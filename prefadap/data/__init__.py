from __future__ import annotations

"""Utilities for loading summarisation datasets and formatting prompts.

This module centralises dataset handling used by generation and evaluation
scripts.  It provides small wrappers around :func:`datasets.load_dataset`
implementing caching, deterministic subsampling and optional dumping of
selected indices.  Basic prompt formatting helpers for the supported datasets
are also exposed so that scripts do not need to duplicate the logic.

The helpers are intentionally lightweight and only depend on the optional
``datasets`` package.  When the library is not available the loading functions
raise :class:`ImportError` instead of failing at import time.
"""



from pathlib import Path
from typing import Any, Mapping, Sequence
import random

from .collators import (
    DataCollatorWithLabelPaddingWithSide,
    DPODataCollator,
    RewardDataCollator,
    collate_to_device,
)
from .dpo import make_dpo_dataset, make_pseudo_dataset
from .indexing import dataset_with_indices
from .kto import make_kto_pseudo_dataset, make_kto_raw_text_dataset
from .loader import get_grpo_datasets, get_lm_datasets
from .processing import tokenization
from .registry import (
    list_supported_datasets,
    normalise_dataset_name,
    validate_dataset_name,
)
from .core import resolve_hf_datasets_cache_dir
from .summarisation import (
    load_processed_summarisation_dataset,
    make_raw_text_dataset,
)
from . import safety as _safety  # noqa: F401  # register safety datasets
from . import shp as _shp  # noqa: F401  # register SHP datasets

try:  # pragma: no cover - datasets is optional during tests
    from datasets import Dataset, concatenate_datasets, load_dataset as hf_load_dataset
except Exception:  # pragma: no cover - graceful fallback
    Dataset = None  # type: ignore[misc]

__all__ = [
    "load_dataset",
    "format_prompt",
    "make_input_example_tldr",
    "make_input_example_cnndm",
    "DataCollatorWithLabelPaddingWithSide",
    "collate_to_device",
    "DPODataCollator",
    "RewardDataCollator",
    "get_lm_datasets",
    "get_grpo_datasets",
    "make_raw_text_dataset",
    "make_dpo_dataset",
    "make_pseudo_dataset",
    "make_kto_raw_text_dataset",
    "make_kto_pseudo_dataset",
    "load_processed_summarisation_dataset",
    "list_supported_datasets",
    "normalise_dataset_name",
    "validate_dataset_name",
    "tokenization",
    "load_source_dataset",
    "load_target_dataset",
    "dataset_with_indices",
]

# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------

def make_input_example_tldr(post: str, title: str, subreddit: str) -> str:
    """Format TL;DR input for summarisation models."""
    return f"SUBREDDIT: r/{subreddit}\nTITLE: {title}\nPOST: {post}\nTL;DR:"

def make_input_example_cnndm(article: str) -> str:
    """Format CNN/DailyMail input for summarisation models."""
    return f"ARTICLE: {article}\nTL;DR:"

PROMPT_BUILDERS = {
    "tldr": make_input_example_tldr,
    "cnndm": make_input_example_cnndm,
}


def format_prompt(dataset: str, example: Mapping[str, Any]) -> str:
    """Return a formatted prompt for ``example`` from ``dataset``.

    ``example`` may either contain the raw fields (``post``, ``title`` …) or an
    ``info`` dictionary holding them.  This mirrors the structures used in the
    original OpenAI summarisation preference datasets.
    """
    info = example.get("info", example)
    if dataset not in PROMPT_BUILDERS:
        raise ValueError(f"Unsupported dataset '{dataset}'")
    if dataset == "tldr":
        return make_input_example_tldr(
            info.get("post", ""), info.get("title", ""), info.get("subreddit", "")
        )
    if dataset == "cnndm":
        return make_input_example_cnndm(info.get("article", ""))
    raise ValueError(f"Unsupported dataset '{dataset}'")

# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

DEFAULT_SPLITS = {
    "tldr": "test",
    "cnndm": "validation",
}


def _deterministic_subsample(
    ds: "Dataset", size: int | None, seed: int | None
) -> tuple["Dataset", list[int]]:
    """Return a deterministic subsample of ``ds`` and the selected indices."""
    if size is None or size >= len(ds):
        indices = list(range(len(ds)))
        return ds, indices
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:size]
    return ds.select(indices), indices


def _dump_indices(indices: Sequence[int], path: str | None) -> None:
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for i in indices:
            fh.write(f"{i}\n")


def _load_tldr(split: str, cache_dir: str | None) -> "Dataset":
    """TL;DR raw dataset from UCL-DARK/openai-tldr-filtered.

    We intentionally use the filtered raw-text corpus, not the SFT comparisons.
    """
    cache_dir = resolve_hf_datasets_cache_dir(cache_dir)
    ds_dict = hf_load_dataset("UCL-DARK/openai-tldr-filtered", cache_dir=cache_dir)
    return ds_dict[split]


def _load_cnndm(split: str, cache_dir: str | None) -> "Dataset":
    """CNN/DailyMail v3.0.0 raw dataset.

    This is the canonical source for CNNDM articles and highlights.
    """
    cache_dir = resolve_hf_datasets_cache_dir(cache_dir)
    ds_dict = hf_load_dataset("cnn_dailymail", "3.0.0", cache_dir=cache_dir)
    return ds_dict[split]


DATASET_LOADERS = {
    "tldr": _load_tldr,
    "cnndm": _load_cnndm,
}


def load_dataset(
    name: str,
    *,
    split: str | None = None,
    cache_dir: str | None = None,
    num_samples: int | None = None,
    seed: int | None = None,
    dump_indices_path: str | None = None,
) -> "Dataset":
    """Load ``name`` and return a :class:`datasets.Dataset` instance.

    Parameters
    ----------
    name:
        Identifier of the dataset.  Supported values are ``tldr``, ``cnndm``
    split:
        Dataset split to load.  Defaults to a sensible split for each dataset.
    cache_dir:
        Optional directory used by the ``datasets`` library for caching.
    num_samples:
        If provided, return a deterministic subsample of this size.
    seed:
        Random seed controlling the subsampling.
    dump_indices_path:
        If given, write the selected indices to this path (one per line).
    """
    if Dataset is None:  # pragma: no cover - datasets missing
        raise ImportError("datasets library is required to load data")

    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}")

    split = split or DEFAULT_SPLITS[name]
    ds = DATASET_LOADERS[name](split, cache_dir)
    ds, indices = _deterministic_subsample(ds, num_samples, seed)
    _dump_indices(indices, dump_indices_path)
    return ds
