from __future__ import annotations

import os
from typing import Any, Dict

from .core import make_filtered_dataset, load_dataset, resolve_pseudo_data_path
from .indexing import dataset_with_indices
from .registry import get_dataset_loader, register_dataset, register_task
from .processing import (
    tokenize_dpo as _tokenize_dpo,
)
from .adapters import ColumnMapping, standardize_preference_dataset


from datasets import DatasetDict




_ASK_PROMPT_FIELDS = ("prompt", "question", "instruction", "input", "source")
_ASK_CHOSEN_FIELDS = (
    "chosen",
    "preferred",
    "better_response",
    "better_answer",
    "response_chosen",
)
_ASK_REJECTED_FIELDS = (
    "rejected",
    "other",
    "worse_response",
    "worse_answer",
    "response_rejected",
)
_ASK_WEIGHT_FIELDS = ("weight", "preference_score", "score")


def _find_column(columns: set[str], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    options = ", ".join(candidates)
    raise KeyError(f"Dataset is missing required columns; expected one of: {options}")


def _standardise_preference_columns(ds: DatasetDict) -> DatasetDict:
    """Rename AskEngineers preference columns to canonical names."""

    processed = {}
    for split, split_ds in ds.items():
        columns = set(split_ds.column_names)
        rename: Dict[str, str] = {}
        prompt_col = _find_column(columns, _ASK_PROMPT_FIELDS)
        chosen_col = _find_column(columns, _ASK_CHOSEN_FIELDS)
        rejected_col = _find_column(columns, _ASK_REJECTED_FIELDS)
        if prompt_col != "prompt":
            rename[prompt_col] = "prompt"
        if chosen_col != "chosen":
            rename[chosen_col] = "chosen"
        if rejected_col != "rejected":
            rename[rejected_col] = "rejected"
        weight_col = next((c for c in _ASK_WEIGHT_FIELDS if c in columns), None)
        if weight_col and weight_col != "weight":
            rename[weight_col] = "weight"
        processed_split = split_ds.rename_columns(rename)

        def _coerce_to_str(batch: Dict[str, list[Any]]) -> Dict[str, list[str]]:
            return {
                "prompt": [str(value) for value in batch["prompt"]],
                "chosen": [str(value) for value in batch["chosen"]],
                "rejected": [str(value) for value in batch["rejected"]],
            }

        processed[split] = processed_split.map(_coerce_to_str, batched=True)
    return DatasetDict(processed)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def make_dpo_samples(examples: Dict[str, list[str]]) -> Dict[str, list[str]]:
    """Convert raw preference pairs into DPO-ready samples.

    If ``weight`` is present in the input examples, it is propagated unchanged.
    """
    result = {
        "prompt": examples["prompt"],
        "chosen": [" " + text for text in examples["chosen"]],
        "rejected": [" " + text for text in examples["rejected"]],
    }
    if "weight" in examples:
        result["weight"] = examples["weight"]
    return result

# Expose shared tokeniser under historical name
tokenization_dpo = _tokenize_dpo

# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------


@register_dataset("tldr", "dpo")
def _load_tldr_dpo(args: Any, cache_dir: str | None = None) -> DatasetDict:
    """Loader for the TL;DR preference dataset used in DPO."""
    ds = load_dataset("trl-lib/tldr-preference", cache_dir=cache_dir)
    ds = make_filtered_dataset(
        ds,
        args.dataset_structured_subset,
        args.dataset_random_subset,
        seed=getattr(args, "seed", None),
    )
    ds = ds.map(make_dpo_samples, batched=True)
    return ds


def _load_and_process_pseudo_dataset(args: Any, cache_dir: str | None = None) -> DatasetDict:
    ds = make_pseudo_raw_text_dataset(args.pseudo_data_path, cache_dir=cache_dir)
    ds = ds.map(make_dpo_samples, batched=True)
    return ds


@register_dataset("pseudo_cnndm_only", "dpo")
def _load_pseudo_cnndm_only(args: Any, cache_dir: str | None = None) -> DatasetDict:
    return _load_and_process_pseudo_dataset(args, cache_dir)


@register_dataset("pseudo_cnndm", "dpo")
def _load_pseudo_cnndm(args: Any, cache_dir: str | None = None) -> DatasetDict:
    return _load_and_process_pseudo_dataset(args, cache_dir)


@register_dataset("mixed_preferences", "dpo")
def _load_mixed_preferences(args: Any, cache_dir: str | None = None) -> DatasetDict:
    return _load_and_process_pseudo_dataset(args, cache_dir)


@register_dataset("mixed_orpo", "dpo")
def _load_mixed_orpo(args: Any, cache_dir: str | None = None) -> DatasetDict:
    return _load_and_process_pseudo_dataset(args, cache_dir)


@register_dataset("pseudo_askculinary_only", "dpo")
def _load_pseudo_askculinary_only(args: Any, cache_dir: str | None = None) -> DatasetDict:
    return _load_and_process_pseudo_dataset(args, cache_dir)


@register_dataset("pseudo_askculinary", "dpo")
def _load_pseudo_askculinary(args: Any, cache_dir: str | None = None) -> DatasetDict:
    return _load_and_process_pseudo_dataset(args, cache_dir)


@register_dataset("askengineers", "dpo")
@register_dataset("askengineers_preference", "dpo")
@register_dataset("askengineers_orpo", "dpo")
def _load_askengineers_preference(
    args: Any, cache_dir: str | None = None
) -> DatasetDict:
    """Load AskEngineers preference dataset.
    
    This loads data from local JSONL files at data/prefadapt_data/SHP/askengineers/
    created by scripts/data/build_shp_domain_slices.py.
    """
    from .shp import load_shp_preference
    from types import SimpleNamespace
    
    # Create minimal args object for SHP loader
    shp_args = SimpleNamespace(
        dataset_name="shp_askengineers",
        dataset="askengineers",
        data_dir=getattr(args, "data_dir", "data")
    )
    ds = load_shp_preference(shp_args, cache_dir=cache_dir)
    ds = _standardise_preference_columns(ds)
    ds = make_filtered_dataset(
        ds,
        args.dataset_structured_subset,
        args.dataset_random_subset,
        seed=getattr(args, "seed", None),
    )
    return ds


def make_dpo_raw_text_dataset(args: Any, cache_dir: str | None = None) -> DatasetDict:
    """Load a raw preference dataset and align it to canonical columns.

    The dataset is obtained via :func:`get_dataset_loader` using
    ``args.dataset_name``. If ``args.column_mapping`` is provided, arbitrary
    field names are translated to the ``prompt``, ``chosen`` and ``rejected``
    schema before further processing.
    """
    loader = get_dataset_loader(args.dataset_name, "dpo")
    ds = loader(args, cache_dir=cache_dir)
    mapping = getattr(args, "column_mapping", None)
    if mapping:
        ds = standardize_preference_dataset(ds, ColumnMapping.from_dict(mapping))
    return ds


def make_dpo_dataset(
    args: Any, tokenizer, cache_dir: str | None = None
) -> tuple[DatasetDict, list[str], list[str]]:
    """Create a raw-text DPO dataset for training.

    The ``tokenizer`` argument is retained for API compatibility but
    tokenisation is deferred to downstream TRL trainers.
    """
    del tokenizer

    ds = make_dpo_raw_text_dataset(args, cache_dir=cache_dir)
    ds, train_idx, eval_idx = dataset_with_indices(ds)

    required = {"prompt", "chosen", "rejected"}
    train_cols = set(ds["train"].column_names)
    if not required <= train_cols:
        missing = ", ".join(sorted(required - train_cols))
        raise ValueError(
            "DPO datasets must contain raw text columns 'prompt', 'chosen', "
            f"and 'rejected'. Missing: {missing}"
        )

    return ds, train_idx, eval_idx


def make_pseudo_raw_text_dataset(
    pseudo_jsonl_path: str | bytes | os.PathLike[str] | os.PathLike[bytes] | None,
    cache_dir: str | None = None,
) -> DatasetDict:
    """Load a pseudo dataset stored in JSONL format.

    The path is resolved from ``pseudo_jsonl_path`` or the ``PSEUDO_DATA_PATH``
    environment variable when the former is omitted.  This matches the
    behaviour of the HPC launcher which exports the variable prior to
    dispatching training jobs.
    """

    path = resolve_pseudo_data_path(pseudo_jsonl_path)
    return load_dataset("json", data_files={"train": path}, cache_dir=cache_dir)


def make_pseudo_dataset(
    args: Any, tokenizer, cache_dir: str | None = None
) -> tuple[DatasetDict, list[str], list[str]]:
    """Create a raw-text DPO dataset from pseudo data.

    The ``tokenizer`` argument is retained for API compatibility but
    tokenisation is deferred to downstream TRL trainers.
    """
    del tokenizer

    ds = make_pseudo_raw_text_dataset(args.pseudo_data_path, cache_dir=cache_dir)
    ds = ds.map(make_dpo_samples, batched=True)
    ds, train_idx, eval_idx = dataset_with_indices(ds)

    required = {"prompt", "chosen", "rejected"}
    train_cols = set(ds["train"].column_names)
    if not required <= train_cols:
        missing = ", ".join(sorted(required - train_cols))
        raise ValueError(
            "Pseudo DPO datasets must contain raw text columns 'prompt', "
            f"'chosen' and 'rejected'. Missing: {missing}"
        )

    return ds, train_idx, eval_idx


# Register task-level helpers -------------------------------------------------

register_task(
    "dpo",
    loader=make_dpo_dataset,
    tokenizer=_tokenize_dpo,
)
