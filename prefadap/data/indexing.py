"""Utilities for managing dataset indices.

Centralised helper functions for hashing dataset splits, optionally
subsampling the resulting indices and persisting them to disk.  Dataset
preparation pipelines rely on these utilities to ensure consistent
handling of example identifiers across tasks.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

try:  # pragma: no cover - ``datasets`` is an optional dependency
    from datasets import DatasetDict
except Exception:  # pragma: no cover - datasets not installed
    DatasetDict = Any  # type: ignore


def _sha256_example(example: Dict[str, Any]) -> str:
    """Return the SHA256 hash of a dataset example."""

    text = json.dumps(example, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_splits(ds: DatasetDict) -> Tuple[List[str], List[str]]:
    """Compute SHA256 hashes for train and evaluation splits."""

    train_idx = [_sha256_example(ex) for ex in ds["train"]] if "train" in ds else []
    eval_split = "validation" if "validation" in ds else ("test" if "test" in ds else None)
    eval_idx = [_sha256_example(ex) for ex in ds[eval_split]] if eval_split else []
    return train_idx, eval_idx


def subsample_indices(
    indices: List[str], proportion: float | None, seed: int | None = None
) -> List[str]:
    """Return a random subset of ``indices``.

    Parameters
    ----------
    indices:
        The list of indices to subsample.
    proportion:
        Fraction of indices to keep. ``None`` or values ``>=1`` return the
        original list unchanged.
    seed:
        Optional random seed used for deterministic sampling.
    """

    if proportion is None or proportion >= 1:
        return indices
    if not (0 < proportion <= 1):  # pragma: no cover - defensive
        raise ValueError("proportion must be in (0, 1]")
    rng = np.random.default_rng(seed)
    n = max(1, int(len(indices) * proportion))
    choice = rng.choice(len(indices), size=n, replace=False)
    return [indices[i] for i in sorted(choice)]


def dump_indices(
    save: bool, train_idx: List[str], eval_idx: List[str], output_dir: str
) -> None:
    """Write indices to ``output_dir`` when ``save`` is ``True``.

    The indices are stored in JSONL format where each line contains
    ``{"index": i, "sha256": sha}``.
    """

    if not save:
        return
    os.makedirs(output_dir, exist_ok=True)

    def _write(path: Path, indices: List[str]) -> None:
        if not indices:
            return
        with path.open("w", encoding="utf-8") as f:
            for i, sha in enumerate(indices):
                json.dump({"index": i, "sha256": sha}, f)
                f.write("\n")

    out = Path(output_dir)
    _write(out / "train_indices.jsonl", train_idx)
    _write(out / "eval_indices.jsonl", eval_idx)


def load_indices(output_dir: str) -> Tuple[List[str], List[str]]:
    """Load indices previously written by :func:`dump_indices`."""

    def _read(path: Path) -> List[str]:
        if not path.exists():
            return []
        data: List[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                data.append(obj["sha256"])
        return data

    out = Path(output_dir)
    return _read(out / "train_indices.jsonl"), _read(out / "eval_indices.jsonl")


def dataset_with_indices(
    ds: DatasetDict,
    *,
    index_subsample: float | None = None,
    seed: int | None = None,
) -> Tuple[DatasetDict, List[str], List[str]]:
    """Return dataset along with hashed indices for its splits."""

    train_idx, eval_idx = hash_splits(ds)
    if index_subsample is not None:
        train_idx = subsample_indices(train_idx, index_subsample, seed)
        eval_idx = subsample_indices(eval_idx, index_subsample, seed)
    return ds, train_idx, eval_idx


__all__ = [
    "hash_splits",
    "subsample_indices",
    "dump_indices",
    "load_indices",
    "dataset_with_indices",
]

