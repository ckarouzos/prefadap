"""Lightweight helpers for canonical dataset identifiers.

This module exposes the dataset mapping utilities used by both the
configuration parser and the training/evaluation pipelines.  It intentionally
avoids importing the broader :mod:`prefadap.data` package so that callers can
normalise dataset names without pulling heavy dependencies (``datasets``,
``transformers`` …).
"""

from __future__ import annotations

from typing import Iterable


SUPPORTED_DATASETS: dict[str, list[str]] = {
    "askculinary": ["train", "validation", "test"],
    "askculinary_only": ["train"],
    "askengineers": ["train", "validation", "test"],
    "askengineers_sft": ["train", "validation"],
    "askengineers_preference": ["train"],
    "askengineers_kto": ["train"],
    "askengineers_orpo": ["train"],
    "tldr": [
        "train",
        "validation",
        "test",
        "full_validation",
        "full_test",
        "ood_validation",
        "ood_test",
    ],
    "shp": ["train", "validation", "test"],
    "shp_sft": ["train", "validation"],
    "shp_preference": ["train"],
    "shp_kto": ["train"],
    "shp_orpo": ["train"],
    "cnndm": ["train", "validation", "test"],
    "mixed": ["train", "validation", "test"],
    "mixed_preferences": ["train"],
    "mixed_orpo": ["train"],
    "cnndm_only": ["train"],
    "pseudo_cnndm_only": ["train"],
    "mixed_unary": ["train"],
}

def normalise_dataset_name(name: str) -> str:
    """Return canonical dataset identifier for ``name`` when possible."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Dataset name must be a non-empty string")

    candidate = name.strip()
    prefix = ""
    remainder = candidate

    lowered = remainder.lower()
    if lowered.startswith("pseudo_"):
        prefix = "pseudo_"
        remainder = remainder[len("pseudo_") :]
        lowered = remainder.lower()

    if remainder in SUPPORTED_DATASETS:
        canonical = remainder
    elif lowered in SUPPORTED_DATASETS:
        canonical = lowered


    if prefix:
        base = canonical
        if base.startswith("pseudo_"):
            base = base[len("pseudo_") :]
        canonical = prefix + base

    return canonical


def list_supported_datasets() -> list[str]:
    """Return all dataset identifiers supported by this project."""

    return sorted(SUPPORTED_DATASETS)


def validate_dataset_name(name: str, allowed: Iterable[str] | None = None) -> None:
    """Validate ``name`` against the list of supported datasets."""

    canonical = normalise_dataset_name(name)
    if allowed is None:
        valid = {dataset for dataset in SUPPORTED_DATASETS}
    else:
        valid = {normalise_dataset_name(option) for option in allowed}

    if canonical not in valid:
        if canonical.startswith("pseudo_"):
            base = canonical[len("pseudo_") :]
            if base in valid:
                return
        options = ", ".join(sorted(valid))
        raise ValueError(
            f"Unsupported dataset '{name}'. Supported datasets: {options}"
        )


__all__ = [
    "SUPPORTED_DATASETS",
    "list_supported_datasets",
    "normalise_dataset_name",
    "validate_dataset_name",
]
