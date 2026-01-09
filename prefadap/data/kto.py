from __future__ import annotations

from typing import Any, Dict

from .dpo import make_pseudo_raw_text_dataset
from .core import make_filtered_dataset, load_dataset
from .indexing import dataset_with_indices
from .registry import get_dataset_loader, register_dataset

from datasets import DatasetDict, concatenate_datasets




_ASK_PROMPT_FIELDS = ("prompt", "question", "instruction", "input", "source")
_ASK_COMPLETION_FIELDS = ("completion", "response", "answer", "output")
_ASK_LABEL_FIELDS = ("label", "preference", "preferred", "score", "is_preferred")
_TRUE_LABELS = {"1", "true", "yes", "preferred", "positive", "accepted"}
_FALSE_LABELS = {"0", "false", "no", "negative", "rejected"}


def _find_column(columns: set[str], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    options = ", ".join(candidates)
    raise KeyError(f"Dataset is missing required columns; expected one of: {options}")


def _coerce_label(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalised = str(value).strip().lower()
    if normalised in _TRUE_LABELS:
        return True
    if normalised in _FALSE_LABELS:
        return False
    return bool(normalised)


def _standardise_unary_columns(ds: DatasetDict) -> DatasetDict:
    processed = {}
    for split, split_ds in ds.items():
        columns = set(split_ds.column_names)
        rename: Dict[str, str] = {}
        prompt_col = _find_column(columns, _ASK_PROMPT_FIELDS)
        completion_col = _find_column(columns, _ASK_COMPLETION_FIELDS)
        label_col = _find_column(columns, _ASK_LABEL_FIELDS)
        if prompt_col != "prompt":
            rename[prompt_col] = "prompt"
        if completion_col != "completion":
            rename[completion_col] = "completion"
        if label_col != "label":
            rename[label_col] = "label"
        processed_split = split_ds.rename_columns(rename)

        def _coerce(batch: Dict[str, list[Any]]) -> Dict[str, list[Any]]:
            return {
                "prompt": [str(value) for value in batch["prompt"]],
                "completion": [str(value) for value in batch["completion"]],
                "label": [_coerce_label(value) for value in batch["label"]],
            }

        processed[split] = processed_split.map(_coerce, batched=True)
    return DatasetDict(processed)


def make_kto_raw_text_dataset(
    args: Any, cache_dir: str | None = None
) -> tuple[DatasetDict, list[str], list[str]]:
    """Create the raw dataset used for KTO training.

    When using the TL;DR preference dataset, the pairwise preferences are
    transformed into single completion examples using
    :func:`trl.unpair_preference_dataset` for each split.
    """
    if args.dataset_name == "tldr":
        ds = load_dataset("trl-lib/tldr-preference", cache_dir=cache_dir)
        ds = make_filtered_dataset(
            ds,
            args.dataset_structured_subset,
            args.dataset_random_subset,
            seed=getattr(args, "seed", None),
        )
        from trl import unpair_preference_dataset

        ds = DatasetDict(
            {split: unpair_preference_dataset(split_ds) for split, split_ds in ds.items()}
        )
    elif args.dataset_name == "cnndm":
        ds = load_dataset("cnn_dailymail", "3.0.0", cache_dir=cache_dir)
        ds = make_filtered_dataset(
            ds,
            args.dataset_structured_subset,
            args.dataset_random_subset,
            seed=getattr(args, "seed", None),
        )
        ds = ds.map(
            lambda example: {
                "prompt": example["article"],
                "completion": example["highlights"],
                "label": True,
            }
        )
 
    elif args.dataset_name == "mixed":
        ds_tldr = load_dataset("trl-lib/tldr-preference", cache_dir=cache_dir)
        ds_tldr = make_filtered_dataset(
            ds_tldr, args.dataset_structured_subset, args.dataset_random_subset
        )
        from trl import unpair_preference_dataset

        ds_tldr_train = unpair_preference_dataset(ds_tldr["train"])
        ds_cnndm = load_dataset("cnn_dailymail", "3.0.0", cache_dir=cache_dir)
        ds_cnndm = make_filtered_dataset(
            ds_cnndm, args.dataset_structured_subset, args.dataset_random_subset
        )
        ds_cnndm = ds_cnndm.map(
            lambda example: {
                "prompt": example["article"],
                "completion": example["highlights"],
                "label": True,
            }
        )
        combined_train = concatenate_datasets([ds_tldr_train, ds_cnndm["train"]])
        ds = DatasetDict({"train": combined_train})
    else:
        try:
            loader = get_dataset_loader(args.dataset_name, "kto")
        except ValueError as exc:
            raise ValueError(f"Unsupported dataset: {args.dataset_name}") from exc
        ds = loader(args, cache_dir=cache_dir)
        ds = make_filtered_dataset(
            ds, args.dataset_structured_subset, args.dataset_random_subset
        )
    return dataset_with_indices(ds)


def make_kto_pseudo_dataset(
    args: Any, cache_dir: str | None = None
) -> tuple[DatasetDict, list[str], list[str]]:
    """Create a KTO-formatted dataset from pseudo preference pairs.

    This function loads a JSONL file containing ``{prompt, chosen, rejected}``
    records, converts it into unpaired samples with boolean labels using
    :func:`trl.unpair_preference_dataset`, and returns a ``DatasetDict`` with a
    single ``train`` split compatible with :class:`trl.KTOTrainer`.
    """

    ds_pairs = make_pseudo_raw_text_dataset(args.pseudo_data_path, cache_dir=cache_dir)
    ds_pairs = make_filtered_dataset(
        ds_pairs, args.dataset_structured_subset, args.dataset_random_subset
    )
    from trl import extract_prompt, unpair_preference_dataset

    processed_splits = {}
    for split, split_ds in ds_pairs.items():
        columns = set(split_ds.column_names)
        if {"chosen", "rejected"} <= columns:
            if "prompt" not in columns:
                split_ds = split_ds.map(
                    extract_prompt,
                    desc="Extracting prompts from preference pairs",
                )
                columns = set(split_ds.column_names)
            keep_cols = {"prompt", "chosen", "rejected"}
            extra_cols = [col for col in split_ds.column_names if col not in keep_cols]
            if extra_cols:
                split_ds = split_ds.remove_columns(extra_cols)
            split_ds = unpair_preference_dataset(split_ds)
        processed_splits[split] = split_ds

    ds = DatasetDict(processed_splits)
    return dataset_with_indices(ds)


@register_dataset("mixed_unary", "kto")
def _load_mixed_unary(args: Any, cache_dir: str | None = None) -> DatasetDict:
    ds, _, _ = make_kto_pseudo_dataset(args, cache_dir=cache_dir)
    return ds


@register_dataset("askengineers", "kto")
@register_dataset("askengineers_kto", "kto")
def _load_askengineers_kto(args: Any, cache_dir: str | None = None) -> DatasetDict:
    """Load AskEngineers KTO dataset.
    
    This loads preference data from local JSONL files at data/prefadapt_data/SHP/askengineers/
    created by scripts/data/build_shp_domain_slices.py, then unpairs it for KTO training.
    """
    from .shp import load_shp_preference
    from types import SimpleNamespace
    from trl import unpair_preference_dataset
    
    # Create minimal args object for SHP loader
    shp_args = SimpleNamespace(
        dataset_name="shp_askengineers",
        dataset="askengineers",
        data_dir=getattr(args, "data_dir", "data")
    )
    ds = load_shp_preference(shp_args, cache_dir=cache_dir)
    
    # Unpair the preference dataset for KTO
    ds = DatasetDict(
        {split: unpair_preference_dataset(split_ds) for split, split_ds in ds.items()}
    )
    
    ds = _standardise_unary_columns(ds)
    ds = make_filtered_dataset(
        ds,
        args.dataset_structured_subset,
        args.dataset_random_subset,
        seed=getattr(args, "seed", None),
    )
    return ds
