from __future__ import annotations

import os
from typing import Any, Dict, Protocol

from .core import make_filtered_dataset, load_dataset, resolve_pseudo_data_path
from .indexing import dataset_with_indices
from .registry import get_dataset_loader, register_dataset, register_task
from .processing import tokenize_summarisation as _tokenize_summarisation

from datasets import Dataset, DatasetDict, concatenate_datasets




_ASK_PROMPT_FIELDS = ("prompt", "question", "instruction", "input", "source")
_ASK_COMPLETION_FIELDS = ("response", "answer", "output", "completion")
_ASK_GROUP_FIELDS = (
    "group_id",
    "domain",
    "topic",
    "category",
    "source_domain",
    "source_subreddit",
)


def _extract_field(
    examples: Dict[str, list[Any]],
    candidates: tuple[str, ...],
    *,
    required: bool = True,
) -> list[str] | None:
    for field in candidates:
        values = examples.get(field)
        if values is not None:
            return [str(value) for value in values]
    if required:
        joined = ", ".join(candidates)
        raise KeyError(f"Dataset is missing required columns; expected one of: {joined}")
    return None


def _format_instruction_samples(examples: Dict[str, list[Any]]) -> Dict[str, list[str]]:
    prompts = _extract_field(examples, _ASK_PROMPT_FIELDS)
    completions = _extract_field(examples, _ASK_COMPLETION_FIELDS)
    result: Dict[str, list[str]] = {
        "queries": prompts,
        "outputs": [" " + text for text in completions],
    }
    group_values = _extract_field(examples, _ASK_GROUP_FIELDS, required=False)
    if group_values is not None:
        result["group_id"] = group_values
    return result


def _maybe_attach_group(
    dataset: DatasetDict,
    group_field: str | None,
) -> DatasetDict:
    if not group_field:
        return dataset

    def add_group(examples: Dict[str, list[Any]]) -> Dict[str, list[str]]:
        if group_field in examples:
            return {"group_id": [str(value) for value in examples[group_field]]}
        length = len(next(iter(examples.values()))) if examples else 0
        return {"group_id": [group_field] * length}

    return dataset.map(add_group, batched=True)


class DatasetArgs(Protocol):
    """Protocol defining the expected structure of args for dataset loading."""
    dataset_structured_subset: str
    dataset_random_subset: float | None
    seed: int | None = None
    group_field: str | None = None


# ---------------------------------------------------------------------------
# Formatting functions
# ---------------------------------------------------------------------------


def make_input_example_tldr(post: str, title: str, subreddit: str) -> str:
    """Format TL;DR input for summarisation models."""
    return f"SUBREDDIT: r/{subreddit}\nTITLE: {title}\nPOST: {post}\nTL;DR:"


def make_tldr_samples(examples: Dict[str, list[str]]) -> Dict[str, list[str]]:
    """Create TL;DR query/response pairs from dataset examples."""
    inputs = [
        make_input_example_tldr(post, title, subreddit)
        for post, title, subreddit in zip(
            examples["post"], examples["title"], examples["subreddit"]
        )
    ]
    outputs = [" " + summary for summary in examples["summary"]]
    result = {"queries": inputs, "outputs": outputs}
    if "group_id" in examples:
        result["group_id"] = examples["group_id"]
    return result


def make_input_example_cnndm(article: str) -> str:
    """Format CNN/DailyMail input for summarisation models."""
    return f"ARTICLE: {article}\nTL;DR:"


def make_cnndm_samples(examples: Dict[str, list[str]]) -> Dict[str, list[str]]:
    """Create CNN/DailyMail query/response pairs from dataset examples."""
    inputs = [make_input_example_cnndm(article) for article in examples["article"]]
    outputs = [" " + highlights for highlights in examples["highlights"]]
    result = {"queries": inputs, "outputs": outputs}
    if "group_id" in examples:
        result["group_id"] = examples["group_id"]
    return result



tokenization = _tokenize_summarisation


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------


def make_tldr_raw_text_dataset(
    dataset_structured_subset: str,
    dataset_random_subset: float | None,
    seed: int | None,
    cache_dir: str | None = None,
    *,
    group_field: str | None = None,
) -> DatasetDict:
    """Load and format the TL;DR dataset."""
    ds_name = "UCL-DARK/openai-tldr-filtered"
    ds = load_dataset(ds_name, cache_dir=cache_dir)
    ds = make_filtered_dataset(
        ds, dataset_structured_subset, dataset_random_subset, seed=seed
    )

    if group_field:
        def add_group(examples):
            if group_field == "subreddit" and "subreddit" in examples:
                return {"group_id": examples["subreddit"]}
            if group_field == "length" and "post" in examples:
                groups = ["long" if len(p.split()) > 100 else "short" for p in examples["post"]]
                return {"group_id": groups}
            if group_field in examples:
                return {"group_id": examples[group_field]}
            return {"group_id": ["default"] * len(examples[next(iter(examples))])}
        ds = ds.map(add_group, batched=True)

    ds = ds.map(make_tldr_samples, batched=True)
    return ds


def make_cnndm_raw_text_dataset(
    cache_dir: str | None = None, *, group_field: str | None = None
) -> DatasetDict:
    """Load and format the CNN/DailyMail dataset."""
    ds = load_dataset("cnn_dailymail", "3.0.0", cache_dir=cache_dir)

    if group_field:
        def add_group(examples):
            if group_field == "length" and "article" in examples:
                groups = ["long" if len(a.split()) > 400 else "short" for a in examples["article"]]
                return {"group_id": groups}
            if group_field in examples:
                return {"group_id": examples[group_field]}
            return {"group_id": ["default"] * len(examples[next(iter(examples))])}
        ds = ds.map(add_group, batched=True)

    ds = ds.map(make_cnndm_samples, batched=True)
    return ds

def make_askengineers_raw_text_dataset(
    dataset_structured_subset: str,
    dataset_random_subset: float | None,
    seed: int | None,
    cache_dir: str | None = None,
    *,
    group_field: str | None = None,
) -> DatasetDict:
    """Load and format the AskEngineers instruction dataset.
    
    This loads data from local JSONL files at data/prefadapt_data/SHP/askengineers/
    created by scripts/data/build_shp_domain_slices.py.
    """
    from .shp import load_shp_sft
    from types import SimpleNamespace
    
    # Create minimal args object for SHP loader
    args = SimpleNamespace(
        dataset_name="shp_askengineers",
        dataset="askengineers",
        data_dir="data"
    )
    ds = load_shp_sft(args, cache_dir=cache_dir)
    ds = make_filtered_dataset(
        ds,
        dataset_structured_subset,
        dataset_random_subset,
        seed=seed,
    )
    ds = _maybe_attach_group(ds, group_field)
    ds = ds.map(_format_instruction_samples, batched=True)
    return ds


def make_askculinary_raw_text_dataset(
    dataset_structured_subset: str,
    dataset_random_subset: float | None,
    seed: int | None,
    cache_dir: str | None = None,
    *,
    group_field: str | None = None,
) -> DatasetDict:
    """Load and format the AskCulinary instruction dataset.
    
    This loads data from local JSONL files at data/prefadapt_data/SHP/askculinary/
    created by scripts/data/build_shp_domain_slices.py.
    """
    from .shp import load_shp_sft
    from types import SimpleNamespace
    
    # Create minimal args object for SHP loader
    args = SimpleNamespace(
        dataset_name="shp_askculinary",
        dataset="askculinary",
        data_dir="data"
    )
    ds = load_shp_sft(args, cache_dir=cache_dir)
    ds = make_filtered_dataset(
        ds,
        dataset_structured_subset,
        dataset_random_subset,
        seed=seed,
    )
    ds = _maybe_attach_group(ds, group_field)
    ds = ds.map(_format_instruction_samples, batched=True)
    return ds


def make_pseudo_raw_text_dataset(
    pseudo_jsonl_path: str | bytes | os.PathLike[str] | os.PathLike[bytes] | None,
    cache_dir: str | None = None,
) -> DatasetDict:
    """Load pseudo-labelled data stored in JSONL format.

    When ``pseudo_jsonl_path`` is omitted the path is resolved from the
    ``PSEUDO_DATA_PATH`` environment variable.  This mirrors the behaviour of
    the HPC launch scripts which export that variable before invoking the
    training pipeline.
    """

    path = resolve_pseudo_data_path(pseudo_jsonl_path)
    raw = load_dataset("json", data_files={"train": path}, cache_dir=cache_dir)["train"]

    if "split" in raw.column_names:
        splits = {
            name: raw.filter(lambda x, s=name: x["split"] == s).remove_columns("split")
            for name in set(raw["split"])
        }
    else:
        splits = {"train": raw}

    def _to_samples(examples: Dict[str, list[str]]) -> Dict[str, list[str]]:
        prompts = examples.get("prompt")
        if prompts is None:
            for key in ("article", "input"):
                if key in examples:
                    prompts = examples[key]
                    break
        completions = examples.get("completion")
        if completions is None:
            for key in ("chosen", "output", "summary"):
                if key in examples:
                    completions = examples[key]
                    break
        outputs = [" " + c for c in completions]
        return {"queries": prompts, "outputs": outputs}

    return DatasetDict(
        {name: ds.map(_to_samples, batched=True) for name, ds in splits.items()}
    )


# Register canonical loaders for summarisation datasets ---------------------


@register_dataset("tldr", "summarisation")
def _load_tldr(args: Any, cache_dir: str | None = None) -> DatasetDict:
    return make_tldr_raw_text_dataset(
        args.dataset_structured_subset,
        args.dataset_random_subset,
        getattr(args, "seed", None),
        cache_dir=cache_dir,
        group_field=getattr(args, "group_field", None),
    )


@register_dataset("cnndm", "summarisation")
def _load_cnndm(args: Any, cache_dir: str | None = None) -> DatasetDict:
    return make_cnndm_raw_text_dataset(
        cache_dir=cache_dir, group_field=getattr(args, "group_field", None)
    )


@register_dataset("pseudo_cnndm", "summarisation")
def _load_pseudo_cnndm(args: Any, cache_dir: str | None = None) -> DatasetDict:
    return make_pseudo_raw_text_dataset(args.pseudo_data_path, cache_dir=cache_dir)


@register_dataset("pseudo_cnndm_only", "summarisation")
def _load_pseudo_cnndm_only(args: Any, cache_dir: str | None = None) -> DatasetDict:
    ds = make_pseudo_raw_text_dataset(args.pseudo_data_path, cache_dir=cache_dir)
    return DatasetDict({"train": ds["train"]})


@register_dataset("pseudo_askculinary", "summarisation")
def _load_pseudo_askculinary(args: Any, cache_dir: str | None = None) -> DatasetDict:
    return make_pseudo_raw_text_dataset(args.pseudo_data_path, cache_dir=cache_dir)


@register_dataset("pseudo_askculinary_only", "summarisation")
def _load_pseudo_askculinary_only(args: Any, cache_dir: str | None = None) -> DatasetDict:
    ds = make_pseudo_raw_text_dataset(args.pseudo_data_path, cache_dir=cache_dir)
    return DatasetDict({"train": ds["train"]})


@register_dataset("mixed_unary", "summarisation")
def _load_mixed_unary(args: Any, cache_dir: str | None = None) -> DatasetDict:
    return make_pseudo_raw_text_dataset(args.pseudo_data_path, cache_dir=cache_dir)


@register_dataset("askengineers", "summarisation")
@register_dataset("askengineers_sft", "summarisation")
def _load_askengineers(args: Any, cache_dir: str | None = None) -> DatasetDict:
    return make_askengineers_raw_text_dataset(
        args.dataset_structured_subset,
        args.dataset_random_subset,
        getattr(args, "seed", None),
        cache_dir=cache_dir,
        group_field=getattr(args, "group_field", None),
    )


@register_dataset("askculinary", "summarisation")
def _load_askculinary(args: Any, cache_dir: str | None = None) -> DatasetDict:
    return make_askculinary_raw_text_dataset(
        args.dataset_structured_subset,
        args.dataset_random_subset,
        getattr(args, "seed", None),
        cache_dir=cache_dir,
        group_field=getattr(args, "group_field", None),
    )


@register_dataset("askculinary_only", "summarisation")
def _load_askculinary_only(args: Any, cache_dir: str | None = None) -> DatasetDict:
    ds = make_askculinary_raw_text_dataset(
        args.dataset_structured_subset,
        args.dataset_random_subset,
        getattr(args, "seed", None),
        cache_dir=cache_dir,
        group_field=getattr(args, "group_field", None),
    )
    return DatasetDict({"train": ds["train"]})


@register_dataset("mixed", "summarisation")
def _load_mixed(args: DatasetArgs, cache_dir: str | None = None) -> DatasetDict:
    """Load and format the mixed dataset (TL;DR + CNN/DailyMail).

    Parameters
    ----------
    args : DatasetArgs
        Configuration object with required attributes:
        - dataset_structured_subset: str - Subset specification for structured filtering
        - dataset_random_subset: float | None - Random subset fraction (0.0-1.0)
        - seed: int | None - Random seed for reproducible sampling
        - group_field: str | None - Optional field for grouping data
    cache_dir : str | None, optional
        Directory for caching downloaded datasets
        
    Returns
    -------
    DatasetDict
        Combined dataset with train split, and validation/test splits if available
        in both source datasets
        
    Raises
    ------
    AttributeError
        If required attributes are missing from args
    """
    # Load TL;DR dataset
    ds_tldr = make_tldr_raw_text_dataset(
        args.dataset_structured_subset,
        args.dataset_random_subset,
        getattr(args, "seed", None),
        cache_dir=cache_dir,
        group_field=getattr(args, "group_field", None),
    )
    
    # Load CNN/DailyMail dataset
    ds_cnndm = make_cnndm_raw_text_dataset(
        cache_dir=cache_dir, 
        group_field=getattr(args, "group_field", None)
    )
    
    # Combine the training sets
    train_ds = concatenate_datasets([ds_tldr["train"], ds_cnndm["train"]])
    
    # Create mixed dataset with train split (other splits can use individual datasets)
    mixed_ds = DatasetDict({"train": train_ds})
    
    # Add validation split if both source datasets have it
    if "validation" in ds_tldr and "validation" in ds_cnndm:
        val_ds = concatenate_datasets([ds_tldr["validation"], ds_cnndm["validation"]])
        mixed_ds["validation"] = val_ds
    
    # Add test split if both source datasets have it  
    if "test" in ds_tldr and "test" in ds_cnndm:
        test_ds = concatenate_datasets([ds_tldr["test"], ds_cnndm["test"]])
        mixed_ds["test"] = test_ds
        
    return mixed_ds


def make_raw_text_dataset(
    args: Any, dataset_name: str | None = None, cache_dir: str | None = None
) -> tuple[DatasetDict, list[str], list[str]]:
    """Load raw text datasets based on CLI arguments.

    Parameters
    ----------
    args:
        Parsed CLI arguments.
    dataset_name:
        Optional dataset name to override ``args.dataset_name``. Used for
        sequential loading where multiple datasets are trained one after
        another.
    """

    resolved_name = dataset_name or getattr(args, "dataset_name", None)
    sequence = getattr(args, "sequence", None)

    def _load_single(name: str) -> DatasetDict:
        loader = get_dataset_loader(name, "summarisation")
        return loader(args, cache_dir=cache_dir)

    if sequence:
        names = []
        for item in sequence:
            if item is None:
                continue
            token = str(item).strip()
            if token:
                names.append(token)
    elif isinstance(resolved_name, str):
        names = [n.strip() for n in resolved_name.split(",") if n.strip()]
    elif resolved_name:
        names = [resolved_name]
    else:
        names = []

    if not names:
        raise ValueError("No dataset names provided for raw text dataset loading")

    if len(names) > 1:
        ds_list = [_load_single(n) for n in names]
        ratios = getattr(args, "mix_ratios", None)
        if ratios and len(ratios) == len(names):
            total = min(
                int(len(ds["train"]) / r) for ds, r in zip(ds_list, ratios)
            )
            train_parts = []
            for ds, r in zip(ds_list, ratios):
                n_samples = int(total * r)
                train_parts.append(ds["train"].shuffle(seed=args.seed).select(range(n_samples)))
            train_ds = concatenate_datasets(train_parts)
            ds = DatasetDict({"train": train_ds})
        else:
            train_ds = concatenate_datasets([ds["train"] for ds in ds_list])
            ds = DatasetDict({"train": train_ds})
    else:
        ds = _load_single(names[0])

    return dataset_with_indices(ds)


def make_lm_dataset(
    args: Any, tokenizer, cache_dir: str | None = None
) -> tuple[DatasetDict, list[str], list[str]]:
    """Create an unlabeled LM dataset tokenised for causal training."""

    if args.dataset_name == "mixed":
        ds_cnndm = load_dataset("cnn_dailymail", "3.0.0", cache_dir=cache_dir).map(
            lambda ex: {"text": ex["article"]}
        )
        ds_tldr = load_dataset("UCL-DARK/openai-tldr-filtered", cache_dir=cache_dir)
        ds_tldr = make_filtered_dataset(
            ds_tldr, args.dataset_structured_subset, args.dataset_random_subset, seed=args.seed
        )
        ds_tldr = ds_tldr.map(lambda ex: {"text": ex["post"]})
        trains_ds = concatenate_datasets([ds_cnndm["train"], ds_tldr["train"]])
        ds = DatasetDict({"train": trains_ds})
    elif args.dataset_name == "cnndm":
        ds = load_dataset("cnn_dailymail", "3.0.0", cache_dir=cache_dir).map(
            lambda ex: {"text": ex["article"]}
        )
    elif args.dataset_name == "tldr":
        ds = load_dataset("UCL-DARK/openai-tldr-filtered", cache_dir=cache_dir)
        ds = make_filtered_dataset(
            ds, args.dataset_structured_subset, args.dataset_random_subset, seed=args.seed
        )
        ds = ds.map(lambda ex: {"text": ex["post"]})
    elif args.dataset_name in {"askengineers", "askculinary"}:
        loader = (
            make_askengineers_raw_text_dataset
            if args.dataset_name == "askengineers"
            else make_askculinary_raw_text_dataset
        )
        raw = loader(
            args.dataset_structured_subset,
            args.dataset_random_subset,
            getattr(args, "seed", None),
            cache_dir=cache_dir,
        )

        def _combine(batch: Dict[str, list[str]]) -> Dict[str, list[str]]:
            prompts = batch.get("queries") or [""] * len(batch.get("outputs", []))
            completions = batch.get("outputs") or [""] * len(prompts)
            texts: list[str] = []
            for prompt, completion in zip(prompts, completions):
                prompt_str = (prompt or "").strip()
                completion_str = (completion or "").strip()
                if prompt_str:
                    texts.append(f"{prompt_str}\n\n{completion_str}" if completion_str else prompt_str)
                else:
                    texts.append(completion_str)
            return {"text": texts}

        ds = DatasetDict(
            {
                split: split_ds.map(
                    _combine,
                    batched=True,
                    remove_columns=split_ds.column_names,
                )
                for split, split_ds in raw.items()
            }
        )

    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    ds, train_idx, eval_idx = dataset_with_indices(ds)
    tokenized = ds.map(
        lambda batch: tokenizer(
            batch["text"], truncation=True, max_length=args.max_seq_length, padding=False
        ),
        batched=True,
        remove_columns=ds["train"].column_names,
        num_proc=8,
    )
    return tokenized, train_idx, eval_idx


def make_summarisation_dataset(
    args: Any, tokenizer, cache_dir: str | None = None
) -> tuple[DatasetDict, list[str], list[str]]:
    """Create a tokenised summarisation dataset for training."""
    ds, train_idx, eval_idx = make_raw_text_dataset(args, cache_dir=cache_dir)
    ds = ds.map(
        tokenization,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            rl_training=args.rl_training,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        ),
        batched=True,
        num_proc=8,
    )
    return ds


def load_processed_summarisation_dataset(
    dataset: str,
    data_dir: str = "data",
):
    """Load packed and indexed summarisation data from disk.

    The processed dataset is expected to live in ``data/processed`` with a
    companion index file in ``data/indexes``.  Each line in the processed file
    should contain a JSON object with the ``text`` field holding the packed
    string.  The index file must contain JSON records with ``sha256``, ``split``
    and ``length_band`` in the same order.  This helper stitches the two files
    together into a :class:`datasets.Dataset` instance.
    """

    if Dataset is None:  # pragma: no cover - datasets is optional
        raise ImportError("datasets library is required")

    from pathlib import Path
    import json

    processed_path = Path(data_dir) / "processed" / f"{dataset}.jsonl"
    index_path = Path(data_dir) / "indexes" / f"{dataset}.jsonl"

    records = []
    with processed_path.open("r", encoding="utf-8") as proc_file, index_path.open(
        "r", encoding="utf-8"
    ) as idx_file:
        for text_line, meta_line in zip(proc_file, idx_file):
            text_obj = json.loads(text_line)
            meta_obj = json.loads(meta_line)
            if isinstance(text_obj, dict) and "text" in text_obj:
                record = {**meta_obj, "text": text_obj["text"]}
            else:
                record = {**meta_obj, "text": text_obj}
            records.append(record)

    return Dataset.from_list(records)


# Register task-level helpers -------------------------------------------------

register_task(
    "summarisation",
    loader=make_summarisation_dataset,
    tokenizer=_tokenize_summarisation,
)
