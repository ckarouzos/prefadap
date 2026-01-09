"""High level dataset loader for language model training.

This module consolidates dataset loading logic used across the various
training pipelines.  Previously each pipeline repeated the same steps:

* load the raw text dataset
* pick the appropriate train/evaluation splits for the algorithm
* tokenise the data for either supervised fine-tuning or RL style training

``get_lm_datasets`` performs all of these steps and returns a mapping with the
train and evaluation datasets.  Most modes return tokenised splits, while
preference-optimisation objectives (DPO/ORPO/Reward) now expose raw text so the
TRL trainers can apply their own preprocessing.  Hashed indices of the raw
splits are always propagated so callers can persist them for provenance
tracking.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import copy
import os
from functools import partial

from prefadap.dataset_names import normalise_dataset_name

from .summarisation import make_raw_text_dataset
from .summarisation import make_lm_dataset
from .summarisation import (
    make_input_example_cnndm,
    make_input_example_tldr,
)
from .dpo import make_dpo_dataset
from .kto import make_kto_raw_text_dataset, make_kto_pseudo_dataset
from .processing import tokenization

def _select_eval_split(ds) -> str | None:
    """Return the appropriate evaluation split for ``ds``.

    Prefers ``"validation"`` when available and falls back to ``"test"``.
    Returns ``None`` if neither split exists.
    """

    if "validation" in ds:
        return "validation"
    if "test" in ds:
        return "test"
    return None


_ASK_PROMPT_FIELDS = ("prompt", "question", "instruction", "input", "source")
_ASK_DATASETS = {
    "askengineers",
    "askengineers_sft",
    "askculinary",
    "askculinary_only",
}


def _can_emit_chat_prompts(tokenizer: Any) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def _extract_prompt_field(
    examples: Dict[str, list[Any]],
    candidates: tuple[str, ...],
    *,
    required: bool = True,
) -> list[Any] | None:
    for field in candidates:
        values = examples.get(field)
        if values is not None:
            return [
                value
                if isinstance(value, (dict, list))
                else str(value)
                for value in values
            ]
    if required:
        joined = ", ".join(candidates)
        raise KeyError(f"Dataset is missing required columns; expected one of: {joined}")
    return None


def _format_messages_as_text(messages: list[Any]) -> str:
    lines: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            role = str(message.get("role", "user")).upper()
            content = str(message.get("content", ""))
            lines.append(f"{role}: {content}".rstrip())
        else:
            lines.append(str(message))
    return "\n".join(lines).strip()


def _normalise_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        if "content" in prompt:
            return str(prompt.get("content", ""))
        return str(prompt)
    if isinstance(prompt, list):
        return _format_messages_as_text(prompt)
    return str(prompt)


def _coerce_chat_message(message: Any) -> dict[str, str]:
    if isinstance(message, dict):
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        return {"role": role, "content": content}
    return {"role": "user", "content": str(message)}


def _normalise_prompt_chat(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, list):
        return [_coerce_chat_message(message) for message in prompt]
    if isinstance(prompt, dict):
        return [_coerce_chat_message(prompt)]
    return [{"role": "user", "content": str(prompt)}]


def _validate_grpo_prompt_schema(ds: Dict[str, Any], use_chat: bool) -> None:
    forbidden_columns = {"input_ids", "attention_mask", "labels"}
    for split_name, split in ds.items():
        if split is None:
            continue
        forbidden = forbidden_columns.intersection(set(split.column_names))
        if forbidden:
            raise ValueError(
                "GRPO datasets must not contain tokenized columns; "
                f"found {sorted(forbidden)} in split '{split_name}'."
            )
        if "prompt" not in split.column_names:
            raise ValueError(
                f"GRPO datasets must contain a prompt column; missing from split '{split_name}'."
            )
        for idx, sample in enumerate(split):
            prompt = sample.get("prompt")
            if use_chat:
                if not isinstance(prompt, list):
                    raise ValueError(
                        "GRPO prompt values must be list[dict] when chat templates are enabled; "
                        f"found {type(prompt)} at {split_name}[{idx}]."
                    )
                for message in prompt:
                    if not isinstance(message, dict):
                        raise ValueError(
                            "GRPO chat prompts must be lists of dict messages; "
                            f"found {type(message)} at {split_name}[{idx}]."
                        )
            else:
                if not isinstance(prompt, str):
                    raise ValueError(
                        "GRPO prompt values must be strings when chat templates are disabled; "
                        f"found {type(prompt)} at {split_name}[{idx}]."
                    )


def _build_summarisation_prompts(
    dataset_name: str,
    examples: Dict[str, list[Any]],
) -> list[str] | None:
    if dataset_name == "tldr" and {"post", "title", "subreddit"} <= set(examples):
        return [
            make_input_example_tldr(post, title, subreddit)
            for post, title, subreddit in zip(
                examples["post"], examples["title"], examples["subreddit"]
            )
        ]
    if dataset_name in {"cnndm", "cnndm_only"} and "article" in examples:
        return [make_input_example_cnndm(article) for article in examples["article"]]
    return None


def _build_ask_prompts(
    dataset_name: str,
    examples: Dict[str, list[Any]],
) -> list[Any] | None:
    if dataset_name in _ASK_DATASETS:
        prompts = _extract_prompt_field(examples, ("queries",), required=False)
        if prompts is None:
            prompts = _extract_prompt_field(examples, ("prompt",), required=False)
        return prompts
    return None


def _build_helpfulness_prompts(
    dataset_name: str,
    examples: Dict[str, list[Any]],
) -> list[Any] | None:
    if dataset_name.startswith("shp_") or dataset_name in {"shp", "shp_all"}:
        return _extract_prompt_field(examples, ("prompt",), required=False)
    return None


def _build_grpo_prompts(
    args: Any,
    examples: Dict[str, list[Any]],
    *,
    use_chat: bool,
) -> Dict[str, list[Any]]:
    prompts = None
    dataset_name = getattr(args, "dataset_name", None)
    if isinstance(dataset_name, str) and dataset_name and "," not in dataset_name:
        canonical_name = normalise_dataset_name(dataset_name)
        prompts = _build_summarisation_prompts(canonical_name, examples)
        if prompts is None:
            prompts = _build_ask_prompts(canonical_name, examples)
        if prompts is None:
            prompts = _build_helpfulness_prompts(canonical_name, examples)

    if prompts is None:
        prompts = _extract_prompt_field(examples, ("prompt", "queries"), required=False)
    if prompts is None:
        prompts = _extract_prompt_field(examples, _ASK_PROMPT_FIELDS, required=False)
    if prompts is None:
        raise KeyError(
            "GRPO datasets must contain a prompt column (prompt, queries, question, "
            "instruction, input, or source) or known summarisation fields."
        )
    if use_chat:
        prompts_out = [_normalise_prompt_chat(prompt) for prompt in prompts]
    else:
        prompts_out = [_normalise_prompt_text(prompt) for prompt in prompts]
    return {"prompt": prompts_out}


def get_lm_datasets(
    args: Any,
    tokenizer,
    mode: str,
) -> Tuple[Dict[str, Any], list[str], list[str]]:
    """Return datasets for LM style training.

    Parameters
    ----------
    args:
        Namespace of CLI arguments describing dataset configuration.
    tokenizer:
        Tokenizer used for processing the raw text.
    mode:
        One of ``"sft"``, ``"rl"``, ``"dpo"`` or ``"kto"`` determining how the
        dataset is prepared.

    Returns
    -------
    datasets:
        Mapping with ``"train"`` and ``"eval"`` datasets.  Depending on
        ``mode`` these may contain token IDs (e.g., SFT, RL) or raw text (DPO).
    train_idx / eval_idx:
        SHA256 hashes for the examples of the raw train/eval splits.
    """

    cache_dir = getattr(args, "hf_cache_dir", None) or getattr(args, "cache_dir", None)

    if mode == "dpo":
        ds, train_idx, eval_idx = make_dpo_dataset(
            args, tokenizer, cache_dir=cache_dir
        )
        eval_split = _select_eval_split(ds)
        ds_train = ds["train"]
        ds_eval = ds[eval_split] if eval_split else None
        return {"train": ds_train, "eval": ds_eval}, train_idx, eval_idx

    if mode == "kto":
        pseudo_configured = getattr(args, "pseudo_data_path", None)
        env_pseudo_path = os.environ.get("PSEUDO_DATA_PATH")
        if (
            getattr(args, "dataset_name", "") == "mixed_unary"
            or pseudo_configured
            or env_pseudo_path
        ):
            ds, train_idx, eval_idx = make_kto_pseudo_dataset(
                args, cache_dir=cache_dir
            )
        else:
            ds, train_idx, eval_idx = make_kto_raw_text_dataset(
                args, cache_dir=cache_dir
            )
        eval_split = _select_eval_split(ds)
        ds_train = ds["train"]
        ds_eval = ds[eval_split] if eval_split else None
        return {"train": ds_train, "eval": ds_eval}, train_idx, eval_idx

    if mode == "lm":
        ds, train_idx, eval_idx = make_lm_dataset(args, tokenizer, cache_dir=cache_dir)
        eval_split = _select_eval_split(ds)
        ds_train = ds["train"]
        ds_eval = ds[eval_split] if eval_split else None
        return {"train": ds_train, "eval": ds_eval}, train_idx, eval_idx

    # Supervised fine tuning or RL style training
    ds_raw, train_idx, eval_idx = make_raw_text_dataset(args, cache_dir=cache_dir)

    remove_cols = ds_raw["train"].column_names

    tok = copy.deepcopy(tokenizer)
    tok.padding_side = "left"
    tok.truncation_side = "left"
    tokenize_fn = partial(
        tokenization,
        tokenizer=tok,
        rl_training=(mode == "rl"),
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    tokenized = ds_raw.map(
        tokenize_fn,
        batched=True,
        num_proc=8,
        remove_columns=remove_cols,
    )

    eval_split = _select_eval_split(tokenized)
    ds_train = tokenized["train"]
    ds_eval = tokenized[eval_split] if eval_split else None
    return {"train": ds_train, "eval": ds_eval}, train_idx, eval_idx


def get_grpo_datasets(
    args: Any,
    tokenizer,
) -> Tuple[Dict[str, Any], list[str], list[str]]:
    """Return GRPO-ready datasets with raw prompt samples.

    GRPO datasets must provide a ``prompt`` field containing either a string
    or a list of ``{role, content}`` chat messages. No tokenised columns are
    produced in this path.
    """
    cache_dir = getattr(args, "hf_cache_dir", None) or getattr(args, "cache_dir", None)
    ds_raw, train_idx, eval_idx = make_raw_text_dataset(args, cache_dir=cache_dir)
    use_chat = _can_emit_chat_prompts(tokenizer)
    ds = ds_raw.map(
        lambda batch: _build_grpo_prompts(args, batch, use_chat=use_chat),
        batched=True,
    )
    _validate_grpo_prompt_schema(ds, use_chat)
    eval_split = _select_eval_split(ds)
    ds_train = ds["train"]
    ds_eval = ds[eval_split] if eval_split else None
    return {"train": ds_train, "eval": ds_eval}, train_idx, eval_idx


__all__ = ["get_lm_datasets", "get_grpo_datasets"]
