from __future__ import annotations

import copy
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from transformers import DataCollatorForTokenClassification

import torch
from torch.nn.utils.rnn import pad_sequence

from .processing import tokenize_dpo
from prefadap.models.tokenization import ensure_pad_token

def collate_to_device(
    collate_fn: Callable[[list[Any]], Dict[str, Any]], device: torch.device
) -> Callable[[list[Any]], Dict[str, Any]]:
    """Wrap ``collate_fn`` to move tensors to ``device``.

    Parameters
    ----------
    collate_fn:
        Original collate function producing a mapping of tensors.
    device:
        Target device where tensors should reside.

    Returns
    -------
    Callable
        A new collate function that places tensor values on ``device``.
    """

    def _move_to_device(value: Any, key: str | None = None) -> Any:
        """Recursively move tensors and numeric sequences to ``device``.

        The custom collators in the dual-domain pipelines often return plain
        Python lists (for example ``domain_labels`` or ``input_ids`` when
        tokenisation happens upstream).  These lists bypass ``torch.utils``'
        default conversion which leads to type errors inside model forwards
        expecting tensors.  This helper eagerly converts such numeric
        sequences while leaving string/metadata fields untouched.
        """

        if isinstance(value, torch.Tensor):
            return value.to(device)

        if isinstance(value, Mapping):
            return {k: _move_to_device(v, k if key is None else f"{key}.{k}") for k, v in value.items()}

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if not value:
                return value

            first = value[0]

            if isinstance(first, torch.Tensor):
                try:
                    return torch.stack([v.to(device) for v in value])
                except Exception:
                    return type(value)(v.to(device) for v in value)

            scalar_types = (numbers.Integral, numbers.Real, bool)
            if all(isinstance(item, scalar_types) for item in value):
                try:
                    return torch.as_tensor(value, device=device)
                except (TypeError, ValueError):
                    return value

            # Recursively move nested sequences to device and stack them when possible.
            converted = [_move_to_device(v, key) for v in value]
            if converted and all(isinstance(item, torch.Tensor) for item in converted):
                first_tensor = converted[0]
                if all(item.shape == first_tensor.shape for item in converted[1:]):
                    try:
                        return torch.stack(converted).to(device)
                    except Exception:
                        pass
                pad_value = -100 if key and "label" in key else 0
                try:
                    return pad_sequence(converted, batch_first=True, padding_value=pad_value).to(device)
                except Exception:
                    try:
                        return torch.stack([item.to(device) for item in converted])
                    except Exception:
                        return converted

            return type(value)(converted)

        if hasattr(value, "to"):
            try:
                return value.to(device)
            except Exception:
                return value

        return value

    def wrapper(batch: list[Any]) -> Dict[str, Any]:
        collated = collate_fn(batch)
        return {k: _move_to_device(v, k) for k, v in collated.items()}

    return wrapper


@dataclass
class DataCollatorWithLabelPaddingWithSide(DataCollatorForTokenClassification):
    """Data collator that applies label padding with a specified padding side.

    The underlying tokenizer is deep-copied so that padding configuration
    changes do not affect shared tokenizer instances.
    """

    padding_side: str = "left"
    base_tokenizer: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_tokenizer = copy.deepcopy(self.tokenizer)
        # Ensure pad token is set to avoid CUDA errors during padding
        ensure_pad_token(self.base_tokenizer)
        try:  # Preserve any parent post-init behaviour if present
            super().__post_init__()  # type: ignore[misc]
        except AttributeError:
            pass

    def __call__(self, examples):  # type: ignore[override]
        if not examples:
            return {}
        self.base_tokenizer.padding_side = self.padding_side
        padding_keys = ["input_ids", "labels", "attention_mask"]

        # Normalize examples to ensure consistent lengths within each example
        normalized_examples = []
        for ex in examples:
            ex_to_normalize = {k: v for k, v in ex.items() if k in padding_keys}
            normalized_ex = self._normalize_example_lengths(ex_to_normalize)
            normalized_examples.append(normalized_ex)

        orig_tokenizer = self.tokenizer
        try:
            self.tokenizer = self.base_tokenizer
            batch = super().__call__(normalized_examples)
        finally:
            self.tokenizer = orig_tokenizer

        batch.update(
            {
                k: [ex[k] for ex in examples]
                for k in examples[0]
                if k not in padding_keys
            }
        )
        return batch

    def _normalize_example_lengths(self, example):
        """Ensure input_ids, labels, and attention_mask have consistent lengths within an example.

        This prevents ValueError when the parent collator tries to convert lists of different
        lengths to tensors. We align all sequences to the length of input_ids by truncating
        or padding as needed.
        """
        if "input_ids" not in example or "labels" not in example:
            return example

        input_ids = example["input_ids"]
        labels = example["labels"]
        attention_mask = example.get("attention_mask", [1] * len(input_ids))

        target_length = len(input_ids)

        # Truncate or pad labels to match input_ids length
        if len(labels) > target_length:
            labels = labels[:target_length]
        elif len(labels) < target_length:
            # Pad labels with ignore_index (-100)
            labels = labels + [-100] * (target_length - len(labels))

        # Truncate or pad attention_mask to match input_ids length
        if len(attention_mask) > target_length:
            attention_mask = attention_mask[:target_length]
        elif len(attention_mask) < target_length:
            # Pad attention_mask with 0 (indicating padding tokens)
            attention_mask = attention_mask + [0] * (target_length - len(attention_mask))

        normalized_example = example.copy()
        normalized_example["input_ids"] = input_ids
        normalized_example["labels"] = labels
        normalized_example["attention_mask"] = attention_mask

        return normalized_example


class DPODataCollator:
    """Data collator for DPO training with configurable padding.

    Clones the tokenizer to adjust padding side without mutating the original
    instance. When raw ``prompt``/``chosen``/``rejected`` strings are provided,
    tokenisation is performed on-the-fly using :func:`tokenize_dpo`.
    """

    def __init__(
        self,
        tokenizer,
        padding_side="left",
        pad_to_multiple_of: int | None = None,
        *,
        max_prompt_length: int | None = None,
        max_completion_length: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.base_tokenizer = copy.deepcopy(tokenizer)
        # Ensure pad token is set to avoid CUDA errors during padding
        ensure_pad_token(self.base_tokenizer)
        self.padding_side = padding_side
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length

    def __call__(self, examples):
        if not examples:
            return {}

        examples = [dict(ex) for ex in examples]
        components = ["prompt", "chosen", "rejected"]
        tokenizer = self.base_tokenizer
        tokenizer.padding_side = self.padding_side

        tokenised_keys = {f"{component}_input_ids" for component in components}
        has_tokenised = tokenised_keys <= set(examples[0].keys())
        if not has_tokenised:
            if not {"prompt", "chosen", "rejected"} <= set(examples[0].keys()):
                raise ValueError(
                    "DPODataCollator requires either pre-tokenised inputs or raw "
                    "'prompt', 'chosen' and 'rejected' columns."
                )
            prompt_len = self.max_prompt_length or getattr(tokenizer, "model_max_length", None)
            completion_len = self.max_completion_length or getattr(tokenizer, "model_max_length", None)
            if prompt_len is None or completion_len is None:
                raise ValueError(
                    "DPODataCollator needs max_prompt_length and max_completion_length "
                    "when collating raw preference data."
                )
            try:
                prompt_len = int(prompt_len)
                completion_len = int(completion_len)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError(
                    "Invalid prompt or completion length supplied to DPODataCollator"
                ) from exc

            encoded = tokenize_dpo(
                {component: [ex[component] for ex in examples] for component in components},
                tokenizer=tokenizer,
                max_source_length=prompt_len,
                max_target_length=completion_len,
            )
            for component in components:
                ids_list = encoded[f"{component}_input_ids"]
                for ex, ids in zip(examples, ids_list):
                    ex[f"{component}_input_ids"] = ids

        batched = {}
        for component in components:
            input_ids = [ex[f"{component}_input_ids"] for ex in examples]
            padded = tokenizer.pad(
                {"input_ids": input_ids},
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            batched[f"{component}_input_ids"] = padded["input_ids"]
            batched[f"{component}_attention_mask"] = padded["attention_mask"]
            # Add aliases for TRL compatibility (TRL expects 'chosen' and 'rejected', not '*_input_ids')
            if component in ("chosen", "rejected"):
                batched[component] = padded["input_ids"]

        batched.update(
            {
                k: [ex[k] for ex in examples]
                for k in examples[0]
                if k not in tokenised_keys
            }
        )
        return batched


class RewardDataCollator:
    """Data collator that concatenates prompt and responses for RM training.

    Uses a cloned tokenizer to avoid mutating shared state when switching the
    padding side.
    """

    def __init__(
        self,
        tokenizer,
        padding_side="left",
        pad_to_multiple_of: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.base_tokenizer = copy.deepcopy(tokenizer)
        # Ensure pad token is set to avoid CUDA errors during padding
        ensure_pad_token(self.base_tokenizer)
        self.padding_side = padding_side
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples):
        if not examples:
            return {}
        chosen = [
            ex["prompt_input_ids"] + ex["chosen_input_ids"] for ex in examples
        ]
        rejected = [
            ex["prompt_input_ids"] + ex["rejected_input_ids"] for ex in examples
        ]
        tokenizer = self.base_tokenizer
        tokenizer.padding_side = self.padding_side
        chosen_padded = tokenizer.pad(
            {"input_ids": chosen},
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        rejected_padded = tokenizer.pad(
            {"input_ids": rejected},
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        return {
            "input_ids_chosen": chosen_padded["input_ids"],
            "attention_mask_chosen": chosen_padded["attention_mask"],
            "input_ids_rejected": rejected_padded["input_ids"],
            "attention_mask_rejected": rejected_padded["attention_mask"],
        }