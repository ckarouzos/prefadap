from __future__ import annotations

"""Shared dataset processing helpers."""

from typing import Any, Dict



def tokenize_summarisation(
    example: Dict[str, list[str]],
    *,
    tokenizer,
    rl_training: bool,
    max_source_length: int,
    max_target_length: int,
) -> Dict[str, list[Any]]:
    """Tokenise inputs and outputs for summarisation datasets.

    Expects ``tokenizer`` to be a worker-local clone configured with left
    padding and truncation. This avoids mutating shared state across
    multiple workers.
    """
    tok = tokenizer

    if rl_training:
        tokenized = tok(
            example["queries"],
            max_length=max_source_length,
            truncation=True,
            padding=False,
        )
        model_inputs = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
    else:
        inputs = example["queries"]
        labels = example["outputs"]
        concatenated = [inp + out for inp, out in zip(inputs, labels)]
        tokenized = tok(
            concatenated,
            max_length=max_source_length + max_target_length,
            truncation=True,
            padding=False,
        )
        input_ids_full = tokenized["input_ids"]
        for ids in input_ids_full:
            if ids[-1] != tok.eos_token_id:
                ids.append(tok.eos_token_id)
        attention_mask = [[1] * len(ids) for ids in input_ids_full]

        source_tokenized = tok(
            inputs,
            max_length=max_source_length + max_target_length,
            truncation=True,
            padding=False,
        )
        source_input_ids = source_tokenized["input_ids"]
        label_ids = [
            [-100] * len(src_ids) + full_ids[len(src_ids):]
            for src_ids, full_ids in zip(source_input_ids, input_ids_full)
        ]
        model_inputs = {
            "input_ids": input_ids_full,
            "labels": label_ids,
            "attention_mask": attention_mask,
        }

    return model_inputs


def tokenize_dpo(
    examples: Dict[str, list[str]],
    *,
    tokenizer,
    max_source_length: int,
    max_target_length: int,
) -> Dict[str, list[Any]]:
    """Tokenise DPO datasets with EOS handling."""
    prompt_tokenized = tokenizer(
        examples["prompt"],
        max_length=max_source_length,
        truncation=True,
        padding=False,
        add_special_tokens=False,
    )

    def _tokenize_with_eos(texts, max_len):
        tokenized = tokenizer(
            texts,
            max_length=max_len - 1,
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )
        for ids in tokenized["input_ids"]:
            ids.append(tokenizer.eos_token_id)
            if len(ids) > max_len:
                raise ValueError("Tokenized sequence exceeds max_target_length")
        return tokenized

    chosen_tokenized = _tokenize_with_eos(examples["chosen"], max_target_length)
    rejected_tokenized = _tokenize_with_eos(examples["rejected"], max_target_length)

    result = {
        "prompt_input_ids": prompt_tokenized["input_ids"],
        "chosen_input_ids": chosen_tokenized["input_ids"],
        "rejected_input_ids": rejected_tokenized["input_ids"],
    }
    if "weight" in examples:
        result["weight"] = examples["weight"]
    return result


# Backwards compatibility export
# ``tokenization`` previously referred to summarisation tokenisation.
tokenization = tokenize_summarisation

__all__ = [
    "tokenize_summarisation",
    "tokenization",
    "tokenize_dpo",
]
