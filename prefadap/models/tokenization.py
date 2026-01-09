"""Tokenizer normalisation helpers.

This module contains small utilities for working with causal language model
tokenizers. It focuses on providing a consistent padding token across models
and safe label masking with a fixed ``ignore_index`` value.
"""

from __future__ import annotations

from typing import Dict, List, Sequence
from transformers import PreTrainedTokenizerBase


# ``-100`` is the value used by ``torch.nn.CrossEntropyLoss`` to skip positions
# during loss computation.
IGNORE_INDEX = -100


def ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> None:
    """Ensure ``tokenizer`` exposes a padding token.

    Some tokenizers – notably the Llama family – ship without an explicit pad
    token.  Many training utilities assume the presence of one, so we add a pad
    token that mirrors the EOS token when missing.  Tokenizers that already
    define a pad token (e.g. Gemma) are left untouched.
    
    This function uses getattr to safely handle simple/dummy tokenizers that
    may not have all attributes defined.
    """

    pad_token = getattr(tokenizer, "pad_token", None)
    if pad_token is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        # Set pad_token and pad_token_id only if EOS info is available
        if eos_token is not None:
            tokenizer.pad_token = eos_token
        if eos_id is not None:
            tokenizer.pad_token_id = eos_id


def mask_target_tokens(
    input_ids: Sequence[int],
    *,
    target_token_count: int,
    ignore_index: int = IGNORE_INDEX,
) -> List[int]:
    """Mask non‑target tokens in ``input_ids``.

    Parameters
    ----------
    input_ids:
        Full sequence consisting of prompt, target and optional padding.
    target_token_count:
        Number of tokens that belong to the target (including an EOS token).
        These tokens are assumed to appear immediately after the prompt and
        before any padding.
    ignore_index:
        Value used to mark tokens that should not contribute to the loss.
    """

    labels = list(input_ids)
    prompt_end = len(input_ids) - target_token_count

    for idx in range(prompt_end):
        labels[idx] = ignore_index
    for idx in range(prompt_end + target_token_count, len(labels)):
        labels[idx] = ignore_index
    return labels


def tokenize_prompt_and_target(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    target: str,
    *,
    max_length: int,
) -> Dict[str, List[int]]:
    """Tokenize ``prompt`` and ``target`` with padding and label masking.

    The tokenizer is first normalised to ensure the presence of a pad token.
    The returned dictionary contains ``input_ids``, ``attention_mask`` and
    ``labels`` where only the target portion contributes to the loss.
    """

    ensure_pad_token(tokenizer)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    target_ids.append(tokenizer.eos_token_id)

    input_ids = prompt_ids + target_ids
    labels = mask_target_tokens(
        input_ids, target_token_count=len(target_ids), ignore_index=IGNORE_INDEX
    )
    attention_mask = [1] * len(input_ids)

    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [IGNORE_INDEX] * pad_len

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


__all__ = [
    "IGNORE_INDEX",
    "ensure_pad_token",
    "mask_target_tokens",
    "tokenize_prompt_and_target",
]
