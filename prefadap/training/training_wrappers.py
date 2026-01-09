"""Compatibility layer re-exporting training pipelines and helpers.

This module preserves historical import locations for training pipelines
and utilities. Only symbols explicitly listed in ``__all__`` are imported
and re-exported.
"""

from __future__ import annotations

from types import SimpleNamespace

# Pipelines
from .pipeline_base import TrainingPipeline
from .pipeline_sft import SFTTrainingPipeline
from .pipeline_dpo import DPOTrainingPipeline
from .pipeline_ppo import PPOTrainingPipeline
from .pipeline_grpo import GRPOTrainingPipeline
from .pipeline_orpo import ORPOTrainingPipeline
from .pipeline_rm import RewardModelTrainingPipeline
from .pipeline_kto import KTOTrainingPipeline


# Trainers / configs
from transformers import TrainingArguments
from trl import (
    SFTTrainer,
    DPOTrainer,
    DPOConfig,
    ORPOTrainer,
    ORPOConfig,
    KTOTrainer,
    KTOConfig,
)

# Data
from prefadap.data import (
    DPODataCollator,
    DataCollatorWithLabelPaddingWithSide,
    get_lm_datasets,
    make_raw_text_dataset,
    make_dpo_dataset,
    make_pseudo_dataset,
    make_kto_raw_text_dataset,
    make_kto_pseudo_dataset,
    tokenization,
)

# Models / LoRA
from prefadap.models.loader import (
    load_model_and_tokenizer,
    load_reward_model_and_tokenizer,
)
from .lora import apply_lora
from .utils import _maybe_apply_lora, TokenKLCallback, save_model_with_lora


# Back-compat alias
maybe_apply_lora = _maybe_apply_lora


def load_source_dataset(args, tokenizer, cache_dir: str | None = None):
    ns = SimpleNamespace(**vars(args))
    ns.dataset_name = getattr(args, "source_dataset_name", None)
    ds, train_idx, eval_idx = get_lm_datasets(ns, tokenizer, mode="dpo")
    return ds["train"], train_idx, eval_idx


def load_target_dataset(args, tokenizer, cache_dir: str | None = None):
    ns = SimpleNamespace(**vars(args))
    ns.dataset_name = getattr(args, "target_dataset_name", None)
    ds, train_idx, eval_idx = get_lm_datasets(ns, tokenizer, mode="sft")
    return ds["train"], train_idx, eval_idx


__all__ = [
    # helpers
    "maybe_apply_lora",
    "_maybe_apply_lora",
    "TokenKLCallback",
    "save_model_with_lora",

    # pipelines
    "TrainingPipeline",
    "SFTTrainingPipeline",
    "DPOTrainingPipeline",
    "PPOTrainingPipeline",
    "GRPOTrainingPipeline",
    "ORPOTrainingPipeline",
    "RewardModelTrainingPipeline",
    "KTOTrainingPipeline",


    # trainers / configs
    "TrainingArguments",
    "SFTTrainer",
    "DPOTrainer",
    "DPOConfig",
    "ORPOTrainer",
    "ORPOConfig",
    "KTOTrainer",
    "KTOConfig",

    # data
    "DPODataCollator",
    "DataCollatorWithLabelPaddingWithSide",
    "get_lm_datasets",
    "make_raw_text_dataset",
    "make_dpo_dataset",
    "make_pseudo_dataset",
    "make_kto_raw_text_dataset",
    "make_kto_pseudo_dataset",
    "tokenization",

    # models
    "load_model_and_tokenizer",
    "load_reward_model_and_tokenizer",
    "apply_lora",

    # dataset loaders
    "load_source_dataset",
    "load_target_dataset",
]
