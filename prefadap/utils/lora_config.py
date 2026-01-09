from __future__ import annotations

import argparse
from typing import List, Optional
from peft import LoraConfig, TaskType


def get_lora_config(
    r: int = 8,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    task_type: TaskType = TaskType.CAUSAL_LM,
    base_model_name_or_path: Optional[str] = None,
) -> LoraConfig:
    """Create a standard LoRA configuration."""
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules or ["q_proj", "v_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type=task_type,
        base_model_name_or_path=base_model_name_or_path,
    )


def add_lora_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Attach common LoRA-related CLI arguments to a parser."""
    parser.add_argument("--lora", action="store_true", help="Enable LoRA training")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument(
        "--lora_dropout", type=float, default=0.1, help="LoRA dropout probability"
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj",
        help="Comma-separated target modules for LoRA",
    )
    return parser


__all__ = ["get_lora_config", "add_lora_arguments"]
