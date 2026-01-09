"""Utilities for applying LoRA adapters during training.

This module centralises the logic for injecting LoRA layers into models.  It
supports specifying target modules either directly as a sequence, as a
comma-separated string or via a YAML/JSON configuration file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import json
import logging


import yaml

logger = logging.getLogger(__name__)


def _parse_target_modules(target_modules: Any) -> list[str]:
    """Return a normalised list of target module names.

    ``target_modules`` may be provided as a sequence of strings, a
    comma-separated string or a path to a YAML/JSON file containing either a
    list of module names or a mapping with a ``target_modules`` key.
    """

    if target_modules is None:
        return ["q_proj", "v_proj"]

    if isinstance(target_modules, (list, tuple, set)):
        return [str(m) for m in target_modules]

    path = Path(str(target_modules))
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            if yaml is not None and path.suffix in {".yml", ".yaml"}:
                data = yaml.safe_load(f)
            else:
                try:
                    data = json.load(f)
                except Exception:
                    f.seek(0)
                    data = f.read()
        if isinstance(data, dict):
            data = data.get("target_modules") or data.get("modules") or data
        if isinstance(data, list):
            return [str(m) for m in data]
        if isinstance(data, str):
            target_modules = data
        else:  # pragma: no cover - invalid specification
            raise ValueError("Invalid LoRA target module specification")

    return [m.strip() for m in str(target_modules).split(",") if m.strip()]


def apply_lora(
    model: Any,
    target_modules: Any,
    r: int,
    alpha: int,
    dropout: float,
    *,
    base_model_name_or_path: Optional[str] = None,
    task_type: Optional[Any] = None,
):
    """Apply a LoRA configuration to ``model``.

    Parameters
    ----------
    model:
        The model to modify.
    target_modules:
        Module names, a comma-separated string or a path to a YAML/JSON file
        listing the modules to adapt.
    r, alpha, dropout:
        Standard LoRA hyperparameters.
    base_model_name_or_path:
        Optional identifier of the base model.  This is recorded in the PEFT
        configuration so that :meth:`PeftModel.save_pretrained` can resolve the
        original model without relying on network access.
    task_type:
        Optional PEFT task type override. Reward-model LoRA should use
        ``TaskType.SEQ_CLS`` to avoid generation-only assumptions.
    """

    modules = _parse_target_modules(target_modules)

    try:  # pragma: no cover - optional dependency
        from peft import get_peft_model, TaskType  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install 'peft' to use LoRA support") from e

    from prefadap.utils.lora_config import get_lora_config

    if task_type is None:
        task_type = TaskType.CAUSAL_LM

    lora_cfg = get_lora_config(
        r=r,
        alpha=alpha,
        dropout=dropout,
        target_modules=modules,
        task_type=task_type,
        base_model_name_or_path=base_model_name_or_path,
    )
    logger.info(
        "Applying get_peft_model to %s with base_model_name_or_path=%s target_modules=%s r=%s alpha=%s dropout=%s",
        type(model),
        base_model_name_or_path,
        modules,
        r,
        alpha,
        dropout,
    )
    adapted = get_peft_model(model, lora_cfg)
    peft_config = getattr(adapted, "peft_config", None)
    if isinstance(peft_config, dict):
        inference_modes = {
            name: getattr(cfg, "inference_mode", None) for name, cfg in peft_config.items()
        }
        logger.info(
            "LoRA applied: type(adapted)=%s adapters=%s inference_mode=%s",
            type(adapted),
            sorted(peft_config.keys()),
            inference_modes,
        )
    else:
        logger.info("LoRA applied: type(adapted)=%s (peft_config missing)", type(adapted))
    return adapted


__all__ = ["apply_lora"]
