"""Lightweight LoRA utilities that don't require torch at import time.

This module provides LoRA-related helper functions that can be imported
without triggering torch imports. This is critical for HPC job templates
that need to parse configs before the container starts.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from transformers import PreTrainedModel

# Import module to allow monkeypatching of apply_lora via module attribute
from . import lora


def _get_adapter_base_model(path: Path) -> Optional[str]:
    """Lazy import and call get_adapter_base_model."""
    from prefadap.models.loader import get_adapter_base_model
    return get_adapter_base_model(path)


def _freeze_peft_adapters_except(
    model: Any, active_adapter: str | Iterable[str]
) -> int:
    """Freeze parameters for all but the ``active_adapter`` names.

    Returns the number of parameters whose ``requires_grad`` flag was updated.
    """

    if isinstance(active_adapter, str):
        active_names: Tuple[str, ...] = (active_adapter,)
    else:
        active_names = tuple(active_adapter)

    frozen = 0

    modules = getattr(model, "modules", None)
    if not callable(modules):
        return frozen

    for module in modules():
        if module is None:
            continue
        for attr in dir(module):
            if not attr.startswith("lora_"):
                continue
            param_dict = getattr(module, attr, None)
            if param_dict is None:
                continue
            items = None
            if isinstance(param_dict, dict):
                items = param_dict.items()
            elif hasattr(param_dict, "items"):
                try:
                    items = param_dict.items()
                except TypeError:  # pragma: no cover - defensive
                    items = None
            if items is None:
                continue
            for name, param in items:
                if name in active_names:
                    should_train = True
                else:
                    should_train = False
                if hasattr(param, "requires_grad") and param.requires_grad != should_train:
                    param.requires_grad = should_train
                    frozen += 1

    return frozen


def _resolve_lora_base_identifier(args: Any, model: Any) -> Optional[str]:
    """Return the preferred base model identifier for LoRA adapters."""

    candidate = getattr(model, "base_model_name_or_path", None)
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()

    arg_source = None
    if args is not None:
        arg_source = getattr(args, "model_name_or_path", None)

    if arg_source is None:
        return None

    resolved: Optional[str] = None
    try:
        resolved = _get_adapter_base_model(Path(arg_source))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        resolved = None

    if resolved:
        return resolved

    if isinstance(arg_source, str) and arg_source.strip():
        return arg_source.strip()

    return None


def _is_sequence_classification_model(model: Any) -> bool:
    """Return True when ``model`` looks like a sequence-classification model."""

    config = getattr(model, "config", None)
    if config is None:
        return False
    if getattr(config, "num_labels", None) is None:
        return False
    if hasattr(model, "prepare_inputs_for_generation"):
        return False
    return True


def _resolve_lora_task_type(args: Any, model: "PreTrainedModel"):
    """Pick the PEFT task type for LoRA adapters based on the training role."""

    from peft import TaskType
    from prefadap.training.config import RMArgs

    experiment_type = getattr(args, "experiment_type", None)
    if isinstance(experiment_type, str) and experiment_type.lower() == "rm":
        # Reward models use sequence classification heads; PEFT must use SEQ_CLS to avoid generation hooks.
        return TaskType.SEQ_CLS
    if isinstance(args, RMArgs):
        # Reward models use sequence classification heads; PEFT must use SEQ_CLS to avoid generation hooks.
        return TaskType.SEQ_CLS
    if _is_sequence_classification_model(model):
        return TaskType.SEQ_CLS
    return TaskType.CAUSAL_LM


def _maybe_apply_lora(args: Any, model: "PreTrainedModel") -> "PreTrainedModel":
    """Return ``model`` with LoRA applied if requested in ``args``."""

    if not getattr(args, "lora", False):
        return model

    base_model_name = getattr(args, "model_name_or_path", None)
    resolved_base = _resolve_lora_base_identifier(args, model)
    if resolved_base:
        base_model_name = resolved_base
    has_existing_adapter = bool(getattr(model, "peft_config", None))
    task_type = _resolve_lora_task_type(args, model)

    if has_existing_adapter:
        policy = getattr(args, "lora_on_peft", None)
        if not policy:
            raise ValueError(
                "Model already contains LoRA adapters. Specify --lora_on_peft"
                " to choose how to handle staged training runs."
            )

        normalised = str(policy).replace("-", "_").lower()
        if normalised in {"merge_then_new", "merge_then_apply", "merge"}:
            if not hasattr(model, "merge_and_unload"):
                raise TypeError(
                    "Expected model with merge_and_unload() when handling existing LoRA adapters."
                )
            model = model.merge_and_unload()
            adapted = lora.apply_lora(
                model,
                target_modules=getattr(args, "lora_target_modules", None),
                r=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                base_model_name_or_path=base_model_name,
                task_type=task_type,
            )
            try:
                setattr(adapted, "_prefadap_merge_lora_on_save", True)
            except Exception:  # pragma: no cover - defensive
                pass
            return adapted

        if normalised in {"new_adapter", "add_adapter"}:
            try:  # pragma: no cover - import guard for optional dependency
                from peft import TaskType
            except ImportError as exc:  # pragma: no cover
                raise ImportError("Install 'peft' to add stacked LoRA adapters") from exc

            from prefadap.utils.lora_config import get_lora_config

            modules = lora._parse_target_modules(getattr(args, "lora_target_modules", None))
            lora_cfg = get_lora_config(
                r=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                target_modules=modules,
                task_type=task_type,
                base_model_name_or_path=base_model_name,
            )

            existing_names = list(getattr(model, "peft_config", {}).keys())
            stage_hint = getattr(args, "stage", None)
            adapter_name = getattr(args, "lora_adapter_name", None)
            if not adapter_name:
                if stage_hint is not None:
                    adapter_name = f"stage_{stage_hint}"
                else:
                    adapter_name = f"adapter_{len(existing_names) + 1}"

            add_adapter = getattr(model, "add_adapter", None)
            set_adapter = getattr(model, "set_adapter", None)
            if add_adapter is None or set_adapter is None:
                raise TypeError(
                    "Model with existing LoRA adapters must expose add_adapter/set_adapter when"
                    " using --lora_on_peft=new_adapter."
                )

            add_adapter(adapter_name, lora_cfg)
            set_adapter(adapter_name)

            train_adapter = getattr(model, "train_adapter", None)
            if callable(train_adapter):
                try:
                    train_adapter(adapter_name)
                except TypeError:  # pragma: no cover - handle list-only signatures
                    train_adapter([adapter_name])
            else:
                _freeze_peft_adapters_except(model, adapter_name)
            try:
                setattr(model, "_prefadap_merge_lora_on_save", False)
            except Exception:  # pragma: no cover - defensive
                pass
            return model

        raise ValueError(f"Unknown lora_on_peft policy: {policy}")

    adapted = lora.apply_lora(
        model,
        target_modules=getattr(args, "lora_target_modules", None),
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        base_model_name_or_path=base_model_name,
        task_type=task_type,
    )
    # For multi-stage training (stage > 1), we should merge adapters on save
    # because the loaded model might already have previous stage's changes merged in
    stage_hint = getattr(args, "stage", None)
    merge_on_save = stage_hint is not None and stage_hint > 1
    try:
        setattr(adapted, "_prefadap_merge_lora_on_save", merge_on_save)
    except Exception:  # pragma: no cover - defensive
        pass
    return adapted


# Alias for back-compat
maybe_apply_lora = _maybe_apply_lora

__all__ = [
    "_freeze_peft_adapters_except",
    "_resolve_lora_base_identifier",
    "_maybe_apply_lora",
    "maybe_apply_lora",
]
