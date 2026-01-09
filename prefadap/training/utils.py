"""Shared helper utilities for training pipelines.

This module contains utility functions and lightweight classes used across
the various training pipeline implementations.  The code was previously
embedded in :mod:`training_wrappers` but is now centralised here so that the
individual pipeline modules can import only what they need.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type, TypeVar, Union, List, Iterable, Tuple

from importlib import metadata, util, import_module

import csv
import json
import os
import time
import math
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import TrainerCallback, PreTrainedModel, GenerationConfig

import wandb
from trl import create_reference_model

from prefadap.utils.run_metadata import collect_run_metadata
from prefadap.utils.distributed import is_main_process
from prefadap.data.core import resolve_pseudo_data_path
from prefadap.models.loader import get_adapter_base_model

# Import module to allow monkeypatching of apply_lora via module attribute
from . import lora

# Constants for memory and model pattern detection
BYTES_TO_GB = 1024**3
MEMORY_THRESHOLD_BYTES = 16 * BYTES_TO_GB  # 16GB in bytes
MEMORY_INTENSIVE_PATTERNS = ["70b", "65b", "30b", "13b", "7b"]

# Model size thresholds (in billions of parameters)
LARGE_MODEL_THRESHOLD_B = 30  # Models >= 30B are considered large

# Module-level logger
logger = logging.getLogger(__name__)

ACCELERATE_MULTI_GPU_LAUNCH = (
    "accelerate launch \\\n"
    "  --num_processes <num_gpus> \\\n"
    "  -m prefadap.cli.run_training ppo --config <config.yaml>"
)


def log_liger_kernel_status(use_liger_kernel: bool, logger: logging.Logger | None = None) -> None:
    """Validate Liger Kernel availability and log the runtime status."""
    log = logger or logging.getLogger(__name__)
    if not use_liger_kernel:
        log.info("[liger] enabled=False")
        return

    if util.find_spec("liger_kernel") is None:
        raise ImportError(
            "use_liger_kernel=True but 'liger-kernel' is not installed. "
            "Install it with `pip install liger-kernel`."
        )

    import_module("liger_kernel")

    version = "unknown"
    try:
        version = metadata.version("liger-kernel")
    except metadata.PackageNotFoundError:
        try:
            version = metadata.version("liger_kernel")
        except metadata.PackageNotFoundError:
            version = "unknown"

    log.info("[liger] enabled=True (liger-kernel v%s)", version)


def accelerate_deepspeed_config_path() -> Path | None:
    """Return the Accelerate DeepSpeed config path when it exists."""
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "configs" / "accelerate_deepspeed_zero3.yaml"
    return path if path.exists() else None

# ---------------------------------------------------------------------------
# Picklable to_dict wrapper for TrainingArguments
# ---------------------------------------------------------------------------


class _PicklableToDictWrapper:
    """Picklable wrapper for TrainingArguments.to_dict() that ensures hub keys exist.
    
    This class wraps the original to_dict() method and ensures that hub-related keys
    (push_to_hub_token, hub_token, etc.) are always present in the returned dict,
    preventing KeyError crashes in TRL trainers.
    
    Unlike a local closure function, this class is picklable because:
    1. It's defined at module level (not inside another function)
    2. It stores references as instance attributes, not closure variables
    3. All attributes are picklable (config object, dict of defaults)
    
    Args:
        config: The TrainingArguments config object
        original_to_dict: The original to_dict method to wrap
        hub_keys_defaults: Dict of hub key names to their default values
    """
    
    def __init__(self, config, original_to_dict, hub_keys_defaults):
        self.config = config
        self.original_to_dict = original_to_dict
        self.hub_keys_defaults = hub_keys_defaults
    
    def __call__(self):
        """Call the original to_dict and ensure hub keys are present."""
        result = self.original_to_dict()
        # Ensure hub keys are in the dict even if not in official params
        for key, default_value in self.hub_keys_defaults.items():
            if key not in result:
                result[key] = getattr(self.config, key, default_value)
        return result
    
    def __reduce__(self):
        """Support pickling by returning constructor args.
        
        This is the key to making this wrapper picklable - we tell pickle
        how to reconstruct this object by providing the class and arguments.
        """
        return (
            _PicklableToDictWrapper,
            (self.config, self.original_to_dict, self.hub_keys_defaults)
        )


# ---------------------------------------------------------------------------
# Centralized Multi-GPU Sharding Helpers
# ---------------------------------------------------------------------------


def apply_default_sharding(kwargs: dict, model_name_or_path: str = None) -> dict:
    """Apply default sharding and dtype settings for multi-GPU model loading.
    
    This is the canonical function that establishes the multi-GPU sharding policy.
    All model loading code should use this to prepare kwargs before from_pretrained().
    
    Policy:
    - On multi-GPU systems (2+ GPUs): Always sets device_map="auto"
    - Sets torch_dtype=torch.bfloat16 for large models (≥20B) on multi-GPU
    - Preserves all user-provided kwargs
    
    Args:
        kwargs: User-provided kwargs for from_pretrained (will be modified in-place)
        model_name_or_path: Model name or path for size estimation (optional for backward compat)
        
    Returns:
        The modified kwargs dict (same object as input, for convenience)
        
    Example:
        >>> kwargs = {"trust_remote_code": True}
        >>> apply_default_sharding(kwargs, "outputs/Olmo-3-1125-32B/stage1/seed0")
        >>> model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    """
    # Always apply device_map="auto" on multi-GPU systems
    if (
        "device_map" not in kwargs
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
    ):
        kwargs["device_map"] = "auto"
        
        # Estimate model size to determine if we should set dtype optimization
        if model_name_or_path is not None:
            model_size_b = estimate_model_size_b(model_name_or_path)
            
            # Set dtype for large models if not already specified
            if model_size_b is not None and model_size_b >= 20.0:
                if "torch_dtype" not in kwargs:
                    kwargs["torch_dtype"] = torch.bfloat16
                
                dtype_name = kwargs.get("torch_dtype", "default")
                if hasattr(dtype_name, "__name__"):
                    dtype_name = dtype_name.__name__
                logger.info(
                    f"[multi-gpu] device_map='auto' applied for {model_size_b}B parameter model "
                    f"across {torch.cuda.device_count()} GPUs (dtype={dtype_name})"
                )
            else:
                logger.info(
                    f"[multi-gpu] device_map='auto' applied across {torch.cuda.device_count()} GPUs"
                )
        else:
            logger.info(
                f"[multi-gpu] device_map='auto' applied across {torch.cuda.device_count()} GPUs"
            )
    
    return kwargs


def safe_move_model(
    model: Any,
    device: Optional[str],
    model_type_name: str = "model"
) -> Any:
    """Safely move model to device, respecting existing device_map sharding.
    
    This is the canonical function for device placement. All code that needs to
    move a model to a device should use this instead of calling model.to() directly.
    
    Policy:
    - Detects device_map using get_effective_hf_device_map() (handles wrappers)
    - If device_map exists: logs info, skips movement (model is already sharded)
    - If no device_map and device specified: performs model.to(device) and sets _to_called flag
    - If no device_map and no device: does nothing
    
    Note: This function does NOT set model.device as models may be sharded/distributed
    and do not have a single device. Use next(model.parameters()).device or 
    get_effective_hf_device_map() to inspect device placement.
    
    Args:
        model: Model to potentially move
        device: Target device (e.g., "cuda", "cpu", "cuda:0") or None
        model_type_name: Human-readable name for logging (e.g., "model", "reward model")
        
    Returns:
        The model (potentially moved to device)
        
    Example:
        >>> model = load_model(...)
        >>> safe_move_model(model, "cuda:0", "policy model")
    """
    dm = get_effective_hf_device_map(model)
    if dm:
        logger.info("Detected hf_device_map; skipping model.to(%s)", device)
        # Model is already sharded - do not move it
        return model

    if hasattr(model, "to") and device is not None:
        result = model.to(device)
        # Set _to_called flag for tests
        if hasattr(model, "_to_called"):
            model._to_called = True
        return result
    return model


def get_effective_hf_device_map(model: Any) -> Optional[Dict[str, Any]]:
    """
    Return the most relevant hf_device_map for a (possibly wrapped) model.

    Large models loaded with device_map="auto" are sharded across GPUs by the
    HuggingFace loader. However, when wrapped with LoRA (via PEFT's PeftModel)
    or TRL's AutoModelForCausalLMWithValueHead, the outer wrapper may not expose
    the hf_device_map attribute, causing the pipeline to mistakenly think the
    model is single-GPU and call .to(device), which collapses all shards onto
    GPU0 and causes OOM.

    This helper checks multiple levels to find the effective device_map:
    - Check the model itself first (only if dict).
    - Check PEFT wrapper base_model attribute.
    - Check TRL wrapper pretrained_model attribute.
    - If nothing is found, return None.

    Args:
        model: The model to inspect (may be wrapped by PEFT, TRL, or both)

    Returns:
        The hf_device_map dict if found on any layer, or None if not found.
        Only returns a value if it's a dict (the expected type for multi-GPU sharding).
        Device map values can be int (GPU index), str (device name like 'cpu', 'cuda:0'),
        or torch.device objects.
    """
    # 1. Check if model has hf_device_map attribute
    if not hasattr(model, "hf_device_map"):
        # Model doesn't have hf_device_map, check if it's a wrapper with inner model
        
        # Check 2a: PEFT wrapper (base_model pattern)
        if hasattr(model, "base_model"):
            base = model.base_model
            if hasattr(base, "hf_device_map") and isinstance(base.hf_device_map, dict):
                return base.hf_device_map
        
        # Check 2b: TRL wrapper (pretrained_model pattern)
        if hasattr(model, "pretrained_model"):
            inner = model.pretrained_model
            inner_dm = getattr(inner, "hf_device_map", None)
            if isinstance(inner_dm, dict):
                return inner_dm
        
        # Not a wrapper or no device map found
        return None
    
    # Model has hf_device_map attribute, check if it's a dict
    dm = model.hf_device_map
    if isinstance(dm, dict):
        return dm
    
    # hf_device_map exists but is not a dict (e.g., string "auto")
    # Check if there's an inner model with actual dict device map
    
    # Check PEFT wrapper first
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "hf_device_map") and isinstance(base.hf_device_map, dict):
            return base.hf_device_map
    
    # Check TRL wrapper
    if hasattr(model, "pretrained_model"):
        inner = model.pretrained_model
        inner_dm = getattr(inner, "hf_device_map", None)
        if isinstance(inner_dm, dict):
            return inner_dm
    
    return None


def resolve_optional_pseudo_data_path(
    path: str | os.PathLike[str] | os.PathLike[bytes] | None,
    *,
    env_var: str = "PSEUDO_DATA_PATH",
    description: str = "pseudo dataset path",
) -> str | None:
    """Return a resolved pseudo dataset path if configured."""

    if path is None and os.environ.get(env_var) is None:
        return None
    return resolve_pseudo_data_path(path, env_var=env_var, description=description)


def _maybe_init_wandb(
    project: str,
    name: str,
    enabled: bool,
    config: Optional[Union[dict, object]] = None,
    tags: Optional[List[str]] = None,
) -> bool:
    """Initialise Weights & Biases if ``enabled``.

    When disabled the ``WANDB_DISABLED`` environment variable is set so that
    downstream libraries are aware that logging should be skipped.

    This function supports auto-login via the WANDB_API_KEY environment variable.
    If WANDB_API_KEY is set, it will be used to authenticate with W&B without
    requiring interactive login.

    Args:
        project: W&B project name
        name: Run name
        enabled: Whether to enable W&B logging
        config: Configuration object to log as hyperparameters
        tags: Optional list of tags for the run
    """

    if enabled:
        if not is_main_process():
            # Avoid duplicate W&B runs in DDP by disabling logging on non-zero ranks.
            os.environ["WANDB_DISABLED"] = "true"
            if wandb is not None:
                try:
                    wandb.init(mode="disabled")
                except Exception as e:
                    logger.warning(f"W&B disabled init failed on worker rank: {e}")
            return False

        # Check if wandb is available
        if wandb is None:
            logger.warning("wandb not available, disabling wandb logging")
            os.environ["WANDB_DISABLED"] = "true"
            return False
        
        # Auto-login from WANDB_API_KEY environment variable if set
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
            try:
                wandb.login(key=wandb_api_key, relogin=False)
                logger.debug("W&B authenticated via WANDB_API_KEY environment variable")
            except Exception as e:
                logger.warning(f"W&B auto-login from WANDB_API_KEY failed: {e}")
        
        # Convert config to dict for W&B if provided
        config_dict = None
        if config is not None:
            if hasattr(config, "__dict__"):
                # Filter to only include basic serializable types
                config_dict = {}
                for k, v in config.__dict__.items():
                    if not k.startswith("_") and not callable(v):
                        # Only include basic JSON-serializable types
                        if isinstance(v, (str, int, float, bool, type(None))):
                            config_dict[k] = v
                        elif isinstance(v, (list, tuple)):
                            # Include lists/tuples if all elements are basic types
                            if all(
                                isinstance(item, (str, int, float, bool, type(None)))
                                for item in v
                            ):
                                config_dict[k] = list(
                                    v
                                )  # Convert tuples to lists for JSON
                        elif isinstance(v, dict):
                            # Include dicts if all values are basic types
                            if all(
                                isinstance(val, (str, int, float, bool, type(None)))
                                for val in v.values()
                            ):
                                config_dict[k] = v
            elif isinstance(config, dict):
                # For dict inputs, apply the same filtering
                config_dict = {}
                for k, v in config.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        config_dict[k] = v
                    elif isinstance(v, (list, tuple)):
                        if all(
                            isinstance(item, (str, int, float, bool, type(None)))
                            for item in v
                        ):
                            config_dict[k] = list(v)
                    elif isinstance(v, dict):
                        if all(
                            isinstance(val, (str, int, float, bool, type(None)))
                            for val in v.values()
                        ):
                            config_dict[k] = v

        wandb.init(project=project, name=name, config=config_dict, tags=tags)
        return True
    os.environ["WANDB_DISABLED"] = "true"
    return False


# ---------------------------------------------------------------------------
# Penalty-aware trainers
# ---------------------------------------------------------------------------


"""Helper functions and lightweight utilities."""


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
        resolved = get_adapter_base_model(Path(arg_source))
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


def _resolve_lora_task_type(args: Any, model: PreTrainedModel):
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


def _maybe_apply_lora(args: Any, model: PreTrainedModel) -> PreTrainedModel:
    """Return ``model`` with LoRA applied if requested in ``args``."""

    if not getattr(args, "lora", False):
        logger.info("LoRA disabled; skipping adapter injection.")
        return model

    base_model_name = getattr(args, "model_name_or_path", None)
    resolved_base = _resolve_lora_base_identifier(args, model)
    if resolved_base:
        base_model_name = resolved_base
    has_existing_adapter = bool(getattr(model, "peft_config", None))
    logger.info(
        "LoRA requested; existing_adapter=%s base_model_name=%s",
        has_existing_adapter,
        base_model_name,
    )
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
            logger.info("LoRA policy merge_then_new; merging existing adapters before applying new LoRA.")
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
            logger.info("LoRA policy new_adapter; adding new adapter to existing PEFT model.")
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

    logger.info("Applying LoRA to base model (no existing adapters detected).")
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


def _maybe_enable_gradient_checkpointing(
    args: Any, model: PreTrainedModel
) -> PreTrainedModel:
    """Enable gradient checkpointing on ``model`` if requested or needed for memory."""

    # Check if explicitly requested
    gradient_checkpointing = getattr(args, "gradient_checkpointing", False)

    # If not explicitly set, check if we should enable it automatically
    if not gradient_checkpointing:
        gradient_checkpointing = _should_enable_gradient_checkpointing_automatically(
            args, model
        )

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled to reduce memory footprint")
    return model


def _should_enable_gradient_checkpointing_automatically(
    args: Any, model: PreTrainedModel
) -> bool:
    """Determine if gradient checkpointing should be enabled automatically based on memory constraints."""

    # Enable for large models (>1B parameters)
    if hasattr(model, "num_parameters"):
        try:
            param_count = model.num_parameters()
            if param_count > 1_000_000_000:  # 1B parameters
                return True
        except Exception:
            pass

    # Enable if CUDA memory is limited (less than 16GB available)
    if torch.cuda.is_available():
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = total_memory - torch.cuda.memory_allocated()
            # Enable if less than 16GB available
            if available_memory < MEMORY_THRESHOLD_BYTES:
                return True
        except Exception:
            pass

    # Enable for specific model patterns known to be memory-intensive
    model_name = getattr(args, "model_name_or_path", "").lower()
    if any(pattern in model_name for pattern in MEMORY_INTENSIVE_PATTERNS):
        return True

    return False


def _maybe_freeze_embeddings(args: Any, model: PreTrainedModel) -> PreTrainedModel:
    """Freeze input embeddings and optionally normalisation layers.

    - If ``args.freeze_embeddings`` is True, freezes input embedding parameters.
    - If ``args.freeze_norm_layers`` is True (default), freezes parameters of
      normalisation layers (modules whose class name contains "Norm" or whose
      attribute name contains "norm").
    """

    if getattr(args, "freeze_embeddings", False):
        embed_fn = getattr(model, "get_input_embeddings", None)
        if callable(embed_fn):
            emb_module = embed_fn()
            for param in emb_module.parameters():
                param.requires_grad = False

    if getattr(args, "freeze_norm_layers", True):
        try:
            norm_layer_names = {"LayerNorm", "BatchNorm", "GroupNorm", "RMSNorm"}
            for name, module in model.named_modules():  # type: ignore[attr-defined]
                cls = module.__class__.__name__
                if cls in norm_layer_names:
                    for p in getattr(module, "parameters", lambda: [])():
                        p.requires_grad = False
        except Exception:
            # Be best-effort if model doesn't expose named_modules in tests
            pass
    return model


def _tune_70b_defaults(args: Any) -> None:
    """Adjust common training arguments for 70B QLoRA models.

    Training 70B models with FSDP on hardware assumes a very small
    per-device memory budget. When ``args.tune_large_defaults`` is ``True`` and
    ``args.fsdp`` is active, this helper applies a set of defaults tailored for
    that environment: the micro batch size is capped at one, LoRA rank is
    increased and gradient checkpointing is enabled. A DeepSpeed configuration
    is no longer injected automatically.
    """

    if not getattr(args, "tune_large_defaults", False):
        return

    model_name = getattr(args, "model_name_or_path", "").lower()
    if "70b" not in model_name or not getattr(args, "fsdp", False):
        return

    logger = logging.getLogger(__name__)
    overrides: list[tuple[str, Any, Any]] = []

    if getattr(args, "per_device_train_batch_size", 1) > 1:
        old = getattr(args, "per_device_train_batch_size")
        args.per_device_train_batch_size = 1
        overrides.append(("per_device_train_batch_size", old, 1))

    if getattr(args, "lora_r", 8) == 8:
        old = getattr(args, "lora_r")
        args.lora_r = 64
        overrides.append(("lora_r", old, 64))

    if getattr(args, "lora_alpha", 32) == 32:
        old = getattr(args, "lora_alpha")
        args.lora_alpha = 16
        overrides.append(("lora_alpha", old, 16))

    if not getattr(args, "gradient_checkpointing", False):
        old = getattr(args, "gradient_checkpointing", False)
        args.gradient_checkpointing = True
        overrides.append(("gradient_checkpointing", old, True))

    for name, old, new in overrides:
        logger.info("Overriding %s: %s -> %s", name, old, new)


def dump_indices(
    save: bool, train_idx: list[str], eval_idx: list[str], output_dir: str
) -> None:
    """Write SHA256 indices to JSONL files if ``save`` is ``True``."""

    if not save:
        return
    os.makedirs(output_dir, exist_ok=True)

    def _write(path: str, indices: list[str]) -> None:
        with open(path, "w") as f:
            for i, sha in enumerate(indices):
                json.dump({"index": i, "sha256": sha}, f)
                f.write("\n")

    if train_idx:
        _write(os.path.join(output_dir, "train_indices.jsonl"), train_idx)
    if eval_idx:
        _write(os.path.join(output_dir, "eval_indices.jsonl"), eval_idx)


def _ensure_adapter_metadata(output_dir: str, trainer, model) -> None:
    """Ensure saved adapters record their base model for downstream tooling."""

    adapter_config = Path(output_dir) / "adapter_config.json"
    if not adapter_config.exists():
        return

    try:
        data = json.loads(adapter_config.read_text())
    except json.JSONDecodeError:
        return

    base = data.get("base_model_name_or_path") or data.get("base_model")
    if isinstance(base, str) and base.strip():
        return

    args = getattr(trainer, "args", None)
    candidate = _resolve_lora_base_identifier(args, model)

    if isinstance(candidate, str) and candidate.strip():
        data["base_model_name_or_path"] = candidate.strip()
        try:
            adapter_config.write_text(json.dumps(data, indent=2, sort_keys=True))
        except Exception:
            pass


def _ensure_adapter_safetensors(output_dir: str) -> None:
    """Convert legacy adapter weights to safetensors when possible.

    Older PEFT releases emitted ``adapter_model.bin`` (or ``adapter_model.pt``)
    when ``safe_serialization`` was unavailable.  Downstream tooling in this
    repository expects ``adapter_model.safetensors`` to detect saved adapters.
    This helper converts legacy artifacts to the safetensors format when both
    ``torch`` and ``safetensors`` are available.
    """

    adapter_path = Path(output_dir) / "adapter_model.safetensors"
    if adapter_path.exists():
        return

    legacy_candidates = [
        Path(output_dir) / "adapter_model.bin",
        Path(output_dir) / "adapter_model.pt",
    ]
    legacy_path = next((candidate for candidate in legacy_candidates if candidate.exists()), None)
    if legacy_path is None:
        return

    try:  # Import lazily so environments without safetensors still work.
        from safetensors.torch import save_file as safetensors_save_file  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning(
            "PEFT adapter saved as %s but safetensors support is unavailable: %s",
            legacy_path.name,
            exc,
        )
        return

    try:
        state_dict = torch.load(str(legacy_path), map_location="cpu")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Failed to load legacy adapter weights from %s for safetensor conversion: %s",
            legacy_path,
            exc,
        )
        return

    try:
        safetensors_save_file(state_dict, str(adapter_path))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Failed to convert %s to safetensors format: %s",
            legacy_path,
            exc,
        )
        return

    logger.info(
        "Converted %s to adapter_model.safetensors for compatibility", legacy_path.name
    )


def save_model_with_lora(trainer, output_dir: str) -> None:
    """Save a model, handling LoRA adapters if present.

    Args:
        trainer: Trainer object with model to save
        output_dir: Directory to save the model
    """
    from peft import PeftModel

    model = getattr(trainer, "model", None)
    merge_on_save = bool(getattr(model, "_prefadap_merge_lora_on_save", False))

    if isinstance(model, PeftModel):
        if merge_on_save and hasattr(model, "merge_and_unload"):
            logger.info("Merging LoRA adapters into the base model before saving")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(output_dir, safe_serialization=True)
            return

        try:
            model.save_pretrained(output_dir, safe_serialization=True)
        except TypeError:
            # Some historical PEFT versions did not accept the safe_serialization
            # keyword.  Retry without it to remain compatible.
            model.save_pretrained(output_dir)
            _ensure_adapter_metadata(output_dir, trainer, model)
            _ensure_adapter_safetensors(output_dir)
        except UnboundLocalError as exc:
            logger.warning(
                "PEFT save_pretrained failed due to missing active adapters; "
                "retrying after merging adapters. Error: %s",
                exc,
            )
            if not hasattr(model, "merge_and_unload"):
                raise
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(output_dir, safe_serialization=True)
        else:
            _ensure_adapter_metadata(output_dir, trainer, model)
            _ensure_adapter_safetensors(output_dir)
        return

    trainer.save_model(output_dir)

    # Generate model card automatically
    try:
        from prefadap.models.model_card import generate_model_card_from_context

        # Extract config from trainer if available
        config = getattr(trainer, "args", None)
        tokenizer = getattr(trainer, "tokenizer", None)

        generate_model_card_from_context(
            output_dir=output_dir,
            config=config,
            trainer=trainer,
            tokenizer=tokenizer,
        )
    except Exception as e:
        # Don't fail the model save if model card generation fails
        logging.error("Model card generation failed: %s", e, exc_info=True)


class TokenKLCallback(TrainerCallback):
    """Track token counts, divergence metrics and wall-clock time."""

    def __init__(
        self,
        *,
        log_token_count: bool,
        log_kl: bool,
        use_wandb: bool,
        ref_model: Optional[PreTrainedModel] = None,
        logger: Optional[Any] = None,
        output_dir: str = ".",
        seed: int = 0,
        metric_logging_steps: int = 50,
        system_logging_steps: int = 500,
    ) -> None:
        self.log_token_count = log_token_count
        self.log_kl = log_kl and ref_model is not None
        self.use_wandb = use_wandb
        self.ref_model = ref_model
        self.logger = logger
        self.output_dir = output_dir
        self.seed = seed
        self.metric_logging_steps = metric_logging_steps
        self.system_logging_steps = system_logging_steps
        self.token_counts: Dict[str, int] = {}
        self.kl_sums: Dict[str, float] = {}
        self.kl_counts: Dict[str, int] = {}
        self.start_time: Optional[float] = None
        self.cumulative_tokens: int = 0
        self.steps: int = 0
        self.last_metric_log: int = 0
        self.last_system_log: int = 0

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _tokens_in_inputs(inputs: Dict[str, Any]) -> int:
        if "input_ids" in inputs:
            return inputs["input_ids"].numel()
        total = 0
        for key in ("prompt_input_ids", "chosen_input_ids", "rejected_input_ids"):
            if key in inputs:
                total += inputs[key].numel()
        return total

    @staticmethod
    def _log_ratio_gap(
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        inputs: Dict[str, Any],
    ) -> float:
        # Compute log-ratio gap for DPO style batches.
        if (
            not {"prompt_input_ids", "chosen_input_ids", "rejected_input_ids"}
            <= inputs.keys()
        ):
            return 0.0
        
        # Get device for policy model and reference model (may be different)
        # Use next(model.parameters()).device for sharded/wrapped models
        policy_device = next(model.parameters()).device
        ref_device = next(ref_model.parameters()).device
        
        # Prepare tensors for policy model (on policy device)
        prompt = inputs["prompt_input_ids"].to(policy_device)
        chosen = inputs["chosen_input_ids"].to(policy_device)
        rejected = inputs["rejected_input_ids"].to(policy_device)
        p_mask = inputs.get("prompt_attention_mask")
        c_mask = inputs.get("chosen_attention_mask")
        r_mask = inputs.get("rejected_attention_mask")
        if p_mask is None:
            p_mask = torch.ones_like(prompt)
        else:
            p_mask = p_mask.to(policy_device)
        if c_mask is None:
            c_mask = torch.ones_like(chosen)
        else:
            c_mask = c_mask.to(policy_device)
        if r_mask is None:
            r_mask = torch.ones_like(rejected)
        else:
            r_mask = r_mask.to(policy_device)
        prompt_lens = p_mask.sum(dim=1)

        def _concat(a, b):
            return torch.cat([a, b], dim=1)

        chosen_ids = _concat(prompt, chosen)
        chosen_mask = _concat(p_mask, c_mask)
        rejected_ids = _concat(prompt, rejected)
        rejected_mask = _concat(p_mask, r_mask)

        def _logps(model, ids, mask, prompt_lens, target_device):
            # Move tensors to the model's device if needed
            if ids.device != target_device:
                ids = ids.to(target_device)
                mask = mask.to(target_device)
                prompt_lens = prompt_lens.to(target_device)
            
            logits = model(ids, attention_mask=mask).logits
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            target = ids[:, 1:]
            gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            gathered = gathered * mask[:, 1:]
            idx = torch.arange(gathered.size(1), device=gathered.device).unsqueeze(0)
            after_prompt = idx >= (prompt_lens.unsqueeze(1) - 1)
            gathered = gathered * after_prompt.to(gathered.dtype)
            return gathered.sum(-1)

        model_was_training = model.training
        ref_was_training = ref_model.training
        try:
            model.eval()
            ref_model.eval()
            with torch.no_grad():
                pol_chosen = _logps(model, chosen_ids, chosen_mask, prompt_lens, policy_device)
                pol_rejected = _logps(model, rejected_ids, rejected_mask, prompt_lens, policy_device)
                ref_chosen = _logps(ref_model, chosen_ids, chosen_mask, prompt_lens, ref_device)
                ref_rejected = _logps(
                    ref_model, rejected_ids, rejected_mask, prompt_lens, ref_device
                )
                # Move ref outputs back to policy device for computation
                ref_chosen = ref_chosen.to(policy_device)
                ref_rejected = ref_rejected.to(policy_device)
        finally:
            model.train(model_was_training)
            ref_model.train(ref_was_training)
        gap = (pol_chosen - pol_rejected) - (ref_chosen - ref_rejected)
        return gap.mean().item()

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------
    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return control

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
        self.steps = state.global_step
        return control

    def on_log(self, args, state, control, model=None, **kwargs):  # type: ignore[override]
        logs = kwargs.get("logs", {})
        inputs = logs.get("inputs")
        domain = logs.get("domain", "train")
        current_step = state.global_step

        # Check if we should log metrics at this step
        should_log_metrics = (
            current_step - self.last_metric_log >= self.metric_logging_steps
        )
        metrics_logged = False

        # Log token counts with metric interval
        if self.log_token_count and inputs is not None:
            tokens = self._tokens_in_inputs(inputs)
            self.cumulative_tokens += tokens
            self.token_counts[domain] = self.token_counts.get(domain, 0) + tokens

            # Use metric logging interval for frequent metrics
            if should_log_metrics:
                if self.use_wandb and wandb is not None:
                    wandb.log(
                        {
                            f"metrics/tokens_{domain}": self.token_counts[domain],
                            "metrics/tokens_total": self.cumulative_tokens,
                            f"metrics/tokens_per_step_{domain}": tokens,
                        },
                        step=current_step,
                    )
                elif self.logger:
                    self.logger.info(f"tokens_{domain}={self.token_counts[domain]}")
                metrics_logged = True

        # Log KL divergence with metric interval
        if self.log_kl and model is not None and self.ref_model is not None:
            if {
                "prompt_input_ids",
                "chosen_input_ids",
                "rejected_input_ids",
            } <= inputs.keys():
                value = self._log_ratio_gap(model, self.ref_model, inputs)
                key = "log_ratio_gap"
            elif "input_ids" in inputs:
                # Get devices for policy and reference models
                # Use next(model.parameters()).device for sharded/wrapped models
                policy_device = next(model.parameters()).device
                ref_device = next(self.ref_model.parameters()).device
                
                # Prepare tensors on policy device first
                ids = inputs["input_ids"].to(policy_device)
                attn = inputs.get("attention_mask")
                if attn is not None:
                    attn = attn.to(policy_device)
                
                model_was_training = model.training
                ref_was_training = self.ref_model.training
                try:
                    model.eval()
                    self.ref_model.eval()
                    with torch.no_grad():
                        logits_model = model(input_ids=ids, attention_mask=attn).logits
                        
                        # Move tensors to reference model's device if different
                        if ref_device != policy_device:
                            ids_ref = ids.to(ref_device)
                            attn_ref = attn.to(ref_device) if attn is not None else None
                            logits_ref = self.ref_model(
                                input_ids=ids_ref, attention_mask=attn_ref
                            ).logits
                            # Move ref logits back to policy device for KL computation
                            logits_ref = logits_ref.to(policy_device)
                        else:
                            logits_ref = self.ref_model(
                                input_ids=ids, attention_mask=attn
                            ).logits
                        
                        value = F.kl_div(
                            logits_model.log_softmax(-1),
                            logits_ref.log_softmax(-1),
                            reduction="batchmean",
                            log_target=True,
                        ).item()
                finally:
                    model.train(model_was_training)
                    self.ref_model.train(ref_was_training)
                key = "kl_div"
            else:
                value = 0.0
                key = "kl_div"
            self.kl_sums[domain] = self.kl_sums.get(domain, 0.0) + value
            self.kl_counts[domain] = self.kl_counts.get(domain, 0) + 1

            # Use metric logging interval for KL metrics
            if should_log_metrics:
                if self.use_wandb and wandb is not None:
                    wandb.log({f"metrics/{key}_{domain}": value}, step=current_step)
                elif self.logger:
                    self.logger.info(f"{key}_{domain}={value}")
                metrics_logged = True

        # Update last metric log time if any metrics were logged
        if metrics_logged:
            self.last_metric_log = current_step

        # Log system metrics with system interval
        if (
            current_step - self.last_system_log >= self.system_logging_steps
            and torch.cuda.is_available()
        ):
            if self.use_wandb and wandb is not None:
                system_metrics = {
                    "system/gpu_memory_allocated_gb": torch.cuda.memory_allocated()
                    / BYTES_TO_GB,
                    "system/gpu_memory_reserved_gb": torch.cuda.memory_reserved()
                    / BYTES_TO_GB,
                }
                if self.start_time is not None:
                    elapsed = time.time() - self.start_time
                    system_metrics["system/elapsed_hours"] = elapsed / 3600
                    system_metrics["system/tokens_per_second"] = (
                        self.cumulative_tokens / elapsed if elapsed > 0 else 0
                    )
                wandb.log(system_metrics, step=current_step)
            self.last_system_log = current_step

        return control

    def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
        end_time = time.time()
        elapsed = end_time - (self.start_time or end_time)
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpu_hours = elapsed / 3600 * num_gpus
        max_mem = (
            torch.cuda.max_memory_allocated() / BYTES_TO_GB
            if torch.cuda.is_available()
            else 0.0
        )

        avg_kl = {
            k: (self.kl_sums[k] / self.kl_counts[k]) if self.kl_counts.get(k) else 0.0
            for k in self.kl_sums
        }
        stats: Dict[str, Any] = {
            "tokens": self.token_counts,
            "avg_kl": avg_kl,
            "wall_clock_seconds": elapsed,
            "gpu_hours": gpu_hours,
            "max_memory_gb": max_mem,
        }

        os.makedirs(self.output_dir, exist_ok=True)
        json_path = os.path.join(self.output_dir, "training_stats.json")
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)
        csv_path = os.path.join(self.output_dir, "training_stats.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for d, v in self.token_counts.items():
                writer.writerow([f"tokens_{d}", v])
            for d, v in avg_kl.items():
                writer.writerow([f"avg_kl_{d}", v])
            writer.writerow(["wall_clock_seconds", elapsed])
            writer.writerow(["gpu_hours", gpu_hours])
            writer.writerow(["max_memory_gb", max_mem])

        if self.use_wandb and wandb is not None:
            flat = {f"final/tokens_{k}": v for k, v in self.token_counts.items()}
            flat.update({f"final/avg_kl_{k}": v for k, v in avg_kl.items()})
            flat.update(
                {
                    "final/wall_clock_seconds": elapsed,
                    "final/gpu_hours": gpu_hours,
                    "final/max_memory_gb": max_mem,
                }
            )
            wandb.log(flat, step=state.global_step)
        if self.logger:
            self.logger.info(f"Training statistics saved to {json_path}")

        metadata = collect_run_metadata(
            self.token_counts,
            avg_kl,
            self.steps,
            self.start_time or end_time,
            self.seed,
        )
        meta_path = os.path.join(self.output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return control


# ---------------------------------------------------------------------------
# Early stopping utilities
# ---------------------------------------------------------------------------


class PerplexityThresholdCallback(TrainerCallback):
    """Stop training when perplexity falls below a threshold."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs or "loss" not in logs:
            return
        ppl = math.exp(logs["loss"])
        if ppl <= self.threshold:
            control.should_save = True
            control.should_training_stop = True


# ---------------------------------------------------------------------------
# Additional helpers
# ---------------------------------------------------------------------------


T = TypeVar("T")


def resolve_device(device: str | torch.device | None = None) -> str:
    """Resolve ``device`` to an actual torch device string.
    
    This function is rank-aware and will select the appropriate device based on
    distributed training environment variables (LOCAL_RANK, RANK, WORLD_SIZE).
    When running in a distributed environment, it maps each rank to its corresponding GPU.

    Parameters
    ----------
    device:
        Device hint from the configuration. ``"auto"`` or ``None`` triggers
        auto-detection, choosing ``"cuda"`` when any CUDA/ROCm GPU is available
        and ``"cpu"`` otherwise. In distributed training, this will select the
        appropriate GPU for the current rank.
    """

    requested = None if device is None else str(device)
    
    # Auto-detect device with rank awareness for distributed training
    if requested is None or requested.lower() in {"", "auto"}:
        if torch.cuda.is_available():
            # Check for distributed training environment variables
            local_rank = os.environ.get("LOCAL_RANK")
            rank = os.environ.get("RANK")
            world_size = os.environ.get("WORLD_SIZE")
            
            # If LOCAL_RANK is set, use it to select the GPU
            if local_rank is not None:
                try:
                    device_id = int(local_rank)
                    gpu_count = torch.cuda.device_count()
                    if device_id < gpu_count:
                        resolved = f"cuda:{device_id}"
                        logger.info(
                            f"Distributed training detected: LOCAL_RANK={local_rank}, RANK={rank}, "
                            f"WORLD_SIZE={world_size}, using device={resolved}, "
                            f"total GPUs available={gpu_count}"
                        )
                    else:
                        logger.warning(
                            f"LOCAL_RANK={local_rank} exceeds available GPUs ({gpu_count}), "
                            f"falling back to cuda:0"
                        )
                        resolved = "cuda:0"
                except ValueError:
                    logger.warning(f"Invalid LOCAL_RANK value: {local_rank}, using cuda:0")
                    resolved = "cuda:0"
            else:
                # Single-GPU or single-node training
                resolved = "cuda:0"
                gpu_count = torch.cuda.device_count()
                logger.info(f"Single-node training: using device=cuda:0, total GPUs available={gpu_count}")
        else:
            resolved = "cpu"
            logger.info("No CUDA devices available, using CPU")
    else:
        resolved = requested

    if resolved.startswith("cuda") and not torch.cuda.is_available():
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
        logger.warning(
            "device '%s' requested but CUDA is not available. CUDA_VISIBLE_DEVICES=%s, torch.version.cuda=%s",
            resolved,
            cuda_visible,
            torch.version.cuda,
        )
    if resolved.startswith("cpu") and torch.cuda.is_available():
        logger.warning(
            "CUDA is available but '%s' was selected. Set device='cuda' to use the GPU.",
            resolved,
        )

    return resolved


def is_distributed_environment() -> bool:
    """Return True when environment variables indicate distributed training."""
    local_rank = os.environ.get("LOCAL_RANK")
    world_size = os.environ.get("WORLD_SIZE")
    accelerate_process_count = os.environ.get("ACCELERATE_PROCESS_COUNT")

    try:
        world_size_val = int(world_size) if world_size is not None else 1
    except ValueError:
        world_size_val = 1
    try:
        accelerate_process_count_val = (
            int(accelerate_process_count) if accelerate_process_count is not None else 1
        )
    except ValueError:
        accelerate_process_count_val = 1

    return (
        world_size_val > 1
        or accelerate_process_count_val > 1
        or local_rank is not None
    )


def _resolve_deepspeed_config_file(state, deepspeed_plugin) -> str | None:
    ds_config_file = None
    if deepspeed_plugin is not None:
        ds_config_file = getattr(deepspeed_plugin, "deepspeed_config_file", None)
        if ds_config_file is None:
            ds_config = getattr(deepspeed_plugin, "deepspeed_config", None)
            if isinstance(ds_config, dict):
                ds_config_file = ds_config.get("deepspeed_config_file")

    if ds_config_file is None:
        state_config = getattr(state, "config", None)
        if isinstance(state_config, dict):
            ds_config_section = state_config.get("deepspeed_config", None)
        else:
            ds_config_section = getattr(state_config, "deepspeed_config", None)
        if isinstance(ds_config_section, dict):
            ds_config_file = ds_config_section.get("deepspeed_config_file")

    return ds_config_file


def _validate_single_gpu_baseline(experiment_name: str) -> None:
    """Ensure PPO/GRPO training runs on a true single-GPU, non-DeepSpeed path."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{experiment_name} single-GPU baseline requires CUDA but none was detected."
        )

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_count = torch.cuda.device_count()
    if device_count != 1:
        raise RuntimeError(
            f"{experiment_name} single-GPU baseline requires exactly one visible CUDA "
            f"device. torch.cuda.device_count()={device_count}, "
            f"CUDA_VISIBLE_DEVICES={cuda_visible or '(not set)'}."
        )

    if torch.distributed.is_initialized():
        raise RuntimeError(
            f"{experiment_name} single-GPU baseline forbids torch.distributed "
            "initialization, but a process group is already active."
        )

    if is_distributed_environment():
        raise RuntimeError(
            f"{experiment_name} single-GPU baseline requires WORLD_SIZE=1 and no "
            "distributed launch variables. Detected distributed environment variables."
        )

    from accelerate import Accelerator
    from accelerate.state import DistributedType

    accelerator = Accelerator()
    state = accelerator.state
    deepspeed_plugin = getattr(state, "deepspeed_plugin", None)
    deepspeed_config_file = _resolve_deepspeed_config_file(state, deepspeed_plugin)

    if (
        state.distributed_type == DistributedType.DEEPSPEED
        or deepspeed_plugin is not None
        or deepspeed_config_file is not None
    ):
        raise RuntimeError(
            f"{experiment_name} single-GPU baseline forbids DeepSpeed. "
            f"Distributed type={state.distributed_type}, "
            f"deepspeed_plugin={'present' if deepspeed_plugin else 'none'}, "
            f"deepspeed_config_file={deepspeed_config_file}, "
            f"ACCELERATE_CONFIG_FILE={os.environ.get('ACCELERATE_CONFIG_FILE')}."
        )

    if state.num_processes != 1:
        raise RuntimeError(
            f"{experiment_name} single-GPU baseline requires a single process. "
            f"Accelerate reports num_processes={state.num_processes}."
        )

    logger.info(
        "%s single-GPU baseline validated: DeepSpeed disabled, world size 1, "
        "CUDA_VISIBLE_DEVICES=%s.",
        experiment_name,
        cuda_visible or "(not set)",
    )


def validate_ppo_single_gpu_baseline(experiment_name: str = "PPO") -> None:
    """Ensure PPO-style training runs on a true single-GPU, non-DeepSpeed path."""
    _validate_single_gpu_baseline(experiment_name)


def validate_grpo_single_gpu_baseline(experiment_name: str = "GRPO") -> None:
    """Ensure GRPO-style training runs on a true single-GPU, non-DeepSpeed path."""
    _validate_single_gpu_baseline(experiment_name)


def validate_ppo_distributed_launch(
    cfg: Any | None = None, experiment_name: str = "PPO/GRPO"
) -> None:
    """Ensure PPO-style training uses multi-process distributed execution."""
    if cfg is not None and getattr(cfg, "ppo_single_gpu_baseline", False):
        deepspeed_enabled = is_deepspeed_enabled()
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        assert not deepspeed_enabled, (
            "PPO mode: single-GPU baseline forbids DeepSpeed."
        )
        assert not torch.distributed.is_initialized(), (
            "PPO mode: single-GPU baseline forbids torch.distributed initialization."
        )
        assert world_size == 1, (
            "PPO mode: single-GPU baseline requires world_size == 1."
        )
        logger.info("PPO mode: single-GPU baseline (NO DeepSpeed, NO Accelerate)")
        return

    if not torch.cuda.is_available():
        return

    dist_initialized = torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size() if dist_initialized else None
    if world_size is None:
        world_size_env = (
            os.environ.get("WORLD_SIZE")
            or os.environ.get("ACCELERATE_PROCESS_COUNT")
        )
        world_size = int(world_size_env) if world_size_env else 1

    if world_size == 1:
        # Single-GPU PPO is a supported baseline mode: it can run without
        # Accelerate/DeepSpeed because there is no distributed memory pressure.
        # ZeRO-3 is mandatory only when PPO/GRPO is launched with world_size > 1.
        logger.info(
            "%s single-GPU run detected (world_size=1); "
            "skipping Accelerate/DeepSpeed requirement.",
            experiment_name,
        )
        return

    config_path = accelerate_deepspeed_config_path()
    accelerate_config_env = os.environ.get("ACCELERATE_CONFIG_FILE")
    if not accelerate_config_env:
        config_hint = (
            f"Expected config path: {config_path}"
            if config_path is not None
            else "Set ACCELERATE_CONFIG_FILE to your DeepSpeed config."
        )
        raise RuntimeError(
            f"{experiment_name} requires DeepSpeed ZeRO-3 via Accelerate. "
            "ACCELERATE_CONFIG_FILE must be set before launch; silent ZeRO-2 "
            "fallbacks are not allowed.\n\n"
            f"{config_hint}"
        )

    from accelerate import Accelerator
    from accelerate.state import DistributedType

    accelerator = Accelerator()
    state = accelerator.state
    deepspeed_enabled = state.distributed_type == DistributedType.DEEPSPEED
    zero_stage = None
    deepspeed_plugin = getattr(state, "deepspeed_plugin", None)
    if deepspeed_plugin is not None:
        zero_stage = getattr(deepspeed_plugin, "zero_stage", None)
        if zero_stage is None:
            ds_config = getattr(deepspeed_plugin, "deepspeed_config", {}) or {}
            zero_stage = ds_config.get("zero_optimization", {}).get("stage")
    resolved_config_file = getattr(state, "config_file", None)
    deepspeed_config_file = _resolve_deepspeed_config_file(state, deepspeed_plugin)
    logger.info(
        "DeepSpeed enabled: %s | ZeRO stage: %s | World size: %s",
        "yes" if deepspeed_enabled else "no",
        zero_stage,
        state.num_processes,
    )

    if not deepspeed_enabled:
        raise RuntimeError(
            f"{experiment_name} requires DeepSpeed ZeRO-3 via Accelerate. "
            "Silent ZeRO-2 fallback is not allowed.\n\n"
            f"ACCELERATE_CONFIG_FILE={accelerate_config_env}\n"
            f"Resolved Accelerate config: {resolved_config_file}\n"
            f"Detected deepspeed_config_file: {deepspeed_config_file}\n"
            f"Expected config path: {config_path}\n"
            "A ZeRO-2 fallback here indicates Accelerate failed to load the "
            "DeepSpeed config-file mode."
        )
    if zero_stage is None or int(zero_stage) != 3:
        raise RuntimeError(
            f"{experiment_name} requires DeepSpeed ZeRO-3. "
            f"Detected ZeRO stage: {zero_stage}. "
            "Silent ZeRO-2 fallback is not allowed.\n\n"
            f"ACCELERATE_CONFIG_FILE={accelerate_config_env}\n"
            f"Resolved Accelerate config: {resolved_config_file}\n"
            f"Detected deepspeed_config_file: {deepspeed_config_file}\n"
            f"Expected config path: {config_path}\n"
            "A ZeRO-2 fallback here indicates Accelerate failed to load the "
            "DeepSpeed config-file mode."
        )
    if torch.distributed.is_initialized():
        return

    raise RuntimeError(
        f"{experiment_name} requires Accelerate with DeepSpeed ZeRO-3. "
        "DeepSpeed is mandatory, and PPO/GRPO without DeepSpeed is unsupported "
        "due to memory constraints.\n\n"
        "Canonical Accelerate launch:\n"
        f"{ACCELERATE_MULTI_GPU_LAUNCH}\n\n"
        f"Use the DeepSpeed config: accelerate launch --config_file {config_path} "
        "-m prefadap.cli.run_training ppo --config <config.yaml>"
    )


def log_accelerate_state(log: logging.Logger | None = None) -> None:
    """Log distributed state to confirm Accelerate/DDP is initialized."""
    log = log or logging.getLogger(__name__)
    local_rank = os.environ.get("LOCAL_RANK")
    world_size_env = (
        os.environ.get("WORLD_SIZE")
        or os.environ.get("ACCELERATE_PROCESS_COUNT")
    )
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    dist_initialized = torch.distributed.is_initialized()
    backend = torch.distributed.get_backend() if dist_initialized else None
    world_size = (
        torch.distributed.get_world_size() if dist_initialized else world_size_env
    )
    device = (
        f"cuda:{torch.cuda.current_device()}"
        if torch.cuda.is_available()
        else "cpu"
    )
    device_map = f"local_rank={local_rank} -> {device}"

    log.info(
        "[accelerate] backend=%s world_size=%s local_rank=%s device_map=%s cuda_visible_devices=%s",
        backend,
        world_size,
        local_rank,
        device_map,
        cuda_visible,
    )


def log_accelerate_config_resolution(log: logging.Logger | None = None) -> None:
    """Log Accelerate config resolution details for DeepSpeed debugging."""
    log = log or logging.getLogger(__name__)
    if os.environ.get("PREFADAP_DEBUG_ACCELERATE") != "1":
        return
    from accelerate import Accelerator

    accelerator = Accelerator()
    if not accelerator.is_main_process:
        return

    state = accelerator.state
    config_file = getattr(state, "config_file", None)

    log.info(
        "[accelerate] ACCELERATE_CONFIG_FILE=%s",
        os.environ.get("ACCELERATE_CONFIG_FILE"),
    )
    log.info("[accelerate] resolved_config_file=%s", config_file)

    deepspeed_plugin = getattr(state, "deepspeed_plugin", None)
    if deepspeed_plugin is None:
        log.info("[accelerate] deepspeed_plugin=None")
        return

    zero_stage = getattr(deepspeed_plugin, "zero_stage", None)
    ds_config = getattr(deepspeed_plugin, "deepspeed_config", None)
    log.info("[accelerate] deepspeed_plugin.zero_stage=%s", zero_stage)
    log.info("[accelerate] deepspeed_plugin.deepspeed_config=%s", ds_config)


def build_ppo_model_load_config(device: str | None) -> dict[str, object]:
    """Return a loader config that disables device_map sharding for PPO."""
    config: dict[str, object] = {"disable_device_map": True}
    if device is not None:
        config["device"] = device
    return config


def is_deepspeed_zero3_enabled() -> bool:
    """Return True when Accelerate reports DeepSpeed ZeRO-3 is active."""
    if not torch.cuda.is_available():
        return False

    from accelerate import Accelerator
    from accelerate.state import DistributedType

    accelerator = Accelerator()
    state = accelerator.state
    if state.distributed_type != DistributedType.DEEPSPEED:
        return False

    deepspeed_plugin = getattr(state, "deepspeed_plugin", None)
    zero_stage = None
    if deepspeed_plugin is not None:
        zero_stage = getattr(deepspeed_plugin, "zero_stage", None)
        if zero_stage is None:
            ds_config = getattr(deepspeed_plugin, "deepspeed_config", {}) or {}
            zero_stage = ds_config.get("zero_optimization", {}).get("stage")

    try:
        return int(zero_stage) == 3
    except (TypeError, ValueError):
        return False


def is_deepspeed_enabled() -> bool:
    """Return True when Accelerate reports DeepSpeed is active."""
    if not torch.cuda.is_available():
        return False

    from accelerate import Accelerator
    from accelerate.state import DistributedType

    accelerator = Accelerator()
    state = accelerator.state
    if state.distributed_type == DistributedType.DEEPSPEED:
        return True

    deepspeed_plugin = getattr(state, "deepspeed_plugin", None)
    deepspeed_config_file = _resolve_deepspeed_config_file(state, deepspeed_plugin)
    return deepspeed_plugin is not None or deepspeed_config_file is not None


def resolve_rollout_forward_batch_size(
    model_name_or_path: str, current_value: int
) -> int:
    """Resolve the rollout forward batch size with model-aware defaults."""
    if current_value and current_value > 1:
        return current_value

    return 4


def estimate_model_size_b(model_name_or_path: str) -> Optional[float]:
    """Estimate model size in billions of parameters from name or config.
    
    This is a heuristic function that tries to extract the model size from
    the model name (e.g., "32B", "7B") or from the model config if available.
    
    Args:
        model_name_or_path: Model name or path
        
    Returns:
        Estimated model size in billions of parameters, or None if unknown
    """
    import re
    
    # Try to extract size from model name (e.g., "Olmo-3-1125-32B", "Llama-3.1-8B")
    # Match patterns like "32B", "32b", "8B", "7B"
    size_pattern = r'(\d+(?:\.\d+)?)\s*[bB](?:\s|$|-|_)'
    match = re.search(size_pattern, model_name_or_path)
    if match:
        try:
            size_b = float(match.group(1))
            return size_b
        except ValueError:
            pass
    
    # Try to load config.json to get actual parameter count
    try:
        from pathlib import Path
        import json
        
        # Check if it's a local path with config.json
        path = Path(model_name_or_path)
        if path.exists() and path.is_dir():
            config_file = path / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    # Try to compute from architecture parameters
                    # This is a simplified heuristic for transformer models
                    hidden_size = config.get('hidden_size', 0)
                    num_layers = config.get('num_hidden_layers', 0)
                    if hidden_size and num_layers:
                        # Rough estimate: params ≈ 12 * hidden_size^2 * num_layers / 1e9
                        estimated = 12 * (hidden_size ** 2) * num_layers / 1e9
                        return estimated
    except Exception:
        pass
    
    return None


def check_gpu_requirements_for_large_model(
    model_name_or_path: str,
    experiment_type: str,
    use_lora: bool = False,
    shared_backbone: bool = False,
) -> None:
    """Validate GPU requirements for large models (≥30B parameters).
    
    This function checks if the current GPU setup can support the requested
    model and training configuration. It will raise an error if:
    - Model is ≥30B AND
    - Only 1 GPU is visible AND
    - DPO/GRPO is requested without shared backbone
    
    Args:
        model_name_or_path: Model name or path
        experiment_type: Training type (sft, dpo, grpo, etc.)
        use_lora: Whether LoRA is enabled
        shared_backbone: Whether shared backbone is enabled for DPO/GRPO
        
    Raises:
        RuntimeError: If GPU requirements are not met for the configuration
    """
    # Estimate model size
    model_size_b = estimate_model_size_b(model_name_or_path)
    
    if model_size_b is None:
        # Can't determine size, allow to proceed
        logger.debug(f"Could not estimate model size for {model_name_or_path}, skipping GPU validation")
        return
    
    if model_size_b < LARGE_MODEL_THRESHOLD_B:
        # Small models don't need special handling
        return
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.warning(
            f"Large model ({model_size_b:.1f}B parameters) requested but no CUDA devices available. "
            f"Training will likely fail or be extremely slow on CPU."
        )
        return
    
    gpu_count = torch.cuda.device_count()
    
    # Log GPU configuration
    logger.info(
        f"Large model detected: {model_name_or_path} (~{model_size_b:.1f}B parameters), "
        f"GPUs available: {gpu_count}, LoRA: {use_lora}, Experiment: {experiment_type}"
    )
    
    # Check if we need reference model (DPO, GRPO, KTO, ORPO)
    needs_reference_model = experiment_type.lower() in {"dpo", "grpo", "kto", "orpo", "online_dpo"}
    
    # If multiple GPUs are available, we're good
    if gpu_count > 1:
        logger.info(f"Multiple GPUs detected ({gpu_count}), proceeding with large model training")
        return
    
    # Single GPU case: validate requirements
    if needs_reference_model and not shared_backbone:
        # DPO/GRPO without shared backbone on single GPU will OOM
        error_msg = (
            f"\n{'='*80}\n"
            f"ERROR: Insufficient GPU resources for {model_size_b:.1f}B model\n"
            f"{'='*80}\n"
            f"Configuration:\n"
            f"  - Model: {model_name_or_path} (~{model_size_b:.1f}B parameters)\n"
            f"  - Experiment type: {experiment_type}\n"
            f"  - GPUs available: {gpu_count}\n"
            f"  - LoRA enabled: {use_lora}\n"
            f"  - Shared backbone: {shared_backbone}\n"
            f"\n"
            f"Problem: {experiment_type.upper()} requires both a policy and reference model.\n"
            f"Loading two full {model_size_b:.1f}B models on a single GPU will cause CUDA OOM.\n"
            f"\n"
            f"Solutions:\n"
            f"  1. Use multiple GPUs (recommended):\n"
            f"     - Request ≥2 GPUs in your job script\n"
            f"     - Run with: CUDA_VISIBLE_DEVICES=0,1 python -m ...\n"
            f"\n"
            f"  2. Enable shared backbone with LoRA (single GPU):\n"
            f"     Add to your config:\n"
            f"       lora: true\n"
            f"       shared_backbone: true\n"
            f"\n"
            f"  3. Use SFT instead (single GPU, no reference model):\n"
            f"     Change experiment_type to 'sft'\n"
            f"\n"
            f"  4. Use a smaller model (e.g., 7B or 8B)\n"
            f"{'='*80}\n"
        )
        raise RuntimeError(error_msg)
    
    # Single GPU with SFT or shared backbone is OK
    if experiment_type.lower() == "sft":
        logger.info(
            f"Single GPU SFT training for {model_size_b:.1f}B model. "
            f"Ensure LoRA is enabled and batch size is small."
        )
    elif shared_backbone:
        logger.info(
            f"Single GPU {experiment_type.upper()} training with shared backbone for {model_size_b:.1f}B model."
        )


def _should_enable_fp16(device: str) -> bool:
    """Return ``True`` when FP16 should be enabled for ``device``.

    FP16 is only appropriate for CUDA devices and when a GPU is available.
    """

    device = resolve_device(device)
    return device.startswith("cuda") and torch.cuda.is_available()


def _should_enable_bf16(device: str) -> bool:
    """Return ``True`` when BF16 should be enabled for ``device``.

    BF16 is only appropriate for CUDA devices with Ampere or newer architecture
    (compute capability >= 8.0) or ROCm devices, and when a GPU is available.
    """
    device = resolve_device(device)
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        return False

    # Check if current device supports BF16
    try:
        device_props = torch.cuda.get_device_properties(0)
        # Ampere and newer (compute capability >= 8.0)
        return device_props.major >= 8
    except (RuntimeError, AttributeError):
        # Fallback: check if torch supports BF16 operations
        return (
            hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        )


def validate_mixed_precision_config(
    fp16: bool = False, bf16: bool = False, device: str = ""
) -> None:
    """Validate mixed precision configuration and raise informative errors.

    Args:
        fp16: Whether FP16 is requested
        bf16: Whether BF16 is requested
        device: Target device string

    Raises:
        ValueError: If the configuration is invalid or unsupported
        RuntimeError: If hardware doesn't support the requested precision
    """
    if fp16 and bf16:
        raise ValueError(
            "Cannot enable both FP16 and BF16 simultaneously. "
            "Please choose only one mixed precision format."
        )

    device = resolve_device(device)
    if not device.startswith("cuda") and (fp16 or bf16):
        if torch.cuda.is_available():
            raise ValueError(
                f"Mixed precision training ({'FP16' if fp16 else 'BF16'}) is only supported on CUDA devices, but device is '{device}'. "
                "A CUDA device is available; set device='cuda' to use it."
            )
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
        raise ValueError(
            f"Mixed precision training ({'FP16' if fp16 else 'BF16'}) is only supported on CUDA devices, but device is '{device}'. "
            f"torch.cuda.is_available() returned False (CUDA_VISIBLE_DEVICES={cuda_visible}, torch.version.cuda={torch.version.cuda})."
        )

    if not torch.cuda.is_available() and (fp16 or bf16):
        raise RuntimeError(
            f"Mixed precision training ({'FP16' if fp16 else 'BF16'}) requires CUDA, "
            "but no CUDA devices are available. Please ensure you have a compatible GPU "
            "and PyTorch with CUDA support installed."
        )

    if bf16:
        if not _should_enable_bf16(device):
            try:
                device_props = torch.cuda.get_device_properties(0)
                compute_capability = f"{device_props.major}.{device_props.minor}"
                raise RuntimeError(
                    f"BF16 training requires GPU with compute capability >= 8.0 (Ampere or newer), "
                    f"but current device has compute capability {compute_capability}. "
                    f"Consider using FP16 instead for older GPUs."
                )
            except (RuntimeError, AttributeError) as e:
                if "compute capability" in str(e):
                    raise e
                raise RuntimeError(
                    "BF16 training is not supported on this device. "
                    "BF16 requires Ampere or newer GPU architecture. "
                    "Consider using FP16 instead."
                )

    # For FP16, we've already validated device type and CUDA availability above.
    # The _should_enable_fp16 check would be redundant at this point since it only
    # checks the same conditions. No additional FP16-specific validation needed.


def build_trl_config(args: Any, cls: Type[T], **extra: Any) -> T:
    """Construct a TRL/transformers config instance from ``args``.

    Parameters
    ----------
    args:
        Namespace containing common training fields.
    cls:
        Configuration class to instantiate (e.g. ``TrainingArguments`` or
        ``DPOConfig``).
    extra:
        Additional keyword arguments specific to ``cls``.
    """
    import inspect

    device = resolve_device(getattr(args, "device", None))
    setattr(args, "device", device)

    # Determine precision settings with user override support
    fp16_requested = getattr(args, "fp16", None)
    bf16_requested = getattr(args, "bf16", None)

    # Auto-detect if neither is explicitly set
    if fp16_requested is None and bf16_requested is None:
        # Prefer BF16 on capable devices, fallback to FP16
        if _should_enable_bf16(device):
            fp16_enabled = False
            bf16_enabled = True
        else:
            fp16_enabled = _should_enable_fp16(device)
            bf16_enabled = False
    else:
        # Use user's explicit settings
        fp16_enabled = fp16_requested if fp16_requested is not None else False
        bf16_enabled = bf16_requested if bf16_requested is not None else False

    # Validate the precision configuration
    validate_mixed_precision_config(fp16_enabled, bf16_enabled, device)

    # Pull core knobs from args with sensible fallbacks
    weight_decay_val = getattr(args, "weight_decay", 0.01)
    lr_sched = getattr(args, "lr_scheduler_type", None)
    warmup = getattr(args, "warmup_ratio", None)
    # Only forward max_steps if explicitly set; TrainingArguments defaults to -1
    max_steps = getattr(args, "max_steps", None)
    save_total_limit = getattr(args, "save_total_limit", 2)

    # Explicitly configure behaviour that Hugging Face/TRL would otherwise infer
    # and warn about.  Custom collators in this project frequently rely on all
    # columns being preserved, so default to ``remove_unused_columns=False``
    # unless the caller opted into another value.  Likewise, ensure
    # ``label_names`` is always populated so PEFT-wrapped causal LM models don't
    # emit warnings during Trainer initialisation.
    remove_unused_columns = getattr(args, "remove_unused_columns", None)
    if remove_unused_columns is None:
        remove_unused_columns = False

    label_names = getattr(args, "label_names", None)
    if label_names is None:
        label_names = ["labels"]
    elif isinstance(label_names, str):
        label_names = [label_names]

    common = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=["wandb"] if getattr(args, "use_wandb", False) else [],
        weight_decay=weight_decay_val,
        lr_scheduler_type=lr_sched,
        warmup_ratio=warmup,
        # skip max_steps if None so default logic in HF/TRL applies
        max_steps=max_steps,
        save_total_limit=save_total_limit,
        fp16=fp16_enabled,
        bf16=bf16_enabled,
        deepspeed=getattr(args, "deepspeed", None),
        remove_unused_columns=remove_unused_columns,
        label_names=label_names,
        # Explicitly set hub-related keys to prevent TRL crashes
        # TRL trainers (e.g., SFTTrainer) may call .pop() on these keys without defaults
        push_to_hub=False,
        push_to_hub_token=None,
    )
    common.update(extra)

    # Remove optional keys with None values to avoid overriding HF/TRL defaults
    # None for these can trigger downstream type issues (e.g., comparisons with ints)
    # Note: push_to_hub and push_to_hub_token are NOT in this list because TRL
    # expects them to always exist (even if None/False) and will crash if missing
    optional_keys_to_remove = [
        "max_steps",
        "lr_scheduler_type",
        "warmup_ratio",
        # Other hub-related keys that may not be supported in all TRL versions
        "hub_token",
        "push_to_hub_model_id",
        "push_to_hub_organization",
    ]
    
    for optional_key in optional_keys_to_remove:
        if common.get(optional_key, "__MISSING__") is None:
            common.pop(optional_key, None)

    # Filter out unsupported kwargs for forward/backward compatibility
    try:
        sig = inspect.signature(cls.__init__)
        supported_params = set(sig.parameters.keys())

        # Separate supported and unsupported parameters
        supported_kwargs = {k: v for k, v in common.items() if k in supported_params}
        unsupported_kwargs = {
            k: v for k, v in common.items() if k not in supported_params
        }

        # Create the config with only supported parameters
        config = cls(**supported_kwargs)

        # Set unsupported parameters as attributes for backward compatibility
        for key, value in unsupported_kwargs.items():
            setattr(config, key, value)

        # Ensure hub-related keys exist with safe defaults to prevent TRL crashes
        # TRL trainers (e.g., SFTTrainer) may call .pop() on these keys without defaults
        # Ensure these exist even if None, so .pop() doesn't raise KeyError
        hub_keys_with_safe_defaults = {
            "push_to_hub_token": None,
            "hub_token": None,
            "push_to_hub_model_id": None,
            "push_to_hub_organization": None,
        }
        for key, default_value in hub_keys_with_safe_defaults.items():
            if not hasattr(config, key):
                setattr(config, key, default_value)
        
        # CRITICAL FIX: Wrap to_dict() to ensure hub keys are always in the dict
        # TRL's SFTTrainer calls to_dict() and then pops these keys without defaults
        # Even if we set attributes, they might not appear in to_dict() output
        # Use a picklable wrapper class instead of a local closure to support checkpointing
        original_to_dict = config.to_dict if hasattr(config, "to_dict") else None
        if original_to_dict:
            config.to_dict = _PicklableToDictWrapper(
                config, original_to_dict, hub_keys_with_safe_defaults
            )

        return config
    except (TypeError, AttributeError, ValueError):
        # Fallback to original behavior if introspection fails
        config = cls(**common)
        
        # Ensure hub-related keys exist with safe defaults (same as above)
        hub_keys_with_safe_defaults = {
            "push_to_hub_token": None,
            "hub_token": None,
            "push_to_hub_model_id": None,
            "push_to_hub_organization": None,
        }
        for key, default_value in hub_keys_with_safe_defaults.items():
            if not hasattr(config, key):
                setattr(config, key, default_value)
        
        # CRITICAL FIX: Wrap to_dict() to ensure hub keys are always in the dict
        # TRL's SFTTrainer calls to_dict() and then pops these keys without defaults
        # Use a picklable wrapper class instead of a local closure to support checkpointing
        original_to_dict = config.to_dict if hasattr(config, "to_dict") else None
        if original_to_dict:
            config.to_dict = _PicklableToDictWrapper(
                config, original_to_dict, hub_keys_with_safe_defaults
            )
        
        return config


def load_reference_model_with_validation(
    model: PreTrainedModel,
    model_name_or_path: str,
    experiment_type: str,
    use_lora: bool = False,
    shared_backbone: bool = False,
    enabled: bool = True,
    device: str = "cpu",
) -> Optional[PreTrainedModel]:
    """Load reference model with GPU validation for large models.
    
    This is the centralized reference model loading function that should be used
    by all pipelines. It validates GPU requirements for large models (≥30B) and
    automatically applies memory-efficient strategies.
    
    Args:
        model: The policy model to create a reference copy of
        model_name_or_path: Model name or path for size detection
        experiment_type: Training type (dpo, grpo, kto, orpo, online_dpo, ppo)
        use_lora: Whether LoRA is enabled
        shared_backbone: Whether shared backbone is enabled
        enabled: Whether to create the reference model
        device: Device to place the reference model on (default: "cpu" to save GPU memory)
        
    Returns:
        Reference model on specified device, or None if not enabled or creation fails
        
    Raises:
        RuntimeError: If GPU requirements are not met for large model configuration
    """
    if not enabled:
        return None
    
    # Validate GPU requirements for large models
    try:
        check_gpu_requirements_for_large_model(
            model_name_or_path=model_name_or_path,
            experiment_type=experiment_type,
            use_lora=use_lora,
            shared_backbone=shared_backbone,
        )
    except RuntimeError as e:
        # Re-raise with additional context about reference model loading
        raise RuntimeError(
            f"GPU validation failed during reference model loading: {e}"
        ) from e
    
    # Determine optimal device for reference model
    model_size_b = estimate_model_size_b(model_name_or_path)
    gpu_count = 1
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
    except Exception:
        pass
    
    # For large models on single GPU, always use CPU for reference
    if model_size_b is not None and model_size_b >= LARGE_MODEL_THRESHOLD_B and gpu_count == 1:
        device = "cpu"
        logger.info(
            f"Large model ({model_size_b:.1f}B) on single GPU: "
            f"placing reference model on CPU to save GPU memory"
        )
    
    try:
        ref_model = create_reference_model(model)
        
        # Only move reference model if it doesn't have a device map
        # Use centralized safe_move_model helper
        if device != "cuda" and hasattr(ref_model, "to"):
            safe_move_model(ref_model, device, "reference model")
        
        return ref_model
    except Exception as e:
        logger.warning(f"Failed to create reference model: {e}")
        return None


def load_explicit_reference_model(
    model_name_or_path: str,
    *,
    training_args: Any,
    tokenizer_name_or_path: str | None = None,
) -> PreTrainedModel:
    """Load an explicit reference model from the base checkpoint."""
    from prefadap.models.loader import load_model_and_tokenizer

    device = resolve_device(getattr(training_args, "device", None))
    config = build_ppo_model_load_config(device)
    ref_model, _ = load_model_and_tokenizer(
        model_name_or_path,
        gradient_checkpointing=False,
        tokenizer_name_or_path=tokenizer_name_or_path,
        config=config,
    )
    if getattr(training_args, "lora", False):
        ref_model = _maybe_apply_lora(training_args, ref_model)
    ref_model.requires_grad_(False)
    ref_model.eval()
    ensure_generation_config(ref_model)
    return ref_model


def _load_reference_model(
    model: PreTrainedModel, enabled: bool = True, device: str = "cpu"
) -> Optional[PreTrainedModel]:
    """Legacy reference model loading function (deprecated, use load_reference_model_with_validation).
    
    This function is kept for backward compatibility but should not be used in new code.
    Use load_reference_model_with_validation instead for proper GPU validation.
    
    Args:
        model: The model to create a reference copy of
        enabled: Whether to create the reference model
        device: Device to place the reference model on (default: "cpu" to save GPU memory)
        
    Returns:
        Reference model on specified device, or None if not enabled or creation fails
    """

    if not enabled:
        return None
    try:
        ref_model = create_reference_model(model)
        # Always move to CPU by default to save GPU memory
        ref_model = ref_model.to(device)
        return ref_model
    except Exception:
        return None


def check_model_exists(output_dir: str) -> bool:
    """Check if a COMPLETE trained model already exists in the output directory.

    Only checks for truly complete models:
    - final_model/ (final checkpoint after training completes)
    - final_model_remerged/ (remerged model after training)
    - best/ (best checkpoint during training)
    - model saved directly in output_dir root

    Does NOT check for checkpoint-*/ directories, as those indicate
    incomplete training that should be RESUMED, not skipped.

    Args:
        output_dir: Directory where model would be saved

    Returns:
        True if a COMPLETE trained model exists, False otherwise
    """
    from pathlib import Path

    output_path = Path(output_dir)
    if not output_path.exists():
        return False

    def _has_model_files(path: Path) -> bool:
        """Return True if the directory contains a saved model or adapter."""

        has_full_model = (path / "config.json").exists()
        has_adapter = (
            (path / "adapter_model.safetensors").exists()
            and (path / "adapter_config.json").exists()
        )
        return has_full_model or has_adapter

    # Check for final model
    final_model = output_path / "final_model"
    if final_model.exists() and _has_model_files(final_model):
        return True

    # Check for final_model_remerged
    final_model_remerged = output_path / "final_model_remerged"
    if final_model_remerged.exists() and _has_model_files(final_model_remerged):
        return True

    # Check for best model
    best_model = output_path / "best"
    if best_model.exists() and _has_model_files(best_model):
        return True

    # Check for model saved directly in output_dir (root)
    if _has_model_files(output_path):
        return True

    # Note: We deliberately do NOT check for checkpoint-* here
    # Checkpoints indicate incomplete training that should be resumed
    return False



def resolve_trl_output_dir(output_dir: str, log: logging.Logger | None = None) -> str:
    """Resolve a TRL trainer output directory to an absolute path.

    TRL trainers may default to a relative ``output_dir`` (e.g., ``trainer_output``),
    which is unsafe in containerized environments with read-only working
    directories. This helper ensures an absolute path is always used.
    """
    path = Path(output_dir).expanduser()
    if path.is_absolute():
        return str(path)

    resolved = path.resolve()
    if log is not None:
        log.info(
            "Rewriting relative TRL output_dir '%s' to absolute path '%s'.",
            output_dir,
            resolved,
        )
    return str(resolved)


__all__ = [
    "apply_default_sharding",
    "safe_move_model",
    "get_effective_hf_device_map",
    "_maybe_apply_lora",
    "_maybe_enable_gradient_checkpointing",
    "_should_enable_gradient_checkpointing_automatically",
    "_maybe_freeze_embeddings",
    "_maybe_init_wandb",
    "_load_reference_model",
    "load_reference_model_with_validation",
    "load_explicit_reference_model",
    "_should_enable_fp16",
    "_should_enable_bf16",
    "validate_mixed_precision_config",
    "build_trl_config",
    "resolve_device",
    "is_distributed_environment",
    "validate_ppo_single_gpu_baseline",
    "validate_grpo_single_gpu_baseline",
    "validate_ppo_distributed_launch",
    "log_accelerate_state",
    "build_ppo_model_load_config",
    "is_deepspeed_zero3_enabled",
    "is_deepspeed_enabled",
    "resolve_rollout_forward_batch_size",
    "ACCELERATE_MULTI_GPU_LAUNCH",
    "accelerate_deepspeed_config_path",
    "estimate_model_size_b",
    "check_gpu_requirements_for_large_model",
    "_tune_70b_defaults",
    "dump_indices",
    "save_model_with_lora",
    "check_model_exists",
    "validate_persistent_storage",
    "resolve_trl_output_dir",
    "TokenKLCallback",
    "PerplexityThresholdCallback",
    "resolve_optional_pseudo_data_path",
    "ensure_trl_policy_value_wrapper_gradient_checkpointing",
    "log_liger_kernel_status",
]
# ---------------------------------------------------------------------------
# TRL PPO helper
# ---------------------------------------------------------------------------


def ensure_generation_config(model: PreTrainedModel) -> None:
    if not hasattr(model, "generation_config") or model.generation_config is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)
    if getattr(model.generation_config, "pad_token_id", None) is None:
        pad_token_id = getattr(model.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(model.config, "eos_token_id", None)
        if pad_token_id is not None:
            model.generation_config.pad_token_id = pad_token_id


def _toggle_gradient_checkpointing(model: Any, enable: bool, *args, **kwargs) -> None:
    if model is None:
        return
    if enable and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(*args, **kwargs)
        return
    if not enable and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        return
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = enable


def ensure_trl_policy_value_wrapper_gradient_checkpointing() -> None:
    """Ensure TRL's policy/value wrapper exposes HF gradient checkpointing helpers."""
    from trl.trainer.ppo_trainer import PolicyAndValueWrapper

    if not hasattr(PolicyAndValueWrapper, "gradient_checkpointing_disable"):
        # TRL expects HF-style gradient checkpointing helpers on the wrapper during generation.
        def gradient_checkpointing_disable(self) -> None:
            _toggle_gradient_checkpointing(self.policy, False)
            critic_backbone = getattr(self, "critic_backbone", None)
            if critic_backbone is not None and critic_backbone is not self.policy:
                _toggle_gradient_checkpointing(critic_backbone, False)

        def gradient_checkpointing_enable(self, *args, **kwargs) -> None:
            _toggle_gradient_checkpointing(self.policy, True, *args, **kwargs)
            critic_backbone = getattr(self, "critic_backbone", None)
            if critic_backbone is not None and critic_backbone is not self.policy:
                _toggle_gradient_checkpointing(critic_backbone, True, *args, **kwargs)

        PolicyAndValueWrapper.gradient_checkpointing_disable = gradient_checkpointing_disable
        PolicyAndValueWrapper.gradient_checkpointing_enable = gradient_checkpointing_enable

    if not isinstance(PolicyAndValueWrapper.__dict__.get("is_gradient_checkpointing"), property):
        def _get_is_gradient_checkpointing(self) -> bool:
            return bool(
                getattr(
                    self.policy,
                    "is_gradient_checkpointing",
                    getattr(self, "_is_gradient_checkpointing", False),
                )
            )

        def _set_is_gradient_checkpointing(self, value: bool) -> None:
            self.__dict__["_is_gradient_checkpointing"] = value

        PolicyAndValueWrapper.is_gradient_checkpointing = property(
            _get_is_gradient_checkpointing,
            _set_is_gradient_checkpointing,
        )
