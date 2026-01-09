"""Utilities for loading models/tokenizers and applying LoRA."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
from typing import List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

try:  # Optional dependency used when loading adapter weights
    from peft import AutoPeftModelForCausalLM, PeftModel  # type: ignore
except Exception:  # pragma: no cover - peft not installed
    AutoPeftModelForCausalLM = None  # type: ignore
    PeftModel = None  # type: ignore

from .registry import register_model
from .tokenization import ensure_pad_token
from ..utils.hf_auth import prepare_hf_from_pretrained_kwargs

# Import olmo3_seqcls to ensure Olmo3ForSequenceClassification is registered
# before any code attempts to load OLMo-3 reward models. This import must happen
# here because loader.py is often imported directly (e.g., from training_wrappers),
# bypassing the prefadap.models.__init__.py import.
from . import olmo3_seqcls  # noqa: F401

# Constants for memory calculations
BYTES_TO_GB = 1024**3

# Module-level logger
logger = logging.getLogger(__name__)

_STAGE_RE = re.compile(r"stage(\d+)$")
_SEED_RE = re.compile(r"seed(\d+)$")
_HF_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


def _count_parameters(model: object) -> Optional[int]:
    params = getattr(model, "parameters", None)
    if not callable(params):
        return None
    try:
        return sum(p.numel() for p in params())
    except Exception:  # pragma: no cover - defensive
        return None


def _has_model_artifacts(path: Path) -> bool:
    """Return ``True`` if ``path`` looks like a Hugging Face checkpoint."""

    if not path.exists() or not path.is_dir():
        return False

    if (path / "config.json").exists():
        return True

    adapter_file = path / "adapter_model.safetensors"
    adapter_config = path / "adapter_config.json"
    if adapter_file.exists() and adapter_config.exists():
        return True

    return False


def is_adapter_directory(path: Path) -> bool:
    """Return ``True`` when ``path`` contains PEFT adapter weights."""

    if not path.exists() or not path.is_dir():
        return False

    adapter_config = path / "adapter_config.json"
    if not adapter_config.exists():
        return False

    adapter_candidates = [
        path / "adapter_model.safetensors",
        path / "adapter_model.bin",
        path / "adapter_model.pt",
    ]

    if any(candidate.exists() for candidate in adapter_candidates):
        return True

    # Support sharded adapter checkpoints (e.g., adapter_model.safetensors.index.json)
    shard_index = path / "adapter_model.safetensors.index.json"
    return shard_index.exists()


def _normalise_identifier(identifier: str) -> str:
    """Return a canonical representation for local paths and repo IDs."""

    value = identifier.strip()
    if not value:
        return value

    looks_like_repo = _HF_REPO_RE.fullmatch(value) is not None
    is_windows_path = os.path.altsep and os.path.altsep in value
    is_unix_path = os.path.sep in value and not looks_like_repo
    if value.startswith((".", "~", os.path.sep)) or is_windows_path or is_unix_path:
        try:
            return str(Path(value).expanduser().resolve())
        except (OSError, RuntimeError):
            return str(Path(value).expanduser())
    return value.rstrip("/\\")


def _identifiers_match(lhs: str, rhs: str) -> bool:
    """Return ``True`` when two identifiers refer to the same source."""

    return _normalise_identifier(lhs) == _normalise_identifier(rhs)


def _extract_model_name_from_resolved_config(config_path: Path) -> Optional[str]:
    """Best-effort extraction of ``model_name_or_path`` from YAML configs."""

    if not config_path.exists():
        return None

    try:  # Try the robust path first when PyYAML is available.
        import yaml  # type: ignore

        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if isinstance(data, dict):
            candidate = data.get("model_name_or_path")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    except Exception:
        # Fall back to a lightweight line scan that works without PyYAML.
        try:
            for line in config_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("model_name_or_path:"):
                    value = stripped.split(":", 1)[1].strip().strip("'\"")
                    if value:
                        return value
        except Exception:
            return None

    return None


def get_adapter_base_model(path: Path) -> Optional[str]:
    """Return the base model specified by a PEFT adapter directory, if any."""

    adapter_config = path / "adapter_config.json"
    if adapter_config.exists():
        try:
            config_data = json.loads(adapter_config.read_text())
        except json.JSONDecodeError:
            config_data = None
        else:
            base = config_data.get("base_model_name_or_path") or config_data.get("base_model")
            if isinstance(base, str) and base.strip():
                return base

    # Some historical runs failed to record the base model.  Fall back to the
    # resolved training configuration which always captures the original
    # ``model_name_or_path`` value.
    resolved_config = path.parent / "resolved_config.yaml"
    fallback = _extract_model_name_from_resolved_config(resolved_config)
    if fallback:
        return fallback

    return None


def _collect_model_candidates(
    path: Path,
    *,
    stage_hint: Optional[int] = None,
    seed_hint: Optional[int] = None,
) -> List[Tuple[int, Optional[int], Optional[int], Optional[int], Path]]:
    """Return candidate model directories beneath ``path``.

    The helper searches run directories following the canonical
    ``outputs/FRAMEWORK/.../stageN/seedM`` layout and collects
    ``final_model`` checkpoints as well as ``best`` and ``checkpoint-*``
    directories. ``stage_hint`` and ``seed_hint`` can be provided to
    prioritise specific stages or seeds.
    """

    candidates: List[Tuple[int, Optional[int], Optional[int], Optional[int], Path]] = []

    def _checkpoint_num(path: Path) -> Optional[int]:
        if not path.name.startswith("checkpoint-"):
            return None
        try:
            return int(path.name.split("checkpoint-", 1)[1])
        except (IndexError, ValueError):
            return None

    def _add_candidate(
        dir_path: Path,
        *,
        priority: int,
        stage: Optional[int],
        seed: Optional[int],
    ) -> None:
        if _has_model_artifacts(dir_path):
            candidates.append((priority, stage, seed, _checkpoint_num(dir_path), dir_path))

    def _walk(dir_path: Path, stage: Optional[int], seed: Optional[int]) -> None:
        if not dir_path.exists() or not dir_path.is_dir():
            return

        current_stage = stage
        current_seed = seed

        stage_match = _STAGE_RE.fullmatch(dir_path.name)
        if stage_match:
            try:
                current_stage = int(stage_match.group(1))
            except ValueError:
                current_stage = stage

        seed_match = _SEED_RE.fullmatch(dir_path.name)
        if seed_match:
            try:
                current_seed = int(seed_match.group(1))
            except ValueError:
                current_seed = seed

        if dir_path.name == "final_model":
            _add_candidate(dir_path, priority=0, stage=current_stage, seed=current_seed)
            return
        if dir_path.name == "best":
            _add_candidate(dir_path, priority=1, stage=current_stage, seed=current_seed)
            return
        if dir_path.name.startswith("checkpoint-"):
            _add_candidate(dir_path, priority=2, stage=current_stage, seed=current_seed)
            return

        final_dir = dir_path / "final_model"
        if final_dir.is_dir():
            _add_candidate(final_dir, priority=0, stage=current_stage, seed=current_seed)

        best_dir = dir_path / "best"
        if best_dir.is_dir():
            _add_candidate(best_dir, priority=1, stage=current_stage, seed=current_seed)

        for child in dir_path.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if name == "final_model":
                continue
            if name == "best":
                continue
            if name.startswith("checkpoint-"):
                _add_candidate(child, priority=2, stage=current_stage, seed=current_seed)
                continue
            if name.startswith("."):
                continue
            if _STAGE_RE.fullmatch(name) or _SEED_RE.fullmatch(name):
                _walk(child, current_stage, current_seed)

    _walk(path, None, None)

    if not candidates:
        return []

    # Normalise hints
    stage_filter: Optional[int]
    seed_filter: Optional[int]
    try:
        stage_filter = int(stage_hint) if stage_hint is not None else None
    except (TypeError, ValueError):
        stage_filter = None

    if stage_filter is not None and stage_filter > 1:
        stage_filter = stage_filter - 1
    else:
        stage_filter = None

    try:
        seed_filter = int(seed_hint) if seed_hint is not None else None
    except (TypeError, ValueError):
        seed_filter = None

    if stage_filter is not None:
        filtered = [c for c in candidates if c[1] == stage_filter]
        if filtered:
            candidates = filtered

    if seed_filter is not None:
        filtered = [c for c in candidates if c[2] == seed_filter]
        if filtered:
            candidates = filtered

    # Sort by priority -> highest stage -> lowest seed -> shallowest path
    def _sort_key(
        item: Tuple[int, Optional[int], Optional[int], Optional[int], Path]
    ) -> Tuple[int, int, int, int, int]:
        priority, stage, seed, checkpoint_num, candidate_path = item
        stage_value = stage if stage is not None else -1
        seed_value = seed if seed is not None else 10**9
        checkpoint_value = checkpoint_num if checkpoint_num is not None else -1
        depth = len(candidate_path.parts)
        return (priority, -stage_value, seed_value, depth, -checkpoint_value)

    candidates.sort(key=_sort_key)
    return candidates


def _resolve_model_path(
    model_name_or_path: str,
    *,
    stage: Optional[int] = None,
    seed: Optional[int] = None,
) -> str:
    """Resolve a model path and ensure that it exists if it's local.

    The training pipeline is often executed from run directories nested deep
    under ``outputs``.  Relative model paths therefore need to be interpreted
    with respect to the repository root rather than the current working
    directory.  This helper resolves such paths by checking several candidate
    locations and raises a clear error if the model still cannot be located.
    """
    # In test mode, enforce strict existence check for local-looking paths
    # before any fallback logic to catch configuration errors early
    if os.getenv("PREFADAP_TEST_MODE") == "1":
        # Check if path looks local (has slashes or starts with local path indicators)
        if ("/" in model_name_or_path and 
            not model_name_or_path.startswith(("http://", "https://"))):
            if not os.path.exists(model_name_or_path):
                raise FileNotFoundError(f"Model path not found: {model_name_or_path}")
    
    looks_like_local_path = False

    if "/" in model_name_or_path:
        slash_count = model_name_or_path.count("/")
        if slash_count > 1:
            looks_like_local_path = True
        elif model_name_or_path.startswith(
            ("./", "../", "/", "outputs/", "models/", "checkpoints/")
        ):
            looks_like_local_path = True

    def _resolve_directory(directory: Path) -> str:
        name = directory.name
        candidates = _collect_model_candidates(
            directory, stage_hint=stage, seed_hint=seed
        )
        if candidates:
            return str(candidates[0][4])

        if _has_model_artifacts(directory):
            return str(directory)

        adapter_file = directory / "adapter_model.safetensors"
        adapter_config = directory / "adapter_config.json"
        if adapter_file.exists() or adapter_config.exists():
            return str(directory)

        expected_final = directory / "final_model"
        raise FileNotFoundError(
            f"Could not locate a 'final_model' checkpoint under '{directory}'. "
            f"Expected directory: '{expected_final}'."
        )

    try:
        path_parts = tuple(Path(model_name_or_path).parts)
    except (TypeError, ValueError):
        path_parts = ()
    
    # Check if "outputs" appears in the path (either at start or anywhere for absolute paths)
    outputs_index = -1
    if path_parts and path_parts[0] == "outputs":
        outputs_index = 0
    elif "outputs" in path_parts:
        try:
            outputs_index = path_parts.index("outputs")
        except ValueError:
            outputs_index = -1
    
    if outputs_index >= 0:
        try:
            # Extract the relative path starting from "outputs"
            relative_parts = path_parts[outputs_index:]
            
            # If the path ends with a model subdirectory (final_model, best),
            # strip it to get the run directory.
            if relative_parts and relative_parts[-1] in {"final_model", "best"}:
                relative_parts = relative_parts[:-1]
            
            relative_path = str(Path(*relative_parts))
            
            # Canonicalize using the same logic as the training/eval pipeline
            from prefadap.utils.paths import resolve_run_dir

            canonical = resolve_run_dir(relative_path, seed=seed)
            canonical_dir = Path(str(canonical))
            if canonical_dir.exists() and canonical_dir.is_dir():
                return _resolve_directory(canonical_dir)
            # Even if the exact directory is missing, attempt to locate
            # a 'final_model' under the intended stage/seed tree.
            return _resolve_directory(canonical_dir)
        except (FileNotFoundError, OSError, ValueError):
            # Fall back to the generic path search below if canonical
            # resolution fails for expected reasons (keeps HF IDs working).
            pass

    if looks_like_local_path:
        if os.path.exists(model_name_or_path):
            abs_path = os.path.abspath(model_name_or_path)
            resolved = Path(abs_path)
            if resolved.is_dir():
                return _resolve_directory(resolved)
            return abs_path

        path_obj = Path(model_name_or_path)
        first_component = path_obj.parts[0] if path_obj.parts else None

        def _iter_search_roots() -> list[Path]:
            roots: list[Path] = []
            seen: set[Path] = set()

            def _add(path: Path) -> None:
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    roots.append(resolved)

            repo_root = Path(__file__).resolve().parents[2]
            _add(repo_root)

            cwd = Path.cwd().resolve()
            _add(cwd)
            for parent in cwd.parents:
                _add(parent)

            if first_component in {"outputs", "models", "checkpoints"}:
                for ancestor in (cwd, *cwd.parents):
                    if ancestor.name == first_component:
                        _add(ancestor.parent)

            env_paths = os.environ.get("PREFADAP_MODEL_PATH_HINTS")
            if env_paths:
                for entry in env_paths.split(os.pathsep):
                    entry = entry.strip()
                    if not entry:
                        continue
                    _add(Path(entry).expanduser())

            return roots

        for base in _iter_search_roots():
            candidate = base / model_name_or_path
            if candidate.exists():
                if candidate.is_dir():
                    return _resolve_directory(candidate)
                return str(candidate)

            if first_component:
                anchored = base / first_component
                if anchored.exists():
                    # Instead of anchored.parent / path_obj (which is redundant),
                    # resolve the path relative to the anchor.
                    candidate = anchored / Path(*path_obj.parts[1:])
                    if candidate.exists():
                        if candidate.is_dir():
                            return _resolve_directory(candidate)
                        return str(candidate)

        logger.debug(
            "Model path '%s' not found locally; returning unchanged for downstream loaders.",
            model_name_or_path,
        )
        return model_name_or_path

    return model_name_or_path


@register_model("causal")
def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    gradient_checkpointing: bool = False,
    tokenizer_name_or_path: str | None = None,
    config: dict[str, object] | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM and its tokenizer with sensible defaults.

    ``model_name_or_path`` may refer either to a full model checkpoint or to a
    PEFT adapter.  Local directories containing ``adapter_model.safetensors``
    or ``adapter_config.json`` are treated as adapters.  Adapters can also be
    loaded from Hugging Face repositories by providing ``base_model`` in
    ``config`` which specifies the base model to combine with the adapter.  If
    ``peft``'s ``AutoPeftModelForCausalLM`` is available, local adapter
    directories without an explicit ``base_model`` will be loaded via that
    helper.

    Parameters
    ----------
    model_name_or_path:
        Hugging Face repo ID or local path.
    gradient_checkpointing:
        Whether to enable gradient checkpointing on the loaded model.
    tokenizer_name_or_path:
        Optional separate tokenizer source.
    config:
        Additional keyword arguments forwarded to
        :func:`transformers.AutoModelForCausalLM.from_pretrained`.  Supports
        two special keys:

        - ``device``: move the returned model to this device after loading.
        - ``base_model``: treat ``model_name_or_path`` as a PEFT adapter and
          load it on top of the specified base model.
    """

    # Log initial memory state if CUDA is available
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated()
        logger.debug(f"Initial GPU memory: {initial_memory / BYTES_TO_GB:.2f} GB")

    env_var = os.getenv("PREFER_EAGER_ATTENTION")
    if env_var is not None:
        prefer_eager = env_var.lower() not in {"0", "false", "no"}
    else:
        prefer_eager = hasattr(torch.version, "hip") or torch.version.cuda is None

    attn_impl = (
        "eager"
        if prefer_eager or "gemma-2" in model_name_or_path.lower()
        else "flash_attention_2"
    )
    load_in_4bit = "70b" in model_name_or_path.lower() and not prefer_eager
    kwargs: dict[str, object] = {"attn_implementation": attn_impl}
    if load_in_4bit:
        kwargs.update(
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "device_map": "auto",
            }
        )
    cfg = config.copy() if config else {}
    base_model_name = cfg.pop("base_model", None)
    allow_adapter_stack = bool(cfg.pop("allow_adapter_stack", False))
    explicit_base_model = base_model_name
    device = cfg.pop("device", None)
    model_type = str(cfg.pop("model_type", "")).strip().lower()
    
    kwargs.update(cfg)
    disable_device_map = bool(kwargs.pop("disable_device_map", False))
    if disable_device_map:
        kwargs.pop("device_map", None)
    elif kwargs.get("device_map") is None:
        kwargs.pop("device_map", None)
    
    # Import centralized sharding helpers (inline to avoid circular dependency)
    from prefadap.training.utils import apply_default_sharding, safe_move_model

    if not model_type:
        for candidate in (model_name_or_path, base_model_name, tokenizer_name_or_path):
            if isinstance(candidate, str) and "qwen3" in candidate.lower():
                model_type = "qwen3"
                break

    if model_type in {"qwen", "qwen2", "qwen3"}:
        kwargs.setdefault("trust_remote_code", True)
    # Check if this is a local path that doesn't exist yet
    model_name_or_path = _resolve_model_path(model_name_or_path)

    model_path = Path(model_name_or_path)
    adapter_file = model_path / "adapter_model.safetensors"
    adapter_config = model_path / "adapter_config.json"
    adapter_config_records_base = False
    adapter_recorded_base: Optional[str] = None
    if adapter_config.exists():
        try:
            data = json.loads(adapter_config.read_text())
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            recorded = data.get("base_model_name_or_path") or data.get("base_model")
            if isinstance(recorded, str) and recorded.strip():
                adapter_config_records_base = True
                candidate_base = recorded.strip()
                if candidate_base:
                    if _identifiers_match(candidate_base, model_name_or_path):
                        candidate_base = None
                    else:
                        candidate_path = Path(candidate_base)
                        if candidate_path.exists() and is_adapter_directory(candidate_path):
                            nested = get_adapter_base_model(candidate_path)
                            if nested and not _identifiers_match(nested, candidate_base):
                                candidate_base = nested
                    adapter_recorded_base = candidate_base
                    if adapter_recorded_base is not None and base_model_name is None:
                        base_model_name = adapter_recorded_base

    adapter_file_exists = adapter_file.exists()
    is_adapter_path = adapter_file_exists or adapter_config.exists() or base_model_name is not None

    if explicit_base_model is not None and adapter_recorded_base is not None:
        if _identifiers_match(explicit_base_model, adapter_recorded_base):
            explicit_base_model = None
            base_model_name = adapter_recorded_base

    if is_adapter_path:
        if (
            adapter_config_records_base
            and AutoPeftModelForCausalLM is not None
            and adapter_file_exists
            and explicit_base_model is None
        ):
            # Apply default sharding policy (device_map="auto" only for large models on multi-GPU)
            if not disable_device_map:
                apply_default_sharding(kwargs, model_name_or_path)
            used_device_map = "device_map" in kwargs
            if used_device_map:
                device = None  # Don't move model manually when using device_map
            
            model = AutoPeftModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
            loaded_params = _count_parameters(model)
            logger.info(
                "Loaded adapter model from %s via AutoPeftModelForCausalLM (params=%s, peft=%s)",
                model_name_or_path,
                loaded_params if loaded_params is not None else "unknown",
                bool(getattr(model, "peft_config", None)),
            )
            # Note: When device_map is used, model is sharded - do not set model.device
            tok_candidate = adapter_recorded_base or model_name_or_path
            tok_src = tokenizer_name_or_path or tok_candidate
        else:
            if base_model_name is None:
                inferred_base = get_adapter_base_model(model_path)
                if inferred_base:
                    base_model_name = inferred_base
            if base_model_name is None:
                raise ValueError(
                    "Adapter path provided but 'base_model' missing from config"
                )
            if PeftModel is None:  # pragma: no cover - peft missing
                raise ImportError("Install 'peft' to load adapter models")
            resolved_base = _resolve_model_path(base_model_name)
            
            # Apply default sharding policy (device_map="auto" only for large models on multi-GPU)
            if not disable_device_map:
                apply_default_sharding(kwargs, resolved_base)
            used_device_map = "device_map" in kwargs
            if used_device_map:
                device = None  # Don't move model manually when using device_map
            
            base_path = Path(resolved_base)
            base_is_adapter = is_adapter_directory(base_path)
            base_adapter_recorded: Optional[str] = None
            tokenizer_base_hint = resolved_base

            if base_is_adapter:
                base_adapter_recorded = get_adapter_base_model(base_path)
                if not allow_adapter_stack and not _identifiers_match(
                    model_name_or_path, str(base_path)
                ):
                    raise ValueError(
                        "Base model "
                        f"'{base_model_name}' already contains LoRA adapters. "
                        "Set config['allow_adapter_stack']=True to chain adapters explicitly."
                    )

                if AutoPeftModelForCausalLM is not None:
                    base_model = AutoPeftModelForCausalLM.from_pretrained(
                        resolved_base, **kwargs
                    )
                    base_params = _count_parameters(base_model)
                    logger.info(
                        "Loaded base adapter model %s via AutoPeftModelForCausalLM (params=%s, peft=%s)",
                        resolved_base,
                        base_params if base_params is not None else "unknown",
                        bool(getattr(base_model, "peft_config", None)),
                    )
                else:
                    if base_adapter_recorded is None:
                        raise ValueError(
                            "Base model "
                            f"'{base_model_name}' is a PEFT adapter without a recorded base."
                        )
                    nested_resolved = _resolve_model_path(base_adapter_recorded)
                    nested_kwargs = prepare_hf_from_pretrained_kwargs(
                        nested_resolved, **kwargs
                    )
                    base_model = AutoModelForCausalLM.from_pretrained(
                        nested_resolved, **nested_kwargs
                    )
                    base_params = _count_parameters(base_model)
                    base_model = PeftModel.from_pretrained(base_model, resolved_base)
                    merged_params = _count_parameters(base_model)
                    logger.info(
                        "Loaded chained adapter %s onto base %s (params: %s -> %s, peft=%s)",
                        resolved_base,
                        nested_resolved,
                        base_params if base_params is not None else "unknown",
                        merged_params if merged_params is not None else "unknown",
                        bool(getattr(base_model, "peft_config", None)),
                    )
                    resolved_base = nested_resolved
                tokenizer_base_hint = base_adapter_recorded or resolved_base
            else:
                resolved_kwargs = prepare_hf_from_pretrained_kwargs(
                    resolved_base, **kwargs
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    resolved_base, **resolved_kwargs
                )
            
            # Note: When device_map is used, model is sharded - do not set model.device

            if base_is_adapter and _identifiers_match(model_name_or_path, str(base_path)):
                model = base_model
                tok_src = tokenizer_name_or_path or tokenizer_base_hint
            else:
                base_params = _count_parameters(base_model)
                model = PeftModel.from_pretrained(base_model, model_name_or_path)
                adapted_params = _count_parameters(model)
                logger.info(
                    "Loaded adapter %s onto base %s (params: %s -> %s, peft=%s)",
                    model_name_or_path,
                    resolved_base,
                    base_params if base_params is not None else "unknown",
                    adapted_params if adapted_params is not None else "unknown",
                    bool(getattr(model, "peft_config", None)),
                )
                if adapted_params is not None and base_params == adapted_params:
                    logger.warning(
                        "Adapter load did not change parameter count for %s (base %s).",
                        model_name_or_path,
                        resolved_base,
                    )
                # Note: When device_map is used, model is sharded - do not set model.device
                tok_candidate = adapter_recorded_base or tokenizer_base_hint
                tok_src = tokenizer_name_or_path or tok_candidate
    else:
        # Apply default sharding policy (device_map="auto" only for large models on multi-GPU)
        if not disable_device_map:
            apply_default_sharding(kwargs, model_name_or_path)
        used_device_map = "device_map" in kwargs
        if used_device_map:
            device = None  # Don't move model manually when using device_map
        
        final_kwargs = prepare_hf_from_pretrained_kwargs(
            model_name_or_path, **kwargs
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **final_kwargs
        )
        # Note: When device_map is used, model is sharded - do not set model.device
        tok_src = tokenizer_name_or_path or model_name_or_path
    tok_kwargs = prepare_hf_from_pretrained_kwargs(
        tok_src, validate_access=False
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_src, **tok_kwargs)
    ensure_pad_token(tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Move model to device if not already sharded
    safe_move_model(model, device, "model")

    if gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        logger.info(
            f"Gradient checkpointing enabled for model {model_name_or_path} to reduce memory footprint"
        )

        # Log memory impact if CUDA is available
        if torch.cuda.is_available():
            post_checkpoint_memory = torch.cuda.memory_allocated()
            logger.info(
                f"GPU memory after checkpointing setup: {post_checkpoint_memory / BYTES_TO_GB:.2f} GB"
            )

    return model, tokenizer


@register_model("reward")
def load_reward_model_and_tokenizer(
    model_name_or_path: str,
    *,
    gradient_checkpointing: bool = False,
    tokenizer_name_or_path: str | None = None,
    config: dict[str, object] | None = None,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load a sequence classification model and tokenizer for reward modelling.

    Parameters are analogous to :func:`load_model_and_tokenizer` with the same
    ``config`` forwarding semantics.
    """

    # Log initial memory state if CUDA is available
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated()
        logger.debug(f"Initial GPU memory: {initial_memory / BYTES_TO_GB:.2f} GB")

    # Check if this is a local path that doesn't exist yet
    model_name_or_path = _resolve_model_path(model_name_or_path)

    cfg = config.copy() if config else {}
    if "num_labels" in cfg and cfg["num_labels"] != 1:
        raise ValueError(
            "Reward model must be scalar (num_labels=1). "
            f"Received num_labels={cfg['num_labels']}."
        )
    if "problem_type" in cfg and cfg["problem_type"] != "regression":
        raise ValueError(
            "Reward model must use regression heads (problem_type='regression'). "
            f"Received problem_type={cfg['problem_type']}."
        )
    cfg["num_labels"] = 1
    cfg["problem_type"] = "regression"
    device = cfg.pop("device", None)
    disable_device_map = bool(cfg.pop("disable_device_map", False))
    if disable_device_map:
        cfg.pop("device_map", None)
    elif cfg.get("device_map") is None:
        cfg.pop("device_map", None)
    
    # Import centralized sharding helpers (inline to avoid circular dependency)
    from prefadap.training.utils import apply_default_sharding, safe_move_model
    
    # Apply default sharding policy (device_map="auto" only for large models on multi-GPU)
    if not disable_device_map:
        apply_default_sharding(cfg, model_name_or_path)
    used_device_map = "device_map" in cfg
    if used_device_map:
        device = None  # Don't move model manually when using device_map
    
    model_kwargs = prepare_hf_from_pretrained_kwargs(
        model_name_or_path, **cfg
    )
    
    # Explicit handling for OLMo-3 models: AutoModelForSequenceClassification
    # cannot load Olmo3Config due to lack of dynamic registration support in
    # transformers 4.55+. Load directly via Olmo3ForSequenceClassification.
    from transformers import AutoConfig
    from transformers.models.olmo3 import Olmo3Config

    model_dir = Path(model_name_or_path)
    adapter_config = model_dir / "adapter_config.json"
    config_path = model_dir / "config.json"
    adapter_only = False
    if model_dir.exists() and model_dir.is_dir() and adapter_config.exists():
        if not config_path.exists():
            adapter_only = True
        else:
            try:
                config_data = json.loads(config_path.read_text())
            except json.JSONDecodeError:
                config_data = None
            if not isinstance(config_data, dict) or not config_data.get("model_type"):
                adapter_only = True

    if adapter_only:
        from peft import PeftModel

        base_model_name = get_adapter_base_model(model_dir)
        if not base_model_name:
            raise ValueError(
                "Legacy RM adapter checkpoint is missing base model metadata. "
                "Expected adapter_config.json to include base_model_name_or_path."
            )

        base_kwargs = prepare_hf_from_pretrained_kwargs(
            base_model_name, **cfg
        )
        config_obj = AutoConfig.from_pretrained(base_model_name, **base_kwargs)
        config_obj.num_labels = 1
        config_obj.problem_type = "regression"

        if isinstance(config_obj, Olmo3Config):
            for k in ("num_labels", "problem_type"):
                base_kwargs.pop(k, None)

            from prefadap.models.olmo3_seqcls import Olmo3ForSequenceClassification

            base_model = Olmo3ForSequenceClassification.from_pretrained(
                base_model_name,
                config=config_obj,
                **base_kwargs,
            )
        else:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name, **base_kwargs
            )

        base_params = _count_parameters(base_model)
        peft_model = PeftModel.from_pretrained(base_model, model_name_or_path)
        adapted_params = _count_parameters(peft_model)
        logger.info(
            "Loaded adapter %s onto reward base %s (params: %s -> %s, peft=%s)",
            model_name_or_path,
            base_model_name,
            base_params if base_params is not None else "unknown",
            adapted_params if adapted_params is not None else "unknown",
            bool(getattr(peft_model, "peft_config", None)),
        )
        if adapted_params is not None and base_params == adapted_params:
            logger.warning(
                "Reward adapter load did not change parameter count for %s (base %s).",
                model_name_or_path,
                base_model_name,
            )
        merged_model = peft_model.merge_and_unload()
        try:
            merged_model.save_pretrained(model_name_or_path, safe_serialization=True)
        except TypeError:
            merged_model.save_pretrained(model_name_or_path)
        model = merged_model
    else:
        # Load config to check if this is an OLMo-3 model
        config_obj = AutoConfig.from_pretrained(model_name_or_path, **model_kwargs)
        config_obj.num_labels = 1
        config_obj.problem_type = "regression"

        if isinstance(config_obj, Olmo3Config):
            # Classification semantics live ONLY on config
            config_obj.num_labels = 1
            config_obj.problem_type = "regression"

            # Remove classification-only kwargs before model construction
            for k in ("num_labels", "problem_type"):
                model_kwargs.pop(k, None)

            from prefadap.models.olmo3_seqcls import Olmo3ForSequenceClassification

            model = Olmo3ForSequenceClassification.from_pretrained(
                model_name_or_path,
                config=config_obj,
                **model_kwargs,
            )
        else:
            # Not an OLMo-3 model, use standard AutoModel
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, **model_kwargs
            )
    if model.config.num_labels != 1:
        raise ValueError(
            "Reward model must be scalar (num_labels=1). "
            f"Classification-style reward models are not supported (got {model.config.num_labels})."
        )
    # Note: When device_map is used, model is sharded - do not set model.device
    tok_src = tokenizer_name_or_path or model_name_or_path
    tok_kwargs = prepare_hf_from_pretrained_kwargs(
        tok_src, validate_access=False
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_src, **tok_kwargs)
    ensure_pad_token(tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Move model to device if not already sharded
    safe_move_model(model, device, "reward model")

    if gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        logger.info(
            f"Gradient checkpointing enabled for reward model {model_name_or_path} to reduce memory footprint"
        )

        # Log memory impact if CUDA is available
        if torch.cuda.is_available():
            post_checkpoint_memory = torch.cuda.memory_allocated()
            logger.info(
                f"GPU memory after checkpointing setup: {post_checkpoint_memory / BYTES_TO_GB:.2f} GB"
            )

    return model, tokenizer


from prefadap.training.lora import apply_lora  # re-export for backward compatibility
