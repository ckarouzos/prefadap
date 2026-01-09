"""Configuration loading and schema utilities.

This module provides helpers for reading configuration files with support for
inheritance/composition and validating them against dataclasses or Pydantic
models.  Configuration files may reference one or more parent files via an
``extends`` field whose entries are resolved relative to the including file.
Later files override earlier ones using a deep merge strategy.

The module also exposes a small :class:`RunConfig` dataclass used by the
experiment launcher utilities.  Users can supply multiple ``--config`` files to
training CLIs; subsequent files act as partial updates, overriding keys from
previous files while maintaining backward compatibility with existing single
file usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Type, TypeVar
import argparse
import json
import copy

from pydantic import BaseModel, TypeAdapter

try:  # optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML may be missing
    yaml = None

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Schema for experiment run configuration files used by the launcher."""

    script: str | List[str]
    model: Dict[str, Any] = field(default_factory=dict)
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    objective: str | None = None
    da_method: str | None = None
    dataset: str | None = None
    source_dataset: str | None = None
    target_dataset: str | None = None
    uda_budget_tokens: int | None = None
    seed: int | None = None
    data_dir: str | None = None
    output_base: str | None = None
    accelerator: Dict[str, Any] = field(default_factory=dict)
    
    # Essential training config fields that the launcher needs to be aware of
    model_name_or_path: str | None = None
    run_id: str | None = None
    experiment_type: str | None = None
    dataset_name: str | None = None


# ---------------------------------------------------------------------------
# Merging and file resolution
# ---------------------------------------------------------------------------


def _merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``update`` into ``base`` (in-place)."""

    for key, value in update.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _read_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        if yaml is not None and path.suffix in {".yml", ".yaml"}:
            return yaml.safe_load(f) or {}
        return json.load(f)


def _load_with_extends(path: Path, seen: set[Path] | None = None) -> Dict[str, Any]:
    """Load ``path`` resolving any ``extends`` entries recursively."""

    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Cyclic config include detected for {path}")
    seen.add(path)

    data = _read_file(path)
    base: Dict[str, Any] = {}
    parents = data.pop("extends", [])
    if isinstance(parents, (str, Path)):
        parents = [parents]
    for rel in parents:
        parent_path = (path.parent / rel).resolve()
        parent_cfg = _load_with_extends(parent_path, seen)
        _merge(base, parent_cfg)
    _merge(base, data)
    return base


def load_dict(path: Path) -> Dict[str, Any]:
    """Return configuration dictionary for ``path`` resolving inheritance.

    Parameters
    ----------
    path:
        Location of the configuration file.  Paths may originate from run
        directories that mirror the repository layout.  If the file does not
        exist, the function also checks the same relative path from the
        repository root.

    Raises
    ------
    FileNotFoundError
        If the configuration file cannot be located either at ``path`` or via
        the repository fallback mechanism.
    """

    path = Path(path)
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [path, repo_root / path]

    for candidate in candidates:
        if candidate.exists():
            return _load_with_extends(candidate.resolve())

    raise FileNotFoundError(
        f"Config file not found: {path}. Ensure the path is correct and "
        "that the repository contains the requested config."
    )


def load_configs(paths: Iterable[Path]) -> Dict[str, Any]:
    """Merge and return configuration dictionaries from ``paths``."""

    cfg: Dict[str, Any] = {}
    for p in paths:
        _merge(cfg, load_dict(p))
    return cfg


# ---------------------------------------------------------------------------
# Validation and schema
# ---------------------------------------------------------------------------


def validate(cls: Type[T], data: Dict[str, Any]) -> T:
    """Validate ``data`` against ``cls`` using :class:`pydantic.TypeAdapter`.
    
    Intelligently handles None values to allow default factories to work while
    preserving intentional None values for proper validation.
    """
    import dataclasses
    
    # Smart None handling: only remove None values for fields that have default factories
    # This prevents masking validation errors for fields that shouldn't be None
    cleaned_data = data.copy()
    
    extras: Dict[str, Any] = {}
    if dataclasses.is_dataclass(cls):
        # Check which fields have default factories
        for field_name, field_info in cls.__dataclass_fields__.items():
            has_default_factory = (
                hasattr(field_info, "default_factory")
                and field_info.default_factory is not dataclasses.MISSING
            )

            if (
                field_name in data
                and data[field_name] is None
                and has_default_factory
            ):
                # Remove None to let default factory work
                cleaned_data.pop(field_name, None)

        # Drop unknown keys for dataclass validation to be tolerant of
        # extra fields injected by higher-level pipeline wrappers.
        allowed = set(cls.__dataclass_fields__.keys())
        allowed_extras = {"_config_metadata"}
        extras = {k: cleaned_data[k] for k in allowed_extras if k in cleaned_data}
        cleaned_data = {k: v for k, v in cleaned_data.items() if k in allowed}
    else:
        # For non-dataclass types, use the original logic as fallback
        cleaned_data = {k: v for k, v in data.items() if v is not None}
    
    adapter = TypeAdapter(cls)
    try:
        result = adapter.validate_python(cleaned_data)
        if extras and dataclasses.is_dataclass(result):
            for key, value in extras.items():
                setattr(result, key, value)
        return result
    except Exception as e:
        error_message = str(e)
        
        # Provide helpful error messages for common validation issues
        if "model_name_or_path" in error_message and "required" in error_message.lower():
            # Check if the field was actually provided in the original data
            if "model_name_or_path" in data and data["model_name_or_path"] is not None:
                raise ValueError(
                    f"model_name_or_path validation failed despite being provided as '{data['model_name_or_path']}'. "
                    "This suggests an internal configuration processing issue."
                ) from e
            else:
                raise ValueError(
                    "model_name_or_path is required but was not provided or is None. "
                    "Please specify a model name such as 'gpt2', 'meta-llama/Meta-Llama-3-8B', "
                    "or another valid model identifier in your configuration file."
                ) from e
        
        # Re-raise the original error for other cases
        raise


def schema(cls: Type[T]) -> Dict[str, Any]:
    """Return a JSON schema for ``cls``."""

    if isinstance(cls, type) and issubclass(cls, BaseModel):
        return cls.model_json_schema()
    adapter = TypeAdapter(cls)
    return adapter.json_schema()


# ---------------------------------------------------------------------------
# CLI loader
# ---------------------------------------------------------------------------


def load_config(cls: Type[T]) -> T:
    """Load ``cls`` from one or more ``--config`` files."""

    parser = argparse.ArgumentParser(description=f"Load {cls.__name__} from file")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to YAML/JSON configuration file(s)",
    )
    args = parser.parse_args()
    paths = [Path(p) for p in args.config]
    data = load_configs(paths)
    return validate(cls, data)


__all__ = [
    "RunConfig",
    "load_dict",
    "load_configs",
    "load_config",
    "schema",
    "validate",
]
