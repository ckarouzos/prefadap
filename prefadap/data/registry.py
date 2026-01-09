"""Dataset registry and validation utilities."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

from prefadap.dataset_names import validate_dataset_name
from prefadap.registry import dataset_registry

# Internal registries ---------------------------------------------------------

# Mapping of task -> helpers such as loader/tokenizer
_TASK_REGISTRY: dict[str, dict[str, Callable[..., object]]] = defaultdict(dict)


def _make_dataset_key(task: str, name: str) -> tuple[str, str]:
    """Construct a unique key for dataset ``name`` within ``task``."""

    return (task, name)


def register_dataset(name: str, task: str) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Register ``name`` for ``task`` with the given loader function."""

    key = _make_dataset_key(task, name)
    return dataset_registry.register(key)


def register_task(
    task: str,
    *,
    loader: Callable[..., object] | None = None,
    tokenizer: Callable[..., object] | None = None,
) -> None:
    """Register helpers for a given ``task``.

    Parameters
    ----------
    task:
        Identifier of the task (e.g. ``"summarisation"``).
    loader:
        Optional dataset builder function.
    tokenizer:
        Optional tokenisation function used by the task.
    """

    if loader is not None:
        _TASK_REGISTRY[task]["loader"] = loader
    if tokenizer is not None:
        _TASK_REGISTRY[task]["tokenizer"] = tokenizer


def get_dataset_loader(name: str, task: str) -> Callable[..., object]:
    """Retrieve the loader registered for ``name`` and ``task``."""

    validate_dataset_name(name)
    key = _make_dataset_key(task, name)
    try:
        return dataset_registry.get(key)
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"No loader registered for dataset '{name}' and task '{task}'"
        ) from exc


def get_task_loader(task: str) -> Callable[..., object]:
    """Return the dataset builder registered for ``task``."""

    try:
        return _TASK_REGISTRY[task]["loader"]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"No loader registered for task '{task}'") from exc


def get_task_tokenizer(task: str) -> Callable[..., object]:
    """Return the tokenisation function registered for ``task``."""

    try:
        return _TASK_REGISTRY[task]["tokenizer"]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"No tokenizer registered for task '{task}'") from exc
