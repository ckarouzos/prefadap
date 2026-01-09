"""Lightweight registries for extensible components.

This module implements a small utility :class:`Registry` which maps keys to
callables. It powers plugin-like registration of models, datasets, metrics and
training pipelines throughout the codebase.

External modules typically import the helper functions exposed by
``prefadap.models.registry`` or ``prefadap.training.registry`` rather than
interacting with these instances directly.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Generic, Hashable, List, TypeVar

K = TypeVar("K", bound=Hashable)


class Registry(Generic[K]):
    """Simple mapping from keys to callables with decorator helpers."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: Dict[K, Callable[..., Any]] = {}

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def register(self, key: K) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register ``key`` with a builder function."""

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            if key in self._registry:  # pragma: no cover - defensive
                raise ValueError(f"{self.name!s} '{key}' already registered")
            self._registry[key] = fn
            return fn

        return decorator

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get(self, key: K) -> Callable[..., Any]:
        """Return the callable registered under ``key``."""

        try:
            return self._registry[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown {self.name!s} '{key}'") from exc

    def create(self, key: K, *args: Any, **kwargs: Any) -> Any:
        """Instantiate ``key`` with provided arguments."""

        builder = self.get(key)
        return builder(*args, **kwargs)

    def available(self) -> List[K]:
        """Return a sorted list of registered keys."""

        return sorted(self._registry)


# Public registries ---------------------------------------------------------

model_registry: Registry[str] = Registry("model")
dataset_registry: Registry[tuple[str, str]] = Registry("dataset")
metric_registry: Registry[str] = Registry("metric")
adaptation_registry: Registry[str] = Registry("adaptation")


__all__ = [
    "Registry",
    "model_registry",
    "dataset_registry",
    "metric_registry",
    "adaptation_registry",
]
