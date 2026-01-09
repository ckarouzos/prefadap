"""Registry helpers for model builder functions."""

from __future__ import annotations

from prefadap.registry import model_registry


register_model = model_registry.register
get_model_builder = model_registry.get
create_model = model_registry.create


__all__ = ["register_model", "get_model_builder", "create_model"]
