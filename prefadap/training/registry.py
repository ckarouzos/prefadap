"""Registry helpers for training pipelines."""

from __future__ import annotations

from prefadap.registry import adaptation_registry


register_adaptation = adaptation_registry.register
get_adaptation = adaptation_registry.get
create_adaptation = adaptation_registry.create


__all__ = ["register_adaptation", "get_adaptation", "create_adaptation"]
