from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml  # type: ignore


def load_decoding_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load decoding parameters from a YAML file.

    The function attempts to parse ``path`` as YAML and falls back to JSON
    when :mod:`PyYAML` is unavailable.  If ``path`` is ``None``, the default
    ``configs/decoding.yaml`` file relative to the repository root is used.
    """

    if path is None:
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / "configs" / "decoding.yaml"
    else:
        path = Path(path)

    print(f"Loading decoding config from {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_policy(dataset: str | None = None) -> Dict[str, Any]:
    """Return decoding parameters for ``dataset``.

    The returned dictionary merges the global settings from
    ``configs/decoding.yaml`` with any dataset-specific overrides. Dataset
    names are matched case-insensitively.  If ``dataset`` is ``None``, only the
    global settings are returned.  A :class:`ValueError` is raised for unknown
    dataset names.
    """

    cfg = load_decoding_config()
    global_cfg = {k: v for k, v in cfg.items() if k not in {"datasets", "seed"}}
    if dataset is None:
        return global_cfg

    datasets = cfg.get("datasets", {})
    lookup = {k.lower(): v for k, v in datasets.items()}
    key = dataset.lower()
    if key not in lookup:
        options = ", ".join(sorted(datasets)) or "<none>"
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available datasets: {options}"
        )
    merged = {**global_cfg, **lookup[key]}
    return merged


__all__ = ["load_decoding_config", "load_policy"]
