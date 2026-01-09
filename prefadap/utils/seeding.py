"""Centralized seeding utilities.

This module provides a single :func:`set_seed` function that initialises the
random number generators of Python's ``random`` module, NumPy and PyTorch.  It
also configures PyTorch to use deterministic algorithms where possible and
disables CuDNN benchmarking to ensure reproducible runs across the codebase.
"""

from __future__ import annotations

import os
import random

try:  # pragma: no cover - optional dependency
    import numpy as np
except ImportError:  # pragma: no cover - numpy may be unavailable
    np = None  # type: ignore

try:  # pragma: no cover - torch may be unavailable in tests
    import torch
except ImportError:  # pragma: no cover - torch may be unavailable
    torch = None  # type: ignore


np_rng: np.random.Generator | None = None


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch when available.

    Parameters
    ----------
    seed:
        The integer seed to apply to all supported libraries.
    """
    global np_rng  # pylint: disable=global-statement

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if np is not None:
        # Seed both the legacy global RNG and expose a seeded ``Generator``.
        np.random.seed(seed)
        try:  # pragma: no cover - ``default_rng`` may be missing on old NumPy
            np_rng = np.random.default_rng(seed)
        except AttributeError:
            # Fallback for older NumPy versions without default_rng
            np_rng = None
    if torch is not None:
        try:  # pragma: no cover - torch may be a lightweight stub
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except (RuntimeError, AttributeError):
            # When ``torch`` is a minimal stub or running on limited hardware,
            # certain attributes or CUDA backends may be unavailable. Swallow
            # those specific errors to keep seeding best‑effort.
            pass


__all__ = ["set_seed", "np_rng"]

