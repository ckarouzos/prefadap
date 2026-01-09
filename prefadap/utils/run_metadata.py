import os
import subprocess
import sys
import time
from typing import Dict, Any

import torch


def build_run_id(alg: str, adaptation: str, src: str, tgt: str, model: str, size: str, sched: str, seed: int) -> str:
    """Return a canonical run identifier.

    The identifier has the following structure::

        ALG-DA-SRC->TGT-MODEL-SIZE-SCHED-SEED
    """

    return f"{alg}-{adaptation}-{src}->{tgt}-{model}-{size}-{sched}-{seed}"


def collect_run_metadata(
    token_counts: Dict[str, int],
    avg_kl: Dict[str, float],
    steps: int,
    start_time: float,
    seed: int,
) -> Dict[str, Any]:
    """Collect metadata about a training or evaluation run."""

    end_time = time.time()
    wall_clock_seconds = end_time - start_time
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.getcwd(), timeout=5
        ).decode().strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        git_sha = "unknown"
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], timeout=30
        ).decode().splitlines()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        freeze = []
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    metadata: Dict[str, Any] = {
        "token_counts": token_counts,
        "avg_kl": avg_kl,
        "steps": steps,
        "total_tokens": int(sum(token_counts.values())),
        "seed": seed,
        "wall_clock_seconds": wall_clock_seconds,
        "git_sha": git_sha,
        "pip_freeze": freeze,
        "gpu": gpu,
        "start_time": start_time,
    }
    return metadata


__all__ = ["build_run_id", "collect_run_metadata"]
