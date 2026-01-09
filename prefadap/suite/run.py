from __future__ import annotations

"""Unified end-to-end runner for training and generation runs."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from prefadap.config import load_dict
import torch

# Path handling -------------------------------------------------------------
# ``run.py`` lives under ``prefadap/suite``. ``ROOT`` should reference the
# repository root so outputs are written relative to the top-level project
# directory and child processes can locate project modules.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_run_id(exp_name: str) -> str:
    """Return a timestamped run identifier for ``exp_name``."""

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{stamp}_{exp_name}"


def build_combinations(
    seeds: Iterable[int | None], hparams: Iterable[Dict[str, Any]]
) -> List[Tuple[int | None, Dict[str, Any]]]:
    """Return Cartesian product of ``seeds`` and ``hparams``.
    """

    s_list = list(seeds) or [None]
    h_list = list(hparams) or [{}]
    return list(product(s_list, h_list))


def _requires_accelerate(pipeline: str, args: Dict[str, Any]) -> bool:
    if pipeline in {"ppo", "grpo"}:
        return True
    if pipeline == "dualrlhf":
        return str(args.get("objective", "")).lower() == "ppo"
    return False


def _resolve_num_processes() -> int:
    env_sources = [
        "ACCELERATE_PROCESS_COUNT",
        "WORLD_SIZE",
    ]
    for key in env_sources:
        value = os.environ.get(key)
        if value:
            try:
                return max(int(value), 1)
            except ValueError:
                continue
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        return max(len([v for v in cuda_visible.split(",") if v.strip()]), 1)
    if torch.cuda.is_available():
        return max(torch.cuda.device_count(), 1)
    return 1


def _build_cmd(pipeline: str, args: Dict[str, Any]) -> List[str]:
    """Return command list to invoke the training pipeline.

    ``prefadap.cli.run_training`` is executed as a module to ensure the
    repository root is on ``sys.path``. Calling the file via a relative path
    can omit the package root from the module search path and cause import
    errors. Executing the module avoids this issue irrespective of the current
    working directory.
    """

    if _requires_accelerate(pipeline, args):
        num_processes = _resolve_num_processes()
        cmd: List[str] = [
            "accelerate",
            "launch",
            "--num_processes",
            str(num_processes),
            "-m",
            "prefadap.cli.run_training",
            pipeline,
        ]
    else:
        cmd = [sys.executable, "-m", "prefadap.cli.run_training", pipeline]
    for k, v in args.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
            continue
        cmd.extend([f"--{k}", str(v)])
    return cmd


def _run_training(pipeline: str, args: Dict[str, Any]) -> int:
    """Invoke the training subprocess with the repo root on ``PYTHONPATH``."""

    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.run(_build_cmd(pipeline, args), env=env).returncode


def _run_generation(run_name: str, generation_config: Dict[str, Any], force: bool) -> int:
    """Invoke generation for a training run."""
    
    from .generation import generate_for_run
    from prefadap.utils.paths import resolve_run_dir

    run_dir = resolve_run_dir(ROOT / "runs" / run_name)
    success = generate_for_run(run_dir, generation_config, force)
    return 0 if success else 1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML configuration file")
    parser.add_argument("--force", action="store_true", help="Re-run all stages")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation step")
    args = parser.parse_args()

    cfg = load_dict(args.config)
    exp = cfg.get("exp_name", "experiment")
    run_id = cfg.get("run_id", generate_run_id(exp))

    base_dir = ROOT / "runs" / run_id
    base_dir.mkdir(parents=True, exist_ok=True)

    seeds = cfg.get("seeds", [None])
    hparams = cfg.get("hparams", [{}])
    combos = build_combinations(seeds, hparams)

    train_cfg = cfg.get("train", {})
    generation_cfg = cfg.get("generation", {})
    
    for idx, (seed, hp) in enumerate(combos):
        pipeline_name = train_cfg.get("pipeline") if train_cfg else None
        skip_post_training = pipeline_name == "rm"
        parts: List[str] = []
        if seed is not None:
            parts.append(f"seed{seed}")
        if hp:
            parts.append(f"hp{idx}")
        sub = "/".join(parts) if parts else "run"
        run_name = f"{run_id}/{sub}"
        run_dir = ROOT / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup markers for tracking completion
        train_marker = run_dir / ".train.done"
        generate_marker = run_dir / ".generate.done"
        
        if args.force:
            train_marker.unlink(missing_ok=True)
            generate_marker.unlink(missing_ok=True)
        
        # Training step
        if train_cfg and not train_marker.exists():
            t_args = train_cfg.get("args", {}).copy()
            if seed is not None:
                t_args["seed"] = seed
            t_args.update(hp)
            # Ensure subprocess writes outputs to the designated run directory
            t_args["run_name"] = run_name
            t_args["run_id"] = run_name
            t_args["output_dir"] = run_dir
            ret = _run_training(train_cfg["pipeline"], t_args)
            if ret != 0:
                sys.exit(ret)

            # Record training artifacts for downstream stages
            from .generation import _get_model_path

            summary = {"output_dir": str(run_dir)}
            final_model = _get_model_path(run_dir)
            if final_model is not None:
                summary["final_model"] = str(final_model)
            with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            train_marker.touch()

        if skip_post_training and train_marker.exists():
            print(f"Skipping generation for RM pipeline run: {run_name}")
            generate_marker.touch()
            continue

        # Generation step (only if training completed and generation not skipped)
        if not args.skip_generation and train_marker.exists() and not generate_marker.exists():
            ret = _run_generation(run_name, generation_cfg, force=args.force)
            if ret != 0:
                print(f"Warning: Generation failed for {run_name}, continuing")
                # Don't exit on generation failure, but mark it as attempted
            generate_marker.touch()



if __name__ == "__main__":  # pragma: no cover
    main()
