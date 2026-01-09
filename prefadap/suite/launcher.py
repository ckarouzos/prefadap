"""Lightweight experiment launch utilities.

This module provides small helpers for expanding configuration sweeps,
flattening nested configuration dictionaries into command line arguments
and orchestrating simple training pipelines.  The functions are
pure and intentionally free of any environment-specific glue so that they can be
reused by different frontends such as the in-package CLI or thin batch
wrappers.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import copy
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
import torch

from prefadap.config import RunConfig, load_dict, validate
from .generation import _get_model_path
from prefadap.utils.paths import resolve_run_dir
import logging
import tempfile
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scratch path resolution
# ---------------------------------------------------------------------------


def _resolve_scratch_base() -> Path:
    """Resolve scratch directory with portable fallback.
    
    Returns a writable base directory for experiment runs, falling back
    to tempfile when system-specific scratch directories are unavailable.
    """
    base = Path(os.environ.get("SCRATCHDIR", "/scratch"))
    if not base.exists() or not os.access(base, os.W_OK):
        base = Path(tempfile.gettempdir()) / "prefadap_runs"
    return base


# ---------------------------------------------------------------------------
# Configuration handling
# ---------------------------------------------------------------------------


def sweep_configs(base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand list-valued hyperparameters and seeds into a config sweep."""

    hyper = base_cfg.get("hyperparams", {})
    sweep_keys = [k for k, v in hyper.items() if isinstance(v, list)]
    sweep_values = [hyper[k] if isinstance(hyper[k], list) else [hyper[k]] for k in sweep_keys]

    seeds = base_cfg.get("seed", [None])
    if not isinstance(seeds, Iterable) or isinstance(seeds, (str, bytes)):
        seeds = [seeds]

    runs: List[Dict[str, Any]] = []
    for combo in itertools.product(*sweep_values, seeds):
        cfg = copy.deepcopy(base_cfg)
        for key, value in zip(sweep_keys, combo[:-1]):
            cfg.setdefault("hyperparams", {})[key] = value
        cfg["seed"] = combo[-1]
        runs.append(cfg)

    return runs or [base_cfg]


def flatten_args(cfg: Dict[str, Any], skip_keys: Iterable[str] = ()) -> Dict[str, Any]:
    """Flatten nested configuration dictionaries into CLI arguments."""

    def add(prefix: str, d: Dict[str, Any], acc: Dict[str, Any]) -> None:
        for key, value in d.items():
            if value is None or key in skip_keys:
                continue
            name = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                add(f"{name}_", value, acc)
            else:
                acc[name] = value

    args: Dict[str, Any] = {}
    add("", cfg.get("model", {}), args)
    add("", cfg.get("hyperparams", {}), args)

    for key in [
        "objective",
        "da_method",
        "dataset",
        "source_dataset",
        "target_dataset",
        "uda_budget_tokens",
        "seed",
        "data_dir",
        "output_base",
    ]:
        if key in cfg and cfg[key] is not None and key not in skip_keys:
            args[key] = cfg[key]

    logging_cfg = cfg.get("logging", {})
    if logging_cfg.get("project"):
        args["wandb_project"] = logging_cfg["project"]
    if logging_cfg.get("run_name"):
        args["run_name"] = logging_cfg["run_name"]
    if logging_cfg.get("wandb"):
        args["wandb"] = logging_cfg["wandb"]
    if logging_cfg.get("log_token_count"):
        args["log_token_count"] = logging_cfg["log_token_count"]
    if logging_cfg.get("log_kl"):
        args["log_kl"] = logging_cfg["log_kl"]
    if logging_cfg.get("log_time"):
        args["log_time"] = logging_cfg["log_time"]

    acc_cfg = cfg.get("accelerator", {})
    if acc_cfg.get("flash_attention"):
        args["use_flash_attention"] = True

    return args


# ---------------------------------------------------------------------------
# Command construction and execution
# ---------------------------------------------------------------------------


def _explicit_deepspeed_requested(cfg: Dict[str, Any]) -> bool:
    accelerator_cfg = cfg.get("accelerator", {})
    if isinstance(accelerator_cfg, dict):
        return any(
            accelerator_cfg.get(key)
            for key in (
                "deepspeed",
                "use_deepspeed",
                "deepspeed_config",
                "deepspeed_config_file",
            )
        )
    return False


def _requires_accelerate_pipeline(pipeline: str, cfg: Dict[str, Any]) -> bool:
    if pipeline == "ppo":
        if cfg.get("ppo_single_gpu_baseline"):
            return False
        if torch.cuda.is_available() and torch.cuda.device_count() == 1:
            if not _explicit_deepspeed_requested(cfg):
                return False
        return True
    if pipeline == "grpo":
        # PPO already bypasses Accelerate on single-GPU runs; GRPO previously
        # always forced Accelerate, causing distributed port conflicts.
        if cfg.get("grpo_single_gpu_baseline"):
            return False
        if torch.cuda.is_available() and torch.cuda.device_count() == 1:
            if not _explicit_deepspeed_requested(cfg):
                return False
        return True
    if pipeline == "dualrlhf":
        return str(cfg.get("objective", "")).lower() == "ppo"
    return False


def _repo_root() -> Path:
    env_root = os.environ.get("PREFADAP_REPO_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _accelerate_deepspeed_config_path() -> Path | None:
    path = _repo_root() / "configs" / "accelerate_deepspeed_zero3.yaml"
    return path if path.exists() else None


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


def _extract_pipeline(script_tokens: List[str], cfg: Dict[str, Any]) -> str:
    if script_tokens and script_tokens[0].endswith("run_training.py"):
        if len(script_tokens) >= 2:
            return script_tokens[1]
    if len(script_tokens) >= 3 and script_tokens[0] == "-m" and script_tokens[1] == "prefadap.cli.run_training":
        return script_tokens[2]
    objective = cfg.get("objective")
    if objective:
        return str(objective)
    raise ValueError("run_training.py requires a pipeline token")


def _build_run_training_cmd(
    script_tokens: List[str],
    cfg: Dict[str, Any],
    config_path: Path,
    accelerate_config: Path | None = None,
) -> List[str]:
    pipeline = _extract_pipeline(script_tokens, cfg)
    if _requires_accelerate_pipeline(pipeline, cfg):
        num_processes = _resolve_num_processes()
        cmd = [
            "accelerate",
            "launch",
            "--num_processes",
            str(num_processes),
        ]
        if accelerate_config:
            cmd.extend(["--config_file", str(accelerate_config)])
        cmd.extend([*script_tokens, "--config", str(config_path)])
        return cmd
    return ["python", *script_tokens, "--config", str(config_path)]


def build_cmd(
    script: str | Iterable[str], args: Dict[str, Any], *, hyphenate: bool = True
) -> List[str]:
    """Return command list to invoke ``script`` with ``args``.

    Parameters
    ----------
    script:
        The training script to execute.  May be a string or an
        iterable of command tokens.
    args:
        Mapping of argument names to values.
    hyphenate:
        When ``True`` (the default) underscores in argument names are replaced
        with hyphens.  This mirrors the behaviour of :mod:`argparse` based
        CLIs.  Set to ``False`` when the target script expects underscores.
    """

    tokens = shlex.split(script) if isinstance(script, str) else list(script)
    cmd = ["python"] + tokens
    for key, value in args.items():
        opt = key.replace("_", "-") if hyphenate else key
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{opt}")
        else:
            cmd.append(f"--{opt}")
            cmd.append(str(value))
    return cmd


def _script_tokens(field: Any) -> List[str]:
    if isinstance(field, list):
        return [str(x) for x in field]
    if isinstance(field, str):
        return shlex.split(field)
    raise TypeError(f"Unsupported script field type: {type(field)}")


def run_single(cfg: Dict[str, Any], results_dir: Path, *, force: bool = False) -> None:
    """Execute a single configuration ``cfg`` and record its outputs."""

    # Auto-populate script field from experiment_type if missing
    if "script" not in cfg and "experiment_type" in cfg:
        experiment_type = cfg["experiment_type"].lower()
        
        # Extract base algorithm from complex experiment_type
        # Available pipelines: sft, dpo, kto, orpo, dapt, grpo, ppo, online_dpo, rm, dualdpo, dann_dpo, dann_kto, dann_orpo, dualrlhf
        available_pipelines = {
            "sft", "dpo", "kto", "orpo", "dapt", "grpo", "ppo", 
            "online_dpo", "rm", "dualdpo", "dann_dpo", "dann_kto", "dann_orpo", "dualrlhf"
        }
        
        # Find matching pipeline by checking if any pipeline name is at the start of experiment_type
        base_algorithm = None
        for pipeline in available_pipelines:
            if experiment_type.startswith(pipeline):
                base_algorithm = pipeline
                break
        
        if base_algorithm is None:
            # Fallback: use the experiment_type as-is (for backward compatibility)
            base_algorithm = experiment_type
        
        cfg["script"] = ["-m", "prefadap.cli.run_training", base_algorithm]

    # Preserve the original run_id before validation to ensure it's not lost
    original_run_id = cfg.get("run_id")
    
    # Strictly validate config using RunConfig; fail fast if validation fails
    try:
        run_config_fields = asdict(validate(RunConfig, cfg))
        # Merge the validated essential fields back into the original config
        for key, value in run_config_fields.items():
            if key not in cfg and value is not None:
                cfg[key] = value
    except Exception as e:
        logger.error("Config validation failed: %s", e)
        raise ValueError(f"Configuration validation failed: {e}")
    # Ensure run_id is preserved even if validation had issues
    if original_run_id and not cfg.get("run_id"):
        cfg["run_id"] = original_run_id
        logger.warning("Restored run_id after validation: %s", original_run_id)
    
    # Define a clear hierarchy for run_name resolution
    run_name = None
    if "logging" in cfg and isinstance(cfg["logging"], dict):
        run_name = cfg["logging"].get("run_name")
    if not run_name:
        run_name = cfg.get("run_name")
    if not run_name:
        run_name = cfg.get("run_id")
    if not run_name or not str(run_name).strip():
        run_name = "run"
        logger.warning("run_name could not be determined from config; using default 'run'.")
    requested_out = cfg.get("output_dir")
    if requested_out:
        out_path = Path(requested_out)
        if os.environ.get("RUN_BASE_ROOT"):
            try:
                # Preserve model namespace (e.g., outputs/qwen/...) when applicable
                out_path = resolve_run_dir(
                    requested_out,
                    model_name_or_path=cfg.get("model_name_or_path"),
                )
            except ValueError:
                out_path = Path(requested_out)
        out_dir = Path(out_path)
        cfg["output_dir"] = str(out_dir)
    else:
        out_dir = results_dir / run_name
        cfg["output_dir"] = str(out_dir)

    if out_dir.exists() and not force:
        report = out_dir / "report.json"
        summary_path = out_dir / "summary.json"
        if report.exists():
            return
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except Exception:  # pragma: no cover - defensive
                summary = {}
            if summary.get("returncode", 1) == 0:
                return
        if any(out_dir.glob("checkpoint*")):
            cfg.setdefault("hyperparams", {})["resume"] = True

    script_tokens = _script_tokens(cfg["script"])
    out_dir.mkdir(parents=True, exist_ok=True)
    # Ensure training pipelines write outputs into the run directory
    cfg.setdefault("run_id", run_name)
    with (out_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    accelerate_config = None
    dispatcher = (
        script_tokens 
        and (
            script_tokens[0].endswith("run_training.py")
            or (len(script_tokens) >= 2 and script_tokens[0] == "-m" and script_tokens[1] == "prefadap.cli.run_training")
        )
    )
    if dispatcher:
        if len(script_tokens) < 2:
            obj = cfg.get("objective")
            if not obj:
                raise ValueError("run_training.py requires a pipeline token")
            script_tokens = [script_tokens[0], str(obj)]
        repo_root = _repo_root()
        accelerate_config = None
        if _requires_accelerate_pipeline(_extract_pipeline(script_tokens, cfg), cfg):
            accelerate_config = _accelerate_deepspeed_config_path()
        cmd = _build_run_training_cmd(
            script_tokens,
            cfg,
            out_dir / "config.yaml",
            accelerate_config,
        )
    else:
        args = flatten_args(cfg)
        cmd = build_cmd(script_tokens, args)

    # Stream logs directly to files so users can tail them live during long runs
    start = time.time()
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    # Ensure subprocess can import project modules regardless of installation
    src_path = Path(__file__).resolve().parents[2]
    env = dict(os.environ, RUN_DIR=str(out_dir))
    repo_root = _repo_root()
    if cmd and cmd[0] == "accelerate":
        accel_env = accelerate_config or _accelerate_deepspeed_config_path()
        if accel_env:
            env["ACCELERATE_CONFIG_FILE"] = str(accel_env)
        logger.info("Accelerate launch cwd=%s", repo_root)
    env["PYTHONPATH"] = f"{src_path}:{env.get('PYTHONPATH', '')}"
    with stdout_path.open("w", encoding="utf-8") as f_out, stderr_path.open(
        "w", encoding="utf-8"
    ) as f_err:
        proc = subprocess.run(
            cmd,
            stdout=f_out,
            stderr=f_err,
            text=True,
            env=env,
            cwd=repo_root,
        )
    end = time.time()
    summary = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "duration": end - start,
        "output_dir": str(out_dir),
    }
    final_model = _get_model_path(out_dir)
    if final_model is not None:
        summary["final_model"] = str(final_model)
    (out_dir / "summary.json").write_text(json.dumps(summary))
    if proc.returncode != 0:
        # Show context from stderr to aid debugging
        try:
            from collections import deque

            with stderr_path.open("r", encoding="utf-8", errors="replace") as f:
                err_excerpt = "".join(deque(f, maxlen=20))
            if err_excerpt.strip():
                print(err_excerpt, file=sys.stderr, end="")
        except Exception:  # pragma: no cover - best effort
            pass

        # Print brief failure context and point to log files
        print(
            f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}",
            file=sys.stderr,
        )
        print(
            f"Full logs written to: {stderr_path} and {stdout_path}",
            file=sys.stderr,
        )
        raise subprocess.CalledProcessError(proc.returncode, cmd)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def _read_config(path: Path) -> Dict[str, Any]:
    data = load_dict(path)
    cfg = validate(RunConfig, data)
    return asdict(cfg)


def _config_paths(config_dir: Path) -> Iterable[Path]:
    for path in sorted(config_dir.iterdir()):
        if path.suffix in {".yml", ".yaml", ".json"}:
            yield path


def run_pipeline(config_dir: Path, runs_dir: Path = Path("runs")) -> int:
    """Execute training runs for configs in ``config_dir``."""

    runs_dir = resolve_run_dir(runs_dir)
    
    # Add /scratch → /tmp fallback for CI permission issues
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        if str(runs_dir).startswith("/scratch"):
            runs_dir = Path("/tmp/prefadap_ci_fallback") / runs_dir.relative_to("/scratch")
            runs_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise

    status = 0
    for cfg_path in _config_paths(config_dir):
        cfg = _read_config(cfg_path)
        run_name = cfg.get("logging", {}).get("run_name") or cfg.get("run_name") or "run"
        run_dir = resolve_run_dir(runs_dir / run_name)
        train_marker = run_dir / ".train.done"
        run_dir.mkdir(parents=True, exist_ok=True)

        if not train_marker.exists():
            script_tokens = _script_tokens(cfg["script"])
            args = flatten_args(cfg)
            accelerate_config = None
            dispatcher = (
                script_tokens
                and (
                    script_tokens[0].endswith("run_training.py")
                    or (
                        len(script_tokens) >= 2
                        and script_tokens[0] == "-m"
                        and script_tokens[1] == "prefadap.cli.run_training"
                    )
                )
            )
            if dispatcher:
                args["run_id"] = run_name
                args["output_dir"] = str(run_dir)
            cmd = build_cmd(script_tokens, args, hyphenate=False)
            if dispatcher:
                pipeline = _extract_pipeline(script_tokens, cfg)
                if _requires_accelerate_pipeline(pipeline, cfg):
                    accelerate_config = _accelerate_deepspeed_config_path()
                    num_processes = _resolve_num_processes()
                    base_cmd = cmd
                    cmd = ["accelerate", "launch", "--num_processes", str(num_processes)]
                    if accelerate_config:
                        cmd.extend(["--config_file", str(accelerate_config)])
                    cmd.extend(base_cmd)
            repo_root = _repo_root()
            env = dict(os.environ, RUN_DIR=str(run_dir))
            if cmd and cmd[0] == "accelerate":
                accel_env = accelerate_config or _accelerate_deepspeed_config_path()
                if accel_env:
                    env["ACCELERATE_CONFIG_FILE"] = str(accel_env)
                logger.info("Accelerate launch cwd=%s", repo_root)
            proc = subprocess.run(cmd, env=env, cwd=repo_root)
            if proc.returncode == 0:
                try:
                    rel_dir = run_dir.relative_to(Path.cwd())
                    out_str = str(rel_dir)
                except ValueError:
                    out_str = str(run_dir)
                summary = {
                    "cmd": cmd,
                    "returncode": proc.returncode,
                    "output_dir": out_str,
                }
                final_model = _get_model_path(run_dir)
                if final_model is not None:
                    summary["final_model"] = str(final_model)
                (run_dir / "summary.json").write_text(json.dumps(summary))
                train_marker.touch()
            else:
                status = proc.returncode or 1
                continue

    return status


__all__ = [
    "sweep_configs",
    "flatten_args",
    "build_cmd",
    "run_single",
    "run_pipeline",
]
