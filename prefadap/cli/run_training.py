#!/usr/bin/env python3
"""Unified training dispatcher.

This script dispatches to different training pipelines based on a
positional *pipeline key*.  The key selects the dataclass used to
load configuration and the corresponding ``TrainingPipeline``
implementation.  Example usage:

```
python -m prefadap.cli.run_training sft --config path/to/config.yaml
python -m prefadap.cli.run_training dpo --config path/to/config.yaml --force-training

# Multi-GPU PPO/GRPO/Dual-RLHF (PPO objective) requires Accelerate + DeepSpeed ZeRO-3:
accelerate launch \
  --num_processes <num_gpus> \
  -m prefadap.cli.run_training ppo --config path/to/config.yaml

# Single-GPU PPO baselines (no DeepSpeed, no distributed init) can be run via:
# python -m prefadap.cli.run_training ppo --config path/to/config.yaml --ppo-single-gpu-baseline
# Single-GPU GRPO baselines (no DeepSpeed, no distributed init) can be run via:
# python -m prefadap.cli.run_training grpo --config path/to/config.yaml --grpo-single-gpu-baseline
```

Configuration files may optionally set ``fp16`` or ``bf16`` to override
automatic precision selection. When omitted, the training pipeline chooses
the most suitable precision based on available hardware.

The ``--force-training`` flag can be used to retrain even if a model already
exists in the output directory.
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path

from prefadap.config import load_config
from prefadap.utils.logging import disable_tqdm_if_not_tty, setup_logging
from prefadap.utils.seeding import set_seed

# NOTE: Avoid importing training dataclasses or pipelines at module import time.
# Some environments running lightweight tests may not have heavy dependencies
# (e.g. transformers/trl/torch) installed. We therefore import the dataclasses
# and pipeline implementations lazily inside helper functions.


# The set of valid pipeline keys supported by this dispatcher.
AVAILABLE_PIPELINES = {
    "sft",
    "dpo",
    "kto",
    "orpo",
    "dapt",
    "grpo",
    "ppo",
    "online_dpo",
    "rm",
    "dualdpo",
    "dann_dpo",
    "dann_sft",
    "dann_kto",
    "dann_orpo",
    "dualrlhf",
    "sft_zero_shot",
}


def _resolve_pipeline(pipeline_key):
    """Return (ArgsDataclass, TrainingPipeline) for the given key.

    Imports are performed lazily to keep module import light-weight.
    """

    # Import here to avoid importing torch/transformers at module import time
    from prefadap.training import config as tconf
    from prefadap.training import training_wrappers as tw

    mapping = {
        "sft": (tconf.SFTArgs, tw.SFTTrainingPipeline),
        "dpo": (tconf.DPOArgs, tw.DPOTrainingPipeline),
        "kto": (tconf.KTOArgs, tw.KTOTrainingPipeline),
        "orpo": (tconf.ORPOArgs, tw.ORPOTrainingPipeline),
        "dapt": (tconf.DAPTArgs, tw.DAPTTrainingPipeline),
        "grpo": (tconf.GRPOArgs, tw.GRPOTrainingPipeline),
        "ppo": (tconf.PPOArgs, tw.PPOTrainingPipeline),
        "online_dpo": (tconf.OnlineDPOArgs, tw.OnlineDPOTrainingPipeline),
        "rm": (tconf.RMArgs, tw.RewardModelTrainingPipeline),
        "dualdpo": (tconf.DualDPOArgs, tw.DualDPOTrainingPipeline),
        "dann_dpo": (tconf.DANNArgs, tw.DANNTrainingPipeline),
        "dann_sft": (tconf.DANNArgs, tw.DANNSFTTrainingPipeline),
        "dann_kto": (tconf.DANNArgs, tw.DANNKTOTrainingPipeline),
        "dann_orpo": (tconf.DANNArgs, tw.DANNORPOTrainingPipeline),
        "dualrlhf": (tconf.DualRLHFArgs, tw.DualRLHFTrainingPipeline),
        "sft_zero_shot": (tconf.SFTArgs, tw.SFTTrainingPipeline),
    }
    return mapping[pipeline_key]


def main() -> None:
    if len(sys.argv) < 2:
        available = ", ".join(sorted(AVAILABLE_PIPELINES))
        raise SystemExit(
            f"Usage: {Path(sys.argv[0]).name} <pipeline> --config CONFIG "
            "[--force-training] [--quiet] [--ppo-single-gpu-baseline] "
            "[--grpo-single-gpu-baseline]\n"
            f"Available pipelines: {available}"
        )

    import logging

    pipeline_key = sys.argv[1].lower()
    if pipeline_key not in AVAILABLE_PIPELINES:
        available = ", ".join(sorted(AVAILABLE_PIPELINES))
        raise SystemExit(f"Unknown pipeline '{pipeline_key}'. Choose from: {available}")

    # ``--quiet`` is parsed here to avoid passing it to dataclass parsers.
    quiet = False
    if "--quiet" in sys.argv:
        quiet = True
        sys.argv.remove("--quiet")

    # ``--force-training`` is parsed here to avoid passing it to dataclass parsers.
    force_training = False
    if "--force-training" in sys.argv:
        force_training = True
        sys.argv.remove("--force-training")

    # ``--ppo-single-gpu-baseline`` is parsed here to avoid passing it to dataclass parsers.
    ppo_single_gpu_baseline = False
    if "--ppo-single-gpu-baseline" in sys.argv:
        ppo_single_gpu_baseline = True
        sys.argv.remove("--ppo-single-gpu-baseline")

    # ``--grpo-single-gpu-baseline`` is parsed here to avoid passing it to dataclass parsers.
    grpo_single_gpu_baseline = False
    if "--grpo-single-gpu-baseline" in sys.argv:
        grpo_single_gpu_baseline = True
        sys.argv.remove("--grpo-single-gpu-baseline")

    disable_tqdm_if_not_tty()
    setup_logging(quiet=quiet)

    logger = logging.getLogger(__name__)
    from prefadap.training.utils import log_accelerate_state

    log_accelerate_state(logger)
    from prefadap.training.utils import log_accelerate_config_resolution

    log_accelerate_config_resolution(logger)

    # Remove the pipeline key so that ``load_config`` only sees the remaining
    # arguments (e.g. ``--config path").
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    args_cls, pipeline_cls = _resolve_pipeline(pipeline_key)
    cfg = load_config(args_cls)

    if ppo_single_gpu_baseline and pipeline_key != "ppo":
        raise SystemExit(
            "--ppo-single-gpu-baseline is only valid when running the 'ppo' pipeline."
        )
    if grpo_single_gpu_baseline and pipeline_key != "grpo":
        raise SystemExit(
            "--grpo-single-gpu-baseline is only valid when running the 'grpo' pipeline."
        )

    if pipeline_key == "ppo":
        from prefadap.training.utils import (
            is_distributed_environment,
            validate_ppo_distributed_launch,
            validate_ppo_single_gpu_baseline,
        )

        import torch

        if ppo_single_gpu_baseline:
            cfg.ppo_single_gpu_baseline = True
        if (
            not cfg.ppo_single_gpu_baseline
            and torch.cuda.is_available()
            and torch.cuda.device_count() == 1
            and not torch.distributed.is_initialized()
            and not is_distributed_environment()
        ):
            cfg.ppo_single_gpu_baseline = True
            logger.info(
                "PPO mode: auto-selected single-GPU baseline (no distributed launch detected)."
            )
        if cfg.ppo_single_gpu_baseline:
            logger.info(
                "PPO mode: single-GPU baseline (DeepSpeed disabled, no distributed init)."
            )
            validate_ppo_single_gpu_baseline()
        else:
            logger.info("PPO mode: standard DeepSpeed ZeRO-3 (Accelerate launch).")
            validate_ppo_distributed_launch(cfg)
    elif pipeline_key == "grpo":
        from prefadap.training.utils import (
            is_distributed_environment,
            validate_grpo_single_gpu_baseline,
            validate_ppo_distributed_launch,
        )

        import torch

        if grpo_single_gpu_baseline:
            cfg.grpo_single_gpu_baseline = True
        if (
            not cfg.grpo_single_gpu_baseline
            and torch.cuda.is_available()
            and torch.cuda.device_count() == 1
            and not torch.distributed.is_initialized()
            and not is_distributed_environment()
        ):
            cfg.grpo_single_gpu_baseline = True
            logger.info(
                "GRPO mode: auto-selected single-GPU baseline (no distributed launch detected)."
            )
        if cfg.grpo_single_gpu_baseline:
            logger.info(
                "GRPO mode: single-GPU baseline (DeepSpeed disabled, no distributed init)."
            )
            validate_grpo_single_gpu_baseline()
        else:
            logger.info("GRPO mode: standard DeepSpeed ZeRO-3 (Accelerate launch).")
            validate_ppo_distributed_launch(cfg)
    elif pipeline_key == "dualrlhf" and getattr(cfg, "objective", "").lower() == "ppo":
        from prefadap.training.utils import validate_ppo_distributed_launch

        validate_ppo_distributed_launch(cfg, experiment_name="Dual-RLHF (PPO)")
    
    # Apply the force_training flag if it was provided
    if force_training:
        cfg.force_training = True
    
    set_seed(cfg.seed)
    pipeline = pipeline_cls(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
