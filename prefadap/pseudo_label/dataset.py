"""Helpers for constructing pseudo preference datasets."""

from __future__ import annotations

import json
import os
import random
from typing import Optional

# Check if W&B is disabled via environment variable
if os.environ.get("DISABLE_WANDB") == "1":
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore
else:
    try:
        import wandb  # type: ignore
        # Guard against shadowed local `wandb/` dirs (namespace package without API)
        WANDB_AVAILABLE = hasattr(wandb, "init")
    except ImportError:
        WANDB_AVAILABLE = False
        wandb = None  # type: ignore

from prefadap.utils.logging import create_logger
from prefadap.utils.seeding import set_seed


def construct_pseudo_dataset(
    pairs_path: str,
    output_path: str,
    max_pairs: Optional[int] = None,
    seed: int = 42,
    logging_dir: str = "./logs",
    wandb_project: str = "construct_pseudo",
) -> None:
    """Subsample and finalise pseudo preference pairs into ``output_path``.

    Parameters
    ----------
    pairs_path:
        Path to a JSONL file containing records with ``prompt``, ``chosen``,
        ``rejected`` and optional ``weight`` fields.
    output_path:
        Location where the sampled pseudo dataset is written.
    max_pairs:
        Optional cap on the number of preference pairs to include.  If ``None``
        all pairs are written.
    seed:
        Random seed used for subsampling.
    logging_dir:
        Directory for log files.
    wandb_project:
        Weights & Biases project under which metrics are recorded.
    """

    logger = create_logger(logging_dir)
    logger.info("Starting pseudo-dataset construction...")
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")
    
    if WANDB_AVAILABLE:
        wandb.init(project=wandb_project, name="construct_pseudo")
    else:
        logger.info("W&B not available, skipping initialization")

    logger.info(f"Loading filtered pairs from: {pairs_path}")
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                new_rec = {
                    "prompt": rec["prompt"],
                    "chosen": rec["chosen"],
                    "rejected": rec["rejected"],
                }
                if "weight" in rec:
                    new_rec["weight"] = rec["weight"]
                pairs.append(new_rec)
            except json.JSONDecodeError:
                continue
    total = len(pairs)
    logger.info(f"Total candidate pairs loaded: {total}")
    if WANDB_AVAILABLE:
        wandb.log({"total_pairs_loaded": total})

    if max_pairs and total > max_pairs:
        pairs = random.sample(pairs, max_pairs)
        logger.info(f"Subsampled to {max_pairs} pairs using seed={seed}")
        if WANDB_AVAILABLE:
            wandb.log({"pairs_subsampled": max_pairs})

    logger.info(f"Writing pseudo dataset to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as out_file:
        for rec in pairs:
            out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if WANDB_AVAILABLE:
        wandb.log({"final_pairs": len(pairs)})
        wandb.finish()
    logger.info("Pseudo-dataset construction complete.")


__all__ = ["construct_pseudo_dataset"]
