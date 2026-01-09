"""CLI for subsampling and finalising pseudo preference datasets.

Run via ``python -m prefadap.cli.construct_pseudo_dataset`` to downsample
filtered prompt/completion pairs and write a JSONL dataset suitable for DPO
training.
"""

from __future__ import annotations

import argparse

from prefadap.pseudo_label.dataset import construct_pseudo_dataset


def main() -> None:
    """Parse CLI arguments and construct the pseudo preference dataset.

    CLI Arguments
    -------------
    --pairs_path:
        Input JSONL file containing filtered prompt/chosen/rejected pairs.
    --max_pairs:
        Maximum number of pairs to include; ``None`` keeps all pairs.
    --seed:
        Random seed controlling the subsampling process.
    --output_path:
        Path where the sampled dataset will be written.
    --logging_dir:
        Directory to store logs.
    --wandb_project:
        Weights & Biases project name for tracking.

    Side Effects
    ------------
    Writes the resulting pseudo dataset to ``output_path`` and logs to
    ``logging_dir``.
    """

    parser = argparse.ArgumentParser(
        description="Subsample and construct pseudo preference dataset for DPO",
    )
    parser.add_argument(
        "--pairs_path",
        type=str,
        required=True,
        help="Input JSONL of filtered prompt/chosen/rejected pairs",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Maximum number of pairs to include (None=all)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for subsampling"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output JSONL path for pseudo dataset",
    )
    parser.add_argument(
        "--logging_dir", type=str, default="./logs", help="Directory for logs"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="construct_pseudo",
        help="Weights & Biases project name",
    )
    args = parser.parse_args()

    construct_pseudo_dataset(
        pairs_path=args.pairs_path,
        output_path=args.output_path,
        max_pairs=args.max_pairs,
        seed=args.seed,
        logging_dir=args.logging_dir,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

