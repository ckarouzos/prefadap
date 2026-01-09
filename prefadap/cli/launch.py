#!/usr/bin/env python3
"""Command line interface for experiment sweeps and pipelines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prefadap.config import load_dict
from prefadap.suite.launcher import run_pipeline, run_single, sweep_configs


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    sweep_p = sub.add_parser("sweep", help="Run a sweep from a config file")
    sweep_p.add_argument("config", type=Path, help="YAML configuration file")
    sweep_p.add_argument("--results-dir", type=Path, default=Path("runs"))
    sweep_p.add_argument("--force", action="store_true", help="Overwrite existing runs")

    pipe_p = sub.add_parser(
        "pipeline", help="Run training/eval/aggregate pipeline for configs"
    )
    pipe_p.add_argument("config_dir", type=Path, help="Directory of config files")
    pipe_p.add_argument("--runs-dir", type=Path, default=Path("runs"))

    args = parser.parse_args()

    if args.mode == "sweep":
        base_cfg = load_dict(args.config)
        for cfg in sweep_configs(base_cfg):
            run_single(cfg, args.results_dir, force=args.force)
        return

    if args.mode == "pipeline":
        sys.exit(run_pipeline(args.config_dir, args.runs_dir))


if __name__ == "__main__":  # pragma: no cover
    main()

