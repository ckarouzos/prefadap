#!/usr/bin/env python3
"""Unified command-line interface for preference adaptation experiments.

This is the main entry point for the prefadap CLI, providing subcommands for
training and listing available resources.

Subcommands:
- train: Train preference-adapted models with various algorithms
- list: List available pipelines and configurations
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

PIPELINES = [
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
]


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    
    parser = argparse.ArgumentParser(
        prog="prefadap",
        description="Unified CLI for preference adaptation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True
    
    # Train subcommand
    train_parser = subparsers.add_parser(
        "train",
        help="Train preference-adapted models with various algorithms"
    )
    train_parser.add_argument(
        "pipeline",
        choices=PIPELINES,
        help="Training pipeline to use"
    )
    train_parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file"
    )
    train_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console logging"
    )
    
    # List subcommand
    list_parser = subparsers.add_parser(
        "list",
        help="List available pipelines, models, and configurations"
    )
    list_parser.add_argument(
        "item",
        choices=["pipelines", "configs"],
        nargs="?",
        default="pipelines",
        help="What to list (default: pipelines)"
    )
    
    return parser


def cmd_train(args: argparse.Namespace) -> None:
    """Execute train subcommand."""
    # Import here to avoid heavy dependencies at startup
    from prefadap.cli.run_training import main as training_main
    
    # Reconstruct argv for run_training.py
    train_argv = [args.pipeline, "--config", args.config]
    if args.quiet:
        train_argv.append("--quiet")
    
    # Call the training main with our constructed argv
    original_argv = sys.argv[:]
    try:
        sys.argv = ["prefadap train"] + train_argv
        training_main()
    finally:
        sys.argv = original_argv


def cmd_list(args: argparse.Namespace) -> None:
    """Execute list subcommand."""
    import os
    
    if args.item == "pipelines":
        # Hardcode the pipelines to avoid importing heavy dependencies
        pipelines = PIPELINES
        print("Available training pipelines:")
        for pipeline in sorted(pipelines):
            print(f"  {pipeline}")
    elif args.item == "configs":
        # Find the configs directory relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        configs_dir = os.path.join(repo_root, "configs")
        
        if not os.path.exists(configs_dir):
            print("configs/ directory not found")
            return
            
        print("Available configuration files:")
        
        # Group configs by category
        categories = {
            "Common/Base configs": [],
            "Algorithm defaults": [],
            "Templates": [],
            "Experiment configs": []
        }
        
        for root, dirs, files in os.walk(configs_dir):
            for file in files:
                if file.endswith('.yaml'):
                    rel_path = os.path.relpath(os.path.join(root, file), configs_dir)
                    
                    if rel_path.startswith('common/') or rel_path == 'decoding.yaml':
                        categories["Common/Base configs"].append(rel_path)
                    elif rel_path == 'algorithm_defaults.yaml' or rel_path == 'plugins.yaml':
                        categories["Algorithm defaults"].append(rel_path)
                    elif rel_path.startswith('templates/'):
                        categories["Templates"].append(rel_path)
                    elif rel_path.startswith('experiments/'):
                        categories["Experiment configs"].append(rel_path)
                    else:
                        categories["Common/Base configs"].append(rel_path)
        
        for category, configs in categories.items():
            if configs:
                print(f"\n{category}:")
                for config in sorted(configs):
                    print(f"  {config}")
                    
        print(f"\nTotal: {sum(len(configs) for configs in categories.values())} configuration files")
        print("Use config files with: prefadap train <pipeline> --config configs/<path>")


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point for the unified CLI."""
    
    # If we have specific known subcommands that need special handling,
    # check for them early
    if argv is None:
        argv = sys.argv[1:]
    
    if not argv:
        # Show help when no args provided
        parser = create_parser()
        parser.print_help()
        return
    
    # Normal argument parsing for other commands
    parser = create_parser()
    args = parser.parse_args(argv)
    
    try:
        if args.command == "train":
            cmd_train(args)
        elif args.command == "list":
            cmd_list(args)
        else:
            parser.error(f"Unknown command: {args.command}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
