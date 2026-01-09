"""Command-line interface for the diversity evaluation pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from prefadap.evaluation import DiversityConfig, compute_diversity, plot_diversity_bars

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_ngrams(value: str) -> tuple[int, ...]:
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("ngram list must contain at least one integer")
    if any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("n-gram orders must be positive integers")
    return tuple(sorted(set(parts)))


def _resolve_output_path(path: Path | None, outputs_dir: Path) -> Path | None:
    """
    Resolve output path to an absolute path.
    
    If path is None, returns None (no output file).
    If path is relative, converts it to absolute rooted at the current working directory.
    If path is already absolute, returns it as-is.
    
    Args:
        path: The output path to resolve (may be None, relative, or absolute)
        outputs_dir: The outputs directory being evaluated (used for default path generation)
        
    Returns:
        Resolved absolute path or None
    """
    if path is None:
        return None
    
    # Convert to Path if string
    path = Path(path)
    
    # If already absolute, return as-is
    if path.is_absolute():
        return path
    
    # If relative, make it absolute rooted at CWD
    # This ensures we don't write to read-only locations inside containers
    abs_path = Path.cwd() / path
    logger.info(f"Resolved relative output path '{path}' to absolute: '{abs_path}'")
    return abs_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate output diversity metrics following the protocol from Section 5.2."
    )
    parser.add_argument("outputs_dir", type=Path, help="Directory containing model generations")
    parser.add_argument(
        "--metrics",
        default="EAD,SBERT,NLI",
        help="Comma separated metric names to compute (default: EAD,SBERT,NLI)",
    )
    parser.add_argument("--model-name", default=None, help="Optional model label for reporting")
    parser.add_argument(
        "--max-inputs",
        type=int,
        default=500,
        help="Number of prompts to evaluate (N, protocol default: 500)",
    )
    parser.add_argument(
        "--generations-per-input",
        type=int,
        default=16,
        help="Number of generations per prompt (K, protocol default: 16)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature used to produce the generations (protocol: 1.0)",
    )
    parser.add_argument(
        "--ead-ngrams",
        type=_parse_ngrams,
        default=(1, 2, 3, 4, 5),
        help="Comma separated list of n-gram orders for the EAD metric",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for subsampling")
    parser.add_argument(
        "--sbert-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Sentence-BERT checkpoint to encode generations",
    )
    parser.add_argument("--sbert-batch-size", type=int, default=32)
    parser.add_argument(
        "--sbert-cache-dir",
        type=Path,
        default=None,
        help="Optional directory to cache SBERT embeddings for reuse",
    )
    parser.add_argument(
        "--nli-model",
        default="roberta-large-mnli",
        help="NLI checkpoint for logical diversity (default: roberta-large-mnli)",
    )
    parser.add_argument("--nli-batch-size", type=int, default=8)
    parser.add_argument(
        "--nli-pairs-per-input",
        type=int,
        default=-1,
        help="Number of NLI pairs per prompt (-1 uses all)",
    )
    parser.add_argument(
        "--nli-pairs-across",
        type=int,
        default=-1,
        help="Number of NLI pairs across prompts (-1 uses all)",
    )
    parser.add_argument(
        "--nli-random-subsample",
        type=float,
        default=None,
        help="Optional fraction of NLI pairs to sample (0<f≤1). Default uses all",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Transformer tokenizer max length")
    parser.add_argument("--allow-partial", action="store_true", help="Allow fewer than N prompts")
    parser.add_argument("--verbose", action="store_true", help="Display progress bars while scoring")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic Torch algorithms when available",
    )
    parser.add_argument(
        "--output-precision",
        type=int,
        default=6,
        help="Digits to round floating-point results to (default: 6)",
    )

    parser.add_argument("--per-input", dest="per_input", action="store_true", help="Compute per-input scores")
    parser.add_argument(
        "--no-per-input",
        dest="per_input",
        action="store_false",
        help="Disable per-input aggregation",
    )
    parser.set_defaults(per_input=True)

    parser.add_argument(
        "--across-input",
        dest="across_input",
        action="store_true",
        help="Compute across-input scores",
    )
    parser.add_argument(
        "--no-across-input",
        dest="across_input",
        action="store_false",
        help="Disable across-input aggregation",
    )
    parser.set_defaults(across_input=True)

    parser.add_argument("--output", type=Path, default=None, help="Optional output file path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for auto-named outputs (<model>_diversity.<ext>)",
    )
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Serialization format when using --output",
    )
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=None,
        help="Directory of prior diversity runs to aggregate and plot",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Render comparison plots on a logarithmic y-axis",
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _estimate_runtime(args: argparse.Namespace) -> None:
    if not args.verbose:
        return
    per_input_pairs = args.generations_per_input * (args.generations_per_input - 1) // 2
    across_pairs = 0
    if args.across_input:
        across_pairs = args.max_inputs * (args.max_inputs - 1) // 2
    message = (
        f"[info] Estimated per-input pairs: {per_input_pairs}; across-input pairs (max): {across_pairs}."
        " Enable --nli-random-subsample for quicker smoke runs."
    )
    print(message, file=sys.stderr)


def _write_csv(path: Path, result: dict) -> None:
    metrics = result.get("metrics", {})
    rows = []
    for name, payload in metrics.items():
        rows.append(
            {
                "metric": name,
                "per_input_mean": payload.get("per_input_mean"),
                "per_input_std": payload.get("per_input_std"),
                "across_input": payload.get("across_input"),
            }
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys())) if rows else None
        if writer is None:
            writer = csv.DictWriter(handle, fieldnames=["metric", "per_input_mean", "per_input_std", "across_input"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_result(path: Path, fmt: str, result: dict) -> None:
    """Persist ``result`` to ``path`` as JSON or CSV."""
    
    # Log where we're saving
    logger.info(f"Saving diversity results to: {path.absolute()}")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        _write_csv(path, result)


def _write_compare_csv(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for entry in results:
        model_name = entry.get("model_name", "model")
        for metric_name, payload in entry.get("metrics", {}).items():
            rows.append(
                {
                    "model_name": model_name,
                    "metric": metric_name,
                    "per_input_mean": payload.get("per_input_mean"),
                    "per_input_std": payload.get("per_input_std"),
                    "across_input": payload.get("across_input"),
                }
            )

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model_name", "metric", "per_input_mean", "per_input_std", "across_input"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned or "model"


def _load_results(directory: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for file_path in sorted(directory.glob("*.json")):
        try:
            results.append(json.loads(file_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return results


def _print_summary_table(result: Dict[str, Any], precision: int) -> None:
    metrics = result.get("metrics", {})
    if not metrics:
        return

    def _format(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.{precision}f}"
        return str(value)

    header = "Metric   | Per-input μ | σ       | Across-input"
    separator = "---------+-------------+---------+--------------"
    lines = [header, separator]
    for name, payload in metrics.items():
        mean = _format(payload.get("per_input_mean"))
        std = _format(payload.get("per_input_std"))
        across = _format(payload.get("across_input"))
        line = f"{name:<8}| {mean:>11} | {std:>7} | {across:>12}"
        lines.append(line)
    print("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]

    _estimate_runtime(args)
    
    # Resolve SBERT cache directory to absolute path
    sbert_cache_dir = args.sbert_cache_dir
    if sbert_cache_dir is not None:
        sbert_cache_dir = Path(sbert_cache_dir)
        if not sbert_cache_dir.is_absolute():
            sbert_cache_dir = Path.cwd() / sbert_cache_dir
            logger.info(f"Resolved relative SBERT cache dir to absolute: '{sbert_cache_dir}'")

    config = DiversityConfig(
        model_name=args.model_name,
        max_inputs=args.max_inputs,
        generations_per_input=args.generations_per_input,
        ngram_range=args.ead_ngrams,
        sampling_temperature=args.temperature,
        random_seed=args.seed,
        sbert_model_name=args.sbert_model,
        sbert_batch_size=args.sbert_batch_size,
        sbert_cache_dir=sbert_cache_dir,
        nli_model_name=args.nli_model,
        nli_batch_size=args.nli_batch_size,
        nli_pairs_per_input=args.nli_pairs_per_input,
        nli_pairs_across=args.nli_pairs_across,
        nli_random_subsample=args.nli_random_subsample,
        tokenizer_max_length=args.max_length,
        require_exact_counts=not args.allow_partial,
        verbose=args.verbose,
        deterministic=args.deterministic,
        output_precision=args.output_precision,
    )

    result = compute_diversity(
        args.outputs_dir,
        metrics=metrics,
        per_input=args.per_input,
        across_input=args.across_input,
        config=config,
    )

    # Resolve output path to absolute
    output_path = _resolve_output_path(args.output, args.outputs_dir)
    
    if output_path is None and args.output_dir is not None:
        # Auto-generate output filename in output_dir
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
            logger.info(f"Resolved relative output dir to absolute: '{output_dir}'")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_name = result.get("model_name", "model")
        slug = _slugify(model_name)
        extension = "json" if args.format == "json" else "csv"
        output_path = output_dir / f"{slug}_diversity.{extension}"

    if output_path is not None:
        _save_result(output_path, args.format, result)
    else:
        print(json.dumps(result, indent=2))

    _print_summary_table(result, args.output_precision)

    if args.compare_dir:
        compare_dir = Path(args.compare_dir)
        if not compare_dir.is_absolute():
            compare_dir = Path.cwd() / compare_dir
            logger.info(f"Resolved relative compare dir to absolute: '{compare_dir}'")
        compare_dir.mkdir(parents=True, exist_ok=True)
        model_name = result.get("model_name", "model")
        compare_file = compare_dir / f"{_slugify(model_name)}.json"
        logger.info(f"Saving comparison result to: {compare_file.absolute()}")
        compare_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

        aggregated = _load_results(compare_dir)
        if aggregated:
            _write_compare_csv(compare_dir / "diversity_metrics.csv", aggregated)
            metric_names = sorted({name for entry in aggregated for name in entry.get("metrics", {})})
            for metric_name in metric_names:
                plot_diversity_bars(
                    aggregated,
                    metric_name,
                    output_path=compare_dir / f"{metric_name.lower()}_diversity.png",
                    log_scale=args.log_scale,
                )


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    main()
