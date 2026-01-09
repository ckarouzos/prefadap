"""Command line interface for the pseudo-label pipeline."""

from __future__ import annotations

import argparse
import logging
import torch
from pathlib import Path
import sys
from typing import List

from prefadap.pseudo_label import PipelineConfig, PseudoLabelPipeline
from prefadap.runtime.vllm_policy import (
    log_resolved_knobs,
    log_version_info,
    resolve_vllm_knobs,
)
from prefadap.utils.seeding import set_seed
from prefadap.models.tokenization import ensure_pad_token


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pseudo-label pipeline")
    sub = parser.add_subparsers(dest="command")
    # Require a subcommand for clearer errors in all Python versions
    try:
        sub.required = True  # type: ignore[attr-defined]
    except Exception:
        pass

    def add_common(p: argparse.ArgumentParser) -> None:
        # Support both positional and flag-style for robustness in wrappers
        p.add_argument("run_id", nargs="?", help="Unique identifier for this run")
        p.add_argument("dataset", nargs="?", help="Path to input dataset (JSONL)")
        p.add_argument("--run_id", dest="run_id_opt", help="Run id (alternative to positional)")
        p.add_argument(
            "--dataset-path",
            dest="dataset_opt",
            help="Dataset path (alternative to positional)",
        )
        p.add_argument("--k", type=int, default=2)
        p.add_argument("--scoring_model", default="random")
        p.add_argument("--tau", type=float, default=0.8)
        p.add_argument("--max_pairs", type=int, default=1)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--output", required=True)

    add_common(sub.add_parser("generate_candidates"))
    add_common(sub.add_parser("score_candidates"))
    add_common(sub.add_parser("select_labels"))
    add_common(sub.add_parser("run"))
    
    # Add enhanced pipeline option
    enhanced = sub.add_parser("enhanced", help="Use enhanced pseudolabeling techniques")
    enhanced.add_argument("technique", choices=["binary_preference", "generative_preference", "comparative", "counterpart"],
                         help="Pseudolabeling technique to use")
    # Support both positional and flag-style for robustness in wrappers
    enhanced.add_argument("run_id", nargs="?", help="Unique identifier for this run")
    enhanced.add_argument("dataset", nargs="?", help="Path to input dataset (JSONL)")
    enhanced.add_argument("--run_id", dest="run_id_opt", help="Run id (alternative to positional)")
    enhanced.add_argument(
        "--dataset-path",
        dest="dataset_opt",
        help="Dataset path (alternative to positional)",
    )
    enhanced.add_argument("--output", required=True, help="Path to output file")
    
    # Model settings
    enhanced.add_argument("--model_name", default="gemini-2.5-flash", 
                         help="Model name (e.g., gemini-2.5-flash, meta-llama/Llama-3-8B-Instruct)")
    enhanced.add_argument("--model_type", choices=["vllm", "gemini"], default="gemini",
                         help="Model type: vllm for local models, gemini for Google API")
    enhanced.add_argument("--free_tier", action="store_true", default=True,
                         help="Enable free tier rate limits for Gemini")
    
    # Output settings
    enhanced.add_argument("--output_format", choices=["dpo", "kto"], default="dpo",
                         help="Output format: dpo for preference pairs, kto for labeled completions")
    
    # Generation settings
    enhanced.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    enhanced.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    enhanced.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    
    # Technique settings
    enhanced.add_argument("--k", type=int, default=3, help="Number of candidates/attempts per prompt")
    enhanced.add_argument("--tau", type=float, default=0.8, help="Score threshold for filtering")
    enhanced.add_argument("--max_pairs", type=int, default=5, help="Maximum preference pairs per prompt")
    enhanced.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    enhanced.add_argument(
        "--max_examples",
        type=int,
        default=0,
        help="Maximum examples to pseudolabel (0 disables the limit)",
    )
    
    # vLLM-specific settings
    enhanced.add_argument("--tensor-parallel-size", dest="tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM")
    enhanced.add_argument("--dtype", default="auto", help="Data type for vLLM weights")
    enhanced.add_argument("--model-size-class", dest="model_size_class", help="Override detected model size class")
    enhanced.add_argument("--max-model-len", dest="max_model_len", type=int, help="Optional cap for model context length in vLLM")
    enhanced.add_argument("--max-batched-tokens", dest="max_batched_tokens", type=int, help="Max prefill tokens across batch (vLLM)")
    enhanced.add_argument("--max-seqs", dest="max_seqs", type=int, help="Max concurrent sequences (vLLM)")
    enhanced.add_argument("--batch-size", type=int, help="Prompt micro-batch size for vLLM calls")
    enhanced.add_argument("--auto-truncate", action="store_true", help="Truncate prompts that exceed context length")
    enhanced.add_argument("--dataset", dest="dataset_name", help="Dataset name for runtime defaults")
    enhanced.add_argument("--profile", action="store_true", help="Enable profiling for pseudolabeling job")
    enhanced.add_argument("--log-every", type=int, default=50, help="Log throughput every N records")
    enhanced.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable automatic resume when the output file already exists",
    )
    enhanced.set_defaults(resume=True)
    enhanced.add_argument("--shard_count", type=int, default=1, help="Split dataset into this many shards")
    enhanced.add_argument("--shard_index", type=int, default=0, help="Shard index for this job [0..shard_count-1]")
    
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    # Be forgiving to wrappers that may inject empty-string args
    if argv is None:
        argv = sys.argv[1:]
    # Filter out stray empty/whitespace-only arguments to avoid argparse errors
    argv = [a for a in argv if isinstance(a, str) and a.strip() != ""]
    args = parser.parse_args(argv)
    
    if args.command == "enhanced":
        # Use enhanced techniques
        from prefadap.pseudo_label import (
            BinaryPreferencePseudolabeler,
            ComparativePseudolabeler,
            CounterpartPseudolabeler,
            GenerativePreferencePseudolabeler,
            PseudolabelConfig,
        )
        # Normalise dual-style inputs
        run_id = args.run_id if getattr(args, "run_id", None) else getattr(args, "run_id_opt", None)
        dataset = args.dataset if getattr(args, "dataset", None) else getattr(args, "dataset_opt", None)
        if not run_id or not dataset:
            parser.error("enhanced requires run_id and dataset (provide positionally or via --run_id/--dataset-path)")

        log = logging.getLogger(__name__)
        set_seed(args.seed)

        dataset_name = args.dataset_name or Path(dataset).stem
        overrides = {
            "batch_size": args.batch_size,
            "max_model_len": args.max_model_len,
            "max_batched_tokens": args.max_batched_tokens,
            "max_seqs": args.max_seqs,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "model_size_class": args.model_size_class,
        }
        if args.model_type == "vllm":
            knobs = resolve_vllm_knobs(args.model_name, "pseudo", dataset_name, overrides)
            log_resolved_knobs("pseudo", args.seed, knobs)
            log_version_info()
        else:
            knobs = {}

        config = PseudolabelConfig(
            run_id=run_id,
            dataset=Path(dataset),
            output_format=args.output_format,
            seed=args.seed,
            resume=bool(getattr(args, "resume", True)),
            model_name=args.model_name,
            model_type=args.model_type,
            free_tier=args.free_tier,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            technique=args.technique,
            k=args.k,
            tau=args.tau,
            max_pairs=args.max_pairs,
            tensor_parallel_size=int(knobs.get("tensor_parallel_size", args.tensor_parallel_size)),
            gpu_memory_utilization=float(knobs.get("gpu_memory_utilization", 0.8)),
            max_model_len=knobs.get("max_model_len"),
            max_num_batched_tokens=knobs.get("max_batched_tokens"),
            max_num_seqs=knobs.get("max_seqs"),
            batch_size=int(knobs.get("batch_size", args.batch_size or 0) or 0),
            dtype=str(knobs.get("dtype", args.dtype)),
            enable_prefix_caching=bool(knobs.get("enable_prefix_caching", True)),
            block_size=int(knobs.get("block_size", 16)),
            swap_space=float(knobs.get("swap_space", 0.1)),
            auto_truncate=bool(args.auto_truncate),
            max_examples=args.max_examples,
            shard_count=getattr(args, "shard_count", 1),
            shard_index=getattr(args, "shard_index", 0),
            profile=bool(getattr(args, "profile", False)),
            log_interval=int(getattr(args, "log_every", 50)),
        )

        if args.technique == "binary_preference":
            labeler = BinaryPreferencePseudolabeler(config)
        elif args.technique == "generative_preference":
            labeler = GenerativePreferencePseudolabeler(config)
        elif args.technique == "comparative":
            labeler = ComparativePseudolabeler(config)
        elif args.technique == "counterpart":
            labeler = CounterpartPseudolabeler(config)
        else:
            raise ValueError(f"Unknown technique: {args.technique}")

        # Ensure pad_token exists (LLaMA models often lack one)
        if args.model_type == "vllm":
            generator = getattr(labeler, "generator", None)
            tokenizer = None
            if generator is not None:
                ensure_tok = getattr(generator, "_ensure_tokenizer", None)
                if callable(ensure_tok):
                    try:
                        tokenizer = ensure_tok()
                    except Exception:
                        tokenizer = None
                elif hasattr(generator, "llm") and hasattr(generator.llm, "get_tokenizer"):
                    try:
                        tokenizer = generator.llm.get_tokenizer()
                    except Exception:
                        tokenizer = None
            if tokenizer is not None:
                ensure_pad_token(tokenizer)

        attempt = 0
        while True:
            try:
                labeler.run(Path(args.output))
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                msg = str(exc).lower()
                oom_hit = "out of memory" in msg or "hiperroroutofmemory" in msg
                if oom_hit and attempt == 0:
                    config.batch_size = max(1, int(config.batch_size) // 2)
                    knobs["batch_size"] = config.batch_size
                    log.warning("OOM encountered; reducing batch_size to %s", config.batch_size)
                    attempt += 1
                    continue
                if oom_hit and attempt == 1 and config.max_num_batched_tokens:
                    config.max_num_batched_tokens = max(1, int(config.max_num_batched_tokens) // 2)
                    knobs["max_batched_tokens"] = config.max_num_batched_tokens
                    log.warning(
                        "OOM encountered; reducing max_batched_tokens to %s", config.max_num_batched_tokens
                    )
                    attempt += 1
                    continue
                raise
        if attempt:
            log_resolved_knobs("pseudo", args.seed, knobs)
        
    else:
        # Legacy pipeline
        # Normalise dual-style inputs
        run_id = args.run_id if getattr(args, "run_id", None) else getattr(args, "run_id_opt", None)
        dataset = args.dataset if getattr(args, "dataset", None) else getattr(args, "dataset_opt", None)
        if not run_id or not dataset:
            parser.error(f"{args.command} requires run_id and dataset (provide positionally or via --run_id/--dataset-path)")

        cfg = PipelineConfig(
            run_id=run_id,
            dataset=Path(dataset),
            k=args.k,
            scoring_model=args.scoring_model,
            tau=args.tau,
            max_pairs=args.max_pairs,
            seed=args.seed,
        )
        pipeline = PseudoLabelPipeline(cfg)
        output = Path(args.output)
        if args.command == "generate_candidates":
            pipeline.generate_candidates(output)
        elif args.command == "score_candidates":
            pipeline.score_candidates(Path(args.dataset), output)
        elif args.command == "select_labels":
            pipeline.select_labels(Path(args.dataset), output)
        else:
            pipeline.run(output)


if __name__ == "__main__":  # pragma: no cover
    main()
