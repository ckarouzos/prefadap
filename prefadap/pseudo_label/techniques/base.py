"""Base classes for pseudolabeling techniques."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Tuple

from transformers import AutoTokenizer  # type: ignore

from prefadap.utils.seeding import set_seed
from prefadap.utils.hf_auth import prepare_hf_from_pretrained_kwargs


@dataclass
class PseudolabelConfig:
    """Configuration for pseudolabeling techniques."""

    # Core settings
    run_id: str
    dataset: Path
    output_format: Literal["dpo", "kto"] = "dpo"
    seed: int = 42
    resume: bool = True
    
    # Model settings  
    model_name: str = "gemini-2.5-flash"
    model_type: Literal["vllm", "gemini"] = "gemini"
    free_tier: bool = True
    
    # Generation settings
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Technique-specific settings
    technique: str = "binary_preference" 
    k: int = 3  # Number of candidates to generate
    tau: float = 0.8  # Score threshold
    max_pairs: int = 1000  # Maximum preference pairs per prompt
    # Limit number of examples to process (0 means no limit)
    max_examples: int = 0
    
    # vLLM-specific settings
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.8
    # Optional cap for vLLM context length to reduce KV cache usage
    max_model_len: int | None = None
    # vLLM batching knobs
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    # Prompt micro-batch size for generation/judging
    batch_size: int = 64
    dtype: str = "auto"
    enable_prefix_caching: bool = True
    block_size: int = 16
    swap_space: float = 0.1
    auto_truncate: bool = False
    # Profiling & logging
    profile: bool = False
    log_interval: int = 50
    # Optional dataset sharding for multi-job pseudolabeling
    shard_count: int = 1
    shard_index: int = 0


class BasePseudolabeler(ABC):
    """Abstract base class for all pseudolabeling techniques."""
    
    def __init__(self, config: PseudolabelConfig):
        self.config = config
        set_seed(config.seed)
        self._setup_model()
    
    @abstractmethod
    def _setup_model(self) -> None:
        """Initialize the model/judge for this technique."""
        pass
    
    @abstractmethod
    def generate_preferences(self, prompts: List[str], references: List[str]) -> Iterator[Dict[str, Any]]:
        """Generate preference data for the given prompts and references.
        
        Parameters
        ----------
        prompts : List[str]
            List of input prompts (e.g., articles to summarize)
        references : List[str] 
            List of reference outputs (e.g., original summaries)
            
        Yields
        ------
        Dict[str, Any]
            Preference records in the specified output format
        """
        pass
    
    def run(self, output_path: Path) -> None:
        """Run the complete pseudolabeling pipeline.

        This method now degrades gracefully when Weights & Biases (``wandb``)
        is unavailable or shadowed by a local ``wandb/`` directory. If W&B is
        disabled via ``WANDB_DISABLED=1`` or the import lacks ``init``, logging
        becomes a no-op and the pipeline proceeds normally.
        """
        from prefadap.utils.profiler import RunProfile

        # Load dataset
        log = logging.getLogger(__name__)

        prompts, references = self._load_dataset()

        total = len(prompts)

        # Determine how many have already been processed
        record_offset = 0
        prompt_offset = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resume_state_path = output_path.with_suffix(output_path.suffix + ".resume.json")
        if not output_path.exists() and resume_state_path.exists():
            try:
                resume_state_path.unlink()
            except OSError:
                pass
        if output_path.exists():
            if not getattr(self.config, "resume", True):
                raise FileExistsError(
                    f"Output file {output_path} exists and resume=False; refusing to overwrite"
                )
            record_offset, prompt_offset = self._infer_resume_offsets(output_path, prompts)
            prompt_offset = min(prompt_offset, total)
            if record_offset or prompt_offset:
                log.info(
                    "Found %d existing examples, resuming from index %d of %d",
                    record_offset,
                    prompt_offset,
                    total,
                )
            if prompt_offset >= total:
                log.info(
                    "Output already contains %d examples; nothing to do.",
                    record_offset,
                )
                return

        # Initialise W&B if available and not explicitly disabled. Prefer the
        # real package; keep a graceful fallback but make failures obvious.
        import importlib

        try:
            wandb = importlib.import_module("wandb")  # type: ignore
            _have_real_wandb = hasattr(wandb, "init")
        except Exception:
            wandb = None  # type: ignore
            _have_real_wandb = False

        use_wandb = _have_real_wandb and os.getenv("WANDB_DISABLED", "0") != "1"

        if not use_wandb:
            print("[WARN] W&B disabled or shadowed. Using no-op logger. Set WANDB_DIR and avoid CWD=repo_root.", flush=True)

            class _NoopWandb:  # type: ignore
                def init(self, *a, **k):
                    class _Run:
                        def log(self, *a, **k):
                            pass

                        def finish(self):
                            pass

                    return _Run()

            wandb = _NoopWandb()  # type: ignore

        run = wandb.init(project="pseudolabeling", name=self.config.run_id, resume="allow")  # type: ignore

        # Profiling & logging setup
        prof = RunProfile()
        log_every_env = os.environ.get("PREFADAP_PSEUDO_LOG_INTERVAL")
        log_every = int(log_every_env) if log_every_env else int(self.config.log_interval or 50)
        jsonl_path = output_path.parent / f"{self.config.dataset.stem}_pseudo_profile.jsonl"
        flush_every = max(1, min(int(log_every), 100))

        def write_resume_state(records: int, prompts_done: int) -> None:
            state = {"record_offset": max(0, int(records)), "prompt_offset": max(0, int(prompts_done))}
            try:
                with resume_state_path.open("w", encoding="utf-8") as state_file:
                    json.dump(state, state_file)
                    state_file.flush()
                    os.fsync(state_file.fileno())
            except OSError:
                pass

        write_resume_state(record_offset, prompt_offset)

        with output_path.open("a", encoding="utf-8") as f:
            t0 = time.perf_counter()
            proc = 0
            token_estimate = 0
            last_logged_docs = 0
            last_logged_tokens = 0
            flush_every = max(1, min(int(log_every), 100))
            prompts_done = prompt_offset
            last_prompt_seen: str | None = None
            for i, record in enumerate(
                self.generate_preferences(
                    prompts[prompt_offset:], references[prompt_offset:]
                )
            ):
                f.write(json.dumps(record) + "\n")
                f.flush()
                proc += 1
                if proc % flush_every == 0:
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass

                # Estimate tokens based on saved text length
                prompt_in_record = None
                if isinstance(record, dict):
                    prompt_in_record = record.get("prompt")
                    if "chosen" in record and "rejected" in record:  # DPO-style
                        token_estimate += len(str(record.get("chosen", "")).split())
                        token_estimate += len(str(record.get("rejected", "")).split())
                    elif "completion" in record:  # KTO-style
                        token_estimate += len(str(record.get("completion", "")).split())
                if isinstance(prompt_in_record, str) and prompt_in_record != last_prompt_seen:
                    prompts_done = min(total, prompts_done + 1)
                    last_prompt_seen = prompt_in_record

                # Live logging
                # Append JSONL snapshot every record (ensures survivability);
                # throttle console logging every N
                prof.update(token_estimate - last_logged_tokens, proc - last_logged_docs)
                last_logged_tokens = token_estimate
                last_logged_docs = proc
                prof.append_jsonl(jsonl_path, {
                    "stage": "pseudolabel",
                    "processed": proc,
                    "rate": (prof.examples / prof.elapsed) if prof.examples else 0.0,
                    "tokens_per_sec": prof.tokens_per_sec,
                    "batch_size": int(self.config.batch_size),
                })
                if proc % max(1, log_every) == 0:
                    dt = max(1e-6, time.perf_counter() - t0)
                    docs_per_s = proc / dt
                    toks_per_s = token_estimate / dt if token_estimate else 0.0
                    processed_total = record_offset + proc
                    log.info(
                        "pseudo: %d/%d rec, %.1f docs/s, %.1f tok/s",
                        processed_total,
                        total,
                        docs_per_s,
                        toks_per_s,
                    )
                    run.log(
                        {
                            "processed": processed_total,
                            "total": total,
                            "docs_per_s": docs_per_s,
                            "tokens_per_s": toks_per_s,
                        }
                    )

            # Finalize profiler

            prof.update(tokens=token_estimate, n_examples=proc)
            prof.finalize()
            if getattr(self.config, "profile", False):
                from prefadap.utils.profiler import consolidate_profile

                prof_path = output_path.parent / f"{self.config.dataset.stem}_pseudo_profile.json"
                consolidate_profile(jsonl_path, prof_path)
                log.info(
                    "pseudo profile: docs=%d, tok=%d, t=%.2fs, tok/s=%.1f, mem=%sMB util=%s%%",
                    proc,
                    token_estimate,
                    prof.elapsed,
                    prof.tokens_per_sec,
                    str(prof.gpu_mem_mb),
                    str(prof.gpu_util_percent),
                )
        write_resume_state(record_offset + proc, min(prompts_done, total))
        run.finish()
    
    def _load_dataset(self) -> tuple[List[str], List[str]]:
        """Load prompts and references from the dataset."""
        prompts = []
        references = []

        max_examples = int(self.config.max_examples or 0)
        shard_count = max(1, int(getattr(self.config, "shard_count", 1)))
        shard_index = max(0, int(getattr(self.config, "shard_index", 0)))
        taken = 0
        with self.config.dataset.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_examples > 0 and taken >= max_examples:
                    break
                if shard_count > 1 and (i % shard_count) != shard_index:
                    continue
                import json
                record = json.loads(line)
                prompts.append(record["prompt"])

                # Handle different reference field names
                if "chosen" in record:
                    references.append(record["chosen"])
                elif "completion" in record:
                    references.append(record["completion"])
                elif "response" in record:
                    references.append(record["response"])
                else:
                    references.append(record.get("reference", ""))
                taken += 1
        if self.config.max_model_len:
            tok_kwargs = prepare_hf_from_pretrained_kwargs(
                self.config.model_name, use_fast=False, validate_access=False
            )
            tok = AutoTokenizer.from_pretrained(self.config.model_name, **tok_kwargs)
            checked: List[str] = []
            for p in prompts:
                ids = tok(p)["input_ids"]
                before = len(ids)
                if before > int(self.config.max_model_len):
                    if self.config.auto_truncate:
                        ids = ids[: int(self.config.max_model_len)]
                        checked.append(tok.decode(ids))
                        logging.getLogger(__name__).info(
                            "Truncated prompt tokens %d -> %d", before, len(ids)
                        )
                    else:
                        raise RuntimeError(
                            f"Prompt length {before} exceeds max_model_len {self.config.max_model_len}; use --auto-truncate or --max-model-len"
                        )
                else:
                    checked.append(p)
            prompts = checked

        return prompts, references

    def _write_output(self, preferences: List[Dict[str, Any]], output_path: Path) -> None:
        """Write preferences to output file."""
        import json
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w", encoding="utf-8") as f:
            for record in preferences:
                f.write(json.dumps(record) + "\n")

    def _infer_resume_offsets(
        self, output_path: Path, prompts: List[str]
    ) -> Tuple[int, int]:
        """Infer record and prompt offsets from an existing output file."""

        record_count = 0
        prompt_idx = 0
        last_prompt: str | None = None


        if not prompts:
            return 0, 0

        prompt_positions: Dict[str, deque[int]] = defaultdict(deque)
        for idx, prompt in enumerate(prompts):
            prompt_positions[prompt].append(idx)

        try:
            with output_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    record_count += 1
                    prompt = rec.get("prompt")
                    if not isinstance(prompt, str):
                        continue
                    if prompt == last_prompt:
                        continue
                    positions = prompt_positions.get(prompt)
                    if not positions:
                        continue
                    while positions and positions[0] < prompt_idx:
                        positions.popleft()
                    if not positions:
                        continue
                    prompt_idx = positions.popleft() + 1
                    last_prompt = prompt
        except OSError:
            return 0, 0

        return record_count, prompt_idx



__all__ = ["PseudolabelConfig", "BasePseudolabeler"]
