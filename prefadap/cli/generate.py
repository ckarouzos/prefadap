"""Unified text generation CLI for summarisation datasets.

This module consolidates several legacy generation utilities into a single
entry point.  It supports generating outputs for multiple datasets using a
specified language model, optional subset selection and dataset specific
decoding configurations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple
import time

try:  # Lazy import; allow module import without numpy
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional in some test envs
    np = None  # type: ignore
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prefadap.data import (
    list_supported_datasets,
    load_processed_summarisation_dataset,
    make_raw_text_dataset,
    normalise_dataset_name,
    validate_dataset_name,
)
from prefadap.data.core import resolve_hf_datasets_cache_dir
from prefadap.models.loader import (
    _resolve_model_path,
    get_adapter_base_model,
    is_adapter_directory,
)
from prefadap.utils.hf_auth import prepare_hf_from_pretrained_kwargs
from prefadap.models.tokenization import ensure_pad_token
from prefadap.data.registry import get_dataset_loader
from prefadap.utils.decoding import load_decoding_config
from prefadap.utils.length_bands import enforce_length_band
from prefadap.utils.paths import resolve_run_dir
from prefadap.utils.profiler import RunProfile, consolidate_profile
from prefadap.runtime import generation as runtime_generation
from prefadap.runtime.generation import (
    build_engine as build_vllm_engine,
    build_sampling_params,
    generate_with_vllm,
)
from prefadap.runtime.vllm_policy import (
    resolve_vllm_knobs,
    log_resolved_knobs,
    log_version_info,
)
from prefadap.utils.seeding import set_seed


try:  # pragma: no cover - optional dependency
    from prefadap.pseudo_label.generators import (
        vllm_generator as _vllm_generator,
    )
except Exception:  # pragma: no cover
    VLLMGenerator = None  # type: ignore[assignment]
    _VLLM_WRAPPER_AVAILABLE = False
    _MERGE_LORA_FOR_VLLM = None
else:  # pragma: no cover
    VLLMGenerator = _vllm_generator.VLLMGenerator
    _VLLM_WRAPPER_AVAILABLE = bool(getattr(_vllm_generator, "VLLM_AVAILABLE", False))
    _MERGE_LORA_FOR_VLLM = getattr(_vllm_generator, "_merge_lora_adapter_for_vllm", None)


def _has_external_vllm_container() -> bool:
    """Check if an external vLLM container is available.
    
    On some HPC systems vLLM may not be installed in the local environment but
    is available via a Singularity/Apptainer container.
    
    Returns
    -------
    bool
        True if VLLM_CONTAINER env var points to an existing container file
    """
    container_path = os.environ.get("VLLM_CONTAINER", "").strip()
    if not container_path:
        return False
    
    container_file = Path(container_path)
    return container_file.exists() and container_file.is_file()


_VLLM_AVAILABLE = (
    (runtime_generation.LLM is not None and _VLLM_WRAPPER_AVAILABLE)
    or _has_external_vllm_container()
)

_QA_DATASETS = {"askengineers", "askculinary"}


def _is_qa_dataset(name: str) -> bool:
    return name.lower() in _QA_DATASETS


def _has_summarisation_loader(name: str) -> bool:
    """Return True when ``name`` has a registered summarisation loader."""

    try:
        get_dataset_loader(name, "summarisation")
    except ValueError:
        return False
    return True


def _treat_as_summarisation_for_generation(name: str) -> bool:
    return _has_summarisation_loader(name) and not _is_qa_dataset(name)


def _apply_simple_qa_template(prompt: str) -> str:
    stripped = prompt.strip()
    lower = stripped.lower()
    if lower.startswith("question:") or lower.startswith("q:"):
        return stripped
    return f"Question: {stripped}\nAnswer:"

def disable_tqdm_if_not_tty() -> None:
    """Disable :mod:`tqdm` progress bars when stderr is not a TTY."""

    if not sys.stderr.isatty():
        os.environ["TQDM_DISABLE"] = "1"


def _resolve_generation_output_dir(
    output_dir: Path,
    *,
    run_dir: Path | None,
    seed: int | None,
) -> Path:
    output_dir = output_dir.expanduser()
    if output_dir.is_absolute():
        return output_dir
    if run_dir is None:
        raise ValueError(
            "Relative --output-dir requires RUN_DIR to be set so outputs are anchored "
            "under the persistent run directory."
        )
    resolved_run_dir = resolve_run_dir(run_dir, seed=seed)
    return (resolved_run_dir / output_dir).resolve()


def _count_parameters(model: object) -> int | None:
    params = getattr(model, "parameters", None)
    if not callable(params):
        return None
    try:
        return sum(p.numel() for p in params())
    except Exception:  # pragma: no cover - defensive
        return None


def _log_lora_status(model: object, logger: logging.Logger) -> None:
    peft_cfg = getattr(model, "peft_config", None)
    if not peft_cfg:
        logger.warning("PEFT config not detected on model; LoRA may be inactive.")
        return

    lora_params = 0
    named_params = getattr(model, "named_parameters", None)
    if callable(named_params):
        for name, _param in named_params():
            if "lora" in name.lower():
                lora_params += 1
    logger.info("PEFT config present; detected %d LoRA parameter tensors.", lora_params)
    if lora_params == 0:
        logger.warning("No LoRA parameters detected; check adapter load.")


def _get_decoding_params(cfg: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    """Return decoding parameters for ``dataset`` from ``cfg``."""

    global_cfg = {k: v for k, v in cfg.items() if k not in {"datasets", "seed"}}
    ds_cfg = cfg.get("datasets", {})
    merged = {**global_cfg, **ds_cfg.get(dataset, ds_cfg.get(dataset.lower(), {}))}
    return merged


def _is_fast_tokenizer_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "Tokenizer class" in message
        or "tokenizer_class" in message
        or "TokenizersBackend" in message
    )


def _load_tokenizer_with_fallback(tokenizer_source: str, tokenizer_kwargs: Dict[str, Any]) -> AutoTokenizer:
    log = logging.getLogger(__name__)
    try:
        return AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    except Exception as exc:
        if not _is_fast_tokenizer_error(exc):
            raise
        log.warning(
            "Fast tokenizer failed (%s); retrying with use_fast=False",
            exc,
        )
        slow_kwargs = dict(tokenizer_kwargs)
        slow_kwargs["use_fast"] = False
        return AutoTokenizer.from_pretrained(tokenizer_source, **slow_kwargs)


def _is_qwen3_tokenizer(tokenizer: AutoTokenizer | None) -> bool:
    """Return ``True`` when ``tokenizer`` corresponds to a Qwen3 family model."""

    if tokenizer is None:
        return False

    name = str(getattr(tokenizer, "name_or_path", "") or "")
    if "qwen3" in name.lower():
        return True

    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str) and "enable_thinking" in template:
        return True

    return False


def _apply_qwen3_decoding_defaults(
    decoding_params: Dict[str, Any], tokenizer: AutoTokenizer | None
) -> None:
    """Populate recommended decoding defaults for Qwen3 models when missing."""

    if not _is_qwen3_tokenizer(tokenizer):
        return

    defaults: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "do_sample": True,
    }
    for key, value in defaults.items():
        decoding_params.setdefault(key, value)


def _resolve_stop_token_ids(tokenizer: AutoTokenizer | None) -> list[int]:
    """Return EOS/EOT token IDs when available."""

    if tokenizer is None:
        return []

    stop_token_ids: list[int] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, int):
        stop_token_ids.append(int(eos_id))

    eot_tok = None
    special_map = getattr(tokenizer, "special_tokens_map", {}) or {}
    if isinstance(special_map, dict):
        eot_tok = special_map.get("eot_token")
    if eot_tok:
        eot_id = tokenizer.convert_tokens_to_ids(eot_tok)
        if isinstance(eot_id, int) and eot_id >= 0:
            stop_token_ids.append(int(eot_id))

    return list({int(x) for x in stop_token_ids})


def _prepare_prompts_for_chat_models(
    prompts: List[str],
    tokenizer: AutoTokenizer | None,
    *,
    enable_thinking: bool,
    logger: logging.Logger,
) -> List[str]:
    """Apply chat template formatting for Qwen3 prompts when available."""

    if not _is_qwen3_tokenizer(tokenizer):
        return prompts

    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        return prompts

    formatted: List[str] = []
    warned = False
    for prompt in prompts:
        if "<|im_start|>" in prompt:
            formatted.append(prompt)
            continue
        messages = [{"role": "user", "content": prompt}]
        try:
            templated = apply_template(  # type: ignore[misc]
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            templated = apply_template(  # type: ignore[misc]
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            if not warned:
                logger.warning(
                    "Falling back to raw prompts; failed to apply Qwen3 chat template: %s",
                    exc,
                )
                warned = True
            formatted.append(prompt)
            continue

        if not isinstance(templated, str):
            templated = str(templated)
        formatted.append(templated)

    if formatted and formatted != prompts:
        logger.debug("Applied Qwen3 chat template to %d prompts", len(formatted))

    return formatted


def _select_subset(
    dataset: Iterable[Dict[str, Any]],
    subset_size: int | None,
    random_subset: bool,
) -> Iterable[Dict[str, Any]]:
    """Return a subset of ``dataset`` if ``subset_size`` is provided."""

    if subset_size is None:
        return dataset

    try:  # datasets.Dataset offers ``select``
        length = len(dataset)
        if subset_size >= length:
            return dataset
        if random_subset:
            if np is not None:
                indices = np.random.choice(length, subset_size, replace=False)
            else:  # fallback with Python stdlib
                import random
                indices = random.sample(range(length), subset_size)
        else:
            indices = range(subset_size)
        return dataset.select(list(map(int, indices)))
    except Exception:  # pragma: no cover - dataset may be list-like
        if random_subset:
            if np is not None:
                idxs = list(map(int, np.random.choice(len(dataset), subset_size, replace=False)))
            else:
                import random
                idxs = random.sample(range(len(dataset)), subset_size)
            return [dataset[i] for i in idxs]
        return list(dataset)[:subset_size]


def _load_dataset(
    name: str, subset_size: int | None, random_subset: bool
) -> Tuple[Dict[str, Iterable[Dict[str, Any]]], bool]:
    """Load ``name`` as a mapping of ``split -> dataset``.

    The function first attempts to load a processed dataset from ``data/``.  If
    that fails, the raw text dataset is downloaded via the HuggingFace hub.

    Returns a tuple ``(datasets, processed)`` where ``datasets`` maps split names
    to iterables of records and ``processed`` indicates whether the processed
    loader was used.  Any loaded datasets are optionally subset according to
    ``subset_size`` and ``random_subset``.
    """

    log = logging.getLogger(__name__)
    
    processed_error: Exception | None = None
    if not _is_qa_dataset(name):
        try:
            log.info(f"[dataset-loader] Attempting to load processed dataset: {name}")
            ds = load_processed_summarisation_dataset(name)
            splits = sorted(set(ds["split"]))
            split_ds = {
                split: _select_subset(
                    ds.filter(lambda ex, sp=split: ex["split"] == sp),
                    subset_size,
                    random_subset,
                )
                for split in splits
            }
            log.info(f"[dataset-loader] Successfully loaded processed dataset: {name}")
            return split_ds, True
        except Exception as e:
            processed_error = e
            log.info(f"[dataset-loader] Processed dataset not available: {e}")
    else:
        log.info("[dataset-loader] Skipping processed dataset for QA dataset: %s", name)

    log.info(f"[dataset-loader] Falling back to raw HuggingFace dataset download for: {name}")
    try:
        cache_dir = resolve_hf_datasets_cache_dir(None)
        log.info("[dataset-loader] Using cache_dir: %s", cache_dir)

        args = SimpleNamespace(
            dataset_name=name,
            dataset_structured_subset=None,
            dataset_random_subset=None,
            seed=None,
        )
        ds_dict, _, _ = make_raw_text_dataset(args, cache_dir=cache_dir)
        split_ds = {
            split: _select_subset(dataset, subset_size, random_subset)
            for split, dataset in ds_dict.items()
        }
        log.info(f"[dataset-loader] Successfully loaded raw dataset: {name}")
        return split_ds, False
    except Exception as fallback_error:
        log.error(f"[dataset-loader] FATAL: Both processed and raw dataset loading failed for {name}")
        if processed_error is not None:
            log.error(f"[dataset-loader] Processed dataset error: {processed_error}")
        log.error(f"[dataset-loader] Raw dataset error: {fallback_error}")
        raise RuntimeError(
            f"Failed to load dataset '{name}'. "
            f"Processed dataset not found (data/processed/{name}.jsonl), "
            f"and fallback to HuggingFace also failed. "
            f"Error: {fallback_error}"
        ) from fallback_error


def _batch_chunks(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def generate_for_split(
    model: AutoModelForCausalLM | None,
    tokenizer: AutoTokenizer | None,
    dataset: Iterable[Dict[str, Any]],
    dataset_name: str,
    split: str,
    decoding_params: Dict[str, Any],
    out_path: Path,
    processed: bool,
    *,
    backend: str = "auto",
    batch_size: int = 8,
    concurrency: int | None = None,
    profile: bool = False,
    vllm_engine: "runtime_generation.LLM | None" = None,
    sampling_params: "runtime_generation.SamplingParams | None" = None,
    vllm_helper: "VLLMGenerator | None" = None,
    write_frequency: int = 512,
    max_model_len: int | None = None,
    auto_truncate: bool = False,
    vllm_model_path: str | None = None,
    vllm_knobs: Dict[str, Any] | None = None,
    enable_thinking: bool = False,
) -> None:
    """Generate text for ``dataset`` writing JSONL records to ``out_path``."""
    log = logging.getLogger(__name__)
    if not out_path.is_absolute():
        raise ValueError(f"Output path must be absolute (got {out_path})")
    log.info("Writing generations to %s", out_path)

    # Normalise backend selection
    use_vllm = False
    if backend == "vllm" or (backend == "auto" and _VLLM_AVAILABLE):
        use_vllm = True
    elif backend not in ("auto", "hf", "vllm"):
        raise ValueError(f"Unknown backend: {backend}")

    # Initialise engines
    if use_vllm:
        # vLLM engine must be provided (constructed once per dataset to reuse)
        if vllm_engine is None:
            raise RuntimeError("vLLM backend requested without provided engine (vllm_engine=None)")
    else:
        assert model is not None and tokenizer is not None
        device = model.device
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        total = len(dataset)  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - length may be unknown
        total = None

    prof = RunProfile()
    jsonl_path = out_path.parent / f"{dataset_name}_{split}_gen_profile.jsonl"
    log_every = max(1, int(os.environ.get("PREFADAP_GEN_LOG_INTERVAL", "50")))

    # Materialise dataset to list for batching (datasets.Dataset supports slicing efficiently)
    ds_list: List[Dict[str, Any]] = list(dataset)

    with out_path.open("w", encoding="utf-8") as f:
        # Prepare prompts and references
        prompts: List[str] = []
        references: List[str] = []
        for ex in ds_list:
            if processed:
                text = ex["text"]
                prompt_part, reference = text.split("TL;DR:", 1)
                prompts.append(prompt_part + "TL;DR:")
                references.append(reference.strip())
            else:
                prompts.append(ex["queries"])  # raw dataset field
                references.append(ex.get("outputs", "").strip())

        if _is_qa_dataset(dataset_name):
            prompts = [_apply_simple_qa_template(prompt) for prompt in prompts]

        # Apply Qwen3 chat template formatting conditionally based on dataset type
        if _is_qwen3_tokenizer(tokenizer):
            # Skip chat wrapping for summarization datasets to avoid conversational outputs
            # instead of summaries. Summarization datasets should use raw prompts.
            # This includes all datasets registered under "summarisation" task:
            # - Core: tldr, cnndm, xsum, arxiv
            # - Pseudo-labeled: pseudo_cnndm, pseudo_xsum, pseudo_askculinary, etc.
            # - SHP variants: shp_askengineers, shp_askculinary, shp_*, etc.
            if not _treat_as_summarisation_for_generation(dataset_name):
                # For non-summarisation datasets, apply chat formatting but disable
                # thinking to avoid <think>...</think> blocks
                prompts = _prepare_prompts_for_chat_models(
                    prompts,
                    tokenizer,
                    enable_thinking=False,
                    logger=log,
                )
        else:
            # For non-Qwen3 models, apply chat wrapping with user-specified enable_thinking
            prompts = _prepare_prompts_for_chat_models(
                prompts,
                tokenizer,
                enable_thinking=enable_thinking,
                logger=log,
            )

        max_source_length = decoding_params.get("max_source_length")
        if max_source_length is not None:
            if tokenizer is None:
                raise RuntimeError(
                    "max_source_length requires a tokenizer to be available."
                )
            max_source_length = int(max_source_length)
            truncated_prompts: List[str] = []
            for p in prompts:
                ids = tokenizer(p)["input_ids"]
                before = len(ids)
                if before > max_source_length:
                    ids = ids[:max_source_length]
                    trunc = tokenizer.decode(ids, skip_special_tokens=True)
                    log.info(
                        "Truncated prompt from %d to %d tokens for dataset=%s "
                        "(max_source_length=%d) to control prefill memory.",
                        before,
                        len(ids),
                        dataset_name,
                        max_source_length,
                    )
                    truncated_prompts.append(trunc)
                else:
                    truncated_prompts.append(p)
            prompts = truncated_prompts

        if max_model_len is not None and tokenizer is not None:
            checked: List[str] = []
            # Ensure pad_token exists (LLaMA models often lack one)
            ensure_pad_token(tokenizer)
            for p in prompts:
                ids = tokenizer(p)["input_ids"]
                before = len(ids)
                if before > max_model_len:
                    if auto_truncate:
                        ids = ids[:max_model_len]
                        trunc = tokenizer.decode(ids, skip_special_tokens=True)
                        log.info("input_len_before=%d after=%d", before, len(ids))
                        checked.append(trunc)
                    else:
                        raise RuntimeError(
                            f"Prompt length {before} exceeds max_model_len {max_model_len}."
                            " Use --auto-truncate or increase --max-model-len."
                        )
                else:
                    checked.append(p)
            prompts = checked


        # vLLM fast path
        if use_vllm:
            if vllm_engine is None:
                raise RuntimeError("vLLM backend requested without an initialised engine")
            if sampling_params is None:
                raise RuntimeError("vLLM backend requires pre-built sampling parameters")

            buffer: List[str] = []
            processed_count = 0
            flush_every = max(1, int(write_frequency))
            batch_unit = max(1, int(batch_size))
            resolved_concurrency = max(1, int(concurrency or batch_size))
            local_engine = vllm_engine
            local_knobs = dict(vllm_knobs or {})

            for start_idx in range(0, len(prompts), resolved_concurrency):
                chunk_prompts = prompts[start_idx : start_idx + resolved_concurrency]
                if not chunk_prompts:
                    continue

                attempt = 0
                while True:
                    try:
                        if vllm_helper is not None:
                            raw_outputs = vllm_helper._generate(chunk_prompts, sampling_params)
                            generation_outputs = runtime_generation.GenerationOutputs(
                                [runtime_generation._extract_texts(item) for item in raw_outputs]
                            )
                        else:
                            generation_outputs = generate_with_vllm(
                                local_engine,
                                chunk_prompts,
                                sampling_params,
                                concurrency=min(resolved_concurrency, len(chunk_prompts)),
                                batch_size=max(1, int(batch_size)),
                                logger=log,
                            )
                        break
                    except Exception as exc:
                        msg = str(exc)
                        engine_core_err = "EngineCoreRequestType" in msg or "EngineCore" in msg
                        if not engine_core_err or vllm_model_path is None or attempt >= 1:
                            raise
                        log.error(
                            "vLLM engine error detected (%s). Rebuilding engine with reduced concurrency and retrying once...",
                            msg,
                        )
                        # Degrade concurrency/max_seqs/max_batched_tokens conservatively
                        resolved_concurrency = max(8, resolved_concurrency // 2)
                        if local_knobs:
                            try:
                                local_knobs["max_seqs"] = max(8, int(local_knobs.get("max_seqs", resolved_concurrency)) // 2)
                            except Exception:
                                local_knobs["max_seqs"] = max(8, resolved_concurrency)
                            try:
                                local_knobs["max_batched_tokens"] = max(2048, int(local_knobs.get("max_batched_tokens", 12288)) // 2)
                            except Exception:
                                local_knobs["max_batched_tokens"] = 8192
                            # Lower util slightly for headroom
                            try:
                                util = float(local_knobs.get("gpu_memory_utilization", 0.86))
                            except Exception:
                                util = 0.86
                            local_knobs["gpu_memory_utilization"] = min(0.86, util)
                        from prefadap.runtime.generation import build_engine as _build_vllm_engine

                        local_engine = _build_vllm_engine(str(vllm_model_path), **local_knobs)
                        attempt += 1

                for offset, generations in enumerate(generation_outputs.generations):
                    idx = start_idx + offset
                    primary = generations[0] if generations else ""
                    g, penalty = enforce_length_band(primary, dataset_name)
                    if penalty:
                        log.info("[%s/%s] penalty=%.2f", dataset_name, split, penalty)

                    # Format for diversity evaluation: one record per prompt with all K outputs
                    rec: Dict[str, Any] = {
                        "key": str(idx),
                        "prompt": prompts[idx],
                        "outputs": generations if generations else [g],
                    }
                    # Optional metadata (not used by diversity evaluator but useful for debugging)
                    meta: Dict[str, Any] = {
                        "prediction": g,
                    }
                    if references[idx]:
                        meta["reference"] = references[idx]
                    if penalty:
                        meta["penalty"] = penalty
                    rec["meta"] = meta

                    buffer.append(json.dumps(rec, ensure_ascii=False))
                    token_estimate = len(primary.split())
                    prof.update(token_estimate, 1)
                    processed_count += 1

                    if len(buffer) >= flush_every:
                        f.write("\n".join(buffer) + "\n")
                        buffer.clear()

                    if profile and processed_count % batch_unit == 0:
                        prof.append_jsonl(
                            jsonl_path,
                            {
                                "stage": "generate",
                                "processed": processed_count,
                                "rate": (prof.examples / prof.elapsed)
                                if prof.examples
                                else 0.0,
                                "tokens_per_sec": prof.tokens_per_sec,
                                "batch_size": batch_unit,
                            },
                        )
                        if (processed_count // batch_unit) % log_every == 0:
                            log.info(
                                "gen %s/%s: %d/%s ex, %.1f tok/s",
                                dataset_name,
                                split,
                                processed_count,
                                total or "?",
                                prof.tokens_per_sec,
                            )

            if buffer:
                f.write("\n".join(buffer) + "\n")
        else:
            # HF batched path
            assert tokenizer is not None and model is not None
            device = model.device
            max_new = int(decoding_params.get("max_new_tokens", 128))
            buffer: List[str] = []
            flush_every = max(1, int(write_frequency))
            for bi, chunk in enumerate(_batch_chunks(list(range(len(prompts))), max(1, int(batch_size)))):
                batch_prompts = [prompts[k] for k in chunk]
                t0 = time.perf_counter()
                # Ensure pad_token exists (LLaMA models often lack one)
                ensure_pad_token(tokenizer)
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)
                with torch.no_grad():
                    # Filter out vLLM-specific or preprocessing parameters that HF
                    # model.generate() doesn't support (e.g., 'n' for multi-generation).
                    hf_params = {
                        k: v for k, v in decoding_params.items()
                        if k
                        not in {
                            "max_new_tokens",
                            "max_source_length",
                            "n",
                            "best_of",
                            "logprobs",
                            "prompt_logprobs",
                        }
                    }
                    hf_params.pop("presets", None)
                    eos_token_ids = _resolve_stop_token_ids(tokenizer)
                    if eos_token_ids:
                        hf_params["eos_token_id"] = (
                            eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids
                        )
                    output = model.generate(
                        **inputs,
                        **hf_params,
                        max_new_tokens=max_new,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                dt = time.perf_counter() - t0
                output_ids = getattr(output, "sequences", output)
                if not torch.is_tensor(output_ids):
                    output_ids = torch.as_tensor(output_ids, device=device)
                prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
                predictions = []
                for i, prompt_len in enumerate(prompt_lens):
                    completion_ids = output_ids[i, int(prompt_len) :]
                    predictions.append(
                        tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
                    )
                # Correct token accounting: generated = output_len - input_len (per example)
                try:
                    out_len = int(output_ids.shape[1])
                    gen_tokens = out_len * len(prompt_lens) - int(sum(int(pl) for pl in prompt_lens))
                except Exception:
                    gen_tokens = sum(len(str(pred).split()) for pred in predictions)
                prof.update(gen_tokens, len(chunk))

                for j, pred in enumerate(predictions):
                    g, penalty = enforce_length_band(pred, dataset_name)
                    if penalty:
                        log.info("[%s/%s] penalty=%.2f", dataset_name, split, penalty)
                    idx = chunk[j]
                    # Format for diversity evaluation: one record per prompt with all K outputs
                    rec = {
                        "key": str(idx),
                        "prompt": prompts[idx],
                        "outputs": [g],  # HF path: single generation, but use consistent format
                    }
                    # Optional metadata
                    meta = {
                        "prediction": g,
                    }
                    if references[idx]:
                        meta["reference"] = references[idx]
                    if penalty:
                        meta["penalty"] = penalty
                    rec["meta"] = meta
                    buffer.append(json.dumps(rec, ensure_ascii=False))
                    if len(buffer) >= flush_every:
                        f.write("\n".join(buffer) + "\n")
                        buffer.clear()

                prof.append_jsonl(jsonl_path, {
                    "stage": "generate",
                    "processed": (bi + 1) * max(1, int(batch_size)),
                    "rate": (prof.examples / prof.elapsed) if prof.examples else 0.0,
                    "tokens_per_sec": prof.tokens_per_sec,
                    "batch_size": batch_size,
                })
                if (bi + 1) % log_every == 0:
                    log.info(
                        "gen %s/%s: %d/%s ex, %.1f tok/s",
                        dataset_name,
                        split,
                        (bi + 1) * max(1, int(batch_size)),
                        total or "?",
                        prof.tokens_per_sec,
                    )
            if buffer:
                f.write("\n".join(buffer) + "\n")
    log.info("Wrote %s", out_path)
    if profile:
        prof.finalize()
        prof_path = out_path.parent / f"{dataset_name}_{split}_gen_profile.json"
        consolidate_profile(jsonl_path, prof_path)
        log.info(
            "Profile: tokens/s=%.1f, time/example=%.3fs, gpu_mem=%sMB, gpu_util=%s%%",
            prof.tokens_per_sec,
            prof.time_per_example,
            str(prof.gpu_mem_mb),
            str(prof.gpu_util_percent),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text for datasets")
    parser.add_argument("--model", type=str, required=True, help="Model path or name")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write JSONL prediction files",
    )
    parser.add_argument(
        "--decoding-config",
        type=str,
        default=None,
        help="YAML/JSON file with decoding parameters",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "hf", "vllm"],
        default="auto",
        help="Inference backend: auto prefers vLLM if available; else HF",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Micro-batch size for batched generation (policy default when omitted)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Maximum in-flight async requests (policy default when omitted)",
    )
    parser.add_argument("--max-model-len", type=int, default=None, help="Cap model context length")
    parser.add_argument("--max-batched-tokens", type=int, default=None, help="Cap total prefill tokens")
    parser.add_argument("--max-seqs", type=int, default=None, help="Cap concurrent sequences")
    parser.add_argument("--dtype", type=str, default=None, help="Model dtype override")
    parser.add_argument("--kv-cache-dtype", type=str, default=None, help="KV cache dtype (auto, fp8, etc.)")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="Tensor parallel degree")
    parser.add_argument("--model-size-class", type=str, default=None, help="Override detected size bucket")
    parser.add_argument("--auto-truncate", action="store_true", help="Truncate inputs exceeding max_model_len")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max_new_tokens; else taken from decoding config",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable throughput profiling and save JSON profile",
    )
    parser.add_argument(
        "--write-frequency",
        type=int,
        default=512,
        help="Flush buffered generations to disk every N records",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Allow Qwen3 models to emit <think>…</think> traces. "
            "Disabled by default for comparability with non-thinking models."
        ),
    )
    default_datasets = [
        d
        for d in list_supported_datasets()
        if d != "mixed" and _has_summarisation_loader(d)
    ]
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=default_datasets,
        help="Datasets to generate for",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Limit each split to the first N examples",
    )
    parser.add_argument(
        "--random-subset",
        action="store_true",
        help="Select a random subset when using --subset-size",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    disable_tqdm_if_not_tty()

    run_dir_env = os.environ.get("RUN_DIR")
    run_dir = Path(run_dir_env) if run_dir_env else None
    args.output_dir = _resolve_generation_output_dir(
        args.output_dir,
        run_dir=run_dir,
        seed=args.seed,
    )

    decoding_cfg = load_decoding_config(args.decoding_config)
    if args.seed is not None:
        set_seed(args.seed)

    canonical_datasets: list[str] = []
    for d in args.datasets:
        canonical = normalise_dataset_name(d)
        validate_dataset_name(canonical)
        canonical_datasets.append(canonical)
    args.datasets = canonical_datasets
    
    # Initialize logger early for use in backend detection
    log = logging.getLogger(__name__)
    
    # CRITICAL: Check if any dataset requires multi-generation (n>1) before loading models
    # Multi-generation requires vLLM backend and must be detected early
    requires_multi_generation = False
    if decoding_cfg:
        for dataset_name in args.datasets:
            dataset_params = _get_decoding_params(decoding_cfg, dataset_name)
            n_param = dataset_params.get("n")
            if n_param is not None and n_param > 1:
                requires_multi_generation = True
                log.info(
                    "Dataset '%s' decoding config specifies n=%d (multi-generation); vLLM backend required",
                    dataset_name,
                    n_param
                )
                break
    
    # Enforce backend selection based on multi-generation requirements
    backend = args.backend
    if requires_multi_generation:
        if backend == "hf":
            raise RuntimeError(
                "Decoding config specifies n>1 for multi-generation, "
                "but --backend is set to 'hf'. HuggingFace models do not support the 'n' parameter. "
                "Use --backend vllm or --backend auto to enable vLLM for diversity generation."
            )
        if not _VLLM_AVAILABLE:
            raise RuntimeError(
                "Decoding config specifies n>1 for multi-generation, "
                "but vLLM is not available. Install vLLM to run diversity generation."
            )
        # Force vLLM backend
        if backend == "auto":
            log.info("Forcing vLLM backend for multi-generation (n>1)")
        backend = "vllm"

    resolved_model = _resolve_model_path(args.model)
    model_path = Path(resolved_model)
    adapter_dir = model_path if model_path.is_dir() and is_adapter_directory(model_path) else None
    tokenizer_source = resolved_model
    vllm_base_model = resolved_model

    if adapter_dir is not None:
        base_model = get_adapter_base_model(adapter_dir)
        if not base_model:
            raise ValueError(
                f"Adapter directory '{adapter_dir}' missing base_model_name_or_path;"
                " cannot run generation pipeline."
            )
        vllm_base_model = _resolve_model_path(base_model)
        tokenizer_source = vllm_base_model

    model: AutoModelForCausalLM | None = None
    tokenizer: AutoTokenizer | None = None
    backend = args.backend

    # Backend resolution rules:
    # - If user asked for vLLM but it's unavailable: raise (no silent fallback)
    # - If user asked for HF: load HF
    # - If auto: prefer vLLM when available; otherwise HF
    if backend == "vllm":
        if not _VLLM_AVAILABLE:
            # Provide helpful error message with container path info
            container_path = os.environ.get("VLLM_CONTAINER", "<not set>")
            container_exists = Path(container_path).exists() if container_path != "<not set>" else False
            error_msg = (
                "Requested backend 'vllm' but vLLM is not installed/available. "
                f"VLLM_CONTAINER={container_path}, exists={container_exists}. "
            )
            if not container_exists and container_path != "<not set>":
                error_msg += (
                    f"Container file not found at {container_path}. "
                    "Please verify the container path or install vLLM locally."
                )
            elif container_path == "<not set>":
                error_msg += (
                    "Install vLLM locally or set VLLM_CONTAINER environment variable "
                    "to point to a vLLM Singularity/Apptainer container."
                )
            raise RuntimeError(error_msg)
        tokenizer_kwargs = prepare_hf_from_pretrained_kwargs(
            str(tokenizer_source), validate_access=False
        )
        tokenizer = _load_tokenizer_with_fallback(str(tokenizer_source), tokenizer_kwargs)
        # vLLM engine constructed in generate_for_split
    elif backend == "hf":
        # Import centralized sharding helpers
        from prefadap.training.utils import apply_default_sharding, safe_move_model
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if adapter_dir is not None:
            try:  # pragma: no cover - optional dependency
                from peft import AutoPeftModelForCausalLM  # type: ignore
            except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
                raise ImportError("Install 'peft' to load adapter models") from exc
            model_kwargs = prepare_hf_from_pretrained_kwargs(
                str(model_path), validate_access=False
            )
            # Apply default sharding policy (only for large models on multi-GPU)
            apply_default_sharding(model_kwargs, str(model_path))
            
            model = AutoPeftModelForCausalLM.from_pretrained(
                str(model_path), **model_kwargs
            )
            loaded_params = _count_parameters(model)
            log.info(
                "Loaded adapter model from %s (params=%s, peft=%s)",
                model_path,
                loaded_params if loaded_params is not None else "unknown",
                bool(getattr(model, "peft_config", None)),
            )
            _log_lora_status(model, log)
            # Safe move to device (respects device_map if present)
            if "device_map" not in model_kwargs:
                safe_move_model(model, str(device), "model")
            
            tokenizer_kwargs = prepare_hf_from_pretrained_kwargs(
                str(tokenizer_source), validate_access=False
            )
            tokenizer = _load_tokenizer_with_fallback(
                str(tokenizer_source), tokenizer_kwargs
            )
        else:
            model_kwargs = prepare_hf_from_pretrained_kwargs(
                str(model_path), validate_access=False
            )
            # Apply default sharding policy (only for large models on multi-GPU)
            apply_default_sharding(model_kwargs, str(model_path))
            
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path), **model_kwargs
            )
            # Safe move to device (respects device_map if present)
            if "device_map" not in model_kwargs:
                safe_move_model(model, str(device), "model")
            
            tokenizer_kwargs = prepare_hf_from_pretrained_kwargs(
                str(model_path), validate_access=False
            )
            tokenizer = _load_tokenizer_with_fallback(str(model_path), tokenizer_kwargs)
        # HF decoder-only generation is faster with left padding (avoids wasted attention).
        tokenizer.padding_side = "left"
        model.eval()
    elif backend == "auto":
        if _VLLM_AVAILABLE:
            backend = "vllm"
            tokenizer_kwargs = prepare_hf_from_pretrained_kwargs(
                str(tokenizer_source), validate_access=False
            )
            tokenizer = _load_tokenizer_with_fallback(
                str(tokenizer_source), tokenizer_kwargs
            )
        else:
            # Import centralized sharding helpers
            from prefadap.training.utils import apply_default_sharding, safe_move_model
            
            backend = "hf"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_kwargs = prepare_hf_from_pretrained_kwargs(
                str(model_path), validate_access=False
            )
            # Apply default sharding policy (only for large models on multi-GPU)
            apply_default_sharding(model_kwargs, str(model_path))
            
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path), **model_kwargs
            )
            # Safe move to device (respects device_map if present)
            if "device_map" not in model_kwargs:
                safe_move_model(model, str(device), "model")
            
            tokenizer_kwargs = prepare_hf_from_pretrained_kwargs(
                str(model_path), validate_access=False
            )
            tokenizer = _load_tokenizer_with_fallback(str(model_path), tokenizer_kwargs)
            # HF decoder-only generation is faster with left padding (avoids wasted attention).
            tokenizer.padding_side = "left"
            model.eval()
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if (
        backend == "vllm"
        and adapter_dir is not None
        and model_path.name.startswith("checkpoint-")
        and VLLMGenerator is None
    ):
        if _MERGE_LORA_FOR_VLLM is None:
            raise RuntimeError(
                "vLLM backend requires merging LoRA adapters, but the merge helper "
                "is unavailable. Install optional dependencies for adapter merging."
            )
        log.info(
            "Merging LoRA adapter checkpoint %s into base model %s for vLLM",
            model_path,
            vllm_base_model,
        )
        merged_dir = _MERGE_LORA_FOR_VLLM(str(vllm_base_model), model_path)
        log.info("LoRA adapter merge completed; using merged model at %s", merged_dir)
        model_path = merged_dir
        adapter_dir = None

    for name in args.datasets:
        split_datasets, processed = _load_dataset(name, args.subset_size, args.random_subset)
        decoding_params = _get_decoding_params(decoding_cfg, name)
        _apply_qwen3_decoding_defaults(decoding_params, tokenizer)
        if name.lower() in {"askengineers", "askculinary"}:
            decoding_params["min_new_tokens"] = 0
            log.info("Setting min_new_tokens=0 for %s to allow EOS-based stopping", name)
        if args.max_new_tokens is not None:
            decoding_params["max_new_tokens"] = int(args.max_new_tokens)

        # vLLM policy resolution is only relevant for the vLLM backend.
        knobs = None
        if backend == "vllm":
            overrides = {
                "batch_size": args.batch_size,
                "concurrency": args.concurrency,
                "max_model_len": args.max_model_len,
                "max_new_tokens": decoding_params.get("max_new_tokens"),
                "max_batched_tokens": args.max_batched_tokens,
                "max_seqs": args.max_seqs,
                "dtype": args.dtype,
                "kv_cache_dtype": args.kv_cache_dtype,
                "tensor_parallel_size": args.tensor_parallel_size,
                "model_size_class": args.model_size_class,
            }
            knobs = resolve_vllm_knobs(str(vllm_base_model), "gen", name, overrides)
            log_resolved_knobs("gen", args.seed, knobs)
            log_version_info()
            decoding_params["max_new_tokens"] = int(knobs.get("max_new_tokens", decoding_params.get("max_new_tokens", 128)))
            # Propagate stop sequences from policy to sampling params
            stop_sequences = knobs.get("stop_sequences")
            if stop_sequences:
                decoding_params["stop"] = list(stop_sequences)  # strings

            batch_size = int(knobs["batch_size"])
            concurrency = int(knobs.get("concurrency", batch_size))
            max_model_len_resolved = int(knobs.get("max_model_len", 0)) or None
            knobs["batch_size"] = batch_size
            knobs["concurrency"] = concurrency
            if max_model_len_resolved is not None:
                knobs["max_model_len"] = max_model_len_resolved
        else:
            explicit_batch = args.batch_size
            if explicit_batch is None:
                explicit_batch = decoding_params.get("batch_size")
            if explicit_batch is None:
                # HF backend defaults: cap batch size to avoid Python-side overhead.
                batch_size = 64
            else:
                batch_size = int(explicit_batch)
            concurrency = int(args.concurrency or batch_size)
            max_model_len_resolved = int(args.max_model_len) if args.max_model_len is not None else None

        sampling_params = None
        vllm_engine = None
        vllm_helper = None
        if backend == "vllm" and _VLLM_AVAILABLE:
            # Log multi-generation configuration if n>1 (diversity sampling)
            n_param = decoding_params.get("n")
            if n_param is not None and n_param > 1:
                log.info(
                    "Using vLLM backend for diversity generation: n=%d generations per prompt",
                    n_param
                )
            
            # Enforce EOS stop via token IDs when tokenizer is available
            stop_token_ids = _resolve_stop_token_ids(tokenizer)
            if stop_token_ids:
                existing = decoding_params.get("stop_token_ids") or []
                try:
                    merged = list({int(x) for x in (list(existing) + stop_token_ids)})
                except Exception:
                    merged = stop_token_ids
                decoding_params["stop_token_ids"] = merged

            sampling_params = build_sampling_params(dict(decoding_params), seed=args.seed)
            if VLLMGenerator is not None:
                vllm_helper = VLLMGenerator(
                    model_name=str(model_path),
                    max_tokens=int(decoding_params["max_new_tokens"]),
                    max_model_len=max_model_len_resolved,
                    temperature=float(decoding_params.get("temperature", 0.7)),
                    top_p=float(decoding_params.get("top_p", 0.9)),
                    tensor_parallel_size=int(knobs.get("tensor_parallel_size", 1)),
                    gpu_memory_utilization=float(knobs.get("gpu_memory_utilization", 0.8)),
                    dtype=str(knobs.get("dtype", "auto")),
                    max_num_batched_tokens=int(knobs.get("max_batched_tokens", batch_size)),
                    max_num_seqs=int(knobs.get("max_seqs", concurrency)),
                    enable_prefix_caching=bool(knobs.get("enable_prefix_caching", True)),
                    block_size=int(knobs.get("block_size", 16)),
                    swap_space=float(knobs.get("swap_space", 0.1)),
                )
                vllm_engine = vllm_helper.llm
                if adapter_dir is not None:
                    lora_kwargs = getattr(vllm_helper, "_lora_generate_kwargs", {})
                    if not lora_kwargs:
                        log.warning(
                            "LoRA adapter detected but no vLLM adapter routing kwargs were inferred."
                        )
            else:
                engine_kwargs = {k: v for k, v in knobs.items()}
                vllm_engine = build_vllm_engine(str(model_path), **engine_kwargs)

        for split, split_ds in split_datasets.items():
            split_str = str(split).lower()
            # Skip train and validation splits - only generate on test splits
            if split_str.startswith("train") or split_str.startswith("val") or split_str == "validation":
                continue
            out_file = args.output_dir / f"{name}_{split}.jsonl"
            log.info("Generating for %s split %s", name, split)
            generate_for_split(
                model if backend == "hf" else model_path,  # type: ignore[arg-type]
                tokenizer,
                split_ds,
                name,
                split,
                decoding_params,
                out_file,
                processed,
                backend=backend,
                batch_size=batch_size,
                concurrency=concurrency,
                profile=bool(args.profile),
                vllm_engine=vllm_engine,
                sampling_params=sampling_params,
                vllm_helper=vllm_helper,
                write_frequency=int(args.write_frequency),
                max_model_len=max_model_len_resolved,
                auto_truncate=bool(args.auto_truncate),
                vllm_model_path=str(model_path) if backend == "vllm" else None,
                vllm_knobs=knobs if backend == "vllm" else None,
                enable_thinking=bool(args.enable_thinking),
            )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
