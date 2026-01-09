"""vLLM-based text generator for pseudolabeling."""

from __future__ import annotations

import copy
import gc
import inspect
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional dependency: vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from prefadap.models.loader import (
    _resolve_model_path,
    get_adapter_base_model,
    is_adapter_directory,
)


log = logging.getLogger(__name__)


def _latest_mtime(path: Path) -> float:
    """Return the most recent modification time for files under ``path``."""

    mtimes = [entry.stat().st_mtime for entry in path.rglob("*") if entry.is_file()]
    return max(mtimes) if mtimes else path.stat().st_mtime


def _has_complete_hf_artifacts(path: Path) -> bool:
    """Check whether ``path`` already contains a full Hugging Face model dump."""

    required_files = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )

    if not any(path.glob("*.safetensors")):
        return False

    return all((path / name).exists() for name in required_files)


def _missing_hf_artifacts(path: Path) -> List[str]:
    """Return the list of missing Hugging Face artifacts under ``path``."""

    required_files = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    missing = [name for name in required_files if not (path / name).exists()]
    if not any(path.glob("*.safetensors")):
        missing.append("*.safetensors")
    return missing


def _ensure_hf_artifacts(
    merged_model,
    merge_dir: Path,
    *,
    tokenizer_loader,
    tokenizer_source: str,
    config_source: str,
):
    """Persist merged weights and tokenizer files for vLLM consumption."""

    config_path = merge_dir / "config.json"
    if not hasattr(merged_model, "save_pretrained"):
        raise RuntimeError(
            "Merged model does not expose 'save_pretrained'; cannot materialise weights for vLLM."
        )

    if not any(merge_dir.glob("*.safetensors")) or not config_path.exists():
        merged_model.save_pretrained(merge_dir, safe_serialization=True)

    # Redundant config_path.exists() check and block removed.

    generation_config_path = merge_dir / "generation_config.json"
    if not generation_config_path.exists():
        generation_config = getattr(merged_model, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "save_pretrained"):
            generation_config.save_pretrained(merge_dir)

    tokenizer_files = (
        merge_dir / "tokenizer.json",
        merge_dir / "tokenizer_config.json",
        merge_dir / "special_tokens_map.json",
    )
    if not all(path.exists() for path in tokenizer_files):
        tokenizer = tokenizer_loader(tokenizer_source)
        tokenizer.save_pretrained(merge_dir)


def _invoke_peft_loader(
    loader,
    base_model: str,
    adapter_dir: Path,
    load_kwargs: Dict[str, object],
) -> object:
    """Call ``loader`` with best-effort compatibility across PEFT versions."""

    adapter_path = str(adapter_dir)
    call_kwargs = dict(load_kwargs)

    try:
        signature = inspect.signature(loader)
    except (TypeError, ValueError):  # pragma: no cover - builtins / cythonised funcs
        signature = None

    if signature is not None:
        params = signature.parameters
        if "base_model_name_or_path" in params:
            call_kwargs["base_model_name_or_path"] = base_model
            return loader(adapter_path, **call_kwargs)
        if "base_model" in params:
            call_kwargs["base_model"] = base_model
            return loader(adapter_path, **call_kwargs)
        if "adapter_name_or_path" not in params:
            # No obvious slot for adapter path -> assume adapter first
            return loader(adapter_path, **call_kwargs)

    # Fall back to legacy signature where base model is positional and adapter
    # path is provided via ``adapter_name_or_path``.
    legacy_kwargs = dict(load_kwargs)
    legacy_kwargs["adapter_name_or_path"] = adapter_path
    return loader(base_model, **legacy_kwargs)


def _count_parameters(model: object) -> Optional[int]:
    """Best-effort parameter count for logging adapter merges."""

    params = getattr(model, "parameters", None)
    if not callable(params):
        return None
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception:  # pragma: no cover - defensive for unexpected model wrappers
        return None


def resolve_peft_base(base_dir: Path) -> str:
    """Return the underlying Hugging Face base model for ``base_dir``.

    Parameters
    ----------
    base_dir:
        Directory that is expected to contain a PEFT adapter.

    Returns
    -------
    str
        The base model identifier or path referenced by the adapter.

    Raises
    ------
    ValueError
        If the directory does not describe a valid adapter with a recorded
        ``base_model_name_or_path``.
    """

    adapter_config = base_dir / "adapter_config.json"
    if not adapter_config.exists():
        raise ValueError(f"Directory '{base_dir}' is not a PEFT adapter (missing adapter_config.json)")

    try:
        config_data = json.loads(adapter_config.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse adapter configuration at '{adapter_config}'") from exc

    for key in ("base_model_name_or_path", "base_model"):
        base = config_data.get(key)
        if isinstance(base, str) and base.strip():
            return base.strip()

    # Fall back to the broader helper which can consult the resolved config
    # emitted during training.
    fallback = get_adapter_base_model(base_dir)
    if fallback:
        return fallback

    raise ValueError(
        f"Adapter directory '{base_dir}' is missing 'base_model_name_or_path' information"
    )


def _load_chained_peft_model(
    base_model_name: str,
    stage_one_adapter: Path,
    stage_two_adapter: Path,
    *,
    load_kwargs: Dict[str, object],
    base_model_loader,
    peft_model_class,
):
    """Load ``stage_one_adapter`` and ``stage_two_adapter`` on ``base_model_name``."""

    base_model = base_model_loader(base_model_name, **load_kwargs)
    base_params = _count_parameters(base_model)
    model = peft_model_class.from_pretrained(
        base_model,
        str(stage_one_adapter),
        adapter_name="stage1",
        is_trainable=False,
    )
    stage_one_params = _count_parameters(model)
    log.info(
        "Loaded stage1 adapter %s onto base %s (params: %s -> %s, peft=%s)",
        stage_one_adapter,
        base_model_name,
        base_params if base_params is not None else "unknown",
        stage_one_params if stage_one_params is not None else "unknown",
        bool(getattr(model, "peft_config", None)),
    )
    if stage_one_params is not None and base_params == stage_one_params:
        log.warning(
            "Stage1 adapter load did not change parameter count for %s (base %s).",
            stage_one_adapter,
            base_model_name,
        )
    if hasattr(model, "set_adapter"):
        try:
            model.set_adapter("stage1")
        except (KeyError, ValueError, RuntimeError) as exc:  # pragma: no cover - defensive
            log.debug("Unable to activate stage1 adapter before merge: %s", exc)

    if hasattr(model, "merge_and_unload"):
        merged = model.merge_and_unload()
        if merged is not None:
            model = merged

    model = peft_model_class.from_pretrained(
        model,
        str(stage_two_adapter),
        adapter_name="stage2",
        is_trainable=False,
    )
    stage_two_params = _count_parameters(model)
    log.info(
        "Loaded stage2 adapter %s onto model %s (params: %s -> %s, peft=%s)",
        stage_two_adapter,
        base_model_name,
        stage_one_params if stage_one_params is not None else "unknown",
        stage_two_params if stage_two_params is not None else "unknown",
        bool(getattr(model, "peft_config", None)),
    )
    if stage_two_params is not None and stage_one_params == stage_two_params:
        log.warning(
            "Stage2 adapter load did not change parameter count for %s (base %s).",
            stage_two_adapter,
            base_model_name,
        )
    if hasattr(model, "set_adapter"):
        try:
            model.set_adapter("stage2")
        except (KeyError, ValueError, RuntimeError) as exc:  # pragma: no cover - defensive
            log.debug("Unable to activate stage2 adapter before merge: %s", exc)

    return model


def _merge_lora_adapter_for_vllm(
    base_model: str,
    adapter_dir: Path,
    *,
    _peft_loader=None,
    _tokenizer_loader=None,
    _torch_module=None,
    _base_model_loader=None,
    _peft_model_class=None,
) -> Path:
    """Materialise a merged adapter directory for vLLM when LoRA APIs are missing.

    Parameters
    ----------
    base_model:
        The base Hugging Face model identifier or path.
    adapter_dir:
        Directory containing PEFT adapter weights.

    Returns
    -------
    Path
        Path to a directory that holds merged weights compatible with vLLM.
    """

    if _peft_loader is None:
        try:  # Import on demand to avoid mandatory dependency when unused
            from peft import AutoPeftModelForCausalLM  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency missing
            raise RuntimeError(
                "vLLM build lacks LoRA APIs and 'peft' is unavailable to merge the adapter."
            ) from exc
        peft_loader = AutoPeftModelForCausalLM.from_pretrained
    else:
        peft_loader = _peft_loader

    if _tokenizer_loader is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - transformers should be installed
            raise RuntimeError(
                "Transformers is required to materialise merged LoRA weights."
            ) from exc
        tokenizer_loader = AutoTokenizer.from_pretrained
    else:
        tokenizer_loader = _tokenizer_loader

    if _torch_module is None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch should be installed
            raise RuntimeError("PyTorch is required to merge LoRA adapters for vLLM.") from exc
        torch_module = torch
    else:
        torch_module = _torch_module

    base_adapter_dir: Optional[Path] = None
    resolved_base_model = base_model
    tokenizer_source = base_model
    base_adapter_mtime: Optional[float] = None

    base_model_path = Path(base_model)
    if base_model_path.is_dir():
        has_adapter_config = (base_model_path / "adapter_config.json").exists()
        has_config = (base_model_path / "config.json").exists()
        if has_adapter_config and not has_config:
            base_adapter_dir = base_model_path
            base_adapter_mtime = _latest_mtime(base_model_path)
            try:
                resolved_base_model = resolve_peft_base(base_model_path)
            except ValueError as exc:
                raise RuntimeError(
                    f"Unable to resolve base model for chained adapter at '{base_model_path}'"
                ) from exc
            tokenizer_source = resolved_base_model
            log.info(
                "Base model %s is a PEFT adapter; resolving chained base %s before merging %s",
                base_model,
                resolved_base_model,
                adapter_dir,
            )

    merge_dir = adapter_dir / "_vllm_merged_adapter"
    metadata_path = merge_dir / "prefadap_metadata.json"

    adapter_mtime = _latest_mtime(adapter_dir)
    target_meta = {"base_model": base_model, "adapter_mtime": adapter_mtime}
    if base_adapter_mtime is not None:
        target_meta["base_adapter_mtime"] = base_adapter_mtime
        target_meta["resolved_base_model"] = resolved_base_model

    if merge_dir.exists() and metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            existing = None
        if existing == target_meta and _has_complete_hf_artifacts(merge_dir):
            log.info(
                "Reusing previously merged adapter for base model %s at %s",
                base_model,
                merge_dir,
            )
            return merge_dir
        if existing == target_meta:
            log.info(
                "Previously merged adapter at %s is missing Hugging Face metadata; regenerating",
                merge_dir,
            )
        shutil.rmtree(merge_dir)

    if merge_dir.exists():
        shutil.rmtree(merge_dir)
    merge_dir.mkdir(parents=True, exist_ok=True)

    last_error: Optional[Exception] = None
    dtype_candidates = [getattr(torch_module, name, None) for name in ("float16", "bfloat16", "float32")]
    config_source = resolved_base_model if base_adapter_dir is not None else base_model

    for dtype in dtype_candidates:
        load_kwargs: Dict[str, object] = {"device_map": {"": "cpu"}}
        if dtype is not None:
            load_kwargs["dtype"] = dtype
        try:
            if base_adapter_dir is not None:
                if _base_model_loader is None:
                    try:
                        from transformers import AutoModelForCausalLM
                    except ImportError as exc:  # pragma: no cover - transformers missing
                        raise RuntimeError(
                            "Transformers is required to resolve chained LoRA adapters."
                        ) from exc
                    base_model_loader = AutoModelForCausalLM.from_pretrained
                else:
                    base_model_loader = _base_model_loader

                if _peft_model_class is None:
                    try:
                        from peft import PeftModel  # type: ignore
                    except ImportError as exc:  # pragma: no cover - optional dependency
                        raise RuntimeError(
                            "PEFT is required to compose chained LoRA adapters for vLLM."
                        ) from exc
                    peft_model_class = PeftModel
                else:
                    peft_model_class = _peft_model_class

                log.info(
                    "Merging chained LoRA adapters %s -> %s onto base %s using dtype=%s",
                    base_adapter_dir,
                    adapter_dir,
                    resolved_base_model,
                    dtype,
                )
                model = _load_chained_peft_model(
                    resolved_base_model,
                    base_adapter_dir,
                    adapter_dir,
                    load_kwargs=load_kwargs,
                    base_model_loader=base_model_loader,
                    peft_model_class=peft_model_class,
                )
            else:
                log.info(
                    "Merging LoRA adapter %s into base model %s using dtype=%s",
                    adapter_dir,
                    base_model,
                    dtype,
                )
                model = _invoke_peft_loader(peft_loader, base_model, adapter_dir, load_kwargs)
                loaded_params = _count_parameters(model)
                log.info(
                    "Loaded adapter %s onto base %s (params=%s, peft=%s)",
                    adapter_dir,
                    base_model,
                    loaded_params if loaded_params is not None else "unknown",
                    bool(getattr(model, "peft_config", None)),
                )
            pre_merge_params = _count_parameters(model)
            log.info(
                "Adapter merge starting (dtype=%s, params=%s)",
                dtype,
                pre_merge_params if pre_merge_params is not None else "unknown",
            )
            merged_model = model.merge_and_unload()
            if merged_model is None:  # Older PEFT versions return the model in-place
                merged_model = model
            post_merge_params = _count_parameters(merged_model)
            log.info(
                "Adapter merge completed (dtype=%s, params=%s)",
                dtype,
                post_merge_params if post_merge_params is not None else "unknown",
            )
            _ensure_hf_artifacts(
                merged_model,
                merge_dir,
                tokenizer_loader=tokenizer_loader,
                tokenizer_source=str(tokenizer_source),
                config_source=str(config_source),
            )
            metadata_path.write_text(json.dumps(target_meta), encoding="utf-8")
            artifacts_ok = _has_complete_hf_artifacts(merge_dir)
            if artifacts_ok:
                log.info("Merged adapter saved at %s with complete HF artifacts.", merge_dir)
            else:
                missing = _missing_hf_artifacts(merge_dir)
                log.warning(
                    "Merged adapter saved at %s but is missing artifacts: %s",
                    merge_dir,
                    ", ".join(missing),
                )
            del merged_model
            del model
            gc.collect()
            return merge_dir
        except Exception as exc:  # pragma: no cover - exercised when dtype unsupported
            last_error = exc
            log.warning(
                "Failed to merge LoRA adapter with dtype=%s due to %s; trying next dtype",
                dtype,
                exc,
            )
            shutil.rmtree(merge_dir, ignore_errors=True)
            merge_dir.mkdir(parents=True, exist_ok=True)
            gc.collect()
            continue

    raise RuntimeError(
        "Unable to merge LoRA adapter %s into base model %s for vLLM usage" % (adapter_dir, base_model)
    ) from last_error


class VLLMGenerator:
    """High-performance text generator using vLLM for local inference."""
    
    def __init__(
        self,
        model_name: str,
        *,
        max_tokens: int = 512,
        max_model_len: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        trust_remote_code: bool = False,
        dtype: str = "auto",
        # vLLM engine batching knobs
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        enable_prefix_caching: bool = True,
        block_size: int = 16,
        swap_space: float = 0.1,
    ):
        """Initialize VLLMGenerator with model and sampling parameters."""
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is required for VLLMGenerator. Install it with: pip install vllm"
            )
        
        resolved_model = _resolve_model_path(model_name)
        model_path = Path(resolved_model)

        adapter_dir: Optional[Path]
        base_model = resolved_model
        adapter_dir = None

        if model_path.is_dir() and is_adapter_directory(model_path):
            adapter_dir = model_path
            base_name = get_adapter_base_model(model_path)
            if not base_name:
                raise ValueError(
                    f"Adapter directory '{model_path}' missing base_model_name_or_path; "
                    "cannot initialise vLLM engine."
                )
            base_model = _resolve_model_path(base_name)
            log.info(
                "Detected PEFT adapter at %s; loading base model %s for vLLM and attaching adapter",
                model_path,
                base_model,
            )

        self.input_model_name = model_name
        self.model_name = base_model
        self.adapter_path = adapter_dir
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._lora_generate_kwargs: Dict[str, object] = {}
        self._merged_model_dir: Optional[Path] = None
        self._tokenizer: Optional[Any] = None
        self._prompt_reserved_tokens: Optional[int] = None

        # Initialize vLLM engine
        llm_kwargs = dict(
            model=base_model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            enable_prefix_caching=enable_prefix_caching,
            block_size=block_size,
            swap_space=swap_space,
        )

        if adapter_dir is not None and not self._llm_class_supports_lora():
            log.warning(
                "vLLM installation does not expose a LoRA loading API; merging adapter %s into base model %s",
                adapter_dir,
                base_model,
            )
            merged_dir = _merge_lora_adapter_for_vllm(base_model, adapter_dir)
            self._merged_model_dir = merged_dir
            llm_kwargs["model"] = str(merged_dir)
            self.model_name = str(merged_dir)
            self.adapter_path = None
            adapter_dir = None
        elif adapter_dir is not None:
            init_sig = inspect.signature(LLM.__init__)
            if "enable_lora" in init_sig.parameters:
                llm_kwargs.setdefault("enable_lora", True)
            if "max_loras" in init_sig.parameters and "max_loras" not in llm_kwargs:
                llm_kwargs["max_loras"] = max(1, int(llm_kwargs.get("max_num_seqs") or 1))

        # Cap model context length if provided to avoid KV cache OOM
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = max_num_seqs
        self.llm = LLM(**llm_kwargs)
        engine = getattr(self.llm, "llm_engine", None)
        self.llm_config = getattr(engine, "model_config", None)
        if self.llm_config is None:
            self.llm_config = getattr(self.llm, "model_config", None)

        if adapter_dir is not None:
            self._attach_lora_adapter(adapter_dir)

        env_auto_truncate = os.environ.get("PSEUDO_AUTO_TRUNCATE", "")
        self._auto_truncate_prompts = env_auto_truncate.lower() in {"1", "true", "yes", "y", "on"}

        resolved_max_model_len: Optional[int] = None
        if max_model_len is not None:
            try:
                resolved_max_model_len = int(max_model_len)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                resolved_max_model_len = None
        if resolved_max_model_len is None:
            env_limit = os.environ.get("MAX_MODEL_LEN")
            if env_limit:
                try:
                    resolved_max_model_len = int(env_limit)
                except ValueError:
                    log.warning("Invalid MAX_MODEL_LEN value '%s'; ignoring", env_limit)

        engine_max_len = (
            getattr(self.llm_config, "max_model_len", None)
            if self.llm_config is not None
            else None
        )
        if resolved_max_model_len is None and engine_max_len is not None:
            try:
                resolved_max_model_len = int(engine_max_len)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                resolved_max_model_len = None

        self._max_model_len = resolved_max_model_len
        if self._auto_truncate_prompts and self._max_model_len is not None:
            log.info(
                "[vllm_generator] Auto-truncating prompts to MAX_MODEL_LEN=%s",
                self._max_model_len,
            )

        # Set up sampling parameters
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["\n\n", "User:", "Human:", "Assistant:", "<|eot_id|>", "<|im_end|>"],
        )
        if self.llm_config is not None:
            max_len = getattr(self.llm_config, "max_model_len", None)
            if max_len is not None:
                self.sampling_params.max_model_len = max_len
    
    def generate(self, prompts: List[str], batch_size: Optional[int] = None) -> List[str]:
        """Generate text for a list of prompts.
        
        Parameters
        ----------
        prompts : List[str]
            List of prompts to generate text for
        batch_size : Optional[int]
            Batch size for generation (None for single batch)
            
        Returns
        -------
        List[str]
            List of generated texts
        """
        if batch_size is None or len(prompts) <= batch_size:
            outputs = self._generate(prompts, self.sampling_params)
            return [
                output.outputs[0].text.strip()
                if output is not None and getattr(output, "outputs", None)
                else ""
                for output in outputs
            ]
        
        # Multiple batches
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            outputs = self._generate(batch, self.sampling_params)
            results.extend(
                [
                    output.outputs[0].text.strip()
                    if output is not None and getattr(output, "outputs", None)
                    else ""
                    for output in outputs
                ]
            )
        
        return results
    
    def generate_single(self, prompt: str) -> str:
        """Generate text for a single prompt.
        
        Parameters
        ---------- 
        prompt : str
            Prompt to generate text for
            
        Returns
        -------
        str
            Generated text
        """
        outputs = self._generate([prompt], self.sampling_params)
        first = outputs[0] if outputs else None
        if first is None or not getattr(first, "outputs", None):
            log.warning(
                "[vllm_generator] No generation produced for prompt due to max_model_len constraints"
            )
            return ""
        return first.outputs[0].text.strip()

    def generate_multi(
        self,
        prompts: List[str],
        *,
        n: int,
        batch_size: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[List[str]]:
        """Generate multiple candidates per prompt using parallel sampling.

        Parameters
        ----------
        prompts:
            List of prompts to generate text for.
        n:
            Number of candidates per prompt (vLLM parallel sampling).
        batch_size:
            Optional micro-batch size for prompts to control VRAM.
        temperature, top_p, max_tokens:
            Optional overrides for this call only.

        Returns
        -------
        List[List[str]]
            For each input prompt, a list of ``n`` candidate texts.
        """
        # Build call-specific sampling params without mutating defaults
        sp = SamplingParams(
            n=n,
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            stop=self.sampling_params.stop,
        )
        if self.llm_config is not None:
            max_len = getattr(self.llm_config, "max_model_len", None)
            if max_len is not None:
                sp.max_model_len = max_len

        if batch_size is None or len(prompts) <= batch_size:
            outputs = self._generate(prompts, sp)
            return [
                [o.text.strip() for o in out.outputs]
                if out is not None and getattr(out, "outputs", None)
                else []
                for out in outputs
            ]

        results: List[List[str]] = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            outs = self._generate(batch, sp)
            results.extend(
                [
                    [o.text.strip() for o in out.outputs]
                    if out is not None and getattr(out, "outputs", None)
                    else []
                    for out in outs
                ]
            )
        return results

    # ------------------------------------------------------------------
    @staticmethod
    def _llm_class_supports_lora() -> bool:
        for attr in ("load_lora_adapter", "load_adapter", "add_lora_adapter", "load_lora"):
            if hasattr(LLM, attr):
                return True
        return False

    def _count_llm_parameters(self) -> Optional[int]:
        model = getattr(self.llm, "model", None)
        if model is None:
            engine = getattr(self.llm, "llm_engine", None)
            model = getattr(engine, "model", None) if engine is not None else None
        if model is None:
            return None
        return _count_parameters(model)

    # ------------------------------------------------------------------
    def _attach_lora_adapter(self, adapter_dir: Path) -> None:
        adapter_name = adapter_dir.name or "adapter"
        loader = None
        for attr in ("load_lora_adapter", "load_adapter", "add_lora_adapter", "load_lora"):
            if hasattr(self.llm, attr):
                loader = getattr(self.llm, attr)
                break
        if loader is None:
            raise RuntimeError(
                "vLLM installation does not expose a LoRA loading API; cannot attach adapter."
            )
        before_params = self._count_llm_parameters()
        loader(adapter_name, str(adapter_dir))
        after_params = self._count_llm_parameters()
        log.info(
            "Attached vLLM adapter %s (params: %s -> %s)",
            adapter_dir,
            before_params if before_params is not None else "unknown",
            after_params if after_params is not None else "unknown",
        )
        if after_params is not None and before_params == after_params:
            log.warning(
                "vLLM adapter load did not change parameter count for %s.",
                adapter_dir,
            )
        self._lora_generate_kwargs = self._infer_lora_kwargs(adapter_name, adapter_dir)
        if not self._lora_generate_kwargs:
            log.warning(
                "LoRA adapter '%s' attached but generate() signature does not expose a recognised "
                "parameter for forwarding it; continuing without explicit adapter binding.",
                adapter_name,
            )

    # ------------------------------------------------------------------
    def _infer_lora_kwargs(self, adapter_name: str, adapter_dir: Path) -> Dict[str, object]:
        generate_sig = inspect.signature(self.llm.generate)

        if "lora_path" in generate_sig.parameters:
            return {"lora_path": str(adapter_dir)}
        if "lora_paths" in generate_sig.parameters:
            return {"lora_paths": [str(adapter_dir)]}
        if "adapter_name" in generate_sig.parameters:
            return {"adapter_name": adapter_name}
        if "lora_name" in generate_sig.parameters:
            return {"lora_name": adapter_name}
        if "adapter" in generate_sig.parameters:
            return {"adapter": adapter_name}
        if "lora_request" in generate_sig.parameters:
            try:
                from vllm.lora import LoRARequest  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency missing
                raise RuntimeError(
                    "vLLM exposes 'lora_request' parameter but 'vllm.lora.LoRARequest' is unavailable"
                ) from exc
            try:
                request = LoRARequest(adapter_name=adapter_name, adapter_path=str(adapter_dir))
            except TypeError:
                request = LoRARequest(adapter_name, str(adapter_dir))
            return {"lora_request": request}
        if "lora_requests" in generate_sig.parameters:
            try:
                from vllm.lora import LoRARequest  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency missing
                raise RuntimeError(
                    "vLLM exposes 'lora_requests' parameter but 'vllm.lora.LoRARequest' is unavailable"
                ) from exc
            try:
                request = LoRARequest(adapter_name=adapter_name, adapter_path=str(adapter_dir))
            except TypeError:
                request = LoRARequest(adapter_name, str(adapter_dir))
            return {"lora_requests": [request]}

        return {}

    # ------------------------------------------------------------------
    def _generate(self, prompts: List[str], sampling_params: SamplingParams):
        prompts = self._maybe_truncate_prompts(prompts)
        limit = self._max_model_len
        if limit is None and self.llm_config is not None:
            limit = getattr(self.llm_config, "max_model_len", None)

        tokenizer = None
        if limit is not None and limit > 0 and prompts:
            tokenizer = self._ensure_tokenizer()
            reserved_tokens = max(0, self._get_prompt_reserved_tokens(tokenizer))
            effective_limit = self._effective_prompt_limit(limit, tokenizer)
            truncated_prompts: List[str] = []
            longest_prompt = 0
            for prompt in prompts:
                token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                if effective_limit is not None and len(token_ids) > effective_limit:
                    token_ids = token_ids[:effective_limit]
                    prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
                actual_length = len(token_ids) + reserved_tokens
                longest_prompt = max(longest_prompt, actual_length)
                truncated_prompts.append(prompt)
            log.debug(
                "[vllm_generator] Longest prompt length after truncation: %s tokens (including %s reserved)",
                longest_prompt,
                reserved_tokens,
            )
            prompts = truncated_prompts

        config_limit = None
        if self.llm_config is not None:
            config_limit = getattr(self.llm_config, "max_model_len", None)
            if config_limit is not None:
                try:
                    sampling_params.max_model_len = config_limit
                except AttributeError:
                    setattr(sampling_params, "max_model_len", config_limit)

        model_limit = config_limit
        if model_limit is None:
            model_limit = getattr(sampling_params, "max_model_len", None)
        if model_limit is None:
            model_limit = self._max_model_len

        if model_limit is None:
            if self._lora_generate_kwargs:
                return self.llm.generate(prompts, sampling_params, **self._lora_generate_kwargs)
            return self.llm.generate(prompts, sampling_params)

        if tokenizer is None:
            tokenizer = self._ensure_tokenizer()

        results: List[Optional[object]] = [None] * len(prompts)
        requested_max_tokens = getattr(sampling_params, "max_tokens", None)
        requested_max_new_tokens = getattr(sampling_params, "max_new_tokens", requested_max_tokens)

        for index, prompt in enumerate(prompts):
            try:
                encoding = tokenizer(prompt)
            except TypeError:
                encoding = None
            prompt_ids = getattr(encoding, "input_ids", None)
            if prompt_ids is None:
                try:
                    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
                except Exception:
                    prompt_ids = []
            prompt_len = len(prompt_ids)
            allowed_gen = model_limit - prompt_len

            if allowed_gen <= 0:
                log.warning(
                    "[vllm_generator] Skipping prompt with length %s tokens; exceeds max_model_len=%s",
                    prompt_len,
                    model_limit,
                )
                continue

            candidate_limit = allowed_gen
            if requested_max_tokens is not None:
                candidate_limit = min(requested_max_tokens, candidate_limit)
            if requested_max_new_tokens is not None:
                candidate_limit = min(requested_max_new_tokens, candidate_limit)
            candidate_limit = max(0, candidate_limit)

            total_tokens = prompt_len + candidate_limit
            log.debug(
                "[vllm_generator] Prompt length=%s tokens; requested max_tokens=%s; adjusted max_new_tokens=%s; "
                "prompt+gen=%s vs max_model_len=%s",
                prompt_len,
                requested_max_tokens,
                candidate_limit,
                total_tokens,
                model_limit,
            )

            if requested_max_tokens is not None and candidate_limit < requested_max_tokens:
                log.info(
                    "[vllm_generator] Truncating generation from %s to %s tokens to respect max_model_len=%s",
                    requested_max_tokens,
                    candidate_limit,
                    model_limit,
                )

            per_request_params = copy.deepcopy(sampling_params)
            try:
                per_request_params.max_tokens = candidate_limit
            except AttributeError:
                setattr(per_request_params, "max_tokens", candidate_limit)
            try:
                per_request_params.max_new_tokens = candidate_limit
            except AttributeError:
                setattr(per_request_params, "max_new_tokens", candidate_limit)
            try:
                per_request_params.max_model_len = model_limit
            except AttributeError:
                setattr(per_request_params, "max_model_len", model_limit)

            if self._lora_generate_kwargs:
                generated = self.llm.generate([prompt], per_request_params, **self._lora_generate_kwargs)
            else:
                generated = self.llm.generate([prompt], per_request_params)
            if generated:
                results[index] = generated[0]

        return results

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.llm.get_tokenizer()
        return self._tokenizer

    def _get_prompt_reserved_tokens(self, tokenizer=None) -> int:
        if self._prompt_reserved_tokens is not None:
            return self._prompt_reserved_tokens

        if tokenizer is None:
            try:
                tokenizer = self._ensure_tokenizer()
            except AttributeError:
                self._prompt_reserved_tokens = 0
                return 0

        try:
            reserved = self._infer_reserved_prompt_tokens(tokenizer)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("[vllm_generator] Unable to infer reserved prompt tokens: %s", exc)
            reserved = 0

        try:
            reserved_int = int(reserved)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            reserved_int = 0
        self._prompt_reserved_tokens = max(0, reserved_int)
        if self._prompt_reserved_tokens:
            log.debug(
                "[vllm_generator] Reserving %s tokenizer special tokens for prompts",
                self._prompt_reserved_tokens,
            )
        return self._prompt_reserved_tokens

    def _effective_prompt_limit(self, limit: Optional[int], tokenizer=None) -> Optional[int]:
        if limit is None:
            return None
        if limit <= 0:
            return 0
        reserved = self._get_prompt_reserved_tokens(tokenizer)
        if reserved <= 0:
            return limit
        effective = limit - reserved
        if effective <= 0:
            return 0
        return effective

    def _infer_reserved_prompt_tokens(self, tokenizer) -> int:
        reserved = 0

        try:
            num_special = tokenizer.num_special_tokens_to_add(pair=False)
        except Exception:  # pragma: no cover - depends on tokenizer API
            num_special = None
        if isinstance(num_special, int):
            reserved = max(reserved, num_special)
        elif isinstance(num_special, (list, tuple)):
            reserved = max(reserved, len(num_special))

        try:
            built = tokenizer.build_inputs_with_special_tokens([])
        except Exception:  # pragma: no cover - tokenizer specific quirks
            built = None
        if isinstance(built, (list, tuple)):
            reserved = max(reserved, len(built))

        sample_text = "prefadap"
        try:
            plain = tokenizer.encode(sample_text, add_special_tokens=False)
            with_special = tokenizer.encode(sample_text, add_special_tokens=True)
        except Exception:  # pragma: no cover - tokenizer limitations
            plain = []
            with_special = []
        if len(with_special) >= len(plain):
            reserved = max(reserved, len(with_special) - len(plain))

        return max(0, reserved)

    def _maybe_truncate_prompts(self, prompts: List[str]) -> List[str]:
        if not self._auto_truncate_prompts or self._max_model_len is None or not prompts:
            return prompts

        tokenizer = self._ensure_tokenizer()
        limit = self._effective_prompt_limit(self._max_model_len, tokenizer)
        if limit is None or limit <= 0:
            return prompts

        truncated_prompts: List[str] = []
        for prompt in prompts:
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokens) <= limit:
                truncated_prompts.append(prompt)
            else:
                truncated_tokens = tokens[: max(0, limit)]
                truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=False)
                truncated_prompts.append(truncated_text)

        return truncated_prompts
__all__ = ["VLLMGenerator"]
