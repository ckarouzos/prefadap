"""Base class for all training pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import os
import time
import torch
from pathlib import Path

import wandb
from transformers import Trainer, PreTrainedModel, AutoTokenizer

from prefadap.plugin import PluginManager, PluginContext
from prefadap.utils.logging import create_logger
from prefadap.utils.seeding import set_seed

from .args import DEFAULTS
from .config import GRPOArgs, PPOArgs, RMArgs
from .utils import (
    _maybe_apply_lora,
    _maybe_init_wandb,
    _maybe_freeze_embeddings,
    _tune_70b_defaults,
    save_model_with_lora,
    check_model_exists,
    validate_persistent_storage,
    resolve_device,
    ensure_generation_config,
    safe_move_model,
    build_ppo_model_load_config,
    validate_ppo_distributed_launch,
)



class TrainingPipeline(ABC):
    """Abstract base class orchestrating common training tasks."""

    def __init__(self, cfg: Any, plugin_manager: PluginManager | None = None) -> None:
        self.cfg = cfg
        _tune_70b_defaults(self.cfg)
        import logging
        logging_dir = getattr(cfg, "logging_dir", ".") if hasattr(cfg, "logging_dir") else None
        self.logger = create_logger(logging_dir) if logging_dir is not None else logging.getLogger(__name__)
        self.model: PreTrainedModel | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.trainer: Trainer | None = None
        self.use_wandb: bool = False
        
        # Plugin system
        # If no plugin manager is provided, create a default one with empty configuration
        # This ensures plugins are only loaded if explicitly configured
        self.plugin_manager = plugin_manager or PluginManager()
        self.plugin_context = PluginContext(config=cfg)
        
        # Load and initialize plugins
        self.plugin_manager.load_plugins()
        self.plugin_manager.initialize_plugins(self.plugin_context)

    # ------------------------------------------------------------------
    # Setup hooks
    # ------------------------------------------------------------------
    def validate_config(self) -> None:  # pragma: no cover - trivial
        if not getattr(self.cfg, "model_name_or_path", None):
            raise ValueError("`model_name_or_path` must be provided")
        out_dir = getattr(self.cfg, "output_dir", "runs")
        os.makedirs(out_dir, exist_ok=True)
        # Stage-2 guardrail: if an explicit reference policy is provided, ensure it exists
        try:
            stage = int(getattr(self.cfg, "stage", 1))
        except Exception:
            stage = 1
        ref_path = getattr(self.cfg, "ref_model_name_or_path", None)
        if stage >= 2 and ref_path:
            if not os.path.exists(ref_path):
                raise FileNotFoundError(f"Reference checkpoint path does not exist: {ref_path}")

    def set_seed(self) -> None:
        seed = getattr(self.cfg, "seed", None)
        if seed is not None:
            set_seed(seed)

    def init_wandb(self) -> None:
        project = getattr(self.cfg, "wandb_project", DEFAULTS.get("project", "training"))
        run_name = getattr(self.cfg, "run_id", getattr(self.cfg, "model_name_or_path", "run"))
        tags = getattr(self.cfg, "wandb_tags", None)
        self.use_wandb = _maybe_init_wandb(project, run_name, getattr(self.cfg, "use_wandb", False), 
                                           config=self.cfg, tags=tags)

    def load_model_tokenizer(self) -> None:
        self.plugin_manager.execute_hook("pre_model_load", self.plugin_context)
        
        args = self.cfg
        model_name = getattr(args, "model_name_or_path", "")
        seed_hint = getattr(args, "seed", None)
        stage_hint = getattr(args, "stage", None)
        from . import training_wrappers as tw  # import here for test monkeypatching

        # Resolve device first to enable rank-aware placement
        device = resolve_device(getattr(args, "device", None))
        args.device = device
        
        # Log GPU configuration
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            self.logger.info(
                f"GPU configuration: {gpu_count} device(s) available, "
                f"selected device: {device}"
            )
            # Log details about each GPU
            try:
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    self.logger.info(
                        f"  GPU {i}: {props.name}, "
                        f"Memory: {props.total_memory / (1024**3):.2f} GB"
                    )
            except Exception:
                pass  # Don't fail if we can't get GPU properties
        else:
            self.logger.info("No CUDA devices available, using CPU")

        if isinstance(args, RMArgs):
            from prefadap.models.loader import _resolve_model_path

            model_name = _resolve_model_path(
                model_name,
                seed=seed_hint,
                stage=stage_hint,
            )
            args.model_name_or_path = model_name
            model, tokenizer = tw.load_reward_model_and_tokenizer(
                model_name,
                gradient_checkpointing=getattr(args, "gradient_checkpointing", False),
                tokenizer_name_or_path=getattr(args, "tokenizer_name_or_path", None),
                config={
                    "num_labels": 1,
                    "problem_type": "regression",
                },
            )
            model = _maybe_apply_lora(args, model)
            if getattr(args, "lora", False):
                named_params = dict(model.named_parameters())
                lora_param_count = sum(1 for name in named_params if "lora_" in name)
                if lora_param_count == 0:
                    raise ValueError(
                        "LoRA was requested for RM training but no LoRA parameters were found. "
                        "Aborting to avoid unintended full-finetuning."
                    )
                score_param_count = 0
                for name, param in named_params.items():
                    if name.startswith("score."):
                        param.requires_grad = True
                        score_param_count += 1
                    elif "lora_" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                self.logger.info(
                    "RM LoRA diagnostics: trainable LoRA params=%s trainable score params=%s",
                    lora_param_count,
                    score_param_count,
                )
        elif isinstance(args, (PPOArgs, GRPOArgs)) or getattr(args, "objective", None) == "ppo":
            from prefadap.models.loader import _resolve_model_path

            model_name = _resolve_model_path(
                model_name,
                seed=seed_hint,
                stage=stage_hint,
            )
            args.model_name_or_path = model_name

            validate_ppo_distributed_launch(args)
            model_config = build_ppo_model_load_config(None)
            model, tokenizer = tw.load_model_and_tokenizer(
                model_name,
                gradient_checkpointing=getattr(args, "gradient_checkpointing", False),
                tokenizer_name_or_path=getattr(args, "tokenizer_name_or_path", None),
                config=model_config,
            )
            model = _maybe_apply_lora(args, model)
            # PPO/GRPO rely on PPOTrainer to attach value heads; keep the backbone plain.
            ensure_generation_config(model)
            model = _maybe_freeze_embeddings(args, model)
        else:
            from prefadap.models.loader import _resolve_model_path

            model_name = _resolve_model_path(
                model_name,
                seed=seed_hint,
                stage=stage_hint,
            )
            args.model_name_or_path = model_name

            model, tokenizer = tw.load_model_and_tokenizer(
                model_name,
                gradient_checkpointing=getattr(args, "gradient_checkpointing", False),
                tokenizer_name_or_path=getattr(args, "tokenizer_name_or_path", None),
            )
            model = _maybe_apply_lora(args, model)
            model = _maybe_freeze_embeddings(args, model)
        
        # Log model information
        try:
            param_count = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(
                f"Model loaded: {model_name}, "
                f"Total params: {param_count:,}, "
                f"Trainable params: {trainable_params:,} "
                f"({100 * trainable_params / param_count:.2f}%)"
            )
        except Exception:
            pass  # Don't fail if we can't count parameters

        self._log_model_diagnostics(model, model_name)
        
        # Initialize _to_called tracking attribute for tests
        if not hasattr(model, "_to_called"):
            model._to_called = False
        
        # Only move model to device if it's not already sharded across GPUs
        # Use centralized safe_move_model helper
        safe_move_model(model, str(device), "policy model")
        
        self.model, self.tokenizer = model, tokenizer
        
        # Update plugin context with loaded model and tokenizer
        self.plugin_context.model = self.model
        self.plugin_context.tokenizer = self.tokenizer
        
        self.plugin_manager.execute_hook("post_model_load", self.plugin_context)

    def train(self) -> Any:  # pragma: no cover - simple delegation
        return self.run()

    # ------------------------------------------------------------------
    def _log_model_diagnostics(self, model: PreTrainedModel, model_name: str) -> None:
        """Log detailed model diagnostics for PPO/GRPO investigations."""

        try:
            self.logger.info(
                "Model diagnostics: counting object type=%s repr=%s",
                type(model),
                repr(model),
            )
            model_type = getattr(getattr(model, "config", None), "model_type", None)
            self.logger.info("Model diagnostics: type(model)=%s", type(model))
            self.logger.info("Model diagnostics: config.model_type=%s", model_type)

            base_model = getattr(model, "base_model", None)
            pretrained_model = getattr(model, "pretrained_model", None)
            if base_model is not None:
                self.logger.info("Model diagnostics: type(model.base_model)=%s", type(base_model))
            if pretrained_model is not None:
                self.logger.info(
                    "Model diagnostics: type(model.pretrained_model)=%s", type(pretrained_model)
                )

            peft_config = getattr(model, "peft_config", None)
            if isinstance(peft_config, dict):
                inference_modes = {
                    name: getattr(cfg, "inference_mode", None) for name, cfg in peft_config.items()
                }
                self.logger.info(
                    "Model diagnostics: peft_config adapters=%s inference_mode=%s",
                    sorted(peft_config.keys()),
                    inference_modes,
                )

            model_path = Path(model_name)
            if model_path.exists() and model_path.is_dir():
                file_flags = {
                    "adapter_model.safetensors": (model_path / "adapter_model.safetensors").exists(),
                    "adapter_config.json": (model_path / "adapter_config.json").exists(),
                    "peft_config.json": (model_path / "peft_config.json").exists(),
                    "pytorch_model.bin": (model_path / "pytorch_model.bin").exists(),
                    "model.safetensors": (model_path / "model.safetensors").exists(),
                    "pytorch_model.bin.index.json": (model_path / "pytorch_model.bin.index.json").exists(),
                    "model.safetensors.index.json": (model_path / "model.safetensors.index.json").exists(),
                }
                sharded_files = [
                    path.name
                    for path in model_path.iterdir()
                    if path.is_file()
                    and (
                        path.name.startswith("model-")
                        and path.suffix == ".safetensors"
                        or path.name.startswith("pytorch_model-")
                        and path.suffix == ".bin"
                    )
                ]
                file_flags["sharded_checkpoints"] = bool(sharded_files)
                self.logger.info("Checkpoint contents for %s: %s", model_path, file_flags)
                if sharded_files:
                    self.logger.info(
                        "Checkpoint shard files for %s: %s",
                        model_path,
                        sorted(sharded_files),
                    )

            output_dir = getattr(self.cfg, "output_dir", None)
            if output_dir:
                final_model_dir = Path(output_dir) / "final_model"
                if final_model_dir.exists() and final_model_dir.is_dir():
                    contents = sorted([p.name for p in final_model_dir.iterdir() if p.is_file()])
                    self.logger.info(
                        "final_model/ contents (up to 50): %s",
                        contents[:50],
                    )

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            self.logger.info(
                "Parameter summary: total=%s trainable=%s frozen=%s (%.2f%% trainable)",
                f"{total_params:,}",
                f"{trainable_params:,}",
                f"{frozen_params:,}",
                100 * trainable_params / total_params if total_params else 0.0,
            )

            named_params = dict(model.named_parameters())
            sorted_names = sorted(named_params.keys())
            self.logger.info("Parameter names (first 20): %s", sorted_names[:20])
            self.logger.info("Parameter names (last 20): %s", sorted_names[-20:])

            lora_names = [name for name in sorted_names if "lora_" in name]
            non_lora_names = [name for name in sorted_names if "lora_" not in name]
            self.logger.info(
                "Parameter name counts: lora_=%s non_lora_=%s",
                len(lora_names),
                len(non_lora_names),
            )

            req_grad_true = [p for p in named_params.values() if p.requires_grad]
            req_grad_false = [p for p in named_params.values() if not p.requires_grad]
            self.logger.info(
                "Requires_grad summary: true=%s params (%s elems), false=%s params (%s elems)",
                len(req_grad_true),
                f"{sum(p.numel() for p in req_grad_true):,}",
                len(req_grad_false),
                f"{sum(p.numel() for p in req_grad_false):,}",
            )

            lora_params = {name: p for name, p in named_params.items() if "lora_" in name}
            non_lora_params = {name: p for name, p in named_params.items() if "lora_" not in name}
            self.logger.info(
                "LoRA param totals: total=%s trainable=%s frozen=%s",
                f"{sum(p.numel() for p in lora_params.values()):,}",
                f"{sum(p.numel() for p in lora_params.values() if p.requires_grad):,}",
                f"{sum(p.numel() for p in lora_params.values() if not p.requires_grad):,}",
            )
            self.logger.info(
                "Non-LoRA param totals: total=%s trainable=%s frozen=%s",
                f"{sum(p.numel() for p in non_lora_params.values()):,}",
                f"{sum(p.numel() for p in non_lora_params.values() if p.requires_grad):,}",
                f"{sum(p.numel() for p in non_lora_params.values() if not p.requires_grad):,}",
            )

            try:
                base_weight = (
                    model.base_model.model.model.layers[0].self_attn.q_proj.weight
                )
                base_weight_numel = base_weight.numel()
                base_weight_name = None
                for name, param in named_params.items():
                    if param is base_weight:
                        base_weight_name = name
                        break
                self.logger.info(
                    "Base weight probe: layers[0].self_attn.q_proj.weight numel=%s in_named_params=%s name=%s",
                    f"{base_weight_numel:,}",
                    base_weight_name is not None,
                    base_weight_name,
                )
            except Exception as exc:
                self.logger.info("Base weight probe failed: %s", exc)

            self.logger.info("Top-level module parameter breakdown:")
            for name, child in model.named_children():
                child_total = sum(p.numel() for p in child.parameters())
                child_trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
                self.logger.info(
                    "  %s: total=%s trainable=%s",
                    name,
                    f"{child_total:,}",
                    f"{child_trainable:,}",
                )
        except Exception:
            self.logger.debug("Model diagnostics failed", exc_info=True)

    # ------------------------------------------------------------------
    # Abstract hooks to be implemented by subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def prepare_datasets(self, training_args: Any) -> Any:
        """Prepare datasets for training.
        
        Args:
            training_args: Training configuration
            
        Returns:
            Prepared datasets
        """
        raise NotImplementedError

    @abstractmethod
    def build_trainer(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        training_args: Any,
        datasets: Any,
    ) -> Trainer:
        """Build trainer for the specific training method.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer for the model
            training_args: Training configuration
            datasets: Prepared datasets
            
        Returns:
            Configured trainer
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    def save_artifacts(self) -> None:
        output_dir = getattr(self.cfg, "output_dir", "./")
        
        # Validate storage location for production environments
        if not validate_persistent_storage(output_dir):
            self.logger.warning(
                f"Output directory '{output_dir}' may not be on persistent storage. "
                f"Models saved here might not survive job completion in HPC environments. "
                f"Consider using a persistent storage path (e.g., shared /data or a network volume) "
                f"for production runs."
            )
        
        # Save model in final_model subdirectory for consistency with discovery functions
        final_model_dir = Path(output_dir) / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect zero-shot mode to handle save differently
        num_epochs = getattr(self.cfg, "num_train_epochs", 1)
        is_zero_shot = (num_epochs == 0)
        
        # Save the model - use trainer if available and not in zero-shot mode
        # In zero-shot mode, use direct save to avoid issues with unused trainer
        if is_zero_shot and self.model is not None:
            # Direct save for zero-shot mode to ensure model is always saved
            self.logger.info("Zero-shot mode: Saving model directly without trainer")
            self._save_model_direct(final_model_dir)
        elif self.trainer is not None:
            save_model_with_lora(self.trainer, str(final_model_dir))
        elif self.model is not None:
            # Direct save for cases where trainer isn't available
            # This handles edge cases beyond zero-shot mode
            self.logger.info("Saving model directly without trainer")
            self._save_model_direct(final_model_dir)
        else:
            self.logger.warning("No trainer or model available to save")
            return
        
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(str(final_model_dir))
        
        # Persist reference policy symlink when configured
        ref_path = getattr(self.cfg, "ref_model_name_or_path", None)
        if ref_path:
            try:
                link_path = final_model_dir.parent / "reference_policy"
                if link_path.exists() or link_path.is_symlink():
                    try:
                        link_path.unlink()
                    except Exception:
                        pass
                link_path.symlink_to(Path(ref_path).resolve())
            except Exception:
                # Best-effort: don't fail training if symlink cannot be created
                pass
        
        # Notify plugins about save
        self.plugin_context.set_metadata("save_path", str(final_model_dir))
        self.plugin_manager.execute_hook("on_save", self.plugin_context)

    def _save_model_direct(self, final_model_dir: Path) -> None:
        """Save model directly without using trainer.
        
        This method handles saving in zero-shot mode or when trainer is unavailable.
        """
        try:
            from peft import PeftModel
            from types import SimpleNamespace
            from prefadap.training.utils import (
                _ensure_adapter_metadata,
                _ensure_adapter_safetensors,
            )
            
            # Handle TRL's (Transformer Reinforcement Learning) AutoModelForCausalLMWithValueHead
            # which wraps the actual model. Check merge flag on the original model BEFORE 
            # unwrapping, since the flag may be set on the wrapper model (not on the 
            # pretrained model inside)
            merge_on_save = bool(getattr(self.model, "_prefadap_merge_lora_on_save", False))
            
            model_to_save = self.model
            # Only unwrap if this is actually an AutoModelForCausalLMWithValueHead
            # We check the class name rather than isinstance() because:
            # 1. Avoids import dependency on TRL in this module
            # 2. Prevents unwrapping MagicMock objects in tests (they auto-generate attributes)
            if (hasattr(self.model, "pretrained_model") and 
                type(self.model).__name__ == "AutoModelForCausalLMWithValueHead"):
                model_to_save = self.model.pretrained_model
                self.logger.info("Unwrapping model from value head wrapper")
            
            if isinstance(model_to_save, PeftModel):
                # Use the merge_on_save flag we checked earlier (before unwrapping)
                
                if merge_on_save and hasattr(model_to_save, "merge_and_unload"):
                    self.logger.info("Merging LoRA adapters into the base model before saving")
                    merged_model = model_to_save.merge_and_unload()
                    try:
                        merged_model.save_pretrained(str(final_model_dir), safe_serialization=True)
                    except TypeError:
                        # Some older model implementations may not support safe_serialization
                        merged_model.save_pretrained(str(final_model_dir))
                    return
                
                # Use the same compatibility logic as save_model_with_lora for PEFT models
                merged = False  # Track whether we saved a merged model
                try:
                    model_to_save.save_pretrained(str(final_model_dir), safe_serialization=True)
                except TypeError:
                    # Some historical PEFT versions did not accept the safe_serialization keyword
                    # Retry without it to remain compatible
                    model_to_save.save_pretrained(str(final_model_dir))
                except UnboundLocalError as exc:
                    self.logger.warning(
                        "PEFT save_pretrained failed due to missing active adapters; "
                        "retrying after merging adapters. Error: %s",
                        exc,
                    )
                    if not hasattr(model_to_save, "merge_and_unload"):
                        raise
                    merged_model = model_to_save.merge_and_unload()
                    merged_model.save_pretrained(str(final_model_dir), safe_serialization=True)
                    merged = True
                
                # Only add metadata if we saved adapters (not if we merged)
                if not merged:
                    dummy_trainer = SimpleNamespace(model=model_to_save, args=self.cfg)
                    _ensure_adapter_metadata(str(final_model_dir), dummy_trainer, model_to_save)
                    _ensure_adapter_safetensors(str(final_model_dir))
            elif hasattr(model_to_save, "save_pretrained"):
                # Handle non-PEFT models (e.g., transformers models)
                try:
                    model_to_save.save_pretrained(str(final_model_dir), safe_serialization=True)
                except TypeError:
                    # Some older model implementations may not support safe_serialization
                    model_to_save.save_pretrained(str(final_model_dir))
                
                # Convert legacy adapter files if present (for non-PEFT adapter saves)
                _ensure_adapter_safetensors(str(final_model_dir))
            else:
                self.logger.warning("Model does not have save_pretrained method")
        except Exception as e:
            self.logger.error("Failed to save model directly: %s", e, exc_info=True)
            raise

    # ------------------------------------------------------------------
    def run(self) -> Any:
        """Run the complete training pipeline."""
        start_time = time.time()
        
        try:
            self.validate_config()
            
            # Check if model already exists and handle accordingly
            output_dir = getattr(self.cfg, "output_dir", "./")
            force_training = getattr(self.cfg, "force_training", False)
            
            if check_model_exists(output_dir) and not force_training:
                self.logger.info(
                    f"Training already completed for output directory: {output_dir}\n"
                    f"Found existing trained model. Skipping training to avoid overwriting.\n"
                    f"Use --force-training to retrain and overwrite the existing model."
                )
                return None
            elif check_model_exists(output_dir) and force_training:
                self.logger.warning(
                    f"Found existing trained model in {output_dir}, but force_training=True. "
                    f"Proceeding with training - this will overwrite the existing model."
                )
            
            self.set_seed()
            # Early progress logging so users see where long runs spend time
            try:
                self.logger.info(
                    "Initialising Weights & Biases: %s",
                    "enabled" if getattr(self.cfg, "use_wandb", False) else "disabled",
                )
            except Exception:
                pass
            self.init_wandb()
            
            # Execute pre-training hooks
            self.plugin_manager.execute_hook("pre_training", self.plugin_context)
            
            # Persist resolved configuration for reproducibility
            try:
                from dataclasses import asdict, is_dataclass
                resolved = asdict(self.cfg) if is_dataclass(self.cfg) else dict(self.cfg.__dict__)
                cfg_out = Path(output_dir) / "resolved_config.yaml"
                import yaml  # type: ignore
                cfg_out.parent.mkdir(parents=True, exist_ok=True)
                with cfg_out.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(resolved, f, sort_keys=False)
            except Exception:
                pass

            try:
                self.logger.info(
                    "Loading model and tokenizer: %s",
                    getattr(self.cfg, "model_name_or_path", "<unknown>"),
                )
            except Exception:
                pass
            self.load_model_tokenizer()
            datasets = self.prepare_datasets(self.cfg)
            
            # Update plugin context with datasets
            self.plugin_context.datasets = datasets
            
            try:
                self.logger.info(
                    "Preparing trainer and starting training on dataset: %s",
                    getattr(self.cfg, "dataset_name", "<unknown>"),
                )
            except Exception:
                pass
            self.trainer = self.build_trainer(self.model, self.tokenizer, self.cfg, datasets)
            
            # Update plugin context with trainer
            self.plugin_context.trainer = self.trainer
            

            
            # Detect zero-shot mode: if num_train_epochs == 0, skip training loop
            num_epochs = getattr(self.cfg, "num_train_epochs", 1)
            is_zero_shot = (num_epochs == 0)
            
            if is_zero_shot:
                try:
                    self.logger.info(
                        "Zero-shot mode: num_train_epochs=0. Skipping training loop. "
                        "Model (with LoRA adapters if configured) will be saved without training."
                    )
                except Exception:
                    pass
                result = None
            else:
                result = self.trainer.train()
            
            if self.use_wandb and wandb is not None:
                wandb.finish()
            
            return result
            
        finally:
            # Always finalize plugins
            self.plugin_manager.finalize_plugins()


__all__ = ["TrainingPipeline"]
