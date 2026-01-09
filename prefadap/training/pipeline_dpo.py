"""Direct Preference Optimisation training pipeline."""

from __future__ import annotations

from transformers import AutoTokenizer
from transformers import PreTrainedModel
from trl import DPOConfig, DPOTrainer

from .config import DPOArgs
from .pipeline_base import TrainingPipeline
from .registry import register_adaptation
from .utils import (
    TokenKLCallback,
    build_trl_config,
    dump_indices,
    resolve_optional_pseudo_data_path,
)
from prefadap.data import DPODataCollator, get_lm_datasets, make_dpo_dataset
from prefadap.models.loader import _resolve_model_path


@register_adaptation("dpo")
class DPOTrainingPipeline(TrainingPipeline):
    """Wrapper around :class:`trl.DPOTrainer`."""

    def prepare_datasets(self, training_args: DPOArgs):
        pseudo_path = resolve_optional_pseudo_data_path(
            getattr(training_args, "pseudo_data_path", None)
        )
        if pseudo_path is not None:
            # Pseudo-data path uses the higher-level loader, which handles
            # tokenisation and split selection.
            self.logger.info(f"Loading pseudo data from: {pseudo_path}")
            ds, train_idx, eval_idx = get_lm_datasets(
                training_args, self.tokenizer, mode="dpo"
            )
            dump_indices(
                training_args.save_indices, train_idx, eval_idx, training_args.output_dir
            )
            return ds

        # For regular datasets, call a module-local make_dpo_dataset symbol so
        # tests can monkeypatch it at this import site. This also allows
        # passing through pre-tokenised datasets without extra processing.
        self.logger.info(f"Loading dataset: {training_args.dataset_name}")
        # Call without keyword-only args so monkeypatched lambdas in tests
        # that accept (args, tokenizer) still work.
        ds_dict, train_idx, eval_idx = make_dpo_dataset(
            training_args, self.tokenizer
        )
        dump_indices(
            training_args.save_indices, train_idx, eval_idx, training_args.output_dir
        )
        eval_split = "validation" if "validation" in ds_dict else ("test" if "test" in ds_dict else None)
        ds = {"train": ds_dict["train"], "eval": ds_dict[eval_split] if eval_split else None}
        return ds

    def build_trainer(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        training_args: DPOArgs,
        datasets,
    ):
        from .utils import load_reference_model_with_validation
        
        # Load reference model with centralized GPU validation
        model_name = getattr(training_args, "model_name_or_path", "")
        use_lora = getattr(training_args, "lora", False)
        shared_backbone = getattr(training_args, "shared_backbone", False)
        
        # Handle explicit reference model path
        ref_path = getattr(training_args, "ref_model_name_or_path", None)
        if ref_path:
            try:
                from .training_wrappers import load_model_and_tokenizer  # lazy import
                resolved_ref = _resolve_model_path(
                    ref_path,
                    seed=getattr(training_args, "seed", None),
                    stage=getattr(training_args, "stage", None),
                )
                ref_model, _ = load_model_and_tokenizer(
                    resolved_ref, gradient_checkpointing=False
                )
                for p in ref_model.parameters():
                    p.requires_grad = False
                self.logger.info(
                    f"Loaded reference model from: {resolved_ref}"
                )
            except Exception as exc:
                self.logger.warning(f"Failed to load explicit reference model at {ref_path}: {exc}. Falling back to cloned reference.")
                ref_model = load_reference_model_with_validation(
                    model=model,
                    model_name_or_path=model_name,
                    experiment_type="dpo",
                    use_lora=use_lora,
                    shared_backbone=shared_backbone,
                    enabled=True,
                )
        else:
            # Use centralized reference model loading with validation
            ref_model = load_reference_model_with_validation(
                model=model,
                model_name_or_path=model_name,
                experiment_type="dpo",
                use_lora=use_lora,
                shared_backbone=shared_backbone,
                enabled=True,
            )
        collator = DPODataCollator(
            tokenizer,
            padding_side="left",
            pad_to_multiple_of=8,
            max_prompt_length=training_args.max_source_length,
            max_completion_length=training_args.max_target_length,
        )
        config = build_trl_config(
            training_args,
            DPOConfig,
            beta=training_args.beta,
            ref_model_device="cpu",  # Move reference model to CPU to save GPU memory
        )

        try:  # local import to avoid circulars at module import time
            from . import training_wrappers as _tw  # type: ignore

            trainer_cls = getattr(_tw, "DPOTrainer", DPOTrainer)
        except Exception:
            trainer_cls = DPOTrainer
        trainer = trainer_cls(
            model=model,
            ref_model=ref_model,
            args=config,
            train_dataset=datasets["train"],
            data_collator=collator,
            processing_class=tokenizer,
        )
        if training_args.log_token_count or training_args.log_kl:
            trainer.add_callback(
                TokenKLCallback(
                    log_token_count=training_args.log_token_count,
                    log_kl=training_args.log_kl,
                    use_wandb=training_args.use_wandb,
                    ref_model=ref_model,
                    logger=self.logger,
                    output_dir=training_args.output_dir,
                    seed=training_args.seed,
                    metric_logging_steps=getattr(training_args, "metric_logging_steps", 50),
                    system_logging_steps=getattr(training_args, "system_logging_steps", 500),
                )
            )
        return trainer

__all__ = ["DPOTrainingPipeline"]
