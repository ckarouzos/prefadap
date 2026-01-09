"""ORPO training pipeline."""

from __future__ import annotations

from transformers import AutoTokenizer
from transformers import PreTrainedModel
from trl import ORPOConfig, ORPOTrainer

from prefadap.data.dpo import make_dpo_raw_text_dataset, make_pseudo_raw_text_dataset, make_dpo_samples
from prefadap.data.indexing import dataset_with_indices
from prefadap.models.loader import _resolve_model_path

from .config import ORPOArgs
from .pipeline_base import TrainingPipeline
from .utils import (
    TokenKLCallback,
    build_trl_config,
    dump_indices,
    resolve_optional_pseudo_data_path,
)


class ORPOTrainingPipeline(TrainingPipeline):
    """Wrapper around :class:`trl.ORPOTrainer`."""

    def prepare_datasets(self, training_args: ORPOArgs):
        pseudo_path = resolve_optional_pseudo_data_path(
            getattr(training_args, "pseudo_data_path", None)
        )
        if pseudo_path is not None:
            self.logger.info(f"Loading pseudo data from: {pseudo_path}")
            ds = make_pseudo_raw_text_dataset(training_args.pseudo_data_path)
            ds = ds.map(make_dpo_samples, batched=True)
            ds, train_idx, eval_idx = dataset_with_indices(ds)
        else:
            self.logger.info(f"Loading dataset: {training_args.dataset_name}")
            ds = make_dpo_raw_text_dataset(training_args)
            ds, train_idx, eval_idx = dataset_with_indices(ds)
        
        dump_indices(
            training_args.save_indices, train_idx, eval_idx, training_args.output_dir
        )
        return ds

    def build_trainer(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        training_args: ORPOArgs,
        datasets,
    ):
        from .utils import load_reference_model_with_validation
        
        ref_model = None
        if training_args.log_kl:
            model_name = getattr(training_args, "model_name_or_path", "")
            use_lora = getattr(training_args, "lora", False)
            shared_backbone = getattr(training_args, "shared_backbone", False)
            
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
                        experiment_type="orpo",
                        use_lora=use_lora,
                        shared_backbone=shared_backbone,
                        enabled=True,
                    )
            else:
                ref_model = load_reference_model_with_validation(
                    model=model,
                    model_name_or_path=model_name,
                    experiment_type="orpo",
                    use_lora=use_lora,
                    shared_backbone=shared_backbone,
                    enabled=True,
                )

        config = build_trl_config(
            training_args,
            ORPOConfig,
            beta=training_args.beta,
            ref_model_device="cpu",  # Move reference model to CPU to save GPU memory
        )
        trainer = ORPOTrainer(
            model=model,
            args=config,
            train_dataset=datasets["train"],
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

__all__ = ["ORPOTrainingPipeline"]
