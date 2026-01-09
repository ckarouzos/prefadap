"""Reward model training pipeline."""

from __future__ import annotations

from peft import PeftModel
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from trl import RewardConfig, RewardTrainer

from prefadap.data import get_lm_datasets
from prefadap.models.tokenization import ensure_pad_token

from .config import RMArgs
from .pipeline_base import TrainingPipeline
from .utils import TokenKLCallback, build_trl_config, dump_indices, _load_reference_model


class RewardModelTrainingPipeline(TrainingPipeline):
    """Wrapper around :class:`trl.RewardTrainer`."""

    def prepare_datasets(self, training_args: RMArgs):
        self.logger.info(f"Loading dataset: {training_args.dataset_name}")
        ds, train_idx, eval_idx = get_lm_datasets(
            training_args, self.tokenizer, mode="dpo"
        )
        dump_indices(
            training_args.save_indices, train_idx, eval_idx, training_args.output_dir
        )
        return ds

    def build_trainer(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        training_args: RMArgs,
        datasets,
    ):
        if getattr(model.config, "num_labels", None) != 1:
            raise ValueError(
                "Reward model must be scalar (num_labels=1). "
                f"Classification-style reward models are not supported (got {model.config.num_labels})."
            )
        if getattr(model.config, "problem_type", None) not in (None, "regression"):
            raise ValueError(
                "Reward model must use regression heads (problem_type='regression'). "
                f"Received problem_type={model.config.problem_type}."
            )
        ref_model = _load_reference_model(model, enabled=training_args.log_kl)
        
        # Ensure tokenizer has a pad_token (required by TRL)
        ensure_pad_token(tokenizer)
        config = build_trl_config(training_args, RewardConfig)

        trainer = RewardTrainer(
            model=model,
            args=config,
            train_dataset=datasets["train"],
            eval_dataset=datasets.get("eval"),
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

    # Inherit base save_artifacts: save to output_dir/final_model and execute hooks.
    def save_artifacts(self) -> None:
        """Save reward model artifacts with a fully loadable HF model directory."""
        model = self.model
        if isinstance(model, PeftModel):
            setattr(model, "_prefadap_merge_lora_on_save", True)
        super().save_artifacts()


__all__ = ["RewardModelTrainingPipeline"]
