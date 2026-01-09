"""Supervised fine-tuning training pipeline."""

from __future__ import annotations

from transformers import AutoTokenizer
from transformers import PreTrainedModel

from prefadap.data import (
    DataCollatorWithLabelPaddingWithSide,
    get_lm_datasets,
)

from .config import SFTArgs
from .pipeline_base import TrainingPipeline
from .registry import register_adaptation

from trl import SFTConfig, SFTTrainer


from .utils import (
    TokenKLCallback,
    build_trl_config,
    dump_indices,
    _load_reference_model,
)


@register_adaptation("sft")
class SFTTrainingPipeline(TrainingPipeline):
    """Wrapper around :class:`trl.SFTTrainer` handling boilerplate."""

    def prepare_datasets(self, training_args: SFTArgs):
        self.logger.info(f"Loading dataset: {training_args.dataset_name}")
        ds, train_idx, eval_idx = get_lm_datasets(
            training_args, self.tokenizer, mode="sft"
        )
        dump_indices(
            training_args.save_indices, train_idx, eval_idx, training_args.output_dir
        )
        return ds

    def build_trainer(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        training_args: SFTArgs,
        datasets,
    ):
        collator = DataCollatorWithLabelPaddingWithSide(tokenizer, padding_side="left")
        hf_args = build_trl_config(
            training_args,
            SFTConfig,
            eval_strategy="no",
        )
        ref_model = _load_reference_model(model, training_args.log_kl)
        try:
            from . import training_wrappers as _tw  # lazy import

            trainer_cls = getattr(_tw, "SFTTrainer", SFTTrainer)
        except Exception:
            trainer_cls = SFTTrainer

        trainer = trainer_cls(
            model=model,
            args=hf_args,
            train_dataset=datasets["train"],
            processing_class=tokenizer,
            data_collator=collator,
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

__all__ = ["SFTTrainingPipeline"]
