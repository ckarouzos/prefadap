"""KTO training pipeline."""

from __future__ import annotations

from typing import Iterable, Tuple

from transformers import AutoTokenizer
from transformers import PreTrainedModel
from trl import KTOConfig

from .trl_compat import KTOTrainer

from .config import KTOArgs
from .pipeline_base import TrainingPipeline
from .utils import (
    TokenKLCallback,
    build_trl_config,
    dump_indices,
    resolve_optional_pseudo_data_path,
)
from prefadap.data import get_lm_datasets


def auto_balance_kto_loss_weights(
    train_dataset,
    desirable_weight: float,
    undesirable_weight: float,
    logger=None,
) -> Tuple[float, float]:
    """Return balanced KTO loss weights using dataset label counts.

    The TRL ``KTOTrainer`` warns when the desirable/undesirable weights are not
    scaled in line with the dataset's label distribution.  This helper inspects
    the ``label`` column of ``train_dataset`` (interpreting truthy values as
    desirable examples, e.g. ``True``/``1``) and rescales the weights so that
    the minority class receives proportionally higher loss weight, keeping the
    effective contribution of both classes comparable.

    When one of the label buckets is empty we fall back to zeroing out the
    corresponding weight.  If the dataset does not expose ``label`` (or is
    empty) the original weights are returned untouched.
    """

    labels: Iterable[int | bool]
    try:
        labels = train_dataset["label"]  # type: ignore[index]
    except Exception:
        if logger is not None:
            logger.warning(
                "Auto balancing requested but train dataset has no 'label' column; "
                "keeping provided KTO loss weights.",
            )
        return desirable_weight, undesirable_weight

    total = 0
    desirable_count = 0
    for value in labels:
        total += 1
        if bool(value):
            desirable_count += 1

    if total == 0:
        if logger is not None:
            logger.warning(
                "Auto balancing requested but train dataset contains no examples; "
                "keeping provided KTO loss weights.",
            )
        return desirable_weight, undesirable_weight

    undesirable_count = total - desirable_count

    if desirable_count == 0:
        if logger is not None:
            logger.warning(
                "KTO dataset has zero desirable examples; setting desirable_weight to 0.0.",
            )
        return 0.0, float(undesirable_weight)

    if undesirable_count == 0:
        if logger is not None:
            logger.warning(
                "KTO dataset has zero undesirable examples; setting undesirable_weight to 0.0.",
            )
        return float(desirable_weight), 0.0

    if desirable_count == undesirable_count:
        return float(desirable_weight), float(undesirable_weight)

    new_desirable = float(desirable_weight)
    new_undesirable = float(undesirable_weight)

    if desirable_count > undesirable_count:
        new_undesirable *= desirable_count / undesirable_count
    else:
        new_desirable *= undesirable_count / desirable_count

    if logger is not None:
        logger.info(
            "Auto-balanced KTO loss weights using %d desirable and %d undesirable examples: "
            "desirable_weight=%.4f, undesirable_weight=%.4f",
            desirable_count,
            undesirable_count,
            new_desirable,
            new_undesirable,
        )

    return new_desirable, new_undesirable


class KTOTrainingPipeline(TrainingPipeline):
    """Wrapper around :class:`trl.KTOTrainer`."""

    def prepare_datasets(self, training_args: KTOArgs):
        pseudo_path = resolve_optional_pseudo_data_path(
            getattr(training_args, "pseudo_data_path", None)
        )
        if pseudo_path is not None:
            self.logger.info(f"Loading pseudo data from: {pseudo_path}")
        else:
            self.logger.info(f"Loading dataset: {training_args.dataset_name}")
        ds, train_idx, eval_idx = get_lm_datasets(
            training_args, self.tokenizer, mode="kto"
        )
        dump_indices(
            training_args.save_indices, train_idx, eval_idx, training_args.output_dir
        )
        return ds

    def build_trainer(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        training_args: KTOArgs,
        datasets,
    ):
        from .utils import load_reference_model_with_validation
        
        # Use centralized reference model loading with GPU validation
        model_name = getattr(training_args, "model_name_or_path", "")
        use_lora = getattr(training_args, "lora", False)
        shared_backbone = getattr(training_args, "shared_backbone", False)
        
        ref_model = load_reference_model_with_validation(
            model=model,
            model_name_or_path=model_name,
            experiment_type="kto",
            use_lora=use_lora,
            shared_backbone=shared_backbone,
            enabled=training_args.log_kl,
        )
        desirable_weight = training_args.desirable_weight
        undesirable_weight = training_args.undesirable_weight

        if getattr(training_args, "auto_balance_kto_weights", False):
            desirable_weight, undesirable_weight = auto_balance_kto_loss_weights(
                datasets["train"], desirable_weight, undesirable_weight, self.logger
            )
            training_args.desirable_weight = desirable_weight
            training_args.undesirable_weight = undesirable_weight

        config = build_trl_config(
            training_args,
            KTOConfig,
            beta=training_args.beta,
            gamma=training_args.gamma,
            tau=training_args.tau,
            desirable_weight=desirable_weight,
            undesirable_weight=undesirable_weight,
            # TRL warns to set this explicitly with KTO/DPO-style collators
            remove_unused_columns=False,
            ref_model_device="cpu",  # Move reference model to CPU to save GPU memory
        )
        trainer = KTOTrainer(
            model=model,
            ref_model=ref_model,
            args=config,
            train_dataset=datasets["train"],
            data_collator=None,
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


__all__ = ["KTOTrainingPipeline"]
