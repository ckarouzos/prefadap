"""GRPO training pipeline built on top of GRPOTrainer.

Implements Group Relative Preference Optimisation using the
:class:`trl.GRPOTrainer`.  The pipeline is typically dispatched via
``python -m prefadap.cli.run_training grpo --config <config.yaml>``.
"""

from __future__ import annotations

from dataclasses import asdict

from transformers import AutoTokenizer
from transformers import PreTrainedModel
from trl import GRPOConfig, GRPOTrainer

from prefadap.data import get_grpo_datasets
from prefadap.models.loader import load_reward_model_and_tokenizer

from .config import GRPOArgs
from .pipeline_base import TrainingPipeline
from .trl_rollout_sync import enforce_single_rank_rollout
from .utils import (
    TokenKLCallback,
    dump_indices,
    build_ppo_model_load_config,
    resolve_trl_output_dir,
    is_deepspeed_zero3_enabled,
    resolve_rollout_forward_batch_size,
    log_liger_kernel_status,
)


class GRPOTrainingPipeline(TrainingPipeline):
    """Simplified GRPO training built on :class:`trl.GRPOTrainer`."""

    def prepare_datasets(self, training_args: GRPOArgs):
        """Load and prepare the training and evaluation datasets.

        Parameters
        ----------
        training_args:
            Configuration dataclass describing the GRPO run.

        Returns
        -------
        datasets.DatasetDict
            Raw dataset with ``train`` and ``eval`` splits containing ``prompt`` fields.

        Side Effects
        ------------
        When ``training_args.save_indices`` is true, index files are written
        to ``training_args.output_dir``.
        """

        self.logger.info(f"Loading dataset: {training_args.dataset_name}")
        ds, train_idx, eval_idx = get_grpo_datasets(training_args, self.tokenizer)
        if "train" in ds:
            self.logger.info(
                "GRPO dataset columns (train): %s", ds["train"].column_names
            )
            if len(ds["train"]) > 0:
                sample = ds["train"][0]
                prompt_value = sample.get("prompt")
                preview = prompt_value
                if isinstance(prompt_value, list):
                    preview = prompt_value[0] if prompt_value else ""
                if isinstance(preview, dict):
                    preview = preview.get("content", preview)
                self.logger.info("GRPO train[0] type: %s", type(sample))
                self.logger.info(
                    "GRPO train[0]['prompt'] type: %s", type(prompt_value)
                )
                self.logger.info(
                    "GRPO train[0]['prompt'] preview: %r",
                    str(preview)[:200],
                )
            reward_columns = getattr(training_args, "reward_kwargs_columns", None)
            if reward_columns:
                missing = [
                    column
                    for column in reward_columns
                    if column not in ds["train"].column_names
                ]
                if missing:
                    raise ValueError(
                        f"GRPO dataset is missing reward kwargs columns: {missing}"
                    )
        dump_indices(
            training_args.save_indices, train_idx, eval_idx, training_args.output_dir
        )
        return ds

    def build_trainer(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        training_args: GRPOArgs,
        datasets,
    ):
        """Instantiate a :class:`trl.GRPOTrainer` configured for GRPO.

        Parameters
        ----------
        model:
            Actor model to be optimised.
        tokenizer:
            Tokeniser corresponding to ``model``.
        training_args:
            GRPO hyperparameters and runtime configuration.
        datasets:
            Dataset dictionary as returned by :meth:`prepare_datasets`.

        Returns
        -------
        GRPOTrainer
            Configured trainer ready for :meth:`~trl.GRPOTrainer.train`.

        Side Effects
        ------------
        May register a :class:`TokenKLCallback` when logging options are
        enabled in ``training_args``.
        """

        model_name = getattr(training_args, "model_name_or_path", "")

        log_liger_kernel_status(training_args.use_liger_kernel, self.logger)

        eval_dataset = datasets.get("eval")
        if eval_dataset is None:
            eval_dataset = datasets["train"].select([])

        zero3_enabled = is_deepspeed_zero3_enabled()
        reward_model, reward_tokenizer = load_reward_model_and_tokenizer(
            training_args.reward_model_name_or_path,
            gradient_checkpointing=False,
            config=build_ppo_model_load_config(getattr(training_args, "device", None)),
        )
        trainer_output_dir = resolve_trl_output_dir(training_args.output_dir, self.logger)
        rl_shared = training_args.rl.shared
        rl_grpo = training_args.rl.grpo
        if rl_grpo is None:
            raise ValueError("GRPO configs must define rl.grpo.")
        ds3_gather_for_generation = rl_shared.ds3_gather_for_generation
        if not ds3_gather_for_generation:
            rollout_forward_batch_size = rl_shared.local_rollout_forward_batch_size
            self.logger.info(
                "ds3_gather_for_generation disabled: enforcing single-rank rollout generation on rank 0."
            )
        elif zero3_enabled:
            rollout_forward_batch_size = 2
        else:
            rollout_forward_batch_size = resolve_rollout_forward_batch_size(
                model_name, rl_shared.local_rollout_forward_batch_size
            )
        if rollout_forward_batch_size != rl_shared.local_rollout_forward_batch_size:
            self.logger.info(
                "Overriding local_rollout_forward_batch_size from %s to %s",
                rl_shared.local_rollout_forward_batch_size,
                rollout_forward_batch_size,
            )
        rl_shared.local_rollout_forward_batch_size = rollout_forward_batch_size
        generation_kwargs = {"eos_token_id": tokenizer.eos_token_id}
        config_kwargs = {
            "output_dir": trainer_output_dir,
            "learning_rate": training_args.learning_rate,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "num_train_epochs": training_args.num_train_epochs,
            "num_generations": rl_grpo.num_generations,
            "max_completion_length": rl_grpo.max_completion_length,
            "temperature": rl_shared.temperature,
            "top_p": rl_shared.top_p,
            "beta": rl_grpo.beta,
            "epsilon": rl_grpo.epsilon,
            "scale_rewards": rl_grpo.scale_rewards,
            "ds3_gather_for_generation": ds3_gather_for_generation,
            "generation_kwargs": generation_kwargs,
            "report_to": ["wandb"] if training_args.use_wandb else [],
            "bf16": getattr(training_args, "bf16", None) or False,
            "fp16": getattr(training_args, "fp16", None) or False,
            "use_liger_kernel": training_args.use_liger_kernel,
        }
        config = GRPOConfig(**config_kwargs)
        config.ds3_gather_for_generation = ds3_gather_for_generation
        if getattr(training_args, "max_grad_norm", None) is not None:
            config.max_grad_norm = training_args.max_grad_norm
        effective_grpo_config = {
            "shared": asdict(rl_shared),
            "grpo": asdict(rl_grpo),
            "trl": {
                "num_train_epochs": config.num_train_epochs,
                "num_generations": rl_grpo.num_generations,
                "max_completion_length": rl_grpo.max_completion_length,
                "temperature": config.temperature,
                "top_p": rl_shared.top_p,
                "beta": config.beta,
                "epsilon": rl_grpo.epsilon,
                "scale_rewards": rl_grpo.scale_rewards,
                "ds3_gather_for_generation": ds3_gather_for_generation,
                "generation_kwargs": generation_kwargs,
                "max_grad_norm": getattr(config, "max_grad_norm", None),
                "bf16": config.bf16,
                "fp16": config.fp16,
            },
        }
        self.logger.info("effective_grpo_config=%s", effective_grpo_config)
        trainer = GRPOTrainer(
            args=config,
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_model,
            train_dataset=datasets["train"],
            eval_dataset=eval_dataset,
            reward_processing_classes=reward_tokenizer,
        )
        enforce_single_rank_rollout(trainer, self.logger)
        if training_args.log_token_count or training_args.log_kl:
            try:
                ref_model = getattr(trainer, "ref_model", None)
                trainer.add_callback(
                    TokenKLCallback(
                        log_token_count=training_args.log_token_count,
                        log_kl=training_args.log_kl,
                        use_wandb=training_args.use_wandb,
                        ref_model=ref_model,
                        logger=self.logger,
                        output_dir=trainer_output_dir,
                        seed=training_args.seed,
                        metric_logging_steps=getattr(training_args, "metric_logging_steps", 50),
                        system_logging_steps=getattr(training_args, "system_logging_steps", 500),
                    )
                )
            except AttributeError:
                pass
        return trainer


__all__ = ["GRPOTrainingPipeline"]
