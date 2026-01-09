"""Proximal Policy Optimization training pipeline."""

from __future__ import annotations

from dataclasses import asdict

import os

from transformers import AutoTokenizer, PreTrainedModel
from trl import PPOConfig, PPOTrainer

from prefadap.data import get_lm_datasets
from prefadap.models.loader import load_reward_model_and_tokenizer

from .config import PPOArgs
from .pipeline_base import TrainingPipeline
from .utils import (
    TokenKLCallback,
    dump_indices,
    ensure_trl_policy_value_wrapper_gradient_checkpointing,
    build_ppo_model_load_config,
    resolve_trl_output_dir,
    log_liger_kernel_status,
)


class PPOTrainingPipeline(TrainingPipeline):
    """Wrapper around :class:`trl.PPOTrainer` for RLHF."""

    def prepare_datasets(self, training_args: PPOArgs):
        self.logger.info("Loading dataset: %s", training_args.dataset_name)
        ds, train_idx, eval_idx = get_lm_datasets(training_args, self.tokenizer, mode="rl")
        dump_indices(training_args.save_indices, train_idx, eval_idx, training_args.output_dir)
        return ds

    def build_trainer(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        training_args: PPOArgs,
        datasets,
    ):


        from .utils import load_reference_model_with_validation

        model_name = getattr(training_args, "model_name_or_path", "")
        use_lora = bool(getattr(training_args, "lora", False))
        shared_backbone = bool(getattr(training_args, "shared_backbone", False))

        log_liger_kernel_status(training_args.use_liger_kernel, self.logger)

        eval_dataset = datasets.get("eval")
        if eval_dataset is None:
            # PPO evaluation is controlled by dataset presence.
            eval_dataset = datasets["train"].select([])


        if use_lora:
            ref_model = None
        else:
            ref_model = load_reference_model_with_validation(
                model=model,
                model_name_or_path=model_name,
                experiment_type="ppo",
                use_lora=use_lora,
                shared_backbone=shared_backbone,
                enabled=True,
                device=getattr(training_args, "device", "cpu"),
            )

        ensure_trl_policy_value_wrapper_gradient_checkpointing()

        rm_load_cfg = build_ppo_model_load_config(getattr(training_args, "device", None))
        reward_model, _ = load_reward_model_and_tokenizer(
            training_args.reward_model_name_or_path,
            gradient_checkpointing=False,
            config=rm_load_cfg,
        )
        value_model, _ = load_reward_model_and_tokenizer(
            training_args.reward_model_name_or_path,
            gradient_checkpointing=False,
            config=rm_load_cfg,
        )
        if value_model.config.num_labels != 1:
            raise ValueError(
                "PPO/GRPO value model must have num_labels=1 "
                f"(got {value_model.config.num_labels})."
            )
        value_model.requires_grad_(False)
        value_model.eval()

        trainer_output_dir = resolve_trl_output_dir(training_args.output_dir, self.logger)

        rl_shared = training_args.rl.shared
        rl_ppo = training_args.rl.ppo
        if rl_ppo is None:
            raise ValueError("PPO configs must define rl.ppo.")

        ds3_gather_for_generation = bool(rl_shared.ds3_gather_for_generation)

        if not ds3_gather_for_generation:
            if ref_model is not None:
                self.logger.info(
                    "ds3_gather_for_generation disabled: disabling reference model to avoid extra memory pressure."
                )
            ref_model = None

        # ----------------------------
        # Build PPOConfig (INPUTS ONLY)
        # ----------------------------
        config_kwargs = {
            "output_dir": trainer_output_dir,
            "learning_rate": training_args.learning_rate,

            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,

            # PPO epochs (prefer explicit ppo_epochs if present)
            "num_ppo_epochs": getattr(training_args, "ppo_epochs", training_args.num_train_epochs),

            # PPO rollout + minibatching inputs
            "response_length": rl_shared.response_length,
            "local_rollout_forward_batch_size": rl_shared.local_rollout_forward_batch_size,
            "num_mini_batches": rl_shared.num_mini_batches,

            # Generation inputs
            "temperature": rl_shared.temperature,
            "stop_token": rl_shared.stop_token,

            # ZeRO-3 gather for generation
            "ds3_gather_for_generation": ds3_gather_for_generation,

            # PPO loss / reward options
            "kl_coef": rl_ppo.kl_coef,
            "cliprange": rl_ppo.cliprange,
            "cliprange_value": rl_ppo.cliprange_value,
            "vf_coef": rl_ppo.vf_coef,
            "whiten_rewards": rl_ppo.whiten_rewards,

            "report_to": ["wandb"] if training_args.use_wandb else [],

            # Explicit stability defaults for PPO/GRPO on H100s.
            "bf16": True,
            "fp16": False,
            "use_liger_kernel": bool(getattr(training_args, "use_liger_kernel", False)),
        }


        config = PPOConfig(**config_kwargs)

        config.target_kl = rl_ppo.target_kl
        config.top_p = rl_shared.top_p  # TRL uses this for generation if wired internally
        if rl_ppo.kl_estimator is not None and hasattr(config, "kl_estimator"):
            config.kl_estimator = rl_ppo.kl_estimator
        if rl_ppo.gamma is not None and hasattr(config, "gamma"):
            config.gamma = rl_ppo.gamma
        if rl_ppo.lam is not None and hasattr(config, "lam"):
            config.lam = rl_ppo.lam
        if getattr(training_args, "max_grad_norm", None) is not None:
            config.max_grad_norm = training_args.max_grad_norm

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        pbs = int(getattr(config, "per_device_train_batch_size", 0) or 0)
        gas = int(getattr(config, "gradient_accumulation_steps", 1) or 1)
        nmb = int(getattr(config, "num_mini_batches", 1) or 1)

        local_batch_size = pbs * gas
        local_mini_batch_size = local_batch_size // max(1, nmb)

        self.logger.info(
            "TRL PPO sizing: per_device_train_batch_size=%d gradient_accumulation_steps=%d "
            "num_mini_batches=%d world_size=%d => local_batch_size=%d local_mini_batch_size=%d",
            pbs, gas, nmb, world_size, local_batch_size, local_mini_batch_size
        )

        if bool(getattr(config, "whiten_rewards", False)) and local_mini_batch_size < 8:
            raise ValueError(
                "Invalid PPO config: whiten_rewards=True requires local_mini_batch_size >= 8.\n"
                f"Computed local_mini_batch_size={local_mini_batch_size} from "
                f"per_device_train_batch_size={pbs}, gradient_accumulation_steps={gas}, "
                f"num_mini_batches={nmb}.\n"
                "Fix by: (a) decreasing num_mini_batches, (b) increasing per_device_train_batch_size "
                "or gradient_accumulation_steps, or (c) setting whiten_rewards=False."
            )

        effective_ppo_config = {
            "shared": asdict(rl_shared),
            "ppo": asdict(rl_ppo),
            "trl": {
                "per_device_train_batch_size": getattr(config, "per_device_train_batch_size", None),
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "num_mini_batches": config.num_mini_batches,
                "response_length": config.response_length,
                "local_rollout_forward_batch_size": config.local_rollout_forward_batch_size,
                "temperature": config.temperature,
                "top_p": getattr(config, "top_p", None),
                "stop_token": config.stop_token,
                "ds3_gather_for_generation": config.ds3_gather_for_generation,
                "num_ppo_epochs": config.num_ppo_epochs,
                "target_kl": getattr(config, "target_kl", None),
                "kl_coef": config.kl_coef,
                "cliprange": config.cliprange,
                "cliprange_value": config.cliprange_value,
                "vf_coef": config.vf_coef,
                "whiten_rewards": config.whiten_rewards,
                "max_grad_norm": getattr(config, "max_grad_norm", None),
                "bf16": config.bf16,
                "fp16": config.fp16,
                "kl_estimator": getattr(config, "kl_estimator", None),
                "gamma": getattr(config, "gamma", None),
                "lam": getattr(config, "lam", None),
            },
        }
        self.logger.info("effective_ppo_config=%s", effective_ppo_config)

        trainer = PPOTrainer(
            args=config,
            model=model,
            ref_model=ref_model,
            processing_class=tokenizer,
            reward_model=reward_model,
            train_dataset=datasets["train"],
            eval_dataset=eval_dataset,
            value_model=value_model,
        )

        if training_args.log_token_count or training_args.log_kl:
            try:
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


__all__ = ["PPOTrainingPipeline"]
