"""Dataclass configurations for training pipelines.

These dataclasses replace the previous Pydantic models used for specifying
training options.  They capture fields shared across pipelines such as the
device to run on, random seeds and precision settings (``fp16``/``bf16``/``dtype``).

The classes mirror the earlier ``training.args`` module but use ``dataclasses``
    so that :func:`prefadap.config.load_config` can directly instantiate them from
YAML or JSON files without requiring Pydantic.

Enhanced with algorithm-specific defaults and configuration template support.
"""

"""
Note: We avoid postponed evaluation of annotations here to ensure that
type hints like Optional[str] are available to pydantic.TypeAdapter
without requiring forward reference resolution.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict, Literal
import os
import time
import logging

from .args import DEFAULTS, get_algorithm_parameter

logger = logging.getLogger(__name__)


@dataclass
class TrainingArgs:
    """Base training arguments shared across pipelines.
    
    Enhanced with support for algorithm-specific defaults and configuration templates.
    """

    model_name_or_path: str
    dataset_name: Optional[str] = None
    dataset: Optional[str] = None
    run_id: str = field(default_factory=lambda: str(int(time.time())))
    data_dir: str = field(default_factory=lambda: os.environ.get("DATA_DIR") or os.environ.get("DATA", "data"))
    output_dir: Optional[str] = None
    logging_dir: Optional[str] = None
    # `auto` defers device selection to runtime so configs generated on CPU-only
    # machines still leverage GPUs when available (including ROCm devices).
    device: str = "auto"
    
    # Training hyperparameters with configurable defaults
    num_train_epochs: int = 1
    per_device_train_batch_size: int = field(
        default_factory=lambda: DEFAULTS.get("batch", {}).get("micro", 1)
    )
    gradient_accumulation_steps: int = field(
        default_factory=lambda: DEFAULTS.get("batch", {}).get("gradient_accumulation", 1)
    )
    learning_rate: float = field(
        default_factory=lambda: DEFAULTS.get("lr", {}).get("learning_rate", 3e-5)
    )
    # Optimizer-related knobs exposed at top-level for simplicity
    weight_decay: float = field(
        default_factory=lambda: get_algorithm_parameter("optimization", "optimizer.weight_decay", 0.01)
    )
    lr_scheduler_type: str = field(
        default_factory=lambda: get_algorithm_parameter("optimization", "scheduler", "cosine")
    )
    warmup_ratio: float = field(
        default_factory=lambda: get_algorithm_parameter("optimization", "warmup_ratio", 0.02)
    )
    max_steps: Optional[int] = None
    
    # Logging and monitoring
    wandb_project: str = DEFAULTS.get("project", "training")
    logging_steps: int = field(
        default_factory=lambda: get_algorithm_parameter("logging", "steps", DEFAULTS.get("logging_steps", 100))
    )
    metric_logging_steps: int = field(
        default_factory=lambda: get_algorithm_parameter("logging", "metric_steps", 50)
    )
    system_logging_steps: int = field(
        default_factory=lambda: get_algorithm_parameter("logging", "system_steps", 500)
    )
    wandb_tags: Optional[List[str]] = field(
        default_factory=lambda: get_algorithm_parameter("logging", "wandb_tags", None)
    )
    save_steps: int = field(
        default_factory=lambda: get_algorithm_parameter("checkpointing", "save_steps", DEFAULTS.get("save_steps", 500))
    )
    save_total_limit: int = field(
        default_factory=lambda: get_algorithm_parameter("checkpointing", "save_total_limit", 2)
    )

    seed: int = DEFAULTS.get("seeds", [42])[0]
    use_wandb: bool = field(
        default_factory=lambda: get_algorithm_parameter("logging", "use_wandb", True)
    )
    
    # LoRA configuration with algorithm defaults
    lora: bool = False
    lora_r: int = field(
        default_factory=lambda: get_algorithm_parameter("lora", "r", DEFAULTS.get("lora", {}).get("r", 8))
    )
    lora_alpha: int = field(
        default_factory=lambda: get_algorithm_parameter("lora", "alpha", DEFAULTS.get("lora", {}).get("alpha", 32))
    )
    lora_dropout: float = field(
        default_factory=lambda: get_algorithm_parameter("lora", "dropout", 0.1)
    )
    lora_target_modules: str | list[str] | None = field(
        default_factory=lambda: get_algorithm_parameter("lora", "target_modules", "q_proj,v_proj")
    )
    lora_on_peft: Optional[str] = field(
        default_factory=lambda: get_algorithm_parameter("lora", "on_peft", "merge_then_new")
    )
    freeze_embeddings: bool = field(
        default_factory=lambda: get_algorithm_parameter("lora", "freeze_embeddings", False)
    )
    # Additionally freeze normalisation layers for stability across methods
    freeze_norm_layers: bool = True
    
    # Training strategy parameters  
    dataset_structured_subset: str = ""
    dataset_random_subset: Optional[float] = None
    gradient_checkpointing: bool = field(
        default_factory=lambda: get_algorithm_parameter("training_strategy", "gradient_checkpointing", False)
    )
    tune_large_defaults: bool = False
    fsdp: bool = False
    force_training: bool = False
    
    # Monitoring and logging flags
    log_token_count: bool = field(
        default_factory=lambda: get_algorithm_parameter("logging", "log_token_count", False)
    )
    log_kl: bool = field(
        default_factory=lambda: get_algorithm_parameter("logging", "log_kl", False)  
    )
    save_indices: bool = field(
        default_factory=lambda: get_algorithm_parameter("logging", "save_indices", False)
    )
    
    # Precision settings
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None
    dtype: Optional[str] = None
    max_grad_norm: Optional[float] = None
    # TRL performance toggle
    use_liger_kernel: bool = True
    
    # Tokenizer override
    tokenizer_name_or_path: Optional[str] = None
    
    # Algorithm-specific configuration override
    algorithm_config: Optional[Dict[str, Any]] = field(default=None)
    config_template: Optional[str] = field(default=None)

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        # Handle explicit None values for optional directories
        if self.output_dir is None:
            self.output_dir = os.path.join(
                os.environ.get("SCRATCH", "runs"), self.run_id
            )
        if self.logging_dir is None:
            self.logging_dir = self.output_dir
            
        # Load algorithm-specific configuration if provided
        if self.algorithm_config:
            self._apply_algorithm_config(self.algorithm_config)
    
    def _apply_algorithm_config(self, config: Dict[str, Any]) -> None:
        """Apply algorithm-specific configuration overrides."""
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
    
    def get_algorithm_default(self, algorithm: str, param_path: str, fallback: Any = None) -> Any:
        """Get algorithm-specific default for this configuration."""
        return get_algorithm_parameter(algorithm, param_path, fallback)


@dataclass(kw_only=True)
class RLSharedConfig:
    """Shared RL rollout/generation configuration for PPO and GRPO."""

    response_length: int
    local_rollout_forward_batch_size: int
    num_mini_batches: int
    ds3_gather_for_generation: bool
    temperature: float
    top_p: float
    stop_token: Literal["eos"]


@dataclass(kw_only=True)
class RLPPOConfig:
    """PPO-specific RL configuration mirroring TRL PPOConfig fields."""

    target_kl: float
    kl_coef: float
    cliprange: float
    cliprange_value: float
    vf_coef: float
    whiten_rewards: bool
    kl_estimator: Optional[str] = None
    gamma: Optional[float] = None
    lam: Optional[float] = None


@dataclass(kw_only=True)
class RLGRPOConfig:
    """GRPO-specific RL configuration mirroring TRL GRPOConfig fields."""

    num_generations: int
    max_completion_length: int
    beta: float
    epsilon: float
    scale_rewards: Literal["group"]


@dataclass(kw_only=True)
class RLConfig:
    """Container for RL configuration blocks."""

    shared: RLSharedConfig
    ppo: RLPPOConfig | None = None
    grpo: RLGRPOConfig | None = None


@dataclass(kw_only=True)
class SFTArgs(TrainingArgs):
    """Arguments for SFT training."""

    max_source_length: int = 512
    max_target_length: int = 48
    sequence: Optional[List[str]] = None
    mix_ratios: Optional[List[float]] = None
    pseudo_data_path: Optional[str] = None
    pseudo_weight: float = 1.0


@dataclass(kw_only=True)
class DAPTArgs(TrainingArgs):
    """Arguments for domain-adaptive pretraining."""

    max_seq_length: int = 512
    perplexity_threshold: float | None = None


@dataclass(kw_only=True)
class GRPOArgs(TrainingArgs):
    """Arguments for GRPO-based RLHF training.
    
    Enhanced with configurable algorithm parameters from algorithm_defaults.yaml
    """

    reward_model_name_or_path: str
    value_model_name_or_path: Optional[str] = None
    max_source_length: int = 512
    max_target_length: int = 48
    per_device_train_batch_size: int = 2
    rl: RLConfig
    gradient_accumulation_steps: int = field(
        default_factory=lambda: DEFAULTS.get("batch", {}).get("gradient_accumulation", 32) * 2
    )
    
    # PPO/GRPO algorithm parameters with configurable defaults
    beta: float = 0.1  # GRPO-specific beta parameter
    ppo_epochs: int = field(
        default_factory=lambda: get_algorithm_parameter("ppo", "epochs", 1)
    )
    vf_coef: float = field(
        default_factory=lambda: get_algorithm_parameter("ppo", "vf_coef", 0.5)
    )
    entropy_coef: float = field(
        default_factory=lambda: get_algorithm_parameter("ppo", "entropy_coef", 0.01)
    )
    clip_range: float = field(
        default_factory=lambda: get_algorithm_parameter("ppo", "clip_range", 0.2)
    )
    
    # GRPO-specific parameters
    group_size: int = 4
    group_field: Optional[str] = None
    fairness_metric: str = "min_ratio"

    pseudo_data_path: Optional[str] = None
    pseudo_weight: float = 1.0
    grpo_single_gpu_baseline: bool = False

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        super().__post_init__()
        if self.value_model_name_or_path is None:
            self.value_model_name_or_path = self.reward_model_name_or_path
        if self.rl is None or self.rl.grpo is None:
            raise ValueError("GRPO configs must define rl.shared and rl.grpo blocks.")



@dataclass(kw_only=True)
class OnlineDPOArgs(TrainingArgs):
    """Arguments for online DPO training."""

    reward_model_name_or_path: str
    dataset_name: str
    max_source_length: int = 512
    max_target_length: int = 48
    per_device_train_batch_size: int = 2
    learning_rate: float = 1e-5
    beta: float = 0.1
    target_kl: float = 0.1
    num_responses: int = 2
    pseudo_data_path: Optional[str] = None
    pseudo_weight: float = 1.0


@dataclass(kw_only=True)
class DPOArgs(TrainingArgs):
    """Arguments for DPO training.
    
    Enhanced with configurable algorithm parameters from algorithm_defaults.yaml
    """

    pseudo_data_path: Optional[str] = None
    pseudo_weight: float = 1.0
    max_source_length: int = 512
    max_target_length: int = 48
    # Optional explicit reference policy path (e.g., Stage-1 checkpoint)
    ref_model_name_or_path: Optional[str] = None
    
    # Override per_device_train_batch_size with DPO-specific default
    per_device_train_batch_size: int = field(
        default_factory=lambda: get_algorithm_parameter("dpo", "per_device_train_batch_size", 1)
    )
    
    # DPO algorithm parameters with configurable defaults
    beta: float = field(
        default_factory=lambda: get_algorithm_parameter("dpo", "beta", 0.1)
    )
    ref_coef: float = field(
        default_factory=lambda: get_algorithm_parameter("dpo", "ref_coef", 1.0)
    )
    label_smoothing: float = field(
        default_factory=lambda: get_algorithm_parameter("dpo", "label_smoothing", 0.0)
    )
    loss_type: str = field(
        default_factory=lambda: get_algorithm_parameter("dpo", "loss_type", "sigmoid")
    )


@dataclass(kw_only=True)
class ORPOArgs(TrainingArgs):
    """Arguments for ORPO training.
    
    Enhanced with configurable algorithm parameters from algorithm_defaults.yaml
    """

    dataset_name: Optional[str] = None
    pseudo_data_path: Optional[str] = None
    pseudo_weight: float = 1.0
    max_source_length: int = 512
    max_target_length: int = 48
    # Optional explicit reference policy path for KL monitoring
    ref_model_name_or_path: Optional[str] = None
    
    # Override per_device_train_batch_size with ORPO-specific default
    per_device_train_batch_size: int = field(
        default_factory=lambda: get_algorithm_parameter("orpo", "per_device_train_batch_size", 2)
    )
    
    # ORPO algorithm parameters with configurable defaults
    beta: float = 0.1  # Legacy parameter, kept for compatibility
    alpha_reg: float = field(
        default_factory=lambda: get_algorithm_parameter("orpo", "alpha_reg", 0.1)
    )
    margin: float = field(
        default_factory=lambda: get_algorithm_parameter("orpo", "margin", 0.1)
    )
    temperature: float = field(
        default_factory=lambda: get_algorithm_parameter("orpo", "temperature", 1.0)
    )


@dataclass(kw_only=True)
class RMArgs(TrainingArgs):
    """Arguments for reward model training."""

    dataset_name: str
    pseudo_data_path: Optional[str] = None
    max_source_length: int = 512
    max_target_length: int = 48
    per_device_train_batch_size: int = 2
    learning_rate: float = 1e-5


@dataclass(kw_only=True)
class DualDPOArgs(TrainingArgs):
    """Arguments for DualDPO training."""

    target_dataset_name: str
    source_dataset_name: Optional[str] = None
    pseudo_data_path: Optional[str] = None
    pseudo_weight: float = 1.0
    per_device_train_batch_size: int = 2
    learning_rate: float = 1e-6
    max_source_length: int = 512
    max_target_length: int = 48
    beta: float = 0.1
    source_batches_per_update: int = 1
    target_batches_per_update: int = 1
    sft_loss_scaling: float = 0.1
    rl_training: bool = False
    ref_model_name_or_path: Optional[str] = None


@dataclass(kw_only=True)
class DANNArgs(TrainingArgs):
    """Arguments for DANN-DPO training."""

    target_dataset_name: str
    source_dataset_name: Optional[str] = None
    pseudo_data_path: Optional[str] = None
    pseudo_weight: float = 1.0
    per_device_train_batch_size: int = 2
    learning_rate: float = 1e-6
    max_source_length: int = 512
    max_target_length: int = 48
    beta: float = 0.1
    mu: float = 1.0
    base_algorithm: str = "dpo"         # sft|dpo|kto|orpo (pipeline uses DPO)
    source_batches_per_update: int = 1
    target_batches_per_update: int = 1
    ref_model_name_or_path: Optional[str] = None


@dataclass(kw_only=True)
class DualRLHFArgs(TrainingArgs):
    """Arguments for dual-domain RLHF (PPO/KTO) training.
    
    Enhanced with configurable algorithm parameters from algorithm_defaults.yaml
    """

    target_dataset_name: str
    source_dataset_name: str
    objective: Literal["ppo", "kto", "orpo"] = "ppo"
    # Reward models are only mandatory when running PPO.  KTO/ORPO stages operate
    # without a learned reward model, so templates may omit this field.
    reward_model_name_or_path: Optional[str] = None
    ref_model_name_or_path: Optional[str] = None
    per_device_train_batch_size: int = 2
    learning_rate: float = 1e-6
    max_source_length: int = 512
    max_target_length: int = 48
    
    # Algorithm-specific parameters with configurable defaults
    beta: float = 0.1  # Used by PPO/KTO/ORPO objectives
    alpha_reg: float = field(
        default_factory=lambda: get_algorithm_parameter("orpo", "alpha_reg", 0.1)
    )
    margin: float = field(
        default_factory=lambda: get_algorithm_parameter("orpo", "margin", 0.1)
    )
    temperature: float = field(
        default_factory=lambda: get_algorithm_parameter("orpo", "temperature", 1.0)
    )
    target_kl: float = field(
        default_factory=lambda: get_algorithm_parameter("ppo", "target_kl", 0.1)
    )
    
    # PPO-specific parameters
    ppo_epochs: int = field(
        default_factory=lambda: get_algorithm_parameter("ppo", "epochs", 1)
    )
    vf_coef: float = field(
        default_factory=lambda: get_algorithm_parameter("ppo", "vf_coef", 0.5)
    )
    entropy_coef: float = field(
        default_factory=lambda: get_algorithm_parameter("ppo", "entropy_coef", 0.01)
    )
    clip_range: float = field(
        default_factory=lambda: get_algorithm_parameter("ppo", "clip_range", 0.2)
    )
    
    # KTO-specific parameters (used when objective="kto")
    kto_lambda: float = field(
        default_factory=lambda: get_algorithm_parameter("kto", "lambda", 1.5)
    )
    kto_alpha: float = field(
        default_factory=lambda: get_algorithm_parameter("kto", "alpha", 0.6)
    )
    
    # Dual domain training parameters
    source_batches_per_update: int = field(
        default_factory=lambda: get_algorithm_parameter("dual_domain", "source_batches_per_update", 1)
    )
    target_batches_per_update: int = field(
        default_factory=lambda: get_algorithm_parameter("dual_domain", "target_batches_per_update", 1)
    )
    sft_loss_scaling: float = field(
        default_factory=lambda: get_algorithm_parameter("dual_domain", "sft_loss_scaling", 1.0)
    )
    
    pseudo_data_path: Optional[str] = None
    pseudo_weight: float = 1.0
    # When mixing SFT batches into RLHF training we still need gold labels.
    # Keep ``rl_training`` explicit so dataset loaders know whether to drop
    # supervised targets.  Defaults to ``False`` for the target SFT dataset.
    rl_training: bool = False


@dataclass(kw_only=True)
class KTOArgs(TrainingArgs):
    """Arguments for KTO training.
    
    Enhanced with configurable algorithm parameters from algorithm_defaults.yaml
    """

    pseudo_data_path: Optional[str] = None
    pseudo_weight: float = 1.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    max_source_length: int = 2000
    max_target_length: int = 48
    
    # KTO algorithm parameters with configurable defaults
    beta: float = field(
        default_factory=lambda: get_algorithm_parameter("kto", "beta", 0.1)
    )
    gamma: float = 1.0  # Keep as non-configurable legacy parameter
    tau: float = 1.0    # Keep as non-configurable legacy parameter
    
    # KTO-specific parameters from algorithm config
    kto_lambda: float = field(
        default_factory=lambda: get_algorithm_parameter("kto", "lambda", 1.5)
    )
    kto_alpha: float = field(
        default_factory=lambda: get_algorithm_parameter("kto", "alpha", 0.6)
    )
    ref_coef: float = field(
        default_factory=lambda: get_algorithm_parameter("kto", "ref_coef", 1.0)
    )

    # Legacy parameters (maintained for backward compatibility)
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0
    auto_balance_kto_weights: bool = True
    resume: bool = False


@dataclass(kw_only=True)
class PPOArgs(TrainingArgs):
    """Arguments for PPO-based RLHF training."""

    reward_model_name_or_path: str
    value_model_name_or_path: Optional[str] = None
    dataset_name: str
    per_device_train_batch_size: int = 2
    rl: RLConfig
    gradient_accumulation_steps: int = field(
        default_factory=lambda: DEFAULTS.get("batch", {}).get("gradient_accumulation", 32) * 2
    )
    learning_rate: float = 1e-5
    max_source_length: int = 512
    max_target_length: int = 48
    beta: float = 0.1
    target_kl: float = 0.1
    pseudo_data_path: Optional[str] = None
    pseudo_weight: float = 1.0
    ppo_single_gpu_baseline: bool = False

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        super().__post_init__()
        if self.value_model_name_or_path is None:
            self.value_model_name_or_path = self.reward_model_name_or_path
        if self.rl is None or self.rl.ppo is None:
            raise ValueError("PPO configs must define rl.shared and rl.ppo blocks.")


__all__ = [
    "TrainingArgs",
    "RLSharedConfig",
    "RLPPOConfig",
    "RLGRPOConfig",
    "RLConfig",
    "SFTArgs",
    "DAPTArgs",
    "GRPOArgs",
    "OnlineDPOArgs",
    "DPOArgs",
    "ORPOArgs",
    "RMArgs",
    "DualDPOArgs",
    "DANNArgs",
    "DualRLHFArgs",
    "KTOArgs",
    "PPOArgs",
]
