"""Configuration utilities for loading and applying algorithm-specific configurations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar
import yaml

from .args import (
    load_algorithm_config,
    load_complete_config,
    merge_configs
)

logger = logging.getLogger(__name__)

T = TypeVar('T')

def load_config_for_pipeline(
    pipeline_type: str,
    config_path: Optional[Path] = None,
    template: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Load configuration for a specific pipeline type.
    
    Args:
        pipeline_type: Type of pipeline (e.g., 'sft', 'dpo', 'kto', 'dual_rlhf')
        config_path: Path to base configuration file
        template: Configuration template name
        overrides: Override parameters
        
    Returns:
        Complete configuration dictionary
    """
    base_config = {}
    
    # Load base configuration file if provided
    if config_path and config_path.exists():
        with config_path.open() as f:
            base_config = yaml.safe_load(f) or {}
    
    # Use template if no explicit template specified but we can infer one
    if template is None:
        template = f"{pipeline_type}_standard"
    
    # Load complete configuration
    config = load_complete_config(
        base_config=base_config,
        template=template,
        algorithm=pipeline_type,
        overrides=overrides
    )

    logger.info(f"Loaded configuration for {pipeline_type} pipeline")
    if template:
        logger.info(f"Using template: {template}")

    return config

def apply_algorithm_defaults(
    config_class: Type[T],
    config_dict: Dict[str, Any],
    algorithm: Optional[str] = None
) -> T:
    """Apply algorithm defaults to a configuration class instance.
    
    Args:
        config_class: Configuration dataclass type
        config_dict: Configuration parameters
        algorithm: Algorithm name for defaults
        
    Returns:
        Configured instance
    """
    # If algorithm is specified, merge in algorithm defaults
    if algorithm:
        algorithm_config = load_algorithm_config(algorithm)
        config_dict = merge_configs(algorithm_config, config_dict)
    
    # Filter config_dict to only include parameters that exist in the config class
    if hasattr(config_class, '__dataclass_fields__'):
        valid_fields = set(config_class.__dataclass_fields__.keys())
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        # Log any filtered out parameters
        filtered_out = set(config_dict.keys()) - valid_fields
        if filtered_out:
            logger.debug(f"Filtered out unknown parameters: {filtered_out}")
    else:
        filtered_config = config_dict
    
    return config_class(**filtered_config)

def validate_algorithm_config(algorithm: str, config: Dict[str, Any]) -> List[str]:
    """Validate that required parameters are present for an algorithm.
    
    Args:
        algorithm: Algorithm name
        config: Configuration to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Define required parameters for each algorithm
    # Only check for parameters that don't have sensible defaults in the dataclass
    required_params = {
        'ppo': ['target_kl'],  # beta has algorithm default
        'dpo': ['beta'],
        'kto': ['lambda', 'alpha', 'beta'],
        'sft': [],
        'dual_rlhf': ['source_batches_per_update', 'target_batches_per_update'],  # objective has default "ppo"
        'orpo': ['alpha_reg', 'margin'],
    }
    
    required = required_params.get(algorithm, [])
    for param in required:
        if param not in config:
            errors.append(f"Missing required parameter for {algorithm}: {param}")
    
    # Algorithm-specific validation
    if algorithm == 'dual_rlhf':
        objective = config.get('objective', 'ppo')  # Default from DualRLHFArgs
        valid_objectives = {'ppo', 'kto', 'orpo'}
        if objective not in valid_objectives:
            errors.append(
                f"Invalid objective for dual_rlhf: {objective}. Must be one of {sorted(valid_objectives)}"
            )
        elif objective == 'orpo':
            for param in ['alpha_reg', 'margin']:
                if param not in config:
                    errors.append(
                        "Missing required parameter for dual_rlhf orpo objective: "
                        f"{param}"
                    )
    
    return errors

def create_experiment_config(
    experiment_type: str,
    domain: str = "CNNDM", 
    model_name: str = "Llama3-8B",
    method: str = "SFT",
    adaptation: str = "NONE",
    seed: int = 0,
    **overrides
) -> Dict[str, Any]:
    """Create a complete experiment configuration.
    
    Args:
        experiment_type: Type of experiment configuration to create
        domain: Target domain
        model_name: Model identifier  
        method: Training method
        adaptation: Adaptation strategy
        seed: Random seed
        **overrides: Additional parameter overrides
        
    Returns:
        Complete experiment configuration
    """
    # Start with base template
    base_config = {
        'experiment_type': experiment_type,
        'model': {
            'model_name_or_path': model_name,
        },
        'data': {
            'target_domain': domain,
        },
        'training': {
            'seed': seed,
            'method': method,
            'adaptation': adaptation,
        }
    }
    
    # Merge with overrides
    config = merge_configs(base_config, overrides)
    
    # Apply algorithm-specific defaults
    algorithm = method.lower()
    complete_config = load_complete_config(
        base_config=config,
        algorithm=algorithm,
        template=f"{algorithm}_standard"
    )

    return complete_config

def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved configuration to {output_path}")

__all__ = [
    'load_config_for_pipeline',
    'apply_algorithm_defaults',
    'validate_algorithm_config',
    'create_experiment_config',
    'save_config'
]
