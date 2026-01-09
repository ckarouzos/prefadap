"""Enhanced configuration loader with algorithm defaults and template support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import logging

logger = logging.getLogger(__name__)

# Path handling: enhanced_loader.py lives under prefadap/training/args/
# ROOT should reference the repository root for config loading
ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = ROOT / "configs"
ALGORITHM_DEFAULTS_FILE = CONFIG_ROOT / "algorithm_defaults.yaml" 
COMMON_DIR = CONFIG_ROOT / "common"
TEMPLATES_DIR = CONFIG_ROOT / "templates"

# Load algorithm defaults
_ALGORITHM_DEFAULTS: Dict[str, Any] = {}
if ALGORITHM_DEFAULTS_FILE.exists():
    with ALGORITHM_DEFAULTS_FILE.open() as f:
        _ALGORITHM_DEFAULTS = yaml.safe_load(f) or {}

# Load common configuration files
_COMMON_CONFIGS: Dict[str, Any] = {}
for config_file in COMMON_DIR.glob("*.yaml"):
    config_name = config_file.stem
    with config_file.open() as f:
        _COMMON_CONFIGS[config_name] = yaml.safe_load(f) or {}

# Backwards compatibility - create DEFAULTS from common configs
DEFAULTS: Dict[str, Any] = _COMMON_CONFIGS.get("base_training", {})

# Add algorithm defaults to the main DEFAULTS
ALGORITHM_DEFAULTS: Dict[str, Any] = _ALGORITHM_DEFAULTS

def load_algorithm_config(algorithm: str) -> Dict[str, Any]:
    """Load algorithm-specific configuration.
    
    Args:
        algorithm: Algorithm name (e.g., 'ppo', 'dpo', 'kto', 'sft')
        
    Returns:
        Dictionary with algorithm-specific configuration parameters
    """
    return _ALGORITHM_DEFAULTS.get(algorithm, {})

def load_template_config(template_name: str) -> Dict[str, Any]:
    """Load a configuration template.
    
    Args:
        template_name: Name of the template (e.g., 'sft_standard', 'dpo_standard')
        
    Returns:
        Dictionary with template configuration
        
    Raises:
        FileNotFoundError: If template doesn't exist
    """
    template_file = TEMPLATES_DIR / f"{template_name}.yaml"
    if not template_file.exists():
        raise FileNotFoundError(f"Template {template_name} not found at {template_file}")
    
    with template_file.open() as f:
        return yaml.safe_load(f) or {}

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration dictionaries.
    
    Later dictionaries override earlier ones. Nested dictionaries are merged recursively.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    result = {}
    
    for config in configs:
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
                
    return result

def load_complete_config(
    base_config: Optional[Dict[str, Any]] = None,
    template: Optional[str] = None,
    algorithm: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Load a complete configuration by merging multiple sources.
    
    Merge order (later overrides earlier):
    1. Algorithm defaults
    2. Template configuration  
    3. Base configuration
    4. Override parameters
    
    Args:
        base_config: Base configuration dictionary
        template: Template name to load
        algorithm: Algorithm name for algorithm-specific defaults
        overrides: Override parameters
        
    Returns:
        Complete merged configuration
    """
    configs_to_merge = []
    
    # 1. Algorithm defaults
    if algorithm:
        alg_config = load_algorithm_config(algorithm)
        if alg_config:
            configs_to_merge.append(alg_config)
            logger.debug(f"Loaded algorithm config for {algorithm}")
    
    # 2. Template configuration
    if template:
        try:
            template_config = load_template_config(template)
            configs_to_merge.append(template_config)
            logger.debug(f"Loaded template config {template}")
        except FileNotFoundError:
            logger.warning(f"Template {template} not found, skipping")
    
    # 3. Base configuration
    if base_config:
        configs_to_merge.append(base_config)
    
    # 4. Overrides
    if overrides:
        configs_to_merge.append(overrides)
    
    if not configs_to_merge:
        return {}
    
    return merge_configs(*configs_to_merge)

def get_algorithm_parameter(
    algorithm: str, 
    parameter_path: str, 
    default: Any = None
) -> Any:
    """Get a specific algorithm parameter using dot notation.
    
    Args:
        algorithm: Algorithm name
        parameter_path: Dot-separated parameter path (e.g., 'ppo.epochs')
        default: Default value if parameter not found
        
    Returns:
        Parameter value or default
        
    Examples:
        >>> get_algorithm_parameter('ppo', 'epochs', 1)
        1
        >>> get_algorithm_parameter('dpo', 'beta', 0.1) 
        0.1
    """
    config = load_algorithm_config(algorithm)
    
    # Navigate through nested dictionaries using dot notation
    keys = parameter_path.split('.')
    current = config
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
            
    return current

def list_available_templates() -> list[str]:
    """List all available configuration templates.
    
    Returns:
        List of template names (without .yaml extension)
    """
    if not TEMPLATES_DIR.exists():
        return []
    
    return [f.stem for f in TEMPLATES_DIR.glob("*.yaml")]

def list_available_algorithms() -> List[str]:
    """List all available algorithm configurations.
    
    Returns:
        List of algorithm names with available configurations
    """
    return list(_ALGORITHM_DEFAULTS.keys())

__all__ = [
    "DEFAULTS", 
    "ALGORITHM_DEFAULTS",
    "load_algorithm_config",
    "load_template_config", 
    "merge_configs",
    "load_complete_config",
    "get_algorithm_parameter",
    "list_available_templates",
    "list_available_algorithms"
]
