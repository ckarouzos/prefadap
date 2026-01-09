"""Load default hyperparameters for training with enhanced configuration support.

This module provides enhanced configuration functionality including:
- Algorithm-specific configuration defaults
- Configuration templates for common experiment types  
- Enhanced parameter loading and merging
"""

from __future__ import annotations

import logging

# Import enhanced configuration functionality
from .enhanced_loader import (
    DEFAULTS,
    ALGORITHM_DEFAULTS, 
    load_algorithm_config,
    load_template_config,
    merge_configs,
    load_complete_config,
    get_algorithm_parameter,
    list_available_templates,
    list_available_algorithms
)

logger = logging.getLogger(__name__)

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
