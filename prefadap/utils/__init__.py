"""Common utilities and helper functions for preference adaptation.

This package provides essential utilities that support all other components of the
preference adaptation framework. It includes logging, seeding, data manipulation,
and other foundational functionality.

## Design Philosophy

The utils package follows these principles:
- **Cross-Framework Compatibility**: Works consistently across PyTorch, NumPy, and Python
- **Reproducibility First**: Deterministic seeding and random state management
- **Flexible Logging**: Structured logging with multiple output formats
- **Performance Aware**: Efficient implementations for common operations

## Key Components

### Reproducibility and Seeding
- **set_seed**: Deterministic seeding across all frameworks (Python, NumPy, PyTorch)
- **Random state management**: Context managers for temporary seed changes
- **Deterministic algorithms**: Enable deterministic mode in PyTorch/cuDNN
- **Reproducible sampling**: Consistent random sampling across runs

### Logging and Debugging
- **setup_logging**: Global logging configuration with file and console outputs
- **create_logger**: Directory-based logger creation that returns a logger instance
- **disable_tqdm_if_not_tty**: Smart progress bar management
- **Structured logging**: JSON and structured log output
- **Debug utilities**: Memory profiling and performance monitoring

### Data Manipulation
- **enforce_length_band**: Filter data by length requirements
- **Text processing helpers**: Common text manipulation operations
- **Data validation**: Input validation and sanitization
- **Format conversion**: Convert between different data formats

### System Utilities
- **Environment detection**: Detect runtime environment capabilities
- **Resource monitoring**: Memory and compute resource tracking
- **Path management**: Cross-platform path handling
- **Configuration helpers**: Configuration loading and validation

## Seeding and Reproducibility

### Comprehensive Seeding
```python
from prefadap.utils import set_seed

# Set deterministic seed across all frameworks
set_seed(42)

# This ensures reproducible results across:
# - Python random module
# - NumPy random state
# - PyTorch random state
# - PyTorch CUDA random state
# - cuDNN deterministic algorithms
```

### Advanced Seeding Control
```python
from prefadap.utils.seeding import (
    seed_context, 
    get_random_state,
    set_deterministic_mode
)

# Temporary seed change
with seed_context(123):
    # Operations with seed 123
    random_data = torch.randn(100)
# Original seed restored

# Enable deterministic algorithms
set_deterministic_mode(True)
```

## Logging System

### Flexible Logging Setup
```python
from prefadap.utils.logging import setup_logging, create_logger

# Global logging configuration
setup_logging(quiet=False)

# Directory-based logging (returns logger instance)
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    logger = create_logger(tmpdir, filename="training.log")
    logger.info("Training started")
```

### Progress Bar Management
```python
from prefadap.utils.logging import disable_tqdm_if_not_tty
from tqdm import tqdm

# Automatically disable progress bars in non-interactive environments
disable_tqdm_if_not_tty()

# Progress bars will only show in interactive terminals
for item in tqdm(data):
    process(item)
```

### Standard Logging
```python
import logging
from prefadap.utils.logging import setup_logging

# Configure logging
setup_logging(log_file=Path("experiment.log"))

# Use standard logging
logger = logging.getLogger(__name__)
logger.info("Training started")
```

## Data Processing Utilities

### Length-Based Filtering
```python
from prefadap.utils.length_bands import enforce_length_band

# Filter dataset by text length
filtered_data = enforce_length_band(
    dataset,
    min_length=10,
    max_length=512,
    length_field="text"
)
```

## Configuration Management

The configuration utilities are available in the main config module:

```python
from prefadap.config import load_dict, load_configs, validate, schema

# Load configuration from file  
config = load_dict(Path("config.yaml"))

# Load and merge multiple configurations
configs = load_configs([Path("base.yaml"), Path("override.yaml")])

# Validate configuration against a dataclass
validated_config = validate(MyConfigClass, config_dict)
```

The utils package provides essential foundation utilities that enable reliable,
reproducible, and efficient operation of all other framework components.

Key utilities included:
- **Seeding**: Deterministic random state management across frameworks  
- **Logging**: Global logging configuration and directory-based logger creation
- **Length bands**: Text filtering by length constraints

For additional data processing, configuration management, and system utilities,
see the dedicated modules in other parts of the framework.

Note: set_seed is imported lazily to avoid importing torch at module load time.
This allows config parsing scripts to import prefadap.utils.paths without requiring torch.
"""

from .length_bands import enforce_length_band
from .logging import disable_tqdm_if_not_tty, setup_logging, create_logger


def __getattr__(name):
    """Lazy import for set_seed to avoid torch dependency at module load time."""
    if name == "set_seed":
        from .seeding import set_seed
        return set_seed
    raise AttributeError(f"module 'prefadap.utils' has no attribute {name!r}")

__all__ = [
    "enforce_length_band",
    "setup_logging",
    "create_logger",
    "disable_tqdm_if_not_tty",
    "set_seed",
    "build_run_dir",
    "canonical_run_dir",
    "is_test_path",
    "normalize_to_canonical_run_dir",
    "resolve_run_dir",
    "validate_parity",
    "ParityError",
]
