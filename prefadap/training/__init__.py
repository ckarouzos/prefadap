"""Preference-based model training algorithms and utilities.

This package implements various preference adaptation methods including Supervised
Fine-Tuning (SFT), Direct Preference Optimization (DPO), Kahneman-Tversky Optimization 
(KTO), and other preference learning algorithms.

## Architecture

The training package uses lazy loading to avoid importing heavy dependencies unless
they are actually needed. Components are accessed through a registry system that
enables pluggable training algorithms.

## Supported Algorithms

- **SFT**: Supervised Fine-Tuning on preferred responses
- **DPO**: Direct Preference Optimization using preference pairs
- **KTO**: Kahneman-Tversky Optimization for preference learning
- **RLHF**: Reinforcement Learning from Human Feedback (when applicable)
- **Custom methods**: Extensible through the plugin architecture

## Key Components

- **TrainingWrappers**: High-level training coordination
- **Algorithm Registry**: Pluggable training method registration
- **Configuration System**: Hierarchical config management with templates
- **LoRA Integration**: Parameter-efficient fine-tuning utilities

## Configuration System

The package features an enhanced configuration system that supports:
- Algorithm-specific default parameters
- Configuration templates for common experiment types
- Hierarchical parameter merging and override
- Backward compatibility with legacy configurations

## Example Usage

```python
from prefadap.training import maybe_apply_lora
from prefadap.training.args import load_algorithm_config

# Load DPO-specific configuration
config = load_algorithm_config("dpo", template="summarization")

# Apply LoRA for parameter-efficient training
model = maybe_apply_lora(model, config)
```

## Integration with Plugin System

Training algorithms can be extended through the plugin architecture:

```python
from prefadap.plugin.interfaces import PreferenceAdaptationMethod

class CustomTrainingMethod(PreferenceAdaptationMethod):
    @property
    def method_name(self) -> str:
        return "custom_method"
    
    def adapt_model(self, model, preference_data, config):
        # Custom training logic
        return adapted_model
```

Training jobs are typically launched through the CLI:

```bash
python -m prefadap.cli.run_training dpo --config config.yaml
```
"""

_EXPORTS = {"maybe_apply_lora"}

__all__ = sorted(_EXPORTS)


def __getattr__(name):  # pragma: no cover - thin lazy passthrough
    if name in _EXPORTS:
        from .lora_utils import maybe_apply_lora as _maybe_apply_lora
        return _maybe_apply_lora
    raise AttributeError(f"module 'prefadap.training' has no attribute {name!r}")
