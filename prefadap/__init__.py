"""Preference Adaptation Framework.

The `prefadap` package provides a comprehensive framework for preference-based model 
adaptation, including tools for data generation, model training, and evaluation.

## Architecture Overview

The framework is organized into several key packages:

- **pseudo_label**: Generate pseudo-preference pairs from datasets for training
- **training**: Implement preference adaptation methods (SFT, DPO, KTO, etc.)
- **evaluation**: Score model outputs and compute evaluation metrics  
- **models**: Model loading, tokenization, and management utilities
- **cli**: Command-line interfaces for all major operations
- **plugin**: Extensible plugin architecture for custom components
- **data**: Dataset loading and processing utilities
- **generation**: Text preprocessing and corpus preparation tools
- **steer**: SteerLM-specific utilities for attribute-controlled generation
- **utils**: Common utilities for logging, seeding, and data manipulation

## Plugin Architecture

The framework features a comprehensive plugin system that allows custom components
to be registered and used throughout the training and evaluation pipeline. See
`prefadap.plugin` for details on creating and using plugins.

## Registry System  

Components like models, datasets, metrics, and adaptation methods are managed
through a registry system that enables easy extension and customization without
modifying core code.

## Usage

The framework is designed to be used primarily through its CLI tools:

```bash
# Generate pseudo-labels from a dataset
python -m prefadap.cli.pseudo_label_pipeline run <run_id> <dataset.jsonl> --output output.jsonl

# Train a preference-adapted model  
python -m prefadap.cli.run_training dpo --config training_config.yaml

# Evaluate diversity metrics for generated outputs
python -m prefadap.cli.evaluate_diversity <outputs_dir>
```

For programmatic usage, import specific modules as needed:

```python
from prefadap.training import maybe_apply_lora
from prefadap.models import load_model_and_tokenizer
from prefadap.evaluation import compute_metrics
```
"""

from __future__ import annotations

import os

# Disable telemetry and non-text transformers backends for HPC environments
for v in [
    "HF_HUB_DISABLE_TELEMETRY",
    "TRANSFORMERS_NO_TF",
    "TRANSFORMERS_NO_FLAX",
    "TRANSFORMERS_NO_TORCHVISION",
    "TRANSFORMERS_NO_GGUF",
    "TRANSFORMERS_NO_GGML",
]:
    os.environ.setdefault(v, "1")

__all__: list[str] = []
