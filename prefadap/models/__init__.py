"""Model loading, tokenization, and management utilities.

This package provides comprehensive utilities for working with transformer models
in the preference adaptation framework. It handles model loading, tokenization,
parameter-efficient fine-tuning integration, and model-specific preprocessing.

## Design Philosophy

The models package follows these principles:
- **Unified Interface**: Consistent API across different model architectures
- **Lazy Loading**: Models loaded on-demand to minimize memory usage
- **Format Agnostic**: Support for various model formats (HuggingFace, local, etc.)
- **Tokenization Safety**: Automatic handling of special tokens and padding

## Key Components

### Model Loading
- **load_model_and_tokenizer**: Main interface for loading language models
- **load_reward_model_and_tokenizer**: Specialized loader for reward models
- **Architecture Detection**: Automatic model architecture identification
- **Memory Management**: Efficient loading for large models

### Tokenization Utilities
- **ensure_pad_token**: Automatic pad token management
- **mask_target_tokens**: Token masking for training objectives
- **tokenize_prompt_and_target**: Structured tokenization for training
- **IGNORE_INDEX**: Standard ignore index for loss computation

### Parameter-Efficient Training
- **LoRA Integration**: Seamless integration with LoRA and other PEFT methods
- **Adapter Management**: Loading and merging adapter weights
- **Memory Optimization**: Efficient fine-tuning for large models

## Supported Model Types

### Language Models
- **Causal LM**: GPT-style autoregressive models (GPT-2, GPT-3.5, LLaMA, Mistral)
- **Encoder-Decoder**: T5, BART, and similar seq2seq models
- **Instruction-Tuned**: Chat models with specific formatting requirements

### Reward Models  
- **Classification Head**: Models with classification layers for scoring
- **Regression Head**: Models outputting continuous preference scores
- **Custom Architectures**: Support for domain-specific reward models

### Specialized Models
- **LoRA Models**: Parameter-efficient adapters
- **Quantized Models**: 4-bit and 8-bit quantized variants
- **Multi-GPU Models**: Models sharded across multiple devices

## Tokenization Features

### Smart Tokenization
```python
from prefadap.models import tokenize_prompt_and_target

# Automatic handling of prompt/response tokenization
input_ids, labels = tokenize_prompt_and_target(
    tokenizer=tokenizer,
    prompt="What is the capital of France?",
    target="The capital of France is Paris.",
    max_length=512
)
```

### Token Management
```python
from prefadap.models import ensure_pad_token, mask_target_tokens

# Ensure tokenizer has pad token
tokenizer = ensure_pad_token(tokenizer)

# Mask tokens for training objectives
labels = mask_target_tokens(input_ids, tokenizer, mask_prompt=True)
```

## Model Loading Examples

### Basic Model Loading
```python
from prefadap.models import load_model_and_tokenizer

# Load standard language model
model, tokenizer = load_model_and_tokenizer(
    "microsoft/DialoGPT-medium",
    config={
        "dtype": "float16",
        "device_map": "auto"
    }
)
```

### Reward Model Loading
```python
from prefadap.models import load_reward_model_and_tokenizer

# Load specialized reward model
reward_model, tokenizer = load_reward_model_and_tokenizer(
    "OpenAssistant/reward-model-deberta-v3-large",
    config={"device": "cuda"}
)
```

### LoRA Model Loading
```python
# Load base model with LoRA adapter
model, tokenizer = load_model_and_tokenizer(
    "meta-llama/Llama-2-7b-hf",
    config={
        "load_in_4bit": True,
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"]
        }
    }
)
```

## Advanced Features

### Memory Optimization
- **Gradient Checkpointing**: Reduce memory usage during training
- **Mixed Precision**: Automatic float16/bfloat16 handling
- **Model Sharding**: Distribute large models across GPUs
- **Quantization**: 4-bit and 8-bit quantization support

### Format Handling
- **Chat Templates**: Automatic handling of conversation formats
- **Special Tokens**: Management of BOS, EOS, and custom tokens
- **Sequence Length**: Dynamic sequence length handling
- **Batching**: Efficient batching with proper padding

### Error Handling
- **Model Validation**: Verify model compatibility before loading
- **Memory Monitoring**: Detect and handle OOM conditions
- **Format Validation**: Ensure tokenizer/model compatibility
- **Graceful Fallbacks**: Fallback strategies for unsupported features

## Integration Points

### Training Integration
```python
# Models integrate seamlessly with training workflows
from prefadap.training import maybe_apply_lora

# Apply LoRA if configured
model = maybe_apply_lora(model, training_config)
```

### Plugin Integration
Custom model loaders can be registered:

```python
from prefadap.plugin.interfaces import ModelComponent

class CustomModelLoader(ModelComponent):
    @property
    def component_name(self) -> str:
        return "custom_loader"
    
    def load_model(self, model_path, config):
        # Custom loading logic
        return model, tokenizer
```

## Performance Considerations

### Loading Performance
- **Caching**: Automatic model and tokenizer caching
- **Parallel Loading**: Multi-threaded loading for large models
- **Memory Mapping**: Efficient file-based model loading
- **Progressive Loading**: Stream loading for very large models

### Runtime Performance  
- **Batched Inference**: Efficient batch processing
- **KV Caching**: Key-value cache management for generation
- **Attention Optimization**: Flash attention and other optimizations
- **Device Placement**: Optimal GPU/CPU placement strategies
"""

from .tokenization import (
    IGNORE_INDEX,
    ensure_pad_token,
    mask_target_tokens,
    tokenize_prompt_and_target,
)
from .loader import load_model_and_tokenizer, load_reward_model_and_tokenizer
from .model_card import ModelCardGenerator, generate_model_card_from_context

# Import olmo3_seqcls to register Olmo3ForSequenceClassification at module load time.
# This must happen before any code attempts to load OLMo-3 reward models with
# AutoModelForSequenceClassification. The olmo3_seqcls module itself gracefully
# handles environments where transformers does not provide the required classes.
from . import olmo3_seqcls  # noqa: F401

__all__ = [
    "IGNORE_INDEX",
    "ensure_pad_token",
    "mask_target_tokens",
    "tokenize_prompt_and_target",
    "load_model_and_tokenizer",
    "load_reward_model_and_tokenizer",
    "ModelCardGenerator",
    "generate_model_card_from_context",
]
