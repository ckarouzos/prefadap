"""OLMo-3 sequence classification model definition.

This module defines Olmo3ForSequenceClassification for use as a reward model
in RLHF pipelines. The model must be loaded directly via:
  - Olmo3ForSequenceClassification.from_pretrained() for explicit loading

The implementation follows the HuggingFace LLaMA pattern:
- Inherit from GenericForSequenceClassification for the classification head
- Inherit from Olmo3PreTrainedModel for the base model architecture
- No custom __init__, forward, or other overrides
- Rely entirely on GenericForSequenceClassification's implementation

Note: Due to AutoModel registration issues in transformers 4.55+, the
load_reward_model_and_tokenizer() function uses direct instantiation
rather than relying on AutoModel registration.
"""

from __future__ import annotations

from transformers.models.olmo3.configuration_olmo3 import Olmo3Config
from transformers.models.olmo3.modeling_olmo3 import Olmo3PreTrainedModel
from transformers.models.llama.modeling_llama import GenericForSequenceClassification


class Olmo3ForSequenceClassification(
    GenericForSequenceClassification,
    Olmo3PreTrainedModel,
):
    """OLMo-3 model with a sequence classification head.

    This class enables using OLMo-3 models as reward models in RLHF
    pipelines by providing a classification head on top of the base
    OLMo-3 architecture.

    The implementation follows the HuggingFace pattern used by LLaMA:
    - Inherits from GenericForSequenceClassification for the head
    - Inherits from Olmo3PreTrainedModel for the base architecture
    - No custom implementation needed

    Usage:
        >>> from prefadap.models.olmo3_seqcls import Olmo3ForSequenceClassification
        >>> model = Olmo3ForSequenceClassification.from_pretrained(
        ...     "allenai/OLMo-3-7B-reward-model"
        ... )
    """

    config_class = Olmo3Config


__all__ = ["Olmo3ForSequenceClassification"]

