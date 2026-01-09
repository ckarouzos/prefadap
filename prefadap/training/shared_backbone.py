"""Shared backbone implementation for memory-efficient DPO/GRPO training.

This module implements a shared-backbone architecture where a single base model
is loaded into memory with two lightweight LoRA adapters: one for the policy
(trainable) and one for the reference (frozen). This approach significantly
reduces memory usage for large models (≥30B parameters) by avoiding the need
to load two full model copies.

The implementation is particularly important for large models like OLMo-3-1125-32B
on systems with limited GPU memory.
"""

from __future__ import annotations

import logging
from typing import Optional, Any
from transformers import PreTrainedModel

try:
    from peft import PeftModel, LoraConfig, get_peft_model, TaskType
except ImportError:
    PeftModel = None  # type: ignore
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore
    TaskType = None  # type: ignore

logger = logging.getLogger(__name__)


class SharedBackboneManager:
    """Manages a shared base model with separate policy and reference LoRA adapters.
    
    This class provides a memory-efficient way to handle DPO/GRPO training by:
    1. Loading a single base model
    2. Attaching two LoRA adapters (policy and reference)
    3. Providing methods to switch between adapters during training
    
    The reference adapter is frozen (non-trainable) while the policy adapter
    is trainable. This approach saves significant GPU memory compared to loading
    two full model copies.
    
    Attributes:
        base_model: The underlying pretrained model
        policy_adapter_name: Name of the policy (trainable) adapter
        reference_adapter_name: Name of the reference (frozen) adapter
        current_adapter: Name of the currently active adapter
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        lora_config: dict[str, Any],
        policy_adapter_name: str = "policy",
        reference_adapter_name: str = "reference",
    ):
        """Initialize the shared backbone manager.
        
        Args:
            base_model: Pretrained model to use as the shared backbone
            lora_config: LoRA configuration dictionary with keys like r, alpha, dropout, etc.
            policy_adapter_name: Name for the policy (trainable) adapter
            reference_adapter_name: Name for the reference (frozen) adapter
        """
        if PeftModel is None or LoraConfig is None or get_peft_model is None:
            raise ImportError(
                "PEFT library is required for shared backbone. "
                "Install with: pip install peft"
            )
        
        self.base_model = base_model
        self.policy_adapter_name = policy_adapter_name
        self.reference_adapter_name = reference_adapter_name
        self.current_adapter: Optional[str] = None
        
        # Extract LoRA parameters
        r = lora_config.get("r", 16)
        alpha = lora_config.get("alpha", 32)
        dropout = lora_config.get("dropout", 0.05)
        target_modules = lora_config.get("target_modules", None)
        
        # Create LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        # Apply first adapter (policy) to the base model
        logger.info(f"Applying policy adapter '{policy_adapter_name}' to base model")
        self.model = get_peft_model(base_model, peft_config, adapter_name=policy_adapter_name)
        
        # Add second adapter (reference) - this will be frozen
        logger.info(f"Adding reference adapter '{reference_adapter_name}' to base model")
        self.model.add_adapter(reference_adapter_name, peft_config)
        
        # Freeze the reference adapter
        self._freeze_adapter(reference_adapter_name)
        
        # Set policy adapter as active by default
        self.set_policy_adapter()
        
        logger.info(
            f"Shared backbone initialized with adapters: "
            f"policy='{policy_adapter_name}' (trainable), "
            f"reference='{reference_adapter_name}' (frozen)"
        )
    
    def _freeze_adapter(self, adapter_name: str) -> None:
        """Freeze all parameters of a specific adapter.
        
        Args:
            adapter_name: Name of the adapter to freeze
        """
        frozen_count = 0
        for name, param in self.model.named_parameters():
            # Check if this parameter belongs to the target adapter
            if adapter_name in name and "lora_" in name:
                param.requires_grad = False
                frozen_count += 1
        
        logger.debug(f"Froze {frozen_count} parameters in adapter '{adapter_name}'")
    
    def set_policy_adapter(self) -> None:
        """Switch to the policy (trainable) adapter."""
        self.model.set_adapter(self.policy_adapter_name)
        self.current_adapter = self.policy_adapter_name
        logger.debug(f"Switched to policy adapter '{self.policy_adapter_name}'")
    
    def set_reference_adapter(self) -> None:
        """Switch to the reference (frozen) adapter."""
        self.model.set_adapter(self.reference_adapter_name)
        self.current_adapter = self.reference_adapter_name
        logger.debug(f"Switched to reference adapter '{self.reference_adapter_name}'")
    
    def get_policy_model(self) -> PeftModel:
        """Get the model with policy adapter active.
        
        Returns:
            Model with policy adapter active (trainable)
        """
        self.set_policy_adapter()
        return self.model
    
    def get_reference_model(self) -> PeftModel:
        """Get the model with reference adapter active.
        
        Returns:
            Model with reference adapter active (frozen)
        """
        self.set_reference_adapter()
        return self.model
    
    def get_current_model(self) -> PeftModel:
        """Get the model with the currently active adapter.
        
        Returns:
            Model with current adapter active
        """
        return self.model


def create_shared_backbone_for_dpo(
    base_model: PreTrainedModel,
    lora_config: dict[str, Any],
) -> tuple[PeftModel, PeftModel]:
    """Create a shared backbone setup for DPO training.
    
    This is a convenience function that creates a SharedBackboneManager and
    returns separate references to the policy and reference models. Both
    references point to the same underlying model but with different adapters
    active.
    
    Args:
        base_model: Pretrained model to use as shared backbone
        lora_config: LoRA configuration dictionary
        
    Returns:
        Tuple of (policy_model, reference_model), both using the same backbone
    """
    manager = SharedBackboneManager(base_model, lora_config)
    
    # Return the model twice, but callers should be aware they share the same backbone
    # The manager handles adapter switching internally
    policy_model = manager.get_policy_model()
    
    # For reference model, we create a view that automatically switches to reference adapter
    # This is a simple wrapper that ensures the reference adapter is active
    class ReferenceModelView:
        """Wrapper that ensures reference adapter is always active."""
        
        def __init__(self, manager: SharedBackboneManager):
            self._manager = manager
        
        def __getattr__(self, name: str):
            # Always ensure reference adapter is active when accessing the model
            self._manager.set_reference_adapter()
            return getattr(self._manager.model, name)
        
        def __call__(self, *args, **kwargs):
            self._manager.set_reference_adapter()
            return self._manager.model(*args, **kwargs)
    
    reference_model = ReferenceModelView(manager)
    
    logger.info("Created shared backbone for DPO with policy and reference adapters")
    
    return policy_model, reference_model  # type: ignore


__all__ = [
    "SharedBackboneManager",
    "create_shared_backbone_for_dpo",
]
