"""Utilities for safe HuggingFace Hub authentication and model loading.

This module provides helpers for:
1. Disabling HF transfer acceleration (breaks when hf_transfer is missing)
2. Getting auth tokens explicitly (required for Transformers >4.57)
3. Validating gated model access before loading
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def disable_hf_transfer() -> None:
    """Disable HF Hub fast transfer mode.
    
    HF_HUB_ENABLE_HF_TRANSFER=1 forces fast-transfer mode and raises ValueError
    if the dependency 'hf_transfer' is missing. On HPC clusters, hf_transfer
    often is not installed. This must be unset before model loads.
    
    Explicit token required for gated HF repos (Transformers >4.57 no longer
    inject implicitly).
    """
    for var in ["HF_HUB_ENABLE_HF_TRANSFER"]:
        if os.getenv(var):
            print(f"[hf-fix] Removing {var} (hf_transfer may not be installed)", file=sys.stderr)
            os.environ.pop(var, None)


def get_hf_token() -> Optional[str]:
    """Get HuggingFace Hub authentication token from environment.
    
    Checks environment variables in the following order:
    1. HUGGINGFACE_HUB_TOKEN (preferred)
    2. HF_TOKEN (fallback)
    
    Returns:
        Token string if available, None otherwise.
    """
    return os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")


def validate_gated_repo_access(repo_id: str, token: Optional[str] = None) -> bool:
    """Validate access to a gated HuggingFace repository.
    
    Purpose:
    - Fail early if token is invalid or missing
    - Detect cluster connectivity issues
    - Avoid silently continuing into "config not found" fallback paths
    
    Args:
        repo_id: Repository identifier (e.g., "meta-llama/Meta-Llama-3.1-8B")
        token: Optional token; if None, will attempt to get from environment
        
    Returns:
        True if access is validated, False otherwise
        
    Raises:
        RuntimeError: If token is required but missing
        SystemExit: If validation fails and we should abort
    """
    if token is None:
        token = get_hf_token()
    
    # Known gated model patterns (can be extended)
    # These are patterns known to require authentication
    known_gated_patterns = [
        "llama", "meta-llama",  # Meta's Llama models
        "mistral",  # Some Mistral models are gated
    ]
    
    # Check if this looks like a gated repo based on known patterns
    is_gated_repo = any(
        pattern in repo_id.lower()
        for pattern in known_gated_patterns
    )
    
    # If we have a token, always try to validate regardless of pattern
    # This handles cases where new gated models might not match our patterns
    if token:
        try:
            from huggingface_hub import model_info
            
            _ = model_info(repo_id, token=token)
            print(f"[hf-fix] HuggingFace gated repo access validated: {repo_id}", file=sys.stderr)
            return True
        except ImportError:
            # huggingface_hub not available, skip validation
            print("[hf-fix] Warning: huggingface_hub not available, skipping validation", file=sys.stderr)
            return True
        except Exception as e:
            # Only treat as fatal for known gated repos
            if is_gated_repo:
                print(f"[hf-fix] ERROR: HuggingFace auth or gating failed for {repo_id}", file=sys.stderr)
                print(f"[hf-fix] Error details: {e}", file=sys.stderr)
                sys.exit(1)
            else:
                # For non-gated repos, just warn
                print(f"[hf-fix] Warning: Could not validate access to {repo_id}: {e}", file=sys.stderr)
                return False
    elif is_gated_repo:
        # Known gated repo but no token provided
        print(
            "[hf-fix] ERROR: Missing required env var: HUGGINGFACE_HUB_TOKEN or HF_TOKEN",
            file=sys.stderr,
        )
        print(
            f"[hf-fix] Required for gated repo: {repo_id}",
            file=sys.stderr,
        )
        raise RuntimeError(
            "Missing required environment variable: HUGGINGFACE_HUB_TOKEN or HF_TOKEN"
        )
    
    # Non-gated repo without token is fine
    return True


def prepare_hf_from_pretrained_kwargs(
    repo_id: str,
    *,
    validate_access: bool = True,
    **kwargs,
) -> dict:
    """Prepare kwargs for AutoModel.from_pretrained with proper auth.
    
    Canonical repo name required to resolve Llama 3.1 models on clean HPC nodes.
    HF_HUB_ENABLE_HF_TRANSFER disabled because it breaks downloads if hf_transfer
    is not installed.
    
    Args:
        repo_id: Repository identifier or local path
        validate_access: Whether to validate gated repo access first
        **kwargs: Additional kwargs to pass to from_pretrained
        
    Returns:
        Dictionary of kwargs including token if appropriate
    """
    # Disable HF transfer if enabled
    disable_hf_transfer()
    
    # Check if this looks like a HuggingFace repo ID (not a local path)
    is_likely_hf_repo = (
        "/" in repo_id
        and not repo_id.startswith(("./", "../", "/"))
        and not os.path.exists(repo_id)
    )
    
    # For HF repos, add token if available and validate access
    if is_likely_hf_repo:
        token = get_hf_token()
        if token and "token" not in kwargs:
            kwargs["token"] = token
        
        if validate_access:
            validate_gated_repo_access(repo_id, token)
    
    return kwargs
