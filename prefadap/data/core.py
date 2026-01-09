from __future__ import annotations

import logging
import os
from pathlib import Path
import numpy as np
from typing import Any, Dict

PathLike = os.PathLike[str] | os.PathLike[bytes] | str | bytes

from datasets import DatasetDict, load_dataset as hf_load_dataset


_HF_DATASETS_CACHE_LOGGED = False


def _rewrite_cluster_path(path: str) -> str:
    """Rewrite configured shared-data prefixes to local storage roots.

    This supports portable configs by mapping a shared prefix to a
    cluster-local or user-specific root when environment variables are set.

    Environment variables
    ---------------------
    PREFADAP_DATA_PREFIX:
        Path prefix to rewrite (e.g., "/data/shared").
    PREFADAP_DATA_PATH_OVERRIDE or PSEUDO_DATA_PATH:
        Full override path used when the prefix matches.
    PERSIST_ROOT or SCRATCHDIR:
        Base path used to rewrite the shared prefix to a writable location.
    """
    if not path:
        return path

    prefix = os.environ.get("PREFADAP_DATA_PREFIX")
    if not prefix:
        return path

    prefix = prefix.rstrip("/")
    if path != prefix and not path.startswith(prefix + "/"):
        return path

    override = os.environ.get("PREFADAP_DATA_PATH_OVERRIDE") or os.environ.get("PSEUDO_DATA_PATH")
    if override and override.strip():
        override = override.strip()
        logging.info("Overriding data path with %s", override)
        return override

    persist_root = os.environ.get("PERSIST_ROOT")
    scratchdir = os.environ.get("SCRATCHDIR")
    if persist_root:
        base = persist_root
    elif scratchdir:
        base = os.path.join(scratchdir, "prefadap")
    else:
        logging.warning(
            "Data path prefix %s matched but no PERSIST_ROOT or SCRATCHDIR is set; "
            "using original path: %s",
            prefix,
            path,
        )
        return path

    relative_path = path[len(prefix):].lstrip("/")
    rewritten = os.path.join(base, relative_path) if relative_path else base
    logging.info("Rewriting data path from %s to %s", path, rewritten)
    return rewritten


def load_dataset(*args, cache_dir: str | None = None, **kwargs):  # type: ignore[override]
    """Wrapper around :func:`datasets.load_dataset` with caching support."""

    cache_dir = resolve_hf_datasets_cache_dir(cache_dir)

    try:
        return hf_load_dataset(*args, cache_dir=cache_dir, **kwargs)
    except OSError as exc:  # pragma: no cover - network failure
        data_dir = os.environ.get("DATA")
        if data_dir:
            logging.warning(
                "Dataset download failed. Please pre-download datasets into %s: %s",
                os.path.join(data_dir, "hf_cache"),
                exc,
            )
        else:
            logging.warning("Dataset download failed: %s", exc)
        raise


def resolve_hf_datasets_cache_dir(cache_dir: str | None) -> str:
    """Resolve the Hugging Face datasets cache directory."""
    if cache_dir:
        resolved = _ensure_writable_cache_dir(cache_dir, source="explicit cache_dir")
        _log_hf_cache_dir(resolved)
        return resolved

    env_cache = os.environ.get("HF_DATASETS_CACHE")
    if env_cache:
        resolved = _ensure_writable_cache_dir(env_cache, source="HF_DATASETS_CACHE")
        _log_hf_cache_dir(resolved)
        return resolved

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        resolved = _ensure_writable_cache_dir(
            os.path.join(hf_home, "datasets"), source="HF_HOME"
        )
        _log_hf_cache_dir(resolved)
        return resolved

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        fallback = os.path.join(xdg_cache, "huggingface", "datasets")
    else:
        fallback = os.path.join(str(Path.home()), ".cache", "huggingface", "datasets")
    resolved = _ensure_writable_cache_dir(fallback, source="fallback")
    _log_hf_cache_dir(resolved)
    return resolved


def _ensure_writable_cache_dir(cache_dir: str, *, source: str) -> str:
    resolved = os.path.abspath(os.path.expanduser(cache_dir))
    if resolved.startswith("/users/"):
        raise OSError(
            "HF datasets cache directory resolved from "
            f"{source} points under /users/ (read-only on some clusters): {resolved}"
        )
    path = Path(resolved)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"HF datasets cache directory resolved from {source} is not writable: {resolved}"
        ) from exc
    if not os.access(resolved, os.W_OK):
        raise OSError(
            f"HF datasets cache directory resolved from {source} is not writable: {resolved}"
        )
    return resolved


def _log_hf_cache_dir(cache_dir: str) -> None:
    global _HF_DATASETS_CACHE_LOGGED
    if _HF_DATASETS_CACHE_LOGGED:
        return
    logging.info("HF datasets cache directory: %s", cache_dir)
    _HF_DATASETS_CACHE_LOGGED = True


def ensure_text_path(path: PathLike | None, *, description: str = "path") -> str:
    """Return ``path`` as a UTF-8 string.

    ``datasets.load_dataset`` expects ``str`` inputs for paths and patterns but
    the configuration pipeline may serialise filesystem locations as
    ``bytes`` (for example when read from environments that inject binary
    YAML scalars).  This helper normalises such values and provides clear
    errors when the resulting string would be empty.

    Args:
        path: Value describing a filesystem location.  May be a ``str``,
            ``bytes`` or :class:`os.PathLike` instance.
        description: Human readable label used in error messages.

    Raises:
        ValueError: If ``path`` is ``None`` or normalises to an empty string.

    Returns:
        Normalised string representation suitable for ``datasets`` helpers.
    """

    if path is None:
        raise ValueError(f"{description.capitalize()} must be provided")

    normalised = os.fsdecode(path).strip()
    if not normalised:
        raise ValueError(f"{description.capitalize()} must not be empty")
    return normalised


def resolve_pseudo_data_path(
    path: PathLike | None,
    *,
    env_var: str = "PSEUDO_DATA_PATH",
    description: str = "pseudo dataset path",
) -> str:
    """Return a normalised pseudo dataset path with environment fallback.
    
    Optionally rewrites paths that match ``PREFADAP_DATA_PREFIX`` to local
    storage roots for environment portability.

    Parameters
    ----------
    path:
        Primary location provided via configuration. May be ``None`` when the
        caller relies on environment injection (e.g. cluster launchers).
    env_var:
        Environment variable checked when ``path`` is missing or empty.
        Defaults to ``"PSEUDO_DATA_PATH"`` which mirrors the value exported by
        the HPC templates.
    description:
        Human readable label used when raising errors.

    Returns
    -------
    str
        UTF-8 encoded filesystem path suitable for :func:`datasets.load_dataset`.

    Raises
    ------
    ValueError
        If neither ``path`` nor ``env_var`` resolve to a usable location.
        
    Notes
    -----
    When ``PREFADAP_DATA_PREFIX`` is set, paths starting with that prefix are
    rewritten to ``PERSIST_ROOT`` or ``SCRATCHDIR`` (with a ``prefadap`` subdir)
    to keep configs portable across environments.
    """

    last_error: ValueError | None = None
    seen: set[str] = set()

    def _to_text(item: PathLike | None) -> str:
        if item is None:
            raise TypeError("cannot normalize None")
        try:
            return os.fsdecode(os.fspath(item))  # type: ignore[arg-type]
        except TypeError:
            return os.fsdecode(item)

    def _normalise_candidate(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [_to_text(v) for v in value if v is not None]
        return [_to_text(value)]

    def _iter_candidates() -> list[str]:
        candidates: list[str] = []

        for candidate in _normalise_candidate(path):
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)

        direct_env = os.environ.get(env_var)
        for candidate in _normalise_candidate(direct_env):
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)

        array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if array_id:
            suffix_key = f"{env_var}_{array_id}"
            suffix_value = os.environ.get(suffix_key)
            suffix_candidates = _normalise_candidate(suffix_value)
            for candidate in suffix_candidates:
                if candidate not in seen:
                    candidates.append(candidate)
                    seen.add(candidate)

        if not direct_env and not candidates:
            # When no unsuffixed variable is present, fall back to any suffixed
            # variants exported by job schedulers (e.g. PSEUDO_DATA_PATH_0).
            suffixed_keys = sorted(
                key for key in os.environ if key.startswith(f"{env_var}_")
            )
            for key in suffixed_keys:
                value = os.environ.get(key)
                for candidate in _normalise_candidate(value):
                    if candidate not in seen:
                        candidates.append(candidate)
                        seen.add(candidate)
                # Prefer the first suffixed match to avoid mixing multiple runs
                if candidates:
                    break

        return candidates

    for candidate in _iter_candidates():
        try:
            normalized_path = ensure_text_path(candidate, description=description)
            # Apply cluster-aware path rewriting for cross-cluster compatibility
            rewritten_path = _rewrite_cluster_path(normalized_path)
            return rewritten_path
        except ValueError as exc:
            last_error = exc
            continue

    error_msg = f"{description.capitalize()} must be provided via configuration or the {env_var} environment variable"
    if last_error is not None:
        raise ValueError(error_msg) from last_error
    raise ValueError(error_msg)

SUBREDDITS = {
    "relationship_advice",
    "AskReddit",
    "relationships",
    "tifu",
    "dating_advice",
}


def get_random_indices(
    length: int, proportion: float, seed: int | None = None
) -> np.ndarray:
    """Generate an array of random indices for selecting a subset of items.

    Args:
        length: Total number of items.
        proportion: Proportion of indices to sample (0 < proportion <= 1).
        seed: Optional random seed for reproducibility.

    Raises:
        ValueError: If ``proportion`` is not within the (0, 1] range.
    """
    if not (0 < proportion <= 1):
        raise ValueError("Proportion must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    num_samples = max(1, int(length * proportion))
    return rng.choice(length, size=num_samples, replace=False)


def get_field(example: Dict[str, Any], field: str, info_key: str | None = None) -> Any:
    """Retrieve a field from ``example`` possibly nested under ``info_key``."""
    return example[info_key][field] if info_key else example[field]


def make_filtered_dataset(
    dataset: DatasetDict,
    dataset_structured_subset: str,
    dataset_random_subset: float | None,
    info_key: str | None = None,
    seed: int | None = None,
) -> DatasetDict:
    """Filter a dataset by random or structured subset.

    If ``dataset_random_subset`` is provided, a random subset of the train and
    validation splits is returned.  If ``dataset_structured_subset`` corresponds
    to a known subreddit, additional in-domain/ out-of-domain splits are
    created.  Otherwise the dataset is returned unchanged.
    """
    if dataset_random_subset is not None:
        new_dataset = DatasetDict(
            {
                split: (
                    dataset[split].select(
                        get_random_indices(
                            len(dataset[split]), dataset_random_subset, seed
                        )
                    )
                    if split in {"train", "validation"}
                    else dataset[split]
                )
                for split in dataset.keys()
            }
        )
    elif dataset_structured_subset and dataset_structured_subset in SUBREDDITS:
        new_dataset = dataset.filter(
            lambda eg: get_field(eg, "subreddit", info_key) == dataset_structured_subset
        )
        new_dataset["full_validation"] = dataset["validation"]
        new_dataset["ood_validation"] = dataset["validation"].filter(
            lambda eg: get_field(eg, "subreddit", info_key) != dataset_structured_subset
        )
        new_dataset["full_test"] = dataset["test"]
        new_dataset["ood_test"] = dataset["test"].filter(
            lambda eg: get_field(eg, "subreddit", info_key) != dataset_structured_subset
        )
    else:
        new_dataset = dataset
    return new_dataset
