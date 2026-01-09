"""Reddit TL;DR dataset builder.

This module exposes a :func:`build` function returning a
``datasets.Dataset`` with normalised fields and deterministic ordering.
The dataset is loaded via :func:`datasets.load_dataset` either from the
HuggingFace hub or a local mirror.

The builder validates the expected schema, performs Unicode
normalisation and optionally checks that examples fit within specified
token limits.
"""

from __future__ import annotations

import hashlib
import unicodedata
from typing import Mapping

from prefadap.data.core import resolve_hf_datasets_cache_dir

try:  # pragma: no cover - datasets is optional in the execution env
    from datasets import Dataset, load_dataset
except Exception:  # pragma: no cover - graceful failure if missing
    Dataset = None  # type: ignore


# Expected columns present in the raw dataset
REQUIRED_FIELDS = {"post", "summary", "title", "subreddit"}


def _normalise(text: str) -> str:
    """Return Unicode NFKC normalised ``text``."""

    return unicodedata.normalize("NFKC", text)


def _validate_record(
    record: Mapping[str, str],
    *,
    tokenizer=None,
    max_source_length: int | None = None,
    max_target_length: int | None = None,
    group_field: str | None = None,
) -> Mapping[str, str]:
    """Validate and clean a single TL;DR record.

    Parameters
    ----------
    record:
        Original example from the dataset.
    tokenizer:
        Optional tokenizer providing ``__call__`` for length checks.
    max_source_length, max_target_length:
        If given, ensure encoded inputs do not exceed these token counts.
    """

    missing = REQUIRED_FIELDS.difference(record)
    if missing:
        raise ValueError(f"Missing fields: {', '.join(sorted(missing))}")

    post = _normalise(str(record["post"]))
    summary = _normalise(str(record["summary"]))
    title = _normalise(str(record["title"]))
    subreddit = _normalise(str(record["subreddit"]))

    # Derive optional group identifier
    group_id = None
    if group_field == "subreddit":
        group_id = subreddit
    elif group_field == "length":
        group_id = "long" if len(post.split()) > 100 else "short"
    elif group_field and group_field in record:
        group_id = _normalise(str(record[group_field]))

    if tokenizer is not None:
        if max_source_length is not None:
            length = len(tokenizer(post)["input_ids"])
            if length > max_source_length:
                raise ValueError(
                    f"Post length {length} exceeds max_source_length={max_source_length}"
                )
        if max_target_length is not None:
            length = len(tokenizer(summary)["input_ids"])
            if length > max_target_length:
                raise ValueError(
                    f"Summary length {length} exceeds max_target_length={max_target_length}"
                )

    sha = hashlib.sha256((post + summary).encode("utf-8")).hexdigest()

    return {
        "post": post,
        "summary": summary,
        "title": title,
        "subreddit": subreddit,
        "sha256": sha,
        "group_id": group_id,
    }


def build(
    split: str,
    *,
    cache_dir: str | None = None,
    tokenizer=None,
    max_source_length: int | None = None,
    max_target_length: int | None = None,
    group_field: str | None = None,
) -> "Dataset":
    """Build and return the TL;DR dataset for ``split``.

    The examples are processed deterministically and sorted by their
    SHA256 checksum so that repeated runs yield the same ordering.
    """

    if Dataset is None:  # pragma: no cover - datasets missing
        raise ImportError("datasets library is required to build TL;DR data")

    cache_dir = resolve_hf_datasets_cache_dir(cache_dir)
    ds = load_dataset(
        "UCL-DARK/openai-tldr-filtered", split=split, cache_dir=cache_dir
    )

    def mapper(record: Mapping[str, str]) -> Mapping[str, str]:
        return _validate_record(
            record,
            tokenizer=tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            group_field=group_field,
        )

    ds = ds.map(mapper)
    ds = ds.sort("sha256")
    return ds


__all__ = ["build"]
