"""CNN/DailyMail dataset builder.

The builder mirrors :mod:`data.tldr` and provides a deterministic and
validated view over the benchmark dataset.  Each example yields the
article and its highlights along with a SHA256 checksum for stable
ordering.
"""

from __future__ import annotations

import hashlib
import unicodedata
from typing import Mapping

from prefadap.data.core import resolve_hf_datasets_cache_dir

try:  # pragma: no cover - datasets optional
    from datasets import Dataset, load_dataset
except Exception:  # pragma: no cover - graceful
    Dataset = None  # type: ignore


REQUIRED_FIELDS = {"article", "highlights"}


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _validate_record(
    record: Mapping[str, str],
    *,
    tokenizer=None,
    max_source_length: int | None = None,
    max_target_length: int | None = None,
    group_field: str | None = None,
) -> Mapping[str, str]:
    missing = REQUIRED_FIELDS.difference(record)
    if missing:
        raise ValueError(f"Missing fields: {', '.join(sorted(missing))}")

    article = _normalise(str(record["article"]))
    highlights = _normalise(str(record["highlights"]))

    group_id = None
    if group_field == "length":
        group_id = "long" if len(article.split()) > 400 else "short"
    elif group_field and group_field in record:
        group_id = _normalise(str(record[group_field]))

    if tokenizer is not None:
        if max_source_length is not None:
            length = len(tokenizer(article)["input_ids"])
            if length > max_source_length:
                raise ValueError(
                    f"Article length {length} exceeds max_source_length={max_source_length}"
                )
        if max_target_length is not None:
            length = len(tokenizer(highlights)["input_ids"])
            if length > max_target_length:
                raise ValueError(
                    f"Summary length {length} exceeds max_target_length={max_target_length}"
                )

    sha = hashlib.sha256((article + highlights).encode("utf-8")).hexdigest()

    return {
        "article": article,
        "highlights": highlights,
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
    """Build the CNN/DailyMail dataset for ``split``."""

    if Dataset is None:  # pragma: no cover - datasets missing
        raise ImportError("datasets library is required to build CNN/DailyMail data")

    cache_dir = resolve_hf_datasets_cache_dir(cache_dir)
    ds = load_dataset("cnn_dailymail", "3.0.0", split=split, cache_dir=cache_dir)

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
