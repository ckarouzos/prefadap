"""Length band utilities for summarisation datasets.

This module defines dataset-specific length bands and a helper function to
clip generated summaries to these bands while reporting a penalty for
violations. The penalties are proportional to the relative amount by which
the text falls outside the allowed band.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

# Definition of length bands for supported datasets.
# Units can be ``sentences``, ``words`` or ``tokens`` (whitespace tokens).
LENGTH_BANDS: Dict[str, Dict[str, int]] = {
    # CNN/DailyMail summaries are expected to be between 3 and 5 sentences.
    "cnn_dailymail": {"unit": "sentences", "min": 3, "max": 5},
    "cnndm": {"unit": "sentences", "min": 3, "max": 5},
    "cnn_dm": {"unit": "sentences", "min": 3, "max": 5},
    "reddit": {"unit": "tokens", "max": 160},
}


def _split_sentences(text: str) -> list[str]:
    """Return a list of sentences using a simple punctuation-based splitter."""

    # Split on punctuation followed by whitespace.  This heuristic is light-weight
    # and avoids extra dependencies (e.g. nltk).
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def enforce_length_band(text: str, dataset: str) -> Tuple[str, float]:
    """Clip ``text`` to the dataset's length band and return a penalty.

    Parameters
    ----------
    text:
        Generated summary text.
    dataset:
        Identifier of the dataset; case-insensitive.  If the dataset is
        unknown, ``text`` is returned unchanged with zero penalty.

    Returns
    -------
    tuple of (str, float)
        The possibly clipped text and a non-negative penalty representing the
        proportional violation of the band (0 when within bounds).
    """

    band = LENGTH_BANDS.get(dataset.lower())
    if band is None:
        return text, 0.0

    unit = band["unit"]
    penalty = 0.0

    if unit == "sentences":
        sentences = _split_sentences(text)
        count = len(sentences)
        min_len = band.get("min", 0)
        max_len = band.get("max", count)
        if count > max_len:
            penalty = (count - max_len) / max_len
            sentences = sentences[:max_len]
        elif count < min_len:
            penalty = (min_len - count) / max_len if max_len else 0.0
        return " ".join(sentences), penalty

    if unit == "words":
        words = text.split()
        count = len(words)
        min_len = band.get("min", 0)
        max_len = band.get("max", count)
        if count > max_len:
            penalty = (count - max_len) / max_len
            words = words[:max_len]
        elif count < min_len:
            penalty = (min_len - count) / max_len if max_len else 0.0
        return " ".join(words), penalty

    if unit == "tokens":
        tokens = text.split()
        count = len(tokens)
        min_len = band.get("min", 0)
        max_len = band.get("max", count)
        if max_len and count > max_len:
            penalty = (count - max_len) / max_len
            tokens = tokens[:max_len]
        elif min_len and count < min_len:
            penalty = (min_len - count) / min_len
        return " ".join(tokens), penalty

    return text, 0.0


__all__ = ["LENGTH_BANDS", "enforce_length_band"]
