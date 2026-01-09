"""Utilities for preparing summarisation datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _compute_length_band(length: int) -> str:
    """Bucket a token length into a coarse length band."""

    bands = [128, 256, 512, 1024]
    for bound in bands:
        if length < bound:
            return f"<{bound}"
    return f">={bands[-1]}"


def _clean_text(text: str) -> str:
    """Remove simple HTML/boilerplate from ``text``."""

    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _load_examples(dataset: str) -> Iterable[Tuple[str, str]]:
    """Yield ``(split, text)`` pairs for ``dataset``."""

    try:  # pragma: no cover - datasets is optional
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise ImportError("datasets library is required") from exc

    from prefadap.data import (
        make_input_example_cnndm,
        make_input_example_tldr,
    )
    from prefadap.data.core import resolve_hf_datasets_cache_dir

    cache_dir = resolve_hf_datasets_cache_dir(None)

    if dataset == "tldr":
        ds = load_dataset("UCL-DARK/openai-tldr-filtered", cache_dir=cache_dir)
        for split, split_ds in ds.items():
            for ex in split_ds:
                prompt = make_input_example_tldr(ex["post"], ex["title"], ex["subreddit"])
                text = f"{prompt} {ex['summary']}"
                yield split, text
    elif dataset == "cnn_dailymail":
        ds = load_dataset("cnn_dailymail", "3.0.0", cache_dir=cache_dir)
        for split, split_ds in ds.items():
            for ex in split_ds:
                prompt = make_input_example_cnndm(ex["article"])
                text = f"{prompt} {ex['highlights']}"
                yield split, text
    else:  # pragma: no cover - defensive programming
        raise ValueError(f"Unknown dataset: {dataset}")


def _write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def strip_boilerplate(text: str) -> str:
    """Return a cleaned version of ``text`` suitable for summarisation corpora."""

    return _clean_text(text)


def deduplicate(records: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Deduplicate records by exact text content, preserving first occurrence."""

    seen: set[str] = set()
    unique: List[Dict[str, str]] = []
    for rec in records:
        text = rec.get("text", "")
        if not text:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(rec)
    return unique


def prepare_corpus(dataset: str, out_dir: Path | str, *, apply_dedup: bool = True) -> List[Dict[str, str]]:
    """Load and clean examples for ``dataset``, writing a JSONL corpus to disk."""

    records: List[Dict[str, str]] = []
    for split, text in _load_examples(dataset):
        cleaned = strip_boilerplate(text)
        if not cleaned:
            continue
        records.append({"split": split, "text": cleaned})

    if apply_dedup:
        records = deduplicate(records)

    out_path = Path(out_dir) / "clean" / f"{dataset}.jsonl"
    _write_jsonl(out_path, records)
    return records


def cmd_clean(args: argparse.Namespace) -> None:
    """Load raw dataset text, clean it, and write the cleaned JSONL file."""

    prepare_corpus(args.dataset, args.out_dir, apply_dedup=False)


def cmd_dedup(args: argparse.Namespace) -> None:
    """Deduplicate a cleaned corpus and write the resulting JSONL file."""

    in_path = Path(args.out_dir) / "clean" / f"{args.dataset}.jsonl"
    out_path = Path(args.out_dir) / "dedup" / f"{args.dataset}.jsonl"

    records: List[Dict[str, str]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    _write_jsonl(out_path, deduplicate(records))

def _pack_records(records: List[Dict[str, str]], context: int) -> Tuple[List[str], List[Dict[str, str]]]:
    packed_texts: List[str] = []
    indexes: List[Dict[str, str]] = []

    current: List[str] = []
    current_len = 0
    current_split = None

    for rec in records:
        tokens = rec["text"].split()
        length = len(tokens)
        split = rec["split"]
        if current_split is None:
            current_split = split

        if current_len + length > context or split != current_split:
            text = "\n\n".join(current)
            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            indexes.append(
                {
                    "sha256": sha,
                    "split": current_split,
                    "length_band": _compute_length_band(current_len),
                }
            )
            packed_texts.append(text)
            current = [rec["text"]]
            current_len = length
            current_split = split
        else:
            current.append(rec["text"])
            current_len += length

    if current:
        text = "\n\n".join(current)
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        indexes.append(
            {
                "sha256": sha,
                "split": current_split,
                "length_band": _compute_length_band(current_len),
            }
        )
        packed_texts.append(text)

    return packed_texts, indexes


def cmd_pack(args: argparse.Namespace) -> None:
    """Pack cleaned samples and write processed data and index files."""

    in_path = Path(args.out_dir) / "clean" / f"{args.dataset}.jsonl"
    processed_path = Path(args.out_dir) / "processed" / f"{args.dataset}.jsonl"
    index_path = Path(args.out_dir) / "indexes" / f"{args.dataset}.jsonl"

    records: List[Dict[str, str]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    packed, indexes = _pack_records(records, args.context_length)

    _write_jsonl(processed_path, ({"text": t} for t in packed))
    _write_jsonl(index_path, indexes)


__all__ = [
    "prepare_corpus",
    "strip_boilerplate",
    "deduplicate",
    "cmd_dedup",
    "cmd_clean",
    "cmd_pack",
]
