"""Helpers for loading the Stanford Human Preferences (SHP) dataset."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping

from datasets import Dataset, DatasetDict

from .registry import register_dataset

_SHP_DATA_ROOT = Path("prefadapt_data") / "SHP"

_SHP_DOMAINS: tuple[str, ...] = (
    "askacademia",
    "askanthropology",
    "askbaking",
    "askcarguys",
    "askculinary",
    "askdocs",
    "askengineers",
    "askhistorians",
    "askhr",
    "askphilosophy",
    "askphysics",
    "askscience",
    "asksciencefiction",
    "asksocialscience",
    "askvet",
    "changemyview",
    "explainlikeimfive",
    "legaladvice",
    "all",
    "smoke",
)


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _normalise_domain(value: str | None) -> str | None:
    if value is None:
        return None
    token = value.strip().lower()
    if not token:
        return None
    if token.startswith("shp_"):
        token = token[len("shp_") :]
    if token == "shp":
        return None
    return token


def _resolve_domain(args: Any) -> str:
    for candidate in (
        getattr(args, "dataset_name", None),
        getattr(args, "dataset", None),
        getattr(args, "shp_domain", None),
    ):
        domain = _normalise_domain(candidate)
        if domain is None:
            continue
        if domain not in _SHP_DOMAINS:
            raise ValueError(f"Unknown SHP domain '{candidate}'")
        return domain
    raise ValueError(
        "SHP loaders require a dataset name like 'shp_askculinary' or a "
        "dataset/shp_domain field identifying the subreddit domain."
    )


def _resolve_paths(args: Any, domain: str) -> dict[str, Path]:
    import os
    
    path_spec = getattr(args, "dataset_path", None)
    if isinstance(path_spec, Mapping):
        return {split: Path(path) for split, path in path_spec.items()}

    if path_spec is not None:
        candidate = Path(path_spec)
        if candidate.is_dir():
            root = candidate
        else:
            return {"train": candidate}
    else:
        data_dir = Path(getattr(args, "data_dir", "data"))
        # If data_dir is relative and REPO_ROOT is set, resolve against REPO_ROOT
        # This handles the case where jobs run from a different working directory
        if not data_dir.is_absolute():
            repo_root = os.environ.get("REPO_ROOT")
            if repo_root:
                data_dir = Path(repo_root) / data_dir
        root = data_dir / _SHP_DATA_ROOT / domain
        
        # If the computed path doesn't exist and data_dir is absolute,
        # fall back to REPO_ROOT/data to handle cases where data_dir
        # points to an HF cache directory but SHP data is in the repository
        if not root.exists() and data_dir.is_absolute():
            repo_root = os.environ.get("REPO_ROOT")
            if repo_root:
                fallback_root = Path(repo_root) / "data" / _SHP_DATA_ROOT / domain
                if fallback_root.exists():
                    root = fallback_root

    if root.is_file():
        return {"train": root}
    if not root.exists():
        raise FileNotFoundError(
            f"Expected SHP data for domain '{domain}' under {root}, but the path does not exist"
        )

    mapping: dict[str, Path] = {}
    for split in ("train", "validation", "test"):
        candidate = root / f"{split}.jsonl"
        if candidate.exists():
            mapping[split] = candidate

    if not mapping:
        for jsonl_file in sorted(root.glob("*.jsonl")):
            mapping[jsonl_file.stem] = jsonl_file

    if not mapping:
        raise FileNotFoundError(
            f"No JSONL files found for SHP domain '{domain}' under {root}"
        )

    return mapping


def _extract_prompt(record: Mapping[str, Any]) -> str:
    history = record.get("history")
    if isinstance(history, str) and history.strip():
        return history.strip()
    if isinstance(history, list):
        parts = [str(item).strip() for item in history if str(item).strip()]
        if parts:
            return "\n\n".join(parts)
    for key in ("post", "question", "context"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_label(record: Mapping[str, Any]) -> int | None:
    label: Any = record.get("labels")
    if isinstance(label, list) and label:
        label = label[0]
    if isinstance(label, (int, float)):
        return 1 if int(label) == 1 else 0
    if isinstance(label, str):
        token = label.strip().lower()
        if token in {"1", "a"}:
            return 1
        if token in {"0", "b"}:
            return 0
    return None


def _extract_preference_pair(record: Mapping[str, Any]) -> tuple[str, str] | None:
    label = _extract_label(record)
    if label is None:
        return None

    ref_a = record.get("human_ref_A")
    ref_b = record.get("human_ref_B")
    if not isinstance(ref_a, str) or not ref_a.strip():
        return None
    if not isinstance(ref_b, str) or not ref_b.strip():
        return None

    chosen, rejected = (ref_a.strip(), ref_b.strip()) if label == 1 else (ref_b.strip(), ref_a.strip())
    return chosen, rejected


def _materialise_dataset(records_by_split: Mapping[str, Iterable[Dict[str, Any]]]) -> DatasetDict:
    datasets: Dict[str, Dataset] = {}
    for split, records in records_by_split.items():
        materialised = list(records)
        if not materialised:
            continue
        datasets[split] = Dataset.from_list(materialised)
    if not datasets:
        raise ValueError("SHP loader received no valid records")
    return DatasetDict(datasets)


def _load_dataset(
    paths: Mapping[str, Path],
    builder: Callable[[Mapping[str, Any]], Dict[str, Any] | None],
) -> DatasetDict:
    records: dict[str, Iterable[Dict[str, Any]]] = {}

    for split, path in paths.items():

        def _iter(path: Path = path) -> Iterator[Dict[str, Any]]:
            for raw in _read_jsonl(path):
                record = builder(raw)
                if record is None:
                    continue
                yield record

        records[split] = _iter()

    return _materialise_dataset(records)


def load_shp_sft(args: Any, cache_dir: str | None = None) -> DatasetDict:
    del cache_dir  # Unused, for API compatibility
    domain = _resolve_domain(args)
    paths = _resolve_paths(args, domain)

    def _builder(raw: Mapping[str, Any]) -> Dict[str, Any] | None:
        prompt = _extract_prompt(raw)
        pair = _extract_preference_pair(raw)
        if not prompt or pair is None:
            return None
        chosen, _ = pair
        return {"prompt": prompt, "completion": chosen}

    return _load_dataset(paths, _builder)


def load_shp_preference(args: Any, cache_dir: str | None = None) -> DatasetDict:
    del cache_dir
    domain = _resolve_domain(args)
    paths = _resolve_paths(args, domain)

    def _builder(raw: Mapping[str, Any]) -> Dict[str, Any] | None:
        prompt = _extract_prompt(raw)
        pair = _extract_preference_pair(raw)
        if not prompt or pair is None:
            return None
        chosen, rejected = pair
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    return _load_dataset(paths, _builder)


for _domain in _SHP_DOMAINS:
    _dataset_name = f"shp_{_domain}"

    @register_dataset(_dataset_name, "summarisation")
    def _register_sft_loader(
        args: Any,
        cache_dir: str | None = None,
        *,
        __domain: str = _domain,
        __dataset_name: str = _dataset_name,
    ) -> DatasetDict:
        if getattr(args, "dataset_name", None) is None:
            setattr(args, "dataset_name", __dataset_name)
        if getattr(args, "dataset", None) is None:
            setattr(args, "dataset", __domain)
        return load_shp_sft(args, cache_dir=cache_dir)

    @register_dataset(_dataset_name, "dpo")
    def _register_preference_loader(
        args: Any,
        cache_dir: str | None = None,
        *,
        __domain: str = _domain,
        __dataset_name: str = _dataset_name,
    ) -> DatasetDict:
        if getattr(args, "dataset_name", None) is None:
            setattr(args, "dataset_name", __dataset_name)
        if getattr(args, "dataset", None) is None:
            setattr(args, "dataset", __domain)
        return load_shp_preference(args, cache_dir=cache_dir)


__all__ = ["load_shp_sft", "load_shp_preference"]

