"""Lightweight manifest runner for training pipelines.

The CLI is intentionally minimal and primarily exists so that the unit tests
can exercise the manifest resolution logic introduced for the SAFETY/PKU track.
When the ``--dry-run`` flag is supplied the command simply prints a JSON
summary describing which configuration files would be executed and which
datasets they target.  Without ``--dry-run`` it forwards each entry to
``prefadap.cli.run_training``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

from prefadap.config import load_dict

SAFETY_DATASET_DEFAULTS = {
    "sft": "pku_sft",
    "dpo": "pku_preference",
    "orpo": "pku_preference",
    "kto": "pku_kto",
}

OBJECTIVE_FALLBACKS = {
    "sft": ("sft",),
    "dpo": ("dpo",),
    "orpo": ("orpo", "dpo"),
    "kto": ("kto",),
}


@dataclass
class ManifestEntry:
    config_path: Path
    objective: str
    dataset_name: str | None
    apply_dataset_override: bool = False


def _normalise_objective(raw: str | None) -> str:
    if not raw:
        raise ValueError("Manifest entry is missing 'experiment_type'/'objective'")
    return raw.strip().lower()


def _resolve_manifest_path(pair_id: str, objective: str, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit

root = Path(os.environ.get("PREFADAP_MANIFEST_ROOT", Path(__file__).resolve().parents[2] / "manifests"))
    parts = [p for p in pair_id.split("/") if p]
    manifest_dir = root.joinpath(*parts)
    if not manifest_dir.exists():
        raise FileNotFoundError(f"Manifest directory {manifest_dir} does not exist")

    candidates = OBJECTIVE_FALLBACKS.get(objective, (objective,))
    for name in candidates:
        path = manifest_dir / f"{name}.txt"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No manifest file found for objective '{objective}' under {manifest_dir}"
    )


def _iter_manifest_entries(path: Path) -> Iterator[Path]:
    text = path.read_text(encoding="utf-8")
    base = path.parent
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = (base / candidate).resolve()
        yield candidate


def _load_manifest(path: Path) -> List[Path]:
    return list(_iter_manifest_entries(path))


def _summarise_entry(config_path: Path, pair_id: str | None) -> ManifestEntry:
    data = load_dict(config_path)
    objective = _normalise_objective(data.get("experiment_type") or data.get("objective"))
    dataset_name = data.get("dataset_name")
    apply_override = False

    if pair_id and pair_id.upper().startswith("SAFETY/PKU"):
        if not dataset_name:
            dataset_name = SAFETY_DATASET_DEFAULTS.get(objective)
            apply_override = dataset_name is not None

    return ManifestEntry(
        config_path=config_path,
        objective=objective,
        dataset_name=dataset_name,
        apply_dataset_override=apply_override,
    )


def _run_training(entry: ManifestEntry, quiet: bool = False) -> None:
    from prefadap.cli import run_training as training_cli

    argv_backup = list(sys.argv)
    override_path: Path | None = None
    try:
        args = [argv_backup[0], entry.objective, "--config", str(entry.config_path)]
        if entry.apply_dataset_override and entry.dataset_name:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
                json.dump({"dataset_name": entry.dataset_name}, handle)
                handle.flush()
                override_path = Path(handle.name)
            args.append(str(override_path))
        if quiet and "--quiet" not in args:
            args.append("--quiet")
        sys.argv = args
        training_cli.main()
    finally:
        sys.argv = argv_backup
        if override_path is not None:
            try:
                override_path.unlink()
            except FileNotFoundError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Path to manifest file (optional when --pair_id is used)")
    parser.add_argument("--pair_id", help="Logical pair identifier such as SAFETY/PKU")
    parser.add_argument(
        "--objective",
        help="Optional objective override (sft, dpo, orpo, kto). Required when --pair_id is used without --manifest",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not launch training; print summary instead")
    parser.add_argument("--quiet", action="store_true", help="Forward --quiet to prefadap.cli.run_training")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    objective = (args.objective or "").strip().lower()
    if not args.manifest and not args.pair_id:
        parser.error("either --manifest or --pair_id must be supplied")
    if args.pair_id and not objective:
        parser.error("--objective is required when using --pair_id")

    manifest_path = _resolve_manifest_path(args.pair_id, objective, args.manifest) if args.pair_id else args.manifest
    entries = [_summarise_entry(path, args.pair_id) for path in _load_manifest(manifest_path)]

    if args.dry_run:
        payload = {
            "manifest": str(manifest_path),
            "entries": [
                {
                    "config": str(entry.config_path),
                    "objective": entry.objective,
                    "dataset_name": entry.dataset_name,
                }
                for entry in entries
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    for entry in entries:
        _run_training(entry, quiet=args.quiet)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
