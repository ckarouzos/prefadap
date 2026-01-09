"""Pseudo-label generation and selection pipeline.

This module exposes a light-weight pseudo-labelling workflow consisting of
three stages: candidate generation, scoring and final label selection.  The
implementation intentionally avoids heavyweight model dependencies so that it
can run in constrained test environments.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List

from prefadap.utils.seeding import set_seed


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration shared across pipeline stages."""

    run_id: str
    dataset: Path
    k: int = 2
    scoring_model: str = "random"
    tau: float = 0.8
    max_pairs: int = 1
    seed: int = 0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _stream_jsonl(path: Path) -> Iterator[Dict[str, str]]:
    """Yield records from a JSONL file one at a time."""

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


# ---------------------------------------------------------------------------
# Pipeline implementation
# ---------------------------------------------------------------------------


class PseudoLabelPipeline:
    """High level orchestrator for pseudo-labelling."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        set_seed(self.cfg.seed)

    # -- Stage 1 -----------------------------------------------------------
    def generate_candidates(self, output: Path) -> None:
        """Generate candidate responses for each prompt."""

        reservoir: List[str] = []
        max_reservoir = 1000
        with self.cfg.dataset.open("r", encoding="utf-8") as src, output.open(
            "w", encoding="utf-8"
        ) as dst:
            for idx, line in enumerate(src):
                rec = json.loads(line)
                meta = {
                    key: rec[key]
                    for key in ("pair_id", "metadata")
                    if key in rec
                }
                base = rec.get("chosen", "")
                for i in range(self.cfg.k):
                    if self.cfg.scoring_model == "random" and reservoir:
                        text = random.choice(reservoir)
                    else:
                        text = f"{base} candidate {i}"
                    dst.write(
                        json.dumps(
                            {
                                "prompt": rec["prompt"],
                                "candidate": text,
                                **meta,
                            }
                        )
                        + "\n"
                    )

                if len(reservoir) < max_reservoir:
                    reservoir.append(base)
                else:
                    j = random.randint(0, idx)
                    if j < max_reservoir:
                        reservoir[j] = base

    # -- Stage 2 -----------------------------------------------------------
    def score_candidates(self, candidates: Path, output: Path) -> None:
        """Assign heuristic scores to each candidate."""

        with candidates.open("r", encoding="utf-8") as src, output.open(
            "w", encoding="utf-8"
        ) as dst:
            for line in src:
                rec = json.loads(line)
                meta = {
                    key: rec[key]
                    for key in ("pair_id", "metadata")
                    if key in rec
                }
                text = rec["candidate"]
                if self.cfg.scoring_model == "random":
                    score = random.random()
                else:
                    score = min(len(text) / 100.0, 1.0)
                dst.write(
                    json.dumps(
                        {
                            "prompt": rec["prompt"],
                            "candidate": text,
                            "score": score,
                            **meta,
                        }
                    )
                    + "\n"
                )

    # -- Stage 3 -----------------------------------------------------------
    def select_labels(self, scored: Path, output: Path) -> None:
        """Select high quality pseudo pairs and merge with original data."""

        scored_stream = _stream_jsonl(scored)
        with self.cfg.dataset.open("r", encoding="utf-8") as data, output.open(
            "w", encoding="utf-8"
        ) as out:
            for line in data:
                rec = json.loads(line)
                out.write(json.dumps(rec) + "\n")

                cands = []
                for _ in range(self.cfg.k):
                    try:
                        cand = next(scored_stream)
                    except StopIteration:
                        break
                    if cand["prompt"] != rec["prompt"]:
                        continue
                    if cand["score"] >= self.cfg.tau:
                        cands.append(cand)

                cands.sort(key=lambda r: r["score"], reverse=True)
                for cand in cands[: self.cfg.max_pairs]:
                    meta = {
                        key: cand[key]
                        for key in ("pair_id", "metadata")
                        if key in cand
                    }
                    out.write(
                        json.dumps(
                            {
                                "prompt": rec["prompt"],
                                "chosen": rec["chosen"],
                                "rejected": cand["candidate"],
                                "weight": 1.0,
                                **meta,
                            }
                        )
                        + "\n"
                    )

    # -- Orchestration ----------------------------------------------------
    def run(self, output: Path) -> None:
        tmp_candidates = output.with_suffix(".candidates.jsonl")
        tmp_scored = output.with_suffix(".scored.jsonl")
        self.generate_candidates(tmp_candidates)
        self.score_candidates(tmp_candidates, tmp_scored)
        self.select_labels(tmp_scored, output)


__all__ = ["PipelineConfig", "PseudoLabelPipeline", "_stream_jsonl"]

