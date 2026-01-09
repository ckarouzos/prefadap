#!/usr/bin/env python3
"""Nano (OpenAI) Batch Judge harness.

This CLI prepares Batch API request files for running GPT-5-nano as a
pairwise summarisation judge over full evaluation splits (e.g., TL;DR and
CNN/DM). It also parses Batch results and aggregates win-rate metrics.

Design goals:
- Offline-safe: preparation and parsing do not perform network calls.
- Minimal dependencies: only Python stdlib.
- Compatible with the OpenAI Chat Completions Batch API shape.

Typical workflow:
1) Prepare request files for one or more datasets
   python -m prefadap.cli.nano_batch_judge prepare --run_dir runs/exp1 --datasets tldr cnndm

   This writes NDJSON request files under runs/exp1/nano_batch/*_requests.jsonl
   and a manifest mapping custom_ids to original record metadata.

2) Submit using Python SDK or submit script (outside this module, on a networked machine)
   Using the submit script:
   python scripts/submit_openai_judge_requests.py runs/exp1/nano_batch/tldr_requests.jsonl --wait --download-output

   Or using Python SDK directly:
   from openai import OpenAI
   client = OpenAI()
   with open("runs/exp1/nano_batch/tldr_requests.jsonl", "rb") as f:
       batch_input_file = client.files.create(file=f, purpose="batch")
   batch = client.batches.create(input_file_id=batch_input_file.id, endpoint="/v1/chat/completions", completion_window="24h")
   # Save batch.id for later retrieval

   (Repeat for cnndm.)

3) After the batch completes, download results with Python SDK and parse them
   batch = client.batches.retrieve("<batch_id>")
   file_response = client.files.content(batch.output_file_id)
   with open("runs/exp1/nano_batch/tldr_results.jsonl", "w") as f: f.write(file_response.text)
   
   python -m prefadap.cli.nano_batch_judge parse --run_dir runs/exp1 --dataset tldr \
       --results-file runs/exp1/nano_batch/tldr_results.jsonl

4) Aggregate/score parsed results
   python -m prefadap.cli.nano_batch_judge score --run_dir runs/exp1 --dataset tldr

Notes
-----
The judge prompt uses four criteria (faithfulness, coverage, style, concision)
and requires strict JSON output for robust parsing. The output is converted to
per-criterion win rates and an overall win rate using the same aggregation
logic used elsewhere in the repo.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from prefadap.prompts.judges.openai_nano_prompting import (
    PLACEHOLDERS,
    SHP_PLACEHOLDERS,
    SAFETY_PLACEHOLDERS,
    OpenAIJudgePromptFactory,
)
from prefadap.utils.batch_requests import (
    BatchRequestWriter,
    MAX_BYTES_PER_FILE,
    MAX_REQUESTS_PER_FILE,
)

try:  # pragma: no cover - optional dependency in lightweight environments
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore


@dataclass(frozen=True)
class NanoBatchProfile:
    """Pricing profile for GPT-5-nano Batch jobs."""

    name: str
    input_cost_per_million: float
    cached_input_cost_per_million: float
    output_cost_per_million: float
    currency: str = "USD"
    default_output_tokens: int = 256
    notes: str = ""

    @property
    def input_cost_per_token(self) -> float:
        return self.input_cost_per_million / 1_000_000

    @property
    def cached_input_cost_per_token(self) -> float:
        return self.cached_input_cost_per_million / 1_000_000

    @property
    def output_cost_per_token(self) -> float:
        return self.output_cost_per_million / 1_000_000

    def estimate_cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_ratio: float = 0.0,
    ) -> float:
        if input_tokens < 0 or output_tokens < 0:
            return 0.0
        cached_ratio = min(max(cached_ratio, 0.0), 1.0)
        cached_tokens = int(round(input_tokens * cached_ratio))
        fresh_tokens = max(input_tokens - cached_tokens, 0)
        return (
            fresh_tokens * self.input_cost_per_token
            + cached_tokens * self.cached_input_cost_per_token
            + output_tokens * self.output_cost_per_token
        )


DEFAULT_NANO_BATCH_PROFILE = NanoBatchProfile(
    name="gpt-5-nano-batch-pricing",
    input_cost_per_million=0.025,
    cached_input_cost_per_million=0.0025,
    output_cost_per_million=0.20,
    notes="OpenAI Batch pricing (Feb 2025).",
)


# ---------------------------- Helpers and IO ----------------------------


def _build_token_counter() -> Callable[[str], int]:
    encoder = None
    if tiktoken is not None:
        try:  # pragma: no cover - may fail without model files
            encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:  # pragma: no cover
            encoder = None

    if encoder is not None:
        return lambda text: len(encoder.encode(text or ""))

    return lambda text: len((text or "").split())


def _discover_output_file(run_dir: Path, dataset: str) -> Path:
    """Find a generated outputs JSONL file for dataset (preferring test split).

    The generation CLI writes ``{dataset}_{split}.jsonl`` into the run dir.
    We first try ``{dataset}_test.jsonl`` then fall back to the first match.
    """
    preferred = run_dir / f"{dataset}_test.jsonl"
    if preferred.exists():
        return preferred
    # Fall back to any file that looks like the dataset
    candidates = sorted(run_dir.glob(f"{dataset}_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"Could not find outputs for dataset '{dataset}' under {run_dir}")
    return candidates[0]


def _load_records(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _first_generation(rec: Dict[str, Any]) -> str:
    if isinstance(rec.get("generations"), list) and rec["generations"]:
        return str(rec["generations"][0])
    if "prediction" in rec:
        return str(rec["prediction"])  # alternate schema
    if "output" in rec:
        return str(rec["output"])  # alternate schema
    return ""


def _reference_text(rec: Dict[str, Any]) -> str:
    for key in ("reference", "target", "summary", "highlights"):
        if key in rec and rec[key]:
            return str(rec[key])
    return ""


def _prompt_text(rec: Dict[str, Any]) -> str:
    for key in ("prompt", "input", "document", "article", "text"):
        if key in rec and rec[key]:
            return str(rec[key])
    return ""


def _extract_shp_responses(rec: Dict[str, Any]) -> Tuple[str, str]:
    response_a = ""
    response_b = ""
    for key in ("response_a", "response_0", "responseA", "response0"):
        if key in rec and rec[key]:
            response_a = str(rec[key])
            break
    for key in ("response_b", "response_1", "responseB", "response1"):
        if key in rec and rec[key]:
            response_b = str(rec[key])
            break
    return response_a, response_b


def _is_shp_dataset(dataset: str) -> bool:
    ds = dataset.lower()
    return ds.startswith("shp") or ds in {"askengineers", "askculinary"}


# ------------------------- Request construction -------------------------


@dataclass
class PreparedItem:
    custom_id: str
    dataset: str
    index: int
    pred_len: int
    ref_len: int
    prompt_tokens: int
    request_file: str
    reference_source: str = "reference"


def _safe_relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def prepare_requests(
    run_dir: Path,
    datasets: List[str],
    *,
    model: str = "gpt-5-nano",
    system_prompt: Optional[str] = None,
    user_template: Optional[str] = None,
    cached_input_ratio: float = 0.0,
    estimated_output_tokens: Optional[int] = None,
    max_requests_per_file: int = MAX_REQUESTS_PER_FILE,
    max_bytes_per_file: int = MAX_BYTES_PER_FILE,
    comparison_run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Prepare NDJSON request files for each dataset.

    Requests are chunked automatically to respect Batch API limits (50k records
    and 200 MiB per file by default). Static judge instructions always precede
    dynamic content to maximise prompt caching efficiency.

    When ``comparison_run_dir`` is provided, candidate B comes from the
    ``comparison_run_dir`` outputs instead of the on-disk ``reference`` field.
    This enables head-to-head evaluation of two model runs without requiring
    ground-truth references.
    """
    if user_template is not None:
        placeholder_sets = (PLACEHOLDERS, SAFETY_PLACEHOLDERS, SHP_PLACEHOLDERS)
        if not any(all(token in user_template for token in tokens) for tokens in placeholder_sets):
            raise ValueError(
                "User template missing required placeholders. "
                "Expected {instruction}, {output_1}, {output_2}; "
                "{prompt}, {response}; or {prompt}, {response_a}, {response_b}."
            )

    if not 0.0 <= cached_input_ratio <= 1.0:
        raise ValueError("cached_input_ratio must be between 0 and 1")
    if estimated_output_tokens is None:
        output_tokens_per_request = DEFAULT_NANO_BATCH_PROFILE.default_output_tokens
    else:
        if estimated_output_tokens < 0:
            raise ValueError("estimated_output_tokens must be non-negative")
        output_tokens_per_request = estimated_output_tokens
    if max_requests_per_file <= 0:
        raise ValueError("max_requests_per_file must be positive")
    if max_bytes_per_file <= 0:
        raise ValueError("max_bytes_per_file must be positive")

    out_dir = run_dir / "nano_batch"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    manifest_f = manifest_path.open("w", encoding="utf-8")
    token_counter = _build_token_counter()
    prompt_factory = OpenAIJudgePromptFactory(
        token_counter=token_counter,
        system_override=system_prompt,
        user_override=user_template,
    )

    dataset_input_tokens: Dict[str, int] = {}
    total_input_tokens = 0
    total_output_tokens = 0
    total_requests = 0
    dataset_request_meta: Dict[str, List[Dict[str, Any]]] = {}
    writers: Dict[str, BatchRequestWriter] = {}

    summary: Dict[str, Any] = {
        "datasets": {},
        "out_dir": str(out_dir),
        "profile": {
            "name": DEFAULT_NANO_BATCH_PROFILE.name,
            "currency": DEFAULT_NANO_BATCH_PROFILE.currency,
            "input_cost_per_million_tokens": DEFAULT_NANO_BATCH_PROFILE.input_cost_per_million,
            "cached_input_cost_per_million_tokens": DEFAULT_NANO_BATCH_PROFILE.cached_input_cost_per_million,
            "output_cost_per_million_tokens": DEFAULT_NANO_BATCH_PROFILE.output_cost_per_million,
            "notes": DEFAULT_NANO_BATCH_PROFILE.notes,
        },
        "cached_input_ratio": cached_input_ratio,
        "assumed_output_tokens_per_request": output_tokens_per_request,
        "estimated_total_cost": 0.0,
        "total_input_tokens": 0,
        "estimated_total_output_tokens": 0,
        "total_requests": 0,
        "batch_limits": {
            "max_requests_per_file": max_requests_per_file,
            "max_bytes_per_file": max_bytes_per_file,
        },
    }

    if comparison_run_dir is not None:
        summary["comparison_run_dir"] = str(comparison_run_dir)

    logger = logging.getLogger(__name__)

    try:
        for ds in datasets:
            src_file = _discover_output_file(run_dir, ds)
            is_shp = _is_shp_dataset(ds)
            if is_shp and comparison_run_dir is not None:
                raise ValueError(
                    f"Comparison runs are not supported for SHP datasets (got {ds})"
                )
            template = prompt_factory.get_template(ds)
            writer = BatchRequestWriter(
                out_dir / f"{ds}_requests.jsonl",
                max_requests=max_requests_per_file,
                max_bytes=max_bytes_per_file,
            )
            writers[ds] = writer
            count = 0
            comp_iter: Optional[Iterator[Dict[str, Any]]] = None
            if comparison_run_dir is not None:
                comp_file = _discover_output_file(comparison_run_dir, ds)
                comp_iter = iter(_load_records(comp_file))
            for i, rec in enumerate(_load_records(src_file)):
                prompt = _prompt_text(rec)
                if not prompt:
                    continue

                if is_shp:
                    resp_a, resp_b = _extract_shp_responses(rec)
                    if not resp_a or not resp_b:
                        continue
                    pred = resp_a
                    ref = resp_b
                    ref_source = "shp_response_b"
                else:
                    pred = _first_generation(rec)
                    if not pred:
                        continue

                    ref_source = "reference"
                    if comp_iter is not None:
                        try:
                            comp_rec = next(comp_iter)
                        except StopIteration:
                            raise ValueError(
                                f"Comparison run {comparison_run_dir} has fewer records than {run_dir} for dataset '{ds}'"
                            ) from None
                        ref = _first_generation(comp_rec)
                        ref_source = "comparison_run_dir"
                        comp_prompt = _prompt_text(comp_rec)
                        if comp_prompt and prompt and comp_prompt.strip() != prompt.strip():
                            logger.warning(
                                "Prompt mismatch between runs for dataset %s index %d; using primary prompt", ds, i
                            )
                    else:
                        ref = _reference_text(rec)

                    if not ref:
                        if comp_iter is not None:
                            logger.warning(
                                "Skipping dataset %s index %d: comparison run is missing a generation", ds, i
                            )
                        continue

                rendered = template.render(
                    instruction=prompt,
                    output_1=pred,
                    output_2=ref,
                )
                body = {
                    "model": model,
                    "messages": rendered.messages,
                }
                custom_id = f"{ds}-{i}"
                line = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                if line["url"] != "/v1/chat/completions":
                    raise ValueError("Invalid request URL for chat completions batch")
                if "messages" not in body or not isinstance(body["messages"], list):
                    raise ValueError("Batch request body must include a messages list")
                current_path = writer.write_record(line)

                man = PreparedItem(
                    custom_id=custom_id,
                    dataset=ds,
                    index=i,
                    pred_len=len(pred.split()),
                    ref_len=len(ref.split()),
                    prompt_tokens=rendered.total_tokens,
                    request_file=_safe_relative_path(current_path, out_dir),
                    reference_source=ref_source,
                )
                manifest_f.write(json.dumps(man.__dict__) + "\n")

                dataset_input_tokens[ds] = dataset_input_tokens.get(ds, 0) + rendered.total_tokens
                total_input_tokens += rendered.total_tokens
                total_requests += 1
                count += 1

            writer.close()
            dataset_request_meta[ds] = [
                {
                    "path": _safe_relative_path(summary_entry.path, out_dir),
                    "request_count": summary_entry.request_count,
                    "byte_count": summary_entry.byte_count,
                }
                for summary_entry in writer.summaries
            ]

            input_tokens = dataset_input_tokens.get(ds, 0)
            est_output_tokens = count * output_tokens_per_request
            est_cost = DEFAULT_NANO_BATCH_PROFILE.estimate_cost(
                input_tokens=input_tokens,
                output_tokens=est_output_tokens,
                cached_ratio=cached_input_ratio,
            )
            file_meta_sorted = sorted(
                dataset_request_meta.get(ds, []), key=lambda item: item["path"]
            )
            summary["datasets"][ds] = {
                "requests": count,
                "request_files": [item["path"] for item in file_meta_sorted],
                "request_file_metadata": file_meta_sorted,
                "input_tokens": input_tokens,
                "estimated_output_tokens": est_output_tokens,
                "estimated_cost": round(est_cost, 4),
            }
            summary["estimated_total_cost"] += est_cost
            total_output_tokens += est_output_tokens

            if comp_iter is not None:
                try:
                    next(comp_iter)
                except StopIteration:
                    pass
                else:
                    raise ValueError(
                        f"Comparison run {comparison_run_dir} has more records than {run_dir} for dataset '{ds}'"
                    )
    finally:
        manifest_f.close()
        for writer in writers.values():
            writer.close()

    summary["total_requests"] = total_requests
    summary["total_input_tokens"] = total_input_tokens
    summary["estimated_total_output_tokens"] = total_output_tokens
    summary["estimated_total_cost"] = round(summary["estimated_total_cost"], 4)
    return summary


# ---------------------------- Results parsing ---------------------------


def _extract_text_from_response(resp: Dict[str, Any]) -> str:
    """Extract plain text from a Responses API response structure.

    Tries ``output_text`` first, then concatenates any ``text`` items found in
    ``output[*].content[*].text``. Also handles OpenAI Batch API format with
    ``body.choices[*].message.content``. Falls back to empty string.
    """
    if "output_text" in resp and isinstance(resp["output_text"], str):
        return resp["output_text"].strip()
    
    # Try OpenAI Batch API format: body.choices[].message.content
    try:
        body = resp.get("body", {})
        choices = body.get("choices", [])
        if choices and isinstance(choices, list):
            for choice in choices:
                message = choice.get("message", {})
                content = message.get("content")
                if content and isinstance(content, str):
                    return content.strip()
    except Exception:
        pass
    
    # Walk output array (original format)
    try:
        outputs = resp.get("output", [])
        texts: List[str] = []
        for msg in outputs:
            for part in msg.get("content", []):
                if part.get("type") in ("output_text", "text") and "text" in part:
                    texts.append(str(part["text"]))
        return "\n".join(t.strip() for t in texts if t.strip())
    except Exception:
        return ""


def _extract_python_literal(text: str) -> Any:
    """Extract a JSON object embedded in text.

    Tries JSON first; then falls back to ``ast.literal_eval`` to support
    Python dict/list outputs as required by the prompt.
    """
    import ast
    s = text.strip()
    # Remove code fences if present
    if s.startswith("```"):
        # Find the first newline after the fence language (if any)
        newline = s.find("\n")
        if newline != -1:
            s = s[newline + 1 :]
        # Trim trailing fence
        end = s.rfind("```")
        if end != -1:
            s = s[:end]
        s = s.strip()
    # Try JSON
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try a top-level {...} or [...] block
    m = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            try:
                return ast.literal_eval(m.group(1))
            except Exception:
                pass
    # Fallback: literal_eval whole string
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def _parse_single_letter_winner(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    if cleaned.upper() in ("A", "B"):
        return cleaned.upper()
    match = re.search(r"\b([AB])\b", cleaned, re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _is_shp_base_name(base_name: str) -> bool:
    base = base_name.lower()
    return base.startswith("shp") or base.startswith("askengineers") or base.startswith("askculinary")


def parse_results(run_dir: Path, base_name: str, results_file: Path, pairwise: bool = False) -> Path:
    """Parse Batch results NDJSON and write per-record scores.

    Parameters
    ----------
    run_dir : Path
        Root directory for outputs
    base_name : str
        Base name for output files (e.g., 'cnndm_test_A3__SFT_src_KTO_src_Eval_tgt')
    results_file : Path
        Path to the batch results JSONL file
    pairwise : bool
        Whether this is pairwise evaluation mode

    Returns
    -------
    Path
        Path to the parsed scores JSONL (``*_scores.jsonl``).
    """
    out_dir = run_dir / "nano_batch"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / f"{base_name}_scores.jsonl"
    is_shp = _is_shp_base_name(base_name)
    
    # Track parsing statistics
    total_responses = 0
    written = 0
    json_parse_failures = 0
    no_text_extracted = 0
    no_python_literal = 0
    unrecognized_winner = 0
    invalid_structure = 0
    
    with scores_path.open("w", encoding="utf-8") as sf:
        with results_file.open("r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                total_responses += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    json_parse_failures += 1
                    logging.debug(f"Failed to parse line as JSON: {line[:100]}")
                    continue
                custom_id = obj.get("custom_id") or obj.get("id") or ""
                resp = obj.get("response") or obj.get("result") or obj
                text = _extract_text_from_response(resp)
                if not text:
                    no_text_extracted += 1
                    logging.debug(f"No text extracted from response for custom_id={custom_id}")
                    continue
                if is_shp:
                    winner_letter = _parse_single_letter_winner(text)
                    if not winner_letter:
                        unrecognized_winner += 1
                        logging.warning(
                            f"Unrecognized winner response for custom_id={custom_id}. "
                            "Expected single-letter A or B."
                        )
                        continue
                    winner = "response_a" if winner_letter == "A" else "response_b"
                    row = {"custom_id": custom_id, "winner": winner}
                else:
                    lit = _extract_python_literal(text)
                    if lit is None:
                        no_python_literal += 1
                        logging.debug(f"Could not extract Python literal for custom_id={custom_id}")
                        continue

                if not is_shp and pairwise:
                    # Pairwise mode: parse winner field (A, B, or Tie with synonyms)
                    if isinstance(lit, dict):
                        winner_raw = lit.get("winner", "")
                        if isinstance(winner_raw, str):
                            # Normalize: uppercase, strip whitespace, replace spaces/hyphens with underscores
                            winner_normalized = winner_raw.strip().upper().replace(" ", "_").replace("-", "_")
                            # Accept A/B with synonyms
                            if winner_normalized in ("A", "FIRST", "1", "MODEL_1", "MODELA", "SUMMARY_A"):
                                winner = "model_1"
                            elif winner_normalized in ("B", "SECOND", "2", "MODEL_2", "MODELB", "SUMMARY_B"):
                                winner = "model_2"
                            # Accept tie with multiple synonyms
                            elif winner_normalized in ("TIE", "DRAW", "EQUAL", "SAME", "NEITHER", "BOTH"):
                                winner = "tie"
                            else:
                                unrecognized_winner += 1
                                logging.warning(
                                    f"Unrecognized winner value '{winner_raw}' for custom_id={custom_id}. "
                                    f"Expected A/B/Tie or synonyms. Skipping this response."
                                )
                                continue
                        else:
                            invalid_structure += 1
                            logging.warning(f"Winner field is not a string for custom_id={custom_id}")
                            continue
                    else:
                        invalid_structure += 1
                        logging.warning(f"Pairwise response not a dict for custom_id={custom_id}")
                        continue
                    row = {"custom_id": custom_id, "winner": winner}
                elif not is_shp:
                    # Standard mode: Accept either list of {model, rank} or dict with a key pointing to that list
                    rankings = None
                    if isinstance(lit, list):
                        rankings = lit
                    elif isinstance(lit, dict):
                        # Common keys: 'ranking', 'ranks', 'result'
                        for k in ("ranking", "ranks", "result", "output"):
                            v = lit.get(k)
                            if isinstance(v, list):
                                rankings = v
                                break
                        if rankings is None:
                            # Some models might directly return a dict mapping model->rank
                            # Convert to list
                            if all(isinstance(v, (int, float)) for v in lit.values()):
                                rankings = [{"model": k, "rank": v} for k, v in lit.items()]
                    if not rankings or not isinstance(rankings, list):
                        invalid_structure += 1
                        logging.debug(f"Invalid ranking structure for custom_id={custom_id}")
                        continue
                    # Compute winner
                    m1_rank = None
                    m2_rank = None
                    for item in rankings:
                        try:
                            name = str(item.get("model"))
                            rank = float(item.get("rank"))
                        except Exception:
                            continue
                        if name == "model_1":
                            m1_rank = rank
                        elif name == "model_2":
                            m2_rank = rank
                    if m1_rank is None or m2_rank is None:
                        invalid_structure += 1
                        logging.debug(f"Missing model ranks for custom_id={custom_id}")
                        continue
                    if m1_rank < m2_rank:
                        winner = "model_1"
                    elif m2_rank < m1_rank:
                        winner = "model_2"
                    else:
                        winner = "tie"
                    row = {"custom_id": custom_id, "winner": winner, "ranks": {"model_1": m1_rank, "model_2": m2_rank}}
                
                sf.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    
    # Log parsing statistics
    total_skipped = total_responses - written
    if total_responses > 0:
        success_rate = (written / total_responses) * 100
        skip_rate = (total_skipped / total_responses) * 100
        
        logging.info(
            f"Parse results for {base_name}: "
            f"{written}/{total_responses} responses successfully parsed ({success_rate:.1f}%)"
        )
        
        if total_skipped > 0:
            logging.info(
                f"Skipped {total_skipped} responses ({skip_rate:.1f}%): "
                f"json_parse={json_parse_failures}, "
                f"no_text={no_text_extracted}, "
                f"no_literal={no_python_literal}, "
                f"unrecognized_winner={unrecognized_winner}, "
                f"invalid_structure={invalid_structure}"
            )
        
        if skip_rate > 10.0:
            logging.warning(
                f"High skip rate ({skip_rate:.1f}%) detected. "
                f"This may indicate issues with the judge model or prompt format."
            )
    
    if written == 0:
        raise RuntimeError(f"No parsable results found in {results_file}")
    return scores_path


# ---------------------------- Metrics aggregation -----------------------


def _bootstrap_ci(values: List[float], *, seed: int = 0, n_samples: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    import random
    import statistics

    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(n_samples):
        sample = [rng.choice(values) for _ in range(len(values))]
        means.append(statistics.mean(sample))
    means.sort()
    lo = int(alpha / 2 * n_samples)
    hi = int((1 - alpha / 2) * n_samples)
    return (means[lo], means[hi])


def score_parsed(run_dir: Path, base_name: str, *, seed: int = 0, pairwise: bool = False, manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    """Aggregate win-rate metrics from parsed scores file.

    Parameters
    ----------
    run_dir : Path
        Root directory with nano_batch subdirectory
    base_name : str
        Base name for files (e.g., 'cnndm_test_A3__SFT_src_KTO_src_Eval_tgt')
    seed : int
        Random seed for bootstrap confidence intervals
    pairwise : bool
        Whether this is pairwise evaluation mode
    manifest_path : Optional[Path]
        Path to manifest file containing is_swapped information

    Returns
    -------
    Dict[str, Any]
        Aggregated metrics including win rate and confidence intervals

    Assumes ``{base_name}_scores.jsonl`` exists under ``run_dir/nano_batch``.
    """
    import statistics

    out_dir = run_dir / "nano_batch"
    scores_path = out_dir / f"{base_name}_scores.jsonl"
    is_shp = _is_shp_base_name(base_name)
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing parsed scores file: {scores_path}; run parse first")

    # Load swap information from manifest if available
    swap_info: Dict[str, bool] = {}
    if pairwise and manifest_path and manifest_path.exists():
        # Resolve manifest path: if directory, look for per-file manifest
        actual_manifest: Optional[Path] = None
        if manifest_path.is_dir():
            # Try to find per-file manifest for this base_name
            per_file_manifest = manifest_path / f"{base_name}_manifest.jsonl"
            if per_file_manifest.exists():
                actual_manifest = per_file_manifest
        else:
            # Direct file path
            actual_manifest = manifest_path
        
        if actual_manifest and actual_manifest.exists():
            manifest_entry_count = 0
            with actual_manifest.open("r", encoding="utf-8") as mf:
                for line in mf:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        custom_id = record.get("custom_id")
                        is_swapped = record.get("is_swapped", False)
                        if custom_id:
                            swap_info[custom_id] = is_swapped
                            manifest_entry_count += 1
                    except json.JSONDecodeError:
                        continue
            
            if manifest_entry_count > 0:
                logging.info(f"Loaded manifest for {base_name}: {manifest_entry_count} entries from {actual_manifest.name}")
    
    # Validate swap information completeness in pairwise mode
    if pairwise and swap_info:
        # Check if all custom_ids in scores have swap info
        missing_swap_ids = []
        with scores_path.open("r", encoding="utf-8") as sf:
            for line in sf:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    custom_id = row.get("custom_id", "")
                    if custom_id and custom_id not in swap_info:
                        missing_swap_ids.append(custom_id)
                except json.JSONDecodeError:
                    continue
        
        if missing_swap_ids:
            logging.warning(
                f"Pairwise mode: {len(missing_swap_ids)} custom_ids from scores missing in manifest. "
                f"These will default to is_swapped=False, which may cause incorrect win rates. "
                f"Missing IDs (first 10): {missing_swap_ids[:10]}"
            )

    # We compute simple win/tie/loss from parsed rankings
    wins: List[float] = []  # 1 for adapted model win, 0 for loss, 0.5 for tie
    n_win = n_loss = n_tie = 0
    
    # For pairwise mode, track wins when prediction was first vs second
    wins_pred_first: List[float] = []
    wins_pred_second: List[float] = []
    
    with scores_path.open("r", encoding="utf-8") as sf:
        for line in sf:
            if not line.strip():
                continue
            row = json.loads(line)
            w = row.get("winner")
            if is_shp and w in ("response_a", "response_b"):
                w = "model_1" if w == "response_a" else "model_2"
            
            # In pairwise mode, adjust for swaps to compute win relative to adapted model
            custom_id = row.get("custom_id", "")
            is_swapped = swap_info.get(custom_id, False) if pairwise else False
            
            if w == "model_1":
                # model_1 won; if swapped, that means base won, else adapted won
                win_val = 0.0 if is_swapped else 1.0
                wins.append(win_val)
                if not is_swapped:
                    n_win += 1
                else:
                    n_loss += 1
            elif w == "model_2":
                # model_2 won; if swapped, that means adapted won, else base won
                win_val = 1.0 if is_swapped else 0.0
                wins.append(win_val)
                if is_swapped:
                    n_win += 1
                else:
                    n_loss += 1
            else:
                wins.append(0.5)
                n_tie += 1
            
            # Track position bias if in pairwise mode
            if pairwise:
                if is_swapped:
                    # When swapped, adapted prediction is in position 2 (model_2)
                    wins_pred_second.append(wins[-1])
                else:
                    # Adapted prediction is in position 1 (model_1)
                    wins_pred_first.append(wins[-1])

    win_rate = statistics.mean(wins) if wins else 0.0
    result: Dict[str, Any] = {
        "base_name": base_name,
        "n": len(wins),
        "wins": n_win,
        "losses": n_loss,
        "ties": n_tie,
        "win_rate": win_rate,
        "win_rate_ci": _bootstrap_ci(wins, seed=seed) if wins else (0.0, 0.0),
    }
    
    # Add pairwise-specific metrics
    if pairwise and (wins_pred_first or wins_pred_second):
        win_rate_pred_first = statistics.mean(wins_pred_first) if wins_pred_first else 0.0
        win_rate_pred_second = statistics.mean(wins_pred_second) if wins_pred_second else 0.0
        side_swap_bias = abs(win_rate_pred_first - win_rate_pred_second)
        
        result["pairwise"] = True
        result["win_rate_pred_first"] = win_rate_pred_first
        result["win_rate_pred_second"] = win_rate_pred_second
        result["side_swap_bias"] = side_swap_bias
        result["n_pred_first"] = len(wins_pred_first)
        result["n_pred_second"] = len(wins_pred_second)

    out_path = out_dir / f"{base_name}_metrics.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def compute_final_scores_from_results(run_dir: Path, base_name: str, results_file: Path, *, seed: int = 0, pairwise: bool = False, manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    """Convenience method: parse a results NDJSON and return aggregate scores.

    Parameters
    ----------
    run_dir : Path
        Root directory for outputs
    base_name : str
        Base name for output files (e.g., 'cnndm_test_A3__SFT_src_KTO_src_Eval_tgt')
    results_file : Path
        Path to the batch results JSONL file
    seed : int
        Random seed for bootstrap confidence intervals
    pairwise : bool
        Whether this is pairwise evaluation mode
    manifest_path : Optional[Path]
        Path to manifest file for swap information

    Returns
    -------
    Dict[str, Any]
        Aggregated win-rate metrics

    This reads the Batch results (NDJSON) for the given base_name, writes a
    parsed rankings file, and returns aggregated win-rate metrics.
    """
    parse_results(run_dir, base_name, results_file, pairwise=pairwise)
    return score_parsed(run_dir, base_name, seed=seed, pairwise=pairwise, manifest_path=manifest_path)


# ------------------------------- CLI entry ------------------------------


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run_dir", type=Path, required=True, help="Run directory with outputs")


def main(argv: List[str] | None = None) -> None:  # pragma: no cover - CLI wrapper
    ap = argparse.ArgumentParser(description="GPT-5-nano Batch judge harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Prepare NDJSON request files from run_dir outputs")
    _add_common_args(p_prep)
    p_prep.add_argument("--datasets", nargs="+", default=["tldr", "cnndm"], help="Datasets to prepare")
    p_prep.add_argument("--model", default="gpt-5-nano", help="OpenAI model name to use in requests")
    p_prep.add_argument(
        "--comparison-run-dir",
        type=Path,
        help="Optional secondary run directory providing candidate B generations",
    )
    p_prep.add_argument(
        "--system-prompt-file",
        type=Path,
        help="Optional path to a custom system prompt. Overrides dataset defaults.",
    )
    p_prep.add_argument(
        "--user-template-file",
        type=Path,
        help=(
            "Optional path to a user message template containing {instruction}, {output_1}, and "
            "{output_2} placeholders (or {prompt}, {response}, {response_a}, {response_b}). "
            "Overrides dataset defaults."
        ),
    )
    p_prep.add_argument(
        "--cached-input-ratio",
        type=float,
        default=0.0,
        help="Fraction of input tokens expected to come from cache (0–1).",
    )
    p_prep.add_argument(
        "--estimated-output-tokens",
        type=int,
        default=DEFAULT_NANO_BATCH_PROFILE.default_output_tokens,
        help="Assumed judge output tokens per request when estimating cost.",
    )
    p_prep.add_argument(
        "--max-requests-per-file",
        type=int,
        default=MAX_REQUESTS_PER_FILE,
        help="Maximum requests per NDJSON file before rotating (default: 50000)",
    )
    p_prep.add_argument(
        "--max-bytes-per-file",
        type=int,
        default=MAX_BYTES_PER_FILE,
        help="Maximum bytes per NDJSON file before rotating (default: 200 MiB)",
    )

    p_parse = sub.add_parser("parse", help="Parse Batch results NDJSON to per-record scores")
    _add_common_args(p_parse)
    p_parse.add_argument("--dataset", help="(Deprecated) Dataset name. If not provided, base name is derived from results-file.")
    p_parse.add_argument("--results-file", type=Path, required=True, help="Path to downloaded Batch results JSONL")
    p_parse.add_argument("--pairwise", action="store_true", help="Enable pairwise evaluation mode parsing")

    p_score = sub.add_parser("score", help="Aggregate win-rate metrics from parsed results")
    _add_common_args(p_score)
    p_score.add_argument("--base-name", help="Base name for files (e.g., 'cnndm_test_A3__SFT_src_KTO_src_Eval_tgt')")
    p_score.add_argument("--dataset", help="(Deprecated) Dataset name. Use --base-name instead.")
    p_score.add_argument("--seed", type=int, default=0)
    p_score.add_argument("--pairwise", action="store_true", help="Enable pairwise evaluation mode scoring")
    p_score.add_argument("--manifest", type=Path, help="Path to manifest file for swap information")

    args = ap.parse_args(argv)

    if args.cmd == "prepare":
        system_prompt_override = (
            args.system_prompt_file.read_text(encoding="utf-8") if args.system_prompt_file else None
        )
        user_template_override = (
            args.user_template_file.read_text(encoding="utf-8") if args.user_template_file else None
        )
        try:
            summary = prepare_requests(
                args.run_dir,
                args.datasets,
                model=args.model,
                system_prompt=system_prompt_override,
                user_template=user_template_override,
                cached_input_ratio=args.cached_input_ratio,
                estimated_output_tokens=args.estimated_output_tokens,
                max_requests_per_file=args.max_requests_per_file,
                max_bytes_per_file=args.max_bytes_per_file,
                comparison_run_dir=args.comparison_run_dir,
            )
        except ValueError as exc:
            ap.error(str(exc))
        print(json.dumps(summary, indent=2))
    elif args.cmd == "parse":
        # Determine base_name: use dataset if provided (for backwards compatibility),
        # otherwise derive from results_file
        if args.dataset:
            base_name = args.dataset
        else:
            # Extract base name from results file
            stem = args.results_file.stem
            # Remove common suffixes
            for suffix in ['_batch_output', '_results', '_batch_errors']:
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
                    break
            base_name = stem
        out = parse_results(args.run_dir, base_name, args.results_file, pairwise=args.pairwise)
        print(str(out))
    elif args.cmd == "score":
        # Determine base_name
        if args.base_name:
            base_name = args.base_name
        elif args.dataset:
            base_name = args.dataset
        else:
            ap.error("Either --base-name or --dataset must be provided for score command")
        res = score_parsed(
            args.run_dir, 
            base_name, 
            seed=args.seed, 
            pairwise=args.pairwise,
            manifest_path=args.manifest
        )
        print(json.dumps(res, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
