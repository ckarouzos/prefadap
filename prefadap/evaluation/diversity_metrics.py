r"""Diversity evaluation for language model generations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import statistics
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from prefadap.utils.seeding import set_seed

# Log module import path for debugging (helps detect old installed code paths)
logger = logging.getLogger(__name__)
logger.info(f"prefadap diversity_metrics loaded from: {__file__}")

similarity2diversity_function = lambda sim_score_list: 1 - np.mean(sim_score_list) if sim_score_list else 0.0

# 384 is the output dimension for sentence-transformers like 'all-MiniLM-L6-v2'
DEFAULT_SENTENCE_EMBEDDING_DIMENSION = 384
# 500 is the default sample size used in the Meta protocol for diversity evaluation
DEFAULT_OVERALL_SAMPLE_SIZE = 500
_STATISTICS_HAS_FMEAN = hasattr(statistics, "fmean")


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean using ``statistics.fmean`` when available."""

    if _STATISTICS_HAS_FMEAN:  # pragma: no branch - constant after import
        return float(statistics.fmean(values))
    return float(statistics.mean(values))


def _maybe_tqdm(iterable: Iterable[Any], enabled: bool, desc: str) -> Iterable[Any]:
    """Wrap an iterable with ``tqdm`` progress reporting when requested."""

    if not enabled:
        return iterable

    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:  # pragma: no cover - tqdm is an optional dep
        return iterable

    return tqdm(iterable, desc=desc, leave=False)


@dataclass(slots=True)
class OutputGroup:
    """Container for the generations associated with a single prompt."""

    input_id: str
    outputs: List[str]
    prompt: Optional[str] = None


@dataclass(slots=True)
class DiversityConfig:
    """Configuration for the diversity evaluation pipeline."""

    model_name: Optional[str] = None
    max_inputs: int = 500
    generations_per_input: int = 16
    ngram_range: Tuple[int, ...] = (1, 2, 3, 4, 5)
    sampling_temperature: float = 1.0
    random_seed: int = 1234
    sbert_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    sbert_batch_size: int = 1024
    sbert_cache_dir: Optional[Path] = None
    nli_model_name: str = "roberta-large-mnli"
    nli_batch_size: int = 1024
    nli_pairs_per_input: int = -1
    nli_pairs_across: int = -1
    nli_random_subsample: Optional[float] = None
    nli_sentence_sample_size: int = 1
    tokenizer_max_length: int = 512
    require_exact_counts: bool = True
    verbose: bool = False
    deterministic: bool = False
    output_precision: int = 6
    sample_overall: bool = True


def _ensure_directory(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Generation directory '{path}' does not exist")
    if not path.is_dir():
        raise ValueError(f"Generation path '{path}' must be a directory")


def _iter_json_objects(path: Path) -> Iterator[Any]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            yield json.load(handle)


def _extract_outputs(record: Any) -> Tuple[Optional[str], List[str]]:
    if isinstance(record, list):
        return None, [str(item) for item in record]

    if isinstance(record, dict):
        if "outputs" in record and isinstance(record["outputs"], Sequence):
            outputs = record["outputs"]
        elif "generations" in record and isinstance(record["generations"], Sequence):
            outputs = record["generations"]
        elif "responses" in record and isinstance(record["responses"], Sequence):
            outputs = record["responses"]
        elif "completions" in record and isinstance(record["completions"], Sequence):
            outputs = record["completions"]
        elif "samples" in record and isinstance(record["samples"], Sequence):
            outputs = record["samples"]
        else:
            raise ValueError("Record does not contain an outputs field")

        if isinstance(outputs, (str, bytes)):
            raise ValueError("Outputs field must contain a sequence of strings")

        prompt = (
            record.get("prompt")
            or record.get("input")
            or record.get("instruction")
            or record.get("question")
        )

        return prompt, [str(item) for item in outputs]

    raise ValueError("Unsupported record type for generation outputs")


def _iter_entries_from_object(obj: Any) -> Iterator[Tuple[str, Optional[str], List[str]]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            prompt, outputs = _extract_outputs(value)
            yield str(key), prompt, outputs
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            key = str(value.get("id", index)) if isinstance(value, dict) else str(index)
            prompt, outputs = _extract_outputs(value)
            yield key, prompt, outputs
    else:
        raise ValueError("Top level JSON object must be a dict or list")


def load_model_generations(model_outputs_dir: Path | str, config: DiversityConfig) -> List[OutputGroup]:
    base_path = Path(model_outputs_dir)
    _ensure_directory(base_path)

    groups: List[OutputGroup] = []
    json_files = sorted(
        [
            path
            for path in base_path.rglob("*")
            if path.suffix in {".json", ".jsonl"}
        ]
    )

    if not json_files:
        raise FileNotFoundError(
            f"No JSON or JSONL files containing generations were found in '{base_path}'"
        )

    for file_path in json_files:
        for obj in _iter_json_objects(file_path):
            # Direct support for new JSONL schema: one record per prompt with all outputs
            # Check if obj is already a single properly-formatted record
            if (
                isinstance(obj, dict)
                and "prompt" in obj
                and ("outputs" in obj or "generations" in obj)
            ):
                # Extract fields from the single record
                outputs_field = obj.get("outputs") or obj.get("generations")
                if not isinstance(outputs_field, list):
                    raise ValueError("Field 'outputs' or 'generations' must be a list of strings")
                
                key = str(obj.get("key") or obj.get("id") or "")
                prompt = obj.get("prompt")
                outputs = [str(out) for out in outputs_field]
                
                # Validate output count
                output_count = len(outputs)
                if output_count < config.generations_per_input:
                    message = (
                        f"Entry '{key}' only has {output_count} generations; expected at least"
                        f" {config.generations_per_input}."
                    )
                    if config.require_exact_counts:
                        raise ValueError(message)
                    warnings.warn(message + " Skipping entry.", RuntimeWarning, stacklevel=2)
                    continue

                if (
                    not config.require_exact_counts
                    and output_count > config.generations_per_input
                ):
                    warnings.warn(
                        (
                            f"Entry '{key}' has {output_count} generations; truncating to"
                            f" {config.generations_per_input}."
                        ),
                        RuntimeWarning,
                        stacklevel=2,
                    )

                truncated = outputs[: config.generations_per_input]
                groups.append(OutputGroup(input_id=key, outputs=truncated, prompt=prompt))

                if len(groups) >= config.max_inputs:
                    return groups
            else:
                # Fallback: handle legacy formats (dict-of-records or list-of-records)
                for key, prompt, outputs in _iter_entries_from_object(obj):
                    output_count = len(outputs)
                    if output_count < config.generations_per_input:
                        message = (
                            f"Entry '{key}' only has {output_count} generations; expected at least"
                            f" {config.generations_per_input}."
                        )
                        if config.require_exact_counts:
                            raise ValueError(message)
                        warnings.warn(message + " Skipping entry.", RuntimeWarning, stacklevel=2)
                        continue

                    if (
                        not config.require_exact_counts
                        and output_count > config.generations_per_input
                    ):
                        warnings.warn(
                            (
                                f"Entry '{key}' has {output_count} generations; truncating to"
                                f" {config.generations_per_input}."
                            ),
                            RuntimeWarning,
                            stacklevel=2,
                        )

                    truncated = [str(out) for out in outputs[: config.generations_per_input]]
                    groups.append(OutputGroup(input_id=key, outputs=truncated, prompt=prompt))

                    if len(groups) >= config.max_inputs:
                        return groups

    if config.require_exact_counts and len(groups) < config.max_inputs:
        raise ValueError(
            f"Only {len(groups)} prompts were found but 'max_inputs' was set to {config.max_inputs}."
            " Set 'require_exact_counts=False' to allow smaller samples."
        )

    return groups


_PUNCTUATION_TO_REMOVE = [".", "\n"]

def lines_to_ngrams(lines: Sequence[str], n: int = 3) -> List[List[Tuple[str, ...]]]:
    ngrams: List[List[Tuple[str, ...]]] = []
    for sentence in lines:
        cleaned = sentence
        for punct in _PUNCTUATION_TO_REMOVE:
            cleaned = cleaned.replace(punct, "")
        words = cleaned.split()
        if len(words) < n or n <= 0:
            ngrams.append([])
            continue
        ngrams.append([tuple(words[index : index + n]) for index in range(len(words) - n + 1)])
    return ngrams


def first_outputs(groups: Sequence[OutputGroup]) -> List[str]:
    return [group.outputs[0] for group in groups if group.outputs]


class EADDiversityCalculator:
    """Compute expectation-adjusted distinct n-gram diversity."""

    def __init__(self, ngram_range: Sequence[int], vocab_size: int = 50257):
        if not ngram_range:
            raise ValueError("ngram_range must contain at least one n")
        self.ngram_range = tuple(sorted(ngram_range))
        self.vocab_size = int(vocab_size)

    def _ead_for_lines(self, lines: Sequence[str], n: int) -> float:
        grouped = lines_to_ngrams(lines, n)
        ngrams: List[Tuple[str, ...]] = [ngram for group in grouped for ngram in group]
        total = len(ngrams)
        if total == 0:
            return 0.0

        unique = len(set(ngrams))
        vocab = float(self.vocab_size)
        denom = vocab * (1.0 - pow((vocab - 1.0) / vocab, total))
        if denom <= 0.0:
            return 0.0

        ead = unique / denom
        return float(min(max(ead, 0.0), 1.0))

    def per_input(self, outputs_per_prompt: Sequence[Sequence[str]]) -> List[float]:
        scores: List[float] = []
        for responses in outputs_per_prompt:
            values = [self._ead_for_lines(responses, n) for n in self.ngram_range]
            scores.append(sum(values) / len(values))
        return scores

    def across_input(self, first_outputs: Sequence[str]) -> float:
        values: List[float] = []
        for n in self.ngram_range:
            grouped = lines_to_ngrams(first_outputs, n)
            ngrams: List[Tuple[str, ...]] = [ngram for group in grouped for ngram in group]

            total = len(ngrams)
            if total == 0:
                values.append(0.0)
                continue

            unique = len(set(ngrams))
            vocab = float(self.vocab_size)
            denom = vocab * (1.0 - pow((vocab - 1.0) / vocab, total))
            if denom <= 0.0:
                values.append(0.0)
                continue

            ead = unique / denom
            values.append(float(min(max(ead, 0.0), 1.0)))

        return sum(values) / len(values) if values else 0.0


class SentenceBERTDiversityCalculator:
    """Semantic diversity using Sentence-BERT embeddings."""

    def __init__(
        self,
        model_name: str,
        batch_size: int,
        max_length: int,
        *,
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        import logging
        import os

        self.model_name = model_name
        self.batch_size = batch_size
        self.batched = True
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        # Lazy model loading: model is loaded on first use
        self._model: Optional[Any] = None

        # Set cache directory with environment variable fallback
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            # Use SBERT_CACHE_DIR env var or cluster-aware default
            default_cache = self._get_default_sbert_cache_dir()
            self.cache_dir = Path(os.environ.get("SBERT_CACHE_DIR", default_cache))
        
        logger = logging.getLogger(__name__)
        logger.info(f"SBERT cache directory: {self.cache_dir}")
        
        self._cache_path: Optional[Path] = None
        self._cache_dirty = False
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", self.model_name)
        self._cache_path = self.cache_dir / f"{safe_name}_L{self.max_length}.pt"
        self._load_cache()

    def _ensure_model_loaded(self) -> Any:
        """Lazy load the SentenceTransformer model on first use.
        
        Returns
        -------
        SentenceTransformer
            The loaded model instance.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.eval()
        
        return self._model
    
    @property
    def model(self) -> Any:
        """Get the SentenceTransformer model, loading it if necessary."""
        return self._ensure_model_loaded()
    
    @staticmethod
    def _get_default_sbert_cache_dir() -> str:
        """Get default SBERT cache directory.
        
        Preference order (if available):
        - PREFADAP_CACHE_ROOT or PERSIST_ROOT
        - SCRATCHDIR
        - XDG_CACHE_HOME
        - ~/.cache
        
        Returns
        -------
        str
            Path to default SBERT cache directory
        """
        cache_root = os.environ.get("PREFADAP_CACHE_ROOT") or os.environ.get("PERSIST_ROOT")
        if cache_root:
            return os.path.join(cache_root, "prefadap", "cache", "sbert")

        scratchdir = os.environ.get("SCRATCHDIR")
        if scratchdir:
            return os.path.join(scratchdir, "prefadap", "cache", "sbert")

        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            return os.path.join(xdg_cache, "prefadap", "sbert")

        return str(Path.home() / ".cache" / "prefadap" / "sbert")

    def _load_cache(self) -> None:
        if self._cache_path is None or not self._cache_path.exists():
            return

        try:
            payload = torch.load(self._cache_path, map_location="cpu")
        except (OSError, RuntimeError):  # pragma: no cover - corrupted cache
            return

        if not isinstance(payload, dict):
            return

        if payload.get("model_name") != self.model_name:
            return
        if int(payload.get("max_length", self.max_length)) != self.max_length:
            return

        embeddings = payload.get("embeddings", {})
        if not isinstance(embeddings, dict):
            return

        cache: Dict[str, torch.Tensor] = {}
        for key, value in embeddings.items():
            if isinstance(value, torch.Tensor):
                cache[str(key)] = value.detach().cpu()
        self._embedding_cache = cache
        self._cache_dirty = False

    def _save_cache(self) -> None:
        if not self._cache_dirty or self._cache_path is None:
            return

        payload = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "embeddings": {key: tensor.cpu() for key, tensor in self._embedding_cache.items()},
        }
        torch.save(payload, self._cache_path)
        self._cache_dirty = False

    @staticmethod
    def _cache_key(model_name: str, max_length: int, text: str) -> str:
        payload = f"{model_name}|{max_length}|{text}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _encode(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            hidden_size = getattr(
                self.model,
                "get_sentence_embedding_dimension",
                lambda: DEFAULT_SENTENCE_EMBEDDING_DIMENSION,
            )()
            return torch.empty(0, hidden_size)

        cached_embeddings: List[Optional[torch.Tensor]] = [None] * len(texts)
        missing_keys: List[str] = []
        missing_texts: List[str] = []
        key_to_indices: Dict[str, List[int]] = {}

        for index, text in enumerate(texts):
            key = self._cache_key(self.model_name, self.max_length, text)
            cached = self._embedding_cache.get(key)
            if cached is not None:
                cached_embeddings[index] = cached.clone()
                continue

            if key not in key_to_indices:
                key_to_indices[key] = []
                missing_keys.append(key)
                missing_texts.append(text)
            key_to_indices[key].append(index)

        if missing_texts:
            new_embeddings: List[torch.Tensor] = []
            batch_iterable = _maybe_tqdm(
                range(0, len(missing_texts), self.batch_size),
                self.verbose,
                "SBERT encoding",
            )
            for start in batch_iterable:
                batch = missing_texts[start : start + self.batch_size]
                with torch.inference_mode():
                    encoded = self.model.encode(
                        batch,
                        batch_size=self.batch_size,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                    )
                new_embeddings.append(encoded.detach().cpu())

            if new_embeddings:
                stacked_new = F.normalize(torch.cat(new_embeddings, dim=0), p=2, dim=1)
            else:
                stacked_new = torch.empty(0)

            for offset, key in enumerate(missing_keys):
                embedding = stacked_new[offset]
                for target_index in key_to_indices[key]:
                    cached_embeddings[target_index] = embedding.clone()
                self._embedding_cache[key] = embedding.clone()
                self._cache_dirty = True

        self._save_cache()

        final_embeddings: List[torch.Tensor] = []
        for embedding in cached_embeddings:
            if embedding is None:  # pragma: no cover - sanity guard
                raise RuntimeError("Sentence embeddings missing after caching step")
            final_embeddings.append(embedding)

        return torch.stack(final_embeddings, dim=0)

    @staticmethod
    def _pairwise_diversity(embeddings: torch.Tensor) -> float:
        k = embeddings.size(0)
        if k <= 1:
            return 0.0

        similarity_matrix = embeddings @ embeddings.T
        similarities: List[float] = []
        for i in range(k):
            for j in range(i + 1, k):
                similarities.append(float(similarity_matrix[i, j].item()))
        return float(similarity2diversity_function(similarities))

    def embed_groups(
        self, groups: Sequence[OutputGroup]
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]:
        texts: List[str] = []
        offsets: List[Tuple[int, int]] = []
        first_indices: List[int] = []
        cursor = 0

        for group in groups:
            texts.extend(group.outputs)
            start = cursor
            cursor += len(group.outputs)
            offsets.append((start, cursor))
            first_indices.append(start)

        embeddings = self._encode(texts)
        return embeddings, offsets, first_indices

    def per_input_from_embeddings(
        self, embeddings: torch.Tensor, offsets: Sequence[Tuple[int, int]]
    ) -> List[float]:
        scores: List[float] = []
        for start, end in offsets:
            segment = embeddings[start:end]
            scores.append(self._pairwise_diversity(segment))
        return scores

    def across_input_from_embeddings(
        self, embeddings: torch.Tensor, indices: Sequence[int]
    ) -> float:
        if not indices:
            return 0.0
        subset = embeddings[torch.tensor(indices, dtype=torch.long)]
        return self._pairwise_diversity(subset)


class NLIDiversityEvaluator:
    """Logical diversity measured with an NLI classifier."""

    LABEL_WEIGHTS: Dict[str, int] = {
        "CONTRADICTION": -1,
        "NEUTRAL": 0,
        "ENTAILMENT": 1,
    }

    def __init__(
        self,
        model_name: str,
        batch_size: int,
        max_length: int,
        pairs_per_input: int,
        pairs_across: int,
        sentence_sample_size: int,
        *,
        random_subsample: Optional[float] = None,
        random_seed: Optional[int] = None,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.batched = True
        self.max_length = max_length
        self.pairs_per_input = pairs_per_input
        self.pairs_across = pairs_across
        if random_subsample is not None and random_subsample <= 0.0:
            raise ValueError("nli_random_subsample must be > 0 when provided")
        if random_subsample is not None and random_subsample > 1.0:
            raise ValueError("nli_random_subsample must be <= 1.0")
        self.random_subsample = random_subsample
        self._rng = random.Random(random_seed) if random_seed is not None else random.Random()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.sentence_sample_size = max(1, int(sentence_sample_size))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Only move model to device if it doesn't have a device map
        # Models distributed across GPUs should not be moved
        if not hasattr(self.model, "hf_device_map"):
            self.model.to(self.device)
        
        self.model.eval()

        id_to_label: Dict[int, str] = {}
        for key, value in self.model.config.id2label.items():
            index = int(key) if not isinstance(key, int) else key
            id_to_label[index] = self._canonical_label(value)
        self.id_to_label = id_to_label
        self.label_to_weight = dict(self.LABEL_WEIGHTS)

    @staticmethod
    def _canonical_label(label: str) -> str:
        label = label.upper()
        if "CONTRADICT" in label:
            return "CONTRADICTION"
        if "ENTAIL" in label:
            return "ENTAILMENT"
        return "NEUTRAL"

    def _select_pairs(self, count: int, limit: int) -> List[Tuple[int, int]]:
        if count <= 1:
            return []

        indices = list(combinations(range(count), 2))

        if self.random_subsample is not None and self.random_subsample < 1.0:
            target = max(1, int(len(indices) * self.random_subsample))
            if target < len(indices):
                indices = self._rng.sample(indices, target)
                indices.sort()

        if limit is not None and limit > 0 and len(indices) > limit:
            indices = indices[:limit]

        return indices

    @staticmethod
    def _collate_pairs(batch: Sequence[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
        if not batch:
            return [], []
        premises, hypotheses = zip(*batch)
        return list(premises), list(hypotheses)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        if not text:
            return []
        cleaned = text.replace("\r", " ").replace("\t", " ")
        segments = re.split(r"[.!?]+\s+|\n+", cleaned)
        sentences = [segment.strip() for segment in segments if segment and segment.strip()]
        if sentences:
            return sentences
        stripped = cleaned.strip()
        return [stripped] if stripped else []

    def _batched_weighted_scores(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        *,
        progress_desc: Optional[str] = None,
    ) -> List[float]:
        if not premises:
            return []

        dataset: List[Tuple[str, str]] = list(zip(premises, hypotheses))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_pairs,
        )

        iterator: Iterable[Tuple[List[str], List[str]]] = loader
        iterator = _maybe_tqdm(
            iterator,
            enabled=self.verbose and progress_desc is not None,
            desc=progress_desc or "NLI scoring",
        )

        scores: List[float] = []
        with torch.inference_mode():
            for premises_batch, hypotheses_batch in iterator:
                if not premises_batch:
                    continue
                inputs = self.tokenizer(
                    premises_batch,
                    hypotheses_batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}

                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                for row in probs:
                    output = [
                        {
                            "label": self.id_to_label.get(idx, "NEUTRAL"),
                            "score": float(row[idx].item()),
                        }
                        for idx in range(row.shape[0])
                    ]
                    scores.append(self.process_output(output))

        return scores

    def process_output(self, output: Sequence[Dict[str, Any]]) -> float:
        return float(
            sum(
                entry["score"] * self.label_to_weight.get(entry["label"], 0)
                for entry in output
            )
        )

    def _pairwise_similarity(
        self,
        texts: Sequence[str],
        indices: Sequence[Tuple[int, int]],
        *,
        progress_desc: Optional[str] = None,
    ) -> List[float]:
        if not indices:
            return []

        premises: List[str] = []
        hypotheses: List[str] = []
        for i, j in indices:
            sentences_i = self._split_sentences(texts[i])
            sentences_j = self._split_sentences(texts[j])
            if not sentences_i or not sentences_j:
                continue
            min_len = min(len(sentences_i), len(sentences_j))
            if min_len == 0:
                continue
            sample_size = min(self.sentence_sample_size, min_len)
            available = list(range(min_len))
            if len(available) > sample_size:
                chosen_indices = self._rng.sample(available, sample_size)
            else:
                chosen_indices = available
            for idx in chosen_indices:
                sentence_a = sentences_i[idx]
                sentence_b = sentences_j[idx]
                premises.extend([sentence_a, sentence_b])
                hypotheses.extend([sentence_b, sentence_a])

        scores = self._batched_weighted_scores(
            premises,
            hypotheses,
            progress_desc=progress_desc,
        )
        if len(scores) % 2 != 0:  # pragma: no cover - sanity guard
            raise RuntimeError("NLI scoring produced an odd number of results")
        return [0.5 * (scores[idx] + scores[idx + 1]) for idx in range(0, len(scores), 2)]

    @staticmethod
    def _diversity_from_similarities(similarities: Sequence[float]) -> float:
        return float(similarity2diversity_function(similarities)) if similarities else 0.0

    def per_input(self, groups: Sequence[OutputGroup]) -> List[float]:
        scores: List[float] = []
        for group in _maybe_tqdm(groups, self.verbose, "NLI per-input"):
            pair_indices = self._select_pairs(len(group.outputs), self.pairs_per_input)
            if not pair_indices:
                scores.append(0.0)
                continue
            similarities = self._pairwise_similarity(
                group.outputs,
                pair_indices,
            )
            scores.append(self._diversity_from_similarities(similarities))
        return scores

    def across_input(self, outputs: Sequence[str]) -> float:
        pair_indices = self._select_pairs(len(outputs), self.pairs_across)
        if not pair_indices:
            return 0.0
        similarities = self._pairwise_similarity(
            outputs,
            pair_indices,
            progress_desc="NLI across-input",
        )
        return self._diversity_from_similarities(similarities)


def _summary_statistics(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = _mean(values)
    if len(values) == 1:
        std = 0.0
    else:
        std = float(statistics.pstdev(values))
    return mean, std


def _round_float(value: float, precision: int) -> float:
    return float(round(value, precision))


def _round_metric_payload(payload: Dict[str, Any], precision: int) -> Dict[str, Any]:
    for key, value in list(payload.items()):
        if isinstance(value, float):
            payload[key] = _round_float(value, precision)
    return payload


def correlate_metrics(metric_scores: Dict[str, Sequence[float]]) -> Dict[str, float]:
    """Compute Pearson correlations between per-input metric scores."""

    names = sorted(metric_scores.keys())
    correlations: Dict[str, float] = {}
    for idx, first in enumerate(names):
        scores_first = metric_scores[first]
        if len(scores_first) < 2:
            continue
        for second in names[idx + 1 :]:
            scores_second = metric_scores[second]
            if len(scores_second) != len(scores_first) or len(scores_second) < 2:
                continue
            mean_first = _mean(scores_first)
            mean_second = _mean(scores_second)
            numerator = 0.0
            denom_first = 0.0
            denom_second = 0.0
            for value_first, value_second in zip(scores_first, scores_second):
                diff_first = value_first - mean_first
                diff_second = value_second - mean_second
                numerator += diff_first * diff_second
                denom_first += diff_first * diff_first
                denom_second += diff_second * diff_second
            denominator = math.sqrt(denom_first * denom_second)
            if denominator == 0.0:
                continue
            key = f"{first}~{second}"
            correlations[key] = numerator / denominator
    return correlations


def compute_diversity(
    model_outputs_dir: Path | str,
    metrics: Sequence[str] | None = None,
    *,
    per_input: bool = True,
    across_input: bool = True,
    config: Optional[DiversityConfig] = None,
) -> Dict[str, Any]:
    """Compute diversity metrics for a directory of model generations."""

    cfg = config or DiversityConfig()
    requested_metrics = [metric.upper() for metric in (metrics or ("EAD", "SBERT", "NLI"))]

    set_seed(cfg.random_seed)
    if cfg.deterministic and hasattr(torch, "use_deterministic_algorithms"):
        try:  # pragma: no cover - depends on torch build
            torch.use_deterministic_algorithms(True)
        except (RuntimeError, AttributeError):
            pass
    groups = load_model_generations(model_outputs_dir, cfg)
    if not groups:
        raise ValueError("No generation groups were loaded from the provided directory")

    outputs_per_group: List[List[str]] = [group.outputs for group in groups]
    all_outputs: List[str] = [output for group in outputs_per_group for output in group]
    all_indices = list(range(len(all_outputs)))
    if cfg.sample_overall and len(all_indices) >= DEFAULT_OVERALL_SAMPLE_SIZE:
        overall_positions = sorted(
            np.random.choice(all_indices, replace=False, size=DEFAULT_OVERALL_SAMPLE_SIZE).tolist()
        )
    else:
        overall_positions = all_indices
    sampled_all_outputs = [all_outputs[idx] for idx in overall_positions]

    first_output_text = first_outputs(groups)
    selection_indices = list(range(len(first_output_text)))
    if cfg.sample_overall and len(selection_indices) >= DEFAULT_OVERALL_SAMPLE_SIZE:
        sampled_positions = sorted(
            np.random.choice(
                selection_indices,
                replace=False,
                size=DEFAULT_OVERALL_SAMPLE_SIZE,
            ).tolist()
        )
    else:
        sampled_positions = selection_indices
    sampled_first_outputs = [first_output_text[idx] for idx in sampled_positions]

    results: Dict[str, Any] = {
        "model_name": cfg.model_name or Path(model_outputs_dir).name,
        "N": len(groups),
        "K": cfg.generations_per_input,
        "metrics": {},
    }

    metrics_payload: Dict[str, Any] = {}
    per_input_scores_map: Dict[str, List[float]] = {}

    if "EAD" in requested_metrics:
        ead_calculator = EADDiversityCalculator(cfg.ngram_range)
        metric_result: Dict[str, Any] = {}
        overall_all = ead_calculator.across_input(sampled_all_outputs)
        overall_single = ead_calculator.across_input(sampled_first_outputs)
        metric_result["overall_EAD"] = overall_all
        metric_result["overall_single_output_EAD"] = overall_single
        if per_input:
            per_input_scores = ead_calculator.per_input(outputs_per_group)
            mean, std = _summary_statistics(per_input_scores)
            metric_result["per_input_mean"] = mean
            metric_result["per_input_std"] = std
            per_input_scores_map["EAD"] = per_input_scores
        if across_input:
            metric_result["across_input"] = overall_single
        metric_result["mean_per_input_EAD"] = metric_result.get("per_input_mean", 0.0)
        metric_result["std_per_input_EAD"] = metric_result.get("per_input_std", 0.0)
        _round_metric_payload(metric_result, cfg.output_precision)
        metrics_payload["EAD"] = metric_result

    if "SBERT" in requested_metrics:
        sbert_calculator = SentenceBERTDiversityCalculator(
            cfg.sbert_model_name,
            cfg.sbert_batch_size,
            cfg.tokenizer_max_length,
            cache_dir=cfg.sbert_cache_dir,
            verbose=cfg.verbose,
        )
        embeddings, offsets, first_indices = sbert_calculator.embed_groups(groups)
        sampled_embedding_indices = [first_indices[idx] for idx in sampled_positions]
        overall_embedding_indices = overall_positions
        metric_result = {}
        overall_single_sbert = sbert_calculator.across_input_from_embeddings(
            embeddings, sampled_embedding_indices
        )
        overall_all_sbert = sbert_calculator.across_input_from_embeddings(
            embeddings, overall_embedding_indices
        )
        if per_input:
            per_input_scores = sbert_calculator.per_input_from_embeddings(embeddings, offsets)
            mean, std = _summary_statistics(per_input_scores)
            metric_result["per_input_mean"] = mean
            metric_result["per_input_std"] = std
            per_input_scores_map["SBERT"] = per_input_scores
        if across_input:
            metric_result["across_input"] = overall_single_sbert
        metric_result["overall_single_output_SBERT"] = overall_single_sbert
        metric_result["overall_SBERT"] = overall_all_sbert
        metric_result["mean_per_input_SBERT"] = metric_result.get("per_input_mean", 0.0)
        metric_result["std_per_input_SBERT"] = metric_result.get("per_input_std", 0.0)
        _round_metric_payload(metric_result, cfg.output_precision)
        metrics_payload["SBERT"] = metric_result

    if "NLI" in requested_metrics:
        nli_evaluator = NLIDiversityEvaluator(
            cfg.nli_model_name,
            cfg.nli_batch_size,
            cfg.tokenizer_max_length,
            cfg.nli_pairs_per_input,
            cfg.nli_pairs_across,
            cfg.nli_sentence_sample_size,
            random_subsample=cfg.nli_random_subsample,
            random_seed=cfg.random_seed,
            verbose=cfg.verbose,
        )
        metric_result = {}
        overall_single_nli = nli_evaluator.across_input(sampled_first_outputs)
        overall_all_nli = nli_evaluator.across_input(sampled_all_outputs)
        if per_input:
            per_input_scores = nli_evaluator.per_input(groups)
            mean, std = _summary_statistics(per_input_scores)
            metric_result["per_input_mean"] = mean
            metric_result["per_input_std"] = std
            per_input_scores_map["NLI"] = per_input_scores
        if across_input:
            metric_result["across_input"] = overall_single_nli
        metric_result["overall_single_output_NLI"] = overall_single_nli
        metric_result["overall_NLI"] = overall_all_nli
        metric_result["mean_per_input_NLI"] = metric_result.get("per_input_mean", 0.0)
        metric_result["std_per_input_NLI"] = metric_result.get("per_input_std", 0.0)
        _round_metric_payload(metric_result, cfg.output_precision)
        metrics_payload["NLI"] = metric_result

    results["metrics"] = metrics_payload

    return results


def calculate_diversity_metrics(
    model_outputs_dir: Path | str,
    metrics: Sequence[str] | None = None,
    *,
    per_input: bool = True,
    across_input: bool = True,
    config: Optional[DiversityConfig] = None,
) -> Dict[str, Any]:
    return compute_diversity(
        model_outputs_dir,
        metrics,
        per_input=per_input,
        across_input=across_input,
        config=config,
    )


def plot_diversity_bars(
    results: Sequence[Dict[str, Any]],
    metric: str,
    *,
    output_path: Optional[Path | str] = None,
    figsize: Tuple[float, float] = (8.0, 5.0),
    log_scale: bool = False,
) -> None:
    """Visualize diversity scores for multiple models.

    Parameters
    ----------
    results:
        Iterable of dictionaries as produced by :func:`compute_diversity`.
    metric:
        Metric name (``"EAD"``, ``"SBERT"``, or ``"NLI"``).
    output_path:
        Optional file path. When provided the plot is saved, otherwise the plot
        is shown interactively.
    figsize:
        Figure size passed to :func:`matplotlib.pyplot.subplots`.
    log_scale:
        When ``True`` the y-axis is rendered in log scale.
    """

    import matplotlib.pyplot as plt  # Local import to keep dependency optional

    metric = metric.upper()
    labels: List[str] = []
    means: List[float] = []
    stds: List[float] = []
    across_values: List[Optional[float]] = []

    for entry in results:
        labels.append(entry.get("model_name", "model"))
        metric_data = entry.get("metrics", {}).get(metric, {})
        means.append(float(metric_data.get("per_input_mean", 0.0)))
        stds.append(float(metric_data.get("per_input_std", 0.0)))
        across_values.append(metric_data.get("across_input"))

    positions = range(len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(positions, means, yerr=stds, alpha=0.7, color="#4c72b0", label="Per-input mean")

    if any(value is not None for value in across_values):
        valid_positions = [pos for pos, value in zip(positions, across_values) if value is not None]
        valid_values = [value for value in across_values if value is not None]
        ax.plot(valid_positions, valid_values, marker="o", linestyle="--", color="#c44e52", label="Across-input")

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(f"{metric} diversity")
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(f"Diversity comparison for {metric}")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()

