"""Generative preference pseudolabeling technique."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterator, List

from ..formats.dpo import DPOFormatter
from ..formats.kto import KTOFormatter
from ..generators.gemini_generator import GeminiGenerator
from ..generators.vllm_generator import VLLMGenerator
from .base import BasePseudolabeler


class GenerativePreferencePseudolabeler(BasePseudolabeler):
    """Creates preference pairs by generating good summaries and using references as worse alternatives."""
    
    def _setup_model(self) -> None:
        """Initialize the generator."""
        if self.config.model_type == "vllm":
            self.generator = VLLMGenerator(
                model_name=self.config.model_name,
                max_tokens=self.config.max_tokens,
                max_model_len=self.config.max_model_len,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                max_num_seqs=self.config.max_num_seqs,
                dtype=self.config.dtype,
                enable_prefix_caching=self.config.enable_prefix_caching,
                block_size=self.config.block_size,
                swap_space=self.config.swap_space,
            )
        elif self.config.model_type == "gemini":
            self.generator = GeminiGenerator(
                model_name=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                free_tier=self.config.free_tier,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        # Load prompts
        self._load_prompts()
    
    def _load_prompts(self) -> None:
        """Load prompt templates for the model type."""
        model_suffix = self.config.model_type
        
        try:
            # Try new location first
            self.system_prompt = (
                resources.files("prefadap.prompts.pseudolabelers")
                .joinpath(f"generative_system_{model_suffix}.txt")
                .read_text("utf-8")
            )
            self.user_template = (
                resources.files("prefadap.prompts.pseudolabelers")
                .joinpath(f"generative_user_{model_suffix}.txt")
                .read_text("utf-8")
            )
        except FileNotFoundError:
            # Fallback to default location
            repo_root = Path(__file__).resolve().parents[3]
            system_path = repo_root / "prefadap" / "prompts" / "pseudolabelers" / f"generative_system_{model_suffix}.txt"
            user_path = repo_root / "prefadap" / "prompts" / "pseudolabelers" / f"generative_user_{model_suffix}.txt"
            
            self.system_prompt = system_path.read_text(encoding="utf-8")
            self.user_template = user_path.read_text(encoding="utf-8")
    
    def _generate_summary(self, article: str) -> str:
        """Generate a high-quality summary for the given article (single attempt)."""
        user_prompt = self.user_template.format(article=article)
        full_prompt = f"{self.system_prompt}\n\n{user_prompt}"
        return self.generator.generate_single(full_prompt)
    
    def generate_preferences(self, prompts: List[str], references: List[str]) -> Iterator[Dict[str, Any]]:
        """Generate preference pairs by creating good summaries and using references as alternatives.

        Uses vLLM batching and parallel sampling (n=k) when available; otherwise
        falls back to sequential generation.
        """
        if self.config.model_type != "vllm":
            for prompt, reference in zip(prompts, references):
                generated_summaries = []
                for _ in range(self.config.k):
                    generated = self._generate_summary(prompt)
                    if generated and generated != reference:
                        generated_summaries.append(generated)
                for generated in generated_summaries:
                    if self.config.output_format == "dpo":
                        yield DPOFormatter.format_preference(
                            prompt=prompt,
                            chosen=generated,
                            rejected=reference,
                            weight=1.0,
                            technique="generative_preference",
                            generated_by=self.config.model_name,
                        )
                    else:
                        yield KTOFormatter.format_completion(
                            prompt=prompt,
                            completion=generated,
                            label=True,
                            weight=1.0,
                            technique="generative_preference",
                            generated_by=self.config.model_name,
                        )
                        yield KTOFormatter.format_completion(
                            prompt=prompt,
                            completion=reference,
                            label=False,
                            weight=1.0,
                            technique="generative_preference",
                        )
            return

        B = max(1, int(self.config.batch_size))
        for i in range(0, len(prompts), B):
            batch_prompts = prompts[i:i + B]
            batch_refs = references[i:i + B]
            full_prompts = []
            for article in batch_prompts:
                user_prompt = self.user_template.format(article=article)
                full_prompts.append(f"{self.system_prompt}\n\n{user_prompt}")

            candidate_lists = self.generator.generate_multi(
                full_prompts,
                n=self.config.k,
                batch_size=self.config.batch_size,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
            )

            for prompt, reference, cands in zip(batch_prompts, batch_refs, candidate_lists):
                for generated in cands:
                    if not generated or generated == reference:
                        continue
                    if self.config.output_format == "dpo":
                        yield DPOFormatter.format_preference(
                            prompt=prompt,
                            chosen=generated,
                            rejected=reference,
                            weight=1.0,
                            technique="generative_preference",
                            generated_by=self.config.model_name,
                        )
                    else:
                        yield KTOFormatter.format_completion(
                            prompt=prompt,
                            completion=generated,
                            label=True,
                            weight=1.0,
                            technique="generative_preference",
                            generated_by=self.config.model_name,
                        )
                        yield KTOFormatter.format_completion(
                            prompt=prompt,
                            completion=reference,
                            label=False,
                            weight=1.0,
                            technique="generative_preference",
                        )


__all__ = ["GenerativePreferencePseudolabeler"]
