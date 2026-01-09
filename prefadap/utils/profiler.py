from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in tests
    torch = None  # type: ignore


@dataclass
class RunProfile:
    examples: int = 0
    total_tokens: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float = 0.0
    gpu_mem_mb: Optional[int] = None
    gpu_util_percent: Optional[int] = None

    def update(self, tokens: int, n_examples: int) -> None:
        self.total_tokens += int(tokens)
        self.examples += int(n_examples)

    @property
    def elapsed(self) -> float:
        end = self.end_time or time.perf_counter()
        return max(1e-9, end - self.start_time)

    @property
    def tokens_per_sec(self) -> float:
        return float(self.total_tokens) / self.elapsed if self.total_tokens else 0.0

    @property
    def time_per_example(self) -> float:
        return self.elapsed / max(1, self.examples)

    def finalize(self) -> None:
        self.end_time = time.perf_counter()
        self._capture_gpu_metrics()

    def _capture_gpu_metrics(self) -> None:
        # Prefer torch memory info if available
        try:
            if torch is not None and torch.cuda.is_available():  # type: ignore[attr-defined]
                mem = torch.cuda.max_memory_allocated()  # bytes
                self.gpu_mem_mb = int(mem / (1024 * 1024))
        except Exception:
            pass

        # Try NVIDIA smi then ROCm smi as fallbacks
        try:
            if shutil.which("nvidia-smi"):
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                line = out.splitlines()[0]
                util_s, mem_s = [x.strip() for x in line.split(",")]
                if self.gpu_util_percent is None:
                    self.gpu_util_percent = int(util_s)
                if self.gpu_mem_mb is None:
                    self.gpu_mem_mb = int(mem_s)
                return
        except Exception:
            pass

        try:
            if shutil.which("rocm-smi"):
                # utilization
                util_txt = subprocess.check_output(
                    ["rocm-smi", "--showuse"], text=True, stderr=subprocess.DEVNULL
                )
                # Parse first occurrence like: "GPU[0] : GPU use: 12%"
                for line in util_txt.splitlines():
                    line = line.strip()
                    if "GPU use:" in line and "%" in line:
                        try:
                            val = line.split("GPU use:")[-1].strip().rstrip("% ")
                            self.gpu_util_percent = int(val)
                        except Exception:
                            pass
                        break
                # memory (vram used). rocm-smi --showmeminfo vram prints used in bytes
                mem_txt = subprocess.check_output(
                    ["rocm-smi", "--showmeminfo", "vram"], text=True, stderr=subprocess.DEVNULL
                )
                # Look for "VRAM Used (B): <bytes>"
                for line in mem_txt.splitlines():
                    if "VRAM Used (B)" in line:
                        try:
                            val = "".join(ch for ch in line if ch.isdigit())
                            if val:
                                used_mb = int(val) // (1024 * 1024)
                                self.gpu_mem_mb = used_mb
                                break
                        except Exception:
                            pass
        except Exception:
            pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "examples": self.examples,
            "comparisons": getattr(self, "comparisons", 0),
            "total_tokens": self.total_tokens,
            "elapsed_sec": self.elapsed,
            "tokens_per_sec": self.tokens_per_sec,
            "time_per_example_sec": self.time_per_example,
            "gpu_mem_mb": self.gpu_mem_mb,
            "gpu_util_percent": self.gpu_util_percent,
        }

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def append_jsonl(self, path: Path, extra: Optional[Dict[str, Any]] = None) -> None:
        """Append a snapshot to a JSONL file and flush immediately."""
        self._capture_gpu_metrics()
        record = self.to_dict()
        if extra:
            record.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()


def consolidate_profile(jsonl_path: Path, out_json: Optional[Path] = None) -> Dict[str, Any]:
    """Consolidate a JSONL profile file into a single summary JSON.

    If ``out_json`` is provided, writes the summary there.
    """
    if not jsonl_path.exists():
        return {}
    last: Dict[str, Any] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except Exception:
                continue
    if out_json is not None and last:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(last, f, indent=2)
    return last


__all__ = ["RunProfile", "consolidate_profile"]
