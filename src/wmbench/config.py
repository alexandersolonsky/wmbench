"""Experiment configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BenchConfig:
    """One benchmark run.

    ``input_dir`` (a user-provided folder of images) and ``distortions`` are the
    parts expected to grow over time; everything else has a sensible default.
    """

    algorithms: list[str]
    input_dir: str
    distortions: list[str] = field(default_factory=lambda: ["none"])
    seed: int = 0
    success_threshold: float = 0.99
    output_dir: str = "results"
    # Reuse cached (algorithm, image, distortion) results + saved watermarked
    # images from a prior run in output_dir; only compute what's missing. Use
    # when ADDING distortions/algorithms — not when changing algorithm settings.
    resume: bool = False
    # How many (algorithm, image) sample triptychs to save for the report's
    # original / watermarked / distorted gallery (0 disables).
    save_samples: int = 3
    # Optional fixed payload; if None, a random payload is drawn per algorithm
    # (each algorithm has its own ``payload_bits`` width).
    message_bits: list[int] | None = None

    @property
    def input_path(self) -> Path:
        return Path(self.input_dir).expanduser()

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir).expanduser()


def load_config(path: str | Path) -> BenchConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "algorithms" not in data or "input_dir" not in data:
        raise ValueError("config must define 'algorithms' and 'input_dir'")
    known = BenchConfig.__dataclass_fields__.keys()
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return BenchConfig(**data)
