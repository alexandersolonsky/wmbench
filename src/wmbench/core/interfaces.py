"""Core data structures and plugin protocols for wmbench.

These define the contract every watermarking algorithm and every distortion must
satisfy. They intentionally depend only on Pillow so the module is import-light.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

from PIL import Image


@dataclass
class ExtractResult:
    """Outcome of running a detector/extractor on one image.

    Adapters normalize their backend's native output into this shape.
    """

    message: list[int]  # recovered payload bits (0/1)
    present: bool  # detector's hard decision: is a watermark present?
    confidence: float  # detector confidence, normalized to [0, 1]


@runtime_checkable
class Watermarker(Protocol):
    """A watermarking algorithm: embed a bit message, later recover it.

    Implementations are registered via ``register_algorithm``. Heavy backends
    (torch, model weights) must be imported lazily inside ``__init__``/methods so
    that merely importing the adapter module never requires them.
    """

    name: str
    payload_bits: int

    def embed(self, image: Image.Image, message: list[int]) -> Image.Image:
        """Return a watermarked copy of ``image`` carrying ``message``."""
        ...

    def extract(self, image: Image.Image) -> ExtractResult:
        """Detect and recover the payload from ``image``."""
        ...


@runtime_checkable
class Distortion(Protocol):
    """A transform applied to a (watermarked) image before extraction.

    The ``none`` distortion is the control used both for the clean quality
    measurement and the false-positive pass. Real distortions are added later and
    register the same way.
    """

    name: str
    group: str   # thematic family, e.g. "compression", "geometric", "none"
    params: dict

    def apply(self, image: Image.Image) -> Image.Image:
        """Return a distorted copy of ``image``."""
        ...


@dataclass
class ResultRow:
    """One (algorithm, image, distortion) cell of the benchmark matrix.

    Serialized into ``results.json`` for the static report. ``quality`` holds the
    perceptual metrics measured on the *undistorted* watermarked image vs. the
    original; any metric whose tool is unavailable is ``None``.
    """

    algo: str
    image: str
    distortion: str
    group: str = "none"
    params: dict = field(default_factory=dict)
    # "watermarked" rows carry a payload; "clean" rows are the no-watermark
    # control used to measure the false-positive rate.
    condition: str = "watermarked"
    quality: dict = field(default_factory=dict)  # vmaf / ssimulacra2 / xpsnr / psnr_hvs_m
    bit_acc: float | None = None
    present: bool | None = None
    confidence: float | None = None
    is_false_positive: bool = False
    embed_s: float | None = None    # time to embed the watermark (per image)
    extract_s: float | None = None  # time to extract/detect (per result)

    def to_dict(self) -> dict:
        return asdict(self)
