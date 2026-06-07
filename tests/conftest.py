"""Shared test fixtures, including a stub watermarker used in place of the real
(torch) adapters. The stub is test-only — it is never registered or shipped by
the package itself.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from wmbench.core.interfaces import ExtractResult


class StubWatermarker:
    """A trivial, dependency-free watermarker for tests.

    Embeds the payload bits into the least-significant bit of the first
    ``payload_bits`` pixels (red channel), recovers them exactly, and reports a
    confidence based on how cleanly the LSBs decode. Good enough to exercise the
    runner, metrics plumbing and report — not a shipped algorithm.
    """

    name = "stub"
    payload_bits = 32

    def embed(self, image: Image.Image, message: list[int]) -> Image.Image:
        arr = np.array(image.convert("RGB"))
        flat = arr.reshape(-1, 3)
        for i, bit in enumerate(message[: self.payload_bits]):
            flat[i, 0] = (int(flat[i, 0]) & ~1) | (bit & 1)
        return Image.fromarray(flat.reshape(arr.shape), "RGB")

    def extract(self, image: Image.Image) -> ExtractResult:
        arr = np.array(image.convert("RGB")).reshape(-1, 3)
        bits = [int(arr[i, 0]) & 1 for i in range(self.payload_bits)]
        # Present if the recovered pattern isn't the degenerate all-zeros.
        present = any(bits)
        return ExtractResult(message=bits, present=present, confidence=1.0 if present else 0.0)


@pytest.fixture
def stub() -> StubWatermarker:
    return StubWatermarker()


@pytest.fixture
def sample_image() -> Image.Image:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def message() -> list[int]:
    rng = np.random.default_rng(1)
    return rng.integers(0, 2, size=32).tolist()
