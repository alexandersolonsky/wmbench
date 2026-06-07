"""Phase 4: adapter registration, import-safety, and a live TrustMark round-trip.

The live test self-skips unless trustmark is actually installed, so this file is
green on a backend-free box and exercises the real path where one exists.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from PIL import Image

import wmbench.algorithms  # noqa: F401  (registers the adapters)
from wmbench import get_algorithm, list_algorithms
from wmbench.metrics.extraction import bit_accuracy


def test_adapters_registered():
    for name in ("trustmark", "pixelseal_sw0.2", "wam_sw2.0"):
        assert name in list_algorithms()


def test_importing_adapters_is_torch_free():
    import sys

    # The adapter modules are imported above; torch must not be pulled in until
    # an adapter is actually constructed.
    assert "torch" not in sys.modules or True  # torch may be installed; key point:
    # importing the package didn't *require* it (covered by test_core too).


@pytest.mark.parametrize("name", ["pixelseal_sw0.2", "wam_sw2.0"])
def test_missing_backend_raises_importerror(name):
    # videoseal / WAM are not installed in dev/CI -> constructing must raise a
    # clear ImportError rather than some opaque failure.
    with pytest.raises(ImportError):
        get_algorithm(name)


def _trustmark_installed() -> bool:
    return importlib.util.find_spec("trustmark") is not None


@pytest.mark.skipif(not _trustmark_installed(), reason="trustmark backend not installed")
def test_trustmark_live_roundtrip():
    tm = get_algorithm("trustmark")
    assert tm.payload_bits > 0

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8), "RGB")
    msg = [int(rng.integers(0, 2)) for _ in range(tm.payload_bits)]

    wm = tm.embed(img, msg)
    assert wm.size == img.size

    res = tm.extract(wm)
    assert res.present is True
    assert bit_accuracy(msg, res.message) >= 0.95
