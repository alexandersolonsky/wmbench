"""Phase 2: metric parsers (pure), extraction metrics, and a guarded integration test."""

from __future__ import annotations

import pytest
from PIL import Image

from wmbench.core.interfaces import ExtractResult
from wmbench.metrics import tools
from wmbench.metrics.extraction import bit_accuracy, false_positive, wilson_interval
from wmbench.metrics.quality import QUALITY_METRICS, measure_quality


# ---- pure parsers --------------------------------------------------------- #
VMAF_JSON_POOLED = """
{"version":"...","pooled_metrics":{"vmaf":{"min":90.1,"max":99.9,"mean":97.5,
"harmonic_mean":97.4},"psnr_hvs":{"min":40.0,"max":45.0,"mean":43.2}}}
"""

VMAF_JSON_FRAMES = """
{"frames":[{"frameNum":0,"metrics":{"vmaf":80.0,"psnr_hvs":30.0}},
{"frameNum":1,"metrics":{"vmaf":90.0,"psnr_hvs":40.0}}]}
"""


def test_parse_vmaf_pooled():
    assert tools.parse_vmaf_json(VMAF_JSON_POOLED) == 97.5
    assert tools.parse_psnr_hvs_json(VMAF_JSON_POOLED) == 43.2


def test_parse_vmaf_frames_mean():
    assert tools.parse_vmaf_json(VMAF_JSON_FRAMES) == 85.0
    assert tools.parse_psnr_hvs_json(VMAF_JSON_FRAMES) == 35.0


def test_parse_psnr_hvs_per_channel_fallback():
    data = '{"pooled_metrics":{"psnr_hvs_y":{"mean":42.0}}}'
    assert tools.parse_psnr_hvs_json(data) == 42.0


def test_parse_xpsnr():
    line = "[Parsed_xpsnr_0 @ 0x55] XPSNR  y: 41.5183  u: 47.1751  v: 46.3409"
    assert tools.parse_xpsnr_stderr(line) == 41.5183


def test_parse_xpsnr_inf():
    assert tools.parse_xpsnr_stderr("XPSNR  y: inf  u: inf  v: inf") == float("inf")


def test_parse_xpsnr_missing():
    assert tools.parse_xpsnr_stderr("no metric here") is None


def test_parse_ssimulacra2_bare_number():
    assert tools.parse_ssimulacra2_stdout("87.61\n") == 87.61


def test_parse_ssimulacra2_labeled():
    assert tools.parse_ssimulacra2_stdout("Score: 87.61\n") == 87.61


# ---- extraction metrics --------------------------------------------------- #
def test_bit_accuracy():
    assert bit_accuracy([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert bit_accuracy([1, 0, 1, 0], [1, 1, 1, 1]) == 0.5
    assert bit_accuracy([], []) == 0.0


def test_wilson_interval_bounds():
    low, high = wilson_interval(10, 10)
    assert low <= 1.0 and high == pytest.approx(1.0, abs=0.05) or high <= 1.0
    assert 0.0 <= low <= high <= 1.0
    # wider band for small n
    low5, high5 = wilson_interval(5, 5)
    low100, high100 = wilson_interval(100, 100)
    assert (high5 - low5) > (high100 - low100)


def test_wilson_interval_empty():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_false_positive_present():
    r = ExtractResult(message=[0, 0], present=True, confidence=0.9)
    assert false_positive(r) is True


def test_false_positive_message_match():
    r = ExtractResult(message=[1, 0, 1], present=False, confidence=0.1)
    assert false_positive(r, registered_message=[1, 0, 1], threshold=0.99) is True
    assert false_positive(r, registered_message=[0, 1, 0], threshold=0.99) is False


def test_false_positive_negative():
    r = ExtractResult(message=[0, 1], present=False, confidence=0.0)
    assert false_positive(r) is False


# ---- guarded integration -------------------------------------------------- #
@pytest.mark.skipif(
    not any(tools.available_metrics().values()),
    reason="no external quality-metric tools installed",
)
def test_measure_quality_integration():
    ref = Image.new("RGB", (128, 128), (120, 130, 140))
    wm = ref.copy()
    wm.putpixel((0, 0), (121, 130, 140))  # tiny perturbation
    q = measure_quality(ref, wm)
    assert set(q) == set(QUALITY_METRICS)
    # At least one metric should have produced a number.
    assert any(v is not None for v in q.values())
