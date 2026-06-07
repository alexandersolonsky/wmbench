"""Phase 1: interfaces, registry, none distortion, import-lightness."""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from wmbench import (
    ExtractResult,
    ResultRow,
    get_distortion,
    list_distortions,
    register_algorithm,
)


def test_import_is_lightweight():
    # Importing wmbench must not drag in heavy/optional backends. Checked in a
    # fresh interpreter so it's unaffected by other tests that load torch.
    code = (
        "import sys, wmbench;"
        "heavy=[m for m in ('torch','videoseal','trustmark') if m in sys.modules];"
        "print(heavy); sys.exit(1 if heavy else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"import wmbench pulled in: {proc.stdout.strip()}"


def test_stub_roundtrip(stub, sample_image, message):
    wm = stub.embed(sample_image, message)
    result = stub.extract(wm)
    assert isinstance(result, ExtractResult)
    assert result.message == message
    assert result.present is True
    assert result.confidence == 1.0


def test_none_distortion_registered(sample_image):
    assert "none" in list_distortions()
    ident = get_distortion("none")
    out = ident.apply(sample_image)
    assert np.array_equal(np.array(out), np.array(sample_image))
    assert out is not sample_image  # returns a copy


def test_register_and_get_algorithm(stub):
    register_algorithm("stub")(lambda: stub)
    from wmbench import get_algorithm, list_algorithms

    assert "stub" in list_algorithms()
    assert get_algorithm("stub") is stub


def test_resultrow_serializes():
    row = ResultRow(
        algo="stub",
        image="a.png",
        distortion="none",
        quality={"vmaf": 99.1, "ssimulacra2": 88.0, "xpsnr": 50.2, "psnr_hvs_m": 47.0},
        bit_acc=1.0,
        present=True,
        confidence=1.0,
    )
    d = row.to_dict()
    assert d["quality"]["vmaf"] == 99.1
    assert d["is_false_positive"] is False
