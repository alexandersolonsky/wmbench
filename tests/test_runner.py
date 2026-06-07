"""Phase 3: config loading + end-to-end runner with the stub watermarker."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from wmbench.config import BenchConfig, load_config
from wmbench.core.registry import register_algorithm
from wmbench.runner import find_images, make_message, run, write_outputs

from conftest import StubWatermarker


@pytest.fixture(autouse=True)
def _register_stub():
    register_algorithm("stub")(lambda: StubWatermarker())


def _make_images(d, n=2):
    rng = np.random.default_rng(7)
    for i in range(n):
        arr = rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(d / f"img{i}.png")


def test_load_config_roundtrip(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("algorithms: [stub]\ninput_dir: ./imgs\n")
    loaded = load_config(cfg)
    assert loaded.algorithms == ["stub"]
    assert loaded.distortions == ["none"]  # default


def test_load_config_rejects_unknown_key(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("algorithms: [stub]\ninput_dir: ./i\nbogus: 1\n")
    with pytest.raises(ValueError):
        load_config(cfg)


def test_make_message_deterministic():
    a = make_message(32, 0, "stub", None)
    b = make_message(32, 0, "stub", None)
    assert a == b and len(a) == 32
    assert make_message(8, 0, "x", [1, 1]) == [1, 1, 0, 0, 0, 0, 0, 0]


def test_find_images(tmp_path):
    _make_images(tmp_path, 3)
    (tmp_path / "notes.txt").write_text("ignore me")
    # images in a resolution subfolder are found recursively
    sub = tmp_path / "1080p"
    sub.mkdir()
    _make_images(sub, 2)
    assert len(find_images(tmp_path)) == 5


def test_run_end_to_end(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    _make_images(imgs, 2)
    out = tmp_path / "out"

    cfg = BenchConfig(algorithms=["stub"], input_dir=str(imgs),
                      distortions=["none"], output_dir=str(out))
    payload = run(cfg)

    assert payload["algorithms"] == ["stub"]
    results = payload["results"]
    # 2 images × 1 distortion × {watermarked, clean} = 4 rows
    assert len(results) == 4

    wm_rows = [r for r in results if r["condition"] == "watermarked"]
    assert all(r["bit_acc"] == 1.0 and r["present"] is True for r in wm_rows)
    assert all(r["quality"]  # quality dict populated for watermarked rows
               for r in wm_rows)

    clean_rows = [r for r in results if r["condition"] == "clean"]
    assert all(isinstance(r["is_false_positive"], bool) for r in clean_rows)

    json_path, csv_path = write_outputs(payload, out)
    assert json_path.exists() and csv_path.exists()
    reloaded = json.loads(json_path.read_text())
    assert reloaded["results"] == results
    # CSV flattens the quality dict into columns
    header = csv_path.read_text().splitlines()[0]
    for col in ("vmaf", "ssimulacra2", "xpsnr", "psnr_hvs_m"):
        assert col in header


def test_resume_only_computes_new_distortions(tmp_path):
    from wmbench.core.registry import register_algorithm, register_distortion
    from wmbench.runner import write_outputs

    class CountStub(StubWatermarker):
        embeds = 0
        extracts = 0
        def embed(self, image, message):
            type(self).embeds += 1
            return super().embed(image, message)
        def extract(self, image):
            type(self).extracts += 1
            return super().extract(image)

    stub = CountStub()
    register_algorithm("cstub")(lambda: stub)

    class _Dummy:
        name = "d_dummy"; group = "test"
        def __init__(self): self.params = {}
        def apply(self, image): return image.copy()
    register_distortion("d_dummy")(lambda: _Dummy())

    imgs = tmp_path / "imgs"; imgs.mkdir(); _make_images(imgs, 2)
    out = tmp_path / "out"

    # first run: only the "none" distortion
    p1 = run(BenchConfig(algorithms=["cstub"], input_dir=str(imgs),
                         distortions=["none"], output_dir=str(out)))
    write_outputs(p1, out)
    embeds_after_first = CountStub.embeds
    assert len(p1["results"]) == 2 * 2  # 2 images x {watermarked, clean}

    # second run, resume, ADD d_dummy: "none" reused, only d_dummy computed,
    # and no re-embedding (watermarked loaded from the gallery).
    p2 = run(BenchConfig(algorithms=["cstub"], input_dir=str(imgs),
                         distortions=["none", "d_dummy"], output_dir=str(out), resume=True))
    assert len(p2["results"]) == 2 * 2 * 2  # 2 images x 2 distortions x 2 conditions
    # embed only fired for the one-shot warmup, not for re-embedding the images
    assert CountStub.embeds - embeds_after_first == 1
    # the reused "none" rows are byte-identical to the first run
    none1 = {(r["image"], r["condition"]): r for r in p1["results"] if r["distortion"] == "none"}
    none2 = {(r["image"], r["condition"]): r for r in p2["results"] if r["distortion"] == "none"}
    assert none1 == none2
