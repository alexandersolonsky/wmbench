"""Merging results from multiple runs (e.g. per-algorithm environments)."""

from __future__ import annotations

import json

from PIL import Image

from wmbench.merge import merge_runs


def _write_run(d, algo, image_name):
    (d / "gallery").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(d / "gallery" / f"{image_name}__source.png")
    Image.new("RGB", (8, 8)).save(d / "gallery" / f"{image_name}__{algo}.png")
    payload = {
        "meta": {"tools": {"vmaf": True, "xpsnr": False}, "success_threshold": 0.99},
        "algorithms": [algo], "distortions": ["none"],
        "results": [{"algo": algo, "image": f"{image_name}.png", "distortion": "none",
                     "condition": "watermarked", "quality": {"vmaf": 90.0},
                     "bit_acc": 1.0, "present": True, "confidence": 1.0,
                     "is_false_positive": False}],
        "gallery": [{"image": f"{image_name}.png", "algo": algo,
                     "source": f"gallery/{image_name}__source.png",
                     "watermarked": f"gallery/{image_name}__{algo}.png",
                     "quality": {"vmaf": 90.0}}],
    }
    (d / "results.json").write_text(json.dumps(payload))


def test_merge_combines_runs_and_gallery(tmp_path):
    r1, r2, out = tmp_path / "r1", tmp_path / "r2", tmp_path / "out"
    r1.mkdir(); r2.mkdir()
    # two algorithms over the SAME image -> one gallery entry, two watermarked
    _write_run(r1, "trustmark", "pic")
    _write_run(r2, "pixelseal", "pic")

    out_json = merge_runs([r1, r2], out)
    merged = json.loads(out_json.read_text())

    assert merged["algorithms"] == ["trustmark", "pixelseal"]
    assert len(merged["results"]) == 2
    assert merged["meta"]["tools"] == {"vmaf": True, "xpsnr": False}

    # one gallery entry for the shared image, with both algorithms' watermarks
    assert len(merged["gallery"]) == 1
    g = merged["gallery"][0]
    assert g["image"] == "pic.png"
    assert g["source"] == "gallery/pic__source.png"
    assert set(g["watermarked"]) == {"trustmark", "pixelseal"}
    assert g["watermarked"]["pixelseal"] == "gallery/pic__pixelseal.png"

    # images were actually copied
    assert (out / "gallery" / "pic__source.png").exists()
    assert (out / "gallery" / "pic__trustmark.png").exists()
    assert (out / "gallery" / "pic__pixelseal.png").exists()
