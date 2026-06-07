"""Phase 5: report aggregation (pure) and static-site build."""

from __future__ import annotations

import json

from wmbench.report.generate import aggregate, build_site


def _payload():
    return {
        "meta": {"generated_at": "2026-06-03T00:00:00Z", "success_threshold": 0.99,
                 "tools": {"vmaf": True, "ssimulacra2": True, "xpsnr": True, "psnr_hvs_m": True}},
        "algorithms": ["a1", "a2"],
        "distortions": ["none"],
        "results": [
            {"algo": "a1", "image": "x.png", "distortion": "none",
             "condition": "watermarked", "quality": {"vmaf": 98.0, "ssimulacra2": 90.0,
             "xpsnr": 50.0, "psnr_hvs_m": 45.0}, "bit_acc": 1.0, "present": True,
             "confidence": 0.99, "is_false_positive": False},
            {"algo": "a1", "image": "x.png", "distortion": "none",
             "condition": "clean", "quality": {}, "bit_acc": 0.5, "present": False,
             "confidence": 0.1, "is_false_positive": False},
            {"algo": "a2", "image": "x.png", "distortion": "none",
             "condition": "watermarked", "quality": {"vmaf": 80.0, "ssimulacra2": 70.0,
             "xpsnr": 40.0, "psnr_hvs_m": 35.0}, "bit_acc": 0.5, "present": True,
             "confidence": 0.6, "is_false_positive": False},
            {"algo": "a2", "image": "x.png", "distortion": "none",
             "condition": "clean", "quality": {}, "bit_acc": 0.9, "present": True,
             "confidence": 0.8, "is_false_positive": True},
        ],
        "gallery": [],
    }


def test_aggregate_quality_and_extraction():
    agg = aggregate(_payload())
    assert agg["quality_by_algo"]["a1"]["vmaf"] == 98.0
    # extraction rate = exact match (extracted ID == embedded), not mean bit-acc
    a1 = next(e for e in agg["extraction"] if e["algo"] == "a1")
    assert a1["extraction_rate"] == 1.0       # bit_acc 1.0 -> exact match
    a2 = next(e for e in agg["extraction"] if e["algo"] == "a2")
    assert a2["extraction_rate"] == 0.0       # bit_acc 0.5 -> not exact
    assert a1["mean_bit_acc"] == 1.0          # bit-acc still reported separately


def test_aggregate_points_for_scatter():
    agg = aggregate(_payload())
    # one point per algorithm: X = exact-match extraction rate
    assert len(agg["points"]) == 2
    pa = {p["algo"]: p for p in agg["points"]}
    assert pa["a1"]["extraction_rate"] == 1.0
    assert pa["a2"]["extraction_rate"] == 0.0
    assert pa["a1"]["quality"]["vmaf"] == 98.0 and "ssimulacra2" in pa["a1"]["quality"]


def test_aggregate_false_positive_rate():
    agg = aggregate(_payload())
    f1 = next(f for f in agg["fpr"] if f["algo"] == "a1")
    f2 = next(f for f in agg["fpr"] if f["algo"] == "a2")
    assert f1["fpr"] == 0.0
    assert f2["fpr"] == 1.0


def test_build_site_is_self_contained(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(json.dumps(_payload()))
    out = tmp_path / "site"
    index = build_site(results, out)

    html = index.read_text()
    assert "wmbench" in html
    assert "cdn." not in html.lower()  # no CDN references -> offline-safe
    assert 'src="assets/chart.umd.min.js"' in html
    assert (out / "assets" / "chart.umd.min.js").exists()
    # embedded data drives the charts
    assert "const DATA =" in html
