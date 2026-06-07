"""Benchmark runner: orchestrate the algorithm × image × distortion matrix.

For each watermarking algorithm and input image it embeds a payload, measures the
perceptual quality cost once (undistorted), then for each distortion extracts and
scores the recovery. A parallel "clean" pass runs the detector on the
*non-watermarked* image to measure false positives. Output is ``results.json``
(for the report) and ``results.csv``.
"""

from __future__ import annotations

import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from wmbench.config import BenchConfig, load_config
from wmbench.core.interfaces import ResultRow, Watermarker
from wmbench.core.registry import get_algorithm, get_distortion
from wmbench.metrics.extraction import bit_accuracy, false_positive
from wmbench.metrics.quality import measure_quality
from wmbench.metrics.tools import available_metrics

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _log(msg: str) -> None:
    print(f"[wmbench] {msg}", file=sys.stderr)


def _ensure_adapters_registered() -> None:
    """Import the adapters package so real algorithms register themselves.

    Import failures (a backend not installed) are non-fatal: those algorithms
    simply won't be available, and the run skips them with a clear message.
    """
    try:
        import wmbench.algorithms  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on environment
        _log(f"note: could not import algorithm adapters ({exc!r}); "
             "only explicitly-registered algorithms are available")


def find_images(input_dir: Path) -> list[Path]:
    """All images under ``input_dir``, searched recursively (so resolution
    subfolders like ``1080p/``, ``4K/`` are picked up)."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def make_message(payload_bits: int, seed: int, name: str,
                 fixed: list[int] | None) -> list[int]:
    """Deterministic payload for (run seed, algorithm). A fixed payload is
    truncated/zero-padded to the algorithm's width."""
    if fixed is not None:
        bits = list(fixed)[:payload_bits]
        return bits + [0] * (payload_bits - len(bits))
    rng = random.Random(f"{seed}:{name}")
    return [rng.randint(0, 1) for _ in range(payload_bits)]


def _resolve_algorithms(names: list[str]) -> list[Watermarker]:
    algos: list[Watermarker] = []
    for name in names:
        try:
            algos.append(get_algorithm(name))
        except Exception as exc:
            _log(f"skipping algorithm {name!r}: {exc}")
    return algos


def run(config: BenchConfig) -> dict:
    _ensure_adapters_registered()
    algos = _resolve_algorithms(config.algorithms)
    if not algos:
        raise RuntimeError(
            f"no usable algorithms among {config.algorithms}; nothing to run")

    distortions = [get_distortion(d) for d in config.distortions]
    images = find_images(config.input_path)
    if not images:
        raise RuntimeError(f"no images found in {config.input_path}")

    _log(f"algorithms={[a.name for a in algos]} images={len(images)} "
         f"distortions={[d.name for d in distortions]}")

    gallery_dir = config.output_path / "gallery"

    # Resume cache: reuse a prior run's rows, quality, embed time, gallery and
    # PSNR scale; only compute (algo, image, distortion) combinations that are
    # missing. Watermarked images are loaded from the gallery to skip re-embed.
    cache_rows: dict[tuple, dict] = {}
    cache_quality: dict[tuple, dict] = {}
    cache_embed_s: dict[tuple, float] = {}
    gallery: list[dict] = []
    psnr_block_min: float | None = None
    psnr_block_max: float | None = None
    if config.resume and (config.output_path / "results.json").exists():
        prev = json.loads((config.output_path / "results.json").read_text(encoding="utf-8"))
        for r in prev.get("results", []):
            cache_rows[(r["algo"], r["image"], r["distortion"], r["condition"])] = r
            if r["condition"] == "watermarked":
                cache_quality.setdefault((r["algo"], r["image"]), r.get("quality") or {})
                if r.get("embed_s") is not None:
                    cache_embed_s.setdefault((r["algo"], r["image"]), r["embed_s"])
        gallery = prev.get("gallery", [])
        pm = prev.get("meta", {})
        psnr_block_min, psnr_block_max = pm.get("psnr_block_min"), pm.get("psnr_block_max")
        _log(f"resume: reusing {len(cache_rows)} cached result rows")

    # One image's original under each distortion (only missing previews are saved).
    distortion_preview = _save_distortion_preview(gallery_dir, images, distortions)
    gallery_seen = {(g["algo"], g["image"]) for g in gallery}

    def _cached(algo_name, img, dist_name):
        return ((algo_name, img, dist_name, "watermarked") in cache_rows
                and (algo_name, img, dist_name, "clean") in cache_rows)

    result_rows: list[dict] = []
    for algo in algos:
        message = make_message(algo.payload_bits, config.seed, algo.name,
                               config.message_bits)

        algo_has_work = any(not _cached(algo.name, p.name, d.name)
                            or (algo.name, p.name) not in cache_quality
                            for p in images for d in distortions)
        if algo_has_work:
            # Warm up once so the first real embed/extract isn't charged for
            # one-time torch init (keeps recorded timings at steady state).
            try:
                warm = Image.open(images[0]).convert("RGB")
                algo.extract(algo.embed(warm, message))
            except Exception as exc:  # pragma: no cover - warmup must never be fatal
                _log(f"warmup skipped for {algo.name}: {exc}")

        for img_path in images:
            img, original = img_path.name, None
            qkey = (algo.name, img)
            to_compute = [d for d in distortions if not _cached(algo.name, img, d.name)]
            quality = cache_quality.get(qkey)
            watermarked = None
            embed_s = cache_embed_s.get(qkey)

            if to_compute or quality is None:
                original = Image.open(img_path).convert("RGB")
                wm_file = gallery_dir / f"{img_path.stem}__{algo.name}.png"
                if config.resume and wm_file.exists():
                    watermarked = Image.open(wm_file).convert("RGB")
                else:
                    t0 = time.perf_counter()
                    watermarked = algo.embed(original, message)
                    embed_s = round(time.perf_counter() - t0, 4)
                if quality is None:
                    quality = measure_quality(original, watermarked)
                    bmin, bmax = _block_psnr_minmax(original, watermarked)
                    if bmin is not None:
                        psnr_block_min = bmin if psnr_block_min is None else min(psnr_block_min, bmin)
                        psnr_block_max = bmax if psnr_block_max is None else max(psnr_block_max, bmax)
                if (algo.name, img) not in gallery_seen and len(gallery) < config.save_samples:
                    gallery.append(_save_gallery_entry(
                        gallery_dir, algo.name, img_path.stem, img, original, watermarked, quality))
                    gallery_seen.add((algo.name, img))

            for dist in distortions:
                wkey = (algo.name, img, dist.name, "watermarked")
                ckey = (algo.name, img, dist.name, "clean")
                if wkey in cache_rows and ckey in cache_rows:
                    result_rows.append(cache_rows[wkey])
                    result_rows.append(cache_rows[ckey])
                    continue
                group = getattr(dist, "group", "none")
                # --- watermarked (treatment) ---
                wm_d = dist.apply(watermarked)
                t1 = time.perf_counter()
                res = algo.extract(wm_d)
                extract_s = time.perf_counter() - t1
                result_rows.append(ResultRow(
                    algo=algo.name, image=img, distortion=dist.name, group=group,
                    params=dict(dist.params), condition="watermarked", quality=quality,
                    bit_acc=bit_accuracy(message, res.message),
                    present=res.present, confidence=res.confidence, is_false_positive=False,
                    embed_s=embed_s, extract_s=round(extract_s, 4),
                ).to_dict())
                # --- clean (false-positive control) ---
                clean_d = dist.apply(original)
                t2 = time.perf_counter()
                res_c = algo.extract(clean_d)
                extract_c_s = time.perf_counter() - t2
                result_rows.append(ResultRow(
                    algo=algo.name, image=img, distortion=dist.name, group=group,
                    params=dict(dist.params), condition="clean", quality={},
                    bit_acc=bit_accuracy(message, res_c.message),
                    present=res_c.present, confidence=res_c.confidence,
                    extract_s=round(extract_c_s, 4),
                    is_false_positive=false_positive(
                        res_c, registered_message=message,
                        threshold=config.success_threshold),
                ).to_dict())

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tools": available_metrics(),
            "success_threshold": config.success_threshold,
            "seed": config.seed,
            "psnr_block_min": psnr_block_min,
            "psnr_block_max": psnr_block_max,
        },
        "algorithms": [a.name for a in algos],
        "algorithm_settings": {a.name: dict(getattr(a, "settings", {})) for a in algos},
        "distortions": [d.name for d in distortions],
        "results": result_rows,
        "gallery": gallery,
        "distortion_preview": distortion_preview,
    }
    return payload


def _block_psnr_minmax(original: Image.Image, watermarked: Image.Image,
                       block: int = 32) -> tuple[float | None, float | None]:
    """Min and max per-32x32-block PSNR between source and watermarked (matches the
    report's localized-PSNR heatmap). Identical blocks (infinite PSNR) are skipped.
    Returns (None, None) if every block is identical."""
    a = np.asarray(original.convert("RGB"), dtype=np.float64)
    b = np.asarray(watermarked.convert("RGB"), dtype=np.float64)
    se = (a - b) ** 2
    h, w = se.shape[:2]
    lo = hi = None
    for by in range(0, h, block):
        for bx in range(0, w, block):
            mse = float(se[by:by + block, bx:bx + block, :].mean())
            if mse <= 0:
                continue
            psnr = 10.0 * np.log10(255.0 * 255.0 / mse)
            lo = psnr if lo is None else min(lo, psnr)
            hi = psnr if hi is None else max(hi, psnr)
    return lo, hi


def _save_distortion_preview(gallery_dir: Path, images: list[Path],
                             distortions: list) -> dict:
    """Save every image's *original* under each distortion (view-only preview),
    so the preview panel can follow whichever image is open in the viewer."""
    gallery_dir.mkdir(parents=True, exist_ok=True)
    by_image: dict[str, list] = {}
    for img_path in images:
        src = Image.open(img_path).convert("RGB")
        entries = []
        for dist in distortions:
            rel = f"gallery/preview__{img_path.stem}__{dist.name}.png"
            out = gallery_dir.parent / rel
            if not out.exists():
                (src if dist.name == "none" else dist.apply(src)).convert("RGB").save(out)
            entries.append({"name": dist.name, "group": getattr(dist, "group", "none"),
                            "path": rel})
        by_image[img_path.name] = entries
    return {"by_image": by_image}


def _save_gallery_entry(gallery_dir: Path, algo: str, stem: str, image: str,
                        original: Image.Image, watermarked: Image.Image,
                        quality: dict) -> dict:
    """Save the full-resolution source (once per image) and watermarked image for
    the web viewer. Filenames are keyed by image stem and algorithm so they never
    collide across per-algorithm runs."""
    gallery_dir.mkdir(parents=True, exist_ok=True)
    source_rel = f"gallery/{stem}__source.png"
    wm_rel = f"gallery/{stem}__{algo}.png"
    source_path = gallery_dir.parent / source_rel
    if not source_path.exists():
        original.convert("RGB").save(source_path)
    watermarked.convert("RGB").save(gallery_dir.parent / wm_rel)
    return {"image": image, "algo": algo, "source": source_rel,
            "watermarked": wm_rel, "quality": quality}


def write_outputs(payload: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = payload["results"]
    fieldnames = ["algo", "image", "distortion", "condition", "bit_acc",
                  "present", "confidence", "is_false_positive",
                  "embed_s", "extract_s",
                  "vmaf", "ssimulacra2", "xpsnr", "psnr_hvs_m"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            flat = dict(r)
            flat.update(r.get("quality") or {})
            writer.writerow(flat)
    return json_path, csv_path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        _log("usage: python -m wmbench.runner <config.yaml>")
        return 2
    config = load_config(argv[0])
    payload = run(config)
    json_path, csv_path = write_outputs(payload, config.output_path)
    _log(f"wrote {json_path} and {csv_path} ({len(payload['results'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
