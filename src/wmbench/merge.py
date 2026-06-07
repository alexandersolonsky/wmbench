"""Merge several ``results.json`` files (and their sample images) into one.

Different algorithms may need different/incompatible Python environments (e.g.
TrustMark vs. PixelSeal vs. WAM). Run each in its own env writing to its own
output dir, then merge the runs into a single results set for the report. Sample
images are copied under run-unique names so they never collide.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def _copy_gallery(src_dir: Path, dst_dir: Path, rel: str) -> str:
    """Copy a gallery image, preserving its name (already unique by image+algo),
    and return its relative path. Identical source images dedupe naturally."""
    name = Path(rel).name
    src = src_dir / name
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / name)
    return f"gallery/{name}"


def merge_runs(input_dirs: list[str | Path], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_gallery = out_dir / "gallery"

    merged: dict = {"meta": {"tools": {}}, "algorithms": [], "distortions": [],
                    "results": [], "gallery": []}
    # image name -> combined gallery entry (source + per-algorithm watermarked)
    gallery_by_image: dict[str, dict] = {}

    for d in input_dirs:
        d = Path(d)
        payload = json.loads((d / "results.json").read_text(encoding="utf-8"))

        meta = payload.get("meta", {})
        # Union tool availability (a tool counts as available if any run had it).
        for k, v in (meta.get("tools") or {}).items():
            merged["meta"]["tools"][k] = merged["meta"]["tools"].get(k, False) or v
        # Common localized-PSNR colour scale spans every run's range.
        if meta.get("psnr_block_min") is not None:
            cur = merged["meta"].get("psnr_block_min")
            merged["meta"]["psnr_block_min"] = meta["psnr_block_min"] if cur is None \
                else min(cur, meta["psnr_block_min"])
        if meta.get("psnr_block_max") is not None:
            cur = merged["meta"].get("psnr_block_max")
            merged["meta"]["psnr_block_max"] = meta["psnr_block_max"] if cur is None \
                else max(cur, meta["psnr_block_max"])
        for k, v in meta.items():
            if k not in ("tools", "psnr_block_min", "psnr_block_max"):
                merged["meta"].setdefault(k, v)

        for a in payload.get("algorithms", []):
            if a not in merged["algorithms"]:
                merged["algorithms"].append(a)
        merged.setdefault("algorithm_settings", {}).update(
            payload.get("algorithm_settings", {}))
        for ds in payload.get("distortions", []):
            if ds not in merged["distortions"]:
                merged["distortions"].append(ds)

        merged["results"].extend(payload.get("results", []))

        src_gallery = d / "gallery"
        # Distortion preview (one image's original under each distortion): take the
        # first run that has it and copy its images.
        if not merged.get("distortion_preview") and payload.get("distortion_preview"):
            dp = payload["distortion_preview"]
            for entries in dp.get("by_image", {}).values():
                for e in entries:
                    _copy_gallery(src_gallery, out_gallery, e["path"])
            merged["distortion_preview"] = dp

        for g in payload.get("gallery", []):
            name = g["image"]
            entry = gallery_by_image.setdefault(
                name, {"image": name, "source": None, "watermarked": {}, "quality": {}})
            entry["source"] = _copy_gallery(src_gallery, out_gallery, g["source"])
            entry["watermarked"][g["algo"]] = _copy_gallery(
                src_gallery, out_gallery, g["watermarked"])
            entry["quality"][g["algo"]] = g.get("quality", {})

    merged["gallery"] = list(gallery_by_image.values())

    out_json = out_dir / "results.json"
    out_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return out_json


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print("usage: python -m wmbench.merge <out_dir> <run_dir> [<run_dir> ...]",
              file=sys.stderr)
        return 2
    out = merge_runs(argv[1:], argv[0])
    print(f"[wmbench] merged {len(argv) - 1} run(s) -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
