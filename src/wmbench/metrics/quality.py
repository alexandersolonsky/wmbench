"""Perceptual quality of a watermarked image vs. the original.

Writes both images to temporary PNGs and runs the external CPU tools. Returns a
dict with one entry per metric; any metric whose tool is unavailable (or which
fails) is ``None`` so the rest of the pipeline can carry on.
"""

from __future__ import annotations

import os
import tempfile

from PIL import Image

from wmbench.metrics import tools

QUALITY_METRICS = ("vmaf", "ssimulacra2", "xpsnr", "psnr_hvs_m")


def _save_png(image: Image.Image, path: str) -> None:
    image.convert("RGB").save(path, format="PNG")


def measure_quality(reference: Image.Image, watermarked: Image.Image) -> dict[str, float | None]:
    """Compute VMAF / SSIMULACRA2 / xPSNR / PSNR-HVS-M for (reference, watermarked)."""
    result: dict[str, float | None] = {k: None for k in QUALITY_METRICS}

    with tempfile.TemporaryDirectory(prefix="wmbench_q_") as d:
        ref_png = os.path.join(d, "ref.png")
        wm_png = os.path.join(d, "wm.png")
        _save_png(reference, ref_png)
        _save_png(watermarked, wm_png)

        vmaf, psnr_hvs = tools.measure_vmaf_and_psnrhvs(ref_png, wm_png)
        result["vmaf"] = vmaf
        result["psnr_hvs_m"] = psnr_hvs
        result["xpsnr"] = tools.measure_xpsnr(ref_png, wm_png)
        result["ssimulacra2"] = tools.measure_ssimulacra2(ref_png, wm_png)

    return result
