"""Built-in distortions, organised into thematic groups.

Each distortion declares a ``group`` (thematic family). The report scores every
distortion equally. ``none`` is the control. Real groups are added one at a time:

  * none         — control (no distortion)
  * compression  — lossy codec round-trips, all via ffmpeg
                   (JPEG q80/50/20, AVC/H.264 CRF 15/22/28)
  * resize/crop  — resampling / framing, all via ffmpeg
                   (downscale to 80/50/20%; centered crop to 80/50/20%)
  * geometrical  — rotation & affine warps, all via ffmpeg (same output size,
                   black fill): rotate 5/10/15 deg; shear-X/Y 10 deg; a general
                   affine warp (combined scale + shear, centred)
  * noise        — additive noise via ffmpeg: Gaussian & uniform at alls 10/20/40
  * blur/filter  — ffmpeg gblur (sigma 1/2/4) and unsharp sharpen (amount 1/2/3)
  * color/tone   — ffmpeg eq/hue photometric edits (brightness, contrast,
                   saturation, hue, gamma), strong magnitudes
  * inpaint      — block-removal inpainting attack: remove the inner 6x6 of every
                   8x8 block and refill with the classical Telea inpainter (telea_grid)
  * c2pa         — content-provenance lifecycle transforms that strip a C2PA
                   manifest (social re-encode, transcode, screenshot, recapture,
                   re-share overlay) — the durable-watermark soft-binding test
"""

from __future__ import annotations

import io
import math
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from wmbench.core.registry import register_distortion
from wmbench.metrics.tools import ffmpeg_path


@register_distortion("none")
class NoDistortion:
    """Pass the image through unchanged (the no-distortion baseline)."""

    name = "none"
    group = "none"

    def __init__(self) -> None:
        self.params: dict = {}

    def apply(self, image: Image.Image) -> Image.Image:
        return image.copy()


# --------------------------------------------------------------------------- #
# Group: compression — lossy codec round-trips, all routed through ffmpeg
# --------------------------------------------------------------------------- #
def _ffmpeg(args: list[str]) -> None:
    ff = ffmpeg_path()
    if not ff:
        raise RuntimeError("ffmpeg is required for compression distortions")
    proc = subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[:300]}")


class Jpeg:
    """JPEG re-encode via ffmpeg's mjpeg encoder. ``quality`` is the 0-100 libjpeg
    scale mapped to ffmpeg's qscale (2-31, lower = better)."""

    group = "compression"

    def __init__(self, quality: int) -> None:
        self.name = f"jpeg_q{quality}"
        self.qscale = max(2, min(31, round((100 - quality) / 100 * 30) + 1))
        self.params = {"codec": "mjpeg", "quality": quality, "qscale": self.qscale}

    def apply(self, image: Image.Image) -> Image.Image:
        with tempfile.TemporaryDirectory(prefix="wmbench_d_") as d:
            src, out = Path(d) / "in.png", Path(d) / "out.jpg"
            image.convert("RGB").save(src)
            _ffmpeg(["-i", str(src), "-c:v", "mjpeg", "-q:v", str(self.qscale), str(out)])
            return Image.open(out).convert("RGB")


class Avc:
    """H.264 (libx264) single-intra-frame round-trip at a given CRF."""

    group = "compression"

    def __init__(self, crf: int) -> None:
        self.name = f"avc_crf{crf}"
        self.crf = crf
        self.params = {"codec": "h264", "crf": crf, "pix_fmt": "yuv420p", "preset": "superfast"}

    def apply(self, image: Image.Image) -> Image.Image:
        with tempfile.TemporaryDirectory(prefix="wmbench_d_") as d:
            src, mp4, out = Path(d) / "in.png", Path(d) / "v.mp4", Path(d) / "out.png"
            image.convert("RGB").save(src)
            # crop to even dimensions (4:2:0 requires it), encode one intra frame
            _ffmpeg(["-i", str(src), "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
                     "-c:v", "libx264", "-crf", str(self.crf), "-preset", "superfast",
                     "-pix_fmt", "yuv420p", "-frames:v", "1", str(mp4)])
            _ffmpeg(["-i", str(mp4), "-frames:v", "1", str(out)])
            return Image.open(out).convert("RGB")


for _q in (80, 50, 20):
    register_distortion(f"jpeg_q{_q}")(lambda q=_q: Jpeg(q))
for _crf in (15, 22, 28):
    register_distortion(f"avc_crf{_crf}")(lambda c=_crf: Avc(c))


# --------------------------------------------------------------------------- #
# Group: resize/crop — resampling / framing, all via ffmpeg filters
# --------------------------------------------------------------------------- #
def _ffmpeg_vf(image: Image.Image, vf: str) -> Image.Image:
    """Apply a single ffmpeg video filter and return the result as a PIL image."""
    with tempfile.TemporaryDirectory(prefix="wmbench_d_") as d:
        src, out = Path(d) / "in.png", Path(d) / "out.png"
        image.convert("RGB").save(src)
        _ffmpeg(["-i", str(src), "-vf", vf, str(out)])
        return Image.open(out).convert("RGB")


class Resize:
    """Downscale to ``percent`` % of each dimension (left at the smaller size)."""

    group = "resize/crop"

    def __init__(self, percent: int) -> None:
        self.name = f"resize_{percent}"
        self.percent = percent
        self.params = {"op": "resize", "percent": percent}

    def apply(self, image: Image.Image) -> Image.Image:
        r = self.percent / 100
        return _ffmpeg_vf(image, f"scale=trunc(iw*{r}):trunc(ih*{r})")


class CenterCrop:
    """Keep the central ``percent`` % of each dimension (left at the smaller size)."""

    group = "resize/crop"

    def __init__(self, percent: int) -> None:
        self.name = f"crop_{percent}"
        self.percent = percent
        self.params = {"op": "center_crop", "percent": percent}

    def apply(self, image: Image.Image) -> Image.Image:
        r = self.percent / 100
        return _ffmpeg_vf(image, f"crop=trunc(iw*{r}):trunc(ih*{r})")


for _p in (80, 50, 20):
    register_distortion(f"resize_{_p}")(lambda p=_p: Resize(p))
    register_distortion(f"crop_{_p}")(lambda p=_p: CenterCrop(p))


# --------------------------------------------------------------------------- #
# Group: geometrical — rotation & affine warps via ffmpeg (same size, black fill)
# --------------------------------------------------------------------------- #
class Rotate:
    """Rotate by ``degrees`` about the centre; output keeps the original size,
    corners that rotate out are filled black."""

    group = "geometrical"

    def __init__(self, degrees: int) -> None:
        self.name = f"rotate_{degrees}"
        self.degrees = degrees
        self.params = {"op": "rotate", "degrees": degrees}

    def apply(self, image: Image.Image) -> Image.Image:
        return _ffmpeg_vf(image, f"rotate={self.degrees}*PI/180:fillcolor=black")


class _Shear:
    """Affine shear via the perspective filter (parallelogram target). Same output
    size; the empty triangle is black."""

    group = "geometrical"
    axis = "x"

    def __init__(self, degrees: int) -> None:
        self.name = f"shear_{self.axis}_{degrees}"
        self.degrees = degrees
        self.params = {"op": f"shear_{self.axis}", "degrees": degrees}

    def _coords(self, w: int, h: int) -> str:
        raise NotImplementedError

    def apply(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        # perspective: input corners (TL,TR,BL,BR) map TO these destination coords
        return _ffmpeg_vf(image, f"perspective={self._coords(w, h)}:sense=destination")


class ShearX(_Shear):
    axis = "x"

    def _coords(self, w: int, h: int) -> str:
        d = round(math.tan(math.radians(self.degrees)) * h)  # bottom shifts right by d
        return f"0:0:{w}:0:{d}:{h}:{w + d}:{h}"


class ShearY(_Shear):
    axis = "y"

    def _coords(self, w: int, h: int) -> str:
        d = round(math.tan(math.radians(self.degrees)) * w)  # right column shifts down by d
        return f"0:0:{w}:{d}:0:{h}:{w}:{h + d}"


class Affine:
    """A general affine warp (linear 2x2 + implicit centring) applied via the
    perspective filter. The 2x2 combines anisotropic scale and shear in both
    axes; it is applied about the image centre so the result stays framed."""

    group = "geometrical"
    # [[a, b], [d, e]] — scale + shear in x and y
    MATRIX = ((0.95, 0.15), (0.10, 0.90))

    def __init__(self) -> None:
        self.name = "affine"
        self.params = {"op": "affine", "matrix": self.MATRIX}

    def apply(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        (a, b), (d, e) = self.MATRIX
        cx, cy = w / 2, h / 2

        def pt(x: float, y: float) -> tuple[int, int]:
            dx, dy = x - cx, y - cy
            return round(a * dx + b * dy + cx), round(d * dx + e * dy + cy)

        tl, tr, bl, br = pt(0, 0), pt(w, 0), pt(0, h), pt(w, h)
        coords = (f"{tl[0]}:{tl[1]}:{tr[0]}:{tr[1]}:"
                  f"{bl[0]}:{bl[1]}:{br[0]}:{br[1]}")
        return _ffmpeg_vf(image, f"perspective={coords}:sense=destination")


for _deg in (5, 10, 15):
    register_distortion(f"rotate_{_deg}")(lambda d=_deg: Rotate(d))
register_distortion("shear_x_10")(lambda: ShearX(10))
register_distortion("shear_y_10")(lambda: ShearY(10))
register_distortion("affine")(lambda: Affine())


# --------------------------------------------------------------------------- #
# Group: noise — additive noise via ffmpeg's noise filter
# --------------------------------------------------------------------------- #
class GaussNoise:
    group = "noise"

    def __init__(self, strength: int) -> None:
        self.name = f"gauss_noise_{strength}"
        self.strength = strength
        self.params = {"type": "gaussian", "alls": strength}

    def apply(self, image: Image.Image) -> Image.Image:
        return _ffmpeg_vf(image, f"noise=alls={self.strength}")


class UniformNoise:
    group = "noise"

    def __init__(self, strength: int) -> None:
        self.name = f"uniform_noise_{strength}"
        self.strength = strength
        self.params = {"type": "uniform", "alls": strength}

    def apply(self, image: Image.Image) -> Image.Image:
        return _ffmpeg_vf(image, f"noise=alls={self.strength}:allf=u")


for _s in (10, 20, 40):
    register_distortion(f"gauss_noise_{_s}")(lambda s=_s: GaussNoise(s))
    register_distortion(f"uniform_noise_{_s}")(lambda s=_s: UniformNoise(s))


# --------------------------------------------------------------------------- #
# Group: blur/filter — ffmpeg gaussian blur and unsharp sharpen
# --------------------------------------------------------------------------- #
class GaussBlur:
    group = "blur/filter"

    def __init__(self, sigma: int) -> None:
        self.name = f"gblur_{sigma}"
        self.sigma = sigma
        self.params = {"type": "gaussian_blur", "sigma": sigma}

    def apply(self, image: Image.Image) -> Image.Image:
        return _ffmpeg_vf(image, f"gblur=sigma={self.sigma}")


class Sharpen:
    group = "blur/filter"

    def __init__(self, amount: int) -> None:
        self.name = f"sharpen_{amount}"
        self.amount = amount
        self.params = {"type": "unsharp", "amount": amount}

    def apply(self, image: Image.Image) -> Image.Image:
        # unsharp=lx:ly:l_amount (5x5 luma kernel; chroma left untouched)
        return _ffmpeg_vf(image, f"unsharp=5:5:{self.amount}:5:5:0.0")


for _sig in (1, 2, 4):
    register_distortion(f"gblur_{_sig}")(lambda s=_sig: GaussBlur(s))
for _amt in (1, 2, 3):
    register_distortion(f"sharpen_{_amt}")(lambda a=_amt: Sharpen(a))


# --------------------------------------------------------------------------- #
# Group: color/tone — photometric edits via ffmpeg eq / hue (strong magnitudes)
# --------------------------------------------------------------------------- #
class ColorTone:
    group = "color/tone"

    def __init__(self, name: str, vf: str, params: dict) -> None:
        self.name = name
        self._vf = vf
        self.params = params

    def apply(self, image: Image.Image) -> Image.Image:
        return _ffmpeg_vf(image, self._vf)


_COLOR_TONE = [
    ("brightness_down", "eq=brightness=-0.4", {"op": "brightness", "value": -0.4}),
    ("brightness_up",   "eq=brightness=0.4",  {"op": "brightness", "value": 0.4}),
    ("contrast_high",   "eq=contrast=1.8",    {"op": "contrast", "value": 1.8}),
    ("grayscale",       "eq=saturation=0.0",  {"op": "saturation", "value": 0.0}),
    ("hue_180",         "hue=h=180",          {"op": "hue", "degrees": 180}),
    ("gamma_0.5",       "eq=gamma=0.5",       {"op": "gamma", "value": 0.5}),
]
for _name, _vf, _params in _COLOR_TONE:
    register_distortion(_name)(lambda n=_name, v=_vf, p=_params: ColorTone(n, v, p))


# --------------------------------------------------------------------------- #
# Group: inpaint — block-removal inpainting attack. telea_grid: remove the inner
# 6x6 of every 8x8 block (keep a 1px lattice) and refill the ~56% removed with
# the classical Telea inpainter. Telea propagates inward from the kept border and
# leaves the mask==0 pixels untouched, so (unlike a learned net) it adds no
# periodic per-block "stamp" — even this fine 8x8 lattice reconstructs without a
# visible grid.
# --------------------------------------------------------------------------- #
class TeleaGridInpaint:
    group = "inpaint"
    name = "telea_grid"
    block = 8
    keep = 1  # 1px border kept -> inner (block - 2*keep)=6x6 removed per block
    radius = 3

    def __init__(self) -> None:
        self._mask_cache: dict = {}
        rm = self.block - 2 * self.keep
        self.params = {"inpainter": "cv2-telea", "radius": self.radius,
                       "block": self.block, "removed": f"{rm}x{rm}",
                       "kept": f"{self.keep}px lattice"}

    def _mask(self, w: int, h: int):
        if (w, h) in self._mask_cache:
            return self._mask_cache[(w, h)]
        import numpy as np
        b, k = self.block, self.keep
        yy, xx = np.mgrid[0:h, 0:w]
        inner = (yy % b >= k) & (yy % b < b - k) & (xx % b >= k) & (xx % b < b - k)
        m = np.zeros((h, w), dtype="uint8")
        m[inner] = 255  # 255 = inpaint
        self._mask_cache[(w, h)] = m
        return m

    def apply(self, image: Image.Image) -> Image.Image:
        import cv2
        import numpy as np
        a = np.asarray(image.convert("RGB"))
        h, w = a.shape[:2]
        out = cv2.inpaint(a, self._mask(w, h), self.radius, cv2.INPAINT_TELEA)
        return Image.fromarray(out)


register_distortion("telea_grid")(lambda: TeleaGridInpaint())


# --------------------------------------------------------------------------- #
# Group: c2pa — content-provenance lifecycle transforms. C2PA stores provenance
# in a signed *manifest* (metadata); these simulate the real-world operations
# that strip that manifest while leaving the image usable — social re-encode,
# format transcode, screenshot, recapture (analog hole), re-share overlay —
# against which a durable watermark is the "soft binding" fallback. Done in PIL so
# the JPEG quality matches what platforms actually report (not ffmpeg's qscale).
# --------------------------------------------------------------------------- #
_CHROMA = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}


def _jpeg_pil(img: Image.Image, quality: int, subsampling: int = 2) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _fit_long_edge(img: Image.Image, n: int) -> Image.Image:
    """Downscale (only) so the long edge is at most ``n`` px, keeping aspect."""
    w, h = img.size
    if max(w, h) <= n:
        return img
    s = n / float(max(w, h))
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)


class C2paResave:
    """Platform re-upload: cap the long edge + JPEG re-encode (strips the C2PA
    manifest and recompresses) — the canonical 'does my mark survive a repost?'."""

    group = "c2pa"

    def __init__(self, name: str, long_edge: int, quality: int, subsampling: int = 2) -> None:
        self.name = name
        self._le, self._q, self._ss = long_edge, quality, subsampling
        self.params = {"long_edge_px": long_edge, "jpeg_quality": quality,
                       "chroma": _CHROMA[subsampling]}

    def apply(self, image: Image.Image) -> Image.Image:
        return _jpeg_pil(_fit_long_edge(image.convert("RGB"), self._le), self._q, self._ss)


class C2paWebp:
    """Format transcode to WebP (common platform/CDN conversion; drops metadata)."""

    group = "c2pa"
    name = "c2pa_webp"

    def __init__(self) -> None:
        self.params = {"format": "webp", "quality": 80}

    def apply(self, image: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, "WEBP", quality=80)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class C2paOverlay:
    """Re-share with a semi-transparent caption/logo bar, then JPEG (republishing)."""

    group = "c2pa"
    name = "c2pa_overlay"

    def __init__(self) -> None:
        self.params = {"overlay": "caption bar (alpha 0.55)", "then": "jpeg_q85"}

    def apply(self, image: Image.Image) -> Image.Image:
        from PIL import ImageDraw
        img = image.convert("RGBA")
        w, h = img.size
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        bar = max(28, h // 12)
        d.rectangle([0, h - bar, w, h], fill=(0, 0, 0, 140))
        d.text((max(8, w // 80), h - bar + bar // 4), "RE-SHARED", fill=(255, 255, 255, 235))
        out = Image.alpha_composite(img, ov).convert("RGB")
        return _jpeg_pil(out, 85, 1)


def _persp_coeffs(out_pts, in_pts):
    """PIL PERSPECTIVE coefficients sampling input ``in_pts[i]`` at output ``out_pts[i]``."""
    import numpy as np
    A, b = [], []
    for (ox, oy), (ix, iy) in zip(out_pts, in_pts):
        A.append([ox, oy, 1, 0, 0, 0, -ox * ix, -oy * ix]); b.append(ix)
        A.append([0, 0, 0, ox, oy, 1, -ox * iy, -oy * iy]); b.append(iy)
    return np.linalg.solve(np.asarray(A, float), np.asarray(b, float)).tolist()


class C2paRecapture:
    """Analog hole: photo-of-screen / print-and-scan — mild keystone perspective,
    optical blur, sensor noise, then JPEG. Metadata is always lost on recapture."""

    group = "c2pa"
    name = "c2pa_recapture"

    def __init__(self) -> None:
        self.params = {"ops": "keystone+blur+sensor-noise", "then": "jpeg_q70"}

    def apply(self, image: Image.Image) -> Image.Image:
        import numpy as np
        from PIL import ImageFilter
        img = image.convert("RGB")
        w, h = img.size
        dx, dy = w * 0.02, h * 0.015
        frame = [(0, 0), (w, 0), (w, h), (0, h)]
        dst = [(dx, dy * 1.5), (w - dx * 0.4, 0), (w, h - dy), (dx * 0.4, h - dy * 0.4)]
        img = img.transform((w, h), Image.PERSPECTIVE,
                            _persp_coeffs(dst, frame), Image.BILINEAR)
        img = img.filter(ImageFilter.GaussianBlur(0.8))
        a = np.asarray(img, np.int16)
        a = np.clip(a + np.random.RandomState(0).normal(0, 4.0, a.shape), 0, 255).astype("uint8")
        return _jpeg_pil(Image.fromarray(a, "RGB"), 70, 2)


register_distortion("c2pa_instagram")(lambda: C2paResave("c2pa_instagram", 1080, 80))
register_distortion("c2pa_facebook")(lambda: C2paResave("c2pa_facebook", 2048, 72))
register_distortion("c2pa_whatsapp")(lambda: C2paResave("c2pa_whatsapp", 1600, 60))
register_distortion("c2pa_screenshot")(lambda: C2paResave("c2pa_screenshot", 1280, 92, 0))
register_distortion("c2pa_webp")(lambda: C2paWebp())
register_distortion("c2pa_overlay")(lambda: C2paOverlay())
register_distortion("c2pa_recapture")(lambda: C2paRecapture())
