"""Locate and drive the external CPU quality-metric tools, and parse their output.

Four perceptual metrics, all computed by external binaries (no CUDA, no torch):

  * VMAF       — ffmpeg with libvmaf
  * PSNR-HVS-M — ffmpeg/libvmaf ``psnr_hvs`` feature
  * xPSNR      — ffmpeg ``xpsnr`` filter
  * SSIMULACRA2 — the ``ssimulacra2`` / ``ssimulacra2_rs`` binary

Parsing is split into pure functions (``parse_*``) so it can be unit-tested
against captured output without the tools installed. The ``measure_*`` functions
shell out and return a float, or ``None`` if the tool is unavailable or fails.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from functools import lru_cache


# --------------------------------------------------------------------------- #
# Tool discovery
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


@lru_cache(maxsize=1)
def _ffmpeg_filters() -> str:
    """Return the text of ``ffmpeg -filters`` (cached), or '' if unavailable."""
    ff = ffmpeg_path()
    if not ff:
        return ""
    try:
        out = subprocess.run(
            [ff, "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout + out.stderr
    except (OSError, subprocess.SubprocessError):
        return ""


def has_libvmaf() -> bool:
    return bool(re.search(r"\blibvmaf\b", _ffmpeg_filters()))


def has_xpsnr() -> bool:
    return bool(re.search(r"\bxpsnr\b", _ffmpeg_filters()))


@lru_cache(maxsize=1)
def ssimulacra2_path() -> str | None:
    for name in ("ssimulacra2", "ssimulacra2_rs"):
        p = shutil.which(name)
        if p:
            return p
    return None


def available_metrics() -> dict[str, bool]:
    """Which of the four metrics can actually be computed on this machine."""
    return {
        "vmaf": has_libvmaf(),
        "psnr_hvs_m": has_libvmaf(),
        "xpsnr": has_xpsnr(),
        "ssimulacra2": ssimulacra2_path() is not None,
    }


# --------------------------------------------------------------------------- #
# Pure parsers (unit-tested against captured output)
# --------------------------------------------------------------------------- #
def _pooled_or_frame_mean(data: dict, *keys: str) -> float | None:
    """Pull a metric mean from a libvmaf JSON log, trying several key spellings.

    Handles both the ``pooled_metrics`` shape (newer libvmaf) and the
    ``frames[].metrics`` shape (older), and several feature-key spellings.
    """
    pooled = data.get("pooled_metrics") or {}
    for k in keys:
        entry = pooled.get(k)
        if isinstance(entry, dict) and isinstance(entry.get("mean"), (int, float)):
            return float(entry["mean"])
    frames = data.get("frames") or []
    for k in keys:
        vals = [
            f["metrics"][k]
            for f in frames
            if isinstance(f.get("metrics"), dict)
            and isinstance(f["metrics"].get(k), (int, float))
        ]
        if vals:
            return float(sum(vals) / len(vals))
    return None


def parse_vmaf_json(text: str) -> float | None:
    return _pooled_or_frame_mean(json.loads(text), "vmaf")


def parse_psnr_hvs_json(text: str) -> float | None:
    # libvmaf reports a combined ``psnr_hvs`` plus per-channel ``psnr_hvs_y``.
    return _pooled_or_frame_mean(json.loads(text), "psnr_hvs", "psnr_hvs_y")


def parse_xpsnr_stderr(text: str) -> float | None:
    """Parse ffmpeg's xpsnr summary line, returning the luma (Y) xPSNR in dB.

    Example line:
      [Parsed_xpsnr_0 @ 0x..] XPSNR  y: 41.5183  u: 47.1751  v: 46.3409
    'inf' (identical inputs) maps to a large finite value.
    """
    m = re.search(r"XPSNR\s+y:\s*(inf|[0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if not m:
        return None
    val = m.group(1).lower()
    return float("inf") if val == "inf" else float(val)


def parse_ssimulacra2_stdout(text: str) -> float | None:
    """Parse a ssimulacra2 score. Both the libjxl tool (prints just a number)
    and ssimulacra2_rs (prints e.g. 'Score: 87.61') are supported: take the last
    float on the last non-empty line."""
    for line in reversed(text.strip().splitlines()):
        nums = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", line)
        if nums:
            return float(nums[-1])
    return None


# --------------------------------------------------------------------------- #
# Shell-out measurement (integration; returns None on any failure)
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def measure_vmaf_and_psnrhvs(ref_png: str, dist_png: str) -> tuple[float | None, float | None]:
    """One ffmpeg/libvmaf pass yields both VMAF and PSNR-HVS-M."""
    ff = ffmpeg_path()
    if not ff or not has_libvmaf():
        return None, None
    # Write the JSON log to stdout via /dev/stdout is unreliable across builds;
    # use a temp file path supplied by the caller-friendly wrapper instead.
    import tempfile, os

    log_fd, log_path = tempfile.mkstemp(suffix=".json")
    os.close(log_fd)
    try:
        lavfi = (
            f"[0:v]format=yuv420p,setsar=1[d];"
            f"[1:v]format=yuv420p,setsar=1[r];"
            f"[d][r]libvmaf=feature=name=psnr_hvs:log_path={log_path}:log_fmt=json"
        )
        cmd = [ff, "-hide_banner", "-nostats", "-i", dist_png, "-i", ref_png,
               "-lavfi", lavfi, "-f", "null", "-"]
        proc = _run(cmd)
        if proc is None:
            return None, None
        try:
            with open(log_path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            return None, None
        return parse_vmaf_json(text), parse_psnr_hvs_json(text)
    finally:
        try:
            os.remove(log_path)
        except OSError:
            pass


def measure_xpsnr(ref_png: str, dist_png: str) -> float | None:
    ff = ffmpeg_path()
    if not ff or not has_xpsnr():
        return None
    lavfi = (
        f"[0:v]format=yuv420p,setsar=1[d];"
        f"[1:v]format=yuv420p,setsar=1[r];"
        f"[d][r]xpsnr"
    )
    cmd = [ff, "-hide_banner", "-i", dist_png, "-i", ref_png,
           "-lavfi", lavfi, "-f", "null", "-"]
    proc = _run(cmd)
    if proc is None:
        return None
    # xpsnr writes its summary to stderr.
    return parse_xpsnr_stderr(proc.stderr + proc.stdout)


def measure_ssimulacra2(ref_png: str, dist_png: str) -> float | None:
    tool = ssimulacra2_path()
    if not tool:
        return None
    # ssimulacra2_rs needs an 'image' subcommand; the libjxl tool takes the two
    # paths directly. Try the subcommand form first, then the bare form.
    for cmd in ([tool, "image", ref_png, dist_png], [tool, ref_png, dist_png]):
        proc = _run(cmd)
        if proc is not None and proc.returncode == 0:
            score = parse_ssimulacra2_stdout(proc.stdout + proc.stderr)
            if score is not None:
                return score
    return None
