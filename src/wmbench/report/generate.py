"""Build the static results web page from a committed ``results.json``.

Aggregation (``aggregate``) is a pure function so it can be unit-tested; the site
builder renders a Jinja2 template, embeds the aggregates + raw rows as JSON for
Chart.js, and copies the vendored Chart.js next to the page (no CDN, works
offline and on GitHub Pages).
"""

from __future__ import annotations

import json
import shutil
import statistics
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from wmbench.metrics.extraction import wilson_interval
from wmbench.metrics.quality import QUALITY_METRICS

_HERE = Path(__file__).parent
_TEMPLATES = _HERE / "templates"
_STATIC = _HERE / "static"


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    return statistics.fmean(vals) if vals else None


def aggregate(payload: dict) -> dict:
    """Summarize raw result rows into per-(algorithm, distortion) statistics."""
    rows = payload.get("results", [])
    threshold = payload.get("meta", {}).get("success_threshold", 0.99)
    algorithms = payload.get("algorithms", [])
    distortions = payload.get("distortions", [])

    wm = [r for r in rows if r.get("condition") == "watermarked"]
    clean = [r for r in rows if r.get("condition") == "clean"]

    # Quality per algorithm (quality is identical across distortions for a given
    # image, so averaging over all watermarked rows yields the per-algo mean).
    quality_by_algo: dict[str, dict] = {}
    for algo in algorithms:
        q = {}
        for metric in QUALITY_METRICS:
            q[metric] = _mean([r.get("quality", {}).get(metric) for r in wm
                               if r["algo"] == algo])
        quality_by_algo[algo] = q

    extraction: list[dict] = []
    fpr: list[dict] = []
    # Extraction rate = fraction of images whose extracted watermark ID exactly
    # matches the embedded one (recovered == embedded, i.e. bit accuracy == 1.0).
    # It is a per-image hit/miss, NOT mean bit accuracy.
    def is_exact(r) -> bool:
        ba = r.get("bit_acc")
        return ba is not None and ba >= 1.0

    for algo in algorithms:
        for dist in distortions:
            cell = [r for r in wm if r["algo"] == algo and r["distortion"] == dist]
            n = len(cell)
            exact = sum(1 for r in cell if is_exact(r))
            low, high = wilson_interval(exact, n)
            extraction.append({
                "algo": algo, "distortion": dist, "n": n,
                "extraction_rate": (exact / n) if n else None,
                "mean_bit_acc": _mean([r.get("bit_acc") for r in cell]),
                "ci_low": low, "ci_high": high,
                "mean_confidence": _mean([r.get("confidence") for r in cell]),
                "mean_embed_s": _mean([r.get("embed_s") for r in cell]),
                "mean_extract_s": _mean([r.get("extract_s") for r in cell]),
            })

            cells_c = [r for r in clean if r["algo"] == algo
                       and r["distortion"] == dist]
            nc = len(cells_c)
            fps = sum(1 for r in cells_c if r.get("is_false_positive"))
            flow, fhigh = wilson_interval(fps, nc)
            fpr.append({
                "algo": algo, "distortion": dist, "n": nc,
                "fpr": (fps / nc) if nc else None,
                "ci_low": flow, "ci_high": fhigh,
            })

    # One point per algorithm for the quality-vs-extraction scatter: extraction
    # rate (X, exact-match fraction over that algorithm's images) and mean of each
    # quality metric (Y, switchable client-side).
    def algo_extraction_rate(algo: str) -> float | None:
        ar = [r for r in wm if r["algo"] == algo]
        return (sum(1 for r in ar if is_exact(r)) / len(ar)) if ar else None

    def algo_fpr(algo: str) -> float | None:
        cr = [r for r in clean if r["algo"] == algo]
        return (sum(1 for r in cr if r.get("is_false_positive")) / len(cr)) if cr else None

    # Thematic groups in the order their distortions first appear.
    group_of = {r["distortion"]: r.get("group", "none") for r in wm}
    groups: list[str] = []
    for d in distortions:
        g = group_of.get(d, "none")
        if g not in groups:
            groups.append(g)

    # Scopes for the extraction-rate axis selector: overall, per group, per distortion.
    scopes = [{"key": "all", "label": "all distortions"}]
    scopes += [{"key": f"group:{g}", "label": f"group · {g}"} for g in groups]
    scopes += [{"key": f"dist:{d}", "label": f"distortion · {d}"} for d in distortions]

    def _rate(subset) -> float | None:
        s = [r for r in subset if r.get("bit_acc") is not None]
        return (sum(1 for r in s if is_exact(r)) / len(s)) if s else None

    points = []
    for algo in algorithms:
        arows = [r for r in wm if r["algo"] == algo]
        rates = {"all": _rate(arows)}
        for g in groups:
            rates[f"group:{g}"] = _rate([r for r in arows if r.get("group") == g])
        for d in distortions:
            rates[f"dist:{d}"] = _rate([r for r in arows if r["distortion"] == d])
        points.append({
            "algo": algo,
            "extraction_rate": rates["all"],
            "rates": rates,
            "mean_confidence": _mean([r.get("confidence") for r in arows]),
            "fpr": algo_fpr(algo),
            "mean_embed_s": _mean([r.get("embed_s") for r in arows]),
            "mean_extract_s": _mean([r.get("extract_s") for r in arows]),
            "quality": quality_by_algo[algo],
        })

    return {
        "algorithms": algorithms,
        "algorithm_settings": payload.get("algorithm_settings", {}),
        "distortions": distortions,
        "groups": groups,
        "group_of": group_of,
        "scopes": scopes,
        "metrics": list(QUALITY_METRICS),
        "threshold": threshold,
        "meta": payload.get("meta", {}),
        "quality_by_algo": quality_by_algo,
        "extraction": extraction,
        "fpr": fpr,
        "points": points,
        # Per-image source + per-algorithm watermarked images for the viewer.
        "gallery": payload.get("gallery", []),
        "distortion_preview": payload.get("distortion_preview"),
        "n_results": len(rows),
    }


def build_site(results_json: str | Path, out_dir: str | Path) -> Path:
    results_json = Path(results_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(results_json.read_text(encoding="utf-8"))
    agg = aggregate(payload)

    # Copy vendored Chart.js and any sample images alongside the page.
    assets = out_dir / "assets"
    assets.mkdir(exist_ok=True)
    shutil.copy2(_STATIC / "chart.umd.min.js", assets / "chart.umd.min.js")

    src_gallery = results_json.parent / "gallery"
    if src_gallery.is_dir():
        dst_gallery = out_dir / "gallery"
        if dst_gallery.exists():
            shutil.rmtree(dst_gallery)
        shutil.copytree(src_gallery, dst_gallery)

    env = Environment(
        loader=FileSystemLoader(_TEMPLATES),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("index.html.j2")
    html = template.render(agg=agg, data_json=json.dumps(agg))
    index = out_dir / "index.html"
    index.write_text(html, encoding="utf-8")
    return index


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    results = argv[0] if argv else "results/results.json"
    out = argv[1] if len(argv) > 1 else "site"
    index = build_site(results, out)
    print(f"[wmbench] built {index}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
