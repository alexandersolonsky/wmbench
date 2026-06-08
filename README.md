# wmbench — image watermarking benchmark

A CPU-only framework for benchmarking image watermarking algorithms. For each
**algorithm × image × distortion** it embeds a payload, distorts the watermarked
image, and measures:

- **Extraction rate** — the fraction of images whose recovered payload **exactly
  matches the embedded ID**. A single wrong bit counts as a miss (this is *not*
  per-bit accuracy). Reported with a per-cell **Wilson 95% confidence interval**.
  Raw per-bit accuracy, detector confidence, and embed/extract timing are recorded too.
- **Quality degradation** of the watermarked image vs. the original (perceptual,
  pro-grade): **VMAF**, **SSIMULACRA2**, **xPSNR**, **PSNR-HVS-M**.
- **False positives** — how often a detector flags a *non-watermarked* image.

Results are written to `results/results.json` and rendered as a **self-contained
static site** (`site/`): a quality-vs-extraction scatter, cascading
group/distortion filters, per-algorithm tables, and an interactive image viewer
(before/after slider, amplified diff, localized 32×32 PSNR heatmap, fullscreen).

## Algorithms

Real adapters only (CPU-only, lazily imported). Each backend can register several
**variants** that vary one knob; variant names encode the setting so they read
consistently on the plot:

| Family | Backend | Variants (registered name → setting) |
|--------|---------|--------------------------------------|
| **TrustMark** | [Adobe TrustMark](https://github.com/adobe/trustmark) | `trustmark` (Q model); P-model **error-correction sweep** `trustmark_P_40 / _61 / _68 / _75 / _100` (BCH schema → usable data bits) |
| **PixelSeal** | [Meta PixelSeal](https://github.com/facebookresearch/videoseal) (`videoseal`) | `pixelseal_sw0.2` (default), `pixelseal_sw0.4` (2× blend strength) — `scaling_w` |
| **Watermark Anything** | [Meta WAM](https://github.com/facebookresearch/watermark-anything) | `wam_sw2.0` (default), `wam_sw1.0` (weaker, higher quality) — `scaling_w` |

TrustMark also ships `P`/`B`/`C` model types: `B` is most robust, `P` highest
visual quality, `C` a compact ResNet-18 decoder. Adding an algorithm or variant
is a drop-in: implement the `Watermarker` protocol and `@register_algorithm("name")`
(see `src/wmbench/algorithms/`).

## Distortions

**38 distortions across 7 groups** (plus the `none` control), all applied with
**ffmpeg** except the inpainting attack (OpenCV):

| Group | Distortions |
|-------|-------------|
| `compression` | `jpeg_q80/50/20`, `avc_crf15/22/28` |
| `resize/crop` | `resize_80/50/20` (downscale), `crop_80/50/20` (centred) |
| `geometrical` | `rotate_5/10/15`, `shear_x_10`, `shear_y_10`, `affine` |
| `noise` | `gauss_noise_10/20/40`, `uniform_noise_10/20/40` |
| `blur/filter` | `gblur_1/2/4`, `sharpen_1/2/3` |
| `color/tone` | `brightness_down/up`, `contrast_high`, `grayscale`, `hue_180`, `gamma_0.5` |
| `inpaint` | `telea_grid` — remove the inner 6×6 of every 8×8 block, refill with the classical Telea inpainter |

Aggregation weights every distortion equally so no group dominates. Distortions
register like algorithms (`@register_distortion`), so adding one is a drop-in.

## Quality-metric tools (external, CPU)

These must be on `PATH` of the machine that runs the benchmark; any that are
missing are reported as `null`:

- `ffmpeg` built with **libvmaf** (VMAF and the `psnr_hvs` feature)
- `ffmpeg` with the **xpsnr** filter
- **ssimulacra2** (`ssimulacra2` or `ssimulacra2_rs`)

```bash
python -c "from wmbench.metrics.tools import available_metrics; print(available_metrics())"
```

## Install

```bash
pip install -e .            # core (Pillow, numpy, PyYAML, Jinja2)
pip install -e ".[dev]"     # + pytest
```

Backends install into their own envs (below). The `telea_grid` distortion needs
OpenCV in whichever env runs it — **pin `opencv-python-headless>=4.10`**: 4.9.0
has a ~500× slower `cv2.inpaint` (Telea) that makes runs crawl.

### WAM weights

```bash
git clone https://github.com/facebookresearch/watermark-anything
export WAM_REPO=$PWD/watermark-anything
mkdir -p checkpoints
# Reliable mirror (fbaipublicfiles throttles/resets); the repo ships params.json:
python -c "from huggingface_hub import hf_hub_download; import shutil; \
  shutil.copy(hf_hub_download('facebook/watermark-anything','checkpoint.pth'), 'checkpoints/checkpoint.pth')"
cp watermark-anything/checkpoints/params.json checkpoints/
export WAM_CHECKPOINT_DIR=$PWD/checkpoints
```

WAM resolves its `configs/*.yaml` relative to the cwd, so the adapter `chdir`s
into `WAM_REPO` during load; pass absolute `input_dir`/`output_dir`.

## Running — one isolated env per algorithm

The backends have **mutually incompatible dependencies**, so each gets its own
virtualenv. `scripts/run_all.sh` runs each in its env, then merges the per-algorithm
`results.json` files into one report:

```bash
bash scripts/run_all.sh        # -> results/results.json + site/index.html
```

| env | python | backend | notes |
|-----|--------|---------|-------|
| `.venv` | 3.14 | — | dev / tests / merge / report only |
| `.venv-tm` | 3.12 | TrustMark | `pip install trustmark`; needs `numpy<2`, so `opencv-python-headless==4.10.x` (not 4.9) |
| `.venv-ps` | 3.12 | PixelSeal | `videoseal` **from git** (PyPI lacks the `pixelseal` card); copy its `configs/` into the installed package; `decord` stubbed on `PYTHONPATH` for arm64 (`.stubs/decord.py`) |
| `.venv-wam` | 3.12 | Watermark Anything | `pip install -r <repo>/requirements.txt`; set `WAM_REPO` + `WAM_CHECKPOINT_DIR` |

Each `configs/<algo>.yaml` sets `input_dir`, the variant list, and writes to
`./.runs/<algo>`. **Resume** is on by default (`resume: true`): cached cells are
reused and watermarked images loaded from the gallery, so adding a variant or
distortion only computes the new cells (delete `.runs/` for a clean run).

### Backend notes (learned the hard way)
- **WAM** expects **ImageNet-normalized** input and returns a normalized image —
  the adapter normalizes in and un-normalizes out. Feeding raw `[0,1]` silently
  degrades extraction at high resolution.
- **TrustMark**'s default `WM_STRENGTH=1.0` can be too weak for some hi-res
  photos; recovers exactly at `≥2.0`. Override with `WMBENCH_TRUSTMARK_STRENGTH`.
  Its `P` model centre-crops non-square inputs to a square — note for wide content.
- **OpenCV 4.9.0** has a pathologically slow Telea `inpaint`; pin `>=4.10`.
- Timing is measured at **steady state** — one warmup embed+extract per algorithm.

## Run a single algorithm

```bash
python -m wmbench.runner configs/demo.yaml                     # -> .runs/.../results.json
python -m wmbench.merge results .runs/tm .runs/ps .runs/wam    # merge per-env runs
python -m wmbench.report.generate results/results.json site    # -> site/index.html
open site/index.html
```

`input_dir` is a folder of your own images (nothing is bundled). A redundancy
pass (result-vector clustering) can trim near-duplicate frames to a lean, diverse set.

## Deploy

The `site/` folder is **fully self-contained** (HTML + vendored Chart.js +
`results.json` + image gallery), so hosting = serve that folder. The gallery is
large (~GBs) and **not committed**, so deploy the *local* `site/` to a host
without a tight size cap. Build it first with `scripts/run_all.sh`, then:

```bash
scripts/deploy.sh cloudflare        # Cloudflare Workers static assets (wrangler.toml -> ./site)
```

Alternatives: `scripts/deploy.sh netlify`, or a VPS via
`WMBENCH_VPS=user@host:/var/www/wmbench scripts/deploy.sh vps`. (GitHub Pages is
not used — its ~1 GB cap can't serve the gallery, and it can't host LFS objects.)

## Layout

```
src/wmbench/
  core/         interfaces (Watermarker, Distortion, ExtractResult, ResultRow) + registry
  algorithms/   trustmark / pixelseal / wam adapters (+ variants)
  distortions/  builtin.py — 7 groups of distortions + the telea_grid attack
  metrics/      quality (VMAF/SSIMULACRA2/xPSNR/PSNR-HVS-M) + extraction (exact-ID, Wilson, FP)
  runner.py     orchestrates the matrix (resume-aware) -> results.json / .csv / gallery
  merge.py      combine per-algorithm env runs into results/
  report/       static-site generator (Jinja2 + vendored Chart.js + image viewer)
```

## Tests

```bash
pytest -q
```

Torch adapters and metric tools are optional: their live tests self-skip when
absent, so the suite is green on a bare box and exercises the real paths where
the backends exist.
