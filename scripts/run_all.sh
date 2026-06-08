#!/usr/bin/env bash
# Run each watermarking algorithm in its OWN isolated environment, then merge the
# per-algorithm results into results/ and build the static report.
#
# One isolated env per backend (their dependencies are mutually incompatible):
#   .venv      py3.14  framework / dev + tests only (no backends)
#   .venv-tm   py3.12  TrustMark + LaMa
#   .venv-ps   py3.12  PixelSeal (videoseal from git; decord stubbed on arm64) + LaMa
#   .venv-wam  py3.12  Watermark Anything (repo on WAM_REPO + checkpoint) + LaMa
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[run_all] TrustMark  (.venv-tm)"
.venv-tm/bin/python -m wmbench.runner configs/trustmark.yaml

echo "[run_all] PixelSeal  (.venv-ps)"
PYTHONPATH="$ROOT/.stubs" .venv-ps/bin/python -m wmbench.runner configs/pixelseal.yaml

echo "[run_all] WAM        (.venv-wam)"
WAM_REPO="$ROOT/vendor/watermark-anything" WAM_CHECKPOINT_DIR="$ROOT/checkpoints" \
  .venv-wam/bin/python -m wmbench.runner configs/wam.yaml

echo "[run_all] InvisMark  (.venv-im)"
INVISMARK_REPO="$ROOT/vendor/InvisMark" INVISMARK_CKPT="$ROOT/checkpoints/invismark/paper.ckpt" \
  .venv-im/bin/python -m wmbench.runner configs/invismark.yaml

echo "[run_all] merge + build report"
.venv/bin/python -m wmbench.merge results .runs/tm .runs/ps .runs/wam .runs/im
.venv/bin/python -m wmbench.report.generate results/results.json site
echo "[run_all] done -> site/index.html"
