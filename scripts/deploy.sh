#!/usr/bin/env bash
# Deploy the self-contained static report (site/, gallery included) to a host
# WITHOUT GitHub Pages' ~1GB cap. Build it first with scripts/run_all.sh.
#
#   scripts/deploy.sh cloudflare     # Cloudflare Workers static assets (recommended)
#   scripts/deploy.sh netlify        # Netlify
#   WMBENCH_VPS=user@host:/var/www/wmbench scripts/deploy.sh vps
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
[ -f site/index.html ] || { echo "site/ not built — run scripts/run_all.sh first"; exit 1; }
echo "deploying site/ ($(du -sh site | cut -f1)) via ${1:-cloudflare}"

case "${1:-cloudflare}" in
  cloudflare) npx wrangler deploy ;;   # Workers static assets (wrangler.toml -> ./site)
  netlify)    npx netlify deploy --dir=site --prod ;;
  vps)        rsync -avz --delete site/ "${WMBENCH_VPS:?set WMBENCH_VPS=user@host:/path}" ;;
  *) echo "usage: deploy.sh [cloudflare|netlify|vps]"; exit 1 ;;
esac
