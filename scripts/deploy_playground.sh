#!/usr/bin/env bash
# Re-export the static playground viewer and deploy it to Vercel production.
# Live at https://duet-playground.vercel.app (and https://plg.duetlabs.co once DNS is in place).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/playground/static_export
.venv/bin/python scripts/playground_export_static.py --out "$OUT"
cd "$OUT"
vercel deploy --prod --yes
