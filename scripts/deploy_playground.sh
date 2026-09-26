#!/usr/bin/env bash
# Re-export the ALLOWLISTED static playground (scripts/playground_publish.json) and deploy it to Vercel production.
# Live at https://plg.duetlabs.co and https://duet-playground.vercel.app (Vercel project duet-playground).
#   scripts/deploy_playground.sh [exporter options, e.g. --allow-partial]
# The Vercel project link lives in $OUT/.vercel/project.json (the exporter carries it over). Without it,
# `vercel deploy --prod --yes` would silently create a NEW project in whatever scope the caller is logged into,
# so this script aborts instead. If any allowlisted episode fails to export, nothing is deployed and the reason is
# printed per episode; --allow-partial (only when given explicitly) deploys the rest. The first run over a site written
# by the previous exporter (no .duet-playground-export marker) needs --adopt-old-export once. The exporter needs the
# playground extra (pip install -e '.[playground]'); PYTHON=/path/to/python overrides the interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/playground/static_export
PY="${PYTHON:-.venv/bin/python}"
if [[ ! -f "$OUT/.vercel/project.json" ]]; then
  echo "abort: $OUT/.vercel/project.json is missing; link the EXISTING duet-playground project first (cd $OUT && vercel link)" >&2
  exit 1
fi
command -v vercel >/dev/null || { echo "abort: vercel CLI not found on PATH" >&2; exit 1; }
rc=0
"$PY" scripts/playground_export_static.py --out "$OUT" "$@" || rc=$?
if [[ $rc -eq 3 ]]; then
  echo "abort: nothing deployed; the live site is unchanged. The episodes marked FAILED above did not export (reason on each line)." >&2
  echo "       Fix them (e.g. put the missing source video back), or deploy only the episodes that exported:" >&2
  echo "         scripts/deploy_playground.sh --allow-partial    # the FAILED episodes then DISAPPEAR from the live site" >&2
  exit 1
elif [[ $rc -ne 0 ]]; then
  echo "abort: the exporter failed (exit $rc, see the error above; missing packages -> PYTHON=... or pip install -e '.[playground]'); nothing deployed" >&2
  exit 1
fi
if [[ ! -f "$OUT/.vercel/project.json" ]]; then
  echo "abort: the export lost $OUT/.vercel/project.json" >&2
  exit 1
fi
cd "$OUT"
vercel deploy --prod --yes
