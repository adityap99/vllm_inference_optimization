#!/usr/bin/env bash
set -euo pipefail
# Generates prometheus.yml from prometheus.yml.template using envsubst.
# Usage:
#   export PREFILL_HOST=localhost PREFILL_PORT=20011
#   export DECODE_HOST=localhost DECODE_PORT=20012
#   ./generate_prometheus_config.sh
# Result: prometheus.yml written next to the template.

TEMPLATE="prometheus.yml.template"
OUT="prometheus.yml"

if [[ ! -f $TEMPLATE ]]; then
  echo "Template $TEMPLATE not found" >&2
  exit 1
fi

# Use only the variables we expect to avoid accidental leaking
export PREFILL_PORT=${PREFILL_PORT:-20011}
export DECODE_PORT=${DECODE_PORT:-20012}
export PROXY_HTTP_PORT=${PROXY_HTTP_PORT:-10005}

if ! command -v envsubst >/dev/null 2>&1; then
  echo "envsubst not found; install gettext package." >&2
  exit 2
fi

echo "[info] Generating $OUT from $TEMPLATE"    
envsubst < "$TEMPLATE" > "$OUT"
echo "[info] Wrote $OUT"

echo "[info] Preview:" && head -n 25 "$OUT" || true
