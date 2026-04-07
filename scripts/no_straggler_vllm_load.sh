#!/usr/bin/env bash
set -euo pipefail

# Baseline workload: only short/medium requests.
# This configuration is the "reference" for apples-to-apples comparison.
# All other experiments MUST keep the small-stream parameters identical.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SMALL_RPS=15
SMALL_IN=128
SMALL_OUT=64
SMALL_NUM_PROMPTS=2000

"${ROOT_DIR}/mixed_vllm_load.sh" \
  --host "${VLLM_HOST:-localhost}" \
  --port "${VLLM_PORT:-10099}" \
  --model "${VLLM_MODEL:-meta-llama/Llama-2-13b-hf}" \
  \
  --rps-small "${SMALL_RPS}" \
  --in-small "${SMALL_IN}" \
  --out-small "${SMALL_OUT}" \
  \
  --rps-long 0.01 \
  --in-long 128 \
  --out-long 1024 \
  \
  --extra-small "--num-prompts ${SMALL_NUM_PROMPTS} --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 90,95,99" \
  --extra-long  "--num-prompts 1"
