#!/usr/bin/env bash
set -euo pipefail

# Heavy straggler (stress) workload:
# - Small stream is STILL IDENTICAL to baseline.
# - Increase long-stream rate and/or length to stress the decode GPU.
# - Keep num-prompts bounded to avoid OOM while clearly showing contention.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SMALL_RPS=15
SMALL_IN=128
SMALL_OUT=64
SMALL_NUM_PROMPTS=2000

LONG_RPS=0.05
LONG_IN=128
LONG_OUT=2048   # longer outputs -> more KV growth
LONG_NUM_PROMPTS=250

"${ROOT_DIR}/mixed_vllm_load.sh" \
  --host "${VLLM_HOST:-localhost}" \
  --port "${VLLM_PORT:-10099}" \
  --model "${VLLM_MODEL:-meta-llama/Llama-2-13b-hf}" \
  \
  --rps-small "${SMALL_RPS}" \
  --in-small "${SMALL_IN}" \
  --out-small "${SMALL_OUT}" \
  \
  --rps-long "${LONG_RPS}" \
  --in-long "${LONG_IN}" \
  --out-long "${LONG_OUT}" \
  \
  --extra-small "--num-prompts ${SMALL_NUM_PROMPTS} --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 90,95,99" \
  --extra-long  "--num-prompts ${LONG_NUM_PROMPTS} --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 90,95,99"
