#!/usr/bin/env bash
set -euo pipefail

# Mixed workload: light baseline with both short and rare long requests.
#
# Design rationale (PROPOSAL.md §6, Condition 2):
#   - Small stream matches the no_straggler baseline exactly so ITL comparisons
#     are apples-to-apples across conditions.
#   - Long stream is infrequent enough not to cause straggler congestion, but
#     present so the proxy sees heterogeneous traffic.
#   - num-prompts for BOTH streams are explicit so the experiment terminates in
#     finite time.  Without an explicit --num-prompts, the vllm bench serve
#     default (1000 prompts at RPS=0.02) would take ~50,000 s to complete.
#
# Duration estimate:
#   small: 2000 / 15 ≈ 133 s
#   long:     4 / 0.02 = 200 s  (concurrent; bounds total runtime)
#   → ~200 s total

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SMALL_RPS=15
SMALL_IN=128
SMALL_OUT=64
SMALL_NUM_PROMPTS=2000

LONG_RPS=0.02
LONG_IN=128
LONG_OUT=512
LONG_NUM_PROMPTS=4   # 4 / 0.02 RPS = 200 s, runs concurrently with small stream

"${ROOT_DIR}/mixed_vllm_load.sh" \
  --host    "${VLLM_HOST:-localhost}" \
  --port    "${VLLM_PORT:-10099}" \
  --model   "${VLLM_MODEL:-meta-llama/Llama-2-13b-hf}" \
  \
  --rps-small "${SMALL_RPS}" \
  --in-small  "${SMALL_IN}" \
  --out-small "${SMALL_OUT}" \
  \
  --rps-long  "${LONG_RPS}" \
  --in-long   "${LONG_IN}" \
  --out-long  "${LONG_OUT}" \
  --ignore-eos \
  \
  --extra-small "--num-prompts ${SMALL_NUM_PROMPTS} --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 90,95,99" \
  --extra-long  "--num-prompts ${LONG_NUM_PROMPTS}  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 90,95,99"
