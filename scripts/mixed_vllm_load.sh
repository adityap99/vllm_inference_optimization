#!/usr/bin/env bash
set -euo pipefail

# Mixed vLLM load generator (CLI-only, vLLM >= 0.10)
# Sends two concurrent loads to a running vLLM server:
#   1) "Small" steady load
#   2) "Long" rare stragglers
#
# Example:
#   scripts/mixed_vllm_load.sh \
#     --host localhost --port 8000 \
#     --model facebook/opt-125m \
#     --duration 60 \
#     --rps-small 40 --in-small 64 --out-small 96 \
#     --rps-long 0.05 --in-long 32 --out-long 512 \
#     --ignore-eos \
#     --extra-small "--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 90,95,99"

# -------- Defaults --------
HOST="${VLLM_HOST:-localhost}"
PORT="${VLLM_PORT:-10099}"
MODEL="${VLLM_MODEL:-meta-llama/Llama-2-13b-hf}"
TOKENIZER="${VLLM_TOKENIZER:-$MODEL}"

# DURATION=300

# Small stream
# RPS_SMALL=60
# IN_TOK_SMALL=128
# OUT_TOK_SMALL=128

# # Long stragglers
# # RPS_LONG=0.15
# RPS_LONG=0.05
# IN_TOK_LONG=128
# # OUT_TOK_LONG=3000
# OUT_TOK_LONG=1024

# IGNORE_EOS=True

RPS_SMALL=10          # Reduce to create more headroom for long requests  
IN_TOK_SMALL=128  
OUT_TOK_SMALL=64      # Shorter outputs to contrast with long requests  
  
RPS_LONG=0.02         # Increase slightly for more frequent long requests  
IN_TOK_LONG=128  
OUT_TOK_LONG=512     # Much longer to exaggerate the effect  
IGNORE_EOS=true       # Critical - ensures long requests don't terminate early

# Extras / logs
EXTRA_SMALL=""
EXTRA_LONG=""
LOG_DIR="logs"
TS="$(date +%Y%m%d_%H%M%S)"

print_usage() {
  cat <<EOF
Usage: $0 [options]

Target:
  --host HOST                 Default: ${HOST}
  --port PORT                 Default: ${PORT}
  --model NAME                Default: ${MODEL}
  --tokenizer NAME            Default: ${TOKENIZER} (defaults to model)

Small stream:
  --rps-small RPS             Default: ${RPS_SMALL}
  --in-small TOKENS           Default: ${IN_TOK_SMALL}
  --out-small TOKENS          Default: ${OUT_TOK_SMALL}

Long stragglers:
  --rps-long RPS              Default: ${RPS_LONG}
  --in-long TOKENS            Default: ${IN_TOK_LONG}
  --out-long TOKENS           Default: ${OUT_TOK_LONG}
  --ignore-eos                Do not stop at EOS (encourage long outputs)

Advanced:
  --extra-small "FLAGS"       Extra flags appended to small CLI
  --extra-long  "FLAGS"       Extra flags appended to long  CLI
  --log-dir DIR               Default: ${LOG_DIR}
  -h, --help

Example:
  $0 --host localhost --port 8000 --model facebook/opt-125m \\
     --duration 60 \\
     --rps-small 40 --in-small 64 --out-small 96 \\
     --rps-long 0.05 --in-long 32 --out-long 512 --ignore-eos
EOF
}

# -------- Args --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --tokenizer) TOKENIZER="$2"; shift 2;;
    --rps-small) RPS_SMALL="$2"; shift 2;;
    --in-small) IN_TOK_SMALL="$2"; shift 2;;
    --out-small) OUT_TOK_SMALL="$2"; shift 2;;
    --rps-long) RPS_LONG="$2"; shift 2;;
    --in-long) IN_TOK_LONG="$2"; shift 2;;
    --out-long) OUT_TOK_LONG="$2"; shift 2;;
    --ignore-eos) IGNORE_EOS=true; shift;;
    --extra-small) EXTRA_SMALL="$2"; shift 2;;
    --extra-long) EXTRA_LONG="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    -h|--help) print_usage; exit 0;;
    *) echo "Unknown arg: $1"; print_usage; exit 1;;
  esac
done

# -------- Preflight --------
if ! command -v vllm >/dev/null 2>&1; then
  echo "[error] 'vllm' CLI not found in PATH."
  exit 1
fi
if ! vllm bench serve --help >/dev/null 2>&1; then
  echo "[error] Your vLLM build does not support 'vllm bench serve'."
  exit 1
fi

mkdir -p "$LOG_DIR"
SMALL_LOG="${LOG_DIR}/small_${TS}.log"
LONG_LOG="${LOG_DIR}/long_${TS}.log"

echo "[info] Target: http://${HOST}:${PORT}"
echo "[info] Model:  ${MODEL} (tokenizer=${TOKENIZER})"
echo "[info] Small:  rps=${RPS_SMALL} in=${IN_TOK_SMALL} out=${OUT_TOK_SMALL}"
echo "[info] Long:   rps=${RPS_LONG} in=${IN_TOK_LONG} out=${OUT_TOK_LONG} ignore_eos=${IGNORE_EOS}"

declare -a PIDS=()
cleanup() {
  echo
  echo "[info] Stopping child processes..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
    fi
  done
  wait || true
}
trap cleanup INT TERM

# -------- Small load --------
  # --profile-dir ./profiles/small/ \

echo "[info] Starting small (cli) -> ${SMALL_LOG}"

CMD_SMALL=(
  vllm bench serve
  --host "$HOST"
  --port "$PORT"
  --backend vllm
  --model "$MODEL"
  --num-prompts 5000
  --tokenizer "$TOKENIZER"
  --ignore-eos
  --dataset-name "random"
  --request-rate "$RPS_SMALL"
  --profile
  --random-input-len "$IN_TOK_SMALL"
  --random-output-len "$OUT_TOK_SMALL"
  --save-result
)

if [[ -n "${EXTRA_SMALL}" ]]; then
  CMD_SMALL+=(${EXTRA_SMALL})
fi

set -x
"${CMD_SMALL[@]}" >"$SMALL_LOG" 2>&1 &
set +x
PIDS+=("$!")

# -------- Long load --------
echo "[info] Starting long (cli)  -> ${LONG_LOG}"
LONG_EXTRA=()
if [[ "$IGNORE_EOS" == "true" ]]; then
  LONG_EXTRA+=(--ignore-eos)
fi

  # --profile-dir ./profiles/long/ \


set -x
CMD_LONG=(
  vllm bench serve
  --host "$HOST"
  --port "$PORT"
  --backend vllm
  --model "$MODEL"
  --num-prompts 50
  --tokenizer "$TOKENIZER"
  --ignore-eos
  --dataset-name "random"
  --request-rate "$RPS_LONG"
  --profile
  --random-input-len "$IN_TOK_LONG"
  --random-output-len "$OUT_TOK_LONG"
  --save-result
)

CMD_LONG+=("${LONG_EXTRA[@]}")

if [[ -n "${EXTRA_LONG}" ]]; then
  CMD_LONG+=(${EXTRA_LONG})
fi

"${CMD_LONG[@]}" >"$LONG_LOG" 2>&1 &
set +x
PIDS+=("$!")

echo "[info] Both loads started."
echo "  small -> ${SMALL_LOG}"
echo "  long  -> ${LONG_LOG}"
echo "[info] Press Ctrl-C to stop."

# -------- Wait for both workloads to finish --------
exit_code=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    exit_code=$?
  fi
done

echo "[info] Both workloads finished."
exit "$exit_code"
