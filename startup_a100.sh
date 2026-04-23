#!/bin/bash

# =============================================================================
# Startup Script — vLLM Disaggregated Serving (4× A100 80 GB)
# =============================================================================
# GPU layout (same as startup.sh):
#   Fast Lane (GPUs 0+1, FP16):  all incoming requests
#     GPU 0: Prefill  port 20098
#     GPU 1: Decode   port 20099
#   Slow Lane (GPUs 2+3, BF16):  migrated stragglers only
#     GPU 2: Prefill  port 20096
#     GPU 3: Decode   port 20097
#   Proxy (port 10099): sole client entry point
#
# A100-specific tuning vs H100:
#   - gpu_memory_utilization slightly lower (leave headroom for NCCL buffers)
#   - nccl_num_channels=16 for SXM NVLink; reduce to 8 for PCIe A100
#   - BF16 supported on Ampere — slow lane keeps BF16
#   - kv_buffer_size same; A100 80GB has ample room
# =============================================================================

SCRATCH_ROOT=${SCRATCH_ROOT:-$HOME/scratch}
CONDA_ENV=${CONDA_ENV:-$SCRATCH_ROOT/vllm_env}
export PATH="$CONDA_ENV/bin:$PATH"
PYTHON_BIN="$CONDA_ENV/bin/python3"
VLLM_BIN="$CONDA_ENV/bin/vllm"

MODEL=${MODEL:-meta-llama/Llama-2-13b-hf}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-30099}
SLOW_PROXY_PORT=${SLOW_PROXY_PORT:-30097}
PROXY_HTTP_PORT=${PROXY_HTTP_PORT:-10099}

HF_CACHE_ROOT=${HF_CACHE_ROOT:-$SCRATCH_ROOT/huggingface_cache}
mkdir -p "$HF_CACHE_ROOT"
export HF_HOME="$HF_CACHE_ROOT"
export HF_HUB_DISABLE_XET=1
echo "HuggingFace cache: $HF_CACHE_ROOT"

# LOG_DIR: directory for server logs. Set by sbatch for per-job isolation.
LOG_DIR="${LOG_DIR:-$(pwd)}"
mkdir -p "$LOG_DIR"

# For PCIe A100 (no NVLink): uncomment these two lines
# export NCCL_P2P_LEVEL=5
# export NCCL_SHM_DISABLE=0

FAST_PREFILL_GPU=${FAST_PREFILL_GPU:-0}
FAST_DECODE_GPU=${FAST_DECODE_GPU:-1}
FAST_PREFILL_PORT=${FAST_PREFILL_PORT:-20098}
FAST_DECODE_PORT=${FAST_DECODE_PORT:-20099}
FAST_PREFILL_KV_PORT=${FAST_PREFILL_KV_PORT:-21098}
FAST_DECODE_KV_PORT=${FAST_DECODE_KV_PORT:-22099}

SLOW_PREFILL_GPU=${SLOW_PREFILL_GPU:-2}
SLOW_DECODE_GPU=${SLOW_DECODE_GPU:-3}
SLOW_PREFILL_PORT=${SLOW_PREFILL_PORT:-20096}
SLOW_DECODE_PORT=${SLOW_DECODE_PORT:-20097}
SLOW_PREFILL_KV_PORT=${SLOW_PREFILL_KV_PORT:-21096}
SLOW_DECODE_KV_PORT=${SLOW_DECODE_KV_PORT:-22097}

echo "============================================================"
echo "Starting vLLM Disaggregated Serving (4× A100 80GB)"
echo "============================================================"
echo "  Model: $MODEL"
echo "  Fast Lane (FP16): prefill GPU $FAST_PREFILL_GPU :$FAST_PREFILL_PORT  decode GPU $FAST_DECODE_GPU :$FAST_DECODE_PORT"
echo "  Slow Lane (BF16): prefill GPU $SLOW_PREFILL_GPU :$SLOW_PREFILL_PORT  decode GPU $SLOW_DECODE_GPU :$SLOW_DECODE_PORT"
echo "  Proxy: ZMQ fast=$PROXY_PORT slow=$SLOW_PROXY_PORT  HTTP=$PROXY_HTTP_PORT"
echo "  Timeout: ${TIMEOUT_SECONDS}s  |  Logs: $LOG_DIR"
echo ""

PIDS=()

cleanup_and_exit() {
    local msg="$1"
    echo "  ✗ $msg"
    echo "  Killing all background processes..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    exit 1
}

# ── Proxy ─────────────────────────────────────────────────────────────────────
echo "Starting migration-aware proxy..."
PROXY_PORT=$PROXY_PORT SLOW_PROXY_PORT=$SLOW_PROXY_PORT \
PROXY_HTTP_PORT=$PROXY_HTTP_PORT MODEL=$MODEL \
    $PYTHON_BIN disagg_proxy_migration.py > "$LOG_DIR/proxy.log" 2>&1 &
PROXY_PID=$!; PIDS+=($PROXY_PID)
echo "  ✓ Proxy started (PID: $PROXY_PID)"
sleep 3

# ── Fast-Lane Prefill (GPU 0, FP16) ───────────────────────────────────────────
echo "Starting fast-lane prefill on GPU $FAST_PREFILL_GPU..."
CUDA_VISIBLE_DEVICES=$FAST_PREFILL_GPU $VLLM_BIN serve $MODEL \
    --enforce-eager \
    --host 0.0.0.0 \
    --port $FAST_PREFILL_PORT \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --dtype float16 \
    --max-num-seqs 64 \
    --max-model-len 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.80 \
    --disable-sliding-window \
    --no-enable-chunked-prefill \
    --kv-cache-dtype auto \
    --kv-transfer-config "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e9\",\"kv_port\":\"$FAST_PREFILL_KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$FAST_PREFILL_PORT\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" \
    > "$LOG_DIR/fast_prefill.log" 2>&1 &
FAST_PREFILL_PID=$!; PIDS+=($FAST_PREFILL_PID)
echo "  ✓ Fast-lane prefill started (PID: $FAST_PREFILL_PID)"

# ── Fast-Lane Decode (GPU 1, FP16) ────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Starting fast-lane decode on GPU $FAST_DECODE_GPU..."
CUDA_VISIBLE_DEVICES=$FAST_DECODE_GPU $VLLM_BIN serve $MODEL \
    --enforce-eager \
    --host 0.0.0.0 \
    --port $FAST_DECODE_PORT \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --dtype float16 \
    --max-num-seqs 32 \
    --max-model-len 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.70 \
    --disable-sliding-window \
    --no-enable-chunked-prefill \
    --kv-cache-dtype auto \
    --kv-transfer-config "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"4e9\",\"kv_port\":\"$FAST_DECODE_KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$FAST_DECODE_PORT\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" \
    > "$LOG_DIR/fast_decode.log" 2>&1 &
FAST_DECODE_PID=$!; PIDS+=($FAST_DECODE_PID)
echo "  ✓ Fast-lane decode started (PID: $FAST_DECODE_PID)"

# ── Slow-Lane Prefill (GPU 2, BF16) ───────────────────────────────────────────
echo "Starting slow-lane prefill on GPU $SLOW_PREFILL_GPU..."
CUDA_VISIBLE_DEVICES=$SLOW_PREFILL_GPU $VLLM_BIN serve $MODEL \
    --enforce-eager \
    --host 0.0.0.0 \
    --port $SLOW_PREFILL_PORT \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --dtype bfloat16 \
    --max-num-seqs 64 \
    --max-model-len 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.70 \
    --disable-sliding-window \
    --no-enable-chunked-prefill \
    --kv-cache-dtype auto \
    --kv-transfer-config "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e9\",\"kv_port\":\"$SLOW_PREFILL_KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$SLOW_PROXY_PORT\",\"http_port\":\"$SLOW_PREFILL_PORT\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" \
    > "$LOG_DIR/slow_prefill.log" 2>&1 &
SLOW_PREFILL_PID=$!; PIDS+=($SLOW_PREFILL_PID)
echo "  ✓ Slow-lane prefill started (PID: $SLOW_PREFILL_PID)"

# ── Slow-Lane Decode (GPU 3, BF16) ────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Starting slow-lane decode on GPU $SLOW_DECODE_GPU..."
CUDA_VISIBLE_DEVICES=$SLOW_DECODE_GPU $VLLM_BIN serve $MODEL \
    --enforce-eager \
    --host 0.0.0.0 \
    --port $SLOW_DECODE_PORT \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --dtype bfloat16 \
    --max-num-seqs 32 \
    --max-model-len 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.70 \
    --disable-sliding-window \
    --no-enable-chunked-prefill \
    --kv-cache-dtype auto \
    --kv-transfer-config "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"4e9\",\"kv_port\":\"$SLOW_DECODE_KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$SLOW_PROXY_PORT\",\"http_port\":\"$SLOW_DECODE_PORT\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" \
    > "$LOG_DIR/slow_decode.log" 2>&1 &
SLOW_DECODE_PID=$!; PIDS+=($SLOW_DECODE_PID)
echo "  ✓ Slow-lane decode started (PID: $SLOW_DECODE_PID)"

echo ""
echo "All servers launched. Logs: $LOG_DIR/{fast_prefill,fast_decode,slow_prefill,slow_decode,proxy}.log"
echo ""

# ── Health checks ──────────────────────────────────────────────────────────────
check_server_ready() {
    local port=$1 name=$2 max_wait=$3
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            if timeout 2 bash -c "echo > /dev/tcp/localhost/$port" 2>/dev/null; then
                return 0
            fi
        fi
        sleep 5; waited=$((waited + 5))
        [ $((waited % 20)) -eq 0 ] && echo "  Waiting for $name... (${waited}s elapsed)"
    done
    return 1
}

echo "Waiting for fast-lane prefill (port $FAST_PREFILL_PORT)..."
check_server_ready $FAST_PREFILL_PORT "fast-lane prefill" $TIMEOUT_SECONDS \
    && echo "  ✓ Fast-lane prefill ready" \
    || { tail -20 "$LOG_DIR/fast_prefill.log"; cleanup_and_exit "Fast-lane prefill failed. Check $LOG_DIR/fast_prefill.log"; }

echo "Waiting for fast-lane decode (port $FAST_DECODE_PORT)..."
check_server_ready $FAST_DECODE_PORT "fast-lane decode" $TIMEOUT_SECONDS \
    && echo "  ✓ Fast-lane decode ready" \
    || { tail -20 "$LOG_DIR/fast_decode.log"; cleanup_and_exit "Fast-lane decode failed. Check $LOG_DIR/fast_decode.log"; }

echo "Waiting for slow-lane prefill (port $SLOW_PREFILL_PORT)..."
check_server_ready $SLOW_PREFILL_PORT "slow-lane prefill" $TIMEOUT_SECONDS \
    && echo "  ✓ Slow-lane prefill ready" \
    || { tail -20 "$LOG_DIR/slow_prefill.log"; cleanup_and_exit "Slow-lane prefill failed. Check $LOG_DIR/slow_prefill.log"; }

echo "Waiting for slow-lane decode (port $SLOW_DECODE_PORT)..."
check_server_ready $SLOW_DECODE_PORT "slow-lane decode" $TIMEOUT_SECONDS \
    && echo "  ✓ Slow-lane decode ready" \
    || { tail -20 "$LOG_DIR/slow_decode.log"; cleanup_and_exit "Slow-lane decode failed. Check $LOG_DIR/slow_decode.log"; }

echo "Waiting for proxy (port $PROXY_HTTP_PORT)..."
check_server_ready $PROXY_HTTP_PORT "proxy" 30 \
    && echo "  ✓ Proxy ready" \
    || { tail -20 "$LOG_DIR/proxy.log"; cleanup_and_exit "Proxy failed. Check $LOG_DIR/proxy.log"; }

echo ""
echo "============================================================"
echo "✓ All servers are ready!"
echo "============================================================"
echo "  Proxy (entry point): http://localhost:$PROXY_HTTP_PORT"
echo "  Fast Lane: prefill http://localhost:$FAST_PREFILL_PORT  decode http://localhost:$FAST_DECODE_PORT"
echo "  Slow Lane: prefill http://localhost:$SLOW_PREFILL_PORT  decode http://localhost:$SLOW_DECODE_PORT"
echo ""
echo "Test:"
echo "  curl -X POST http://localhost:$PROXY_HTTP_PORT/v1/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"$MODEL\", \"prompt\": \"Hello\", \"max_tokens\": 20}'"
echo ""
echo "To stop all servers: ./cleanup.sh"
echo ""
