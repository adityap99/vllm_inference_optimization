#!/bin/bash

# =============================================================================
# Startup Script — vLLM Disaggregated Serving (4× L40S + Llama-2-7B)
# =============================================================================
# Tuned for:  GPU: NVIDIA L40S 48 GB (PCIe, NO NVLink)
#             Model: meta-llama/Llama-2-7b-hf (14 GB weights in FP16)
#
# Memory budget per GPU (48 GB):
#   Weights (FP16):          14 GB
#   CUDA context + overhead:  2 GB
#   Remaining at util=0.80:  48×0.80=38.4 - 14 - 2 = 22.4 GB for KV cache
#   kv_buffer (decode side):  4 GB  →  net 18.4 GB for KV seqs
#   KV per token (7B):        32 layers × 2 × 32 heads × 128 dim × 2B = 524,288 B ≈ 0.5 MB
#   Max concurrent seqs @4096 tokens (decode): 18.4e9 / 524288 / 4096 ≈ 8.6 → cap at 16
#
# Prefill gets util=0.85 (no kv_buffer space reserved there):
#   Budget: 48×0.85=40.8 - 14 - 2 = 24.8 GB for KV → ~11.5 seqs — cap at 32
#
# PCIe P2P bandwidth:   ~32 GB/s (vs H100 NVLink 600 GB/s)
# KV transfer at T_MIN=200 tok: 200 × 0.5 MB = 100 MB → ~3 ms (negligible)
# Decode throughput @ batch=1: ~864 GB/s / 14 GB ≈ 62 tok/s
# =============================================================================

SCRATCH_ROOT=${SCRATCH_ROOT:-$HOME/scratch}
CONDA_ENV=${CONDA_ENV:-$SCRATCH_ROOT/sysml_research4}
export PATH="$CONDA_ENV/bin:$PATH"
PYTHON_BIN="$CONDA_ENV/bin/python3"
VLLM_BIN="$CONDA_ENV/bin/vllm"

# Configuration
MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-900}   # 7B loads faster than 13B; 900s is generous
PROXY_PORT=${PROXY_PORT:-30099}
SLOW_PROXY_PORT=${SLOW_PROXY_PORT:-30097}
PROXY_HTTP_PORT=${PROXY_HTTP_PORT:-10099}

# Migration thresholds — lowered for L40S decode speed
# R_SLOW=15: migrate if per-request decode rate drops below 15 tok/s
# (H100 uses 30; L40S+7B tops out at ~62 tok/s single-req, ~15 under load)
MIGRATION_R_SLOW=${MIGRATION_R_SLOW:-15}
MIGRATION_T_MIN=${MIGRATION_T_MIN:-200}

# HuggingFace cache
HF_CACHE_ROOT=${HF_CACHE_ROOT:-$SCRATCH_ROOT/huggingface_cache}
mkdir -p "$HF_CACHE_ROOT"
export HF_HOME="$HF_CACHE_ROOT"
export HF_HUB_DISABLE_XET=1

# --- GPU assignments (same physical slots as H100 run) ---
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

# --- NCCL tuning for PCIe-only topology ------------------------------------
# L40S has NO NVLink. NCCL will use PCIe P2P automatically.
# These env vars help NCCL pick the right transport and minimize overhead.
export NCCL_P2P_LEVEL=5            # allow P2P across NUMA domains (PCIe switches)
export NCCL_SHM_DISABLE=0          # keep shared memory for intra-socket (GPU 0↔1, 2↔3)
export NCCL_ALGO=Ring              # optimal for 2-rank communicators
export NCCL_PROTO=Simple           # lowest latency; LL128 adds overhead for small KV tensors
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "Starting vLLM Disaggregated Serving (4× L40S + Llama-2-7B)"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Migration: T_MIN=$MIGRATION_T_MIN tok, R_SLOW=$MIGRATION_R_SLOW tok/s"
echo "  Fast Lane (FP16): GPU $FAST_PREFILL_GPU→$FAST_DECODE_GPU, ports $FAST_PREFILL_PORT/$FAST_DECODE_PORT"
echo "  Slow Lane (BF16): GPU $SLOW_PREFILL_GPU→$SLOW_DECODE_GPU, ports $SLOW_PREFILL_PORT/$SLOW_DECODE_PORT"
echo "  Proxy: HTTP $PROXY_HTTP_PORT, ZMQ fast=$PROXY_PORT slow=$SLOW_PROXY_PORT"
echo "  Timeout: ${TIMEOUT_SECONDS}s"
echo ""

PIDS=()

cleanup_and_exit() {
    local msg="$1"
    echo "  ✗ $msg"
    echo "  Killing all background processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    exit 1
}

# =============================================================================
# Launch Migration-Aware Proxy
# =============================================================================
echo "Starting migration-aware proxy server..."
PROXY_PORT=$PROXY_PORT \
SLOW_PROXY_PORT=$SLOW_PROXY_PORT \
PROXY_HTTP_PORT=$PROXY_HTTP_PORT \
MODEL=$MODEL \
MIGRATION_R_SLOW=$MIGRATION_R_SLOW \
MIGRATION_T_MIN=$MIGRATION_T_MIN \
    $PYTHON_BIN disagg_proxy_migration.py > proxy.log 2>&1 &
PROXY_PID=$!
PIDS+=($PROXY_PID)
echo "  ✓ Proxy started (PID: $PROXY_PID)"
sleep 3

# =============================================================================
# Launch Fast-Lane Prefill (GPU 0, FP16)
# Memory budget:  48 × 0.85 = 40.8 GB → weights 14 GB → 24.8 GB for KV
# nccl_num_channels=8 (reduced from 16; fewer channels sufficient for PCIe bandwidth)
# =============================================================================
echo "Starting fast-lane prefill server (GPU $FAST_PREFILL_GPU, FP16)..."
CUDA_VISIBLE_DEVICES=$FAST_PREFILL_GPU $VLLM_BIN serve $MODEL \
    --enforce-eager \
    --host 0.0.0.0 \
    --port $FAST_PREFILL_PORT \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --dtype float16 \
    --max-num-seqs 32 \
    --max-model-len 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --disable-sliding-window \
    --no-enable-chunked-prefill \
    --kv-cache-dtype auto \
    --kv-transfer-config "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e9\",\"kv_port\":\"$FAST_PREFILL_KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$FAST_PREFILL_PORT\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"8\"}}" \
    > fast_prefill.log 2>&1 &
FAST_PREFILL_PID=$!
PIDS+=($FAST_PREFILL_PID)
echo "  ✓ Fast prefill started (PID: $FAST_PREFILL_PID)"

# =============================================================================
# Launch Fast-Lane Decode (GPU 1, FP16)
# Memory budget: 48 × 0.80 = 38.4 GB → weights 14 GB → 22.4 GB; kv_buffer 4 GB → 18.4 GB for KV
# max-num-seqs=16: conservative cap so KV pool isn't exhausted
# =============================================================================
echo "Starting fast-lane decode server (GPU $FAST_DECODE_GPU, FP16)..."
CUDA_VISIBLE_DEVICES=$FAST_DECODE_GPU $VLLM_BIN serve $MODEL \
    --enforce-eager \
    --host 0.0.0.0 \
    --port $FAST_DECODE_PORT \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --dtype float16 \
    --max-num-seqs 16 \
    --max-model-len 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.80 \
    --disable-sliding-window \
    --no-enable-chunked-prefill \
    --kv-cache-dtype auto \
    --kv-transfer-config "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"4e9\",\"kv_port\":\"$FAST_DECODE_KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$FAST_DECODE_PORT\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"8\"}}" \
    > fast_decode.log 2>&1 &
FAST_DECODE_PID=$!
PIDS+=($FAST_DECODE_PID)
echo "  ✓ Fast decode started (PID: $FAST_DECODE_PID)"

# =============================================================================
# Launch Slow-Lane Prefill (GPU 2, BF16)
# Same memory budget as fast-prefill; BF16 has same per-param size as FP16
# =============================================================================
echo "Starting slow-lane prefill server (GPU $SLOW_PREFILL_GPU, BF16)..."
CUDA_VISIBLE_DEVICES=$SLOW_PREFILL_GPU $VLLM_BIN serve $MODEL \
    --enforce-eager \
    --host 0.0.0.0 \
    --port $SLOW_PREFILL_PORT \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --dtype bfloat16 \
    --max-num-seqs 32 \
    --max-model-len 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.80 \
    --disable-sliding-window \
    --no-enable-chunked-prefill \
    --kv-cache-dtype auto \
    --kv-transfer-config "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e9\",\"kv_port\":\"$SLOW_PREFILL_KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$SLOW_PROXY_PORT\",\"http_port\":\"$SLOW_PREFILL_PORT\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"8\"}}" \
    > slow_prefill.log 2>&1 &
SLOW_PREFILL_PID=$!
PIDS+=($SLOW_PREFILL_PID)
echo "  ✓ Slow prefill started (PID: $SLOW_PREFILL_PID)"

# =============================================================================
# Launch Slow-Lane Decode (GPU 3, BF16)
# =============================================================================
echo "Starting slow-lane decode server (GPU $SLOW_DECODE_GPU, BF16)..."
CUDA_VISIBLE_DEVICES=$SLOW_DECODE_GPU $VLLM_BIN serve $MODEL \
    --enforce-eager \
    --host 0.0.0.0 \
    --port $SLOW_DECODE_PORT \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --dtype bfloat16 \
    --max-num-seqs 16 \
    --max-model-len 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.80 \
    --disable-sliding-window \
    --no-enable-chunked-prefill \
    --kv-cache-dtype auto \
    --kv-transfer-config "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"4e9\",\"kv_port\":\"$SLOW_DECODE_KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$SLOW_PROXY_PORT\",\"http_port\":\"$SLOW_DECODE_PORT\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"8\"}}" \
    > slow_decode.log 2>&1 &
SLOW_DECODE_PID=$!
PIDS+=($SLOW_DECODE_PID)
echo "  ✓ Slow decode started (PID: $SLOW_DECODE_PID)"

echo ""
echo "All servers launched. Waiting for initialization..."
echo "Logs: fast_prefill.log  fast_decode.log  slow_prefill.log  slow_decode.log  proxy.log"
echo ""

# =============================================================================
# Health-Check Loop (same as startup.sh)
# =============================================================================
check_server_ready() {
    local port=$1
    local name=$2
    local max_wait=$3
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            if timeout 2 bash -c "echo > /dev/tcp/localhost/$port" 2>/dev/null; then
                return 0
            fi
        fi
        sleep 5
        waited=$((waited + 5))
        if [ $((waited % 60)) -eq 0 ]; then
            echo "  Waiting for $name... (${waited}s elapsed)"
        fi
    done
    return 1
}

echo "Waiting for fast-lane prefill (port $FAST_PREFILL_PORT)..."
if check_server_ready $FAST_PREFILL_PORT "fast-lane prefill" $TIMEOUT_SECONDS; then
    echo "  ✓ fast-lane prefill ready"
else
    cleanup_and_exit "fast-lane prefill failed to start within ${TIMEOUT_SECONDS}s. Check fast_prefill.log"
fi

echo "Waiting for fast-lane decode (port $FAST_DECODE_PORT)..."
if check_server_ready $FAST_DECODE_PORT "fast-lane decode" $TIMEOUT_SECONDS; then
    echo "  ✓ fast-lane decode ready"
else
    cleanup_and_exit "fast-lane decode failed to start within ${TIMEOUT_SECONDS}s. Check fast_decode.log"
fi

echo "Waiting for slow-lane prefill (port $SLOW_PREFILL_PORT)..."
if check_server_ready $SLOW_PREFILL_PORT "slow-lane prefill" $TIMEOUT_SECONDS; then
    echo "  ✓ slow-lane prefill ready"
else
    cleanup_and_exit "slow-lane prefill failed to start within ${TIMEOUT_SECONDS}s. Check slow_prefill.log"
fi

echo "Waiting for slow-lane decode (port $SLOW_DECODE_PORT)..."
if check_server_ready $SLOW_DECODE_PORT "slow-lane decode" $TIMEOUT_SECONDS; then
    echo "  ✓ slow-lane decode ready"
else
    cleanup_and_exit "slow-lane decode failed to start within ${TIMEOUT_SECONDS}s. Check slow_decode.log"
fi

echo ""
echo "============================================================"
echo "✓ All 4 vLLM servers + proxy are running."
echo "  Fast Lane:  prefill=$FAST_PREFILL_PORT  decode=$FAST_DECODE_PORT"
echo "  Slow Lane:  prefill=$SLOW_PREFILL_PORT  decode=$SLOW_DECODE_PORT"
echo "  Proxy HTTP: $PROXY_HTTP_PORT"
echo "============================================================"
