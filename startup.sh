#!/bin/bash

# =============================================================================
# Startup Script for vLLM Disaggregated Serving (4-GPU: Fast Lane + Slow Lane)
# =============================================================================
# This script launches the migration-aware proxy and two independent
# prefill-decode pairs organized as:
#
#   Fast Lane  (GPUs 0+1, FP16)  — all incoming requests
#     GPU 0: Prefill   port 20098
#     GPU 1: Decode    port 20099
#
#   Slow Lane  (GPUs 2+3, BF16)  — migrated stragglers only
#     GPU 2: Prefill   port 20096
#     GPU 3: Decode    port 20097
#
# The proxy (port 10099) is the sole client entry point.
# The system will be ready to handle requests after all servers start up.
# =============================================================================

# Activate conda environment
# Set SCRATCH_ROOT to your HPC scratch directory (default: $HOME/scratch)
SCRATCH_ROOT=${SCRATCH_ROOT:-$HOME/scratch}
CONDA_ENV=${CONDA_ENV:-$SCRATCH_ROOT/envs/sysml_research4}
export PATH="$CONDA_ENV/bin:$PATH"
PYTHON_BIN="$CONDA_ENV/bin/python3"
VLLM_BIN="$CONDA_ENV/bin/vllm"

# Configuration
MODEL=${MODEL:-meta-llama/Llama-2-13b-hf}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-30099}
PROXY_HTTP_PORT=${PROXY_HTTP_PORT:-10099}

# Hugging Face cache relocation (use scratch space instead of ~/.cache)
HF_CACHE_ROOT=${HF_CACHE_ROOT:-$SCRATCH_ROOT/huggingface_cache}
mkdir -p "$HF_CACHE_ROOT"
export HF_HOME="$HF_CACHE_ROOT"
echo "HuggingFace cache set to: $HF_CACHE_ROOT (HF_HOME)"

# Fast Lane PD pair (FP16, latency-optimized)
FAST_PREFILL_GPU=${FAST_PREFILL_GPU:-0}
FAST_DECODE_GPU=${FAST_DECODE_GPU:-1}
FAST_PREFILL_PORT=${FAST_PREFILL_PORT:-20098}
FAST_DECODE_PORT=${FAST_DECODE_PORT:-20099}
FAST_PREFILL_KV_PORT=${FAST_PREFILL_KV_PORT:-21098}
FAST_DECODE_KV_PORT=${FAST_DECODE_KV_PORT:-22099}

# Slow Lane PD pair (BF16, straggler-tolerant)
SLOW_PREFILL_GPU=${SLOW_PREFILL_GPU:-2}
SLOW_DECODE_GPU=${SLOW_DECODE_GPU:-3}
SLOW_PREFILL_PORT=${SLOW_PREFILL_PORT:-20096}
SLOW_DECODE_PORT=${SLOW_DECODE_PORT:-20097}
SLOW_PREFILL_KV_PORT=${SLOW_PREFILL_KV_PORT:-21096}
SLOW_DECODE_KV_PORT=${SLOW_DECODE_KV_PORT:-22097}

echo "============================================================"
echo "Starting vLLM Disaggregated Serving (4-GPU: Fast + Slow Lane)"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Fast Lane (FP16):"
echo "    Prefill GPU: $FAST_PREFILL_GPU, Port: $FAST_PREFILL_PORT, KV Port: $FAST_PREFILL_KV_PORT"
echo "    Decode  GPU: $FAST_DECODE_GPU, Port: $FAST_DECODE_PORT, KV Port: $FAST_DECODE_KV_PORT"
echo "  Slow Lane (BF16):"
echo "    Prefill GPU: $SLOW_PREFILL_GPU, Port: $SLOW_PREFILL_PORT, KV Port: $SLOW_PREFILL_KV_PORT"
echo "    Decode  GPU: $SLOW_DECODE_GPU, Port: $SLOW_DECODE_PORT, KV Port: $SLOW_DECODE_KV_PORT"
echo "  Proxy Port: $PROXY_PORT (ZMQ), $PROXY_HTTP_PORT (HTTP)"
echo "  Timeout: ${TIMEOUT_SECONDS}s"
echo ""

# Array to store PIDs
PIDS=()

# =============================================================================
# Launch Migration-Aware Proxy Server
# =============================================================================
echo "Starting migration-aware proxy server..."
PROXY_PORT=$PROXY_PORT PROXY_HTTP_PORT=$PROXY_HTTP_PORT $PYTHON_BIN disagg_proxy_migration.py > proxy.log 2>&1 &
PROXY_PID=$!
PIDS+=($PROXY_PID)
echo "  ✓ Proxy server started (PID: $PROXY_PID)"

# Give proxy time to start
sleep 3

# =============================================================================
# Launch Fast Lane Prefill Server (GPU 0, FP16)
# =============================================================================
echo "Starting fast-lane prefill server on GPU $FAST_PREFILL_GPU..."
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
    --gpu-memory-utilization 0.85 \
    --disable-sliding-window \
    --no-enable-chunked-prefill \
    --kv-cache-dtype auto \
    --kv-transfer-config "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e9\",\"kv_port\":\"$FAST_PREFILL_KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$FAST_PREFILL_PORT\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" > fast_prefill.log 2>&1 &
FAST_PREFILL_PID=$!
PIDS+=($FAST_PREFILL_PID)
echo "  ✓ Fast-lane prefill server started (PID: $FAST_PREFILL_PID)"

# =============================================================================
# Launch Fast Lane Decode Server (GPU 1, FP16)
# =============================================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting fast-lane decode server on GPU $FAST_DECODE_GPU..."
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
    --gpu-memory-utilization 0.6 \
    --disable-sliding-window \
    --no-enable-chunked-prefill \
    --kv-cache-dtype auto \
    --kv-transfer-config "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"1.4e10\",\"kv_port\":\"$FAST_DECODE_KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$FAST_DECODE_PORT\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" > fast_decode.log 2>&1 &
FAST_DECODE_PID=$!
PIDS+=($FAST_DECODE_PID)
echo "  ✓ Fast-lane decode server started (PID: $FAST_DECODE_PID)"

# =============================================================================
# Launch Slow Lane Prefill Server (GPU 2, BF16)
# =============================================================================
echo "Starting slow-lane prefill server on GPU $SLOW_PREFILL_GPU..."
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
    --gpu-memory-utilization 0.7 \
    --disable-sliding-window \
    --no-enable-chunked-prefill \
    --kv-cache-dtype auto \
    --kv-transfer-config "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e9\",\"kv_port\":\"$SLOW_PREFILL_KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$SLOW_PREFILL_PORT\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" > slow_prefill.log 2>&1 &
SLOW_PREFILL_PID=$!
PIDS+=($SLOW_PREFILL_PID)
echo "  ✓ Slow-lane prefill server started (PID: $SLOW_PREFILL_PID)"

# =============================================================================
# Launch Slow Lane Decode Server (GPU 3, BF16)
# =============================================================================
echo "Starting slow-lane decode server on GPU $SLOW_DECODE_GPU..."
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
    --gpu-memory-utilization 0.7 \
    --disable-sliding-window \
    --no-enable-chunked-prefill \
    --kv-cache-dtype auto \
    --kv-transfer-config "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"1.4e10\",\"kv_port\":\"$SLOW_DECODE_KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$SLOW_DECODE_PORT\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" > slow_decode.log 2>&1 &
SLOW_DECODE_PID=$!
PIDS+=($SLOW_DECODE_PID)
echo "  ✓ Slow-lane decode server started (PID: $SLOW_DECODE_PID)"

echo ""
echo "All servers launched. Waiting for initialization..."
echo "Log files: fast_prefill.log, fast_decode.log, slow_prefill.log, slow_decode.log, proxy.log"
echo ""

# =============================================================================
# Wait for Servers to Start
# =============================================================================

# Function to check if server is ready by checking if port is listening
check_server_ready() {
    local port=$1
    local name=$2
    local max_wait=$3
    local waited=0

    while [ $waited -lt $max_wait ]; do
        # Check if port is listening
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            # Also check if we can connect
            if timeout 2 bash -c "echo > /dev/tcp/localhost/$port" 2>/dev/null; then
                return 0
            fi
        fi
        sleep 5
        waited=$((waited + 5))
        if [ $((waited % 20)) -eq 0 ]; then
            echo "  Waiting for $name... (${waited}s elapsed)"
        fi
    done
    return 1
}

echo "Waiting for fast-lane prefill server (port $FAST_PREFILL_PORT)..."
if check_server_ready $FAST_PREFILL_PORT "fast-lane prefill" 120; then
    echo "  ✓ Fast-lane prefill server ready"
else
    echo "  ✗ Fast-lane prefill server failed to start. Check fast_prefill.log"
    tail -20 fast_prefill.log
    exit 1
fi

echo "Waiting for fast-lane decode server (port $FAST_DECODE_PORT)..."
if check_server_ready $FAST_DECODE_PORT "fast-lane decode" 120; then
    echo "  ✓ Fast-lane decode server ready"
else
    echo "  ✗ Fast-lane decode server failed to start. Check fast_decode.log"
    tail -20 fast_decode.log
    exit 1
fi

echo "Waiting for slow-lane prefill server (port $SLOW_PREFILL_PORT)..."
if check_server_ready $SLOW_PREFILL_PORT "slow-lane prefill" 120; then
    echo "  ✓ Slow-lane prefill server ready"
else
    echo "  ✗ Slow-lane prefill server failed to start. Check slow_prefill.log"
    tail -20 slow_prefill.log
    exit 1
fi

echo "Waiting for slow-lane decode server (port $SLOW_DECODE_PORT)..."
if check_server_ready $SLOW_DECODE_PORT "slow-lane decode" 120; then
    echo "  ✓ Slow-lane decode server ready"
else
    echo "  ✗ Slow-lane decode server failed to start. Check slow_decode.log"
    tail -20 slow_decode.log
    exit 1
fi

echo "Waiting for proxy server (port $PROXY_HTTP_PORT)..."
if check_server_ready $PROXY_HTTP_PORT "proxy" 30; then
    echo "  ✓ Proxy server ready"
else
    echo "  ✗ Proxy server failed to start. Check proxy.log"
    tail -20 proxy.log
    exit 1
fi

echo ""
echo "============================================================"
echo "✓ All servers are ready!"
echo "============================================================"
echo ""
echo "System Status:"
echo "  Proxy (entry point): http://localhost:$PROXY_HTTP_PORT"
echo "  Fast Lane:"
echo "    Prefill: http://localhost:$FAST_PREFILL_PORT"
echo "    Decode:  http://localhost:$FAST_DECODE_PORT"
echo "  Slow Lane:"
echo "    Prefill: http://localhost:$SLOW_PREFILL_PORT"
echo "    Decode:  http://localhost:$SLOW_DECODE_PORT"
echo ""
echo "Running processes:"
ps aux | grep -E "(vllm|disagg_proxy)" | grep -v grep | awk '{printf "  PID %s: %s\n", $2, $NF}'
echo ""
echo "Test with:"
echo "  curl -X POST http://localhost:$PROXY_HTTP_PORT/v1/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"$MODEL\", \"prompt\": \"Hello, my name is\", \"max_tokens\": 20, \"temperature\": 0.0}'"
echo ""
echo "To stop all servers, run: ./cleanup.sh"
echo ""
