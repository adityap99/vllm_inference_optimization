#!/bin/bash

# =============================================================================
# Startup Script for vLLM Disaggregated Serving (1P1D)
# =============================================================================
# This script launches the proxy server, prefill server, and decode server
# for disaggregated serving with 1 Prefill + 1 Decode configuration.
# 
# The system will be ready to handle requests after all servers start up.
# =============================================================================

# Activate conda environment
CONDA_ENV="/home/hice1/apandit63/scratch/envs/sysml_research4"
export PATH="$CONDA_ENV/bin:$PATH"
PYTHON_BIN="$CONDA_ENV/bin/python3"
VLLM_BIN="$CONDA_ENV/bin/vllm"

# Configuration
MODEL=${MODEL:-meta-llama/Llama-2-13b-hf}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-30099}
PROXY_HTTP_PORT=${PROXY_HTTP_PORT:-10099}

# Hugging Face cache relocation (use scratch space instead of ~/.cache)
HF_CACHE_ROOT="/home/hice1/apandit63/scratch/huggingface_cache"
mkdir -p "$HF_CACHE_ROOT"
export HF_HOME="$HF_CACHE_ROOT"
echo "HuggingFace cache set to: $HF_CACHE_ROOT (HF_HOME)"

# 1P1D configuration
PREFILL_GPU=${PREFILL_GPU:-0}
DECODE_GPU=${DECODE_GPU:-1}
PREFILL_PORT=${PREFILL_PORT:-20098}
DECODE_PORT=${DECODE_PORT:-20099}
PREFILL_KV_PORT=${PREFILL_KV_PORT:-21098}
DECODE_KV_PORT=${DECODE_KV_PORT:-22099}

echo "=========================================="
echo "Starting vLLM Disaggregated Serving (1P1D)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Prefill GPU: $PREFILL_GPU, Port: $PREFILL_PORT, KV Port: $PREFILL_KV_PORT"
echo "  Decode GPU: $DECODE_GPU, Port: $DECODE_PORT, KV Port: $DECODE_KV_PORT"
echo "  Proxy Port: $PROXY_PORT (ZMQ), $PROXY_HTTP_PORT (HTTP)"
echo "  Timeout: ${TIMEOUT_SECONDS}s"
echo ""

# Array to store PIDs
PIDS=()

# =============================================================================
# Launch Proxy Server
# =============================================================================
echo "Starting proxy server..."
PROXY_PORT=$PROXY_PORT PROXY_HTTP_PORT=$PROXY_HTTP_PORT $PYTHON_BIN disagg_proxy_p2p_nccl_xpyd.py > proxy.log 2>&1 &
PROXY_PID=$!
PIDS+=($PROXY_PID)
echo "  ✓ Proxy server started (PID: $PROXY_PID)"

# Give proxy time to start
sleep 3

# =============================================================================
# Launch Prefill Server
# =============================================================================
echo "Starting prefill server on GPU $PREFILL_GPU..."
CUDA_VISIBLE_DEVICES=$PREFILL_GPU $VLLM_BIN serve $MODEL \
    --enforce-eager \
    --host 0.0.0.0 \
    --port $PREFILL_PORT \
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
    --kv-transfer-config "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e9\",\"kv_port\":\"$PREFILL_KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$PREFILL_PORT\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" > prefill.log 2>&1 &
PREFILL_PID=$!
PIDS+=($PREFILL_PID)
echo "  ✓ Prefill server started (PID: $PREFILL_PID)"

# =============================================================================
# =============================================================================
# Launch Decode Server
# =============================================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting decode server on GPU $DECODE_GPU..."
CUDA_VISIBLE_DEVICES=$DECODE_GPU $VLLM_BIN serve $MODEL \
    --enforce-eager \
    --host 0.0.0.0 \
    --port $DECODE_PORT \
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
    --kv-transfer-config "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"1.4e10\",\"kv_port\":\"$DECODE_KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$DECODE_PORT\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" > decode.log 2>&1 &
DECODE_PID=$!
PIDS+=($DECODE_PID)
echo "  ✓ Decode server started (PID: $DECODE_PID)"

echo ""
echo "All servers launched. Waiting for initialization..."
echo "Log files: prefill.log, decode.log, proxy.log"
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

echo "Waiting for prefill server (port $PREFILL_PORT)..."
if check_server_ready $PREFILL_PORT "prefill" 120; then
    echo "  ✓ Prefill server ready"
else
    echo "  ✗ Prefill server failed to start. Check prefill.log"
    tail -20 prefill.log
    exit 1
fi

echo "Waiting for decode server (port $DECODE_PORT)..."
if check_server_ready $DECODE_PORT "decode" 120; then
    echo "  ✓ Decode server ready"
else
    echo "  ✗ Decode server failed to start. Check decode.log"
    tail -20 decode.log
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
echo "=========================================="
echo "✓ All servers are ready!"
echo "=========================================="
echo ""
echo "System Status:"
echo "  Proxy:   http://localhost:$PROXY_HTTP_PORT"
echo "  Prefill: http://localhost:$PREFILL_PORT"
echo "  Decode:  http://localhost:$DECODE_PORT"
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
