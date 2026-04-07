#!/bin/bash

# =============================================================================
# Cleanup Script for vLLM Disaggregated Serving
# =============================================================================
# This script stops all vLLM and proxy processes and cleans up network
# connections used by the disaggregated serving system.
# =============================================================================

echo "=========================================="
echo "Cleaning up vLLM Disaggregated Serving"
echo "=========================================="
echo ""

# Kill all vLLM processes
echo "Stopping vLLM processes..."
pkill -9 -f "vllm serve" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ vLLM processes terminated"
else
    echo "  ℹ No vLLM processes found"
fi

# Kill proxy server
echo "Stopping proxy server..."
pkill -9 -f "disagg_proxy" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ Proxy server terminated"
else
    echo "  ℹ No proxy server found"
fi

# Wait for processes to terminate
sleep 2

# Force kill any remaining Python processes related to vLLM (be careful with this)
REMAINING=$(ps aux | grep -E "(vllm|disagg_proxy)" | grep -v grep | wc -l)
if [ $REMAINING -gt 0 ]; then
    echo "  ⚠ Found $REMAINING remaining processes, forcing cleanup..."
    ps aux | grep -E "(vllm|disagg_proxy)" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
    sleep 1
fi

# Clean up ports used by the system
PORTS=(10099 20096 20097 20098 20099 21096 21098 22097 22099 30099)
echo ""
echo "Cleaning up network ports..."
for PORT in "${PORTS[@]}"; do
    # Check if port is in use
    if netstat -tuln 2>/dev/null | grep -q ":$PORT "; then
        echo "  Freeing port $PORT..."
        fuser -k ${PORT}/tcp 2>/dev/null
        sleep 0.5
    fi
done

# Wait for ports to be freed
sleep 2

# Verify cleanup
echo ""
echo "Verifying cleanup..."
ACTIVE_PROCESSES=$(ps aux | grep -E "(vllm|disagg_proxy)" | grep -v grep | wc -l)
ACTIVE_PORTS=$(netstat -tuln 2>/dev/null | grep -E ":(10099|20096|20097|20098|20099|21096|21098|22097|22099|30099)" | wc -l)

if [ $ACTIVE_PROCESSES -eq 0 ]; then
    echo "  ✓ All processes stopped"
else
    echo "  ⚠ Warning: $ACTIVE_PROCESSES processes still running"
    ps aux | grep -E "(vllm|disagg_proxy)" | grep -v grep
fi

if [ $ACTIVE_PORTS -eq 0 ]; then
    echo "  ✓ All ports freed"
else
    echo "  ⚠ Warning: $ACTIVE_PORTS ports still in use"
    netstat -tuln 2>/dev/null | grep -E ":(10099|20096|20097|20098|20099|21096|21098|22097|22099|30099)"
fi

# Clean up log files (optional, commented out by default)
# echo ""
# echo "Cleaning up log files..."
# rm -f prefill.log decode.log proxy.log startup.log
# echo "  ✓ Log files removed"

echo ""
echo "=========================================="
echo "Cleanup complete!"
echo "=========================================="
