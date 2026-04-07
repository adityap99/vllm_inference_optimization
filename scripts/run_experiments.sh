#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh — Master experiment runner (PROPOSAL.md §6)
# =============================================================================
#
# Runs all 5 evaluation conditions against the 4-GPU migration-aware serving
# stack, organizing results per condition for analysis.
#
# EXPERIMENTAL DESIGN
# ───────────────────
# The key comparison is straggler ITL *without* migration vs *with* migration.
# To obtain a clean no-migration baseline on the same hardware, the proxy is
# restarted with MIGRATION_T_MIN=999999 (effectively infinite — migration never
# fires) for the four baseline conditions.  For the treatment condition
# (migration_straggler) it is restarted with the default T_MIN=200.
#
# Phase 1 – migration DISABLED (T_MIN=999999)
#   Condition 1: no_straggler       → clean baseline ITL
#   Condition 2: mixed              → light mixed-load ITL
#   Condition 3: straggler          → ITL degradation with stragglers, no eviction
#   Condition 4: stress_straggler   → heavy straggler pressure
#
# Phase 2 – migration ENABLED (T_MIN=200)
#   Condition 5: migration_straggler → same load as straggler, migration active
#
# PREREQUISITES
# ─────────────
#   startup.sh must have been run and all 5 servers must be healthy:
#     GPU 0 fast-prefill :20098, GPU 1 fast-decode :20099,
#     GPU 2 slow-prefill :20096, GPU 3 slow-decode :20097,
#     proxy :10099
#
# USAGE
# ─────
#   bash scripts/run_experiments.sh
#   SKIP_EXISTING=1 bash scripts/run_experiments.sh   # resume interrupted run
#   COOLDOWN=30     bash scripts/run_experiments.sh   # shorter cooldown (testing)
#
# OUTPUTS
# ───────
#   results/<condition>/*.json          vllm bench --save-result files
#   results/<condition>/logs/           per-stream stdout logs
#   results/<condition>/run.log         orchestration log
#   results/<condition>/metadata.json   timing and proxy-log offsets
#   proxy.log                           live proxy events (migration events here)
#
# Then run: python scripts/analyze_results.py
# =============================================================================

set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$REPO_DIR/scripts"
RESULTS_DIR="$REPO_DIR/results"
PROXY_LOG="$REPO_DIR/proxy.log"

# ─── Configuration ────────────────────────────────────────────────────────────

export VLLM_HOST="${VLLM_HOST:-localhost}"
export VLLM_PORT="${VLLM_PORT:-10099}"
export VLLM_MODEL="${VLLM_MODEL:-meta-llama/Llama-2-13b-hf}"
MODEL="$VLLM_MODEL"

# ZMQ proxy ports (must match startup.sh)
PROXY_ZMQ_PORT="${PROXY_ZMQ_PORT:-30099}"
SLOW_ZMQ_PORT="${SLOW_ZMQ_PORT:-30097}"
PROXY_HTTP_PORT="${VLLM_PORT}"

# Python binary — reuse the conda env that startup.sh set SCRATCH_ROOT/CONDA_ENV for
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"
CONDA_ENV="${CONDA_ENV:-$SCRATCH_ROOT/envs/sysml_research4}"
if [[ -x "$CONDA_ENV/bin/python3" ]]; then
    PYTHON_BIN="$CONDA_ENV/bin/python3"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

# Experiment settings
COOLDOWN="${COOLDOWN:-90}"        # seconds between conditions
RESTART_WAIT="${RESTART_WAIT:-30}" # seconds to wait after proxy restart for re-registration
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# Grafana annotations (marks condition start/end as region annotations in dashboards)
# Overridable via env vars if your Grafana setup differs from defaults.
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
GRAFANA_USER="${GRAFANA_USER:-admin}"
GRAFANA_PASS="${GRAFANA_PASS:-admin}"

# ─── Helpers ──────────────────────────────────────────────────────────────────

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Post a Grafana region annotation via the HTTP API.
# Gracefully does nothing if Grafana is unreachable.
# Args: text  tags_json  start_ms  [end_ms]
#   tags_json: JSON array literal without outer [], e.g. '"experiment","straggler"'
grafana_annotate() {
    local text="$1"
    local tags_json="$2"
    local time_ms="${3:-$(( $(date +%s) * 1000 ))}"
    local time_end_ms="${4:-}"

    local json
    if [[ -n "$time_end_ms" ]]; then
        json="{\"text\":\"${text}\",\"tags\":[${tags_json}],\"time\":${time_ms},\"timeEnd\":${time_end_ms}}"
    else
        json="{\"text\":\"${text}\",\"tags\":[${tags_json}],\"time\":${time_ms}}"
    fi

    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time 5 \
        -X POST "${GRAFANA_URL}/api/annotations" \
        -u "${GRAFANA_USER}:${GRAFANA_PASS}" \
        -H "Content-Type: application/json" \
        -d "$json" 2>/dev/null) || http_code="ERR"

    if [[ "$http_code" == "200" ]]; then
        log "  [grafana] ✓ Annotation posted: ${text}"
    else
        log "  [grafana] Annotation skipped (HTTP ${http_code} — Grafana may not be running)"
    fi
}

# Send a test inference.  Returns the HTTP status code.
_probe_proxy() {
    curl -s -o /dev/null -w "%{http_code}" \
        --max-time 15 \
        -X POST "http://${VLLM_HOST}:${PROXY_HTTP_PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"prompt\":\"Hi\",\"max_tokens\":1}" \
        2>/dev/null || echo "000"
}

# Block until the proxy returns HTTP 200 for a real inference request.
# This confirms both the proxy process AND the vLLM lane registrations are up.
wait_proxy_ready() {
    local max_wait="${1:-180}"
    local waited=0
    log "Waiting for proxy at ${VLLM_HOST}:${PROXY_HTTP_PORT} to accept inferences..."
    while [[ $waited -lt $max_wait ]]; do
        local code
        code="$(_probe_proxy)"
        if [[ "$code" == "200" ]]; then
            log "✓ Proxy ready (HTTP 200)."
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        if [[ $((waited % 30)) -eq 0 ]]; then
            log "  still waiting... (${waited}s, last status: $code)"
        fi
    done
    log "✗ Proxy did not become ready within ${max_wait}s."
    return 1
}

# Send several warmup requests to bring CUDA graphs and KV allocator into
# a steady state before measuring starts.
send_warmup() {
    log "Sending warmup requests..."
    for i in 1 2 3 4 5; do
        curl -s -o /dev/null \
            --max-time 60 \
            -X POST "http://${VLLM_HOST}:${PROXY_HTTP_PORT}/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":20}" \
            2>/dev/null || true
    done
    log "  ✓ Warmup done. Sleeping 10s for KV cache to settle..."
    sleep 10
}

# Kill any running migration proxy and start a fresh one with the given T_MIN.
# The vLLM servers will re-register via ZMQ within RESTART_WAIT seconds.
restart_proxy() {
    local t_min="${1:-200}"
    local label="${2:-migration ${1}}"

    echo ""
    echo "──────────────────────────────────────────────────────────"
    log "Restarting migration proxy: MIGRATION_T_MIN=${t_min} (${label})"
    echo "──────────────────────────────────────────────────────────"

    # Kill existing proxy (by script name, not PID, since startup.sh didn't write a PID file)
    if pkill -f "disagg_proxy_migration.py" 2>/dev/null; then
        log "  Killed existing proxy process."
    else
        log "  No running proxy found (may have already exited)."
    fi

    # Wait for the ZMQ port sockets to close
    sleep 5

    # Append a separator to proxy.log so events can be attributed per phase
    {
        echo ""
        echo "=== PROXY RESTART: MIGRATION_T_MIN=${t_min}  time=$(date -Iseconds) ==="
        echo ""
    } >> "$PROXY_LOG"

    # Start new proxy with updated T_MIN
    PROXY_PORT="$PROXY_ZMQ_PORT" \
    SLOW_PROXY_PORT="$SLOW_ZMQ_PORT" \
    PROXY_HTTP_PORT="$PROXY_HTTP_PORT" \
    MODEL="$MODEL" \
    MIGRATION_T_MIN="$t_min" \
        "$PYTHON_BIN" "$REPO_DIR/disagg_proxy_migration.py" >>"$PROXY_LOG" 2>&1 &

    local new_pid=$!
    log "  New proxy started (PID ${new_pid})."

    # Give vLLM connectors time to detect the reconnect and re-send their ZMQ pings.
    # DEFAULT_PING_SECONDS=5 in the proxy; connectors typically ping multiple times
    # per second, so 30s is conservative.
    log "  Waiting ${RESTART_WAIT}s for vLLM servers to re-register..."
    sleep "$RESTART_WAIT"

    # Verify the proxy is accepting real inferences (confirms registration)
    wait_proxy_ready 120 || {
        log "ERROR: Proxy did not become ready after restart. Aborting."
        exit 1
    }
}

# Run one experimental condition.
# Pushes into results/<name> so --save-result JSONs and logs land there.
run_experiment() {
    local name="$1"
    local script="$2"
    local result_dir="$RESULTS_DIR/$name"

    mkdir -p "$result_dir"

    if [[ "$SKIP_EXISTING" == "1" && -f "$result_dir/.done" ]]; then
        log "SKIP: $name (.done marker present)"
        return 0
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════"
    log "EXPERIMENT: $name"
    echo "════════════════════════════════════════════════════════════"
    log "  output dir : $result_dir"
    log "  proxy      : http://${VLLM_HOST}:${PROXY_HTTP_PORT}"

    # Record proxy.log line offset so the analysis script can isolate events
    local proxy_start_line=0
    proxy_start_line=$(wc -l < "$PROXY_LOG" 2>/dev/null || echo 0)

    # Capture wall-clock start time (Unix epoch in both seconds and ms for Prometheus/Grafana)
    local start_epoch start_epoch_ms
    start_epoch=$(date +%s)
    start_epoch_ms=$(( start_epoch * 1000 ))

    # Post Grafana start-of-region annotation
    grafana_annotate "START: ${name}" "\"experiment\",\"start\",\"${name}\"" "${start_epoch_ms}"

    # Write pre-run metadata
    cat > "$result_dir/metadata.json" <<EOF
{
  "experiment": "$name",
  "start_time": "$(date -Iseconds)",
  "start_epoch": $start_epoch,
  "proxy_host": "$VLLM_HOST",
  "proxy_port": "$PROXY_HTTP_PORT",
  "model": "$MODEL",
  "proxy_log_start_line": $proxy_start_line,
  "status": "running"
}
EOF

    # Run the workload from inside the result directory so --save-result JSONs
    # and the log subdirectory (logs/small_*.log, logs/long_*.log) land there.
    local exit_code=0
    pushd "$result_dir" >/dev/null
    bash "$script" 2>&1 | tee run.log || exit_code=${PIPESTATUS[0]}
    popd >/dev/null

    # Capture wall-clock end time
    local end_epoch end_epoch_ms
    end_epoch=$(date +%s)
    end_epoch_ms=$(( end_epoch * 1000 ))

    # Close the Grafana region annotation
    grafana_annotate "END: ${name}" "\"experiment\",\"end\",\"${name}\"" "${start_epoch_ms}" "${end_epoch_ms}"

    # Record post-run metadata
    local proxy_end_line=0
    proxy_end_line=$(wc -l < "$PROXY_LOG" 2>/dev/null || echo 0)

    cat > "$result_dir/metadata.json" <<EOF
{
  "experiment": "$name",
  "start_time": "$(date -Iseconds)",
  "end_time": "$(date -Iseconds)",
  "start_epoch": $start_epoch,
  "end_epoch": $end_epoch,
  "exit_code": $exit_code,
  "proxy_host": "$VLLM_HOST",
  "proxy_port": "$PROXY_HTTP_PORT",
  "model": "$MODEL",
  "proxy_log_start_line": $proxy_start_line,
  "proxy_log_end_line": $proxy_end_line,
  "status": "$( [[ $exit_code -eq 0 ]] && echo completed || echo failed )"
}
EOF

    if [[ $exit_code -eq 0 ]]; then
        touch "$result_dir/.done"
        log "✓ $name completed (exit 0)"
    else
        log "⚠ $name exited with code ${exit_code} — results may be partial"
    fi

    return 0  # continue to next experiment even on partial failure
}

# ─── Main ─────────────────────────────────────────────────────────────────────

echo "============================================================"
echo " vLLM Straggler Migration — Experiment Runner"
echo " $(date)"
echo "============================================================"
log "REPO_DIR    = $REPO_DIR"
log "RESULTS_DIR = $RESULTS_DIR"
log "Model       = $MODEL"
log "Proxy       = http://${VLLM_HOST}:${PROXY_HTTP_PORT}"
log "Python      = $PYTHON_BIN"
log "Cooldown    = ${COOLDOWN}s between conditions"
log "Skip if done = $SKIP_EXISTING"
echo ""

# Verify the proxy is alive before touching anything
log "Checking proxy is alive before starting experiments..."
wait_proxy_ready 60 || {
    echo ""
    echo "ERROR: Proxy is not responding. Did you run startup.sh first?"
    echo "  startup.sh must be run before run_experiments.sh."
    exit 1
}

mkdir -p "$RESULTS_DIR"

# =============================================================================
# PHASE 1 — Migration DISABLED  (T_MIN=999999, effectively infinite)
# =============================================================================
# Long requests stay on the fast lane for their full duration, causing ITL
# degradation for co-running short requests.  This is the "no migration" baseline.
# =============================================================================

echo ""
echo "████████████████████████████████████████████████████████████"
log "PHASE 1: Baseline conditions (migration DISABLED)"
echo "████████████████████████████████████████████████████████████"
restart_proxy 999999 "migration disabled"
send_warmup

# ── Condition 1: clean baseline (no stragglers) ──────────────────────────────
run_experiment "no_straggler" "$SCRIPTS_DIR/no_straggler_vllm_load.sh"
log "Cooldown ${COOLDOWN}s..."
sleep "$COOLDOWN"

# ── Condition 2: light mixed load ────────────────────────────────────────────
run_experiment "mixed" "$SCRIPTS_DIR/mixed_vllm_load.sh"
log "Cooldown ${COOLDOWN}s..."
sleep "$COOLDOWN"

# ── Condition 3: realistic straggler injection ───────────────────────────────
run_experiment "straggler" "$SCRIPTS_DIR/straggler_vllm_load.sh"
log "Cooldown ${COOLDOWN}s..."
sleep "$COOLDOWN"

# ── Condition 4: heavy straggler stress ──────────────────────────────────────
run_experiment "stress_straggler" "$SCRIPTS_DIR/stress_straggler_vllm_load.sh"

# Extra cooldown after the heaviest workload: stress_straggler can have
# 500 long requests in-flight; give the KV cache time to drain.
log "Post-stress cooldown $((COOLDOWN * 2))s (KV cache drain)..."
sleep "$((COOLDOWN * 2))"

# =============================================================================
# PHASE 2 — Migration ENABLED  (T_MIN=200, default)
# =============================================================================
# Same load as straggler.  The proxy detects stragglers at N=200 tokens and
# evicts them to the slow lane, freeing the fast lane.
# =============================================================================

echo ""
echo "████████████████████████████████████████████████████████████"
log "PHASE 2: Treatment condition (migration ENABLED, T_MIN=200)"
echo "████████████████████████████████████████████████████████████"
restart_proxy 200 "migration enabled"
send_warmup

# ── Condition 5: migration treatment ─────────────────────────────────────────
run_experiment "migration_straggler" "$SCRIPTS_DIR/migration_straggler_vllm_load.sh"

# =============================================================================
# Done
# =============================================================================

echo ""
echo "============================================================"
log "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Results directory layout:"
find "$RESULTS_DIR" -name "*.json" -not -name "metadata.json" | sort | head -40
echo ""
echo "Next step:"
echo "  python scripts/analyze_results.py"
echo "  (plots land in plots/)"
echo ""
