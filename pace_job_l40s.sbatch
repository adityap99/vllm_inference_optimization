#!/bin/bash
# =============================================================================
# PACE SLURM Job Script — vLLM Straggler-Aware Migration Experiments
# ALTERNATIVE CONFIG: 4× L40S (48 GB PCIe) + Llama-2-7B
# =============================================================================
# Use this when H100 nodes are backlogged. L40S nodes (coc-gpu, ice-gpu)
# are typically much shorter-queue. Scientific validity is preserved because
# the experiment measures *relative* speedup of migration, not absolute TFLOPS.
#
# Submit with:
#   sbatch pace_job_l40s.sbatch
#
# Key differences from pace_job.sbatch:
#   - GPU: 4× L40S 48 GB (PCIe, no NVLink)   → was 4× H100 80 GB (NVLink)
#   - Model: Llama-2-7b-hf                    → was Llama-2-13b-hf
#   - Startup: startup_l40s.sh (tuned params) → was startup.sh
#   - MIGRATION_R_SLOW=15 tok/s               → was 30 tok/s (L40S is slower)
#   - NCCL PCIe P2P env vars added
# =============================================================================

# --- Resource allocation -------------------------------------------------------
#SBATCH --job-name=vllm-migration-l40s
#SBATCH --nodes=1                      # Must be 1: P2pNcclConnector requires same node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32             # 4 vLLM servers × ~6 cores + proxy + headroom
#SBATCH --mem=128G                     # 7B model × 4 servers peak ~60 GB CPU RAM; 128G safe
#SBATCH --gres=gpu:l40s:4             # Four L40S 48 GB GPUs on a single node
#SBATCH --time=01:30:00               # 1.5h: model load (~5 min) + all workloads (~62 min) + buffer

# --- Partition / account -------------------------------------------------------
#SBATCH --partition=ice-gpu            # L40S nodes are in ice-gpu (and coc-gpu)
#SBATCH --account=coc                  # PACE allocation account
# To use coc-gpu instead (16-hour limit, same L40S nodes):
##SBATCH --partition=coc-gpu

# --- Output / logging ----------------------------------------------------------
#SBATCH --output=logs/slurm_l40s_%j.out
#SBATCH --error=logs/slurm_l40s_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Environment setup
# =============================================================================

set -euo pipefail

echo "========================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "Config:    L40S × 4 + Llama-2-7B (alternative run)"
echo "========================================================"

mkdir -p logs

# --- Paths -------------------------------------------------------------------
export SCRATCH_ROOT="${SCRATCH_ROOT:-${SCRATCH:-/home/hice1/$USER/scratch}}"
export CONDA_ENV="${SCRATCH_ROOT}/sysml_research4"

# HuggingFace
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN not set. Model download will fail if weights are not cached."
fi
export HF_HOME="${SCRATCH_ROOT}/huggingface_cache"
export MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"   # 7B for L40S (14 GB weights, fits 48 GB)
export HF_HUB_DISABLE_XET=1

# --- Migration threshold for L40S + 7B -----------------------------------
# L40S achieves ~55-65 tok/s on Llama-2-7B at batch=1.
# Under moderate batch load (4-8 seqs), per-request rate drops to 8-15 tok/s.
# Setting R_SLOW=30 would trigger migration on ALL requests under any load.
# R_SLOW=15 targets requests that are genuinely slow (sustained long generation).
export MIGRATION_R_SLOW="${MIGRATION_R_SLOW:-15}"
export MIGRATION_T_MIN="${MIGRATION_T_MIN:-200}"

# --- NCCL tuning for PCIe P2P (no NVLink on L40S) --------------------------
# NCCL auto-detects PCIe P2P; these flags help it choose the right path.
export NCCL_P2P_LEVEL=5            # Allow PCIe P2P across NUMA domains
export NCCL_SHM_DISABLE=0          # Keep shared memory for intra-socket pairs
export NCCL_ALGO=Ring              # Ring is optimal for 2-rank P2P comms
export NCCL_PROTO=Simple           # Simple protocol has lowest latency for small KV tensors

# --- Load system modules -----------------------------------------------------
module purge
module load anaconda3 2>/dev/null || module load Anaconda3 2>/dev/null || {
    echo "WARNING: Could not load anaconda3 module — relying on PATH from .bashrc"
}

# Activate the conda environment
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV" 2>/dev/null || {
    echo "WARNING: conda activate failed; trying direct PATH prepend"
    export PATH="$CONDA_ENV/bin:$PATH"
}

echo "Python:  $(which python3)"
echo "vllm:    $(which vllm 2>/dev/null || echo 'not found — check CONDA_ENV path')"
echo "Model:   $MODEL"
echo "MIGRATION_R_SLOW: $MIGRATION_R_SLOW tok/s"
echo ""

# --- Project directory -------------------------------------------------------
REPO_DIR="${REPO_DIR:-$HOME/vllm_inference_optimization}"
cd "$REPO_DIR"
echo "Working directory: $PWD"
echo ""

# =============================================================================
# Step 0 — Check that model weights are cached (pre-download with download_model.sh)
# =============================================================================
MODEL_CACHE="$HF_HOME/hub/models--$(echo $MODEL | tr '/' '--')"
if [[ ! -d "$MODEL_CACHE" ]]; then
    echo "------------------------------------------------------------"
    echo "STEP 0: Model not cached. Downloading $MODEL ..."
    echo "------------------------------------------------------------"
    python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='${MODEL}',
    cache_dir='${HF_HOME}/hub',
    token=os.environ.get('HF_TOKEN'),
    ignore_patterns=['*.msgpack','*.h5','flax_model*','tf_model*'],
)
print('Download complete.')
"
else
    echo "STEP 0: Model cache found at $MODEL_CACHE — skipping download."
fi

# =============================================================================
# Step 1 — Start vLLM serving stack (L40S-tuned startup)
# =============================================================================
echo "------------------------------------------------------------"
echo "STEP 1: Starting vLLM 4-GPU disaggregated serving stack (L40S)..."
echo "------------------------------------------------------------"

bash startup_l40s.sh
if [[ $? -ne 0 ]]; then
    echo "ERROR: startup_l40s.sh failed — aborting job."
    exit 1
fi

echo ""
echo "All servers ready. Waiting 30s for full model warm-up before load testing..."
sleep 30

# =============================================================================
# Step 2 — Run all evaluation conditions
# =============================================================================
export VLLM_HOST="localhost"
export VLLM_PORT="10099"
export VLLM_MODEL="${MODEL}"
# Pass SCRATCH_ROOT, CONDA_ENV, HF_HOME, HF_TOKEN, MODEL, HF_HUB_DISABLE_XET down
export SCRATCH_ROOT CONDA_ENV HF_HOME HF_TOKEN MODEL HF_HUB_DISABLE_XET
export MIGRATION_R_SLOW MIGRATION_T_MIN
export PATH

echo "------------------------------------------------------------"
echo "STEP 2: Running evaluation conditions via run_experiments.sh"
echo "------------------------------------------------------------"
echo ""

bash scripts/run_experiments.sh 2>&1 | tee "logs/experiments_l40s_${SLURM_JOB_ID}.log"
EXPERIMENTS_EXIT=${PIPESTATUS[0]}
if [[ $EXPERIMENTS_EXIT -ne 0 ]]; then
    echo "WARNING: run_experiments.sh returned exit code ${EXPERIMENTS_EXIT}"
    echo "  Some conditions may have partial results — continuing to analysis."
fi

# =============================================================================
# Step 3 — Analyze results and generate plots
# =============================================================================
echo "------------------------------------------------------------"
echo "STEP 3: Analyzing results and generating plots"
echo "------------------------------------------------------------"
echo ""

python3 scripts/analyze_results.py \
    --results-dir results \
    --plots-dir    plots/l40s \
    --proxy-log    proxy.log \
    2>&1 | tee "logs/analysis_l40s_${SLURM_JOB_ID}.log" || {
    echo "WARNING: Analysis script failed or produced partial output."
}

# =============================================================================
# Step 4 — Cleanup
# =============================================================================
echo "------------------------------------------------------------"
echo "STEP 4: Cleaning up background servers"
echo "------------------------------------------------------------"
bash cleanup.sh || true

echo ""
echo "========================================================"
echo "Job complete: $(date)"
echo "Results:  results/"
echo "Plots:    plots/l40s/"
echo "Logs:     logs/slurm_l40s_${SLURM_JOB_ID}.out"
echo "========================================================"
