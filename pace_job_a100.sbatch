#!/bin/bash
# =============================================================================
# SLURM Job Script — vLLM Straggler-Aware Migration (4× A100 80 GB)
# =============================================================================
# Tested target: 4× NVIDIA A100 80 GB SXM (Ampere, NVLink preferred)
#                Falls back to PCIe A100 — set NCCL_P2P_LEVEL=5 below if needed.
#
# BEFORE SUBMITTING — edit the four lines marked with  ← CONFIGURE
# =============================================================================

# --- Resource allocation ------------------------------------------------------
#SBATCH --job-name=vllm-migration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=02:00:00

# --- Partition / account  ← CONFIGURE ----------------------------------------
#SBATCH --partition=YOUR_GPU_PARTITION      # e.g. gpu, gpu-a100, ai4science
#SBATCH --account=YOUR_ACCOUNT             # your allocation / project account
##SBATCH --qos=YOUR_QOS                    # uncomment if your cluster needs a QOS

# --- Output / logging ---------------------------------------------------------
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=YOUR_EMAIL@example.com   # uncomment to get email alerts

# =============================================================================
# Environment setup
# =============================================================================

set -euo pipefail

echo "========================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "========================================================"

mkdir -p logs

# ← CONFIGURE: path to your scratch / work filesystem
export SCRATCH_ROOT="${SCRATCH_ROOT:-${SCRATCH:-$HOME/scratch}}"

# Path to the conda environment with vllm==0.11.0 installed (see RUNNING.md)
export CONDA_ENV="${CONDA_ENV:-$SCRATCH_ROOT/vllm_env}"

# HuggingFace cache — keep on scratch to avoid filling home quota
export HF_HOME="${SCRATCH_ROOT}/huggingface_cache"

# ← CONFIGURE: your HuggingFace token (needed for Llama-2 gated repo)
# Prefer setting HF_TOKEN in your ~/.bashrc rather than hardcoding here.
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN not set. Set it in ~/.bashrc or pass via sbatch --export."
fi

export MODEL="${MODEL:-meta-llama/Llama-2-13b-hf}"
export HF_HUB_DISABLE_XET=1   # prevent xet/CAS backend OOM during model load

# Migration threshold (tokens/s below which a request is considered a straggler)
export MIGRATION_R_SLOW="${MIGRATION_R_SLOW:-30}"

# --- Module loading -----------------------------------------------------------
# Uncomment / adapt to match your cluster's module system.
# Common patterns:
#   module load cuda/12.1  python/3.10  (generic HPC)
#   module load anaconda3               (Anaconda-based clusters)
module purge
module load anaconda3 2>/dev/null || module load Anaconda3 2>/dev/null || {
    echo "WARNING: Could not load anaconda3 module — relying on PATH from .bashrc"
}

source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV" 2>/dev/null || {
    echo "WARNING: conda activate failed; prepending PATH directly"
    export PATH="$CONDA_ENV/bin:$PATH"
}

echo "Python:  $(which python3)"
echo "vllm:    $(which vllm 2>/dev/null || echo 'not found — check CONDA_ENV')"
echo ""

# Path to the cloned repository
REPO_DIR="${REPO_DIR:-$HOME/vllm_inference_optimization}"
cd "$REPO_DIR"
echo "Working directory: $PWD"
echo ""

# Per-job directories — prevents concurrent jobs from clobbering each other
export LOG_DIR="logs/job_${SLURM_JOB_ID}"
export PROXY_LOG="$LOG_DIR/proxy.log"
mkdir -p "$LOG_DIR"

# =============================================================================
# Step 1 — Start vLLM serving stack
# =============================================================================
echo "------------------------------------------------------------"
echo "STEP 1: Starting vLLM 4-GPU disaggregated serving stack..."
echo "------------------------------------------------------------"

bash startup_a100.sh
if [[ $? -ne 0 ]]; then
    echo "ERROR: startup_a100.sh failed — aborting."
    exit 1
fi

echo ""
echo "All servers ready. Waiting 30s for full model warm-up..."
sleep 30

# =============================================================================
# Step 2 — Run all evaluation conditions
# =============================================================================
export VLLM_HOST="localhost"
export VLLM_PORT="10099"
export VLLM_MODEL="${MODEL}"

echo "------------------------------------------------------------"
echo "STEP 2: Running evaluation conditions"
echo "------------------------------------------------------------"
echo ""

bash scripts/run_experiments.sh 2>&1 | tee "logs/experiments_${SLURM_JOB_ID}.log"
EXPERIMENTS_EXIT=${PIPESTATUS[0]}
if [[ $EXPERIMENTS_EXIT -ne 0 ]]; then
    echo "WARNING: run_experiments.sh returned exit code ${EXPERIMENTS_EXIT}"
    echo "  Some conditions may have partial results — continuing to analysis."
fi

# =============================================================================
# Step 3 — Analyze results
# =============================================================================
echo "------------------------------------------------------------"
echo "STEP 3: Analyzing results and generating plots"
echo "------------------------------------------------------------"
echo ""

python scripts/analyze_results.py \
    --results-dir results \
    --plots-dir   plots \
    --proxy-log   "$PROXY_LOG" \
    --no-prometheus \
    2>&1 | tee "logs/analysis_${SLURM_JOB_ID}.log"

echo "Plots saved to: plots/"

# =============================================================================
# Step 4 — Cleanup
# =============================================================================
echo "------------------------------------------------------------"
echo "STEP 4: Cleaning up server processes..."
echo "------------------------------------------------------------"
bash cleanup.sh

echo ""
echo "========================================================"
echo "Job complete: $(date)"
echo "Results : $PWD/results/"
echo "Plots   : $PWD/plots/"
echo "========================================================"
