#!/bin/bash
# =============================================================================
# PACE SLURM Job Script — vLLM Straggler-Aware Migration Experiments
# =============================================================================
# Submits a single-node, 4-GPU job that:
#   1. Starts the 4-GPU disaggregated serving stack (startup.sh)
#   2. Runs the full evaluation suite (all 5 workload scripts)
#   3. Cleans up
#
# Submit with:
#   sbatch pace_job.sbatch
#
# To run only specific workloads, edit the "Run Experiments" section below.
# =============================================================================

# --- Resource allocation -------------------------------------------------------
#SBATCH --job-name=vllm-migration
#SBATCH --nodes=1                      # Must be 1: NVSwitch P2P KV transfers require same node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64             # 4 vLLM servers × ~12 cores + proxy + headroom
#SBATCH --mem=256G
#SBATCH --gres=gpu:H100:4              # Four H100 80GB (same node, NVSwitch)
#                                      # Fallback: gpu:A100:4  (A100 80GB SXM4)
#SBATCH --time=06:00:00                # 6h: allow for model load (~15 min) + all workloads

# --- Partition / account -------------------------------------------------------
# Uncomment and set the correct values for your PACE allocation.
# ICE cluster (H100 nodes):   --cluster=ice  --partition=gpu    (or gpu-h100)
# Phoenix cluster (A100 nodes): no --cluster flag, --partition=gpu-a100-p3
#
#SBATCH --partition=gpu-h100           # ← adjust for your cluster
##SBATCH --cluster=ice                 # ← uncomment if submitting to ICE
##SBATCH --account=YOUR_ACCOUNT        # ← uncomment and fill in if required

# --- Output / logging ----------------------------------------------------------
#SBATCH --output=logs/slurm_%j.out    # stdout goes here
#SBATCH --error=logs/slurm_%j.err     # stderr goes here
#SBATCH --mail-type=END,FAIL           # get emailed on completion or failure
##SBATCH --mail-user=YOUR_EMAIL@gatech.edu

# =============================================================================
# Environment setup
# =============================================================================

set -euo pipefail

echo "========================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "========================================================"

# Create log directory (output/error dirs must exist before job starts, but just in case)
mkdir -p logs

# --- Paths -------------------------------------------------------------------
# PACE scratch is $SCRATCH or /storage/scratch1/$USER
# Adjust SCRATCH_ROOT if your scratch path differs.
export SCRATCH_ROOT="${SCRATCH:-/storage/scratch1/$USER}"
export CONDA_ENV="${SCRATCH_ROOT}/envs/sysml_research4"

# HuggingFace credential (set HF_TOKEN in your ~/.bashrc or pass via sbatch --export)
# The model meta-llama/Llama-2-13b-hf requires a HF token with gated-model access.
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN not set. Model download will fail if weights are not cached."
    echo "  Set it in ~/.bashrc:  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx"
    echo "  Or pass at submit:    sbatch --export=ALL,HF_TOKEN=hf_xxx pace_job.sbatch"
fi
export HF_HOME="${SCRATCH_ROOT}/huggingface_cache"

# --- Load system modules -----------------------------------------------------
# PACE provides Anaconda via the module system. Adjust the module name if needed.
# Common names: Anaconda3/2022.05  Anaconda3/2023.09-0  anaconda3
module purge
module load anaconda3 2>/dev/null || module load Anaconda3 2>/dev/null || {
    echo "WARNING: Could not load anaconda3 module — relying on PATH from .bashrc"
}

# Activate the conda environment
# (startup.sh also sets PATH, but doing it here ensures sbatch login shell picks it up too)
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV" 2>/dev/null || {
    echo "WARNING: conda activate failed; trying direct PATH prepend"
    export PATH="$CONDA_ENV/bin:$PATH"
}

echo "Python:  $(which python3)"
echo "vllm:    $(which vllm 2>/dev/null || echo 'not found — check CONDA_ENV path')"
echo ""

# --- Project directory -------------------------------------------------------
# Adjust if you cloned the repo somewhere other than $HOME
REPO_DIR="${REPO_DIR:-$HOME/vllm_inference_optimization}"
cd "$REPO_DIR"
echo "Working directory: $PWD"
echo ""

# =============================================================================
# Step 1 — Start vLLM serving stack
# =============================================================================
echo "------------------------------------------------------------"
echo "STEP 1: Starting vLLM 4-GPU disaggregated serving stack..."
echo "------------------------------------------------------------"

# startup.sh runs healthchecks and exits 0 once all servers are listening.
# The 5 server processes remain running in the background after startup.sh exits.
bash startup.sh
if [[ $? -ne 0 ]]; then
    echo "ERROR: startup.sh failed — aborting job."
    exit 1
fi

echo ""
echo "All servers ready. Waiting 30s for full model warm-up before load testing..."
sleep 30

# =============================================================================
# Step 2 — Run all evaluation conditions
# =============================================================================
# run_experiments.sh orchestrates all 5 conditions (PROPOSAL.md §6):
#
#   Phase 1 – migration DISABLED (T_MIN=999999):
#     no_straggler, mixed, straggler, stress_straggler
#
#   Phase 2 – migration ENABLED (T_MIN=200):
#     migration_straggler
#
# Results land in results/<condition>/ as vllm bench JSON files.
# The proxy is restarted between phases; vLLM servers keep running throughout.
# =============================================================================

export VLLM_HOST="localhost"
export VLLM_PORT="10099"
export VLLM_MODEL="$MODEL"

echo "------------------------------------------------------------"
echo "STEP 2: Running evaluation conditions via run_experiments.sh"
echo "------------------------------------------------------------"
echo ""

bash scripts/run_experiments.sh 2>&1 | tee "logs/experiments_${SLURM_JOB_ID}.log"
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

python scripts/analyze_results.py \
    --results-dir results \
    --plots-dir    plots \
    --proxy-log    proxy.log \
    --no-prometheus \
    2>&1 | tee "logs/analysis_${SLURM_JOB_ID}.log"

echo ""
echo "Plots and tables saved to: plots/"

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
