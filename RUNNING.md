# Running vLLM Straggler Migration Experiments on 4× A100 80 GB

This guide is self-contained. Follow the steps in order.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| NVIDIA A100 80 GB × 4 (single node) | SXM or PCIe |
| CUDA | 12.1+ |
| Python | 3.10 |
| vLLM | 0.11.0 |
| HuggingFace account with Llama-2 access | — |

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/adityap99/vllm_inference_optimization.git
cd vllm_inference_optimization
git checkout a100-external
```

---

## Step 2 — Create the conda environment

```bash
conda create -n vllm_env python=3.10 -y
conda activate vllm_env

pip install vllm==0.11.0
pip install quart aiohttp pyzmq msgpack transformers
```

> If your cluster has no internet on compute nodes, run this on a login node
> and note the full path to the env (e.g. `/scratch/yourname/vllm_env`).

---

## Step 3 — Download model weights (login node)

Llama-2-13b-hf is gated. You need a HuggingFace token with access granted at
https://huggingface.co/meta-llama/Llama-2-13b-hf

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx        # your token
export HF_HOME=/scratch/yourname/huggingface_cache
export HF_HUB_DISABLE_XET=1

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-2-13b-hf')
print('Done')
"
```

This downloads ~26 GB. Run once on a login node; compute nodes will use the cache.

---

## Step 4 — Configure the SLURM job script

Edit `pace_job_a100.sbatch`. The only lines you **must** change are marked
`← CONFIGURE`:

```bash
#SBATCH --partition=YOUR_GPU_PARTITION   # ← your cluster's GPU partition name
#SBATCH --account=YOUR_ACCOUNT           # ← your project/allocation account
```

Also verify these environment variables at the top of the script match your
cluster's paths:

```bash
export SCRATCH_ROOT="..."     # your scratch filesystem, e.g. /scratch/$USER
export CONDA_ENV="..."        # full path to the vllm_env created in Step 2
```

### PCIe A100 (no NVLink)
If your A100 nodes are PCIe (not SXM), uncomment these two lines in
`startup_a100.sh`:

```bash
# export NCCL_P2P_LEVEL=5
# export NCCL_SHM_DISABLE=0
```

---

## Step 5 — Submit the job

```bash
# From the repo root:
mkdir -p logs
sbatch pace_job_a100.sbatch
```

Monitor progress:
```bash
squeue -u $USER
tail -f logs/slurm_<JOBID>.out
```

Expected timeline on A100 80 GB:
- Model load + health checks: ~5–8 min
- All 5 experiment conditions: ~40–60 min
- Total walltime needed: ≤ 2 h

---

## Step 6 — Collect results

After the job completes:

```bash
ls results/          # per-condition vllm-bench JSON files
ls plots/            # auto-generated figures (ITL CDFs, migration timeline)
cat logs/slurm_<JOBID>.out   # full run log
```

The key output for the paper is **ITL (inter-token latency)** across conditions:

| Condition | What it shows |
|-----------|--------------|
| `no_straggler` | baseline ITL, no long requests |
| `mixed` | light mixed traffic |
| `straggler` | ITL degradation from long requests (no migration) |
| `stress_straggler` | heavy straggler pressure (no migration) |
| `migration_straggler` | same load as straggler, **migration active** |

---

## Troubleshooting

**vLLM server won't start**
```bash
tail -50 logs/job_<JOBID>/fast_prefill.log
```
Common causes: wrong `CONDA_ENV` path, model weights not downloaded, port already in use.

**`MODEL: unbound variable`**
Ensure you are on the `a100-external` branch. The bug was fixed here.

**OOM during startup**
Reduce `--gpu-memory-utilization` in `startup_a100.sh` (e.g. 0.70 → 0.65 for decode servers).

**KV transfer failures / NCCL errors**
For PCIe A100, uncomment the `NCCL_P2P_LEVEL=5` lines in `startup_a100.sh`.

**Proxy not ready after restart**
Increase `RESTART_WAIT` in `scripts/run_experiments.sh` (default 30 s → 60 s).

---

## File reference

| File | Purpose |
|------|---------|
| `pace_job_a100.sbatch` | SLURM job script — edit partition/account |
| `startup_a100.sh` | Launches 4 vLLM servers + proxy |
| `disagg_proxy_migration.py` | Migration-aware proxy (no changes needed) |
| `scripts/run_experiments.sh` | Orchestrates all 5 experiment conditions |
| `scripts/mixed_vllm_load.sh` | Core load generator |
| `cleanup.sh` | Kills all server processes |
| `scripts/analyze_results.py` | Generates plots from result JSONs |
