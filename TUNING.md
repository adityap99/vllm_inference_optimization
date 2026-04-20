# Hardware / Model / Workload Tuning Guide

This document answers: **"Can I run on fewer/different resources with a smaller model?"**  
and provides a complete tuning reference for every dimension of the experiment.

---

## 1. Current Resource State (as of survey)

| Node type | Count | GPUs/node | VRAM | Interconnect | State |
|-----------|-------|-----------|------|--------------|-------|
| **H100 SXM** | 5 nodes | 8× H100 80 GB | 80 GB | NVLink 900 GB/s | `mixed-` (queued) |
| **H200 SXM** | 4 nodes | 8× H200 80 GB | 80 GB | NVLink 900 GB/s | `mixed-` / drained |
| **L40S**   | 4 nodes | 8× L40S 48 GB | 48 GB | PCIe 4.0 ~32 GB/s | `mixed-` (all 8 GPUs currently allocated) |
| **A100**   | 4 nodes | **2× A100 40 GB** | 40 GB | PCIe | Not enough GPUs |
| **V100**   | many  | 1–4× V100 16 GB | 16 GB | PCIe | Too little VRAM |
| **RTX 6000** | 2 nodes | 4× Quadro RTX 6000 24 GB | 24 GB | PCIe | 1 GPU free on one node; other draining |

**Right now**: no node in `ice-gpu`/`coc-gpu` has 4+ free GPUs.  
Both your queued H100 jobs (4920576, 5020644) are pending and will run when slots open.

---

## 2. Can I Use Fewer GPUs? No — Architecture Requires 4

The 4-GPU disaggregated design is non-negotiable:

```
GPU 0: fast-prefill  |   GPU 2: slow-prefill
GPU 1: fast-decode   |   GPU 3: slow-decode
```

`P2pNcclConnector` creates **2-rank NCCL communicators** (prefill↔decode) per lane.  
Each rank must be a separate GPU device. You need exactly 4 GPUs on **one node** (same-node requirement for `cuda:{local_rank}` device assignment).

**There is no 2-GPU version** of this experiment — you'd lose the fast/slow lane comparison.

---

## 3. Does P2pNcclConnector Require NVLink?

**No.** Source analysis of `p2p_nccl_engine.py` confirms:

- Rendezvous is TCP-based (ZMQ sockets over `tcp://hostname:port`)
- NCCL selects the best available transport **automatically**:
  1. **NVLink** (H100/A100 same node) — 600–900 GB/s ← fastest
  2. **PCIe P2P** (L40S, RTX 6000 same node) — 32–64 GB/s ← still fast
  3. **Shared memory** (intra-socket) — ~200 GB/s
  4. **TCP/IP** (fallback, never needed here) — ~10 GB/s

KV transfer at migration (T_MIN=200 tokens):

| Model | KV per token | 200-token transfer | PCIe time | NVLink time |
|-------|-------------|---------------------|-----------|-------------|
| Llama-2-7B  | 0.50 MB | 100 MB | **3.1 ms** | 0.17 ms |
| Llama-2-13B | 0.78 MB | 156 MB | **4.9 ms** | 0.27 ms |

Both are negligible vs. the decode time saved by migrating away from the straggler. **L40S works.**

---

## 4. Model Options: Fit Analysis

| Model | Weights (FP16) | HF Token? | V100 16 GB | A100 40 GB | L40S 48 GB | H100 80 GB |
|-------|---------------|-----------|------------|------------|------------|------------|
| OPT-125M    | 0.25 GB | No  | ✅ | ✅ | ✅ | ✅ |
| OPT-1.3B    | 2.6 GB  | No  | ✅ | ✅ | ✅ | ✅ |
| OPT-6.7B    | 13.4 GB | No  | ❌ OOM | ✅ | ✅ | ✅ |
| **Llama-2-7B**  | **14.0 GB** | **Yes** | ❌ | ✅ | **✅ ← recommended** | ✅ |
| Llama-3.1-8B | 16.0 GB | Yes | ❌ | ✅ | ✅ | ✅ |
| Mistral-7B  | 14.0 GB | No  | ❌ | ✅ | ✅ | ✅ |
| **Llama-2-13B** | **26.0 GB** | **Yes** | ❌ | ❌ | ✅ (tight) | **✅ ← original** |

**Best alternative: Llama-2-7B on L40S**  
- Requires same HF token you already have  
- Same architecture family as 13B — results are directly comparable  
- Already downloadable with: `HF_TOKEN=hf_xxx python3 -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-2-7b-hf', cache_dir='$SCRATCH/huggingface_cache/hub')"`

**Token-free alternative: Mistral-7B**  
- Same parameter count and similar architecture (GQA with 8 KV heads)  
- No gating — download without HF token  
- Warning: fewer KV heads means different KV transfer sizes; experiment still valid

---

## 5. GPU Memory Tuning

### L40S + Llama-2-7B (48 GB GPU, 14 GB weights)

```
Available VRAM budget = 48 GB × utilization
Overhead (CUDA ctx + framework) ≈ 2 GB
KV per token (7B) = 32 layers × 2 × 32 heads × 128 dim × 2 B = 524,288 B ≈ 0.5 MB
```

| Role | util | Budget | After weights | KV available | kv_buffer | Net KV | Max seqs @4096 tok |
|------|------|--------|--------------|-------------|-----------|--------|-------------------|
| fast-prefill | 0.85 | 40.8 GB | 24.8 GB | 24.8 GB | 1 GB | 23.8 GB | **11** → cap 32 |
| fast-decode  | 0.80 | 38.4 GB | 22.4 GB | 22.4 GB | 4 GB | 18.4 GB | **8** → cap 16 |
| slow-prefill | 0.80 | 38.4 GB | 22.4 GB | 22.4 GB | 1 GB | 21.4 GB | **10** → cap 32 |
| slow-decode  | 0.80 | 38.4 GB | 22.4 GB | 22.4 GB | 4 GB | 18.4 GB | **8** → cap 16 |

### H100 + Llama-2-13B (80 GB GPU, 26 GB weights) — reference

| Role | util | Budget | After weights | kv_buffer | Net KV | Max seqs @4096 tok |
|------|------|--------|--------------|-----------|--------|-------------------|
| fast-prefill | 0.85 | 68 GB | 40 GB | 1 GB | 39 GB | **18** → cap 64 |
| fast-decode  | 0.70 | 56 GB | 28 GB | 4 GB | 24 GB | **11** → cap 32 |
| slow-prefill | 0.70 | 56 GB | 28 GB | 1 GB | 27 GB | **12** → cap 64 |
| slow-decode  | 0.70 | 56 GB | 28 GB | 4 GB | 24 GB | **11** → cap 32 |

---

## 6. Workload Tuning for L40S + 7B

### Throughput reference

```
L40S memory bandwidth:  864 GB/s
Llama-2-7B weights:     14 GB (FP16)
Peak decode, batch=1:   864 / 14 ≈ 62 tok/s
Under load (batch=4):   62 / 4 ≈ 15 tok/s per request
Under load (batch=8):   62 / 8 ≈  8 tok/s per request
```

### Critical: Migration threshold (MIGRATION_R_SLOW)

The proxy flags a request as a straggler when its decode rate drops below `R_SLOW`.

- **H100 + 13B**: `R_SLOW=30` tok/s — correct because H100 easily exceeds 80 tok/s at batch=1
- **L40S + 7B**: `R_SLOW=15` tok/s — must be lowered; otherwise ANY request under batch=4 triggers migration

`startup_l40s.sh` sets `MIGRATION_R_SLOW=15`. You can override:
```bash
MIGRATION_R_SLOW=10 sbatch pace_job_l40s.sbatch
```

### Workload script parameters

The workload scripts (straggler_vllm_load.sh etc.) do not embed model-specific RPS values —  
they use concurrency/num-prompts. These are valid as-is for 7B because:
- **15 RPS for short requests**: Both H100/L40S can sustain this (headroom remains)
- **3072-token long requests**: Still creates straggler effect on decode GPU
- **T_MIN=200 tokens**: ~3.2 s on L40S at 62 tok/s → still meaningful threshold

No workload script changes needed for L40S.

---

## 7. NCCL Tuning for PCIe (L40S)

```bash
export NCCL_P2P_LEVEL=5          # enable PCIe P2P across NUMA domains
export NCCL_SHM_DISABLE=0        # keep shared-memory for intra-socket pairs
export NCCL_ALGO=Ring            # optimal for 2-rank NCCL comms
export NCCL_PROTO=Simple         # lowest latency; LL128 adds overhead for small tensors
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

These are set in `startup_l40s.sh` and `pace_job_l40s.sbatch`. No change needed.

`nccl_num_channels=8` (reduced from 16 in H100 run) — PCIe bandwidth does not benefit  
from 16+ channels. 8 is optimal.

---

## 8. Scientific Validity: 7B vs 13B

The experiment measures:
1. **Time-to-First-Token (TTFT)** under straggler pressure  
2. **Per-request p99 latency** in fast lane vs slow lane  
3. **Throughput degradation** caused by long requests blocking decode GPU  
4. **Migration effectiveness**: how quickly fast lane recovers after migration

All four are **ratios** between conditions (straggler vs no-straggler; migration vs no-migration).  
These ratios are model-architecture-independent. A 7B on L40S shows the same qualitative  
straggler blocking behavior and the same migration relief as a 13B on H100.

The absolute throughput numbers (tok/s) will differ but the paper/thesis conclusion —  
"migration reduces p99 latency by X% under straggler load" — holds at both scales.

---

## 9. Recommended Action Plan

### Immediate (today)

1. **Cancel the duplicate job**: `scancel 4920576` (4920576 is older/lower priority;  
   5020644 is the one actually estimated to run first on `atl1-1-03-012-28-0`)

2. **Download Llama-2-7B** while you wait (reuses your HF token):
   ```bash
   HF_HOME=$SCRATCH/huggingface_cache python3 -c "
   from huggingface_hub import snapshot_download
   snapshot_download(
       'meta-llama/Llama-2-7b-hf',
       cache_dir='$SCRATCH/huggingface_cache/hub',
       ignore_patterns=['*.msgpack','*.h5','flax_model*','tf_model*'],
   )
   "
   ```
   Expected size: ~14 GB (fast, ~5 minutes)

3. **Submit the L40S job** to get into the coc-gpu queue (shorter wait):
   ```bash
   sbatch pace_job_l40s.sbatch
   ```
   L40S nodes (`atl1-1-03-004-{21,23,25,27}-0`) should free up as current jobs complete.

### If L40S nodes stay full

4. Submit to `coc-gpu` partition (16-hour limit) with same sbatch:
   ```bash
   # Edit pace_job_l40s.sbatch: change --partition=ice-gpu to --partition=coc-gpu
   sbatch pace_job_l40s.sbatch
   ```
   The `coc-gpu` queue for L40S is typically less congested than `ice-gpu`.

### Longer term

5. Keep `5020644` in queue — let it run on H100 when it eventually gets scheduled.  
   This gives you both a 7B/L40S run and a 13B/H100 run to compare across scales.

---

## 10. Quick-Reference: Which Job to Submit for What Goal

| Goal | GPU | Model | Sbatch | Queue |
|------|-----|-------|--------|-------|
| Full-scale paper results | H100 × 4 | Llama-2-13B | `pace_job.sbatch` | ice-gpu (days) |
| Faster iteration / validation | L40S × 4 | Llama-2-7B | `pace_job_l40s.sbatch` | ice-gpu or coc-gpu (hours) |
| Functional testing (no token) | L40S × 4 | Mistral-7B | `pace_job_l40s.sbatch` + `MODEL=mistralai/Mistral-7B-v0.1` | ice-gpu (hours) |
| Debug / CI sanity check | any × 4 | OPT-1.3B | `pace_job_l40s.sbatch` + `MODEL=facebook/opt-1.3b` | fastest queue |
