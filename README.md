# Straggler-Aware Dynamic Migration for Disaggregated LLM Inference

A proxy-mediated straggler detection and live migration system for vLLM disaggregated prefill-decode (PD) serving. The system identifies slow "straggler" requests mid-execution, evicts them from a fast-lane GPU, and transparently re-admits them to a dedicated slow-lane GPU — restoring near-isolated performance for all co-running short requests without any modifications to the vLLM engine.

---

## The Problem

In autoregressive decoding, all active sequences in a batch advance together — one token per forward pass. A single request generating 3,000 tokens does not just consume its own time; it slows every other request in the batch for its entire duration. This causes:

- Elevated inter-token latency (ITL) for short requests
- Head-of-line blocking at the GPU batch level
- Unpredictable tail latency that worsens with straggler rate
- KV cache pressure as long-running sequences hold pages for extended periods

Admission-time routing on `max_tokens` is insufficient because users rarely declare output length — the model's natural EOS token is the real terminator. A request with `max_tokens=4096` may complete at 60 tokens or 3,500. Runtime detection is the only reliable signal.

---

## Architecture

```
                        Client
                          │
                          ▼
              ┌─────────────────────┐
              │   Proxy (port 10099) │
              │   Token tracking     │
              │   Migration logic    │
              └──────────┬──────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
   ┌─────────────────┐       ┌─────────────────┐
   │   Fast Lane     │       │   Slow Lane     │
   │   (FP16)        │       │   (BF16)        │
   │                 │       │                 │
   │ GPU 0 Prefill   │       │ GPU 2 Prefill   │
   │ port 20098      │       │ port 20096      │
   │ ZMQ  21098      │       │ ZMQ  21096      │
   │                 │       │                 │
   │ GPU 1 Decode    │       │ GPU 3 Decode    │
   │ port 20099      │       │ port 20097      │
   │ ZMQ  22099      │       │ ZMQ  22097      │
   └─────────────────┘       └─────────────────┘
   ZMQ discovery: 30099       ZMQ discovery: 30097
```

All client traffic enters the proxy on port 10099. Every request starts on the **fast lane**. The proxy tracks token output rate per request, and when a straggler is detected, transparently migrates it to the **slow lane** while continuing to stream tokens to the client.

### Lane Design

| Property | Fast Lane | Slow Lane |
|---|---|---|
| GPUs | 0 (prefill), 1 (decode) | 2 (prefill), 3 (decode) |
| Precision | FP16 | BF16 |
| max-num-seqs decode | 32 | 32 |
| GPU memory utilization | prefill 0.85, decode 0.6 | prefill 0.7, decode 0.7 |
| Purpose | All incoming requests | Migrated stragglers only |

**Why two separate ZMQ ports?** vLLM's `P2pNcclConnector` registers servers by type (`"P"` / `"D"`) only — there is no lane metadata in the registration message. Running two separate ZMQ discovery ports (30099 for fast, 30097 for slow) is the only way to prevent a fast-prefill from being accidentally paired with a slow-decode (FP16 ↔ BF16 KV mismatch → corrupt output).

---

## Straggler Detection

The proxy evaluates a two-condition rule every 16 generated tokens:

$$\text{migrate} \iff \underbrace{N \geq T_{min}}_{\text{evidence gate}} \;\wedge\; \underbrace{\frac{N}{t_{elapsed}} < R_{slow}}_{\text{congestion signal}}$$

| Parameter | Default | Env variable |
|---|---|---|
| $T_{min}$ — min tokens before evaluation | 200 | `MIGRATION_T_MIN` |
| $R_{slow}$ — rate threshold (tok/s) | 30.0 | `MIGRATION_R_SLOW` |
| Evaluation interval | 16 tokens | (hardcoded) |

The evidence gate prevents false positives on requests that are just slow to start. The congestion signal is the real diagnostic: under straggler-induced batch drag, per-request token throughput drops measurably below healthy rates (~50–200 tok/s).

---

## Migration Protocol

When detection fires for a request that has generated $N$ tokens:

1. **Evict from fast lane.** The proxy closes the fast-decode connection; vLLM drops the sequence and frees its KV pages.

2. **Build re-admission payload.** Using raw token IDs (no detokenize/retokenize roundtrip):
   ```
   prompt_token_ids = original_prompt_ids + generated_ids[0..N]
   max_tokens = original_max_tokens - N
   ```

3. **Slow-lane prefill.** The proxy submits `prompt_token_ids` to the slow-lane prefill server, which processes all $L + N$ tokens and transfers KV state to the slow-lane decode GPU via P2pNcclConnector.

4. **Resume client stream.** The proxy forwards slow-lane decode output starting from token $N+1$. The client observes a brief pause (slow-lane prefill latency, typically 100–300ms for Llama-2-13B at 200–400 context tokens) then the stream resumes.

---

## Repository Structure

```
.
├── disagg_proxy_migration.py      # Migration-aware proxy (main entry point)
├── disagg_proxy_p2p_nccl_xpyd.py # Baseline proxy — no migration (for comparison)
├── startup.sh                     # Launch 4-GPU system (fast + slow lane)
├── startup_baseline.sh            # Launch 1P1D baseline (fast lane only)
├── cleanup.sh                     # Kill all servers and free all ports
├── requirements.txt               # Python dependencies (vllm==0.11.0)
├── PROPOSAL.md                    # Full system design and evaluation plan
│
├── scripts/
│   ├── mixed_vllm_load.sh              # Core load dispatcher (used by all below)
│   ├── no_straggler_vllm_load.sh       # Baseline: short requests only
│   ├── straggler_vllm_load.sh          # Straggler injection (no migration)
│   ├── stress_straggler_vllm_load.sh   # Heavy straggler stress test
│   └── migration_straggler_vllm_load.sh # Same as straggler, hits migration proxy
│
└── prometheus_grafana/
    ├── docker-compose.yaml
    ├── prometheus_pd_host.yaml          # Scrape config for all 4 vLLM servers
    ├── grafana_pd_revised_modified_cleaned_labels.json
    └── start_monitoring.sh
```

---

## Dependencies

- Python 3.10+
- `vllm==0.11.0`
- `quart==0.20.0`
- `aiohttp==3.13.2`
- `pyzmq==27.1.0`
- `msgpack==1.1.2`
- `transformers==4.57.1`
- 4× NVIDIA GPU (A100/H100 class recommended for Llama-2-13B)
- Model: `meta-llama/Llama-2-13b-hf` (accessible via Hugging Face)

---

## Quickstart

### Environment

```bash
export SCRATCH_ROOT=/path/to/your/scratch
export CONDA_ENV=$SCRATCH_ROOT/envs/sysml_research4
export HF_HOME=$SCRATCH_ROOT/huggingface_cache
export MODEL=meta-llama/Llama-2-13b-hf
```

### Start the 4-GPU Migration System

```bash
./startup.sh
```

Launches in order: migration proxy → fast-prefill (GPU 0) → fast-decode (GPU 1) → slow-prefill (GPU 2) → slow-decode (GPU 3). Healthchecks all five. If any server fails, all processes are killed automatically.

### Start the 1P1D Baseline (for comparison)

```bash
./startup_baseline.sh
```

Launches baseline proxy + one prefill-decode pair (GPUs 0+1, FP16).

### Quick Smoke Test

```bash
curl -X POST http://localhost:10099/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "meta-llama/Llama-2-13b-hf", "prompt": "The capital of France is", "max_tokens": 20, "temperature": 0.0}'
```

### Stop Everything

```bash
./cleanup.sh
```

---

## Environment Variables

### `startup.sh` / `startup_baseline.sh`

| Variable | Default | Description |
|---|---|---|
| `SCRATCH_ROOT` | `$HOME/scratch` | HPC scratch directory |
| `CONDA_ENV` | `$SCRATCH_ROOT/envs/sysml_research4` | Conda environment path |
| `MODEL` | `meta-llama/Llama-2-13b-hf` | Model name |
| `PROXY_PORT` | `30099` | ZMQ discovery port — fast lane |
| `SLOW_PROXY_PORT` | `30097` | ZMQ discovery port — slow lane |
| `PROXY_HTTP_PORT` | `10099` | Client-facing HTTP port |
| `HF_HOME` / `HF_CACHE_ROOT` | `$SCRATCH_ROOT/huggingface_cache` | HuggingFace model cache |

### `disagg_proxy_migration.py`

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `meta-llama/Llama-2-13b-hf` | Tokenizer to load |
| `PROXY_PORT` | `30099` | ZMQ discovery port — fast lane |
| `SLOW_PROXY_PORT` | `30097` | ZMQ discovery port — slow lane |
| `PROXY_HTTP_PORT` | `10099` | HTTP listen port |
| `MIGRATION_T_MIN` | `200` | Min tokens before migration evaluation |
| `MIGRATION_R_SLOW` | `30.0` | Congestion threshold (tok/s) |
| `OPENAI_API_KEY` | *(empty)* | Forwarded in Authorization header |

### Load Scripts

| Variable | Description |
|---|---|
| `VLLM_HOST` | Target host (default: `localhost`) |
| `VLLM_PORT` | Target port (default: `10099`) |
| `VLLM_MODEL` | Model name passed to `vllm bench serve` |

---

## Evaluation Workloads

All workloads use `vllm bench serve` and emit structured metrics (TTFT, TPOT, ITL, E2EL at P90/P95/P99).

| Script | Small stream | Long stream | Purpose |
|---|---|---|---|
| `no_straggler_vllm_load.sh` | RPS=15, out=64, N=2000 | 1 token (negligible) | Clean baseline — no stragglers |
| `straggler_vllm_load.sh` | RPS=15, out=64, N=2000 | RPS=0.5, out=3072, N=40 | Realistic straggler injection |
| `stress_straggler_vllm_load.sh` | RPS=15, out=64, N=2000 | RPS=1.5, out=3072, N=500 | Heavy straggler stress |
| `migration_straggler_vllm_load.sh` | RPS=15, out=64, N=2000 | RPS=0.5, out=3072, N=40 | Same as straggler → migration proxy |

The key comparison is **straggler vs migration_straggler**: same load, different proxy. The primary metric is P99 ITL on the fast-lane decode GPU.

---

## Monitoring

Start Prometheus + Grafana:

```bash
cd prometheus_grafana
./start_monitoring.sh
```

The `prometheus_pd_host.yaml` scrapes all four vLLM servers with `lane: fast/slow` labels. Key metrics:

| Metric | What to watch |
|---|---|
| `vllm:inter_token_latency_seconds` | P99 drops to near no-straggler baseline with migration |
| `vllm:e2e_request_latency_seconds` | P99 for short requests recovers |
| `vllm:kv_cache_usage_perc` | Fast lane shows lower peak usage after migration |
| `vllm:num_preemptions_total` | Should drop to ~0 with migration active |

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Migration mechanism | Re-admission via `prompt_token_ids` | Works within vLLM's public API; no engine modifications |
| Slow-lane precision | BF16 | Negligible quality difference on Llama-2; same VRAM footprint as FP16 |
| Two ZMQ discovery ports | 30099 (fast), 30097 (slow) | Prevents FP16↔BF16 KV cross-pairing without engine changes |
| Token buffering | Raw token IDs | Exact, no tokenizer roundtrip error; works with any tokenizer |
| Detection layer | Proxy | Intermediates all traffic; owns the complete token stream |
| Detection granularity | Every 16 tokens | Low overhead; aligned with decode batch rhythm |
| Slow-lane warm state | Pre-warmed at startup | Model load takes minutes; cannot start on-demand |

## Known Limitations

- **Re-admission cost grows with detection delay.** Migrating at $N=200$ costs ~1.5× the prefill of the original prompt. Later detection reduces false positives but raises recomputation cost.
- **Stream pause is visible.** Clients on streaming APIs observe a brief stall during slow-lane prefill (~100–300ms for Llama-2-13B). Acceptable for batch workloads; would need masking for interactive SLAs.
- **BF16/FP16 output divergence.** Tokens 1..N are generated in FP16 and tokens N+1..end in BF16. In practice imperceptible on Llama-2 but worth noting for reproducibility.
- **Slow lane always running.** Fixed cost of 2 GPUs allocated at all times, regardless of actual straggler rate.
- **Chat completions: no migration.** The `/v1/chat/completions` endpoint is forwarded to the fast lane only; `prompt_token_ids` is a completions-API extension not supported by the chat format.
