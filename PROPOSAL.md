# Straggler-Aware Dynamic Migration for Disaggregated LLM Inference

## Abstract

Production LLM inference systems serving autoregressive models face an inherent mixed-workload problem: the majority of requests are short and latency-sensitive, while a small fraction generate long outputs and occupy GPU decode capacity for extended periods. In disaggregated prefill-decode (PD) serving, the decode GPU is a shared batch processor — every active sequence advances one token per step, meaning a handful of long-output "straggler" requests directly inflate inter-token latency for all co-running requests. We propose a proxy-mediated straggler detection and dynamic migration system that identifies straggler requests mid-execution using runtime signals, evicts them from the fast-lane decode GPU, and re-admits them to a dedicated slow-lane PD pair, restoring the fast lane to near-isolated performance without modifications to the underlying inference engine.

---

## 1. Motivation

Modern LLM serving workloads are heterogeneous by nature. A chat system serves both one-sentence clarifications and multi-page document drafts through the same endpoint. A coding assistant handles both quick completions and full function generation. Operators cannot predict at admission time how long any given request will run — users do not declare output length, and the model's natural stopping behavior is determined by content, not by any parameter known in advance.

This creates a structural fairness and performance problem in autoregressive decoding. The standard vLLM continuous batching scheduler groups all active sequences into a single decode batch, advancing each by one token per forward pass. The wall-clock time per step is bounded by the largest request in the batch. A single request generating 3,000 tokens does not merely consume its own GPU time — it slows every other request in the batch for the duration of its execution.

The consequences are measurable:
- **Elevated inter-token latency (ITL)** for short requests, even when they are nearly complete
- **Head-of-line blocking** at the GPU batch level, not just the network or scheduler level
- **Unpredictable tail latency** that degrades as straggler rate increases
- **KV cache pressure** as long-running sequences hold pages for extended periods, reducing available slots for incoming requests

---

## 2. Problem Statement

**Why admission-time routing is insufficient.**
The natural first approach is to route requests with large `max_tokens` values to a dedicated slow lane at admission. This works correctly only when clients explicitly declare output length, which is not the common case in production. Real users rely on the model's EOS token, so `max_tokens` is either absent or set conservatively high as a safety cap. A request declared with `max_tokens=4096` may complete naturally at 60 tokens or may genuinely run to 3,500. Admission-time routing on this signal would misclassify the majority of requests.

**The core difficulty.**
Autoregressive decoding generates tokens one at a time, with no a priori knowledge of final output length. The system cannot determine at request start whether a sequence will become a straggler. Runtime signals — tokens generated so far, decoding rate, elapsed time in decode phase — are the only available evidence, and they only become informative after the request has been running long enough to reveal its trajectory.

**The key insight.**
Even imperfect mid-flight detection is sufficient. A straggler caught at step 200 of a 3,000-token generation still has 2,800 steps remaining. The fast-lane benefit of evicting it — restored ITL for all remaining short requests — far exceeds the cost of re-processing its first 200 tokens on the slow lane.

---

## 3. Proposed System

### 3.1 Architecture

The system deploys four GPUs organized as two independent prefill-decode pairs, coordinated by a single migration-aware proxy:

```
┌─────────────────────────────────────────────────────────┐
│                      Proxy (port 10099)                  │
│          Token stream tracking · Migration logic         │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
   ┌──────▼──────┐           ┌──────▼──────┐
   │  Fast Lane  │           │  Slow Lane  │
   │  PD Pair    │           │  PD Pair    │
   │             │           │             │
   │ GPU 0       │           │ GPU 2       │
   │ Prefill     │           │ Prefill     │
   │ FP16        │           │ BF16        │
   │ port 20098  │           │ port 20096  │
   │             │           │             │
   │ GPU 1       │           │ GPU 3       │
   │ Decode      │           │ Decode      │
   │ FP16        │           │ BF16        │
   │ port 20099  │           │ port 20097  │
   └─────────────┘           └─────────────┘
```

**Fast lane** serves all incoming requests. It is optimized for latency: FP16 precision, `max-num-seqs=32` on the decode GPU, low KV cache headroom to minimize queue depth.

**Slow lane** receives migrated stragglers exclusively. It runs BF16 precision (no perceptible quality difference for Llama-2, no additional model weight preparation required). It is pre-warmed at system startup and idles between migration events.

**Proxy** is the sole entry point for all client traffic. It forwards requests to the fast lane, tracks token stream progress per request, evaluates the migration condition, and — when triggered — executes the re-admission protocol transparently to the client.

### 3.2 Online Straggler Detection

The proxy maintains per-request state:

```
{
  request_id:     str,
  original_prompt_token_ids: List[int],
  generated_token_ids:       List[int],
  start_time:     float,
  last_token_time: float,
  migrated:       bool
}
```

The detection condition is evaluated every 16 generated tokens (aligned with decode batch boundaries):

$$\text{migrate} \iff \underbrace{N \geq T_{min}}_{\text{evidence gate}} \;\wedge\; \underbrace{\frac{N}{t_{elapsed}} < R_{slow}}_{\text{congestion signal}}$$

Where:
- $N$ = tokens generated so far
- $T_{min}$ = minimum token threshold before evaluation begins (default: **200 tokens**)
- $t_{elapsed}$ = wall-clock time since decode phase started
- $R_{slow}$ = minimum acceptable decoding rate (default: **30 tokens/sec**)

The evidence gate prevents false positives on requests that simply haven't reached their natural end yet. The congestion signal is the key observable: in a batch under straggler pressure, ITL rises and therefore tokens/sec per request falls measurably below healthy decode rates (~50–200 tok/s). A slow rate with high token count is the fingerprint of a genuine straggler causing batch drag.

### 3.3 Migration Protocol

When the detection condition fires for request $r$ having generated $N$ tokens:

1. **Evict from fast lane.** The proxy stops forwarding the fast-lane stream for request $r$ and signals the fast-lane decode server to drop the sequence. The fast-lane KV pages for $r$ are immediately freed.

2. **Build re-admission payload.** The new prompt is constructed from buffered token IDs:
   ```
   input_ids = original_prompt_token_ids + generated_token_ids[0..N]
   max_tokens = original_max_tokens - N
   ```
   Using raw token IDs directly avoids detokenization/retokenization roundtrips and is exact.

3. **Submit to slow lane.** The proxy POSTs the re-admission request to the slow-lane proxy endpoint. The slow-lane prefill GPU processes all `L + N` tokens and transfers its KV cache to the slow-lane decode GPU.

4. **Resume client stream.** The proxy forwards the slow-lane token stream to the client starting from token $N+1$. From the client's perspective, the token stream pauses briefly during slow-lane prefill and then resumes — the migration is invisible at the API level.

### 3.4 Migration Cost and Break-Even

Re-admission incurs a one-time recomputation cost: the slow-lane prefill must process $L + N$ tokens to reconstruct KV state. Let $C_{prefill}(L+N)$ denote this latency. Migration produces a per-step benefit on the fast lane: $\Delta_{ITL}$, the reduction in decode step time for remaining fast-lane requests after the straggler is evicted.

Migration is worthwhile when:

$$R_{remaining} \cdot \Delta_{ITL} > C_{prefill}(L + N)$$

For the default `T_min = 200` with a 128-token input prompt, the slow-lane prefill sees ~328 tokens — approximately 100–300ms on Llama-2-13B. With typical straggler lengths of 2,000–3,000 tokens, $R_{remaining} \approx 1800$–$2800$ at detection time. Empirical $\Delta_{ITL}$ values from this project's existing straggler experiments are on the order of tens to hundreds of milliseconds per step at high straggler injection rates. The break-even analysis favors migration decisively at `T_min = 200`, with substantial margin.

Detecting later (higher $T_{min}$) reduces false positives but increases recomputation cost and reduces the number of fast-lane steps that benefit. Detecting earlier increases false-positive risk. The threshold is a tunable parameter that can be calibrated from observed workload distributions.

---

## 4. Design Decisions and Rationale

| Decision | Choice | Rationale |
|---|---|---|
| Migration mechanism | Re-admission (not KV transfer) | Works within vLLM's public API; no engine modifications |
| Slow-lane precision | BF16 | Same VRAM as FP16; negligible quality difference on Llama-2; avoids FP8 calibration complexity |
| Detection layer | Proxy | Already intermediates all traffic; owns the full stream; natural place for per-request state |
| Token buffering | Raw token IDs | Exact, avoids tokenizer roundtrip, works with any tokenizer |
| Slow-lane warm state | Pre-warmed at startup | Model load takes minutes; cannot be started on-demand per migration event |
| Detection granularity | Every 16 tokens | Aligned with decode batch rhythm; low overhead; timely enough for the expected straggler lengths |

---

## 5. Expected Benefits

**For short requests (primary objective):**
- P99 ITL on the fast-lane decode GPU should approach the no-straggler baseline
- P95 E2E latency for short requests should recover to near-isolated performance
- KV cache pressure on the fast lane drops as straggler pages are freed

**For long requests (secondary, explicitly tolerable degradation):**
- Migration pause of 100–300ms while slow-lane prefill runs
- Subsequent decode on BF16 slow lane — completion time slightly longer per step
- This tradeoff is explicitly acceptable: the system optimizes the common case at the cost of edge-case throughput

**System-level:**
- Preemption rate on the fast lane drops to near zero (straggler KV exhaustion is preempted before it occurs)
- Better resource utilization: fast-lane GPUs stay loaded with short requests, slow-lane GPUs handle long requests in isolation

---

## 6. Evaluation Plan

### Baseline (existing infrastructure)
Run the four existing workload scripts against the current proxy (no migration):
1. `no_straggler_vllm_load.sh` — clean baseline
2. `mixed_vllm_load.sh` — light mixed load
3. `straggler_vllm_load.sh` — realistic straggler injection
4. `stress_straggler_vllm_load.sh` — heavy straggler stress

### Treatment (new)
5. `migration_straggler_vllm_load.sh` — same parameters as `straggler_vllm_load.sh`, hitting the migration-aware proxy

### Metrics

| Metric | Source | Hypothesis |
|---|---|---|
| P50/P95/P99 ITL (fast decode) | `vllm:inter_token_latency_seconds` | Drops to near no-straggler baseline |
| P99 E2E latency (short requests) | `vllm:e2e_request_latency_seconds` | Drops significantly |
| Migration rate | New proxy counter | Matches straggler injection rate |
| Migration pause duration | New proxy histogram | < 500ms in median case |
| Straggler E2E latency | vllm bench output | Increases ~10–20% — expected and acceptable |
| Fast-lane KV cache utilization | `vllm:kv_cache_usage_perc` | Lower peak, less variance |
| Preemption rate (fast decode) | `vllm:num_preemptions_total` | Drops to ~0 |

### Target Result

The **primary claim** is demonstrated by a single comparison plot: P99 ITL of the fast-lane decode GPU under straggler load, with and without migration, alongside the no-straggler baseline. A successful result shows the migration curve tracking the no-straggler baseline, while the no-migration curve shows the degradation documented in the existing experiments.

---

## 7. Implementation Scope

### What needs to be built

1. **Migration-aware proxy** (`disagg_proxy_migration.py`)
   - Extends the existing proxy with per-request token tracking
   - Implements the two-condition detection logic
   - Executes re-admission and client stream stitching

2. **4-GPU startup script** (`startup_4gpu.sh`)
   - Launches fast PD pair (GPUs 0+1, FP16, existing config)
   - Launches slow PD pair (GPUs 2+3, BF16, `--gpu-memory-utilization 0.7`)
   - Same healthcheck loop, extended to all four servers

3. **Updated monitoring config**
   - `prometheus_pd_host.yaml` extended with slow-prefill (port 20096) and slow-decode (port 20097) scrape targets
   - Two new Grafana panels: migration rate over time, migration pause duration histogram

4. **Evaluation workload script** (`scripts/migration_straggler_vllm_load.sh`)
   - Parameters identical to `straggler_vllm_load.sh`
   - Targets migration-aware proxy

### What does not need to be built
- No vLLM engine modifications
- No custom NCCL kernels or KV extraction logic
- No changes to the fast-lane or slow-lane vLLM servers themselves

---

## 8. Limitations

- **Re-admission cost grows with detection delay.** Migrating later is cheaper for detection accuracy but more expensive for re-processing. The optimal `T_min` is workload-dependent.
- **Slow lane must always run.** Fixed idle cost of a full 4-GPU allocation even during periods with no stragglers.
- **Stream pause is visible.** Clients on streaming APIs will observe a brief stall during slow-lane prefill. This is acceptable for non-interactive batch workloads but would require additional handling for strict latency SLAs on the straggler path.
- **BF16 vs FP16 output divergence.** Tokens 1..N (FP16) and N+1..end (BF16) are generated by numerically distinct model instances. In practice imperceptible for Llama-2, but worth noting as a limitation on output consistency.
