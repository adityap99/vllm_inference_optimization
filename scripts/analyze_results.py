#!/usr/bin/env python3
"""
analyze_results.py — Parse vllm bench --save-result JSONs and proxy.log,
then generate comparison plots and a summary table (PROPOSAL.md §6).

Usage
─────
    python scripts/analyze_results.py
    python scripts/analyze_results.py --results-dir results --plots-dir plots
    python scripts/analyze_results.py --proxy-log proxy.log

Reads
─────
    results/<condition>/*.json          vllm bench --save-result outputs
    results/<condition>/metadata.json   timing + proxy-log line offsets
    proxy.log                           migration events from disagg_proxy_migration.py

Writes
──────
    plots/01_itl_p99_primary_claim.png   THE primary claim plot (PROPOSAL §6)
    plots/02_itl_all_percentiles.png     P50 / P99 ITL grouped bars
    plots/03_ttft_comparison.png         TTFT P50 / P99 — queueing & prefill health
    plots/04_e2e_comparison.png          E2E P99 short vs long requests
    plots/05_straggler_cost.png          Long-request E2E: migration cost
    plots/06_migration_pause_hist.png    Histogram of migration pause durations
    plots/summary_table.csv             All metrics in one CSV
    plots/summary_table.md              Markdown table copy (for README paste)
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# ─── Optional heavy imports — degrade gracefully if absent ────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")  # headless rendering on PACE/HPC
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    _HAS_PLOT = True
except ImportError:
    _HAS_PLOT = False
    print("[warn] matplotlib / numpy not found — plots will be skipped. "
          "Install with: pip install matplotlib numpy")

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False
    print("[warn] pandas not found — CSV output will be skipped. "
          "Install with: pip install pandas")

# ─── Experiment ordering and display metadata ─────────────────────────────────

EXPERIMENT_ORDER = [
    "no_straggler",
    "mixed",
    "straggler",
    "stress_straggler",
    "migration_straggler",
]

LABELS = {
    "no_straggler":       "No Straggler\n(baseline)",
    "mixed":              "Mixed Load",
    "straggler":          "Straggler\n(no migration)",
    "stress_straggler":   "Stress Straggler\n(no migration)",
    "migration_straggler":"Straggler\n(+ migration)",
}

SHORT_LABELS = {
    "no_straggler":       "Baseline",
    "mixed":              "Mixed",
    "straggler":          "Straggler",
    "stress_straggler":   "StressStraggler",
    "migration_straggler":"Migration",
}

# Color palette: blue family for baselines, red for no-migration straggler,
# green for migration treatment.
COLORS = {
    "no_straggler":       "#2E75B6",
    "mixed":              "#9DC3E6",
    "straggler":          "#E74C3C",
    "stress_straggler":   "#922B21",
    "migration_straggler":"#27AE60",
}

# Rate threshold to classify short vs long streams in the benchmark JSON.
# Short stream: request_rate >= SHORT_RPS_THRESHOLD (typically 15 RPS)
# Long  stream: request_rate <  SHORT_RPS_THRESHOLD (typically 0.5–1.5 RPS)
SHORT_RPS_THRESHOLD = 5.0


# ─── Result loading ───────────────────────────────────────────────────────────

def _get_pct(data: dict, metric: str, pct: int) -> float | None:
    """
    Extract a percentile value from a vllm bench --save-result JSON.

    vLLM stores percentiles as either:
      - list of [p, value] pairs:  [[90, 45.2], [95, 67.1], [99, 120.3]]
      - flat key in dict:          {"p99_itl_ms": 120.3}  (older format)
    P50 is stored as median_<metric>_ms.
    """
    # P50 lives in median_*_ms
    if pct == 50:
        v = data.get(f"median_{metric}_ms")
        if v is not None:
            return float(v)

    # Try percentiles_<metric>_ms list first (vLLM >= 0.4)
    key = f"percentiles_{metric}_ms"
    raw = data.get(key)
    if raw is not None:
        if isinstance(raw, list):
            for item in raw:
                # item may be [percentile, value] or {"percentile": p, "value": v}
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    p, v = item
                    if int(p) == pct:
                        return float(v)
                elif isinstance(item, dict):
                    if int(item.get("percentile", -1)) == pct:
                        return float(item["value"])
        elif isinstance(raw, dict):
            for k in (str(pct), f"p{pct}"):
                if k in raw:
                    return float(raw[k])

    # Fallback: flat key e.g. p99_itl_ms
    for k in (f"p{pct}_{metric}_ms", f"{metric}_p{pct}_ms"):
        if k in data:
            return float(data[k])

    return None


def _classify_stream(data: dict) -> str:
    """Return 'short' or 'long' based on the request rate in the result JSON."""
    rate = data.get("request_rate") or data.get("actual_request_rate") or 0.0
    try:
        rate = float(rate)
    except (TypeError, ValueError):
        rate = 0.0
    return "short" if rate >= SHORT_RPS_THRESHOLD else "long"


def _load_result_json(path: str) -> dict | None:
    """Load and validate a vllm bench result JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
        # Must have at least one latency metric to be useful
        if not any(k in data for k in (
            "mean_itl_ms", "median_itl_ms", "percentiles_itl_ms",
            "mean_ttft_ms", "median_ttft_ms",
        )):
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def find_and_load(results_dir: str) -> dict[str, dict[str, dict]]:
    """
    Walk results_dir and return a nested dict:
      { condition_name: { "short": <data>, "long": <data> | None } }

    For each condition, we pick the most-recently-modified JSON that matches
    the short/long classification.  If multiple JSONs exist for the same
    stream (re-runs), the latest one wins.
    """
    output: dict[str, dict] = {}

    for cond in EXPERIMENT_ORDER:
        cond_dir = os.path.join(results_dir, cond)
        if not os.path.isdir(cond_dir):
            continue

        candidates: dict[str, list[tuple[float, dict]]] = defaultdict(list)

        for path in glob.glob(os.path.join(cond_dir, "*.json")):
            if os.path.basename(path) in ("metadata.json",):
                continue
            data = _load_result_json(path)
            if data is None:
                continue
            stream = _classify_stream(data)
            mtime = os.path.getmtime(path)
            candidates[stream].append((mtime, data))

        if not candidates:
            print(f"[info] No result JSONs found for condition: {cond}")
            continue

        output[cond] = {}
        for stream, items in candidates.items():
            # Pick latest file
            items.sort(key=lambda x: x[0], reverse=True)
            output[cond][stream] = items[0][1]
            print(f"[info] Loaded {cond}/{stream}: "
                  f"rps={items[0][1].get('request_rate')}, "
                  f"completed={items[0][1].get('completed')}")

    return output


# ─── Metrics extraction ───────────────────────────────────────────────────────

def _row(cond: str, stream: str, data: dict) -> dict:
    """Extract all relevant metrics from a single result JSON into a flat dict."""
    return {
        "condition":         cond,
        "stream":            stream,
        "request_rate":      data.get("request_rate"),
        "completed":         data.get("completed"),
        "req_throughput":    data.get("request_throughput"),
        "out_throughput":    data.get("output_throughput"),
        # ITL
        "itl_mean":          data.get("mean_itl_ms"),
        "itl_p50":           _get_pct(data, "itl", 50),
        "itl_p90":           _get_pct(data, "itl", 90),
        "itl_p95":           _get_pct(data, "itl", 95),
        "itl_p99":           _get_pct(data, "itl", 99),
        # TTFT
        "ttft_mean":         data.get("mean_ttft_ms"),
        "ttft_p50":          _get_pct(data, "ttft", 50),
        "ttft_p90":          _get_pct(data, "ttft", 90),
        "ttft_p95":          _get_pct(data, "ttft", 95),
        "ttft_p99":          _get_pct(data, "ttft", 99),
        # TPOT (time-per-output-token ≈ avg ITL)
        "tpot_mean":         data.get("mean_tpot_ms"),
        "tpot_p50":          _get_pct(data, "tpot", 50),
        "tpot_p99":          _get_pct(data, "tpot", 99),
        # E2E
        "e2el_mean":         data.get("mean_e2el_ms"),
        "e2el_p50":          _get_pct(data, "e2el", 50),
        "e2el_p90":          _get_pct(data, "e2el", 90),
        "e2el_p95":          _get_pct(data, "e2el", 95),
        "e2el_p99":          _get_pct(data, "e2el", 99),
    }


def build_rows(all_data: dict[str, dict]) -> list[dict]:
    rows = []
    for cond, streams in all_data.items():
        for stream, data in streams.items():
            rows.append(_row(cond, stream, data))
    return rows


# ─── Proxy log parsing ────────────────────────────────────────────────────────

_EVICT_RE   = re.compile(
    r"\[proxy/MIGRATE\].*N=(\d+).*rate=([\d.]+) tok/s"
)
_PAUSE_RE   = re.compile(
    r"\[proxy/MIGRATE\] Slow-lane prefill done in ([\d.]+)s"
)


def parse_proxy_log(proxy_log: str,
                    start_line: int = 0,
                    end_line: int | None = None) -> dict:
    """
    Extract migration statistics from proxy.log between two line numbers.
    Returns a dict with:
        migrations_total: int
        eviction_token_counts: list[int]   — N at eviction
        eviction_rates: list[float]        — tok/s at eviction
        migration_pauses_s: list[float]    — slow-lane prefill durations
    """
    stats = {
        "migrations_total": 0,
        "eviction_token_counts": [],
        "eviction_rates": [],
        "migration_pauses_s": [],
    }
    try:
        with open(proxy_log) as f:
            lines = f.readlines()
    except OSError:
        return stats

    relevant = lines[start_line:end_line]
    for line in relevant:
        m = _EVICT_RE.search(line)
        if m:
            stats["migrations_total"] += 1
            stats["eviction_token_counts"].append(int(m.group(1)))
            stats["eviction_rates"].append(float(m.group(2)))
        m2 = _PAUSE_RE.search(line)
        if m2:
            stats["migration_pauses_s"].append(float(m2.group(1)))

    return stats


def load_migration_stats(results_dir: str, proxy_log: str) -> dict | None:
    """Load metadata for migration_straggler to find proxy log offsets."""
    meta_path = os.path.join(results_dir, "migration_straggler", "metadata.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    start = int(meta.get("proxy_log_start_line", 0))
    end   = meta.get("proxy_log_end_line")
    end   = int(end) if end is not None else None
    return parse_proxy_log(proxy_log, start, end)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _bar_labels(ax, bars, fmt="{:.1f}", offset_frac=0.01):
    """Add value annotations above each bar."""
    ymax = max((b.get_height() for b in bars if b.get_height() > 0), default=1)
    for bar in bars:
        h = bar.get_height()
        if h and h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + ymax * offset_frac,
                fmt.format(h),
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )


def plot_itl_p99_primary_claim(rows: list[dict], plots_dir: str) -> None:
    """
    Plot 01 — THE primary claim (PROPOSAL §6):
    P99 ITL for short requests across all conditions.
    Migration curve should track the no-straggler baseline.
    """
    conditions = [c for c in EXPERIMENT_ORDER
                  if any(r["condition"] == c and r["stream"] == "short" for r in rows)]
    vals = []
    for cond in conditions:
        v = next((r["itl_p99"] for r in rows
                  if r["condition"] == cond and r["stream"] == "short"), None)
        vals.append(v)

    if not any(v is not None for v in vals):
        print("[warn] No short-stream ITL P99 data found — skipping plot 01")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(conditions))
    bars = ax.bar(
        x,
        [v if v is not None else 0 for v in vals],
        color=[COLORS[c] for c in conditions],
        width=0.55,
        edgecolor="black", linewidth=0.6,
        zorder=3,
    )
    _bar_labels(ax, bars)

    # Baseline dashed line
    baseline_val = next((r["itl_p99"] for r in rows
                         if r["condition"] == "no_straggler" and r["stream"] == "short"), None)
    if baseline_val is not None:
        ax.axhline(
            y=baseline_val, color=COLORS["no_straggler"],
            linestyle="--", linewidth=1.8,
            label=f"No-straggler baseline  ({baseline_val:.1f} ms)",
            zorder=2,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[c] for c in conditions], fontsize=9)
    ax.set_ylabel("P99 Inter-Token Latency  (ms)", fontsize=11)
    ax.set_title(
        "P99 ITL — Short Requests  •  Primary Claim\n"
        "Migration should restore ITL to near the no-straggler baseline",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ymax = max((v for v in vals if v), default=1)
    ax.set_ylim(0, ymax * 1.25)

    plt.tight_layout()
    out = os.path.join(plots_dir, "01_itl_p99_primary_claim.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out}")


def plot_itl_all_percentiles(rows: list[dict], plots_dir: str) -> None:
    """
    Plot 02 — P50 and P99 ITL grouped bars for short requests.
    Shows the full distribution shift, not just the tail.
    """
    conditions = [c for c in EXPERIMENT_ORDER
                  if any(r["condition"] == c and r["stream"] == "short" for r in rows)]

    p50s = [next((r["itl_p50"] for r in rows
                  if r["condition"] == c and r["stream"] == "short"), None)
            for c in conditions]
    p99s = [next((r["itl_p99"] for r in rows
                  if r["condition"] == c and r["stream"] == "short"), None)
            for c in conditions]

    if not any(v is not None for v in p99s):
        print("[warn] No ITL data — skipping plot 02")
        return

    n = len(conditions)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    b50 = ax.bar(x - width / 2,
                 [v or 0 for v in p50s],
                 width, label="P50 (median)",
                 color=[COLORS[c] for c in conditions],
                 alpha=0.6, edgecolor="black", linewidth=0.5, zorder=3)
    b99 = ax.bar(x + width / 2,
                 [v or 0 for v in p99s],
                 width, label="P99",
                 color=[COLORS[c] for c in conditions],
                 alpha=1.0, edgecolor="black", linewidth=0.5, zorder=3)
    _bar_labels(ax, b50)
    _bar_labels(ax, b99)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in conditions], fontsize=9)
    ax.set_ylabel("Inter-Token Latency  (ms)", fontsize=11)
    ax.set_title("ITL Distribution — Short Requests\n"
                 "Solid = P99  |  Faded = P50 (median)", fontsize=12)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Custom legend showing percentile shade
    legend_handles = [
        mpatches.Patch(facecolor="gray", alpha=0.6, label="P50 (median)"),
        mpatches.Patch(facecolor="gray", alpha=1.0, label="P99"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)
    ymax = max((v for v in p99s if v), default=1)
    ax.set_ylim(0, ymax * 1.25)

    plt.tight_layout()
    out = os.path.join(plots_dir, "02_itl_all_percentiles.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out}")


def plot_ttft_comparison(rows: list[dict], plots_dir: str) -> None:
    """
    Plot 03 — TTFT P50 / P99 for short requests.
    TTFT captures admission queueing depth and prefill server health.
    High TTFT under straggler load → KV cache pressure is spilling into queuing.
    """
    conditions = [c for c in EXPERIMENT_ORDER
                  if any(r["condition"] == c and r["stream"] == "short" for r in rows)]

    p50s = [next((r["ttft_p50"] for r in rows
                  if r["condition"] == c and r["stream"] == "short"), None)
            for c in conditions]
    p99s = [next((r["ttft_p99"] for r in rows
                  if r["condition"] == c and r["stream"] == "short"), None)
            for c in conditions]

    if not any(v is not None for v in p99s):
        print("[warn] No TTFT data — skipping plot 03")
        return

    n = len(conditions)
    x = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    b50 = ax.bar(x - width / 2, [v or 0 for v in p50s], width,
                 color=[COLORS[c] for c in conditions],
                 alpha=0.6, edgecolor="black", linewidth=0.5, zorder=3)
    b99 = ax.bar(x + width / 2, [v or 0 for v in p99s], width,
                 color=[COLORS[c] for c in conditions],
                 alpha=1.0, edgecolor="black", linewidth=0.5, zorder=3)
    _bar_labels(ax, b50)
    _bar_labels(ax, b99)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in conditions], fontsize=9)
    ax.set_ylabel("Time-to-First-Token  (ms)", fontsize=11)
    ax.set_title("TTFT — Short Requests\n"
                 "Solid = P99  |  Faded = P50", fontsize=12)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    legend_handles = [
        mpatches.Patch(facecolor="gray", alpha=0.6, label="P50"),
        mpatches.Patch(facecolor="gray", alpha=1.0, label="P99"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)
    ymax = max((v for v in p99s if v), default=1)
    ax.set_ylim(0, ymax * 1.25)

    plt.tight_layout()
    out = os.path.join(plots_dir, "03_ttft_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out}")


def plot_e2e_comparison(rows: list[dict], plots_dir: str) -> None:
    """
    Plot 04 — P99 E2E latency for short AND long requests side by side.
    Migration helps short requests (lower E2E) and costs long requests (higher E2E).
    Both effects are visible and expected per PROPOSAL §5.
    """
    conditions = [c for c in EXPERIMENT_ORDER
                  if any(r["condition"] == c for r in rows)]

    short_vals = [next((r["e2el_p99"] for r in rows
                        if r["condition"] == c and r["stream"] == "short"), None)
                  for c in conditions]
    long_vals  = [next((r["e2el_p99"] for r in rows
                        if r["condition"] == c and r["stream"] == "long"), None)
                  for c in conditions]

    if not any(v is not None for v in short_vals + long_vals):
        print("[warn] No E2E data — skipping plot 04")
        return

    n = len(conditions)
    x = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(13, 5))

    b_short = ax.bar(x - width / 2, [v or 0 for v in short_vals], width,
                     color=[COLORS[c] for c in conditions],
                     edgecolor="black", linewidth=0.5, alpha=1.0,
                     label="Short requests", zorder=3)
    b_long  = ax.bar(x + width / 2, [v or 0 for v in long_vals], width,
                     color=[COLORS[c] for c in conditions],
                     edgecolor="black", linewidth=0.5, alpha=0.45,
                     label="Long requests", zorder=3)
    _bar_labels(ax, b_short, fmt="{:.0f}")
    _bar_labels(ax, b_long,  fmt="{:.0f}")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in conditions], fontsize=9)
    ax.set_ylabel("P99 End-to-End Latency  (ms)", fontsize=11)
    ax.set_title(
        "P99 E2E Latency — Short (solid) vs Long / Straggler (faded)\n"
        "Migration lowers short-request E2E; long-request E2E increases (expected tradeoff)",
        fontsize=12,
    )
    legend_handles = [
        mpatches.Patch(facecolor="gray", alpha=1.0,  label="Short requests"),
        mpatches.Patch(facecolor="gray", alpha=0.45, label="Long/straggler requests"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    out = os.path.join(plots_dir, "04_e2e_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out}")


def plot_straggler_cost(rows: list[dict], plots_dir: str) -> None:
    """
    Plot 05 — Long-request E2E for straggler vs migration_straggler.
    Quantifies the migration cost on the straggler itself (PROPOSAL §5):
    migration_straggler E2E > straggler E2E  (expected and acceptable).
    """
    conds = ["straggler", "migration_straggler"]
    found = [c for c in conds if any(r["condition"] == c and r["stream"] == "long" for r in rows)]
    if len(found) < 1:
        print("[warn] Not enough straggler/migration data for plot 05 — skipping")
        return

    metrics = ["e2el_p50", "e2el_p99", "ttft_p50", "ttft_p99"]
    metric_labels = ["E2E P50", "E2E P99", "TTFT P50", "TTFT P99"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    width = 0.35

    for i, cond in enumerate(found):
        vals = []
        for m in metrics:
            v = next((r[m] for r in rows
                      if r["condition"] == cond and r["stream"] == "long"), None)
            vals.append(v or 0)
        offset = (i - len(found) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=LABELS[cond], color=COLORS[cond],
                      edgecolor="black", linewidth=0.5, zorder=3)
        _bar_labels(ax, bars, fmt="{:.0f}")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Latency  (ms)", fontsize=11)
    ax.set_title(
        "Long / Straggler Request Latency\n"
        "Migration cost: straggler E2E increases due to re-prefill on slow lane "
        "(expected & acceptable)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    out = os.path.join(plots_dir, "05_straggler_cost.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out}")


def plot_migration_pause_histogram(stats: dict, plots_dir: str) -> None:
    """
    Plot 06 — Histogram of migration pause durations (slow-lane prefill time).
    PROPOSAL §6 target: median < 500 ms.
    """
    pauses_ms = [p * 1000 for p in stats.get("migration_pauses_s", [])]
    if not pauses_ms:
        print("[warn] No migration pause data found in proxy.log — skipping plot 06")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    n_bins = min(30, max(10, len(pauses_ms) // 5))
    ax.hist(pauses_ms, bins=n_bins, color=COLORS["migration_straggler"],
            edgecolor="black", linewidth=0.5, alpha=0.85, zorder=3)

    med = float(np.median(pauses_ms))
    p95 = float(np.percentile(pauses_ms, 95))
    ax.axvline(med, color="black",   linestyle="--", linewidth=1.5,
               label=f"Median  {med:.0f} ms")
    ax.axvline(p95, color="darkred", linestyle=":",  linewidth=1.5,
               label=f"P95     {p95:.0f} ms")
    ax.axvline(500, color="orange",  linestyle="-.", linewidth=1.2,
               label="PROPOSAL target  500 ms")

    ax.set_xlabel("Migration Pause (slow-lane prefill time)  (ms)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        f"Migration Pause Duration  (n={len(pauses_ms)} migrations)\n"
        f"Median={med:.0f} ms   P95={p95:.0f} ms",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    out = os.path.join(plots_dir, "06_migration_pause_hist.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out}")


# ─── Summary table ────────────────────────────────────────────────────────────

def _pct_recovery(baseline: float | None, straggler: float | None,
                  migration: float | None) -> str:
    """Compute what fraction of straggler degradation was recovered by migration."""
    if None in (baseline, straggler, migration) or straggler == baseline:
        return "N/A"
    degradation = straggler - baseline
    recovered   = straggler - migration
    pct = 100.0 * recovered / degradation
    return f"{pct:+.1f}%"


def print_summary(rows: list[dict], migration_stats: dict | None) -> None:
    """Print a human-readable summary table to stdout."""
    print()
    print("=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)

    # Short-request ITL summary
    print()
    print("Short-request Inter-Token Latency (ITL, ms)")
    print(f"  {'Condition':<26} {'P50':>8} {'P90':>8} {'P95':>8} {'P99':>8} {'Mean':>8}")
    print(f"  {'-'*26} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for cond in EXPERIMENT_ORDER:
        r = next((x for x in rows if x["condition"] == cond and x["stream"] == "short"), None)
        if r is None:
            continue
        def _f(v):
            return f"{v:8.1f}" if v is not None else "       —"
        print(f"  {SHORT_LABELS[cond]:<26} {_f(r['itl_p50'])} {_f(r['itl_p90'])}"
              f" {_f(r['itl_p95'])} {_f(r['itl_p99'])} {_f(r['itl_mean'])}")

    # Primary claim: recovery percentage
    print()
    baseline_p99  = next((r["itl_p99"] for r in rows
                          if r["condition"] == "no_straggler" and r["stream"] == "short"), None)
    straggler_p99 = next((r["itl_p99"] for r in rows
                          if r["condition"] == "straggler" and r["stream"] == "short"), None)
    migration_p99 = next((r["itl_p99"] for r in rows
                          if r["condition"] == "migration_straggler" and r["stream"] == "short"), None)

    recovery = _pct_recovery(baseline_p99, straggler_p99, migration_p99)
    print("P99 ITL — Primary Claim")
    print(f"  No-straggler baseline:  {baseline_p99:.1f} ms" if baseline_p99 else "  No-straggler baseline: N/A")
    print(f"  Straggler (no migr.):   {straggler_p99:.1f} ms" if straggler_p99 else "  Straggler (no migr.): N/A")
    print(f"  Straggler + migration:  {migration_p99:.1f} ms" if migration_p99 else "  Straggler + migration: N/A")
    print(f"  ITL degradation recovered by migration: {recovery}")

    # Migration stats from proxy.log
    if migration_stats:
        pauses = migration_stats.get("migration_pauses_s", [])
        counts = migration_stats.get("eviction_token_counts", [])
        print()
        print("Migration Events (from proxy.log)")
        print(f"  Total migrations triggered: {migration_stats['migrations_total']}")
        if pauses:
            pauses_ms = [p * 1000 for p in pauses]
            print(f"  Pause duration (ms):  median={float(np.median(pauses_ms)):.0f}  "
                  f"P95={float(np.percentile(pauses_ms, 95)):.0f}  "
                  f"max={max(pauses_ms):.0f}")
        if counts:
            print(f"  Tokens at eviction:   median={float(np.median(counts)):.0f}  "
                  f"min={min(counts)}  max={max(counts)}")

    print()


def save_csv(rows: list[dict], plots_dir: str) -> None:
    if not _HAS_PANDAS:
        return
    df = pd.DataFrame(rows)
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].round(2)
    csv_path = os.path.join(plots_dir, "summary_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"  [saved] {csv_path}")

    # Markdown format, short-stream only
    short_df = df[df["stream"] == "short"].copy()
    cols = ["condition", "itl_p50", "itl_p99", "ttft_p50", "ttft_p99",
            "e2el_p50", "e2el_p99", "req_throughput"]
    short_df = short_df[[c for c in cols if c in short_df.columns]]
    short_df.columns = [c.replace("_", " ").title() for c in short_df.columns]
    md_path = os.path.join(plots_dir, "summary_table.md")
    with open(md_path, "w") as f:
        f.write("## Short-Request Metrics Summary\n\n")
        f.write(short_df.to_markdown(index=False))
        f.write("\n\n*All latencies in ms.  "
                "ITL = inter-token latency, TTFT = time to first token, "
                "E2EL = end-to-end latency.*\n")
    print(f"  [saved] {md_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze vllm bench results and generate comparison plots."
    )
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing per-condition result subdirs (default: results)")
    parser.add_argument("--plots-dir",   default="plots",
                        help="Output directory for plots and tables (default: plots)")
    parser.add_argument("--proxy-log",   default="proxy.log",
                        help="Path to proxy.log for migration event parsing (default: proxy.log)")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    plots_dir   = os.path.abspath(args.plots_dir)
    proxy_log   = os.path.abspath(args.proxy_log)

    if not os.path.isdir(results_dir):
        print(f"ERROR: results directory not found: {results_dir}")
        print("       Run scripts/run_experiments.sh first.")
        sys.exit(1)

    os.makedirs(plots_dir, exist_ok=True)

    print(f"Results dir : {results_dir}")
    print(f"Plots dir   : {plots_dir}")
    print(f"Proxy log   : {proxy_log}")
    print()

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading result JSONs...")
    all_data = find_and_load(results_dir)
    if not all_data:
        print("ERROR: No result JSON files found. "
              "Check that run_experiments.sh completed and --save-result was active.")
        sys.exit(1)

    rows = build_rows(all_data)

    # ── Load migration stats ───────────────────────────────────────────────
    migration_stats = None
    if os.path.exists(proxy_log):
        migration_stats = load_migration_stats(results_dir, proxy_log)
        if migration_stats:
            print(f"Loaded proxy.log: {migration_stats['migrations_total']} migration events found")
        else:
            print("proxy.log found but no migration stats extracted "
                  "(metadata.json missing or no migration events)")
    else:
        print(f"proxy.log not found at {proxy_log} — migration stats unavailable")

    # ── Summary to stdout ──────────────────────────────────────────────────
    print_summary(rows, migration_stats)

    # ── Plots ──────────────────────────────────────────────────────────────
    if not _HAS_PLOT:
        print("[skip] matplotlib not available — skipping all plots")
    else:
        print("Generating plots...")
        plot_itl_p99_primary_claim(rows, plots_dir)
        plot_itl_all_percentiles(rows,   plots_dir)
        plot_ttft_comparison(rows,       plots_dir)
        plot_e2e_comparison(rows,        plots_dir)
        plot_straggler_cost(rows,        plots_dir)
        if migration_stats:
            plot_migration_pause_histogram(migration_stats, plots_dir)

    # ── CSV / Markdown ─────────────────────────────────────────────────────
    if _HAS_PANDAS:
        print("Saving summary tables...")
        save_csv(rows, plots_dir)

    print()
    print("Done.  Results in:", plots_dir)


if __name__ == "__main__":
    main()
