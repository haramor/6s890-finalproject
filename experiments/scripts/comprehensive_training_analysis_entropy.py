#!/usr/bin/env python3
"""
Comprehensive Training Analysis (Entropy vs Standard)
====================================================

Same as your clean version, but compares:
  - experiments/results/expert_entropy_2k/logs
  - experiments/results/expert_standard_2k/logs

Behavior:
- Loads each TensorBoard event file separately
- Strategy:
    latest = use newest event file in logs/
    merge  = merge all event files by step, keeping the value with latest wall_time
- Filters to <= max_step (default 10_000)
- Prints: steps/sec (if wall_time exists), steps-to-threshold (auto-chosen),
  divergence/NaN onset, plus clean CSV blocks for train/loss and train/top1_acc

Usage:
  python training_analysis_entropy_vs_standard.py --max_step 10000 --strategy latest
  python training_analysis_entropy_vs_standard.py --max_step 10000 --strategy merge
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import math

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("ERROR: tensorboard not available. Install with: pip install tensorboard")
    sys.exit(1)

PREFERRED_LOSS_TAGS = ["train/loss", "train/ce_loss", "loss"]
PREFERRED_TOP1_TAGS = ["train/top1_acc", "top1_acc", "accuracy"]

def load_event_file(event_file: Path):
    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    data = {}
    for tag in tags:
        evs = ea.Scalars(tag)
        data[tag] = {
            "steps": [e.step for e in evs],
            "values": [e.value for e in evs],
            "wall_times": [e.wall_time for e in evs],
        }
    return data

def pick_metric(metrics: dict, candidates):
    for c in candidates:
        if c in metrics:
            return c
    return None

def filter_by_max_step(metric_data, max_step):
    s = np.array(metric_data["steps"])
    v = np.array(metric_data["values"], dtype=float)
    t = np.array(metric_data["wall_times"], dtype=float)
    mask = s <= max_step
    return {"steps": s[mask], "values": v[mask], "wall_times": t[mask]}

def merge_runs_by_step_keep_latest(list_of_metric_data):
    # Merge multiple (steps, values, wall_times) into one by keeping, for each step,
    # the value with the largest wall_time.
    best = {}
    for md in list_of_metric_data:
        for s, v, wt in zip(md["steps"], md["values"], md["wall_times"]):
            s = int(s)
            if (s not in best) or (wt > best[s][1]):
                best[s] = (float(v), float(wt))
    steps = np.array(sorted(best.keys()), dtype=int)
    values = np.array([best[s][0] for s in steps], dtype=float)
    wall_times = np.array([best[s][1] for s in steps], dtype=float)
    return {"steps": steps, "values": values, "wall_times": wall_times}

def last_finite(x):
    for v in reversed(x):
        if math.isfinite(float(v)):
            return float(v)
    return float("nan")

def steps_to_threshold(steps, values, thr):
    for s, v in zip(steps, values):
        if math.isfinite(float(v)) and float(v) >= float(thr):
            return int(s)
    return None

def nan_onset_step(steps, values):
    for s, v in zip(steps, values):
        if not math.isfinite(float(v)):
            return int(s)
    return None

def calc_steps_per_sec(steps, wall_times):
    if len(steps) < 2:
        return None
    dt = float(wall_times[-1] - wall_times[0])
    ds = int(steps[-1] - steps[0])
    if dt <= 0 or ds <= 0:
        return None
    return ds / dt

def downsample(steps, values, k=120):
    if len(steps) <= k:
        return steps, values
    idx = np.linspace(0, len(steps) - 1, k, dtype=int)
    return steps[idx], values[idx]

def choose_thresholds(top1_values, base_thr=0.10):
    """
    Your note: these runs didn't do well, so 'steps_to_threshold' shouldn't use
    optimistic fixed targets like 0.5.

    Best-judgement policy:
    - Always include: 0.05, 0.10 (very low)
    - Then add a couple near what the run actually reaches:
        60%, 80%, 90% of max_finite (clipped to [0.01, 0.95])
    - Deduplicate + sort
    """
    v = np.array(top1_values, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return [0.05, 0.10]

    mx = float(np.max(finite))
    # If mx is tiny, still keep low thresholds
    dynamic = []
    for frac in (0.60, 0.80, 0.90):
        thr = frac * mx
        thr = max(0.01, min(0.95, thr))
        dynamic.append(thr)

    thrs = [0.05, 0.10] + dynamic
    # keep unique w/ rounding for stability
    thrs = sorted({round(t, 3) for t in thrs})
    return thrs

def build_series(run_metrics, tag_candidates, max_step, strategy):
    series_list = []
    chosen_tag = None

    for rm in run_metrics:
        tag = pick_metric(rm, tag_candidates)
        if tag is None:
            continue
        chosen_tag = chosen_tag or tag
        md = filter_by_max_step(rm[tag], max_step)
        if len(md["steps"]) > 0:
            series_list.append(md)

    if not series_list:
        return None, None

    if strategy == "merge":
        merged = merge_runs_by_step_keep_latest(series_list)
        return chosen_tag, merged

    # latest: only one run_metrics entry should exist, still sort by step
    md = series_list[0]
    order = np.argsort(md["steps"])
    md = {k: md[k][order] for k in md}
    return chosen_tag, md

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_step", type=int, default=10_000)
    ap.add_argument(
        "--strategy",
        choices=["latest", "merge"],
        default="latest",
        help="latest = use newest event file; merge = merge all by step keeping latest wall_time",
    )
    ap.add_argument("--csv_points", type=int, default=60, help="downsample CSV points per series")
    ap.add_argument("--n_thresholds_note", action="store_true",
                    help="print a short note about how thresholds were chosen")
    args = ap.parse_args()

    base_dir = Path("/workspace/6s890-finalproject")
    experiments = {
        "expert_entropy_2k":  base_dir / "experiments/results/expert_entropy_2k/logs",
        "expert_standard_2k": base_dir / "experiments/results/expert_standard_2k/logs",
    }

    print("=" * 88)
    print("CLEAN TRAINING ANALYSIS (expert_entropy_2k vs expert_standard_2k)")
    print(f"max_step={args.max_step} | strategy={args.strategy} | csv_points={args.csv_points}")
    print("=" * 88)

    for name, log_dir in experiments.items():
        print(f"\n--- {name.upper()} ---")
        if not log_dir.exists():
            print(f"  ✗ Missing log dir: {log_dir}")
            continue

        event_files = sorted(log_dir.glob("events.out.tfevents.*"))
        if not event_files:
            print(f"  ✗ No event files in: {log_dir}")
            continue

        # choose runs
        if args.strategy == "latest":
            chosen = max(event_files, key=lambda p: p.stat().st_mtime)
            runs = [chosen]
            print(f"  Event files: {len(event_files)} | using latest: {chosen.name}")
        else:
            runs = event_files
            print(f"  Event files: {len(event_files)} | merging all")

        # load
        run_metrics = [load_event_file(ef) for ef in runs]

        # build clean series
        loss_tag, loss = build_series(run_metrics, PREFERRED_LOSS_TAGS, args.max_step, args.strategy)
        top1_tag, top1 = build_series(run_metrics, PREFERRED_TOP1_TAGS, args.max_step, args.strategy)

        # summaries
        if loss is not None:
            v = loss["values"]
            s = loss["steps"]
            wt = loss["wall_times"]
            n_nan = int(np.sum(~np.isfinite(v)))
            onset = nan_onset_step(s, v)
            min_fin = float(np.nanmin(v[np.isfinite(v)])) if np.any(np.isfinite(v)) else float("nan")
            print(f"  LOSS tag: {loss_tag} | points={len(v)} | nonfinite={n_nan} | nan_onset={onset}")
            print(f"    initial={float(v[0]):.6f} | last_finite={last_finite(v):.6f} | min_finite={min_fin:.6f}")
            sps = calc_steps_per_sec(s, wt)
            if sps is not None:
                print(f"    steps/sec ≈ {sps:.2f} (wall-clock)")

        if top1 is not None:
            v = np.array(top1["values"], dtype=float)
            s = np.array(top1["steps"], dtype=int)
            n_nan = int(np.sum(~np.isfinite(v)))
            onset = nan_onset_step(s, v)
            finite = v[np.isfinite(v)]
            mx = float(np.max(finite)) if finite.size else float("nan")
            print(f"  TOP1 tag: {top1_tag} | points={len(v)} | nonfinite={n_nan} | nan_onset={onset}")
            if finite.size:
                print(f"    initial={float(v[0]):.6f} | last_finite={last_finite(v):.6f} | max_finite={mx:.6f}")
            else:
                print(f"    initial={float(v[0]):.6f} | last_finite={last_finite(v):.6f} | max_finite=N/A")

            thrs = choose_thresholds(v)
            if args.n_thresholds_note:
                print("  threshold_policy: [0.05, 0.10] + {0.60,0.80,0.90}*max_finite (clipped to [0.01,0.95])")

            print("  steps_to_threshold:")
            for thr in thrs:
                st = steps_to_threshold(s, v, thr)
                print(f"    top1≥{thr:.3f}: {st if st is not None else 'N/A'}")

        # CSV blocks (downsampled)
        print("\n  CSV (downsampled)")
        if loss is not None:
            ds_s, ds_v = downsample(np.array(loss["steps"]), np.array(loss["values"], dtype=float), k=args.csv_points)
            print("  experiment,step,train_loss")
            for ss, vv in zip(ds_s, ds_v):
                print(f"  {name},{int(ss)},{float(vv):.6f}")
        if top1 is not None:
            ds_s, ds_v = downsample(np.array(top1["steps"]), np.array(top1["values"], dtype=float), k=args.csv_points)
            print("  experiment,step,train_top1")
            for ss, vv in zip(ds_s, ds_v):
                print(f"  {name},{int(ss)},{float(vv):.6f}")

if __name__ == "__main__":
    main()
