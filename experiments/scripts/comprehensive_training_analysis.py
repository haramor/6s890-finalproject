"""
Comprehensive Training Analysis (Clean)
- Loads each TensorBoard event file separately
- Selects the latest run per experiment (or merges safely)
- Filters to <= max_step (default 10_000)
- Prints: steps/sec (if wall_time exists), steps-to-threshold, divergence/NaN onset,
  plus clean CSV blocks for train/loss and train/top1_acc

Usage:
  python comprehensive_training_analysis_clean.py --max_step 10000 --strategy latest
  python comprehensive_training_analysis_clean.py --max_step 10000 --strategy merge
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import math
from collections import defaultdict

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
            if (s not in best) or (wt > best[s][1]):
                best[s] = (v, wt)
    steps = np.array(sorted(best.keys()), dtype=int)
    values = np.array([best[s][0] for s in steps], dtype=float)
    wall_times = np.array([best[s][1] for s in steps], dtype=float)
    return {"steps": steps, "values": values, "wall_times": wall_times}

def last_finite(x):
    for v in reversed(x):
        if math.isfinite(v):
            return v
    return float("nan")

def steps_to_threshold(steps, values, thr):
    for s, v in zip(steps, values):
        if math.isfinite(v) and v >= thr:
            return int(s)
    return None

def nan_onset_step(steps, values):
    for s, v in zip(steps, values):
        if not math.isfinite(v):
            return int(s)
    return None

def calc_steps_per_sec(steps, wall_times):
    # uses first/last; robust enough for summary
    if len(steps) < 2:
        return None
    dt = float(wall_times[-1] - wall_times[0])
    ds = int(steps[-1] - steps[0])
    if dt <= 0:
        return None
    return ds / dt

def downsample(steps, values, k=120):
    if len(steps) <= k:
        return steps, values
    idx = np.linspace(0, len(steps)-1, k, dtype=int)
    return steps[idx], values[idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_step", type=int, default=10_000)
    ap.add_argument("--strategy", choices=["latest", "merge"], default="latest",
                    help="latest = use newest event file; merge = merge all by step keeping latest wall_time")
    args = ap.parse_args()

    base_dir = Path("/workspace/6s890-finalproject")
    experiments = {
        "baseline": base_dir / "experiments/results/baseline_mixed_skill/logs",
        "expert":   base_dir / "experiments/results/expert_LE22ct/logs",
        "random":   base_dir / "experiments/results/random_skill/logs",
    }

    print("="*88)
    print("CLEAN TRAINING ANALYSIS (per-step efficiency + wall-clock if available)")
    print(f"max_step={args.max_step} | strategy={args.strategy}")
    print("="*88)

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
        run_metrics = []
        for ef in runs:
            run_metrics.append(load_event_file(ef))

        # collect tags present
        all_tags = sorted(set(t for rm in run_metrics for t in rm.keys()))
        print(f"  Scalar tags: {len(all_tags)}")
        # Not printing all tags to keep terminal clean; uncomment if you want:
        # for t in all_tags: print("   -", t)

        # build clean series for top1 and loss
        def build_series(tag_candidates):
            series_list = []
            for rm in run_metrics:
                tag = pick_metric(rm, tag_candidates)
                if tag is None:
                    continue
                md = filter_by_max_step(rm[tag], args.max_step)
                if len(md["steps"]) > 0:
                    series_list.append(md)
            if not series_list:
                return None, None
            if args.strategy == "merge":
                merged = merge_runs_by_step_keep_latest(series_list)
                return pick_metric(run_metrics[0], tag_candidates) or tag_candidates[0], merged
            # latest: just use the first (only) run’s chosen tag
            tag = pick_metric(run_metrics[0], tag_candidates)
            md = series_list[0]
            # sort by step
            order = np.argsort(md["steps"])
            md = {k: md[k][order] for k in md}
            return tag, md

        loss_tag, loss = build_series(PREFERRED_LOSS_TAGS)
        top1_tag, top1 = build_series(PREFERRED_TOP1_TAGS)

        # summary stats
        if loss is not None:
            v = loss["values"]
            s = loss["steps"]
            wt = loss["wall_times"]
            n_nan = int(np.sum(~np.isfinite(v)))
            onset = nan_onset_step(s, v)
            print(f"  LOSS tag: {loss_tag} | points={len(v)} | nonfinite={n_nan} | nan_onset={onset}")
            print(f"    initial={v[0]:.4f} | last_finite={last_finite(v):.4f} | min_finite={np.nanmin(v):.4f}")
            sps = calc_steps_per_sec(s, wt)
            if sps is not None:
                print(f"    steps/sec ≈ {sps:.2f} (wall-clock)")

        if top1 is not None:
            v = top1["values"]; s = top1["steps"]
            n_nan = int(np.sum(~np.isfinite(v)))
            onset = nan_onset_step(s, v)
            print(f"  TOP1 tag: {top1_tag} | points={len(v)} | nonfinite={n_nan} | nan_onset={onset}")
            print(f"    initial={v[0]:.4f} | last_finite={last_finite(v):.4f} | max_finite={np.nanmax(v):.4f}")

            # steps-to-thresholds
            thrs = [0.10, 0.20, 0.30, 0.40, 0.50]
            print("  steps_to_threshold:")
            for thr in thrs:
                st = steps_to_threshold(s, v, thr)
                print(f"    top1≥{thr:.2f}: {st if st is not None else 'N/A'}")

        # CSV blocks (downsampled)
        print("\n  CSV (downsampled)")
        if loss is not None:
            ds_s, ds_v = downsample(loss["steps"], loss["values"], k=60)
            print("  experiment,step,train_loss")
            for ss, vv in zip(ds_s, ds_v):
                print(f"  {name},{int(ss)},{vv:.6f}")
        if top1 is not None:
            ds_s, ds_v = downsample(top1["steps"], top1["values"], k=60)
            print("  experiment,step,train_top1")
            for ss, vv in zip(ds_s, ds_v):
                print(f"  {name},{int(ss)},{vv:.6f}")

if __name__ == "__main__":
    main()
