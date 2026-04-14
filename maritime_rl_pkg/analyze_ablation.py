"""
Aggregate and analyze ablation study results from multiple training runs.

This script reads ablation_summary.json files from multiple runs with different
num_workers settings and generates a summary comparison.

Usage:
    python -m maritime_rl_pkg.maritime_rl.analyze_ablation --results_root ./results_dir

The results_root should contain subdirectories like:
    results_dir/MARL_ppo_case2_20240101-120000/seed_0/ablation_summary.json
    ...
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results_root",
        type=str,
        required=True,
        help="Root directory containing training results subdirectories"
    )
    return p.parse_args()


def aggregate_ablation_results(results_root):
    """
    Scan results_root for ablation_summary.json files and aggregate by num_workers.
    
    Returns dict: {
        num_workers: [
            {ablation_summary fields...},
            ...
        ],
        ...
    }
    """
    results_root = Path(results_root)
    aggregated = defaultdict(list)
    
    # Recursively find all ablation_summary.json files
    for summary_file in results_root.rglob("ablation_summary.json"):
        try:
            with open(summary_file, "r") as f:
                summary = json.load(f)
            
            num_workers = summary.get("num_workers")
            if num_workers is not None:
                aggregated[num_workers].append(summary)
                print(f"[loaded] {summary_file} | num_workers={num_workers}")
        except Exception as e:
            print(f"[WARNING] Failed to load {summary_file}: {e}")
    
    return dict(aggregated)


def compute_statistics(summaries_by_workers):
    """
    Compute mean and std for each num_workers setting.
    
    Returns dict: {
        num_workers: {
            "count": int,
            "mean_steps_per_sec": float,
            "std_steps_per_sec": float,
            "mean_train_wall_time_s": float,
            "mean_eval_wall_time_s": float,
            "mean_total_run_time_s": float,
        },
        ...
    }
    """
    stats = {}
    
    for num_workers in sorted(summaries_by_workers.keys()):
        summaries = summaries_by_workers[num_workers]
        
        steps_per_sec = [
            s["mean_approx_train_steps_per_sec"]
            for s in summaries
            if np.isfinite(s["mean_approx_train_steps_per_sec"])
        ]
        train_times = [s["mean_train_wall_time_s"] for s in summaries]
        eval_times = [s["mean_eval_wall_time_s"] for s in summaries]
        total_times = [s["total_run_time_s"] for s in summaries]
        
        stats[num_workers] = {
            "count": len(summaries),
            "mean_steps_per_sec": float(np.mean(steps_per_sec)) if steps_per_sec else float("nan"),
            "std_steps_per_sec": float(np.std(steps_per_sec)) if len(steps_per_sec) > 1 else 0.0,
            "mean_train_wall_time_s": float(np.mean(train_times)),
            "mean_eval_wall_time_s": float(np.mean(eval_times)),
            "mean_total_run_time_s": float(np.mean(total_times)),
        }
    
    return stats


def save_ablation_report(stats, output_path):
    """Save ablation statistics to JSON file."""
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[saved] {output_path}")


def plot_ablation_results(stats, output_path):
    """Create plot: num_workers vs steps/sec with error bars."""
    num_workers_list = sorted(stats.keys())
    means = [stats[nw]["mean_steps_per_sec"] for nw in num_workers_list]
    stds = [stats[nw]["std_steps_per_sec"] for nw in num_workers_list]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        num_workers_list,
        means,
        yerr=stds,
        marker="o",
        markersize=8,
        linestyle="-",
        linewidth=2,
        capsize=5,
        capthick=2,
        label="mean ± std"
    )
    
    ax.set_xlabel("Number of Workers (num_workers)", fontsize=12)
    ax.set_ylabel("Approximate Training Steps/Second", fontsize=12)
    ax.set_title("Ablation Study: Worker Parallelization Effect on Training Throughput", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {output_path}")


def print_ablation_summary(stats):
    """Print human-readable summary of ablation results."""
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY: Worker Parallelization Effect")
    print("="*70)
    print(
        f"{'num_workers':<15} {'runs':<8} {'steps/sec':<20} {'train_time':<15} {'total_time':<15}"
    )
    print("-"*70)
    
    for num_workers in sorted(stats.keys()):
        s = stats[num_workers]
        print(
            f"{num_workers:<15} "
            f"{s['count']:<8} "
            f"{s['mean_steps_per_sec']:.1f} ± {s['std_steps_per_sec']:.1f}  "
            f"{s['mean_train_wall_time_s']:.3f}s  "
            f"{s['mean_total_run_time_s']:.1f}s"
        )
    
    print("="*70 + "\n")


def main():
    args = parse_args()
    
    print(f"\n[scanning] {args.results_root} for ablation results...")
    summaries_by_workers = aggregate_ablation_results(args.results_root)
    
    if not summaries_by_workers:
        print("[ERROR] No ablation_summary.json files found!")
        return
    
    print(f"\n[found] results for {len(summaries_by_workers)} worker configurations")
    
    # Compute statistics
    stats = compute_statistics(summaries_by_workers)
    print_ablation_summary(stats)
    
    # Save report
    results_root = Path(args.results_root)
    report_path = results_root / "ablation_report.json"
    save_ablation_report(stats, report_path)
    
    # Generate plot
    plot_path = results_root / "ablation_throughput_comparison.png"
    plot_ablation_results(stats, plot_path)
    
    print("\n[done] Ablation study analysis complete!")


if __name__ == "__main__":
    main()
