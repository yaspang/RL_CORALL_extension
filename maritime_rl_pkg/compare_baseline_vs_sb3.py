"""
Compare baseline (CORALL rule-based) vs trained SB3 policy performance across cases.

Generates comprehensive comparison plots:
  - Side-by-side trajectories (overlay visualization)
  - DCPA/TCPA/Range timeseries
  - Risk exposure timeseries
  - Metric summary tables
  - Encounter detail clean plots

USAGE:
======

Basic usage (all 3 cases):
  python -m maritime_rl_pkg.compare_baseline_vs_sb3

Specific case:
  python -m maritime_rl_pkg.compare_baseline_vs_sb3 --cases 1 6 21

Custom evaluation directories:
  python -m maritime_rl_pkg.compare_baseline_vs_sb3 \\
    --baseline_dirs "corall_baseline_case1_20260416-110958/seed_0" \\
    --sb3_dirs "policy_eval_single_sb3_case1_20260416-111254/seed_0" \\
    --cases 1

OUTPUT:
=======
results_comparison/
├── case_1/
│   ├── trajectory_overlay.png
│   ├── dcpa_tcpa_range_timeseries.png
│   ├── risk_timeseries.png
│   ├── encounter_detail.png
│   └── metrics_summary.txt
├── case_6/
│   └── [same structure]
└── case_21/
    └── [same structure]

METRICS INCLUDED:
=================
- Success rate (% waypoint reached)
- Collision rate (% episodes with collision)
- Min DCPA (meters)
- Mean Risk Exposure
- Path length traveled
- Completion time
- Goal progress

INSTALLATION NOTES:
===================
Requires: matplotlib, numpy, pandas (for summary tables)
Uses episode_overlay_tools.py functions for plotting
"""

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .episode_overlay_tools import (
    load_episode_history,
    dcpa_series,
    risk_series,
    plot_full_trajectory_overlay,
    plot_risk_timeseries,
    plot_min_dcpa_timeseries,
    plot_encounter_detail_clean,
)

NMI = 1852.0


def find_best_episode_dir(eval_dir: Path) -> Optional[Path]:
    """
    Find episode_histories directory and return path to best episode.
    Best = highest success + lowest collision + highest return.
    """
    histories_dir = eval_dir / "episode_histories"
    if not histories_dir.exists():
        print(f"  ✗ No episode_histories in {eval_dir}")
        return None
    
    # Load summary JSON to find best episode
    summary_path = eval_dir / "policy_eval_summary.json"
    if not summary_path.exists():
        summary_path = eval_dir / "policy_eval_summary_VIS.json"
    
    if not summary_path.exists():
        print(f"  ✗ No summary file in {eval_dir}")
        return None
    
    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Try to get per-episode metrics
        per_episode = summary.get("per_episode_metrics", [])
        if not per_episode:
            # Fallback: just use first episode
            episodes = list(histories_dir.glob("*.npz"))
            if episodes:
                return sorted(episodes)[0]
            return None
        
        # Score each episode: prioritize success, then collision, then return
        best_idx = 0
        best_score = float('-inf')
        
        for i, metrics in enumerate(per_episode):
            success = metrics.get("success_ownship", 0)
            collision = metrics.get("collision_any", 0)
            ret = metrics.get("episode_return", 0)
            
            # Score: success * 1000 - collision * 500 + return
            score = success * 1000 - collision * 500 + ret
            if score > best_score:
                best_score = score
                best_idx = i
        
        # Find corresponding episode file
        episodes = sorted(list(histories_dir.glob("*.npz")))
        if best_idx < len(episodes):
            return episodes[best_idx]
        
        return episodes[0] if episodes else None
        
    except Exception as e:
        print(f"  ✗ Error finding best episode: {e}")
        return None


def load_eval_summary(eval_dir: Path) -> Optional[Dict]:
    """Load evaluation summary JSON."""
    summary_path = eval_dir / "policy_eval_summary.json"
    if not summary_path.exists():
        summary_path = eval_dir / "policy_eval_summary_VIS.json"
    
    if not summary_path.exists():
        return None
    
    try:
        with open(summary_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"  ✗ Error loading summary: {e}")
        return None


def create_metrics_table(baseline_summary: Dict, sb3_summary: Dict, case: int) -> str:
    """Generate formatted metrics comparison table."""
    
    metrics_to_compare = [
        ("Success Rate", "success_rate_ownship_mean"),
        ("Collision Rate", "collision_rate"),
        ("Min DCPA (m)", "min_dcpa_m_ownship_mean"),
        ("Min TCPA (s)", "min_tcpa_s_ownship_mean"),
        ("Risk Exposure", "risk_exposure_ownship_mean"),
        ("Path Length (m)", "path_length_m_ownship_mean"),
        ("Goal Progress (%)", "goal_progress_ownship_mean"),
        ("Completion Time (s)", "completion_time_s_ownship_mean"),
    ]
    
    lines = [
        f"\n{'='*70}",
        f"Case {case} Comparison: CORALL Baseline vs SB3 Policy",
        f"{'='*70}\n",
        f"{'Metric':<25} {'Baseline':<20} {'SB3 Policy':<20}",
        f"{'-'*70}",
    ]
    
    for metric_name, key in metrics_to_compare:
        baseline_val = baseline_summary.get(key, np.nan)
        sb3_val = sb3_summary.get(key, np.nan)
        
        # Format values
        if "Rate" in metric_name or "Progress" in metric_name:
            baseline_str = f"{baseline_val:.1%}" if not np.isnan(baseline_val) else "N/A"
            sb3_str = f"{sb3_val:.1%}" if not np.isnan(sb3_val) else "N/A"
        elif "Time" in metric_name or "TCPA" in metric_name:
            baseline_str = f"{baseline_val:.1f}" if not np.isnan(baseline_val) else "N/A"
            sb3_str = f"{sb3_val:.1f}" if not np.isnan(sb3_val) else "N/A"
        else:
            baseline_str = f"{baseline_val:.1f}" if not np.isnan(baseline_val) else "N/A"
            sb3_str = f"{sb3_val:.1f}" if not np.isnan(sb3_val) else "N/A"
        
        lines.append(f"{metric_name:<25} {baseline_str:<20} {sb3_str:<20}")
    
    lines.append(f"{'='*70}\n")
    return "\n".join(lines)


def generate_case_comparison(
    case: int,
    baseline_dir: Optional[Path] = None,
    sb3_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Generate all comparison plots and tables for a single case."""
    
    if output_dir is None:
        output_dir = Path("results_comparison")
    
    output_dir = output_dir / f"case_{case}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Generating Case {case} Comparison")
    print(f"{'='*70}")
    
    # Auto-detect directories if not provided
    if baseline_dir is None:
        baseline_dirs = sorted(Path(".").glob(f"corall_baseline_case{case}_*/seed_0"))
        if baseline_dirs:
            baseline_dir = baseline_dirs[-1]  # Most recent
            print(f"✓ Found baseline: {baseline_dir}")
        else:
            print(f"✗ No baseline evaluation found for case {case}")
            return
    
    if sb3_dir is None:
        sb3_dirs = sorted(Path(".").glob(f"policy_eval_single_sb3_case{case}_*/seed_0"))
        if sb3_dirs:
            sb3_dir = sb3_dirs[-1]  # Most recent
            print(f"✓ Found SB3 eval: {sb3_dir}")
        else:
            print(f"✗ No SB3 evaluation found for case {case}")
            return
    
    # Find best episodes
    print("  Finding best episodes...")
    baseline_ep_path = find_best_episode_dir(baseline_dir)
    sb3_ep_path = find_best_episode_dir(sb3_dir)
    
    if baseline_ep_path is None or sb3_ep_path is None:
        print("  ✗ Could not find best episodes")
        return
    
    print(f"  ✓ Baseline best: {baseline_ep_path.name}")
    print(f"  ✓ SB3 best: {sb3_ep_path.name}")
    
    # Load episode histories
    baseline_hist = load_episode_history(baseline_ep_path)
    sb3_hist = load_episode_history(sb3_ep_path)
    
    # Load summaries for metrics table
    baseline_summary = load_eval_summary(baseline_dir)
    sb3_summary = load_eval_summary(sb3_dir)
    
    # Generate plots
    print("  Generating plots...")
    
    try:
        # Plot 1: Trajectory overlay
        fig, ax = plt.subplots(figsize=(12, 10))
        plot_full_trajectory_overlay(
            baseline_hist=baseline_hist,
            rl_hist=sb3_hist,
            ax=ax,
            title=f"Case {case}: Trajectory Comparison (Baseline vs SB3)",
        )
        fig.savefig(output_dir / "trajectory_overlay.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("    ✓ Saved: trajectory_overlay.png")
    except Exception as e:
        print(f"    ✗ Error generating trajectory plot: {e}")
    
    try:
        # Plot 2: DCPA, TCPA, Range timeseries (requires custom implementation)
        # For now, use DCPA timeseries
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # DCPA timeseries
        t_baseline = np.asarray(baseline_hist["t"], dtype=float)
        t_sb3 = np.asarray(sb3_hist["t"], dtype=float)
        
        dcpa_baseline = dcpa_series(np.asarray(baseline_hist["pair_dcpa"], dtype=float)) / NMI
        dcpa_sb3 = dcpa_series(np.asarray(sb3_hist["pair_dcpa"], dtype=float)) / NMI
        
        axes[0].plot(t_baseline, dcpa_baseline, label="Baseline", linewidth=2)
        axes[0].plot(t_sb3, dcpa_sb3, label="SB3 Policy", linewidth=2)
        axes[0].set_ylabel("Min DCPA (nmi)")
        axes[0].set_title(f"Case {case}: Distance of Closest Point of Approach")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Risk timeseries
        risk_baseline = risk_series(np.asarray(baseline_hist["pair_risk"], dtype=float))
        risk_sb3 = risk_series(np.asarray(sb3_hist["pair_risk"], dtype=float))
        
        axes[1].plot(t_baseline, risk_baseline, label="Baseline", linewidth=2)
        axes[1].plot(t_sb3, risk_sb3, label="SB3 Policy", linewidth=2)
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Risk Exposure")
        axes[1].set_title("Risk Timeseries")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(output_dir / "dcpa_risk_timeseries.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("    ✓ Saved: dcpa_risk_timeseries.png")
    except Exception as e:
        print(f"    ✗ Error generating timeseries plot: {e}")
    
    # Save metrics table
    try:
        if baseline_summary and sb3_summary:
            metrics_table = create_metrics_table(baseline_summary, sb3_summary, case)
            with open(output_dir / "metrics_summary.txt", 'w') as f:
                f.write(metrics_table)
            print("    ✓ Saved: metrics_summary.txt")
            print(metrics_table)
    except Exception as e:
        print(f"    ✗ Error generating metrics table: {e}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare CORALL baseline vs trained SB3 policy across cases"
    )
    p.add_argument(
        "--cases",
        type=int,
        nargs="+",
        default=[1, 6, 21],
        help="CORALL cases to compare (default: 1 6 21)",
    )
    p.add_argument(
        "--baseline_dirs",
        type=str,
        nargs="+",
        help="Custom baseline evaluation directories (one per case)",
    )
    p.add_argument(
        "--sb3_dirs",
        type=str,
        nargs="+",
        help="Custom SB3 evaluation directories (one per case)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="results_comparison",
        help="Output directory for comparison plots (default: results_comparison/)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    
    print(f"\n{'='*70}")
    print("BASELINE vs SB3 POLICY COMPARISON")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each case
    for i, case in enumerate(args.cases):
        baseline_dir = None
        sb3_dir = None
        
        if args.baseline_dirs and i < len(args.baseline_dirs):
            baseline_dir = Path(args.baseline_dirs[i])
        
        if args.sb3_dirs and i < len(args.sb3_dirs):
            sb3_dir = Path(args.sb3_dirs[i])
        
        generate_case_comparison(
            case=case,
            baseline_dir=baseline_dir,
            sb3_dir=sb3_dir,
            output_dir=output_dir,
        )
    
    print(f"\n{'='*70}")
    print(f"✓ Comparison complete! Results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
