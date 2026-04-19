"""
Aggregate and compare policy_eval_summary statistics across multiple cases.

Usage:
    python -m maritime_rl_pkg.compare_case_summaries \
        --eval_dirs policy_eval_case1_... policy_eval_case6_... policy_eval_case21_... \
        --output_dir comparison_results \
        --case_numbers 1 6 21 \
        --plot_metrics

Or for all 23 cases:
    python -m maritime_rl_pkg.compare_case_summaries \
        --eval_dir_pattern "policy_eval_single_sb3_case*" \
        --output_dir comparison_results \
        --plot_metrics
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd


def load_eval_summary(summary_path: str | Path) -> Dict:
    """Load a single policy_eval_summary.json file."""
    with open(summary_path, "r") as f:
        return json.load(f)


def find_case_eval_dirs(base_dir: str | Path, case_numbers: Optional[List[int]] = None) -> Dict[int, Path]:
    """
    Find evaluation directories for specific cases.
    Looks for patterns like: policy_eval_single_sb3_case1_*
    Extracts case number from directory name automatically.
    """
    import re
    base_dir = Path(base_dir)
    case_dirs = {}
    
    if case_numbers:
        # Search for specific case numbers
        for case_num in case_numbers:
            pattern = f"*case{case_num}_*"
            matches = list(base_dir.glob(pattern))
            if matches:
                # Take most recent (latest timestamp)
                latest = max(matches, key=lambda p: p.stat().st_mtime)
                case_dirs[case_num] = latest
    else:
        # Search for all case* directories and extract case number
        for path in base_dir.glob("*case*"):
            if path.is_dir():
                # Extract case number from name (e.g., "policy_eval_single_sb3_case5_20260417-001017" -> 5)
                match = re.search(r'case(\d+)', path.name)
                if match:
                    case_num = int(match.group(1))
                    case_dirs[case_num] = path
    
    return case_dirs


def extract_metrics_from_summary(summary: Dict) -> Dict[str, float]:
    """
    Extract key metrics from a policy_eval_summary.json.
    
    Handles the actual structure from eval script which contains:
    - episodes: total number of episodes
    - aggregate_metrics: contains mean/std values for all metrics
    
    Key metrics:
    - episode_return_mean: average episode return
    - collision_any_mean: collision rate (fraction)
    - near_miss_any_mean: near-miss rate
    - min_dcpa_m_ownship_mean: mean predicted DCPA
    - min_actual_sep_m_ownship_mean: mean actual minimum separation
    - min_tcpa_s_ownship_mean: mean time-to-collision
    """
    metrics = {}
    
    # Basic stats
    metrics["total_episodes"] = summary.get("episodes", 0)
    
    # Extract from aggregate_metrics if available
    agg = summary.get("aggregate_metrics", {})
    
    # Return stats
    metrics["mean_return"] = agg.get("episode_return_mean", 0) or 0
    metrics["std_return"] = agg.get("episode_return_std", 0) or 0
    
    # Safety rates (0-1)
    collision_rate = agg.get("collision_any_mean", 0)
    metrics["collision_rate"] = collision_rate if collision_rate is not None and not np.isinf(collision_rate) else 0
    metrics["collision_count"] = int(metrics["total_episodes"] * metrics["collision_rate"]) if metrics["collision_rate"] else 0
    
    near_miss_rate = agg.get("near_miss_any_mean", 0)
    metrics["near_miss_rate"] = near_miss_rate if near_miss_rate is not None and not np.isinf(near_miss_rate) else 0
    metrics["near_miss_count"] = int(metrics["total_episodes"] * metrics["near_miss_rate"]) if metrics["near_miss_rate"] else 0
    
    # Distance/CPA stats (handle Infinity values)
    dcpa_val = agg.get("min_dcpa_m_ownship_mean", np.nan)
    metrics["min_dcpa_m"] = dcpa_val if dcpa_val is not None and not np.isinf(dcpa_val) else np.nan
    metrics["mean_dcpa_m"] = dcpa_val if dcpa_val is not None and not np.isinf(dcpa_val) else np.nan
    
    sep_val = agg.get("min_actual_sep_m_ownship_mean", np.nan)
    metrics["min_actual_sep_m"] = sep_val if sep_val is not None else np.nan
    metrics["mean_actual_sep_m"] = sep_val if sep_val is not None else np.nan
    
    # Time-to-collision stats
    tcpa_val = agg.get("min_tcpa_s_ownship_mean", np.nan)
    metrics["min_tcpa_s"] = tcpa_val if tcpa_val is not None else np.nan
    metrics["mean_tcpa_s"] = tcpa_val if tcpa_val is not None else np.nan
    
    # Success rate
    metrics["success_rate"] = agg.get("success_ownship_mean", 0) or 0
    
    return metrics


def aggregate_case_metrics(eval_dirs: Dict[int, Path]) -> pd.DataFrame:
    """
    Load and aggregate metrics from all case evaluation directories.
    Looks in both root and seed_0/ subdirectories.
    
    Returns DataFrame with cases as rows and metrics as columns.
    """
    all_metrics = {}
    
    for case_num in sorted(eval_dirs.keys()):
        eval_dir = eval_dirs[case_num]
        
        # Try multiple locations for summary file
        possible_paths = [
            eval_dir / "policy_eval_summary.json",  # Root
            eval_dir / "seed_0" / "policy_eval_summary.json",  # seed_0 subdirectory
        ]
        
        summary_path = None
        for path in possible_paths:
            if path.exists():
                summary_path = path
                break
        
        if summary_path:
            summary = load_eval_summary(summary_path)
            all_metrics[case_num] = extract_metrics_from_summary(summary)
            print(f"  Case {case_num}: Found summary at {summary_path}")
        else:
            print(f"Warning: No summary found for Case {case_num}")
            print(f"  Tried: {possible_paths}")
    
    # Convert to DataFrame
    if not all_metrics:
        print("Error: No summary files were found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_metrics).T
    df.index.name = "Case"
    return df


def print_summary_table(df: pd.DataFrame, key_metrics: Optional[List[str]] = None):
    """Print a formatted summary table of key metrics."""
    if key_metrics is None:
        key_metrics = [
            "mean_return", 
            "collision_rate", 
            "near_miss_rate",
            "mean_dcpa_m",
            "mean_actual_sep_m",
        ]
    
    # Filter to available columns
    available_cols = [col for col in key_metrics if col in df.columns]
    summary_df = df[available_cols].copy()
    
    print("\n" + "="*100)
    print("CASE-BY-CASE EVALUATION SUMMARY")
    print("="*100)
    print(summary_df.to_string())
    print("="*100)


def compute_aggregate_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Compute aggregate statistics across all cases."""
    stats = {}
    
    if df.empty:
        print("Error: DataFrame is empty, cannot compute statistics")
        return stats
    
    # Average performance metrics
    stats["avg_return"] = df["mean_return"].mean() if "mean_return" in df.columns else np.nan
    stats["avg_collision_rate"] = df["collision_rate"].mean() if "collision_rate" in df.columns else np.nan
    stats["avg_near_miss_rate"] = df["near_miss_rate"].mean() if "near_miss_rate" in df.columns else np.nan
    
    # Average safety metrics
    stats["avg_min_dcpa"] = df["min_dcpa_m"].mean() if "min_dcpa_m" in df.columns else np.nan
    stats["avg_mean_dcpa"] = df["mean_dcpa_m"].mean() if "mean_dcpa_m" in df.columns else np.nan
    stats["avg_min_sep"] = df["min_actual_sep_m"].mean() if "min_actual_sep_m" in df.columns else np.nan
    stats["avg_mean_sep"] = df["mean_actual_sep_m"].mean() if "mean_actual_sep_m" in df.columns else np.nan
    
    # Time-to-collision
    stats["avg_min_tcpa"] = df["min_tcpa_s"].mean() if "min_tcpa_s" in df.columns else np.nan
    stats["avg_mean_tcpa"] = df["mean_tcpa_s"].mean() if "mean_tcpa_s" in df.columns else np.nan
    
    # Total collisions
    stats["total_collisions"] = df["collision_count"].sum() if "collision_count" in df.columns else 0
    stats["total_near_misses"] = df["near_miss_count"].sum() if "near_miss_count" in df.columns else 0
    stats["total_episodes"] = df["total_episodes"].sum() if "total_episodes" in df.columns else 0
    
    return stats


def plot_case_metrics(df: pd.DataFrame, output_dir: str | Path):
    """Generate comparison plots across cases."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Return by case
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df.index, df["mean_return"], color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Case Number")
    ax.set_ylabel("Mean Return")
    ax.set_title("Mean Return by Case")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "01_returns_by_case.png", dpi=300, bbox_inches="tight", format='png')
    fig.savefig(output_dir / "01_returns_by_case.pdf", bbox_inches="tight", format='pdf')
    plt.close()
    
    # 2. Collision and near-miss rates
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df["collision_rate"], width, label="Collision Rate", color="red", alpha=0.7)
    ax.bar(x + width/2, df["near_miss_rate"], width, label="Near-Miss Rate", color="orange", alpha=0.7)
    ax.set_xlabel("Case Number")
    ax.set_ylabel("Rate (fraction)")
    ax.set_title("Safety Metrics by Case")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "02_safety_rates_by_case.png", dpi=300, bbox_inches="tight", format='png')
    fig.savefig(output_dir / "02_safety_rates_by_case.pdf", bbox_inches="tight", format='pdf')
    plt.close()
    
    # 3. Distance metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["mean_dcpa_m"], marker="o", label="Mean DCPA", linewidth=2, markersize=8)
    ax.plot(df.index, df["mean_actual_sep_m"], marker="s", label="Mean Actual Sep", linewidth=2, markersize=8)
    ax.axhline(y=60, color="red", linestyle="--", label="Collision Threshold (60m)", alpha=0.7)
    ax.set_xlabel("Case Number")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Distance Metrics by Case")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "03_distance_metrics_by_case.png", dpi=300, bbox_inches="tight", format='png')
    fig.savefig(output_dir / "03_distance_metrics_by_case.pdf", bbox_inches="tight", format='pdf')
    plt.close()
    
    # 4. Time-to-collision
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["mean_tcpa_s"], marker="o", label="Mean TCPA", linewidth=2, markersize=8)
    ax.fill_between(range(len(df)), 0, df["mean_tcpa_s"], alpha=0.3)
    ax.set_xlabel("Case Number")
    ax.set_ylabel("Time to Collision (s)")
    ax.set_title("Mean Time-to-Collision by Case")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "04_tcpa_by_case.png", dpi=300, bbox_inches="tight", format='png')
    fig.savefig(output_dir / "04_tcpa_by_case.pdf", bbox_inches="tight", format='pdf')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and compare policy_eval_summary statistics across multiple cases"
    )
    
    # Allow either specific eval dirs or pattern-based search
    parser.add_argument(
        "--eval_dirs",
        type=str,
        nargs="+",
        default=None,
        help="Specific evaluation directories to compare"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="Base directory to search for eval_*_case* patterns"
    )
    parser.add_argument(
        "--case_numbers",
        type=int,
        nargs="+",
        default=None,
        help="Case numbers to include (e.g., 1 6 21). If None, searches for all."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_results",
        help="Output directory for results and plots"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots"
    )
    parser.add_argument(
        "--save_summary",
        action="store_true",
        default=True,
        help="Save summary to CSV"
    )
    
    args = parser.parse_args()
    
    # Find evaluation directories
    if args.eval_dirs:
        # Parse explicit directories, filtering out empty/whitespace args
        valid_dirs = [d.strip() for d in args.eval_dirs if d.strip()]
        eval_dirs = {}
        for i, dir_path in enumerate(valid_dirs):
            # Try to extract case number from directory name
            import re
            match = re.search(r'case(\d+)', dir_path)
            if match:
                case_num = int(match.group(1))
                eval_dirs[case_num] = Path(dir_path)
            else:
                # Fallback: use index if no case number found
                eval_dirs[i] = Path(dir_path)
    else:
        print(f"Searching for case evaluation directories in: {args.base_dir}")
        eval_dirs = find_case_eval_dirs(args.base_dir, args.case_numbers)
    
    if not eval_dirs:
        print("Error: No evaluation directories found!")
        return
    
    print(f"Found {len(eval_dirs)} case evaluation directories:")
    for case_num, dir_path in sorted(eval_dirs.items()):
        print(f"  Case {case_num}: {dir_path.name if hasattr(dir_path, 'name') else dir_path}")
    
    # Load and aggregate metrics
    df = aggregate_case_metrics(eval_dirs)
    
    # Print summary
    print_summary_table(df)
    
    # Compute aggregate statistics
    agg_stats = compute_aggregate_stats(df)
    print("\n" + "="*100)
    print("AGGREGATE STATISTICS")
    print("="*100)
    for key, val in agg_stats.items():
        if isinstance(val, float):
            print(f"  {key:.<30} {val:>15.4f}")
        else:
            print(f"  {key:.<30} {val:>15}")
    print("="*100)
    
    # Save summary
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_summary:
        csv_path = output_dir / "case_summaries.csv"
        df.to_csv(csv_path)
        print(f"\nSaved summary CSV to: {csv_path}")
    
    # Generate plots
    if args.plot:
        plot_case_metrics(df, output_dir)
        print(f"Saved comparison plots to: {output_dir}")


if __name__ == "__main__":
    main()
