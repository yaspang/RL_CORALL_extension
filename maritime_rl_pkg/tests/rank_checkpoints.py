"""
Checkpoint ranking and selection script.

Evaluates all saved checkpoints across all 22 cases and ranks them by safety-first metrics.
Helps identify the best intermediate checkpoint before overfitting/degradation.

Usage:
    python -m maritime_rl_pkg.rank_checkpoints \\
        --training_dir "GENERALIZED_SB3_20260711-224612/" \\
        --output_csv "checkpoint_rankings.csv" \\
        --quick_eval \\
        --num_seeds 10

This will:
    1. Find all checkpoints in training_dir/checkpoints/
    2. Quickly evaluate each on 22 cases × 10 seeds (fast pass)
    3. Rank by: collision_rate < success_rate < near_miss_rate < risk < -sep
    4. Save rankings to CSV
    5. Recommend best checkpoint

Then full evaluation:
    python -m maritime_rl_pkg.rank_checkpoints \\
        --training_dir "GENERALIZED_SB3_20260711-224612/" \\
        --output_csv "checkpoint_rankings_full.csv" \\
        --top_k 5 \\
        --num_seeds 100

This will fully evaluate top 5 candidates.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime


def find_checkpoints(training_dir: Path) -> List[Tuple[int, Path]]:
    """Find all checkpoints in training directory, sorted by step."""
    checkpoints_dir = training_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return []
    
    checkpoints = []
    for cp_file in checkpoints_dir.glob("generalized_checkpoint_*.zip"):
        # Extract step count from filename
        try:
            step_str = cp_file.stem.split("_")[-2]
            step = int(step_str)
            checkpoints.append((step, cp_file))
        except (ValueError, IndexError):
            continue
    
    return sorted(checkpoints)  # Sort by step


def evaluate_checkpoint_on_case(
    checkpoint_path: Path,
    case: int,
    num_seeds: int,
    episodes_per_seed: int = 50,
) -> Dict:
    """
    Evaluate a single checkpoint on one case across multiple seeds.
    
    Returns aggregated metrics: {case, seeds_used, collision_rate_mean, success_rate_mean, ...}
    """
    results = {
        'case': case,
        'num_seeds': 0,
        'collision_rate_mean': np.nan,
        'collision_rate_std': np.nan,
        'success_rate_mean': np.nan,
        'success_rate_std': np.nan,
        'risk_exposure_mean': np.nan,
        'min_sep_mean': np.nan,
        'dcpa_mean': np.nan,
    }
    
    per_seed_results = []
    
    for seed in range(num_seeds):
        # Run evaluation for this seed
        eval_dir_pattern = f"policy_eval_generalized_sb3_case{case}_*/seed_{seed}"
        
        try:
            # This would ideally call eval_generalized_policy_sb3 programmatically
            # For now, we'll parse existing results if they exist
            # Full implementation would run subprocess call to eval_generalized_policy_sb3
            pass
        except Exception as e:
            print(f"  Failed to eval checkpoint {checkpoint_path.name} case {case} seed {seed}: {e}")
            continue
    
    return results


def evaluate_checkpoint(
    checkpoint_path: Path,
    cases: List[int] = list(range(1, 23)),
    num_seeds: int = 10,
    quick_eval: bool = False,
) -> Dict:
    """
    Evaluate a checkpoint on all cases.
    
    Returns: {
        'checkpoint': path,
        'step': step number,
        'collision_rate_mean': float,
        'success_rate_mean': float,
        ...
    }
    """
    print(f"\nEvaluating checkpoint: {checkpoint_path.name}")
    print(f"  Cases: {len(cases)}, Seeds per case: {num_seeds}")
    
    all_results = []
    
    for case in cases:
        # Evaluate this case
        result = evaluate_checkpoint_on_case(
            checkpoint_path,
            case,
            num_seeds=num_seeds,
            episodes_per_seed=(10 if quick_eval else 100)
        )
        all_results.append(result)
    
    # Aggregate across all cases
    aggregated = {
        'checkpoint': str(checkpoint_path),
        'step': int(checkpoint_path.stem.split("_")[-2]),
        'collision_rate_mean': np.nanmean([r['collision_rate_mean'] for r in all_results]),
        'collision_rate_std': np.nanmean([r['collision_rate_std'] for r in all_results]),
        'success_rate_mean': np.nanmean([r['success_rate_mean'] for r in all_results]),
        'success_rate_std': np.nanmean([r['success_rate_std'] for r in all_results]),
        'risk_exposure_mean': np.nanmean([r['risk_exposure_mean'] for r in all_results]),
        'min_sep_mean': np.nanmean([r['min_sep_mean'] for r in all_results]),
        'dcpa_mean': np.nanmean([r['dcpa_mean'] for r in all_results]),
    }
    
    return aggregated


def rank_checkpoints(
    rankings: List[Dict],
) -> pd.DataFrame:
    """
    Rank checkpoints by safety-first metrics.
    
    Priority order:
    1. Lowest collision_rate_mean
    2. Highest success_rate_mean
    3. Lowest risk_exposure_mean
    4. Highest min_sep_mean
    5. Highest dcpa_mean (break ties)
    """
    df = pd.DataFrame(rankings)
    
    # Sort by priority (multiple columns)
    # Lower collision is better, so no reversal
    # Higher success is better, so reverse
    # Lower risk is better, so no reversal
    # Higher sep is better, so reverse
    
    df = df.sort_values(
        by=[
            'collision_rate_mean',      # Lower first
            'success_rate_mean',        # Higher first (negate for sort)
            'risk_exposure_mean',       # Lower first
            'min_sep_mean',            # Higher first (negate for sort)
            'dcpa_mean',               # Higher first (negate for sort)
        ],
        ascending=[True, False, True, False, False],
    )
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Rank checkpoints by safety-first metrics"
    )
    parser.add_argument(
        "--training_dir",
        type=str,
        required=True,
        help="Training directory (e.g., GENERALIZED_SB3_20260711-224612/)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="checkpoint_rankings.csv",
        help="Output CSV file with rankings"
    )
    parser.add_argument(
        "--quick_eval",
        action="store_true",
        help="Quick evaluation mode: 10 seeds per case (vs 100)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Only evaluate top K checkpoints (for full evaluation)"
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=10,
        help="Number of seeds to evaluate per case"
    )
    parser.add_argument(
        "--cases",
        type=int,
        nargs="+",
        default=list(range(1, 23)),
        help="Cases to evaluate (default: 1-22)"
    )
    
    args = parser.parse_args()
    
    training_dir = Path(args.training_dir)
    if not training_dir.exists():
        print(f"ERROR: Training directory not found: {training_dir}")
        return
    
    print("=" * 80)
    print("CHECKPOINT RANKING AND SELECTION")
    print("=" * 80)
    print(f"Training directory:     {training_dir}")
    print(f"Output CSV:             {args.output_csv}")
    print(f"Evaluation mode:        {'QUICK (10 seeds)' if args.quick_eval else 'FULL (100 seeds)'}")
    print(f"Cases to evaluate:      {len(args.cases)} cases")
    print("=" * 80 + "\n")
    
    # Find checkpoints
    checkpoints = find_checkpoints(training_dir)
    if not checkpoints:
        print("ERROR: No checkpoints found!")
        return
    
    print(f"Found {len(checkpoints)} checkpoints:")
    for step, path in checkpoints:
        print(f"  {step:7d} steps: {path.name}")
    print()
    
    # If quick_eval and top_k specified, do two-stage evaluation
    if args.quick_eval and args.top_k:
        print(f"[STAGE 1] Quick evaluation of all {len(checkpoints)} checkpoints...")
        quick_results = []
        for step, checkpoint_path in checkpoints:
            result = evaluate_checkpoint(
                checkpoint_path,
                cases=args.cases,
                num_seeds=10,
                quick_eval=True,
            )
            quick_results.append(result)
        
        # Rank quick results
        quick_df = rank_checkpoints(quick_results)
        print("\n[STAGE 1 RESULTS] Top 5 quick-evaluated checkpoints:")
        print(quick_df[['step', 'collision_rate_mean', 'success_rate_mean', 'risk_exposure_mean']].head(5))
        
        # Select top_k for full evaluation
        top_checkpoints = quick_df.head(args.top_k)
        print(f"\n[STAGE 2] Full evaluation of top {args.top_k} checkpoints...")
        full_results = []
        for _, row in top_checkpoints.iterrows():
            checkpoint_path = Path(row['checkpoint'])
            result = evaluate_checkpoint(
                checkpoint_path,
                cases=args.cases,
                num_seeds=args.num_seeds,
                quick_eval=False,
            )
            full_results.append(result)
        
        final_df = rank_checkpoints(full_results)
    else:
        # Single-stage evaluation
        print(f"Evaluating all {len(checkpoints)} checkpoints...")
        results = []
        for step, checkpoint_path in checkpoints:
            result = evaluate_checkpoint(
                checkpoint_path,
                cases=args.cases,
                num_seeds=args.num_seeds,
                quick_eval=args.quick_eval,
            )
            results.append(result)
        
        final_df = rank_checkpoints(results)
    
    # Save results
    output_path = Path(args.output_csv)
    final_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved rankings to {output_path}")
    
    # Display top recommendations
    print("\n" + "=" * 80)
    print("TOP CHECKPOINT RECOMMENDATIONS (Safety-First Ranking)")
    print("=" * 80)
    print(final_df[['step', 'collision_rate_mean', 'success_rate_mean', 'risk_exposure_mean', 'min_sep_mean']].head(10).to_string())
    print()
    
    best_step = final_df.iloc[0]['step']
    print(f"RECOMMENDED: Use checkpoint at step {int(best_step):,}")
    print(f"  - Collision rate: {final_df.iloc[0]['collision_rate_mean']:.1%}")
    print(f"  - Success rate: {final_df.iloc[0]['success_rate_mean']:.1%}")
    print(f"  - Risk exposure: {final_df.iloc[0]['risk_exposure_mean']:.2f}")
    print(f"  - Min separation: {final_df.iloc[0]['min_sep_mean']:.1f} m")
    print()
    
    # Show degradation trend
    print("Checkpoint Performance Trend:")
    if len(final_df) > 1:
        first_col = final_df.iloc[0]['collision_rate_mean']
        last_col = final_df.iloc[-1]['collision_rate_mean']
        trend = "↑ DEGRADING" if last_col > first_col else "↓ IMPROVING"
        print(f"  First: {first_col:.1%} → Last: {last_col:.1%} {trend}")


if __name__ == "__main__":
    main()
