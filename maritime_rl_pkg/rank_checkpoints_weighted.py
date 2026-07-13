"""
Checkpoint ranking with explicit weighted scoring.

Evaluates all checkpoints and ranks by a safety-focused weighted score:

    score = w_collision * collision_rate 
          + w_success * (1 - success_rate)
          + w_risk * risk_exposure
          + w_sep * (1 - normalized_min_sep)
          
Lower score is better.

Weights are configurable and transparent. Default prioritizes:
    1. Collision avoidance (50% weight)
    2. Mission success (30% weight)
    3. Risk management (15% weight)
    4. Separation maintenance (5% weight)

Usage:
    python -m maritime_rl_pkg.rank_checkpoints_weighted \\
        --training_dir "GENERALIZED_SB3_TIMESTAMP/" \\
        --output_csv "checkpoint_scores.csv" \\
        --quick_eval \\
        --num_seeds 10 \\
        --w_collision 0.5 \\
        --w_success 0.3 \\
        --w_risk 0.15 \\
        --w_sep 0.05
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from collections import defaultdict


def find_checkpoints(training_dir: Path) -> List[Tuple[int, Path]]:
    """Find all saved checkpoints, sorted by step."""
    checkpoints_dir = training_dir / "checkpoints"
    if not checkpoints_dir.exists():
        print(f"ERROR: Checkpoints directory not found: {checkpoints_dir}")
        return []
    
    checkpoints = []
    for cp_file in checkpoints_dir.glob("generalized_checkpoint_*.zip"):
        try:
            # Extract step from filename: generalized_checkpoint_500000_steps.zip
            parts = cp_file.stem.split("_")
            step = int(parts[-2])
            checkpoints.append((step, cp_file))
        except (ValueError, IndexError):
            continue
    
    # Sort by numeric step, not lexicographic filename
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def evaluate_checkpoint_on_case(
    checkpoint_path: Path,
    case: int,
    num_seeds: int = 10,
    output_dir: str = ".",
    desired_cross_x_nmi: float = 1.05,
    target_speed_mps: float = 10.0,
    ownship_speed_mps: float = 10.0,
    sim_time: float = 900.0,
) -> Optional[Dict]:
    """
    Evaluate checkpoint on single case by calling eval_generalized_policy_sb3.
    
    Returns aggregated metrics over all seeds, or None if evaluation fails.
    """
    try:
        # Call evaluator with matching encounter geometry parameters
        cmd = [
            "python", "-m", "maritime_rl_pkg.eval_generalized_policy_sb3",
            "--checkpoint", str(checkpoint_path),
            "--case", str(case),
            "--episodes", str(num_seeds),
            "--seed", "0",
            "--output_dir", output_dir,
            "--desired_cross_x_nmi", str(desired_cross_x_nmi),
            "--target_speed_mps", str(target_speed_mps),
            "--sim_time", str(sim_time),
        ]
        # Only pass ownship_speed_mps if explicitly specified (None = use case native speed)
        if ownship_speed_mps is not None:
            cmd += ["--ownship_speed_mps", str(ownship_speed_mps)]
        
        # Set proper encoding for Windows subprocess output
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            env=env,
        )
        if result.returncode != 0:
            print(f"\n    Eval failed (case {case})")
            print(f"    Command:")
            print(f"    " + " ".join(cmd))
            print(f"\n    ---- STDOUT tail ----")
            print(result.stdout[-3000:])
            print(f"\n    ---- STDERR tail ----")
            print(result.stderr[-5000:])
            return None
        
        # Parse policy_eval_summary.json (it's in seed_0 subdirectory)
        summary_path = Path(output_dir) / "seed_0" / "policy_eval_summary.json"
        if not summary_path.exists():
            print(f"    Summary not found: {summary_path}")
            return None
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        # Extract aggregate metrics (evaluator saves *_mean and *_std keys)
        required_keys = [
            "collision_any_mean",
            "success_ownship_mean",
            "risk_exposure_ownship_mean",
            "min_actual_sep_m_ownship_mean",
        ]
        
        missing = [k for k in required_keys if k not in summary]
        if missing:
            print(f"    Missing keys in summary: {missing}")
            return None
        
        return {
            'collision_rate': float(summary["collision_any_mean"]),
            'success_rate': float(summary["success_ownship_mean"]),
            'risk_exposure': float(summary["risk_exposure_ownship_mean"]),
            'min_sep': float(summary["min_actual_sep_m_ownship_mean"]),
            'completion_time': float(summary.get("completion_time_s_ownship_mean", np.nan)),
            'num_episodes': int(summary.get("num_seeds", 1)),
        }
    
    except subprocess.TimeoutExpired:
        print(f"    Timeout on case {case}")
        return None
    except Exception as e:
        print(f"    ERROR on case {case}: {e}")
        return None


def compute_weighted_score(
    collision_rate: float,
    success_rate: float,
    risk_exposure: float,
    min_sep: float,
    w_collision: float = 0.50,
    w_success: float = 0.30,
    w_risk: float = 0.15,
    w_sep: float = 0.05,
    sep_max: float = 500.0,  # Normalization reference for separation
) -> float:
    """
    Compute weighted safety score. Lower is better.
    
    Args:
        collision_rate: [0, 1] - proportion of episodes with collision
        success_rate: [0, 1] - proportion of episodes reaching goal
        risk_exposure: [0, ∞] - average risk metric
        min_sep: [0, ∞] - average minimum separation (meters)
        w_collision, w_success, w_risk, w_sep: Weights (should sum to 1.0)
        sep_max: Reference value for normalizing separation (higher is better)
    
    Returns:
        score: [0, ∞) - Lower is better
    """
    # Normalize separation (higher sep = lower contribution)
    # Cap at sep_max for numerical stability
    sep_normalized = max(0.0, 1.0 - min(min_sep / sep_max, 1.0))
    
    # Normalize risk to [0, 1] range (assume risk_exposure typically 0-100)
    risk_normalized = min(risk_exposure / 100.0, 1.0)
    
    # Compute weighted score
    score = (
        w_collision * collision_rate +
        w_success * (1.0 - success_rate) +
        w_risk * risk_normalized +
        w_sep * sep_normalized
    )
    
    return score


def format_checkpoint_row(step: int, metrics: Dict, score: float) -> Dict:
    """Format checkpoint metrics for DataFrame."""
    return {
        'step': step,
        'collision_rate': metrics.get('collision_rate', np.nan),
        'success_rate': metrics.get('success_rate', np.nan),
        'risk_exposure': metrics.get('risk_exposure', np.nan),
        'min_sep_m': metrics.get('min_sep', np.nan),
        'completion_time_s': metrics.get('completion_time', np.nan),
        'weighted_score': score,
        'acceptable': (
            metrics.get('collision_rate', 1.0) == 0.0
            and metrics.get('success_rate', 0.0) == 1.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Rank checkpoints using weighted safety score"
    )
    parser.add_argument(
        "--training_dir",
        type=str,
        required=True,
        help="Training output directory (e.g., GENERALIZED_SB3_20260711-224612/)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="checkpoint_scores.csv",
        help="Output CSV with scores"
    )
    parser.add_argument(
        "--quick_eval",
        action="store_true",
        help="Quick mode: 10 seeds/case (vs 100)"
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=10,
        help="Seeds to evaluate per case"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Only fully evaluate top K quick-pass candidates"
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Smoke test mode: evaluate only checkpoint 50k + case 1 for validation"
    )
    
    # Weights (should sum to 1.0)
    parser.add_argument(
        "--desired_cross_x_nmi",
        type=float,
        default=1.05,
        help="Encounter crossing distance in NMI (must match training; default: 1.05)"
    )
    parser.add_argument(
        "--target_speed_mps",
        type=float,
        default=10.0,
        help="Target vessel speed in m/s (default: 10.0)"
    )
    parser.add_argument(
        "--ownship_speed_mps",
        type=float,
        default=None,
        help="Ownship speed in m/s. Omit to use each case's native speed (matches training with ownship_speed_mps=null)"
    )
    parser.add_argument(
        "--sim_time",
        type=float,
        default=900.0,
        help="Episode horizon in seconds passed to evaluator (default: 900.0)"
    )

    # Weights (should sum to 1.0)
    parser.add_argument(
        "--w_collision",
        type=float,
        default=0.50,
        help="Weight for collision rate (default: 0.50)"
    )
    parser.add_argument(
        "--w_success",
        type=float,
        default=0.30,
        help="Weight for success rate (default: 0.30)"
    )
    parser.add_argument(
        "--w_risk",
        type=float,
        default=0.15,
        help="Weight for risk exposure (default: 0.15)"
    )
    parser.add_argument(
        "--w_sep",
        type=float,
        default=0.05,
        help="Weight for separation maintenance (default: 0.05)"
    )
    
    args = parser.parse_args()
    
    # Verify weights
    total_weight = args.w_collision + args.w_success + args.w_risk + args.w_sep
    if abs(total_weight - 1.0) > 0.01:
        print(f"WARNING: Weights sum to {total_weight:.2f}, should be 1.0")
        # Normalize
        args.w_collision /= total_weight
        args.w_success /= total_weight
        args.w_risk /= total_weight
        args.w_sep /= total_weight
    
    training_dir = Path(args.training_dir)
    if not training_dir.exists():
        print(f"ERROR: Training directory not found: {training_dir}")
        return
    
    print("=" * 80)
    print("CHECKPOINT RANKING WITH WEIGHTED SAFETY SCORE")
    print("=" * 80)
    print(f"Training dir:           {training_dir}")
    print(f"Output CSV:             {args.output_csv}")
    print(f"\nWeighting scheme:")
    print(f"  Collision avoidance:  {args.w_collision:.0%}")
    print(f"  Mission success:      {args.w_success:.0%}")
    print(f"  Risk management:      {args.w_risk:.0%}")
    print(f"  Separation safety:    {args.w_sep:.0%}")
    print(f"  ────────────────────────")
    print(f"  Total:                {(args.w_collision + args.w_success + args.w_risk + args.w_sep):.0%}")
    print(f"\nEvaluation mode:        {'QUICK (10 seeds/case)' if args.quick_eval else 'FULL (100 seeds/case)'}")
    if args.top_k:
        print(f"Two-stage: Quick all, then full eval top {args.top_k}")
    if args.smoke_test:
        print(f"Smoke test mode:        ON (checkpoint 50k + case 1 only)")
    print("=" * 80 + "\n")
    
    # Find checkpoints
    checkpoints = find_checkpoints(training_dir)
    if not checkpoints:
        print("ERROR: No checkpoints found!")
        return
    
    print(f"Found {len(checkpoints)} checkpoints:")
    for step, path in checkpoints:
        print(f"  Step {step:7d}: {path.name}")
    print()
    
    # Smoke test: only evaluate one checkpoint and one case before full sweep
    # Use --smoke_test flag to enable
    if args.smoke_test:
        checkpoints = [(step, path) for step, path in checkpoints if step == 50000]
        cases = [1]  # Just case 1
        print(f"[WARN] SMOKE TEST MODE: Using only checkpoint 50k and case 1")
        print(f"       Once this works, rerun without --smoke_test for full sweep.\n")
    else:
        cases = list(range(1, 23))
    
    # Define cases to evaluate (Imazu 22-case validation set)
    if 'cases' not in locals():
        cases = list(range(1, 23))
    num_seeds = 10 if args.quick_eval else args.num_seeds
    
    print(f"Evaluating {len(checkpoints)} checkpoints × {len(cases)} cases × {num_seeds} seeds")
    print(f"Total evaluations: {len(checkpoints) * len(cases) * num_seeds:,}\n")
    
    # Stage 1: Quick evaluation on all checkpoints
    results = []
    
    for checkpoint_idx, (step, checkpoint_path) in enumerate(checkpoints):
        print(f"\n[{checkpoint_idx+1}/{len(checkpoints)}] Checkpoint {step:,} steps")
        
        checkpoint_metrics = defaultdict(list)
        
        for case in cases:
            # Create unique output dir for this checkpoint/case
            eval_output_dir = str(training_dir / f"eval_cp{step}_case{case}")
            Path(eval_output_dir).mkdir(parents=True, exist_ok=True)
            
            print(f"  Case {case:2d}: ", end="", flush=True)
            
            metrics = evaluate_checkpoint_on_case(
                checkpoint_path,
                case,
                num_seeds=num_seeds,
                output_dir=eval_output_dir,
                desired_cross_x_nmi=args.desired_cross_x_nmi,
                target_speed_mps=args.target_speed_mps,
                ownship_speed_mps=args.ownship_speed_mps,
                sim_time=args.sim_time,
            )
            
            if metrics:
                print(f"[OK] collision={metrics['collision_rate']:.1%} success={metrics['success_rate']:.1%}")
                for key, val in metrics.items():
                    if key != 'num_episodes':
                        checkpoint_metrics[key].append(val)
            else:
                print("✗ FAILED")
        
        # Aggregate across all cases for this checkpoint
        if checkpoint_metrics:
            agg_metrics = {
                key: float(np.mean(vals)) for key, vals in checkpoint_metrics.items()
            }
            
            score = compute_weighted_score(
                agg_metrics['collision_rate'],
                agg_metrics['success_rate'],
                agg_metrics['risk_exposure'],
                agg_metrics['min_sep'],
                w_collision=args.w_collision,
                w_success=args.w_success,
                w_risk=args.w_risk,
                w_sep=args.w_sep,
            )
            
            row = format_checkpoint_row(step, agg_metrics, score)
            results.append(row)
    
    # Create DataFrame
    if not results:
        print("\nERROR: No checkpoint evaluations succeeded.")
        print("Check the evaluator command above. Most likely eval_generalized_policy_sb3 failed before writing summary.")
        return
    
    df = pd.DataFrame(results)
    
    # -----------------------------------------------------------------
    # Constraint-based ranking:
    #   "Acceptable" = collision_rate == 0.0 AND success_rate == 1.0
    #   Among acceptable: lowest risk_exposure, then highest min_sep_m,
    #                     then lowest completion_time (tie-breaker)
    #   Unacceptable checkpoints are listed below, sorted by collision
    #   rate ascending then success rate descending.
    # -----------------------------------------------------------------
    df_acceptable = df[df['acceptable']].copy()
    df_unacceptable = df[~df['acceptable']].copy()
    
    if not df_acceptable.empty:
        df_acceptable = df_acceptable.sort_values(
            by=['risk_exposure', 'min_sep_m', 'completion_time_s'],
            ascending=[True, False, True],
        )
    
    df_unacceptable = df_unacceptable.sort_values(
        by=['collision_rate', 'success_rate'],
        ascending=[True, False],
    )
    
    df_ranked = pd.concat([df_acceptable, df_unacceptable], ignore_index=True)
    
    # Stage 2: If two-stage mode, re-evaluate top-K with more seeds
    if args.top_k and args.top_k < len(df_ranked) and not args.quick_eval:
        print("\n" + "=" * 80)
        print(f"STAGE 2: Full evaluation of top {args.top_k} candidates")
        print("=" * 80 + "\n")
        
        top_k_steps = df_ranked.head(args.top_k)['step'].tolist()
        results_full = []
        
        for step in top_k_steps:
            checkpoint_path = training_dir / "checkpoints" / f"generalized_checkpoint_{step}_steps.zip"
            
            print(f"\nFull eval: Checkpoint {step:,} steps (100 seeds/case)")
            
            checkpoint_metrics = defaultdict(list)
            
            for case in cases:
                eval_output_dir = str(training_dir / f"eval_cp{step}_case{case}_full")
                Path(eval_output_dir).mkdir(parents=True, exist_ok=True)
                
                print(f"  Case {case:2d}: ", end="", flush=True)
                
                metrics = evaluate_checkpoint_on_case(
                    checkpoint_path,
                    case,
                    num_seeds=100,
                    output_dir=eval_output_dir,
                    desired_cross_x_nmi=args.desired_cross_x_nmi,
                    target_speed_mps=args.target_speed_mps,
                    ownship_speed_mps=args.ownship_speed_mps,
                    sim_time=args.sim_time,
                )
                
                if metrics:
                    print(f"[OK] collision={metrics['collision_rate']:.1%}")
                    for key, val in metrics.items():
                        if key != 'num_episodes':
                            checkpoint_metrics[key].append(val)
                else:
                    print("✗ FAILED")
            
            if checkpoint_metrics:
                agg_metrics = {
                    key: float(np.mean(vals)) for key, vals in checkpoint_metrics.items()
                }
                
                score = compute_weighted_score(
                    agg_metrics['collision_rate'],
                    agg_metrics['success_rate'],
                    agg_metrics['risk_exposure'],
                    agg_metrics['min_sep'],
                    w_collision=args.w_collision,
                    w_success=args.w_success,
                    w_risk=args.w_risk,
                    w_sep=args.w_sep,
                )
                
                row = format_checkpoint_row(step, agg_metrics, score)
                results_full.append(row)
        
        # Replace results with full evaluation results
        df_full = pd.DataFrame(results_full)
        
        df_acc_full = df_full[df_full['acceptable']].copy()
        df_unacc_full = df_full[~df_full['acceptable']].copy()
        if not df_acc_full.empty:
            df_acc_full = df_acc_full.sort_values(
                by=['risk_exposure', 'min_sep_m', 'completion_time_s'],
                ascending=[True, False, True],
            )
        df_unacc_full = df_unacc_full.sort_values(
            by=['collision_rate', 'success_rate'], ascending=[True, False]
        )
        df_ranked = pd.concat([df_acc_full, df_unacc_full], ignore_index=True)
    
    # Save to CSV
    output_path = Path(args.output_csv)
    df_ranked.to_csv(output_path, index=False)
    print(f"\n[OK] Saved results to {output_path}\n")
    
    # Show recommendations
    n_acceptable = int(df_ranked['acceptable'].sum())
    print("=" * 80)
    print(f"CHECKPOINT RANKING  ({n_acceptable}/{len(df_ranked)} acceptable)")
    print("  Acceptable = collision_rate==0.0 AND success_rate==1.0")
    print("  Ranked by: risk_exposure asc → min_sep_m desc → completion_time asc")
    print("=" * 80)
    print()
    display_cols = ['step', 'acceptable', 'collision_rate', 'success_rate',
                    'risk_exposure', 'min_sep_m', 'completion_time_s']
    print(df_ranked[display_cols].head(10).to_string(index=False))
    print()
    
    best = df_ranked.iloc[0]
    status = "ACCEPTABLE" if best['acceptable'] else "NOT ACCEPTABLE (no perfect checkpoint found)"
    print(f"BEST CHECKPOINT [{status}]: Step {int(best['step']):,}")
    print(f"  Collision Rate:   {best['collision_rate']:.1%}")
    print(f"  Success Rate:     {best['success_rate']:.1%}")
    print(f"  Risk Exposure:    {best['risk_exposure']:.2f}")
    print(f"  Min Separation:   {best['min_sep_m']:.1f} m")
    print(f"  Completion Time:  {best['completion_time_s']:.1f} s")
    print()


if __name__ == "__main__":
    main()
