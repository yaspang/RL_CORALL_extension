"""
Per-case failure breakdown for a specific checkpoint step.

Reads policy_eval_summary.json files already produced by rank_checkpoints_weighted
(stored under {training_dir}/eval_cp{step}_case{N}/seed_0/) and prints a sorted
table showing which Imazu cases are driving collisions.

Usage:
    python -m maritime_rl_pkg.case_failure_breakdown \\
        --training_dir "GENERALIZED_SB3_20260713-142238" \\
        --step 350000

    # Omit --step to auto-select the best step from checkpoint_scores*.csv:
    python -m maritime_rl_pkg.case_failure_breakdown \\
        --training_dir "GENERALIZED_SB3_20260713-142238" \\
        --scores_csv "checkpoint_scores_procedural.csv"
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_case_summaries(training_dir: Path, step: int) -> pd.DataFrame:
    """
    Find all eval_cp{step}_case* subdirs and load their policy_eval_summary.json.
    Returns a DataFrame sorted by collision_rate descending.
    """
    pattern = f"eval_cp{step}_case*/seed_0/policy_eval_summary.json"
    paths = sorted(training_dir.glob(pattern))

    if not paths:
        print(f"No eval results found for step {step:,} in {training_dir}")
        print(f"  (looked for: {training_dir / pattern})")
        return pd.DataFrame()

    rows = []
    for p in paths:
        m = re.search(r"case(\d+)", str(p))
        if not m:
            continue
        case = int(m.group(1))
        with open(p) as f:
            s = json.load(f)

        rows.append({
            "case":             case,
            "collision_rate":   s.get("collision_any_mean",              np.nan),
            "success_rate":     s.get("success_ownship_mean",            np.nan),
            "risk_exposure":    s.get("risk_exposure_ownship_mean",      np.nan),
            "min_sep_m":        s.get("min_actual_sep_m_ownship_mean",   np.nan),
            "completion_time_s": s.get("completion_time_s_ownship_mean", np.nan),
        })

    df = pd.DataFrame(rows).sort_values(
        ["collision_rate", "success_rate"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return df


def pick_best_step(scores_csv: Path) -> int:
    """Return the step with the lowest weighted_score from a checkpoint_scores CSV."""
    df = pd.read_csv(scores_csv)
    if "weighted_score" not in df.columns or "step" not in df.columns:
        raise ValueError(f"Expected 'step' and 'weighted_score' columns in {scores_csv}")
    best_row = df.iloc[0]
    return int(best_row["step"])


def print_breakdown(df: pd.DataFrame, step: int) -> None:
    print()
    print("=" * 70)
    print(f"  PER-CASE FAILURE BREAKDOWN  —  checkpoint step {step:,}")
    print("=" * 70)
    print(
        f"  {'Case':>4}  {'Collision':>10}  {'Success':>9}  "
        f"{'Risk':>8}  {'MinSep(m)':>10}  {'Time(s)':>8}"
    )
    print("  " + "-" * 66)

    collision_cases = []
    for _, row in df.iterrows():
        case = int(row["case"])
        col  = row["collision_rate"]
        suc  = row["success_rate"]
        risk = row["risk_exposure"]
        sep  = row["min_sep_m"]
        t    = row["completion_time_s"]

        flag = " ◄ HIGH" if col >= 0.20 else ("  ↑" if col >= 0.10 else "")
        if col > 0:
            collision_cases.append(case)

        print(
            f"  {case:>4}  {col:>9.1%}  {suc:>9.1%}  "
            f"{risk:>8.1f}  {sep:>10.1f}  {t:>8.1f}{flag}"
        )

    print("  " + "-" * 66)
    overall_col = df["collision_rate"].mean()
    overall_suc = df["success_rate"].mean()
    print(
        f"  {'AVG':>4}  {overall_col:>9.1%}  {overall_suc:>9.1%}  "
        f"{df['risk_exposure'].mean():>8.1f}  {df['min_sep_m'].mean():>10.1f}  "
        f"{df['completion_time_s'].mean():>8.1f}"
    )
    print("=" * 70)

    if collision_cases:
        print(f"\n  Cases with any collisions ({len(collision_cases)}): {collision_cases}")
        high = [int(r['case']) for _, r in df.iterrows() if r['collision_rate'] >= 0.20]
        if high:
            print(f"  High-collision cases (≥20%):  {high}")
    else:
        print("\n  No collisions across all cases — looks clean!")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Per-case collision breakdown for a checkpoint step"
    )
    parser.add_argument(
        "--training_dir",
        type=str,
        required=True,
        help="Training output directory (e.g., GENERALIZED_SB3_20260713-142238/)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step to inspect (e.g., 350000). "
             "Omit to auto-pick lowest-score step from --scores_csv.",
    )
    parser.add_argument(
        "--scores_csv",
        type=str,
        default=None,
        help="checkpoint_scores*.csv to auto-select best step when --step is omitted.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to save the per-case table as CSV.",
    )

    args = parser.parse_args()
    training_dir = Path(args.training_dir)

    if not training_dir.exists():
        print(f"ERROR: training_dir not found: {training_dir}")
        sys.exit(1)

    # Resolve step
    step = args.step
    if step is None:
        csv_path = Path(args.scores_csv) if args.scores_csv else None
        if csv_path is None:
            # Try to find a scores CSV automatically
            candidates = list(Path(".").glob("checkpoint_scores*.csv"))
            if not candidates:
                print("ERROR: Provide --step or --scores_csv to select a checkpoint.")
                sys.exit(1)
            csv_path = sorted(candidates)[-1]
            print(f"[INFO] Auto-detected scores CSV: {csv_path}")
        step = pick_best_step(csv_path)
        print(f"[INFO] Best step from {csv_path.name}: {step:,}")

    df = load_case_summaries(training_dir, step)
    if df.empty:
        sys.exit(1)

    print_breakdown(df, step)

    out_csv = args.output_csv or str(training_dir / f"case_breakdown_step{step}.csv")
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}\n")


if __name__ == "__main__":
    main()
