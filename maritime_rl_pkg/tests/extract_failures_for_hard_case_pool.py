"""
Extract failed cases/seeds from evaluation results and update hard-case pool.

This utility processes evaluation summaries to identify (case, seed) pairs that failed
(collision or high-risk episodes) and adds them to the hard-case pool for training.

Usage:
    python -m maritime_rl_pkg.extract_failures_for_hard_case_pool \\
        --eval_dir "policy_eval_generalized_sb3_case22_TIMESTAMP/seed_0/" \\
        --output_pool "hard_case_pool.json" \\
        --collision_threshold 0.0 \\
        --dcpa_threshold 100.0

This will:
    1. Read policy_eval_summary.json from eval_dir
    2. Extract episodes with collision_flag or DCPA < threshold
    3. Save (case, seed) pairs to output_pool JSON file
    4. Display statistics

Then use in training:
    python -m maritime_rl_pkg.train_generalized_policy_sb3 \\
        --hard_case_oversampling \\
        --hard_case_pool_file "hard_case_pool.json" \\
        ...other args...
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np


def extract_failed_episodes(
    summary_file: str,
    collision_threshold: float = 0.0,
    dcpa_threshold: float = 100.0,
) -> Tuple[Dict[int, Set[int]], List[str]]:
    """
    Extract failed (case, seed) pairs from evaluation summary.
    
    Args:
        summary_file: Path to policy_eval_summary.json
        collision_threshold: Count episodes with collision_flag >= this as failures
        dcpa_threshold: Count episodes with DCPA < this as failures (meters)
    
    Returns:
        (failures_dict, diagnostics)
        failures_dict: {case: {seed1, seed2, ...}}
        diagnostics: list of log messages
    """
    diagnostics = []
    failures_dict: Dict[int, Set[int]] = {}
    
    summary_path = Path(summary_file)
    if not summary_path.exists():
        diagnostics.append(f"ERROR: File not found: {summary_file}")
        return {}, diagnostics
    
    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    except Exception as e:
        diagnostics.append(f"ERROR: Failed to read {summary_file}: {e}")
        return {}, diagnostics
    
    # Extract per-episode results
    per_episode = summary.get("per_episode_results", [])
    if not per_episode:
        diagnostics.append("WARNING: No per_episode_results in summary")
        return {}, diagnostics
    
    diagnostics.append(f"Read {len(per_episode)} episodes from summary")
    
    # Process each episode
    total_failed = 0
    for ep_data in per_episode:
        case = ep_data.get("case")
        seed = ep_data.get("seed")
        collision_flag = ep_data.get("collision_flag", 0)
        dcpa = ep_data.get("dcpa", float('inf'))
        
        if case is None or seed is None:
            continue
        
        case = int(case)
        seed = int(seed)
        
        # Check if episode failed
        is_failure = False
        failure_reason = ""
        
        if collision_flag > collision_threshold:
            is_failure = True
            failure_reason = f"collision_flag={collision_flag}"
        
        if dcpa < dcpa_threshold:
            is_failure = True
            if failure_reason:
                failure_reason += f", "
            failure_reason += f"dcpa={dcpa:.1f}m"
        
        if is_failure:
            if case not in failures_dict:
                failures_dict[case] = set()
            failures_dict[case].add(seed)
            total_failed += 1
    
    diagnostics.append(f"Identified {total_failed} failed episodes")
    diagnostics.append(f"Failed cases: {sorted(failures_dict.keys())}")
    for case in sorted(failures_dict.keys()):
        num_seeds = len(failures_dict[case])
        diagnostics.append(f"  Case {case:2d}: {num_seeds:3d} failed seeds")
    
    return failures_dict, diagnostics


def update_hard_case_pool(
    pool_file: str,
    new_failures: Dict[int, Set[int]],
) -> List[str]:
    """
    Update hard-case pool by merging new failures.
    
    Args:
        pool_file: Path to hard_case_pool.json (will be created if doesn't exist)
        new_failures: {case: {seed1, seed2, ...}} to add
    
    Returns:
        diagnostics: list of log messages
    """
    diagnostics = []
    pool_path = Path(pool_file)
    
    # Load existing pool
    existing_pool: Dict[int, Set[int]] = {}
    if pool_path.exists():
        try:
            with open(pool_path, 'r') as f:
                pool_data = json.load(f)
            # Convert lists back to sets
            existing_pool = {
                int(case): set(seeds)
                for case, seeds in pool_data.items()
            }
            diagnostics.append(f"Loaded existing pool from {pool_file}")
        except Exception as e:
            diagnostics.append(f"WARNING: Failed to load existing pool: {e}")
    
    # Merge new failures
    merged_count = 0
    for case, seeds in new_failures.items():
        if case not in existing_pool:
            existing_pool[case] = set()
        
        new_seeds = seeds - existing_pool[case]
        merged_count += len(new_seeds)
        existing_pool[case].update(new_seeds)
    
    # Save updated pool
    try:
        pool_data = {
            str(case): sorted(list(seeds))
            for case, seeds in existing_pool.items()
        }
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pool_path, 'w') as f:
            json.dump(pool_data, f, indent=2)
        diagnostics.append(f"Saved updated pool to {pool_file} (added {merged_count} new failures)")
    except Exception as e:
        diagnostics.append(f"ERROR: Failed to save pool: {e}")
    
    return diagnostics


def main():
    parser = argparse.ArgumentParser(
        description="Extract failed cases/seeds from evaluation results"
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Directory containing policy_eval_summary.json (e.g., policy_eval_generalized_sb3_case22_TIMESTAMP/seed_0/)"
    )
    parser.add_argument(
        "--output_pool",
        type=str,
        required=True,
        help="Output path for hard_case_pool.json"
    )
    parser.add_argument(
        "--collision_threshold",
        type=float,
        default=0.0,
        help="Episodes with collision_flag >= this are counted as failures (default: 0.0)"
    )
    parser.add_argument(
        "--dcpa_threshold",
        type=float,
        default=100.0,
        help="Episodes with DCPA < this (meters) are counted as failures (default: 100.0)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EXTRACT FAILURES FOR HARD-CASE POOL")
    print("=" * 80)
    print(f"Evaluation dir:         {args.eval_dir}")
    print(f"Output pool file:       {args.output_pool}")
    print(f"Collision threshold:    {args.collision_threshold}")
    print(f"DCPA threshold:         {args.dcpa_threshold} m")
    print("=" * 80 + "\n")
    
    # Extract failures
    summary_file = Path(args.eval_dir) / "policy_eval_summary.json"
    failures, extract_diags = extract_failed_episodes(
        str(summary_file),
        collision_threshold=args.collision_threshold,
        dcpa_threshold=args.dcpa_threshold,
    )
    
    for line in extract_diags:
        print(line)
    
    if not failures:
        print("\nNo failures found. Exiting.")
        return
    
    print()
    
    # Update pool
    update_diags = update_hard_case_pool(args.output_pool, failures)
    for line in update_diags:
        print(line)
    
    print("\n" + "=" * 80)
    print("Done! Use this in training:")
    print(f"  python -m maritime_rl_pkg.train_generalized_policy_sb3 \\")
    print(f"    --hard_case_oversampling \\")
    print(f"    --hard_case_pool_file \"{args.output_pool}\" \\")
    print(f"    ...other args...")
    print("=" * 80)


if __name__ == "__main__":
    main()
