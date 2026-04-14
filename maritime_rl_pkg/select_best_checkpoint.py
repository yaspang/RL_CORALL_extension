"""
Utility to select the best checkpoint from a training run directory.

Usage:
    python select_best_checkpoint.py --run_dir MARL_ppo_case2_20260412-091424/seed_0
    
Output:
    - Prints path to best checkpoint directory
    - Prints training metrics for the best checkpoint
    - Can be piped to eval_trained_policy.py for automatic evaluation
"""

import argparse
import json
from pathlib import Path
import sys


def find_best_checkpoint(run_dir):
    """
    Find best checkpoint in a training run directory.
    
    Args:
        run_dir: Path to run directory (e.g., MARL_ppo_case2_XXXXXX/seed_0)
    
    Returns:
        dict with keys: 'checkpoint_dir', 'iteration', 'eval_return_mean', 'json_path'
    """
    run_path = Path(run_dir).resolve()
    
    if not run_path.exists():
        print(f"ERROR: Run directory not found: {run_path}")
        sys.exit(1)
    
    # Check for best checkpoint info saved by training script
    best_info_path = run_path / "best_checkpoint_info.json"
    if best_info_path.exists():
        with open(best_info_path, "r") as f:
            info = json.load(f)
        
        # Find corresponding checkpoint directory
        checkpoint_dir = run_path / "checkpoints"
        iteration = info["iteration"]
        
        print(f"✓ Best checkpoint found (from training metadata)")
        print(f"  Iteration: {iteration}")
        print(f"  Eval Return: {info['eval_return_mean']:.2f}")
        print(f"  Eval Ep Length: {info['eval_ep_length_mean']:.2f}")
        
        return {
            "checkpoint_dir": checkpoint_dir,
            "iteration": iteration,
            "eval_return_mean": info["eval_return_mean"],
            "json_path": best_info_path,
        }
    
    # Fall back: Look at training_metrics.csv to find best eval return
    csv_path = run_path / "training_metrics.csv"
    if csv_path.exists():
        best_eval_return = float("-inf")
        best_iteration = None
        
        with open(csv_path, "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                # Skip header, find row with max eval_return_mean (column 3)
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 4:
                        try:
                            iteration = int(parts[0])
                            eval_return = float(parts[3])
                            if eval_return > best_eval_return and not math.isnan(eval_return):
                                best_eval_return = eval_return
                                best_iteration = iteration
                        except ValueError:
                            pass
        
        if best_iteration is not None:
            print(f"✓ Best checkpoint inferred from training_metrics.csv")
            print(f"  Iteration: {best_iteration}")
            print(f"  Eval Return: {best_eval_return:.2f}")
            
            checkpoint_dir = run_path / "checkpoints"
            return {
                "checkpoint_dir": checkpoint_dir,
                "iteration": best_iteration,
                "eval_return_mean": best_eval_return,
                "json_path": None,
            }
    
    # Last resort: Use latest checkpoint
    checkpoint_dir = run_path / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = sorted([d for d in checkpoint_dir.iterdir() if d.is_dir()])
        if checkpoints:
            latest = checkpoints[-1]
            print(f"⚠ Using latest checkpoint (best info not found)")
            print(f"  Checkpoint: {latest.name}")
            return {
                "checkpoint_dir": checkpoint_dir,
                "iteration": None,
                "eval_return_mean": None,
                "json_path": None,
            }
    
    print(f"ERROR: No checkpoints found in {checkpoint_dir}")
    sys.exit(1)


def list_all_checkpoints(run_dir):
    """List all available checkpoints in a run directory."""
    run_path = Path(run_dir).resolve()
    checkpoint_dir = run_path / "checkpoints"
    
    if not checkpoint_dir.exists():
        print(f"No checkpoints directory found at {checkpoint_dir}")
        return
    
    checkpoints = sorted([d for d in checkpoint_dir.iterdir() if d.is_dir()])
    
    print(f"\nAll checkpoints in {run_path.name}:")
    for i, ckpt in enumerate(checkpoints, 1):
        print(f"  {i}. {ckpt.name}")
    
    if not checkpoints:
        print("  (none found)")


def main():
    parser = argparse.ArgumentParser(
        description="Find and describe the best checkpoint from a training run"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to training run directory (e.g., MARL_ppo_case2_XXXXXX/seed_0)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available checkpoints"
    )
    
    args = parser.parse_args()
    
    result = find_best_checkpoint(args.run_dir)
    
    if args.list:
        list_all_checkpoints(args.run_dir)
    
    # Print full checkpoint path for piping to other scripts
    if result["checkpoint_dir"]:
        print(f"\nCheckpoint directory: {result['checkpoint_dir']}")


if __name__ == "__main__":
    main()
