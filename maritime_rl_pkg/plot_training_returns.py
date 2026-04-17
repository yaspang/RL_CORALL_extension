"""
Plot training convergence by evaluating checkpoints on a test case.

Usage:
    python -m maritime_rl_pkg.plot_training_returns \
      --training_dir "GENERALIZED_SB3_20260416-115712" \
      --case 1 --episodes 20 --seed 42
      
Output: training_returns_comparison.png showing policy improvement over time
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from .path_setup import ensure_paths
ensure_paths()

from .env_single_agent_sb3 import SingleAgentOwnshipEnv

NMI = 1852.0


def evaluate_checkpoint(
    checkpoint_path: Path,
    case: int,
    episodes: int = 10,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Evaluate a checkpoint on a case and return mean/std return.
    
    Returns: (mean_return, std_return)
    """
    try:
        model = PPO.load(str(checkpoint_path), device='cpu')
    except Exception as e:
        print(f"  ✗ Failed to load {checkpoint_path.name}: {e}")
        return np.nan, np.nan
    
    env = SingleAgentOwnshipEnv(
        case_number=case,
        dt=0.5,
        sim_time=1950.0,
        seed=seed,
    )
    
    returns = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_return = 0.0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated
        
        returns.append(ep_return)
    
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser(description="Plot training convergence")
    parser.add_argument("--training_dir", type=str, required=True,
                        help="Path to training output directory (e.g., GENERALIZED_SB3_*)")
    parser.add_argument("--case", type=int, default=1,
                        help="Case number for evaluation")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per checkpoint evaluation")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    training_dir = Path(args.training_dir)
    if not training_dir.exists():
        print(f"ERROR: Training directory not found: {training_dir}")
        return
    
    checkpoints_dir = training_dir / "checkpoints"
    if not checkpoints_dir.exists():
        print(f"ERROR: Checkpoints directory not found: {checkpoints_dir}")
        return
    
    # Find all checkpoint files sorted by step count
    checkpoint_files = sorted(checkpoints_dir.glob("checkpoint_*.zip"))
    if not checkpoint_files:
        print(f"ERROR: No checkpoint files found in {checkpoints_dir}")
        return
    
    print(f"\nFound {len(checkpoint_files)} checkpoints")
    print("Evaluating each checkpoint...")
    
    # Extract step count from filename: checkpoint_NNNNNN.zip
    steps = []
    means = []
    stds = []
    
    for cp_file in checkpoint_files:
        # Parse step number from filename
        stem = cp_file.stem  # "checkpoint_NNNNNN"
        try:
            step = int(stem.split("_")[1])
        except (ValueError, IndexError):
            print(f"  ⚠ Skipping {cp_file.name}: could not parse step count")
            continue
        
        print(f"  Evaluating {cp_file.name} ({step:,} steps)... ", end="", flush=True)
        mean_ret, std_ret = evaluate_checkpoint(
            cp_file,
            case=args.case,
            episodes=args.episodes,
            seed=args.seed,
        )
        
        if np.isnan(mean_ret):
            print("FAILED")
            continue
        
        print(f"return={mean_ret:+.2f}±{std_ret:.2f}")
        steps.append(step)
        means.append(mean_ret)
        stds.append(std_ret)
    
    if not steps:
        print("ERROR: No valid checkpoints could be evaluated")
        return
    
    # Plot convergence
    fig, ax = plt.subplots(figsize=(12, 6))
    
    steps_array = np.array(steps) / 1000  # Convert to thousands for readability
    means_array = np.array(means)
    stds_array = np.array(stds)
    
    # Plot with error bands
    ax.errorbar(steps_array, means_array, yerr=stds_array, 
                fmt='o-', linewidth=2, markersize=6, capsize=5,
                label=f"Case {args.case} (μ ± σ over {args.episodes} episodes)")
    
    ax.fill_between(steps_array, 
                    means_array - stds_array, 
                    means_array + stds_array,
                    alpha=0.2)
    
    ax.set_xlabel("Training Steps (×1000)", fontsize=12)
    ax.set_ylabel("Episode Return", fontsize=12)
    ax.set_title(f"Training Convergence: Generalized SB3 Policy\n"
                 f"Case {args.case}, {args.episodes} episodes per checkpoint",
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Save figure
    output_path = Path("training_returns_convergence.png")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    print(f"\n✓ Training convergence plot saved to: {output_path}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  First checkpoint: {means[0]:+.2f} ± {stds[0]:.2f}")
    print(f"  Last checkpoint:  {means[-1]:+.2f} ± {stds[-1]:.2f}")
    print(f"  Improvement:      {means[-1] - means[0]:+.2f}")


if __name__ == "__main__":
    main()
