"""
Train a generalized single-agent PPO policy using procedurally generated encounters.

Usage
-----
  python -m src.train_generalized_policy_sb3 \\
      --num_steps 3000000 --sim_time 900.0

Key arguments
-------------
  --num_steps (int)             Total training timesteps (default: 1 000 000)
  --checkpoint_freq (int)       Save checkpoint every N steps (default: 50 000)
  --train_batch (int)           PPO minibatch size (default: 256)
  --rollout_frag (int)          Rollout fragment per update (default: 256)
  --lr (float)                  Learning rate (default: 1e-4)
  --sim_time (float)            Episode horizon in seconds (default: 900.0)
  --desired_cross_x_nmi (float) Encounter crossing distance in nmi (default: 1.0)
  --target_speed_mps (float)    Obstacle speed m/s (default: 10.0)
  --ownship_speed_mps (float)   Ownship speed m/s — (recommend passing 10.0 explicitly for reproducibility)
  --master_seed (int)           Seed for reproducible encounter sequence
  --resume_from (str)           Warm-start from a .zip checkpoint

Outputs
-------
  GENERALIZED_SB3_YYYYMMDD-HHMMSS/
  ├── checkpoints/generalized_checkpoint_<N>_steps.zip
  └── training_config.json
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.env_procedural_encounter_sb3 import RandomEncounterEnv


class GeneralizedTrainingMetricsCallback(BaseCallback):
    """Track training metrics using EpisodeReturnTracker wrapper.
    
    Tracks RAW returns for training, but computes NORMALIZED returns for visualization.
    This gives the policy real reward signals (collisions are bad!) while providing
    clean convergence plots that show balanced improvement across scenario types.
    """
    
    def __init__(self, episode_tracker=None, verbose=1, output_dir=None):
        super().__init__(verbose)
        self.episode_tracker = episode_tracker  # Reference to EpisodeReturnTracker wrapper
        self.verbose = verbose
        self.output_dir = output_dir
        self.timesteps = []
        self.train_returns = []  # RAW returns for logging
        self.train_returns_normalized = []  # NORMALIZED returns for plotting
        self.val_returns = []
        self.returns_2ship = []  # RAW
        self.returns_3ship = []  # RAW
        self.returns_4ship = []  # RAW
        self.returns_2ship_normalized = []  # NORMALIZED
        self.returns_3ship_normalized = []  # NORMALIZED
        self.returns_4ship_normalized = []  # NORMALIZED
        self.episode_counter = 0
        self.logger_keys_printed = False
        self.valid_key = None
        self.diagnostics_file = None
        if output_dir:
            self.diagnostics_file = output_dir / "logger_diagnostics.txt"
        
        # Cache for unwrapped base environment (found once on first access, reused thereafter)
        self._base_env_cached = None
        self._base_env_cache_attempted = False
        self._curriculum_update_count = 0  # Track how many times curriculum update succeeded
        self._curriculum_logged = False  # Track if we've logged curriculum status
        
        # Observation shape tracking for diagnostics
        self._obs_shapes_seen = {}  # {step: (shape, case, num_ships)}
        self._last_logged_step = 0
        
        # Running statistics for normalization (separate per ship count)
        self.ship_count_stats = {
            2: {"returns": [], "mean": 0.0, "std": 1.0},
            3: {"returns": [], "mean": 0.0, "std": 1.0},
            4: {"returns": [], "mean": 0.0, "std": 1.0},
        }
    
    def _normalize_value(self, value: float, ship_count: int) -> float:
        """Normalize a value using running statistics for the ship count."""
        if not np.isfinite(value):
            return np.nan
        stats = self.ship_count_stats.get(ship_count, {})
        mean = stats.get("mean", 0.0)
        std = max(stats.get("std", 1.0), 1e-8)
        normalized = (value - mean) / std
        # Clip extreme outliers to prevent visualization artifacts
        return np.clip(normalized, -10.0, 10.0)
    
    def _update_stats(self, returns_dict: dict):
        """Update running statistics based on observed returns."""
        # Returns_dict: {2: mean_return_2ship, 3: mean_return_3ship, 4: mean_return_4ship}
        for ship_count, ret in returns_dict.items():
            if np.isfinite(ret):
                self.ship_count_stats[ship_count]["returns"].append(ret)
                # Recompute mean and std
                all_ret = self.ship_count_stats[ship_count]["returns"]
                self.ship_count_stats[ship_count]["mean"] = np.mean(all_ret)
                if len(all_ret) > 1:
                    std_val = float(np.std(all_ret))
                    # Ensure std doesn't get too small (causes numerical issues)
                    self.ship_count_stats[ship_count]["std"] = max(std_val, 1e-6)
    
    def _get_base_env(self):
        """Unwrap and cache the env with update_step() (called once, result reused)."""
        if self._base_env_cache_attempted:
            return self._base_env_cached
        self._base_env_cache_attempted = True

        if self.model is None or self.model.env is None:
            return None

        env = self.model.env
        # Strategy 1: via .envs[0] (DummyVecEnv)
        try:
            envs = getattr(env, 'envs', None)
            if envs is not None and isinstance(envs, (list, tuple)) and len(envs) > 0:
                current = envs[0]
                for _ in range(10):
                    if hasattr(current, 'update_step'):
                        self._base_env_cached = current
                        return self._base_env_cached
                    next_env = getattr(current, 'env', None)
                    if next_env is not None and next_env != current:
                        current = next_env
                    else:
                        break
        except Exception:
            pass

        # Strategy 2: direct unwrap via .env
        try:
            current = env
            for _ in range(10):
                if hasattr(current, 'update_step'):
                    self._base_env_cached = current
                    return self._base_env_cached
                next_env = getattr(current, 'env', None)
                if next_env is not None and next_env != current:
                    current = next_env
                else:
                    break
        except Exception:
            pass

        return None
    
    def _on_step(self) -> bool:
        """Called after each environment step."""
        
        # Track observation shapes for diagnostics at curriculum transitions
        if self.num_timesteps % 10000 == 0:
            try:
                # Try to get observation from base env (ImazuCaseEnv has current_case info)
                base_env = self._get_base_env() if not hasattr(self, '_obs_diagnostics_done') else None
                if base_env is not None and hasattr(base_env, 'current_case'):
                    case_list = base_env.env_method("__getattr__", "current_case")
                    case = case_list[0] if case_list else None
                    # Infer ship count from case
                    ships = 2 if case in range(1, 5) else (3 if case in range(5, 12) else 4 if case in range(12, 24) else 0)
                    self._obs_shapes_seen[self.num_timesteps] = (case, ships)
            except:
                pass
        
        # Update curriculum step via cached base environment
        # Get base_env once per training session (cached after first call)
        base_env = self._get_base_env()
        
        # Call update_step if both base_env exists and has the method
        update_step_fn = getattr(base_env, 'update_step', None) 
        if base_env is not None and callable(update_step_fn):
            try:
                update_step_fn(self.num_timesteps)
                self._curriculum_update_count += 1
                
                if not self._curriculum_logged:
                    print(f"[Curriculum] ACTIVE via {type(base_env).__name__}")
                    self._curriculum_logged = True
                    
            except Exception as e:
                if self.verbose and self.num_timesteps == 1000:
                    print(f"  Warning: Curriculum update_step() failed: {e}")
        elif not self._curriculum_logged and self.num_timesteps > 5000:
            if self.verbose:
                print(f"[Curriculum] NOT ACTIVE after {self.num_timesteps} steps — training on uniform distribution")
            self._curriculum_logged = True
        
        # Print available logger keys on first substantial step (after ~1% of data collected)
        if not self.logger_keys_printed and self.num_timesteps > 500:
            self._print_available_keys()
            self.logger_keys_printed = True
        
        # Log metrics periodically
        if self.num_timesteps % 1000 == 0 and self.num_timesteps > 0:
            # Training return: windowed mean of recent episodes (RAW)
            train_return = self._get_mean_episode_return()
            
            # Per-ship-count returns (RAW)
            by_ships = {}
            if self.episode_tracker is not None:
                by_ships = self.episode_tracker.get_mean_return_by_ships()
            
            # Update statistics for normalization
            self._update_stats(by_ships)
            
            # Compute normalized versions for visualization
            train_return_norm = self._normalize_value(train_return, 2) if not np.isnan(train_return) else np.nan
            returns_by_ships_norm = {
                ships: self._normalize_value(ret, ships) 
                for ships, ret in by_ships.items() if np.isfinite(ret)
            }
            
            # Store both raw and normalized
            self.timesteps.append(self.num_timesteps)
            self.train_returns.append(train_return)
            self.train_returns_normalized.append(train_return_norm)
            self.val_returns.append(np.nan)
            
            self.returns_2ship.append(by_ships.get(2, np.nan))
            self.returns_3ship.append(by_ships.get(3, np.nan))
            self.returns_4ship.append(by_ships.get(4, np.nan))
            
            self.returns_2ship_normalized.append(returns_by_ships_norm.get(2, np.nan))
            self.returns_3ship_normalized.append(returns_by_ships_norm.get(3, np.nan))
            self.returns_4ship_normalized.append(returns_by_ships_norm.get(4, np.nan))
            
            if self.verbose:
                parts = [f"[Step {self.num_timesteps:7d}] Return: {train_return:8.1f}"]
                for ns in [2, 3, 4]:
                    v = by_ships.get(ns, np.nan)
                    if not np.isnan(v):
                        parts.append(f"{ns}ship:{v:8.1f}")
                print("  ".join(parts))
        
        return True
    
    
    def _print_available_keys(self):
        """Print and save all available logger keys for debugging."""
        header = "\n" + "=" * 80
        title = "LOGGER KEYS AVAILABLE AT FIRST CHECKPOINT"
        footer = "=" * 80 + "\n"
        
        lines = [header, title, footer]
        
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            logger_dict = self.model.logger.name_to_value
            lines.append(f"Total keys: {len(logger_dict)}")
            lines.append(f"All keys: {sorted(logger_dict.keys())}")
            lines.append("\nKey values:")
            for key in sorted(logger_dict.keys()):
                val = logger_dict[key]
                lines.append(f"  {key}: {val}")
        else:
            lines.append("ERROR: No logger found on model!")
        
        lines.append(footer)
        
        # Print to console
        output = "\n".join(lines)
        print(output)
        
        # Save to file
        if self.diagnostics_file:
            try:
                with open(self.diagnostics_file, 'w') as f:
                    f.write(output)
                if self.verbose:
                    print(f"[Info] Logger diagnostics saved to: {self.diagnostics_file}")
            except Exception as e:
                print(f"[Warning] Failed to save diagnostics to file: {e}")
    
    def _get_mean_episode_return(self) -> float:
        """Extract mean episode return from EpisodeReturnTracker."""
        try:
            if self.episode_tracker is not None:
                return self.episode_tracker.get_mean_return()
            return 0.0
        except Exception as e:
            if self.verbose:
                print(f"  [Debug] Return extraction failed: {type(e).__name__}: {e}")
            return 0.0
    
    def _on_training_end(self) -> None:
        """Called when training finishes."""
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE - CONVERGENCE METRICS")
        print("=" * 80)
        
        # Curriculum diagnostics
        print(f"\nCurriculum Learning Status:")
        if self._curriculum_update_count > 0:
            print(f"  ✓ Active: {self._curriculum_update_count:,} successful update_step() calls")
        else:
            print(f"  ✗ NOT ACTIVE: No curriculum updates (training on fixed distribution)")
        
        if not self.logger_keys_printed:
            print("\n[Info] Logger keys were never printed - first checkpoint may not have been reached")
        
        # Diagnostics for raw returns
        if self.train_returns:
            valid_raw = [r for r in self.train_returns if np.isfinite(r) and r != 0.0]
            if valid_raw:
                print(f"\nRaw Training Episode Return Statistics (from {len(valid_raw)}/{len(self.train_returns)} checkpoints):")
                print(f"  Final return:    {valid_raw[-1]:10.2f}")
                print(f"  Best return:     {np.max(valid_raw):10.2f}")
                print(f"  Mean return:     {np.mean(valid_raw):10.2f}")
                print(f"  Std return:      {np.std(valid_raw):10.2f}")
                print(f"  Range:           [{np.min(valid_raw):10.2f}, {np.max(valid_raw):10.2f}]")
            else:
                print(f"\n⚠️  All {len(self.train_returns)} raw returns are 0.0 or NaN!")
        else:
            print("\n⚠️  No raw metrics were logged during training!")
        
        # Diagnostics for normalized returns
        if self.train_returns_normalized:
            valid_norm = [r for r in self.train_returns_normalized if np.isfinite(r)]
            if valid_norm:
                print(f"\nNormalized Training Episode Return Statistics ({len(valid_norm)}/{len(self.train_returns_normalized)} valid):")
                print(f"  Final normalized: {valid_norm[-1]:10.4f}")
                print(f"  Mean normalized:  {np.mean(valid_norm):10.4f}")
                print(f"  Std normalized:   {np.std(valid_norm):10.4f}")
                print(f"  Range:            [{np.min(valid_norm):10.4f}, {np.max(valid_norm):10.4f}]")
            else:
                print(f"\n⚠️  All {len(self.train_returns_normalized)} normalized returns are NaN!")
        
        # Show normalization statistics
        print("\nNormalization Statistics (per scenario type):")
        for n_ships in [2, 3, 4]:
            stats = self.ship_count_stats.get(n_ships, {})
            n_samples = len(stats.get("returns", []))
            mean = stats.get("mean", 0.0)
            std = stats.get("std", 1.0)
            print(f"  {n_ships}-ship: {n_samples:4d} samples, mean={mean:10.2f}, std={std:10.4f}")
        
        print("=" * 80 + "\n")


def create_output_dir(timestamp: str) -> Path:
    """Create timestamped output directory."""
    output_dir = Path(f"GENERALIZED_SB3_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    return output_dir


def plot_training_convergence(timesteps: list, train_returns_normalized: list, val_returns: list, output_path: Path,
                              returns_2ship_normalized: Optional[list] = None, returns_3ship_normalized: Optional[list] = None, 
                              returns_4ship_normalized: Optional[list] = None, train_returns_raw: Optional[list] = None,
                              returns_2ship_raw: Optional[list] = None, returns_3ship_raw: Optional[list] = None,
                              returns_4ship_raw: Optional[list] = None,
                              use_raw_for_plot: bool = True, show_overall: bool = True):
    """
    Plot training convergence curve with raw returns.
    
    Shows actual episode returns to reveal the real learning signal.
    Includes per-scenario-type breakdown to show balanced learning across 2/3/4-ship.
    
    Parameters:
        timesteps: Training step counts
        train_returns_raw: Raw overall returns (used by default)
        train_returns_normalized: Normalized returns (for comparison if needed)
        returns_*ship_raw: Raw returns per scenario type
        returns_*ship_normalized: Normalized per-scenario returns
        use_raw_for_plot: If True (default), plot raw returns; if False, plot normalized
        show_overall: Include overall training return line
    """
    if not timesteps:
        print("  [Skipped convergence plot - no training data collected]")
        return
    
    # Choose data source
    if use_raw_for_plot and train_returns_raw:
        train_data = train_returns_raw
        returns_2ship_data = returns_2ship_raw if returns_2ship_raw else returns_2ship_normalized
        returns_3ship_data = returns_3ship_raw if returns_3ship_raw else returns_3ship_normalized
        returns_4ship_data = returns_4ship_raw if returns_4ship_raw else returns_4ship_normalized
        data_type = "Raw"
    else:
        train_data = train_returns_normalized
        returns_2ship_data = returns_2ship_normalized
        returns_3ship_data = returns_3ship_normalized
        returns_4ship_data = returns_4ship_normalized
        data_type = "Normalized"
    
    if not train_data:
        print("  [Skipped convergence plot - no training data collected]")
        return
    
    print(f"  Plotting {len(timesteps)} checkpoints ({data_type} returns)...")
    
    # Filter to only finite values for plotting
    timesteps_arr = np.array(timesteps, dtype=float)
    train_arr = np.array(train_data, dtype=float)
    
    # Create mask for finite values
    mask = np.isfinite(train_arr)
    if not np.any(mask):
        print(f"  [Skipped convergence plot - no valid {data_type} data (all NaN/inf)]")
        return
    
    # Filter arrays
    timesteps_valid = timesteps_arr[mask]
    train_valid = train_arr[mask]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    timesteps_m = timesteps_valid / 1e6
    
    # Plot overall training returns
    if show_overall:
        ax.plot(timesteps_m, train_valid, linewidth=2.5, marker='o', 
                markersize=5, color='steelblue', label='Overall (last 50 ep)', zorder=3)
    
    # Plot per-ship-count returns
    ship_colors = {2: '#2ca02c', 3: '#ff7f0e', 4: '#d62728'}
    ship_labels = {2: '2-ship cases', 3: '3-ship cases', 4: '4-ship cases'}
    for n_ships, data in [(2, returns_2ship_data), (3, returns_3ship_data), (4, returns_4ship_data)]:
        if data and len(data) == len(timesteps):
            arr = np.array(data, dtype=float)
            # Use same mask for consistency
            arr_valid = arr[mask]
            valid_mask = np.isfinite(arr_valid)
            if np.any(valid_mask):
                ax.plot(timesteps_m[valid_mask], arr_valid[valid_mask], linewidth=1.5, alpha=0.7,
                        color=ship_colors[n_ships], label=ship_labels[n_ships], zorder=2)
    
    # Add trend line with exponential smoothing
    if show_overall and len(train_valid) > 3:
        alpha = 0.05  # Heavy smoothing to reveal trend
        ema = np.zeros_like(train_valid)
        ema[0] = train_valid[0]
        for i in range(1, len(train_valid)):
            ema[i] = alpha * train_valid[i] + (1 - alpha) * ema[i-1]
        ax.plot(timesteps_m, ema, linewidth=3.5, linestyle='--', 
                color='navy', alpha=0.8, label='Trend (EMA)', zorder=2.5)
    
    # Reference lines
    if data_type == "Normalized":
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='Baseline (μ)')
    
    ax.set_xlabel('Training Steps (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Episode Return', fontsize=12, fontweight='bold')
    ax.set_title(f'Generalized Policy Training Return ({data_type} - No Distortion)', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=1)
    ax.legend(fontsize=11, loc='best')
    
    # Auto-scale with some padding - consider ALL plotted data, not just overall
    all_valid_values = [train_valid]  # Start with overall data
    
    # Add per-ship data to auto-scale calculation
    for n_ships, data in [(2, returns_2ship_data), (3, returns_3ship_data), (4, returns_4ship_data)]:
        if data and len(data) == len(timesteps):
            arr = np.array(data, dtype=float)
            arr_valid = arr[mask]
            valid_mask = np.isfinite(arr_valid)
            if np.any(valid_mask):
                all_valid_values.append(arr_valid[valid_mask])
    
    # Compute min/max across all plotted data
    all_data_combined = np.concatenate(all_valid_values)
    y_min, y_max = np.min(all_data_combined), np.max(all_data_combined)
    y_pad = max(0.5, (y_max - y_min) * 0.1)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"  ✓ Convergence plot saved to: {output_path.name} ({len(train_valid)} valid points, {data_type})")


def main():
    parser = argparse.ArgumentParser(
        description="Train generalized PPO policy across multiple cases and seeds"
    )
    parser.add_argument("--num_steps", type=int, default=1000000,
                        help="Total training steps (default: 1000000)")
    parser.add_argument("--checkpoint_freq", type=int, default=50000,
                        help="Save checkpoint every N steps (default: 50000)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel environments (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--train_batch", type=int, default=256,
                        help="Training batch size (default: 512)")
    parser.add_argument("--rollout_frag", type=int, default=256,
                        help="Rollout fragment length (default: 256)")
    parser.add_argument("--mlp_hiddens", type=int, nargs=2, default=[256, 256],
                        help="MLP hidden layer sizes (default: 256 256)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--master_seed", type=int, default=None,
                        help="Master seed for reproducible case/seed sequence (default: None = random)")
    parser.add_argument("--desired_cross_x_nmi", type=float, default=1.0,
                        help="Encounter cluster distance along 2 nmi route (default: 1.0, try 1.2-1.3 for more engagement)")
    parser.add_argument("--target_speed_mps", type=float, default=10.0,
                        help="Obstacle speed in m/s (default: 10.0, use 8.5-9.0 for slower obstacles)")
    parser.add_argument("--ownship_speed_mps", type=float, default=None,
                        help="Ownship cruising speed in m/s (default: None = inherit from first obstacle). Use 11.0 for slight speed advantage.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a .zip checkpoint to warm-start from (policy weights only; optimizer is reset for fine-tuning)")
    parser.add_argument("--sim_time", type=float, default=900.0,
                        help="Episode horizon in seconds. Use 900.0 for primary mission training/evaluation.")
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = create_output_dir(timestamp)
    
    print("=" * 80)
    print("TRAINING GENERALIZED POLICY ACROSS MULTIPLE CASES")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Encounter mode:            PROCEDURAL (random generation)")
    print(f"    - Stochastic mixed curriculum (active from step 0):")
    print(f"        0-250k  :  80% one target, 15% two targets,  5% three targets")
    print(f"        250k-750k: 60% one target, 30% two targets, 10% three targets")
    print(f"        750k+  :  40% one target, 40% two targets, 20% three targets")
    print(f"    - Target speed range: {6.0}-{14.0} m/s")
    print(f"    - Ownship nominal speed: {args.ownship_speed_mps if args.ownship_speed_mps else 10.0} m/s")
    print(f"    - Ownship speed action bins (m/s): [7.0, 8.25, 9.5, 10.75, 12.0]")
    print(f"  Total steps:               {args.num_steps:,}")
    print(f"  Checkpoint freq:           {args.checkpoint_freq:,}")
    print(f"  Parallel workers:          {args.num_workers}")
    print(f"  Batch size:                {args.train_batch}")
    print(f"  Rollout fragment length:   {args.rollout_frag}")
    print(f"  Learning rate:             {args.lr}")

    print(f"  Desired encounter distance (nmi): {args.desired_cross_x_nmi}")
    print(f"  Obstacle speed (m/s):      {args.target_speed_mps}")
    print(f"  Ownship speed (m/s):       {args.ownship_speed_mps if args.ownship_speed_mps else 'inherit from obstacles'}")
    print(f"  Master seed:               {args.master_seed if args.master_seed is not None else 'random'}")
    print(f"  MLP architecture:          {args.mlp_hiddens}")
    print(f"  Resume from checkpoint:    {args.resume_from if args.resume_from else 'none (train from scratch)'}")
    print(f"\nOutput directory:            {output_dir}")
    print(f"Timestamp:                   {timestamp}")
    print("=" * 80 + "\n")
    
    model = None  # Initialize to handle potential unbound errors in exception handlers
    
    try:
        # Create environment with procedurally generated random encounters
        env = RandomEncounterEnv(
            ownship_speed_mps=args.ownship_speed_mps if args.ownship_speed_mps is not None else 10.0,
            target_speed_range=(6.0, 14.0),
            desired_cross_x_nmi=args.desired_cross_x_nmi,
            dt=0.5,
            sim_time=args.sim_time,
            n_heading=7,
            n_speed=5,
            max_heading_change_deg=25.0,
            loa_m=30.0,
            route_len_nmi=2.0,
            master_seed=args.master_seed,
        )
        
        
        # Wrap with Monitor for proper episode return tracking
        env = Monitor(env)
        
        obs_space = env.observation_space
        act_space = env.action_space
        
        print(f"Observation space: {obs_space}")
        print(f"Action space:      {act_space}\n")
        
        # Create PPO policy
        from torch import nn
        policy_kwargs = {
            "net_arch": {"pi": args.mlp_hiddens, "vf": args.mlp_hiddens},
            "activation_fn": nn.Tanh,
        }
        
        if args.resume_from:
            # Fine-tuning: load policy weights from checkpoint, reset optimizer for clean LR
            print(f"[Fine-tune] Loading policy weights from: {args.resume_from}")
            pretrained = PPO.load(args.resume_from, env=env, device="auto")
            # Create fresh model with new hyperparameters (new optimizer, same architecture)
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=args.lr,
                n_steps=args.rollout_frag,
                batch_size=args.train_batch,
                n_epochs=10,
                gamma=args.gamma,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.005,
                vf_coef=0.5,
                max_grad_norm=0.5,
                use_sde=False,
                policy_kwargs=policy_kwargs,
                verbose=1,
                device="auto",
            )
            # Copy only the policy network weights (not optimizer state)
            model.policy.load_state_dict(pretrained.policy.state_dict())
            print(f"[Fine-tune] Policy weights loaded. Optimizer reset. LR = {args.lr}")
            del pretrained
        else:
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=args.lr,
                n_steps=args.rollout_frag,
                batch_size=args.train_batch,
                n_epochs=10,
                gamma=args.gamma,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.005,
                vf_coef=0.5,
                max_grad_norm=0.5,
                use_sde=False,
                policy_kwargs=policy_kwargs,
                verbose=1,
                device="auto",
            )
        
        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(output_dir / "checkpoints"),
            name_prefix="generalized_checkpoint",
            save_replay_buffer=False,
        )
        
        metrics_callback = GeneralizedTrainingMetricsCallback(
            episode_tracker=None,
            verbose=1,
            output_dir=output_dir
        )
        
        # Train
        print(f"Starting training with {args.num_steps:,} steps...\n")
        model.learn(
            total_timesteps=args.num_steps,
            callback=[checkpoint_callback, metrics_callback],
            progress_bar=True,
        )
        
        # Diagnostic: log observed cases and their ship counts during training
        if hasattr(metrics_callback, '_obs_shapes_seen') and metrics_callback._obs_shapes_seen:
            print("\n[OBSERVATION SHAPE DIAGNOSTICS]")
            print("Cases observed during training (at 10k-step checkpoints):")
            for step in sorted(metrics_callback._obs_shapes_seen.keys()):
                case, ships = metrics_callback._obs_shapes_seen[step]
                print(f"  Step {step:7d}M: Case {case:2d} ({ships}-ship)")
            sys.stdout.flush()
        
        # Save final model
        final_path = output_dir / "best_checkpoint.zip"
        model.save(str(final_path))
        print(f"\n✓ Final model saved to: {final_path}")
        
        # Plot training convergence (using normalized returns for cleaner visualization)
        print(f"\nGenerating convergence plots...")
        
        # Plot training convergence (using normalized returns for cleaner visualization)
        print(f"\nGenerating convergence plots...")
        
        # Try raw plot first (more stable than normalized early on)
        raw_valid = [x for x in metrics_callback.train_returns if np.isfinite(x)]
        norm_valid = [x for x in metrics_callback.train_returns_normalized if np.isfinite(x)]
        
        if len(raw_valid) > 0:
            # Plot raw returns (with overall line and trend) - NO NORMALIZATION
            plot_training_convergence(
                metrics_callback.timesteps, 
                metrics_callback.train_returns_normalized,  # dummy param
                metrics_callback.val_returns,
                output_dir / "training_convergence_raw.png",
                returns_2ship_normalized=metrics_callback.returns_2ship_normalized,  # dummy
                returns_3ship_normalized=metrics_callback.returns_3ship_normalized,  # dummy
                returns_4ship_normalized=metrics_callback.returns_4ship_normalized,  # dummy
                train_returns_raw=metrics_callback.train_returns,  # REAL RAW DATA
                returns_2ship_raw=metrics_callback.returns_2ship,  # REAL RAW DATA
                returns_3ship_raw=metrics_callback.returns_3ship,  # REAL RAW DATA
                returns_4ship_raw=metrics_callback.returns_4ship,  # REAL RAW DATA
                use_raw_for_plot=True,  # Use raw, not normalized
                show_overall=True
            )
        
        if len(norm_valid) > 0:
            # Plot normalized returns per scenario type for reference
            plot_training_convergence(
                metrics_callback.timesteps, 
                metrics_callback.train_returns_normalized,  # NORMALIZED DATA
                metrics_callback.val_returns,
                output_dir / "training_convergence_normalized.png",
                returns_2ship_normalized=metrics_callback.returns_2ship_normalized,  # NORMALIZED DATA
                returns_3ship_normalized=metrics_callback.returns_3ship_normalized,  # NORMALIZED DATA
                returns_4ship_normalized=metrics_callback.returns_4ship_normalized,  # NORMALIZED DATA
                train_returns_raw=metrics_callback.train_returns,  # raw for reference only
                use_raw_for_plot=False,  # Use normalized
                show_overall=False
            )
        
        if len(raw_valid) == 0 and len(norm_valid) == 0:
            print("  No valid training data to plot!")
        
        # Save training config
        config = {
            "num_steps": args.num_steps,
            "checkpoint_freq": args.checkpoint_freq,
            "num_workers": args.num_workers,
            "learning_rate": args.lr,
            "gamma": args.gamma,
            "train_batch_size": args.train_batch,
            "rollout_fragment_length": args.rollout_frag,
            "mlp_hiddens": args.mlp_hiddens,

            "seed": args.seed,
            "encounter_mode": "procedural",
            "desired_cross_x_nmi": args.desired_cross_x_nmi,
            "target_speed_mps": args.target_speed_mps,
            "ownship_speed_mps": args.ownship_speed_mps,
            "n_heading": 7,
            "n_speed": 5,
            "speed_options_mps": [7.0, 8.25, 9.5, 10.75, 12.0],
            "curriculum_type": "stochastic_mixed",
            "curriculum_phases": {
                "0":      [0.80, 0.15, 0.05],
                "250000": [0.60, 0.30, 0.10],
                "750000": [0.40, 0.40, 0.20],
            },
            "sim_time": args.sim_time,
            "dt": 0.5,
            "route_len_nmi": 2.0,
            "timestamp": timestamp,
        }
        
        config_path = output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"✓ Config saved to: {config_path}")
        
        # Summary
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n[Training interrupted by user]")
        # Save current model if possible
        if model is not None:
            try:
                interrupted_path = output_dir / "interrupted_checkpoint.zip"
                model.save(str(interrupted_path))
                print(f"Saved interrupted checkpoint to: {interrupted_path}")
            except:
                pass
        raise
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise


if __name__ == "__main__":
    main()
