"""
Train a generalized policy across multiple cases and random seeds.

This script trains a single PPO policy on randomized cases (1, 6, 21) and seeds,
producing a generalized collision avoidance policy that works across difficulty levels.

TRAINING STRATEGY:
==================
- Total steps: 1,000,000 (configurable)
- Each reset: random case (1, 6, 21) and random seed [0, 99]
- Policy learns to generalize across all encounter geometries and difficulties
- Single checkpoint output (vs. three case-specific checkpoints)

EXPECTED RESULTS:
=================
- Policy generalizes to loose (Case 1), medium (Case 6), tight (Case 21) encounters
- Slightly lower performance on each individual case vs case-specific policies
- Much better performance on unseen scenarios and real-world deployment
- Training time: ~6-8 hours on GPU, ~12-16 hours on CPU

USAGE:
======
python -m maritime_rl_pkg.train_generalized_policy_sb3 \\
    --num_steps 1000000 \\
    --checkpoint_freq 50000 \\
    --num_workers 4

EVALUATION:
===========
After training, evaluate on each case with:
    python -m maritime_rl_pkg.eval_single_agent_sb3 \\
        --checkpoint "/path/to/generalized/best_checkpoint.zip" \\
        --case 1 --episodes 100 --seed 0 --save_histories

Then compare performance across all three cases.
"""

import argparse
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from maritime_rl_pkg.env_random_case_sb3 import RandomCaseEnv
from maritime_rl_pkg.env_reward_normalizer import RewardNormalizerByShipCount
from maritime_rl_pkg.episode_tracker import EpisodeReturnTracker


class GeneralizedTrainingMetricsCallback(BaseCallback):
    """Track training metrics using EpisodeReturnTracker wrapper."""
    
    def __init__(self, episode_tracker=None, verbose=1, output_dir=None):
        super().__init__(verbose)
        self.episode_tracker = episode_tracker  # Reference to EpisodeReturnTracker wrapper
        self.verbose = verbose
        self.output_dir = output_dir
        self.timesteps = []
        self.train_returns = []
        self.val_returns = []
        self.returns_2ship = []
        self.returns_3ship = []
        self.returns_4ship = []
        self.episode_counter = 0
        self.logger_keys_printed = False
        self.valid_key = None
        self.diagnostics_file = None
        if output_dir:
            self.diagnostics_file = output_dir / "logger_diagnostics.txt"
    
    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Print available logger keys on first substantial step (after ~1% of data collected)
        if not self.logger_keys_printed and self.num_timesteps > 500:
            self._print_available_keys()
            self.logger_keys_printed = True
        
        # Log metrics periodically
        if self.num_timesteps % 1000 == 0 and self.num_timesteps > 0:
            # Training return: windowed mean of recent episodes
            train_return = self._get_mean_episode_return()
            
            # Validation return: mean of only recent episodes (last ~10% of what we've seen)
            val_return = np.nan
            
            self.timesteps.append(self.num_timesteps)
            self.train_returns.append(train_return)
            self.val_returns.append(val_return)
            
            # Per-ship-count returns
            by_ships = {}
            if self.episode_tracker is not None:
                by_ships = self.episode_tracker.get_mean_return_by_ships()
            self.returns_2ship.append(by_ships.get(2, np.nan))
            self.returns_3ship.append(by_ships.get(3, np.nan))
            self.returns_4ship.append(by_ships.get(4, np.nan))
            
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
        
        if not self.logger_keys_printed:
            print("\n[Info] Logger keys were never printed - first checkpoint may not have been reached")
        
        if self.train_returns:
            valid_returns = [r for r in self.train_returns if r != 0.0]
            if valid_returns:
                print(f"\nTraining Episode Return Statistics (from {len(valid_returns)}/{len(self.train_returns)} checkpoints):")
                print(f"  Final return:    {valid_returns[-1]:10.2f}")
                print(f"  Best return:     {np.max(valid_returns):10.2f}")
                print(f"  Mean return:     {np.mean(valid_returns):10.2f}")
                print(f"  Std return:      {np.std(valid_returns):10.2f}")
                
                # Check for convergence/overfitting
                if len(valid_returns) >= 3:
                    recent_mean = np.mean(valid_returns[-3:])
                    early_mean = np.mean(valid_returns[:3])
                    improvement = recent_mean - early_mean
                    print(f"\nConvergence Signal:")
                    print(f"  Early mean (first 3):   {early_mean:10.2f}")
                    print(f"  Recent mean (last 3):   {recent_mean:10.2f}")
                    print(f"  Improvement:            {improvement:10.2f}")
            else:
                print(f"\n⚠️  All {len(self.train_returns)} checkpoint returns are 0.0!")
        else:
            print("\n⚠️  No metrics were logged during training!")
        
        print("=" * 80 + "\n")


def create_output_dir(timestamp: str) -> Path:
    """Create timestamped output directory."""
    output_dir = Path(f"GENERALIZED_SB3_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    return output_dir


def plot_training_convergence(timesteps: list, train_returns: list, val_returns: list, output_path: Path,
                              returns_2ship: list = None, returns_3ship: list = None, returns_4ship: list = None):
    """Plot training convergence curve with train and per-ship-count breakdowns."""
    if not timesteps or not train_returns:
        print("  [Skipped convergence plot - no training data collected]")
        return
    
    print(f"  Plotting {len(timesteps)} checkpoints...")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    timesteps_m = np.array(timesteps) / 1e6
    
    # Plot overall training returns
    ax.plot(timesteps_m, train_returns, linewidth=2.5, marker='o', 
            markersize=5, color='steelblue', label='Overall (last 50 ep)', zorder=3)
    
    # Plot per-ship-count returns
    ship_colors = {2: '#2ca02c', 3: '#ff7f0e', 4: '#d62728'}
    ship_labels = {2: '2-ship cases', 3: '3-ship cases', 4: '4-ship cases'}
    for n_ships, data in [(2, returns_2ship), (3, returns_3ship), (4, returns_4ship)]:
        if data and len(data) == len(timesteps):
            arr = np.array(data, dtype=float)
            mask = np.isfinite(arr)
            if np.any(mask):
                ax.plot(timesteps_m[mask], arr[mask], linewidth=1.5, alpha=0.7,
                        color=ship_colors[n_ships], label=ship_labels[n_ships], zorder=2)
    
    # Add trend line
    if len(train_returns) > 3:
        window = max(1, len(train_returns) // 5)
        train_trend = np.convolve(train_returns, np.ones(window)/window, mode='valid')
        trend_timesteps = timesteps_m[window-1:]
        ax.plot(trend_timesteps, train_trend, linewidth=2.5, linestyle='--', 
                color='navy', alpha=0.6, label='Overall Trend', zorder=2)
    
    ax.set_xlabel('Training Steps (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Episode Return', fontsize=12, fontweight='bold')
    ax.set_title('Generalized Policy Training Return', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=1)
    ax.legend(fontsize=11, loc='best')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"  ✓ Convergence plot saved to: {output_path.name}")


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
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--train_batch", type=int, default=512,
                        help="Training batch size (default: 512)")
    parser.add_argument("--rollout_frag", type=int, default=256,
                        help="Rollout fragment length (default: 256)")
    parser.add_argument("--normalize_rewards", action="store_true", default=True,
                        help="Normalize rewards by scenario type for smoother training (default: True)")
    parser.add_argument("--no_normalize_rewards", dest="normalize_rewards", action="store_false",
                        help="Disable reward normalization (for ablation studies)")
    parser.add_argument("--mlp_hiddens", type=int, nargs=2, default=[128, 128],
                        help="MLP hidden layer sizes (default: 128 128)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--master_seed", type=int, default=None,
                        help="Master seed for reproducible case/seed sequence (default: None = random)")
    parser.add_argument("--cases", type=int, nargs="+", default=[1, 6, 21],
                        help="Cases to train on (default: 1 6 21, use --cases 1 2 3 ... 23 for all)")
    parser.add_argument("--desired_cross_x_nmi", type=float, default=1.0,
                        help="Encounter cluster distance along 2 nmi route (default: 1.0, try 1.2-1.3 for more engagement)")
    parser.add_argument("--target_speed_mps", type=float, default=10.0,
                        help="Obstacle speed in m/s (default: 10.0, use 8.5-9.0 for slower obstacles)")
    parser.add_argument("--ownship_speed_mps", type=float, default=None,
                        help="Ownship cruising speed in m/s (default: None = inherit from first obstacle). Use 11.0 for slight speed advantage.")
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = create_output_dir(timestamp)
    
    print("=" * 80)
    print("TRAINING GENERALIZED POLICY ACROSS MULTIPLE CASES")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Cases to train on:         {args.cases}")
    print(f"  Total steps:               {args.num_steps:,}")
    print(f"  Checkpoint freq:           {args.checkpoint_freq:,}")
    print(f"  Parallel workers:          {args.num_workers}")
    print(f"  Batch size:                {args.train_batch}")
    print(f"  Rollout fragment length:   {args.rollout_frag}")
    print(f"  Learning rate:             {args.lr}")
    print(f"  Reward normalization:      {'ON (by scenario type)' if args.normalize_rewards else 'OFF'}")
    print(f"  Desired encounter distance (nmi): {args.desired_cross_x_nmi}")
    print(f"  Obstacle speed (m/s):      {args.target_speed_mps}")
    print(f"  Ownship speed (m/s):       {args.ownship_speed_mps if args.ownship_speed_mps else 'inherit from obstacles'}")
    print(f"  Master seed:               {args.master_seed if args.master_seed is not None else 'random'}")
    print(f"  MLP architecture:          {args.mlp_hiddens}")
    print(f"\nOutput directory:            {output_dir}")
    print(f"Timestamp:                   {timestamp}")
    print("=" * 80 + "\n")
    
    model = None  # Initialize to handle potential unbound errors in exception handlers
    
    try:
        # Create environment (will randomize case/seed at each reset with reproducible sequence)
        env = RandomCaseEnv(
            cases_to_train=args.cases,
            num_seeds=100,
            dt=0.5,
            sim_time=490.0,
            n_heading=7,
            max_heading_change_deg=25.0,
            loa_m=30.0,
            route_len_nmi=2.0,
            master_seed=args.master_seed,
            desired_cross_x_nmi=args.desired_cross_x_nmi,
            target_speed_mps=args.target_speed_mps,
            ownship_speed_mps=args.ownship_speed_mps,
        )
        
        # Wrap with reward normalizer to stabilize training across scenario types
        env = RewardNormalizerByShipCount(env, normalize_rewards=args.normalize_rewards, verbose=False)
        
        # Wrap with Monitor for proper episode return tracking
        env = Monitor(env)
        
        # Wrap with EpisodeReturnTracker for direct episode return access in callback
        episode_tracker = EpisodeReturnTracker(env)
        env = episode_tracker
        
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
            episode_tracker=episode_tracker,
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
        
        # Save final model
        final_path = output_dir / "best_checkpoint.zip"
        model.save(str(final_path))
        print(f"\n✓ Final model saved to: {final_path}")
        
        # Plot training convergence
        print(f"\nGenerating convergence plots...")
        plot_training_convergence(metrics_callback.timesteps, metrics_callback.train_returns, 
                                 metrics_callback.val_returns,
                                 output_dir / "training_convergence.png",
                                 returns_2ship=metrics_callback.returns_2ship,
                                 returns_3ship=metrics_callback.returns_3ship,
                                 returns_4ship=metrics_callback.returns_4ship)
        
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
            "normalize_rewards": args.normalize_rewards,
            "seed": args.seed,
            "cases_trained": args.cases,
            "num_seeds": 100,
            "desired_cross_x_nmi": args.desired_cross_x_nmi,
            "target_speed_mps": args.target_speed_mps,
            "ownship_speed_mps": args.ownship_speed_mps,
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
