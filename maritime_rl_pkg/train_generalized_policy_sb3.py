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


class GeneralizedTrainingMetricsCallback(BaseCallback):
    """Track training metrics using SB3's Monitor wrapper and logger."""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.verbose = verbose
        self.timesteps = []
        self.mean_returns = []
        self.episode_counter = 0
    
    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Log every 5000 steps by querying the model's logger
        if self.num_timesteps % 5000 == 0 and self.num_timesteps > 0:
            mean_reward = self._get_mean_episode_return()
            
            self.timesteps.append(self.num_timesteps)
            self.mean_returns.append(mean_reward)
            
            if self.verbose:
                print(f"[Step {self.num_timesteps:7d}] Mean Episode Return: {mean_reward:10.2f}")
        
        return True
    
    def _get_mean_episode_return(self) -> float:
        """Extract mean episode return from SB3's logger."""
        try:
            # For PPO, the key is 'rollout/ep_rew_mean' or sometimes 'rollout/ep_mean_reward'
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                logger_dict = self.model.logger.name_to_value
                
                # Try multiple possible keys
                for key in ['rollout/ep_rew_mean', 'rollout/ep_mean_reward', 'ep_rew_mean']:
                    if key in logger_dict:
                        val = float(logger_dict.get(key, 0.0))
                        if val != 0.0:  # Only return if non-zero
                            return val
            
            # Fallback: return 0
            return 0.0
        except Exception as e:
            if self.verbose:
                print(f"  [Warning] Could not retrieve mean reward from logger: {e}")
            return 0.0
    
    def _on_training_end(self) -> None:
        """Called when training finishes."""
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE - CONVERGENCE METRICS")
        print("=" * 80)
        
        if self.mean_returns:
            valid_returns = [r for r in self.mean_returns if r != 0.0]
            if valid_returns:
                print(f"\nEpisode Return Statistics (from {len(valid_returns)} non-zero checkpoints):")
                print(f"  Final return:    {valid_returns[-1]:10.2f}")
                print(f"  Best return:     {np.max(valid_returns):10.2f}")
                print(f"  Mean return:     {np.mean(valid_returns):10.2f}")
                print(f"  Std return:      {np.std(valid_returns):10.2f}")
            else:
                print("\n⚠️  WARNING: No non-zero episode returns logged during training!")
                print("    This may indicate a logging issue. Check:")
                print("    1. Is the environment properly wrapped with Monitor()?")
                print("    2. Are episodes actually completing during training?")
                print("    3. Check SB3 logger keys with: model.logger.name_to_value.keys()")
        else:
            print("\n⚠️  WARNING: No metrics were logged during training!")
        
        print("=" * 80 + "\n")


def create_output_dir(timestamp: str) -> Path:
    """Create timestamped output directory."""
    output_dir = Path(f"GENERALIZED_SB3_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    return output_dir


def plot_training_convergence(timesteps: list, mean_returns: list, output_path: Path):
    """Plot training convergence curve."""
    if not timesteps or not mean_returns:
        print("  [Skipped convergence plot - no training data collected]")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert to millions of steps for readability
    timesteps_m = np.array(timesteps) / 1e6
    
    ax.plot(timesteps_m, mean_returns, linewidth=2.5, marker='o', 
            markersize=6, color='steelblue', label='Mean Episode Return')
    
    # Add a trend line (simple moving average)
    if len(mean_returns) > 3:
        window = max(1, len(mean_returns) // 5)
        trend = np.convolve(mean_returns, np.ones(window)/window, mode='valid')
        trend_timesteps = timesteps_m[window-1:]
        ax.plot(trend_timesteps, trend, linewidth=2.5, linestyle='--', 
                color='darkorange', alpha=0.7, label='Trend (Moving Avg)')
    
    ax.set_xlabel('Training Steps (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Episode Return', fontsize=12, fontweight='bold')
    ax.set_title('Generalized Policy Training Convergence', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
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
    parser.add_argument("--train_batch", type=int, default=256,
                        help="Training batch size (default: 256)")
    parser.add_argument("--rollout_frag", type=int, default=128,
                        help="Rollout fragment length (default: 128)")
    parser.add_argument("--mlp_hiddens", type=int, nargs=2, default=[128, 128],
                        help="MLP hidden layer sizes (default: 128 128)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--master_seed", type=int, default=None,
                        help="Master seed for reproducible case/seed sequence (default: None = random)")
    parser.add_argument("--cases", type=int, nargs="+", default=[1, 6, 21],
                        help="Cases to train on (default: 1 6 21, use --cases 1 2 3 ... 23 for all)")
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = create_output_dir(timestamp)
    
    print("=" * 80)
    print("TRAINING GENERALIZED POLICY ACROSS MULTIPLE CASES")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Cases to train on:  {args.cases}")
    print(f"  Total steps:        {args.num_steps:,}")
    print(f"  Checkpoint freq:    {args.checkpoint_freq:,}")
    print(f"  Parallel workers:   {args.num_workers}")
    print(f"  Learning rate:      {args.lr}")
    print(f"  Master seed:        {args.master_seed if args.master_seed is not None else 'random'}")
    print(f"  MLP architecture:   {args.mlp_hiddens}")
    print(f"\nOutput directory:     {output_dir}")
    print(f"Timestamp:            {timestamp}")
    print("=" * 80 + "\n")
    
    model = None  # Initialize to handle potential unbound errors in exception handlers
    
    try:
        # Create environment (will randomize case/seed at each reset with reproducible sequence)
        env = RandomCaseEnv(
            cases_to_train=args.cases,
            num_seeds=100,
            dt=0.5,
            sim_time=1950.0,
            n_heading=7,
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
        
        metrics_callback = GeneralizedTrainingMetricsCallback(verbose=1)
        
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
        plot_training_convergence(metrics_callback.timesteps, metrics_callback.mean_returns, 
                                 output_dir / "training_convergence.png")
        
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
            "cases_trained": [1, 6, 21],
            "num_seeds": 100,
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
        print(f"\nNext steps:")
        print(f"1. Evaluate on Case 1:")
        print(f"   python -m maritime_rl_pkg.eval_single_agent_sb3 \\")
        print(f"     --checkpoint '{final_path}' --case 1 \\")
        print(f"     --episodes 100 --seed 0 --save_histories")
        print(f"\n2. Evaluate on Case 6:")
        print(f"   python -m maritime_rl_pkg.eval_single_agent_sb3 \\")
        print(f"     --checkpoint '{final_path}' --case 6 \\")
        print(f"     --episodes 100 --seed 0 --save_histories")
        print(f"\n3. Evaluate on Case 21:")
        print(f"   python -m maritime_rl_pkg.eval_single_agent_sb3 \\")
        print(f"     --checkpoint '{final_path}' --case 21 \\")
        print(f"     --episodes 100 --seed 0 --save_histories")
        print(f"\n4. Compare results with baseline and case-specific policies")
        print("=" * 80 + "\n")
        
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
