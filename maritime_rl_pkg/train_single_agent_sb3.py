"""
Train single-agent PPO on ownship collision avoidance using Stable-Baselines3.

OVERVIEW:
=========
This script trains a reinforcement learning policy to control an ownship through maritime
collision avoidance scenarios. Only the ownship (agent_0) is trained; obstacles use
deterministic scripted actions (maintain constant heading/speed).

CASES & DIFFICULTY:
===================
- Case 1: scale=1.0  → Original Imazu geometry, loose encounters
- Case 6: scale=0.75 → 25% closer targets, medium difficulty  
- Case 21: scale=0.5 → 50% closer targets, tight encounters

All cases:
  - Waypoint distance: 2.0 NMI (3,704 m) straight ahead
  - Obstacle speeds: uniform 9.5 m/s
  - Episode length: 1950 seconds (~32.5 min simulation)
  - Obstacles head toward ownship from initial positions

TRAINING PROCESS:
=================
1. Environment: SingleAgentOwnshipEnv wraps MultiShipParallelEnv
   - Ownship receives reward from collision penalty + goal progress
   - Obstacles maintain scripted collision-course heading

2. Algorithm: Stable-Baselines3 PPO
   - Policy: neural network (MLP) mapping observations → actions
   - Observations: ownship state (7 dim) + waypoint (4 dim) + obstacles' relative positions
   - Actions: discrete heading + speed (7×5 = 35 total action combinations)
   - Hyperparameters tuned for maritime collision avoidance

3. Callbacks:
   - CheckpointCallback: saves model every 10k steps
   - EvalCallback: runs 3-episode validation every 10k steps
   - TrainingMetricsCallback: logs training progress

4. Output:
   - SINGLE_AGENT_SB3_case{X}_{timestamp}/
     ├── checkpoints/        (model snapshots every 10k steps)
     ├── best_model/         (best validation checkpoint)
     ├── eval_logs/          (evaluation npz files)
     ├── best_checkpoint.zip (final saved model)
     ├── training_return.png (convergence plot)
     └── training_metrics.json

USAGE:
======
python -m maritime_rl_pkg.train_single_agent_sb3 --case 6 --num_steps 100000 --seed 0

Full three-case training:
  python -m maritime_rl_pkg.train_single_agent_sb3 --case 1  --num_steps 100000 --seed 0
  python -m maritime_rl_pkg.train_single_agent_sb3 --case 6  --num_steps 100000 --seed 0
  python -m maritime_rl_pkg.train_single_agent_sb3 --case 21 --num_steps 100000 --seed 0

EXPECTED TRAINING TIME:
======================
- 100k steps ≈ 25 episodes ≈ 500s sim + RL overhead ≈ 30-50 min per case
- Total for 3 cases: approximately 2-3 hours (depending on CPU)

CONVERGENCE INDICATORS:
======================
- Episode return gradually increases (less negative) over time
- Goal progress approaches max value
- Collision rate decreases as policy learns avoidance
- Validation return (3-ep evals) should track training trend
"""

import argparse
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback


class TrainingMetricsCallback(BaseCallback):
    """Track training metrics from model logger."""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.verbose = verbose
    
    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Log every 1000 steps
        if self.num_timesteps % 1000 == 0 and self.num_timesteps > 0:
            # Try to get mean episode reward from logger
            if hasattr(self.model.logger, 'name_to_value'):
                mean_reward = self.model.logger.name_to_value.get('rollout/ep_rew_mean', 0.0)
                if self.verbose:
                    print(f"[Step {self.num_timesteps:6d}] Mean Ep Return: {mean_reward:8.2f}")
        
        return True
    
    def extract_training_logs(self):
        """Extract training metrics from model logger."""
        if not hasattr(self.model.logger, 'name_to_value'):
            return [0], [0]
        
        logger = self.model.logger
        # SB3 logger stores lists under each key name
        timesteps_key = 'time/total_timesteps'
        reward_key = 'rollout/ep_rew_mean'
        
        timesteps = getattr(logger, 'name_to_value', {}).get(timesteps_key, [])
        rewards = getattr(logger, 'name_to_value', {}).get(reward_key, [])
        
        # If logger is empty, try to extract from histories
        if not timesteps or not rewards:
            # Fallback: return placeholder
            timesteps = [0, self.model.num_timesteps]
            rewards = [0, 0]
        
        return timesteps, rewards
    
    def save_metrics(self, output_dir):
        """Save metrics to JSON."""
        timesteps, rewards = self.extract_training_logs()
        metrics = {
            "timesteps": timesteps,
            "episode_returns": rewards,
        }
        json_path = output_dir / "training_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        return json_path


def parse_args():
    p = argparse.ArgumentParser(description="Train single-agent PPO on ownship using Stable-Baselines3")
    p.add_argument("--case", type=int, required=True, help="CORALL case number")
    p.add_argument("--num_steps", type=int, default=100000, help="Total number of environment steps")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--dt", type=float, default=0.5, help="Time step (seconds)")
    p.add_argument("--sim_time", type=float, default=1950.0, help="Episode length (seconds)")
    p.add_argument("--route_len_nmi", type=float, default=2.0, help="Route length (NMI)")
    p.add_argument("--checkpoint_freq", type=int, default=10000, help="Save checkpoint every N steps")
    
    return p.parse_args()


def plot_training_curves(model, eval_log_dir, output_dir):
    """Plot training + validation convergence curves."""
    import os
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Extract training returns from model logger
    train_returns = []
    train_steps = []
    
    if hasattr(model.logger, 'name_to_value'):
        try:
            # Try to get from logger's internal storage
            for key, val in model.logger.name_to_value.items():
                if 'ep_rew_mean' in key and isinstance(val, (list, np.ndarray)):
                    train_returns = list(val)
                    break
        except:
            pass
    
    if not train_returns:
        train_returns = [0]
    if not train_steps:
        train_steps = list(range(0, model.num_timesteps, max(1, model.num_timesteps // len(train_returns))))
        train_steps = train_steps[:len(train_returns)]
    
    # Plot training returns
    if train_returns and len(train_returns) > 1:
        window = max(1, len(train_returns) // 10)
        if window > 1:
            smoothed = np.convolve(train_returns, np.ones(window)/window, mode='valid')
            smoothed_steps = train_steps[window-1:len(smoothed)+window-1]
            ax.plot(smoothed_steps, smoothed, linewidth=2.5, label='Training (smoothed)', color='blue', alpha=0.8)
        ax.plot(train_steps, train_returns, alpha=0.2, label='Training (raw)', color='blue')
    
    # Try to load eval results
    if eval_log_dir and os.path.exists(eval_log_dir):
        try:
            eval_data = np.load(os.path.join(eval_log_dir, 'evaluations.npz'), allow_pickle=True)
            timesteps_array = eval_data['timesteps']
            results_array = eval_data['results']
            
            # Extract mean validation return per evaluation
            eval_returns = [np.mean(r) for r in results_array] if len(results_array) > 0 else []
            eval_steps = list(timesteps_array) if len(timesteps_array) > 0 else []
            
            if eval_returns and eval_steps:
                ax.plot(eval_steps, eval_returns, 'o-', linewidth=2, markersize=6, 
                       label='Validation (3-episode eval)', color='red', alpha=0.7)
        except:
            pass
    
    ax.set_xlabel('Timesteps', fontsize=12)
    ax.set_ylabel('Episode Return', fontsize=12)
    ax.set_title('Training Convergence: Training vs Validation Returns', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    
    plot_path = output_dir / "training_return.png"
    plt.savefig(str(plot_path), dpi=300, format='png')
    pdf_plot_path = output_dir / "training_return.pdf"
    plt.savefig(str(pdf_plot_path), format='pdf')
    print(f"✓ Convergence plot saved to: {plot_path} and {pdf_plot_path}")
    
    plt.close()


def train_single_agent_sb3(args):
    """
    Train single-agent PPO policy using Stable-Baselines3.
    
    HYPERPARAMETER RATIONALE:
    =========================
    learning_rate=3e-4:
        - Standard value for PPO on continuous/discrete tasks
        - Lower than supervised learning due to on-policy sampling
        - Policy updates are conservative to maintain stability
    
    n_steps=2048:
        - Trajectory collection size per update
        - ~5 episodes per batch at ~400-500 steps per episode
        - Larger batches reduce sample efficiency but improve gradient estimates
    
    batch_size=64:
        - Mini-batch for SGD within each epoch
        - Divides n_steps into 32 gradient updates per epoch
        - Smaller batches = more frequent updates but noisier gradients
    
    n_epochs=10:
        - Reuses same trajectory 10 times with different mini-batches
        - Balances sample efficiency (reuse) vs stale data (old rollouts)
    
    gamma=0.99:
        - Discount factor for long-horizon rewards
        - 0.99 weights steps 100 timesteps ahead at 37% importance
        - Maritime incidents can unfold over 30-60 timesteps, so need long-term credit
    
    gae_lambda=0.95:
        - Generalized Advantage Estimation parameter
        - 0.95 = blend between TD(0) and Monte Carlo return
        - Good for continuous state spaces with function approximation
    
    clip_range=0.2:
        - PPO clipping range for policy ratio
        - Prevents overly large policy updates
        - 20% range is standard (ratios between 0.8-1.2)
    
    ent_coef=0.01:
        - Entropy bonus strength (exploration)
        - Encourages policy to not become too deterministic too early
        - 0.01 is light exploration, avoids random behavior in later training
    """
    
    from maritime_rl_pkg.env_single_agent_sb3 import SingleAgentOwnshipEnv
    
    # Create environment
    env = SingleAgentOwnshipEnv(
        case_number=args.case,
        dt=args.dt,
        sim_time=args.sim_time,
        route_len_nmi=args.route_len_nmi,
        seed=args.seed,
    )
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(f"SINGLE_AGENT_SB3_case{args.case}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Single-Agent PPO Training (Stable-Baselines3)")
    print(f"{'='*70}")
    print(f"Case: {args.case}")
    print(f"Total steps: {args.num_steps}")
    print(f"Checkpoint frequency: {args.checkpoint_freq} steps")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="rl_model",
        save_replay_buffer=False,
    )
    
    # Create PPO model with tuned hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        seed=args.seed,
        policy_kwargs=dict(net_arch=[256, 256]),  # Increased from default [64, 64] for better convergence
    )
    
    # Callbacks
    metrics_callback = TrainingMetricsCallback(verbose=1)
    
    eval_env = SingleAgentOwnshipEnv(
        case_number=args.case,
        dt=args.dt,
        sim_time=args.sim_time,
        route_len_nmi=args.route_len_nmi,
        seed=args.seed + 100,  # Different seed for eval
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=args.checkpoint_freq,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )
    
    print("Starting training...\n")
    model.learn(
        total_timesteps=args.num_steps,
        callback=[checkpoint_callback, metrics_callback, eval_callback],
        progress_bar=True,
    )
    
    # Save training metrics
    metrics_json = metrics_callback.save_metrics(output_dir)
    print(f"\n✓ Training metrics saved to: {metrics_json}")
    
    # Plot training + validation curves
    plot_training_curves(
        model=model,
        eval_log_dir=str(output_dir / "eval_logs"),
        output_dir=output_dir,
    )
    
    # Save final model
    final_path = output_dir / "final_model"
    model.save(str(final_path))
    print(f"✓ Final model saved to: {final_path}")
    
    # Also save as "best_checkpoint" for consistency
    best_path = output_dir / "best_checkpoint"
    model.save(str(best_path))
    print(f"✓ Checkpoint saved to: {best_path}")
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    env.close()
    eval_env.close()


def main():
    args = parse_args()
    train_single_agent_sb3(args)


if __name__ == "__main__":
    main()
