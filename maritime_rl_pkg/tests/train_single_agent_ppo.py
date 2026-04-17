"""
Train single-agent PPO on ownship only using Maritime env wrapper.

Only ownship (agent 0) learns; obstacles use scripted actions (maintain heading).

Usage:
python -m maritime_rl_pkg.train_single_agent_ppo \
  --case 6 \
  --num_iterations 50 \
  --workers 2 \
  --seed 0
"""

import argparse
from pathlib import Path
from datetime import datetime

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.gymnasium_env import GymWrapper


def parse_args():
    p = argparse.ArgumentParser(description="Train single-agent PPO on ownship only")
    p.add_argument("--case", type=int, required=True, help="CORALL case number")
    p.add_argument("--num_iterations", type=int, default=50, help="Number of PPO training iterations")
    p.add_argument("--workers", type=int, default=2, help="Number of parallel env workers")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--dt", type=float, default=0.5, help="Time step (seconds)")
    p.add_argument("--sim_time", type=float, default=1950.0, help="Episode length (seconds)")
    p.add_argument("--route_len_nmi", type=float, default=2.0, help="Route length (NMI)")
    p.add_argument("--checkpoint_freq", type=int, default=5, help="Save checkpoint every N iterations")
    
    return p.parse_args()


def train_single_agent_ppo(args):
    """Train single-agent PPO policy."""
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Environment creator
    def env_creator(config):
        from maritime_rl_pkg.tests.env_single_agent_ppo import SingleAgentOwnshipEnv
        return SingleAgentOwnshipEnv(
            case_number=config.get("case_number", args.case),
            dt=config.get("dt", args.dt),
            sim_time=config.get("sim_time", args.sim_time),
            route_len_nmi=config.get("route_len_nmi", args.route_len_nmi),
            seed=config.get("seed", args.seed),
        )
    
    # Register environment
    env_name = f"SingleAgentOwnship_case{args.case}"
    tune.register_env(env_name, lambda cfg: GymWrapper(env_creator(cfg)))
    
    # Probe environment for spaces
    tmp_env = env_creator({
        "case_number": args.case,
        "seed": args.seed,
    })
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    tmp_env.close()
    
    # PPO Config
    config = (
        PPOConfig()
        .environment(
            env=env_name,
            env_config={
                "case_number": args.case,
                "dt": args.dt,
                "sim_time": args.sim_time,
                "route_len_nmi": args.route_len_nmi,
                "seed": args.seed,
            },
        )
        .framework("torch")
        .rl_module(
            model_config_dict={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "use_lstm": False,
            }
        )
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_clip_param=0.3,
            num_sgd_iter=20,
            sgd_minibatch_size=128,
        )
        .rollouts(
            num_rollout_workers=args.workers,
            num_envs_per_worker=1,
            rollout_fragment_length="auto",
        )
        .resources(num_gpus=0)
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
    )
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(f"SINGLE_AGENT_ppo_case{args.case}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build algorithm
    algo = config.build()
    
    print(f"\n{'='*70}")
    print(f"Single-Agent PPO Training (Ownship Only)")
    print(f"{'='*70}")
    print(f"Case: {args.case}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Workers: {args.workers}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Training loop
    best_reward = -float('inf')
    
    for iteration in range(args.num_iterations):
        result = algo.train()
        
        episode_reward_mean = result.get("env_runners", {}).get("episode_reward_mean", 0.0)
        
        print(f"[Iter {iteration+1:3d}] "
              f"train_return={result['env_runners']['episode_reward_mean']:7.2f} "
              f"policy_loss={result['learner']['default_policy']['policy_loss']:7.4f} "
              f"episodes={result['env_runners']['num_episodes']:5.0f}")
        
        # Save checkpoint periodically
        if (iteration + 1) % args.checkpoint_freq == 0:
            checkpoint_path = algo.save(str(output_dir / f"checkpoint_{iteration+1:03d}"))
            print(f"  → Checkpoint saved: {checkpoint_path}")
        
        # Track best
        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean
            best_checkpoint_path = algo.save(str(output_dir / "best_checkpoint"))
            print(f"  → Best checkpoint updated: {best_checkpoint_path} (reward={best_reward:.2f})")
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    algo.stop()
    ray.shutdown()
    
    return output_dir


def main():
    args = parse_args()
    train_single_agent_ppo(args)


if __name__ == "__main__":
    main()
