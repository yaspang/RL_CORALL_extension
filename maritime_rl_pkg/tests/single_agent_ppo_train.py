"""
Single-agent PPO training script using Ray RLlib for CORALL environment

This is a practice script to test training stability and incorporate validation signals
on the single-agent reactive avoidance task before scaling to multi-agent.

Example usage:
python -m maritime_rl_pkg.maritime_rl.train_single_agent_ppo --iters 50 --num_workers 1 --ckpt_every 10
"""

import argparse
import time
from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO


def parse_args():
    """Parse command line arguments"""
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=50, help="Training iterations")
    p.add_argument("--num_workers", type=int, default=1, help="Number of rollout workers")
    p.add_argument("--rollout_frag", type=int, default=1000, help="Rollout fragment length")
    p.add_argument("--train_batch", type=int, default=4000, help="Training batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--eval_every", type=int, default=5, help="Evaluate every N iterations")
    p.add_argument("--n_eval_episodes", type=int, default=5, help="Number of eval episodes")
    p.add_argument("--ckpt_every", type=int, default=10, help="Save checkpoint every N iterations")
    return p.parse_args()


def run_policy_evaluation(algo, env_creator, n_eval_episodes=5):
    """
    Run deterministic evaluation rollouts with current policy
    Returns average episode metrics
    """
    eval_returns = []
    eval_lengths = []

    for ep in range(n_eval_episodes):
        env = env_creator({})
        obs, infos = env.reset()

        ep_return = 0.0
        ep_len = 0

        while True:
            action = algo.compute_single_action(
                obs,
                policy_id="default_policy",
                explore=False,
            )

            obs, reward, terminated, truncated, infos = env.step(action)

            ep_len += 1
            ep_return += float(reward)

            if terminated or truncated:
                break

        eval_returns.append(float(ep_return))
        eval_lengths.append(ep_len)

        if hasattr(env, "close"):
            env.close()

    return {
        "eval_return_mean": float(np.mean(eval_returns)) if eval_returns else float("nan"),
        "eval_ep_length_mean": float(np.mean(eval_lengths)) if eval_lengths else float("nan"),
    }


def moving_average(values, window=10):
    """Compute moving average"""
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return vals
    out = np.full_like(vals, np.nan, dtype=float)
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        chunk = vals[lo : i + 1]
        chunk = chunk[np.isfinite(chunk)]
        if len(chunk) > 0:
            out[i] = np.mean(chunk)
    return out


def main():
    args = parse_args()
    print(f"[DEBUG] Parsed args: {args}")

    # Create output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(f"single_agent_ppo_{timestamp}") / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[logging] writing outputs to: {output_dir.resolve()}")

    # Local imports
    # from ray.rllib.env.wrappers.gym_wrapper import GymWrapper
    from .single_agent_ppo_env import CORALL_ReactiveAvoidanceGymEnv

    def env_creator(config):
        """Create single-agent environment"""
        return CORALL_ReactiveAvoidanceGymEnv(
                case_number=2,
                dt=0.2,
                K_obstacles=1,
                max_steps_cap=20000,
                seed=config.get("seed", args.seed),
            )

    # Register environment
    tune.register_env("corall_single_agent_env", env_creator)

    # Create temporary env to get observation and action spaces
    tmp_env = CORALL_ReactiveAvoidanceGymEnv(case_number=2, dt=0.2, K_obstacles=1, max_steps_cap=20000, seed=args.seed)
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    tmp_env.close() if hasattr(tmp_env, "close") else None

    # Build PPO configuration
    config = (
        PPOConfig()
        .environment(
            env="corall_single_agent_env",
            env_config={"seed": args.seed},
        )
        .framework("torch")
        .env_runners(
            num_env_runners=args.num_workers,
            rollout_fragment_length=args.rollout_frag,
            batch_mode="complete_episodes",
        )
        .training(
            lr=args.lr,
            gamma=0.99,
            train_batch_size=args.train_batch,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.0,
            lambda_=0.95,
            num_epochs=10,
            minibatch_size=128,
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )

    # Instantiate algorithm
    algo = PPO(config=config)

    # Create CSV for metrics
    csv_path = output_dir / "training_metrics.csv"
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "iteration",
            "episode_return_mean",
            "episode_len_mean",
            "eval_return_mean",
            "eval_ep_length_mean",
        ])

    xs_train, ys_train = [], []
    xs_eval, ys_eval = [], []

    # Training loop
    for i in range(args.iters):
        result = algo.train()

        runner_stats = result.get("env_runners", {})
        rew = runner_stats.get("episode_return_mean", float("nan"))
        ep_len = runner_stats.get("episode_len_mean", float("nan"))

        xs_train.append(i + 1)
        ys_train.append(rew)

        # Periodic evaluation
        eval_return_mean = float("nan")
        eval_ep_length_mean = float("nan")

        if (i + 1) % args.eval_every == 0:
            try:
                eval_results = run_policy_evaluation(
                    algo,
                    env_creator,
                    n_eval_episodes=args.n_eval_episodes,
                )
                eval_return_mean = eval_results["eval_return_mean"]
                eval_ep_length_mean = eval_results["eval_ep_length_mean"]
                xs_eval.append(i + 1)
                ys_eval.append(eval_return_mean)
            except Exception as e:
                print(f"[WARNING] Evaluation failed at iteration {i+1}: {e}")

        print(
            f"Iter {i+1}/{args.iters} | "
            f"train_return={rew:.2f} | "
            f"ep_len={ep_len:.2f} | "
            f"eval_return={eval_return_mean:.2f}"
        )

        with open(csv_path, mode="a", newline="") as csv_file:
            csv.writer(csv_file).writerow([
                i + 1,
                rew,
                ep_len,
                eval_return_mean,
                eval_ep_length_mean,
            ])

        if (i + 1) % args.ckpt_every == 0:
            ckpt = algo.save()
            print(f"Checkpoint saved at iteration {i+1}: {ckpt}")

    final_ckpt = algo.save()
    print(f"Training complete. Final checkpoint saved at: {final_ckpt}")

    # Save plots
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs_train, ys_train, label="train_return", color="blue", alpha=0.5)
    ax.plot(xs_train, moving_average(ys_train, window=5), label="train_return (moving avg)", color="darkblue", linestyle="--")

    if len(xs_eval) > 0:
        ax.plot(xs_eval, ys_eval, marker="o", label="eval_return", color="orange", linestyle="-", markersize=6)

    ax.set_title(f"Single-Agent PPO Training | Seed {args.seed}")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Mean Episode Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "training_return.png", dpi=300, format='png')
    plt.close(fig)

    print(f"Saved training plots to: {output_dir}")


if __name__ == "__main__":
    main()
