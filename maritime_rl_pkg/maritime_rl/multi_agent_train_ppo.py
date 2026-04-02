"""
RL Lib PPO training script for MultiShipParallelEnv

Notes: 
- adjust imports depending on RL version
- supports PettingZoo ParallelEnv via PettingZooEnv wrapper

Example usage:
python -m maritime_rl_pkg.maritime_rl.multi_agent_train_ppo --case 2 --iters 50 --num_workers 1 --rollout_frag 1000 --train_batch 4000 --seed 0

"""

import argparse
import time
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for plotting in headless environments
import matplotlib.pyplot as plt
import numpy as np
import torch

def parse_args():
    """Make it easier to make case numbers and training outputs clear after trained
    
    outer loop parameters:
    --case: which CORALL Imazu case to train on 
    --iters: number of "train iteration" training library should run 
    --num_workers: how many rollout worker processes collect experience in parallel
    --rollout_frag: how many env steps each worker collects per 'fragment' before sending to the learner for training. Larger means less communication overhead but more stale data.
    --train_batch: how many total env steps aggregated into training batch per iteration
    --lr: learning rate for policy optimization
    --gamma: discount factor for future rewards
    --seed: random seed for reproducibility
    
    """
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=int, default=2, help="CORALL Imazu case_number")
    p.add_argument("--iters", type=int, default=200, help="Training iterations")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--rollout_frag", type=int, default=2000)
    p.add_argument("--train_batch", type=int, default=8000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--route_len_nmi", type=float, default=2.0, help="Route length in nautical miles (scaling)")
    p.add_argument("--sim_time", type=float, default=300.0, help="Max simulation time in seconds (scaling)")
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--n_eval_episodes", type=int, default=3, help="Number of evaluation episodes to run at each evaluation checkpoint")
    p.add_argument("--ckpt_every", type=int, default=25, help="How often (in training iterations) to save checkpoints")
    return p.parse_args()

def run_policy_evaluation(algo, env_creator, n_eval_episodes=5):
    """
    Run deterministic evaluation rollouts with current PPO policy during training
    Returns average episode-level metrics 
    """
    eval_returns = []
    eval_lengths = []

    for ep in range(n_eval_episodes):
        env = env_creator({})
        obs, infos = env.reset()

        ep_return_by_agent = {agent: 0.0 for agent in env.agents}
        ep_len = 0

        while True: 
            actions = {}

            for agent_id, agent_obs in obs.items():
                action = algo.compute_single_action(
                    agent_obs, 
                    policy_id="shared_policy",
                    explore=False,
                )
                actions[agent_id] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)

            ep_len += 1
            for agent_id, reward in rewards.items():
                ep_return_by_agent[agent_id] += float(reward)

            # In multi-agent maritime scenarios, continue until ALL agents are done 
            # to preserve complete trajectories for each agent's learning
            if all(terminations.values()) or all(truncations.values()):
                break

        # mean return over agents for this eval episode
        eval_returns.append(float(np.mean(list(ep_return_by_agent.values()))))
        eval_lengths.append(ep_len)
        
        if hasattr(env, "close"):
            env.close()

    return {
        "eval_return_mean": float(np.mean(eval_returns)) if eval_returns else float("nan"),
        "eval_ep_length_mean": float(np.mean(eval_lengths)) if eval_lengths else float("nan"),
    }

def moving_average(values, window=10):
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0: 
        return vals
    out = np.full_like(vals, np.nan, dtype=float)
    for i in range(len(vals)):
        lo = max(0, i-window+1)
        chunk = vals[lo:i + 1]
        chunk = chunk[np.isfinite(chunk)]
        if len(chunk) > 0:
            out[i] = np.mean(chunk)
    return out 

def main():
    args = parse_args()
    print(f"[DEBUG] Parsed args: {args}")

    # Create an output folder for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(f"MARL_ppo_case{args.case}_{timestamp}") / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[logging] writing outputs to: {output_dir.resolve()}")

    # local imports so file can be imported without Ray installed
    import ray
    from ray import tune
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.ppo import PPO

    from maritime_rl_pkg.maritime_rl.multi_agent_env_ppo import MultiShipParallelEnv

    # Explicitly initialize Ray with timeout to avoid hangs on Windows
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, _temp_dir=None, include_dashboard=False)

    def env_creator(config):
        case_number = config.get("case_number", args.case)
        env = MultiShipParallelEnv(
            case_number=case_number, 
            dt=config.get("dt", 0.5),
            sim_time=config.get("sim_time", args.sim_time),
            route_len_nmi=config.get("route_len_nmi", args.route_len_nmi),
            seed=config.get("seed", args.seed)
        )
        return env

    # register PettingZoo parallel env with RLlib 
    tune.register_env("corall_mappo_env", lambda cfg: ParallelPettingZooEnv(env_creator(cfg)))


    # create temporary env instance to read spaces and agent ids 
    # --> probe env once to get spaces + agent_ids (so RLlib can build policy model with correct input/output sizes)
    tmp_env = env_creator({"case_number": args.case, "seed": args.seed})
    obs_space = tmp_env.observation_space(tmp_env.agents[0])
    act_space = tmp_env.action_space(tmp_env.agents[0])
    # agent_ids = tmp_env.agents
    tmp_env.close() if hasattr(tmp_env, "close") else None

    def policy_mapping_fn(agent_id, *args, **kwargs):
        """
        Tell RLlib that all agents share one policy (indicate MAPPO type policy)
        """
        return "shared_policy"
    
    # Build PPO algorithm configuration using RLlib's config API based on docs available
    config = (
        PPOConfig()
        .environment(
            env="corall_mappo_env",
            env_config={
                "case_number": args.case,
                "seed": args.seed,
                "sim_time": args.sim_time,
                "route_len_nmi": args.route_len_nmi,
            }, 
        )
        .framework("torch")
        .env_runners(
            num_env_runners=args.num_workers,
            rollout_fragment_length=args.rollout_frag
        )
        .training(
            lr=args.lr,
            gamma=args.gamma,
            train_batch_size=args.train_batch,
            clip_param=0.2, 
            vf_clip_param=10.0,
            entropy_coeff=0.0,
            lambda_=0.95, 
            num_epochs=10,
            minibatch_size=128,
        )
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})}, 
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"]
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
    )

    # instantiate PPO algorithm with config
    algo = PPO(config=config)

    # Create subdirectory for checkpoints
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "training_metrics.csv"
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "iteration", 
            "episode_return_mean", 
            "episode_len_mean",
            "eval_return_mean",
            "eval_ep_length_mean"
        ])
    
    xs_train, ys_train = [], []
    xs_eval, ys_eval = [], []

    for i in range(args.iters):
        result = algo.train()

        runner_stats = result.get("env_runners", {})
        rew = runner_stats.get("episode_return_mean", float("nan"))
        ep_len = runner_stats.get("episode_len_mean", float("nan"))

        xs_train.append(i+1)
        ys_train.append(rew)

        # Run evaluation periodically
        eval_return_mean = float("nan")
        eval_ep_length_mean = float("nan")
        
        if (i+1) % args.eval_every == 0:
            try:
                eval_results = run_policy_evaluation(
                    algo, 
                    env_creator, 
                    n_eval_episodes=args.n_eval_episodes
                )
                eval_return_mean = eval_results["eval_return_mean"]
                eval_ep_length_mean = eval_results["eval_ep_length_mean"]
                xs_eval.append(i+1)
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
                i+1, 
                rew, 
                ep_len,
                eval_return_mean,
                eval_ep_length_mean
            ])
        
        if (i+1) % args.ckpt_every == 0:
            ckpt_path = algo.save(str(checkpoint_dir))
            print(f"Checkpoint saved at iteration {i+1}: {ckpt_path}")
    
    final_ckpt_path = algo.save(str(checkpoint_dir))
    print(f"Training complete. Final checkpoint saved at: {final_ckpt_path}")

    # Final training plot
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(xs_train, ys_train, label="train_return", color="blue", alpha=0.5)
    ax1.plot(xs_train, moving_average(ys_train, window=10), label="train_return (moving avg)", color="darkblue", linestyle="--")
    
    if len(xs_eval) > 0:
        ax1.plot(xs_eval, ys_eval, marker="o", label="eval_return", color="orange", linestyle="-", markersize=6)
    
    ax1.set_title(f"PPO training vs evaluation return | case {args.case} | seed {args.seed}")
    ax1.set_xlabel("Training Iteration")
    ax1.set_ylabel("Mean Episode Return")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(output_dir / "training_return.png", dpi=200)
    plt.close(fig1)

    # Clean up Ray resources
    import ray
    if ray.is_initialized():
        ray.shutdown()

if __name__ == "__main__":
    main()