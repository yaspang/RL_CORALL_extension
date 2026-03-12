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
import matplotlib.pyplot as plt
import numpy as np

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
    return p.parse_args()

def run_policy_evaluation(algo, env_config, n_eval_episodes=5):
    """
    Run deterministic evaluation rollouts with current PPO policy during training
    Returns average episode-level metrics 
    """
    eval_returns = []
    eval_lengths = []
    eval_collision_flags = []
    eval_success_rates = []
    eval_path_lengths = []
    eval_min_dcpas = []
    eval_risk_exposures = []
    eval_completion_times = []

    policy = algo.get_policy("shared_policy")

    for ep in range(n_eval_episodes):
        env = env_config({})
        obs, infos = env.reset()

        done = {agent: False for agent in env.agents}
        truncated = {agent: False for agent in env.agents}

        ep_return_by_agent = {agent: 0.0 for agent in env.agents}
        ep_len = 0
        final_infos = None

        while True: 
            actions = {}

            for agent_id, agent_obs in obs.items():
                action, _, _ = policy.compute_single_action(agent_obs, explore=False) # deterministic evaluation
                actions[agent_id] = action

            obs, rewards, terminations, truncated, infos = env.step(actions)

            ep_len += 1
            for agent_id, reward in rewards.items():
                ep_return_by_agent[agent_id] += float(reward)
            
            final_infos = infos

            any_done = any(terminations.values())
            any_trunc = any(truncated.values())
            if any_done or any_trunc:
                break

        # mean return over agents for this eval episode
        eval_returns.append(float(np.mean(list(ep_return_by_agent.values()))))
        eval_lengths.append(ep_len)

        # read episode metrics from final infos 
        metrics_by_agent = []
        if final_infos is not None:
            for agent_id, info in final_infos.items():
                epm = info.get("episode_metrics", None)
                if isinstance(epm, dict):
                    metrics_by_agent.append(epm)
        
        if metrics_by_agent:
            eval_collision_flags.append(float(any(int(epm.get("collision", 0)) for epm in metrics_by_agent)))
            eval_success_rates.append(float(np.mean([int(epm.get("success", 0)) for epm in metrics_by_agent])))
            eval_path_lengths.append(float(np.nanmean([epm.get("path_length_m", np.nan) for epm in metrics_by_agent])))
            eval_min_dcpas.append(float(np.nanmean([epm.get("min_dcpa_m", np.nan) for epm in metrics_by_agent])))
            eval_risk_exposures.append(float(np.nanmean([epm.get("risk_exposure", np.nan) for epm in metrics_by_agent])))

            cts = [epm.get("completion_time_s", np.nan) for epm in metrics_by_agent]
            cts = [x for x in cts if x is not None and not np.isnan(x)]

            if len(cts) > 0: 
                eval_completion_times.append(float(np.mean(cts)))
        
        if hasattr(env, "close"):
            env.close()

    return {
        "eval_return_mean": float(np.mean(eval_returns)) if eval_returns else float("nan"),
        "eval_ep_length_mean": float(np.mean(eval_lengths)) if eval_lengths else float("nan"),
        "eval_collision_rate_mean": float(np.mean(eval_collision_flags)) if eval_collision_flags else float("nan"),
        "eval_success_rate_mean": float(np.mean(eval_success_rates)) if eval_success_rates else float("nan"),
        "eval_path_length_m_mean": float(np.mean(eval_path_lengths)) if eval_path_lengths else float("nan"),
        "eval_min_dcpa_m_mean": float(np.mean(eval_min_dcpas)) if eval_min_dcpas else float("nan"),
        "eval_risk_exposure_mean": float(np.mean(eval_risk_exposures)) if eval_risk_exposures else float("nan"),
        "eval_completion_time_s_mean": float(np.mean(eval_completion_times)) if eval_completion_times else float("nan"),
    }


def main():
    args = parse_args()

    # Create an output folder for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(f"MARL_ppo_case{args.case}_{timestamp}") / f"seed_{args.seed}" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[logging] writing outputs to: {output_dir.resolve()}")

    # local imports so file can be imported without Ray installed
    from ray import tune
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.ppo import PPO

    from maritime_rl_pkg.maritime_rl.multi_agent_env_ppo import MultiShipParallelEnv
    

    def env_creator(config):
        case_number = config.get("case_number", args.case)
        return MultiShipParallelEnv(
            case_number=case_number, 
            dt=config.get("dt", 0.2),
            sim_time=config.get("sim_time", 300.0),
            seed=config.get("seed", args.seed)
        )

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
    print("Creating PPOConfig...")    
    config = PPO.get_default_config()

    # set environment
    config["env"] = "corall_mappo_env"
    config["env_config"] = {"case_number": args.case, "seed": args.seed}

    # set framework
    config["framework"] = "torch"

    # set rollout workers
    config["num_rollout_workers"] = args.num_workers
    config["rollout_fragment_length"] = args.rollout_frag

    # set training parameters
    config["lr"] = args.lr
    config["gamma"] = args.gamma
    config["train_batch_size"] = args.train_batch
    # optional PPO hyperparameters (tune as needed)
    config["clip_param"] = 0.2
    config["vf_clip_param"] = 10.0
    config["entropy_coeff"] = 0.0
    config["lambda"] = 0.95
    config["num_sgd_iter"] = 10
    config["sgd_minibatch_size"] = 2048 

    # multi-agent setup
    config["multiagent"] = {
        "policies": {"shared_policy": (None, obs_space, act_space, {})},
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": ["shared_policy"],
    }

    # seed
    config["seed"] = args.seed

    # log checkpoints to output folder
    config["logger_config"] = {
    "type": "ray.tune.logger.UnifiedLogger",
    "logdir": str(output_dir),
    }

    print(f"Config: {config}")

    algo = config.build()

    
    plt.ion()

    # Plot live rewards during training 
    fig_train, ax_train = plt.subplots()
    ax_train.set_title(f"PPO training reward | case {args.case} | seed {args.seed}")
    ax_train.set_xlabel("Training Iteration")
    ax_train.set_ylabel("Episode Reward Mean")
    xs_train, ys_train = [], []
    (line_train,) = ax_train.plot(xs_train, ys_train, label="reward")

    # Plot live evaluation of returns during training
    fig_eval, ax_eval = plt.subplots()
    ax_eval.set_title(f"PPO evaluation return | case {args.case} | seed {args.seed}")
    ax_eval.set_xlabel("Training Iteration")
    ax_eval.set_ylabel("Evaluation Mean Episode Return")
    xs_eval, ys_eval = [], []
    (line_eval,) = ax_eval.plot(xs_eval, ys_eval, label="eval_return")


    # CSV logging setup 
    csv_path = output_dir / "training_metrics.csv"
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "iteration", 
            "episode_reward_mean", 
            "episode_len_mean", 
            "custom_metrics/path_length_m_mean",
            "custom_metrics/min_dcpa_m_mean",
            "custom_metrics/risk_exposure_mean",
            "custom_metrics/collision_rate_episode_mean",
            "custom_metrics/success_rate_mean",
            "custom_metrics/collision_rate_mean",
            "custom_metrics/completion_time_s_mean",
            "eval_return_mean", 
            "eval_collision_rate_mean",
            "eval_success_rate_mean",
            "eval_path_length_m_mean",
            "eval_min_dcpa_m_mean",
            "eval_risk_exposure_mean", 
            "eval_completion_time_s_mean",
        ])


    # training loop (
    ## collect rollouts and print desired metrics at each checkpoint)
    plot_every = 1
    save_png_every = 10
    ckpt_every = 25
    eval_every = 5
    n_eval_episodes = 5

    for i in range(args.iters):
        result = algo.train()

        custom = result.get("custom_metrics", {})

        # Data for mean episodic return over training (validation signal)
        eval_return_mean = float("nan")

        # Data for rewards over training
        rew = result.get("episode_reward_mean", float("nan"))
        ep_len = result.get("episode_len_mean", float("nan"))

        if (i+1) % eval_every == 0:
            eval_results = run_policy_evaluation(algo, env_creator, n_eval_episodes=n_eval_episodes)
            eval_return_mean = eval_results["eval_return_mean"]            
            xs_eval.append(i)
            ys_eval.append(eval_return_mean)

        # print progress
        if i % plot_every == 0:
            # training reward plot 
            xs_train.append(i)
            ys_train.append(rew)
            line_train.set_data(xs_train, ys_train)
            ax_train.relim()
            ax_train.autoscale_view()
            fig_train.canvas.draw()
            fig_train.canvas.flush_events()

            # evaluation return plot
            line_eval.set_data(xs_eval, ys_eval)
            ax_eval.relim()
            ax_eval.autoscale_view()
            fig_eval.canvas.draw()
            fig_eval.canvas.flush_events()


        # write csv row 
        with open(csv_path, mode="a", newline="") as csv_file:
            csv.writer(csv_file).writerow([
                i, 
                rew, 
                ep_len,
                eval_return_mean
            ])

        # checkpointing
        if (i+1) % ckpt_every == 0:
            ckpt = algo.save()
            print(f"Checkpoint saved at iteration {i+1}: {ckpt}")
        
        # save plot snapshot
        if (i+1) % save_png_every == 0:
            fig_train.savefig(output_dir / f"training_plot_iter{i+1}.png", dpi=150)
            fig_eval.savefig(output_dir / f"evaluation_plot_iter{i+1}.png", dpi=150)
        
        # save checkpoint
        fig_train.savefig(output_dir / f"reward_plot_iter{i+1}.png", dpi=200)
        fig_eval.savefig(output_dir / f"eval_return_plot_iter{i+1}.png", dpi=200)

    # final save
    fig_train.savefig(output_dir / f"final_reward_plot.png", dpi=200)
    ckpt = algo.save()
    print("Saved finalcheckpoint:", ckpt)

if __name__ == "__main__":
    main()