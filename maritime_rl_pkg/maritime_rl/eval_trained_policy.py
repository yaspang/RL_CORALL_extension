"""
Evaluate a trained policy for a trained shared-policy PPO model using CORALL Imazu case on the MultiShipParallelEnv environment.

Usage example:
python -m maritime_rl_pkg.maritime_rl.evaluate_trained_policy ^
    --checkpoint "C:/path/to/checkpoint_dir/checkpoint_000200" ^
    --case 2 ^
    --episodes 50 ^
    --seed 0

> rebuild PPO algorithm with same env + shared policy config as training, load checkpoint, run evaluation episodes, and print/save results (e.g. to csv or json)
> restore trained policy from checkpoint, run evaluation episodes, and print/save results (e.g. to csv or json)
> run deterministic rollouts with explore=False to evaluate the learned policy's performance without stochasticity from action sampling, and print/save results (e.g. to csv or json)
> log per-episode and aggregate COLAV metrics

"""

import argparse
import json
import csv
import time
from pathlib import Path

import numpy as np
import ray

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint to load for evaluation")
    p.add_argument("--case", type=int, required=True, help="CORALL Imazu case number for the evaluation")
    p.add_argument("--episodes", type=int, default=20, help="Number of episodes to run for evaluation")
    p.add_argument("--seed", type=int, default=0, help="Base random seed for evaluation")
    p.add_argument("--dt", type=float, default=0.2, help="Time step duration in seconds for the environment")
    p.add_argument("--sim_time", type=float, default=300.0, help="Total simulation time in seconds for each episode")
    p.add_argument("--route_len_nmi", type=float, default=40.0, help="Route length in nautical miles (scaling factor for environment)")
    p.add_argument("--num_workers", type=int, default=0, help="Number of parallel workers to use for evaluation (default: 0 for standalone eval)")
    p.add_argument("--render", action="store_true", help="Whether to render the environment during evaluation")

    return p.parse_args()

def safe_mean(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if len(vals) > 0 else float('nan')

def build_algo_and_env(args):
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    
    from maritime_rl_pkg.maritime_rl.multi_agent_env_ppo import MultiShipParallelEnv

    def env_creator(config):
        return MultiShipParallelEnv(
            case_number=config.get("case_number", args.case),
            dt=config.get("dt", args.dt),
            sim_time=config.get("sim_time", args.sim_time),
            route_len_nmi=config.get("route_len_nmi", args.route_len_nmi),
            render_mode="human" if args.render else "none",
            seed=config.get("seed", args.seed),
        )
    
    # register env under a unique name for eval run
    env_name = f"corall_ppo_eval_env_case{args.case}_seed{args.seed}"
    tune.register_env(env_name, lambda cfg: ParallelPettingZooEnv(env_creator(cfg)))

    # probe spaces
    tmp_env = env_creator({"case_number": args.case, "seed": args.seed})
    obs_space = tmp_env.observation_space(tmp_env.agents[0])
    act_space = tmp_env.action_space(tmp_env.agents[0])
    tmp_env.close() if hasattr(tmp_env, "close") else None

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return "shared_policy"

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
        .env_runners(
            num_env_runners=args.num_workers,
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
        .debugging(seed=args.seed)
    )
    

    algo = config.build_algo()
    algo.restore(args.checkpoint)

    return algo, env_creator

def run_one_episode(algo, env_creator, seed):
    env = env_creator({"seed": seed})
    obs, infos = env.reset(seed=seed)

    policy = algo.get_policy("shared_policy")

    done = False
    step_count = 0
    reward_by_agent = {agent: 0.0 for agent in env.agents}
    final_infos = None

    while not done: 
        actions = {}

        for agent_id, agent_obs in obs.items():
            action, _, _ = policy.compute_single_action(agent_obs, explore=False)
            actions[agent_id] = action
        
        obs, rewards, terminations, truncations, infos = env.step(actions)

        step_count += 1

        for agent_id, reward in rewards.items():
            reward_by_agent[agent_id] += float(reward)
        
        final_infos = infos
        done = all(terminations.values()) or any(truncations.values())

    # extract episode metrics
    metrics_by_agent = {}

    if final_infos is not None:
        for agent_id, info in final_infos.items():
            epm = info.get("episode_metrics", None)
            if isinstance(epm, dict):
                metrics_by_agent[agent_id] = epm
    
    env.close() if hasattr(env, "close") else None

    # fallbacks if episode_metrics not present
    if not metrics_by_agent:
        return{
            "episode_steps": step_count, 
            "episode_return_mean": float(np.mean(list(reward_by_agent.values()))),
            "episode_return_ownship": float(reward_by_agent.get("ship_0", float('nan'))),
            "collision_any": float("nan"), 
            "success_rate_agents": float("nan"),
            "path_length_m_mean": float("nan"),
            "path_length_m_ownship": float("nan"),
            "min_dcpa_m_mean": float("nan"),
            "min_dcpa_m_ownship": float("nan"),
            "risk_exposure_mean": float("nan"),
            "risk_exposure_ownship": float("nan"),
            "completion_time_s_mean": float("nan"),
            "completion_time_s_ownship": float("nan"),

        }
    
    # debug to see if episode_metrics are being logged correctly
    if not metrics_by_agent:
        print(f"[WARNING] No episode_metrics found for seed={seed}, steps={step_count}")

    per_agent_path = []
    per_agent_dcpa = []
    per_agent_risk = []
    per_agent_success = []
    per_agent_collision = []
    per_agent_ct = []

    for agent_id, epm in metrics_by_agent.items():
        per_agent_path.append(float(epm.get("path_length_m", np.nan)))
        per_agent_dcpa.append(float(epm.get("min_dcpa_m", np.nan)))
        per_agent_risk.append(float(epm.get("risk_exposure", np.nan)))
        per_agent_success.append(int(bool(epm.get("success", 0))))
        per_agent_collision.append(int(bool(epm.get("collision", 0))))
        
        ct = epm.get("completion_time_s", None)
        if ct is not None: 
            per_agent_ct.append(float(ct))
        
    own = metrics_by_agent.get("ship_0", {})

    return {
        "episode_steps": step_count, 
        "episode_return_mean": float(np.mean(list(reward_by_agent.values()))),
        "episode_return_ownship": float(reward_by_agent.get("ship_0", np.nan)),
        "collision_any": float(any(per_agent_collision)),
        "success_rate_agents": float(np.mean(per_agent_success)),
        "path_length_m_mean": float(np.nanmean(per_agent_path)),
        "path_length_m_ownship": float(own.get("path_length_m", np.nan)),
        "min_dcpa_m_mean": float(np.nanmean(per_agent_dcpa)),
        "min_dcpa_m_ownship": float(own.get("min_dcpa_m", np.nan)),
        "risk_exposure_mean": float(np.nanmean(per_agent_risk)),
        "risk_exposure_ownship": float(own.get("risk_exposure", np.nan)),
        "completion_time_s_mean": float(np.nanmean(per_agent_ct)) if per_agent_ct else float("nan"),
        "completion_time_s_ownship": float(own.get("completion_time_s", np.nan)),
    }
        
def main():
    import ray 
    
    args = parse_args()

    # Initialize Ray BEFORE importing RLlib to avoid Windows import hangs
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True, 
            _temp_dir=None, 
            include_dashboard=False,
            num_cpus=1
        )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(f"policy_eval_case{args.case}_{timestamp}") / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    algo, env_creator = build_algo_and_env(args)

    per_episode_results = []

    for ep in range(args.episodes):
        ep_seed = args.seed + ep  
        row = run_one_episode(algo, env_creator, seed=ep_seed)
        row["episode_index"] = ep
        row["episode_seed"] = ep_seed
        per_episode_results.append(row)

        print(
            f"[eval ep {ep+1}/{args.episodes}] " 
            f"return_mean={row['episode_return_mean']:.3f}, " 
            f"collision_any={row['collision_any']}, "
            f"success_rate_agents={row['success_rate_agents']:.3f}, "
            f"min_dcpa_ownship={row['min_dcpa_m_ownship']:.3f}m, "
            f"risk_exposure_ownship={row['risk_exposure_ownship']:.3f}"

        )

    summary = {
        "checkpoint": args.checkpoint,
        "case": args.case,
        "episodes": args.episodes,
        "seed_base": args.seed,
        "episode_return_mean": safe_mean([r["episode_return_mean"] for r in per_episode_results]),
        "episode_return_ownship_mean": safe_mean([r["episode_return_ownship"] for r in per_episode_results]),
        "collision_rate": safe_mean([r["collision_any"] for r in per_episode_results]),
        "success_rate_agents_mean": safe_mean([r["success_rate_agents"] for r in per_episode_results]),
        "path_length_m_mean": safe_mean([r["path_length_m_mean"] for r in per_episode_results]),
        "path_length_m_ownship_mean": safe_mean([r["path_length_m_ownship"] for r in per_episode_results]),
        "min_dcpa_m_mean": safe_mean([r["min_dcpa_m_mean"] for r in per_episode_results]),
        "min_dcpa_m_ownship_mean": safe_mean([r["min_dcpa_m_ownship"] for r in per_episode_results]),
        "risk_exposure_mean": safe_mean([r["risk_exposure_mean"] for r in per_episode_results]),
        "risk_exposure_ownship_mean": safe_mean([r["risk_exposure_ownship"] for r in per_episode_results]),
        "completion_time_s_mean": safe_mean([r["completion_time_s_mean"] for r in per_episode_results]),
        "completion_time_s_ownship_mean": safe_mean([r["completion_time_s_ownship"] for r in per_episode_results]), 

    }
    
    # save per-episode csv
    csv_path = output_dir / "policy_eval_per_episode.csv"
    fieldnames = [
        "episode_index",
        "episode_seed",
        "episode_steps",
        "episode_return_mean",
        "episode_return_ownship",
        "collision_any",
        "success_rate_agents",
        "path_length_m_mean",
        "path_length_m_ownship",
        "min_dcpa_m_mean",
        "min_dcpa_m_ownship",
        "risk_exposure_mean",
        "risk_exposure_ownship",
        "completion_time_s_mean",
        "completion_time_s_ownship"
    ]

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_episode_results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # Save summary JSON
    summary_path = output_dir / "policy_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\nSaved per-episode CSV to: {csv_path}")
    print(f"Saved summary JSON to: {summary_path}")

    # Clean up Ray resources
    import ray
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main()