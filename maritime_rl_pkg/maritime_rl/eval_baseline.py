import argparse
import json
import csv
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
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

def build_env_creator(args):
    from maritime_rl_pkg.maritime_rl.multi_agent_env_ppo import MultiShipParallelEnv

    def env_creator(config):
        return MultiShipParallelEnv(
            case_number=config.get("case_number", args.case),
            dt=config.get("dt", args.dt),
            sim_time=config.get("sim_time", args.sim_time),
            route_len_nmi=config.get("route_len_nmi", args.route_len_nmi),
            render_mode="human" if args.render else None, 
            seed=config.get("seed", args.seed),
        )
    
    return env_creator

def run_one_episode_baseline(env_creator, seed, args):
    env = env_creator({"seed": seed})
    obs, infos = env.reset(seed=seed)

    done = False
    step_count = 0
    max_steps = int(np.ceil(args.sim_time / args.dt)) + 5
    reward_by_agent = {agent: 0.0 for agent in env.agents}
    final_infos = None

    
    while (not done) and (step_count < max_steps):
        actions = {}

        for agent_id in obs.keys():
            # simple baseline: always go full speed ahead (action=0) if not done
            if agent_id == "ship_0":
                actions[agent_id] = env.compute_corall_baseline_action(agent_id)
            else: 
                actions[agent_id] = env.get_default_agent_action(agent_id)

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

    own = metrics_by_agent.get("ship_0", {})
    ownship_success = float(bool(own.get("success", 0)))
    
    env.close() if hasattr(env, "close") else None

    # fallbacks if episode_metrics not present
    if not metrics_by_agent:
        return{
            "episode_steps": step_count, 
            "episode_return_mean": float(np.mean(list(reward_by_agent.values()))),
            "episode_return_ownship": float(reward_by_agent.get("ship_0", float('nan'))),
            "collision_any": float("nan"), 
            "success_ownship": ownship_success,
            "success_rate_agents": float("nan"),
            "path_length_m_mean": float("nan"),
            "path_length_m_ownship": float("nan"),
            "min_dcpa_m_mean": float("nan"),
            "min_dcpa_m_ownship": float("nan"),
            "risk_exposure_mean": float("nan"),
            "risk_exposure_ownship": float("nan"),
            "min_actual_sep_m_mean": float("nan"),
            "min_actual_sep_m_ownship": float("nan"),
            "near_miss_any": float("nan"),
            "goal_progress_mean": float("nan"),
            "goal_progress_ownship": float("nan"),
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
    per_agent_near_miss = []
    per_agent_min_sep = []
    per_agent_goal_progress = []
    per_agent_ct = []

    for agent_id, epm in metrics_by_agent.items():
        per_agent_path.append(float(epm.get("path_length_m", np.nan)))
        per_agent_dcpa.append(float(epm.get("min_dcpa_m", np.nan)))
        per_agent_risk.append(float(epm.get("risk_exposure", np.nan)))
        per_agent_success.append(int(bool(epm.get("success", 0))))
        per_agent_collision.append(int(bool(epm.get("collision", 0))))
        per_agent_near_miss.append(int(bool(epm.get("near_miss", 0))))
        per_agent_min_sep.append(float(epm.get("min_actual_sep_m", np.nan)))
        per_agent_goal_progress.append(float(epm.get("goal_progress", np.nan)))
        
        ct = epm.get("completion_time_s", None)
        if ct is not None and np.isfinite(ct): 
            per_agent_ct.append(float(ct))
        
    own = metrics_by_agent.get("ship_0", {})

    return {
        "episode_steps": step_count, 
        "episode_return_mean": float(np.mean(list(reward_by_agent.values()))),
        "episode_return_ownship": float(reward_by_agent.get("ship_0", np.nan)),
        "collision_any": float(any(per_agent_collision)),
        "success_ownship": ownship_success,
        "success_rate_agents": float(np.mean(per_agent_success)),
        "path_length_m_mean": float(np.nanmean(per_agent_path)),
        "path_length_m_ownship": float(own.get("path_length_m", np.nan)),
        "min_dcpa_m_mean": float(np.nanmean(per_agent_dcpa)),
        "min_dcpa_m_ownship": float(own.get("min_dcpa_m", np.nan)),
        "risk_exposure_mean": float(np.nanmean(per_agent_risk)),
        "risk_exposure_ownship": float(own.get("risk_exposure", np.nan)),
        "min_actual_sep_m_mean": float(np.nanmean(per_agent_min_sep)),
        "min_actual_sep_m_ownship": float(own.get("min_actual_sep_m", np.nan)),
        "near_miss_any": float(any(per_agent_near_miss)),
        "goal_progress_mean": float(np.nanmean(per_agent_goal_progress)),
        "goal_progress_ownship": float(own.get("goal_progress", np.nan)),
        "completion_time_s_mean": float(np.nanmean(per_agent_ct)) if per_agent_ct else float("nan"),
        "completion_time_s_ownship": float(own.get("completion_time_s", np.nan)),
    }

        
def main():
    args = parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(f"corall_baseline_case{args.case}_{timestamp}") / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env_creator = build_env_creator(args)

    per_episode_results = []

    for ep in range(args.episodes):
        ep_seed = args.seed + ep  
        row = run_one_episode_baseline(env_creator, seed=ep_seed, args=args)
        row["episode_index"] = ep
        row["episode_seed"] = ep_seed
        per_episode_results.append(row)

        print(
            f"[eval ep {ep+1}/{args.episodes}] " 
            f"return_mean={row['episode_return_mean']:.3f}, " 
            f"collision_any={row['collision_any']}, "
            f"success_ownship={row['success_ownship']:.0f}, "
            f"success_rate_agents={row['success_rate_agents']:.3f}, "
            f"min_dcpa_ownship={row['min_dcpa_m_ownship']:.3f}m, "
            f"risk_exposure_ownship={row['risk_exposure_ownship']:.3f}"

        )

    summary = {
        "baseline": "CORALL_rule_based",
        "case": args.case,
        "episodes": args.episodes,
        "seed_base": args.seed,
        "episode_return_mean": safe_mean([r["episode_return_mean"] for r in per_episode_results]),
        "episode_return_ownship_mean": safe_mean([r["episode_return_ownship"] for r in per_episode_results]),
        "collision_rate": safe_mean([r["collision_any"] for r in per_episode_results]),
         "success_rate_ownship_mean": safe_mean([r["success_ownship"] for r in per_episode_results]),
        "success_rate_agents_mean": safe_mean([r["success_rate_agents"] for r in per_episode_results]),
        "path_length_m_mean": safe_mean([r["path_length_m_mean"] for r in per_episode_results]),
        "path_length_m_ownship_mean": safe_mean([r["path_length_m_ownship"] for r in per_episode_results]),
        "min_dcpa_m_mean": safe_mean([r["min_dcpa_m_mean"] for r in per_episode_results]),
        "min_dcpa_m_ownship_mean": safe_mean([r["min_dcpa_m_ownship"] for r in per_episode_results]),
        "risk_exposure_mean": safe_mean([r["risk_exposure_mean"] for r in per_episode_results]),
        "risk_exposure_ownship_mean": safe_mean([r["risk_exposure_ownship"] for r in per_episode_results]),
        "min_actual_sep_m_mean": safe_mean([r["min_actual_sep_m_mean"] for r in per_episode_results]),
        "min_actual_sep_m_ownship_mean": safe_mean([r["min_actual_sep_m_ownship"] for r in per_episode_results]),
        "near_miss_rate": safe_mean([r["near_miss_any"] for r in per_episode_results]),
        "goal_progress_mean": safe_mean([r["goal_progress_mean"] for r in per_episode_results]),
        "goal_progress_ownship_mean": safe_mean([r["goal_progress_ownship"] for r in per_episode_results]),
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
        "success_ownship",
        "success_rate_agents",
        "path_length_m_mean",
        "path_length_m_ownship",
        "min_dcpa_m_mean",
        "min_dcpa_m_ownship",
        "risk_exposure_mean",
        "risk_exposure_ownship",
        "min_actual_sep_m_mean",
        "min_actual_sep_m_ownship",
        "near_miss_any",
        "goal_progress_mean",
        "goal_progress_ownship",
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