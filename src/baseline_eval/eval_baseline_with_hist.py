"""
Evaluate CORALL rule-based baseline on a single Imazu case and save episode histories.

Usage
-----
  python -m src.baseline_eval.eval_baseline_with_hist \\
      --case 6 --episodes 100 --seed 0 \\
      --save_histories --output_dir corall_baseline_case6

Key arguments
-------------
  --case (int)                   Imazu case number (required)
  --episodes (int)               Episodes to run (default: 20)
  --seed (int)                   Base random seed (default: 0)
  --sim_time (float)             Episode horizon in seconds (default: 900.0)
  --route_len_nmi (float)        Route length in nmi (default: 2.0)
  --desired_cross_x_nmi (float)  Encounter crossing distance (default: 1.0)
  --target_speed_mps (float)     Target vessel speed m/s (default: 10.0)
  --ownship_speed_mps (float)    Ownship speed m/s (default: None = case native)
  --save_histories               Save per-step NPZ histories for visualization
  --output_dir (str)             Output directory (default: corall_baseline_case<N>_<timestamp>)

Outputs
-------
  corall_baseline_case<N>_<timestamp>/seed_<S>/
  ├── policy_eval_per_episode.csv
  ├── policy_eval_summary.json
  └── episode_histories/*.npz
"""

import argparse
import json
import csv
import time
from pathlib import Path

import numpy as np

from src.visualizations.episode_overlay_tools import save_episode_history

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=int, required=True, help="CORALL Imazu case number for the evaluation")
    p.add_argument("--episodes", type=int, default=20, help="Number of episodes to run for evaluation")
    p.add_argument("--seed", type=int, default=0, help="Base random seed for evaluation")
    p.add_argument("--dt", type=float, default=0.5, help="Time step duration in seconds for the environment")
    p.add_argument("--sim_time", type=float, default=900.0, help="Total simulation time in seconds for each episode")
    p.add_argument("--route_len_nmi", type=float, default=2.0, help="Route length in nautical miles (scaling factor for environment)")
    p.add_argument("--num_workers", type=int, default=0, help="Number of parallel workers to use for evaluation (default: 0 for standalone eval)")
    p.add_argument("--render", action="store_true", help="Whether to render the environment during evaluation")
    p.add_argument("--save_histories", action="store_true", help="Save per-step state histories for all episodes")
    p.add_argument("--no_first_history", action="store_true",
                   help="Skip the default first-episode history save (no NPZ written unless --save_histories)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Override output base directory name (default: corall_baseline_case{N}_{timestamp}). "
                        "Useful for batch runs to keep results under a fixed folder name.")
    p.add_argument("--desired_cross_x_nmi", type=float, default=1.0)
    p.add_argument("--target_speed_mps", type=float, default=10.0)
    p.add_argument("--ownship_speed_mps", type=float, default=None)
    return p.parse_args()


def safe_mean(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if len(vals) > 0 else float('nan')


def build_env_creator(args):
    from src.baseline_eval.env_baseline import CORALLComparisonEnv

    def env_creator(config):
        return CORALLComparisonEnv(
            case_number=config.get("case_number", args.case),
            dt=config.get("dt", args.dt),
            sim_time=config.get("sim_time", args.sim_time),
            route_len_nmi=config.get("route_len_nmi", args.route_len_nmi),
            seed=config.get("seed", args.seed),
            desired_cross_x_nmi=config.get("desired_cross_x_nmi", args.desired_cross_x_nmi),
            target_speed_mps=config.get("target_speed_mps", args.target_speed_mps),
            ownship_speed_mps=config.get("ownship_speed_mps", args.ownship_speed_mps),
        )
        
    return env_creator


def init_history(env, seed, args):
    # Get final waypoint for ownship (baseline has Xwpt/Ywpt as simple lists, not Xwpt_all)
    Xwpt = env.Xwpt if hasattr(env, 'Xwpt') else []
    Ywpt = env.Ywpt if hasattr(env, 'Ywpt') else []
    final_waypoint_x = float(Xwpt[-1]) if len(Xwpt) > 0 else None
    final_waypoint_y = float(Ywpt[-1]) if len(Ywpt) > 0 else None
    
    return {
        "t": [float(env.t)],
        "X_all": [env.X_all.copy()],
        "pair_risk": [env.pair_risk.copy()],
        "pair_dcpa": [env.pair_dcpa.copy()],
        "pair_dist": [env.pair_dist.copy()],
        "pair_tcpa": [env.pair_tcpa.copy()],
        "agents": list(env.agents),
        "case": int(args.case),
        "seed": int(seed),
        "baseline": "CORALL_rule_based",
        "checkpoint": "",
        "final_waypoint_x_nmi": final_waypoint_x,
        "final_waypoint_y_nmi": final_waypoint_y,
    }


def append_history(history, env):
    history["t"].append(float(env.t))
    history["X_all"].append(env.X_all.copy())
    history["pair_risk"].append(env.pair_risk.copy())
    history["pair_dcpa"].append(env.pair_dcpa.copy())
    history["pair_dist"].append(env.pair_dist.copy())
    history["pair_tcpa"].append(env.pair_tcpa.copy())



def run_one_episode_baseline(env_creator, seed, args, capture_history=False):
    env = env_creator({"seed": seed})
    obs, infos = env.reset(seed=seed)

    done = False
    step_count = 0
    max_steps = int(np.ceil(args.sim_time / args.dt)) + 5
    reward_by_agent = {agent: 0.0 for agent in env.agents}
    final_infos = None
    history = init_history(env, seed, args) if capture_history else None

    
    while (not done) and (step_count < max_steps):
        # CORALL baseline: environment ignores action and uses internal CORALL guidance
        # Just pass dummy action for the required agent (Discrete(1) action space)
        actions = {"ship_0": 0}

        obs, rewards, terminations, truncations, infos = env.step(actions)
        step_count += 1

        if capture_history: 
            append_history(history, env)

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
        row = {
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
            "min_tcpa_s_mean": float("nan"),
            "min_tcpa_s_ownship": float("nan"),
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
        return row, history

    per_agent_path, per_agent_dcpa, per_agent_tcpa, per_agent_risk = [], [], [], []
    per_agent_success, per_agent_collision, per_agent_near_miss = [], [], []
    per_agent_min_sep, per_agent_goal_progress, per_agent_ct = [], [], []

    for agent_id, epm in metrics_by_agent.items():
        per_agent_path.append(float(epm.get("path_length_m", np.nan)))
        per_agent_dcpa.append(float(epm.get("min_dcpa_m", np.nan)))
        per_agent_tcpa.append(float(epm.get("min_tcpa_s", np.nan)))
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

    row = {
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
        "min_tcpa_s_mean": float(np.nanmean(per_agent_tcpa)),
        "min_tcpa_s_ownship": float(own.get("min_tcpa_s", np.nan)),
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
    return row, history

        
def main():
    args = parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_name = args.output_dir if args.output_dir else f"corall_baseline_case{args.case}_{timestamp}"
    output_dir = Path(base_name) / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    histories_dir = output_dir / "episode_histories"
    if args.save_histories or not args.no_first_history:
        histories_dir.mkdir(parents=True, exist_ok=True)

    env_creator = build_env_creator(args)
    per_episode_results = []

    for ep in range(args.episodes):
        ep_seed = args.seed + ep  
        capture_history = bool(args.save_histories or (ep == 0 and not args.no_first_history))
        row, history = run_one_episode_baseline(env_creator, seed=ep_seed, args=args, capture_history=capture_history)
        row["episode_index"] = ep
        row["episode_seed"] = ep_seed
        per_episode_results.append(row)

        if history is not None: 
            hist_path = histories_dir / f"baseline_case{args.case}_seed{ep_seed}_ep{ep:03d}.npz"
            save_episode_history(history, hist_path)
            print(f"[saved] history -> {hist_path}")

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
        "min_tcpa_s_mean": safe_mean([r["min_tcpa_s_mean"] for r in per_episode_results]),
        "min_tcpa_s_ownship_mean": safe_mean([r["min_tcpa_s_ownship"] for r in per_episode_results]),
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
        "min_tcpa_s_mean",
        "min_tcpa_s_ownship",
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

if __name__ == "__main__":
    main()