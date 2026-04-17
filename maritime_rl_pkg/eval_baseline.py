"""
Evaluates rule-based baseline (CORALL reactive avoidance + waypoint planning) in the same environment and metrics as the trained RL policy evaluation for direct comparison.
against a trained RL policy in the same environment and using the same evaluation metrics.
"""

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
    p.add_argument("--dt", type=float, default=0.5, help="Time step duration in seconds for the environment")
    p.add_argument("--sim_time", type=float, default=1950.0, help="Total simulation time in seconds for each episode")
    p.add_argument("--route_len_nmi", type=float, default=2.0, help="Route length in nautical miles (scaling factor for environment)")
    p.add_argument("--num_workers", type=int, default=0, help="Number of parallel workers to use for evaluation (default: 0 for standalone eval)")
    p.add_argument("--render", action="store_true", help="Whether to render the environment during evaluation")

    return p.parse_args()

def safe_mean(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if len(vals) > 0 else float('nan')

def build_env_creator(args):
    from maritime_rl_pkg.env_baseline import CORALLComparisonEnv

    def env_creator(config):
        return CORALLComparisonEnv(
            case_number=config.get("case_number", args.case),
            dt=config.get("dt", args.dt),
            sim_time=config.get("sim_time", args.sim_time),
            route_len_nmi=config.get("route_len_nmi", args.route_len_nmi),
            render_mode="human" if args.render else None, 
            seed=config.get("seed", args.seed),
        )
    
    return env_creator

def run_one_episode_baseline(env_creator, seed, args):
    """Run one evaluation episode with CORALL baseline (reactive avoidance + waypoint planning).
    
    Actions are ignored; env computes internal CORALL guidance automatically.
    
    Returns: Dict with ownship-only metrics for direct comparison
    """
    env = env_creator({"seed": seed})
    obs, infos = env.reset(seed=seed)

    done = False
    step_count = 0
    max_steps = int(np.ceil(args.sim_time / args.dt)) + 5
    final_infos = None

    while (not done) and (step_count < max_steps):
        # Action is ignored by CORALLComparisonEnv (env uses internal CORALL guidance)
        actions = {"ship_0": np.array([0.0])}  # dummy action
        obs, rewards, terminations, truncations, infos = env.step(actions)
        step_count += 1
        final_infos = infos
        done = all(terminations.values()) or any(truncations.values())
    
    # Extract episode metrics (ownship only - ship_0)
    metrics_by_agent = {}

    if final_infos is not None:
        for agent_id, info in final_infos.items():
            epm = info.get("episode_metrics", None)
            if isinstance(epm, dict):
                metrics_by_agent[agent_id] = epm

    own = metrics_by_agent.get("ship_0", {})
    ownship_success = float(bool(own.get("success", 0)))
    
    env.close() if hasattr(env, "close") else None

    # Fallbacks if episode_metrics not present
    if not metrics_by_agent:
        return{
            "episode_steps": step_count, 
            "collision_any": float("nan"), 
            "success_ownship": ownship_success,
            "path_length_m_ownship": float("nan"),
            "min_dcpa_m_ownship": float("nan"),
            "min_tcpa_s_ownship": float("nan"),
            "risk_exposure_ownship": float("nan"),
            "min_actual_sep_m_ownship": float("nan"),
            "near_miss_any": float("nan"),
            "goal_progress_ownship": float("nan"),
            "completion_time_s_ownship": float("nan"),
        }
    
    # Debug logging
    if not metrics_by_agent:
        print(f"[WARNING] No episode_metrics found for seed={seed}, steps={step_count}")

    # Extract ownship (ship_0) metrics ONLY
    own = metrics_by_agent.get("ship_0", {})
    collision_any = float(bool(own.get("collision", 0)))
    near_miss_any = float(bool(own.get("near_miss", 0)))

    return {
        "episode_steps": step_count, 
        "collision_any": collision_any,
        "success_ownship": ownship_success,
        "path_length_m_ownship": float(own.get("path_length_m", np.nan)),
        "min_dcpa_m_ownship": float(own.get("min_dcpa_m", np.nan)),
        "min_tcpa_s_ownship": float(own.get("min_tcpa_s", np.nan)),
        "risk_exposure_ownship": float(own.get("risk_exposure", np.nan)),
        "min_actual_sep_m_ownship": float(own.get("min_actual_sep_m", np.nan)),
        "near_miss_any": near_miss_any,
        "goal_progress_ownship": float(own.get("goal_progress", np.nan)),
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
            f"steps={row['episode_steps']}, " 
            f"path_length={row['path_length_m_ownship']:.0f}m, "
            f"collision={row['collision_any']}, "
            f"success={row['success_ownship']:.0f}, "
            f"min_dcpa={row['min_dcpa_m_ownship']:.1f}m, "
            f"min_tcpa={row['min_tcpa_s_ownship']:.1f}s, "
            f"risk_exposure={row['risk_exposure_ownship']:.3f}"
        )

    summary = {
        "baseline_type": "CORALL_reactive_avoidance_with_waypoint_planning",
        "metrics_scope": "ownship_only (ship_0)",
        "case": args.case,
        "episodes": args.episodes,
        "seed_base": args.seed,
        "collision_rate": safe_mean([r["collision_any"] for r in per_episode_results]),
        "success_rate": safe_mean([r["success_ownship"] for r in per_episode_results]),
        "path_length_m_mean": safe_mean([r["path_length_m_ownship"] for r in per_episode_results]),
        "path_length_m_total": float(np.nansum([r["path_length_m_ownship"] for r in per_episode_results])),
        "min_dcpa_m_mean": safe_mean([r["min_dcpa_m_ownship"] for r in per_episode_results]),
        "min_tcpa_s_mean": safe_mean([r["min_tcpa_s_ownship"] for r in per_episode_results]),
        "risk_exposure_mean": safe_mean([r["risk_exposure_ownship"] for r in per_episode_results]),
        "min_actual_sep_m_mean": safe_mean([r["min_actual_sep_m_ownship"] for r in per_episode_results]),
        "near_miss_rate": safe_mean([r["near_miss_any"] for r in per_episode_results]),
        "goal_progress_mean": safe_mean([r["goal_progress_ownship"] for r in per_episode_results]),
        "completion_time_s_mean": safe_mean([r["completion_time_s_ownship"] for r in per_episode_results]),
    }
    
    # Save per-episode CSV
    csv_path = output_dir / "corall_baseline_eval_per_episode.csv"
    fieldnames = [
        "episode_index",
        "episode_seed",
        "episode_steps",
        "collision_any",
        "success_ownship",
        "path_length_m_ownship",
        "min_dcpa_m_ownship",
        "min_tcpa_s_ownship",
        "risk_exposure_ownship",
        "min_actual_sep_m_ownship",
        "near_miss_any",
        "goal_progress_ownship",
        "completion_time_s_ownship",
    ]

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_episode_results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # Save summary JSON
    summary_path = output_dir / "corall_baseline_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== CORALL Baseline Evaluation Summary ===")
    print("[NOTE] All metrics below are for OWNSHIP (ship_0) ONLY for direct comparison\n")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\nSaved per-episode CSV to: {csv_path}")
    print(f"Saved summary JSON to: {summary_path}")


if __name__ == "__main__":
    main()