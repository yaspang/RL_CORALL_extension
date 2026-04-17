"""
Evaluate a trained single-agent ownship PPO policy.

Usage:
python -m maritime_rl_pkg.eval_single_agent_ppo \
  --checkpoint "SINGLE_AGENT_ppo_case6_20260415-120000/checkpoint_050" \
  --case 6 \
  --episodes 10 \
  --seed 0 \
  --save_histories
"""

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.gymnasium_env import GymWrapper

from ..episode_overlay_tools import save_episode_history


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate single-agent trained ownship policy")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    p.add_argument("--case", type=int, required=True, help="CORALL case number")
    p.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    p.add_argument("--seed", type=int, default=0, help="Base random seed")
    p.add_argument("--dt", type=float, default=0.5, help="Time step (seconds)")
    p.add_argument("--sim_time", type=float, default=1950.0, help="Episode length (seconds)")
    p.add_argument("--route_len_nmi", type=float, default=2.0, help="Route length (NMI)")
    p.add_argument("--save_histories", action="store_true", help="Save episode histories")
    
    return p.parse_args()


def init_history(env, seed, args):
    """Initialize history tracking for an episode."""
    # Get final waypoint for ownship
    Xwpt = env.env_multi.Xwpt_all[0]
    Ywpt = env.env_multi.Ywpt_all[0]
    final_waypoint_x = float(Xwpt[-1]) if len(Xwpt) > 0 else None
    final_waypoint_y = float(Ywpt[-1]) if len(Ywpt) > 0 else None
    
    return {
        "t": [float(env.env_multi.t)],
        "X_all": [env.env_multi.X_all.copy()],
        "pair_risk": [env.env_multi.pair_risk.copy()],
        "pair_dcpa": [env.env_multi.pair_dcpa.copy()],
        "pair_dist": [env.env_multi.pair_dist.copy()],
        "pair_tcpa": [env.env_multi.pair_tcpa.copy()],
        "agents": ["ship_0"],  # Only ownship
        "case": int(args.case),
        "seed": int(seed),
        "checkpoint": str(args.checkpoint),
        "final_waypoint_x_nmi": final_waypoint_x,
        "final_waypoint_y_nmi": final_waypoint_y,
    }


def append_history(history, env):
    """Append current state to history."""
    history["t"].append(float(env.env_multi.t))
    history["X_all"].append(env.env_multi.X_all.copy())
    history["pair_risk"].append(env.env_multi.pair_risk.copy())
    history["pair_dcpa"].append(env.env_multi.pair_dcpa.copy())
    history["pair_dist"].append(env.env_multi.pair_dist.copy())
    history["pair_tcpa"].append(env.env_multi.pair_tcpa.copy())


def run_one_episode(algo, env_creator, seed, args, capture_history=False):
    """Run a single evaluation episode."""
    env = env_creator({"seed": seed})
    obs, info = env.reset(seed=seed)
    
    # Get policy
    policy = algo.get_policy()
    
    history = init_history(env, seed, args) if capture_history else None
    
    episode_return = 0.0
    step = 0
    
    while step < int(args.sim_time / args.dt):
        # Get action (deterministic, no exploration)
        action, _, _ = policy.compute_single_action(obs, explore=False)
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        
        if capture_history:
            append_history(history, env)
        
        if terminated or truncated:
            break
        
        step += 1
    
    # Collect ownship metrics
    ownship_metrics = env.get_ownship_metrics()
    
    # Get pairwise geometry
    pairwise = env.get_pairwise_geometry()
    
    # Compute evaluation metrics (ownship only)
    state_last = env.get_state()
    ownship_dcpa = pairwise["pair_dcpa"][0, 1:].min() if env.env_multi.n_agents > 1 else np.inf
    
    metrics = {
        "episode_return": float(episode_return),
        "episode_steps": int(step),
        "collision_any": int(ownship_metrics.get("collision", 0)),
        "success_ownship": int(ownship_metrics.get("success", 0)),
        "path_length_m_ownship": float(ownship_metrics.get("path_length_m", 0.0)),
        "min_dcpa_m_ownship": float(ownship_metrics.get("min_dcpa_m", np.inf)),
        "min_tcpa_s_ownship": float(ownship_metrics.get("min_tcpa_s", np.inf)),
        "risk_exposure_ownship": float(ownship_metrics.get("risk_exposure", 0.0)),
        "min_actual_sep_m_ownship": float(ownship_metrics.get("min_actual_sep_m", np.inf)),
        "near_miss_any": int(ownship_metrics.get("near_miss", 0)),
        "goal_progress_ownship": float(ownship_metrics.get("goal_progress", 0.0)),
        "completion_time_s_ownship": float(ownship_metrics.get("completion_time_s", np.nan)),
    }
    
    env.close()
    
    return metrics, history


def evaluate_policy(args):
    """Evaluate trained policy."""
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Setup
    checkpoint_path = Path(args.checkpoint)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(f"policy_eval_single_case{args.case}_{timestamp}") / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    env_name = f"SingleAgentOwnship_eval_case{args.case}_{args.seed}"
    from ray import tune
    tune.register_env(env_name, lambda cfg: GymWrapper(env_creator(cfg)))
    
    # Build algorithm
    config = PPOConfig().environment(env=env_name)
    algo = config.build()
    algo.restore(str(checkpoint_path))
    
    print(f"\n{'='*70}")
    print(f"Single-Agent Policy Evaluation")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Case: {args.case}")
    print(f"Episodes: {args.episodes}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Run episodes
    all_metrics = []
    histories_dir = output_dir / "episode_histories" if args.save_histories else None
    if histories_dir:
        histories_dir.mkdir(parents=True, exist_ok=True)
    
    for ep in range(args.episodes):
        seed = args.seed + ep
        capture_hist = args.save_histories and ep == 0
        
        print(f"[{ep+1:2d}/{args.episodes}] Evaluating episode (seed={seed})...", end=" ", flush=True)
        
        metrics, history = run_one_episode(algo, env_creator, seed, args, capture_history=capture_hist)
        all_metrics.append(metrics)
        
        if capture_hist and history is not None:
            hist_path = histories_dir / f"trained_case{args.case}_seed{args.seed}_ep{ep:03d}.npz"
            save_episode_history(history, str(hist_path))
            print(f"✓ (return={metrics['episode_return']:.2f}, saved history)")
        else:
            print(f"✓ (return={metrics['episode_return']:.2f})")
    
    # Aggregate metrics
    print(f"\n{'='*70}")
    print("Aggregated Metrics (Ownship Only)")
    print(f"{'='*70}")
    
    agg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isnan(m.get(key, np.nan))]
        if values:
            agg_metrics[f"{key}_mean"] = float(np.mean(values))
            agg_metrics[f"{key}_std"] = float(np.std(values))
    
    for key, val in sorted(agg_metrics.items()):
        print(f"  {key:40s}: {val:10.4f}")
    
    # Save CSV
    csv_path = output_dir / "policy_eval_per_episode.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\n✓ Saved per-episode CSV to: {csv_path}")
    
    # Save JSON summary
    summary = {
        "checkpoint": str(checkpoint_path),
        "case": args.case,
        "episodes": args.episodes,
        "seed_base": args.seed,
        "aggregate_metrics": agg_metrics,
        "per_episode_metrics": all_metrics,
    }
    summary_path = output_dir / "policy_eval_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary JSON to: {summary_path}")
    
    print(f"\n{'='*70}\n")
    
    algo.stop()
    ray.shutdown()


def main():
    args = parse_args()
    evaluate_policy(args)


if __name__ == "__main__":
    main()
