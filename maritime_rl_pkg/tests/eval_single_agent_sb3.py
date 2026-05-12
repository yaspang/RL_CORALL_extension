"""
Evaluate a trained single-agent ownship PPO policy (Stable-Baselines3).

OVERVIEW:
=========
This script loads a trained policy checkpoint and runs it through multiple evaluation episodes
to collect metrics on collision avoidance performance. Metrics are aggregated to get mean/std
across episodes.

EVALUATION METRICS (OWNSHIP ONLY):
==================================
- episode_return: cumulative reward per episode (-10 for collision, +goal progress, etc.)
- goal_progress_ownship: % of waypoint distance traveled (0-100%)
- success_ownship: binary (1 if reached waypoint without collision, 0 otherwise)
- min_dcpa_m_ownship: closest distance of closest point of approach to all obstacles (meters)
- min_tcpa_s_ownship: time to closest point of closest approach (seconds)
- collision_any: binary (1 if any collision detected, 0 otherwise)
- near_miss_any: binary (1 if any near-miss event detected)
- path_length_m_ownship: total distance traveled by ownship (meters)
- risk_exposure_ownship: cumulative risk metric across episode
- completion_time_s_ownship: time to reach waypoint (seconds)

USAGE:
======
Evaluate a single case:
  python -m maritime_rl_pkg.eval_single_agent_sb3 \\
    --checkpoint "SINGLE_AGENT_SB3_case6_20260415-120000/best_checkpoint.zip" \\
    --case 6 \\
    --episodes 10 \\
    --seed 0 \\
    --save_histories

Evaluate all three cases:
  python -m maritime_rl_pkg.eval_single_agent_sb3 \\
    --checkpoint "SINGLE_AGENT_SB3_case1_TIMESTAMP/best_checkpoint.zip" \\
    --case 1 --episodes 10 --seed 0
  
  python -m maritime_rl_pkg.eval_single_agent_sb3 \\
    --checkpoint "SINGLE_AGENT_SB3_case6_TIMESTAMP/best_checkpoint.zip" \\
    --case 6 --episodes 10 --seed 0
  
  python -m maritime_rl_pkg.eval_single_agent_sb3 \\
    --checkpoint "SINGLE_AGENT_SB3_case21_TIMESTAMP/best_checkpoint.zip" \\
    --case 21 --episodes 10 --seed 0

OUTPUT STRUCTURE:
=================
policy_eval_single_sb3_case{X}_{timestamp}/
└── seed_{base_seed}/
    ├── policy_eval_per_episode.csv      (per-episode metrics, one row per ep)
    ├── policy_eval_summary.json         (aggregated mean/std across episodes)
    └── episode_histories/               (optional, with --save_histories)
        └── trained_case{X}_seed{S}_ep{N}.npz

CSV columns: episode_return, collision_any, success_ownship, goal_progress_ownship, 
             min_dcpa_m_ownship, min_tcpa_s_ownship, risk_exposure_ownship, etc.

JSON summary: checkpoint path, case number, aggregate_metrics (mean/std for all columns),
              per_episode_metrics (list of all metric dicts)

INTERPRETATION:
===============
Compare across cases 1, 6, 21:
  - If returns are similar: policy generalizes across difficulty levels
  - If case 1 > case 6 > case 21: harder cases drive more negative rewards (expected)
  - If collision rate increases: difficult cases overwhelm learned behavior
  - If min_dcpa similar: good separation margins maintained consistently

Baseline comparison:
  - Random policy: ~-20 return, high collision rate, 0% goal progress
  - Heuristic baseline (turn toward waypoint): -10 to -5 return, 5-10% collisions
  - Trained policy (expected): -5 to 0 return, <5% collisions, >90% goal progress
"""

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
from stable_baselines3 import PPO

from ..episode_overlay_tools import save_episode_history

# Observation padding for multi-case compatibility
CASE_OBS_SIZES = {
    1: 12,   # 7 (ownship) + 1*5 (1 obstacle)
    6: 17,   # 7 (ownship) + 2*5 (2 obstacles)
    21: 22,  # 7 (ownship) + 3*5 (3 obstacles)
}
MAX_OBS_SIZE = 22


def pad_observation(obs: np.ndarray, case: int) -> np.ndarray:
    """Pad observation to MAX_OBS_SIZE for consistency across cases.
    
    When evaluating a generalized policy trained with RandomCaseEnv,
    observations from different cases have different sizes:
    - Case 1: 12 dims → pad to 22
    - Case 6: 17 dims → pad to 22
    - Case 21: 22 dims → no padding
    
    Zero-fills unused obstacle slots.
    """
    obs_size = len(obs) if hasattr(obs, '__len__') else 1
    
    if obs_size == MAX_OBS_SIZE:
        return np.asarray(obs, dtype=np.float32)
    
    padded = np.zeros(MAX_OBS_SIZE, dtype=np.float32)
    padded[:obs_size] = obs
    return padded


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate single-agent trained ownship policy (SB3)")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained model .zip file")
    p.add_argument("--case", type=int, required=True, help="CORALL case number")
    p.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    p.add_argument("--seed", type=int, default=0, help="Base random seed")
    p.add_argument("--dt", type=float, default=0.5, help="Time step (seconds)")
    p.add_argument("--sim_time", type=float, default=1950.0, help="Episode length (seconds)")
    p.add_argument("--route_len_nmi", type=float, default=2.0, help="Route length (NMI)")
    p.add_argument("--save_histories", action="store_true", help="Save best episode history (highest return, no collision)")
    p.add_argument("--save_all_histories", action="store_true", help="Save all episode histories")
    
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


def run_one_episode(model, env, seed, args, capture_history=False):
    """
    Run a single evaluation episode and collect metrics.
    
    EVALUATION PROCESS:
    ===================
    1. Reset environment with given seed (deterministic for reproducibility)
    2. Loop over max_steps or until episode terminates:
       - Get deterministic action from trained policy (no exploration)
       - Step environment
       - Accumulate rewards
       - If capture_history=True, record full trajectory for visualization
    3. Extract ownship-only metrics from environment
    
    METRIC COLLECTION (OWNSHIP ONLY):
    =================================
    - episode_return: sum of all rewards (includes collision penalty & goal progress)
    - goal_progress_ownship: normalized distance traveled toward waypoint (0-100%)
    - success_ownship: 1 if reached waypoint without collision, 0 otherwise
    - min_dcpa_m_ownship: closest distance of closest point of approach (meters)
      * DCPA = minimum future distance if both continue current heading/speed
      * Lower DCPA = tighter pass, higher collision risk
    - min_tcpa_s_ownship: time until closest point of approach (seconds)
      * How long until vessels are closest (if no avoidance action taken)
    - collision_any: 1 if any separation < vessel length at any time, 0 otherwise
    - path_length_m_ownship: total distance traveled (accounting for heading changes)
    - risk_exposure_ownship: cumulative CPA-based risk metric over episode
    
    Note: All metrics extracted from ownship only (agents 1...K are obstacles, not evaluated)
    """
    obs, info = env.reset(seed=seed)
    
    # Only apply padding if model was trained with padded observations (generalized policy)
    # Detect by checking if model's observation space is 22-dim (generalized = trained across all cases)
    try:
        model_obs_size = model.observation_space.shape[0] if (hasattr(model.observation_space, 'shape') and model.observation_space.shape is not None) else None
    except (AttributeError, TypeError, IndexError):
        model_obs_size = None
    should_pad = (model_obs_size == MAX_OBS_SIZE)  # 22-dim = generalized policy
    
    if should_pad:
        obs = pad_observation(obs, args.case)
    
    history = init_history(env, seed, args) if capture_history else None
    
    episode_return = 0.0
    step = 0
    done = False
    
    while not done and step < int(args.sim_time / args.dt):
        # Get action (deterministic, no exploration noise)
        obs_for_predict = pad_observation(obs, args.case) if should_pad else obs
        action, _ = model.predict(obs_for_predict, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        if should_pad:
            obs = pad_observation(obs, args.case)
        episode_return += float(reward)
        
        if capture_history:
            append_history(history, env)
        
        done = terminated or truncated
        step += 1
    
    # Collect ownship metrics
    ownship_metrics = env.get_ownship_metrics()
    
    # Get pairwise geometry
    pairwise = env.get_pairwise_geometry()
    
    # Compute evaluation metrics (ownship only)
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
    
    return metrics, history


def evaluate_policy(args):
    """Evaluate trained policy."""
    
    from maritime_rl_pkg.env_single_agent_sb3 import SingleAgentOwnshipEnv
    
    # Setup
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(f"policy_eval_single_sb3_case{args.case}_{timestamp}") / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = PPO.load(str(checkpoint_path))
    
    # Create environment
    env = SingleAgentOwnshipEnv(
        case_number=args.case,
        dt=args.dt,
        sim_time=args.sim_time,
        route_len_nmi=args.route_len_nmi,
        seed=args.seed,
    )
    
    # Detect policy type (generalized vs case-specific)
    try:
        policy_obs_size = model.observation_space.shape[0] if (hasattr(model.observation_space, 'shape') and model.observation_space.shape is not None) else None
    except (AttributeError, TypeError, IndexError):
        policy_obs_size = None
    policy_type = "GENERALIZED" if policy_obs_size == MAX_OBS_SIZE else f"CASE-SPECIFIC ({policy_obs_size}-dim)"
    
    print(f"\n{'='*70}")
    print(f"Single-Agent Policy Evaluation (Stable-Baselines3)")
    print(f"{'='*70}")
    print(f"Policy Type: {policy_type}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Case: {args.case}")
    print(f"Episodes: {args.episodes}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Run episodes
    all_metrics = []
    histories_dir = output_dir / "episode_histories" if (args.save_histories or args.save_all_histories) else None
    if histories_dir:
        histories_dir.mkdir(parents=True, exist_ok=True)
    
    best_episode = None
    best_score = float('-inf')  # Higher return is better
    
    for ep in range(args.episodes):
        seed = args.seed + ep
        
        # Save history if --save_all_histories, or capture for later processing
        capture_hist = args.save_all_histories or (args.save_histories and ep < args.episodes)
        
        print(f"[{ep+1:2d}/{args.episodes}] Evaluating episode (seed={seed})...", end=" ", flush=True)
        
        metrics, history = run_one_episode(model, env, seed, args, capture_history=capture_hist)
        all_metrics.append(metrics)
        
        # Track best episode (highest return + no collision preferred)
        # Scoring: prioritize no collision, then higher return
        collision_penalty = 100 if metrics["collision_any"] else 0
        episode_score = metrics["episode_return"] - collision_penalty
        if episode_score > best_score:
            best_score = episode_score
            best_episode = (ep, history, metrics)
        
        if args.save_all_histories and history is not None and histories_dir is not None:
            hist_path = histories_dir / f"trained_case{args.case}_seed{args.seed}_ep{ep:03d}.npz"
            save_episode_history(history, str(hist_path))
            print(f"✓ (return={metrics['episode_return']:.2f}, collision={metrics['collision_any']})")
        else:
            print(f"✓ (return={metrics['episode_return']:.2f}, collision={metrics['collision_any']})")
    
    # Aggregate metrics
    print(f"\n{'='*70}")
    print("Aggregated Metrics (Ownship Only)")
    print(f"{'='*70}")
    
    # Save best episode history if requested
    if args.save_histories and best_episode is not None and histories_dir is not None:
        best_ep_num, best_history, best_metrics = best_episode
        if best_history is not None:
            hist_path = histories_dir / f"trained_case{args.case}_seed{args.seed}_BEST_ep{best_ep_num:03d}_return{best_metrics['episode_return']:.1f}.npz"
            save_episode_history(best_history, str(hist_path))
            print(f"✓ Saved best episode history: ep {best_ep_num} (return={best_metrics['episode_return']:.2f}, collision={best_metrics['collision_any']})")
            print(f"  Path: {hist_path}\n")
    
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
    
    env.close()


def main():
    args = parse_args()
    evaluate_policy(args)


if __name__ == "__main__":
    main()
