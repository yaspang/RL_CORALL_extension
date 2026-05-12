"""
Evaluate a trained policy for a trained shared-policy PPO model using CORALL Imazu case on the MultiShipParallelEnv environment.

Usage example:
python -m maritime_rl_pkg.evaluate_trained_policy ^
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
import torch
import ray

from ..episode_overlay_tools import save_episode_history

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
    p.add_argument("--save_histories", action="store_true", help="Save per-step state histories for all episodes")
    p.add_argument("--save_first_history", action="store_true", help="Save only the first episode history (useful for overlay figures)")
    return p.parse_args()

def safe_mean(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if len(vals) > 0 else float('nan')

def build_algo_and_env(args):
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from maritime_rl_pkg.env_multi_agent_ppo import MultiShipParallelEnv

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
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .debugging(seed=args.seed)
    )

    algo = config.build_algo()
    algo.restore(args.checkpoint)
    return algo, env_creator

def init_history(env, seed, args):
    # Get final waypoint for ownship
    Xwpt = env.Xwpt_all[0]
    Ywpt = env.Ywpt_all[0]
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
        "baseline": "", 
        "checkpoint": str(args.checkpoint),
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


def run_one_episode(algo, env_creator, seed, args, capture_history=False):
    env = env_creator({"seed": seed})
    obs, infos = env.reset(seed=seed)
    
    # Get RLModule for new API stack
    rl_module = algo.get_module("shared_policy")
    
    # Get action space info from environment
    first_agent = env.agents[0]
    action_space = env.action_space(first_agent)
    action_space_shape = None
    if hasattr(action_space, 'nvec'):  # MultiDiscrete
        action_space_shape = list(action_space.nvec)
    elif hasattr(action_space, 'n'):  # Discrete
        action_space_shape = [action_space.n]
    elif hasattr(action_space, 'shape'):  # Box (continuous)
        action_space_shape = list(action_space.shape)

    done = False
    step_count = 0
    max_steps = int(np.ceil(args.sim_time / args.dt)) + 5
    reward_by_agent = {agent: 0.0 for agent in env.agents}
    final_infos = None
    history = init_history(env, seed, args) if capture_history else None

    while (not done) and (step_count < max_steps):
        actions = {}

        for agent_id, agent_obs in obs.items():
            # Convert observation to tensor batch format (add batch dimension)
            obs_tensor = torch.from_numpy(np.array([agent_obs])).float()
            
            # Inference with the new API
            with torch.no_grad():
                output = rl_module.forward_inference(batch={"obs": obs_tensor})
                
                # Extract action from RLModule output
                if isinstance(output, dict) and "action_dist_inputs" in output:
                    logits = output["action_dist_inputs"][0].cpu().numpy()  # shape [num_logits]
                    
                    # For MultiDiscrete action space
                    if action_space_shape and len(action_space_shape) > 1:
                        action = []
                        offset = 0
                        for num_categories in action_space_shape:
                            component_logits = logits[offset:offset + num_categories]
                            action.append(int(np.argmax(component_logits)))
                            offset += num_categories
                        action = np.array(action, dtype=np.int64)
                    else:
                        # Single discrete action
                        action = np.array([int(np.argmax(logits))], dtype=np.int64)
                else:
                    raise KeyError(f"Cannot extract action from output keys: {list(output.keys())}")
            
            actions[agent_id] = action
        
        obs, rewards, terminations, truncations, infos = env.step(actions)
        step_count += 1

        if capture_history: 
            append_history(history, env)

        for agent_id, reward in rewards.items():
            reward_by_agent[agent_id] += float(reward)
        
        final_infos = infos
        done = all(terminations.values()) or any(truncations.values())

        if infos.get("ship_0", {}).get("success", False):
            done = True  # end episode early if ownship reaches goal successfully   

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
    histories_dir = output_dir / "episode_histories"
    histories_dir.mkdir(parents=True, exist_ok=True)
    algo, env_creator = build_algo_and_env(args)
    per_episode_results = []

    for ep in range(args.episodes):
        ep_seed = args.seed + ep  
        capture_history = bool(args.save_histories or (args.save_first_history and ep == 0))
        row, history = run_one_episode(algo, env_creator, seed=ep_seed, args=args, capture_history=capture_history)
        row["episode_index"] = ep
        row["episode_seed"] = ep_seed
        per_episode_results.append(row)

        if history is not None: 
            hist_path = histories_dir / f"trained_case{args.case}_seed{ep_seed}_ep{ep:03d}.npz"
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
        "checkpoint": args.checkpoint,
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
    csv_path = output_dir / "policy_eval_per_episode_VIS.csv"
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
    summary_path = output_dir / "policy_eval_summary_VIS.json"
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