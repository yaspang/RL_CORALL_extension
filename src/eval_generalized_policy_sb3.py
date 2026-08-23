"""
Evaluate a trained generalized PPO policy on a single Imazu case.

Usage
-----
  python -m src.eval_generalized_policy_sb3 \\
      --checkpoint "GENERALIZED_SB3_YYYYMMDD-HHMMSS/checkpoints/generalized_checkpoint_850000_steps.zip" \\
      --case 6 --episodes 100 --seed 0 --save_histories

  # With LLM intent overlay:
  python -m src.eval_generalized_policy_sb3 \\
      --checkpoint <path>.zip --case 18 --episodes 100 --seed 0 \\
      --llm --llm_provider openai --llm_interval 10 \\
      --llm_env_file "third_party/CORALL/.env" \\
      --save_histories --output_dir results_llmapi_case18

Key arguments
-------------
  --checkpoint STR         Path to .zip checkpoint (required)
  --case INT               Imazu case number 1–22 (required)
  --episodes INT           Evaluation episodes (default: 50)
  --seed INT               Base random seed (default: 0)
  --sim_time FLOAT         Episode horizon in seconds (default: 900.0)
  --desired_cross_x_nmi F  Encounter crossing distance in nmi (default: 1.0)
  --target_speed_mps F     Target vessel speed m/s (default: 10.0)
  --ownship_speed_mps F    Ownship speed m/s (default: None = case native)
  --save_histories         Save per-step NPZ histories for visualization
  --output_dir STR         Output directory (default: policy_eval_generalized_sb3_case<N>_<ts>)
  --llm                    Enable LLM intent generation (Stage 2)
  --llm_rule_only          Log COLREGs rule classifications only (no API)
  --llm_provider STR       LLM provider: openai | claude
  --llm_interval FLOAT     Seconds between LLM queries (default: 30.0)
  --llm_env_file STR       Path to .env file with API keys

Outputs
-------
  <output_dir>/seed_<S>/
  ├── policy_eval_per_episode.csv
  ├── policy_eval_summary.json
  ├── episode_histories/*.npz          (if --save_histories)
  ├── llm_intent_log.csv               (if --llm or --llm_rule_only)
  └── execution_log.csv                (if --llm)
"""

import argparse
from pathlib import Path
from datetime import datetime
import json
import csv

import numpy as np
from stable_baselines3 import PPO

from src.visualizations.episode_overlay_tools import save_episode_history


# Observation dimensions for each case
CASE_OBS_SIZES = {
    1: 8 + 1*6,   # 1 obstacle
    6: 8 + 2*6,   # 2 obstacles
    21: 8 + 3*6,  # 3 obstacles
}
MAX_OBS_SIZE = 29  # v8: 8 (own) + 3 (goal bearing/distance) + 18 (3 obstacles × 6)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate generalized SB3 PPO policy across cases"
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained generalized policy checkpoint (.zip)"
    )
    p.add_argument(
        "--case",
        type=int,
        required=True,
        help="CORALL case for evaluation (1-23; policy trained on all cases)"
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for evaluation"
    )
    p.add_argument(
        "--dt",
        type=float,
        default=0.5,
        help="Simulation timestep (seconds)"
    )
    p.add_argument(
        "--sim_time",
        type=float,
        default=900.0,
        help="Episode length (seconds)"
    )
    p.add_argument(
        "--route_len_nmi",
        type=float,
        default=2.0,
        help="Route length (NMI)"
    )
    p.add_argument(
        "--save_histories",
        action="store_true",
        help="Save episode histories for animation"
    )
    p.add_argument(
        "--save_first_history",
        action="store_true",
        help="Save only first episode history"
    )

    p.add_argument(
        "--desired_cross_x_nmi",
        type=float,
        default=1.0,
        help="Encounter cluster distance along route (must match training if comparing fairly)"
    )
    p.add_argument(
        "--target_speed_mps",
        type=float,
        default=10.0,
        help="Default / fallback target speed used by synchronized-speed generator"
    )
    p.add_argument(
        "--ownship_speed_mps",
        type=float,
        default=None,
        help="Ownship cruising speed used during evaluation (should match training)"
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory for evaluation results"
    )

    # ── Stage-1 LLM intent logging ──────────────────────────────────────────
    p.add_argument(
        "--llm",
        action="store_true",
        default=False,
        help=(
            "Enable LLM intent logging alongside the RL policy (Stage 1: "
            "logging only — the PPO policy is unchanged). "
            "Requires OPENAI_API_KEY or CLAUDE_API_KEY in environment / "
            "third_party/CORALL/.env. "
            "NOTE: LLM API calls add ~1-5 s per call. For large episode "
            "counts use --llm_interval 60 or --episodes 1."
        ),
    )
    p.add_argument(
        "--llm_rule_only",
        action="store_true",
        default=False,
        help=(
            "Log COLREGs rule classifications at each decision interval "
            "without making LLM API calls (fast, no API key required). "
            "Implies --llm behaviour except the LLM text column is omitted."
        ),
    )
    p.add_argument(
        "--llm_provider",
        type=str,
        default=None,
        choices=["openai", "claude"],
        help="LLM provider for --llm mode. Reads LLM_PROVIDER env var if not set.",
    )
    p.add_argument(
        "--llm_interval",
        type=float,
        default=30.0,
        help="Interval between LLM decision queries in simulation seconds (default 30 s).",
    )
    p.add_argument(
        "--llm_env_file",
        type=str,
        default=None,
        help=(
            "Explicit path to the .env file that holds LLM API keys "
            "(e.g. third_party/CORALL/.env). "
            "When omitted the loader searches third_party/CORALL/.env then "
            "the workspace root .env in that order."
        ),
    )

    return p.parse_args()


def init_history(env, seed, args):
    """Initialize history dict for episode tracking."""
    # Get multi-agent env (ImazuCaseEnv.env_multi or direct access)
    multi_env = env.env_multi if hasattr(env, 'env_multi') else env.env.env_multi
    
    # Get final waypoint for ownship
    Xwpt = multi_env.Xwpt_all[0]
    Ywpt = multi_env.Ywpt_all[0]
    final_waypoint_x = float(Xwpt[-1]) if len(Xwpt) > 0 else None
    final_waypoint_y = float(Ywpt[-1]) if len(Ywpt) > 0 else None
    
    return {
        "t": [float(multi_env.t)],
        "X_all": [multi_env.X_all.copy()],
        "pair_risk": [multi_env.pair_risk.copy()],
        "pair_dcpa": [multi_env.pair_dcpa.copy()],
        "pair_dist": [multi_env.pair_dist.copy()],
        "pair_tcpa": [multi_env.pair_tcpa.copy()],
        "agents": list(multi_env.agents),
        "case": int(args.case),
        "seed": int(seed),
        "baseline": "",
        "checkpoint": str(args.checkpoint),
        "final_waypoint_x_nmi": final_waypoint_x,
        "final_waypoint_y_nmi": final_waypoint_y,
    }


def append_history(history, env):
    """Append current step to episode history."""
    # Get multi-agent env (ImazuCaseEnv.env_multi or direct access)
    multi_env = env.env_multi if hasattr(env, 'env_multi') else env.env.env_multi
    
    history["t"].append(float(multi_env.t))
    history["X_all"].append(multi_env.X_all.copy())
    history["pair_risk"].append(multi_env.pair_risk.copy())
    history["pair_dcpa"].append(multi_env.pair_dcpa.copy())
    history["pair_dist"].append(multi_env.pair_dist.copy())
    history["pair_tcpa"].append(multi_env.pair_tcpa.copy())


def run_one_episode(model, env, seed, args, capture_history=False,
                    intent_service=None, intent_arbiter=None, execution_logger=None,
                    episode_idx=0):
    """
    Run a single deterministic evaluation episode.

    Parameters
    ----------
    intent_service : LLMIntentService or None
        # CHANGED: replaces the old Stage-1-only ``llm_logger``. Generates and
        # persists IntentCommand objects (see llm_intent_service.py).
    intent_arbiter : IntentActionArbiter or None
        # CHANGED (new): applies the active intent as a constraint on the RL
        # action before env.step(), with a deterministic emergency override on
        # top (see intent_arbiter.py). The RL policy IS affected when this is set.
    execution_logger : ExecutionLogger or None
        # CHANGED (new): records proposed vs. applied action + intent/arbitration
        # context at every step (see execution_logger.py).
    episode_idx : int
        Zero-based episode index written into log records.

    Returns
    -------
    metrics : dict
    history : dict or None
    """
    obs, info = env.reset(seed=seed)

    # CHANGED: make episode independence explicit — without this, a stale
    # _active_intent / previous_heading_index from episode n could carry into
    # episode n+1 (state persists across resets since these objects are reused).
    if intent_service is not None:
        intent_service.reset_episode()
    if intent_arbiter is not None:
        intent_arbiter.reset_episode()

    history = init_history(env, seed, args) if capture_history else None

    # Cache multi-agent env reference once (object is stable across steps)
    _multi_env = env.env_multi if hasattr(env, 'env_multi') else env.env.env_multi

    episode_return = 0.0
    step = 0
    done = False

    while not done and step < int(args.sim_time / args.dt):
        current_time = float(_multi_env.t)

        # OLD (Stage 1, logging only, ran AFTER env.step — kept here for reference):
        #   action, _ = model.predict(obs, deterministic=True)
        #   obs, reward, terminated, truncated, info = env.step(action)
        #   episode_return += float(reward)
        #   if capture_history:
        #       append_history(history, env)
        #   if llm_logger is not None and llm_logger.should_log(step):
        #       llm_logger.log_step(step=step, time_s=float(_multi_env.t),
        #                            episode_idx=episode_idx, episode_seed=seed,
        #                            multi_env=_multi_env)
        #   done = terminated or truncated
        #   step += 1

        # CHANGED (item 3/4): request/refresh the intent BEFORE predicting the
        # action, so a freshly-issued or still-persisted intent is available to
        # constrain this step's action rather than only being logged after it.
        active_intent = None
        if intent_service is not None:
            intent_service.maybe_request_update(
                step=step, time_s=current_time, episode_idx=episode_idx,
                episode_seed=seed, multi_env=_multi_env,
            )
            active_intent = intent_service.get_active_intent(
                time_s=current_time, multi_env=_multi_env,
            )

        rl_action, _ = model.predict(obs, deterministic=True)

        # CHANGED (item 4/5/6): arbitrate the RL action against the active
        # intent (with a deterministic emergency override on top) before
        # stepping the environment. This creates a real control connection
        # instead of a purely observational one.
        if intent_arbiter is not None:
            applied_action, arbitration = intent_arbiter.apply(
                proposed_action=rl_action, intent=active_intent, multi_env=_multi_env,
            )
        else:
            applied_action, arbitration = rl_action, {"mode": "rl_only", "reason": None}

        obs, reward, terminated, truncated, info = env.step(applied_action)
        episode_return += float(reward)

        if capture_history:
            append_history(history, env)

        # CHANGED (item 4): log proposed vs. applied action + intent/arbitration.
        if execution_logger is not None:
            execution_logger.log(
                step=step, time_s=current_time, episode_idx=episode_idx,
                episode_seed=seed, proposed_action=rl_action,
                applied_action=applied_action, intent=active_intent,
                arbitration=arbitration, multi_env=_multi_env,
            )

        done = terminated or truncated
        step += 1
    
    # Extract ownship metrics (unwrap: ImazuCaseEnv -> SingleAgentOwnshipEnv)
    unwrapped_env = env.env if hasattr(env, 'env') else env
    ownship_metrics = unwrapped_env.get_ownship_metrics()
    
    # Read episode-minimum DCPA from tracked episode_metrics (running min over encounter-filtered
    # steps, same convention as baseline).  The old post-episode pairwise snapshot is unreliable
    # because the encounter is resolved by the time the episode ends.
    ownship_dcpa = float(ownship_metrics.get("min_dcpa_m", np.inf))

    metrics = {
        "episode_return": float(episode_return),
        "episode_steps": int(step),
        "collision_any": int(ownship_metrics.get("collision", 0)),
        "success_ownship": int(ownship_metrics.get("success", 0)),
        "path_length_m_ownship": float(ownship_metrics.get("path_length_m", 0.0)),
        "min_dcpa_m_ownship": float(ownship_dcpa),
        "min_actual_sep_m_ownship": float(ownship_metrics.get("min_actual_sep_m", np.inf)),
        "min_tcpa_s_ownship": float(ownship_metrics.get("min_tcpa_s", np.inf)),
        "risk_exposure_ownship": float(ownship_metrics.get("risk_exposure", 0.0)),
        "completion_time_s_ownship": float(ownship_metrics.get("completion_time_s", np.inf)),
        "goal_progress_ownship": float(ownship_metrics.get("goal_progress", 0.0)),
        "near_miss_ownship": int(ownship_metrics.get("near_miss", 0)),
    }
    
    return metrics, history


def evaluate_policy(args):
    """Main evaluation loop."""
    from src.env_single_agent_sb3 import SingleAgentOwnshipEnv
    # Stage-2 LLM integration modules
    from src.llm_integration.llm_intent_service import LLMIntentService
    from src.llm_integration.intent_arbiter import IntentActionArbiter
    from src.llm_integration.execution_logger import ExecutionLogger
    
    # Load checkpoint
    print(f"\n{'='*70}")
    print("Generalized Policy Evaluation (Stable-Baselines3)")
    print(f"{'='*70}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Case: {args.case}")
    print(f"Episodes: {args.episodes}")
    print(f"{'='*70}\n")
    
    model = PPO.load(args.checkpoint, device="auto")
    
    # Verify model observation space matches expectation
    if model.observation_space is None:
        print("❌ ERROR: Model observation_space is None")
        print(f"   Checkpoint: {args.checkpoint}")
        print("   This typically means the checkpoint is corrupted or incompatible.")
        raise RuntimeError("Cannot load model: observation_space metadata missing")
    
    model_obs_size = model.observation_space.shape[0]
    if model_obs_size != MAX_OBS_SIZE:
        print(f"[WARN] Model expects {model_obs_size}-dim observations, expected {MAX_OBS_SIZE}")
    
    # Create environment using ImazuCaseEnv (ensures 29-dim padding)
    from src.env_imazu_case_sb3 import ImazuCaseEnv
    
    # Create in fixed-case mode (don't randomize case during eval)
    env = ImazuCaseEnv(
        cases_to_train=[args.case],  # Only use target case
        num_seeds=10000,
        dt=args.dt,
        sim_time=args.sim_time,
        n_speed=5,
        route_len_nmi=args.route_len_nmi,
        master_seed=None,
        desired_cross_x_nmi=args.desired_cross_x_nmi,
        target_speed_mps=args.target_speed_mps,
        ownship_speed_mps=args.ownship_speed_mps,
    )
    
    # Output directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path(f"policy_eval_generalized_sb3_case{args.case}_{timestamp}")
    seed_dir = output_dir / f"seed_{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    histories_dir = seed_dir / "episode_histories"
    if args.save_histories or args.save_first_history:
        histories_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage-2 LLM intent generation + arbitration + execution logging ─────
    # Stage-2 LLM: llm_intent_service generates intent, intent_arbiter constrains actions,
    # execution_logger records proposed vs. applied actions (all in src/llm_integration/).
    _use_llm_logging = getattr(args, "llm", False) or getattr(args, "llm_rule_only", False)
    intent_service = None
    intent_arbiter = None
    execution_logger = None
    if _use_llm_logging:
        from src.llm_integration.llm_intent_service import load_env_file
        # Load .env for API keys before constructing MultiLLMCOLREGSInterpreter.
        # Mirrors the original CORALL load_env_file() call in simulation.py.
        if getattr(args, "llm", False):
            found = load_env_file(env_path=getattr(args, "llm_env_file", None))
            if not found:
                print(
                    "[WARN] No .env file found. Set OPENAI_API_KEY / CLAUDE_API_KEY "
                    "as environment variables, or use --llm_env_file to point to "
                    "your .env (e.g. --llm_env_file third_party/CORALL/.env)."
                )
        intent_service = LLMIntentService(
            use_llm=getattr(args, "llm", False),
            provider=getattr(args, "llm_provider", None),
            interval_s=getattr(args, "llm_interval", 30.0),
            dt=args.dt,
            env_file=getattr(args, "llm_env_file", None),
        )
        # CHANGED (new): arbiter constrains the RL action using the intent;
        # execution_logger records proposed vs. applied per step.
        intent_arbiter = IntentActionArbiter(n_heading=getattr(env, "n_heading", 7))
        execution_logger = ExecutionLogger()

    # Run episodes
    per_episode_results = []

    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        capture_hist = bool(args.save_histories or (args.save_first_history and ep == 0))

        # OLD: metrics, history = run_one_episode(
        #          model, env, ep_seed, args, capture_history=capture_hist,
        #          llm_logger=llm_logger, episode_idx=ep,
        #      )
        metrics, history = run_one_episode(
            model, env, ep_seed, args,
            capture_history=capture_hist,
            intent_service=intent_service,
            intent_arbiter=intent_arbiter,
            execution_logger=execution_logger,
            episode_idx=ep,
        )
        metrics["episode_index"] = ep
        metrics["episode_seed"] = ep_seed
        per_episode_results.append(metrics)

        # Save episode history
        if history is not None:
            hist_path = histories_dir / f"case{args.case}_seed{ep_seed}_ep{ep:03d}.npz"
            save_episode_history(history, hist_path)
            print(f"[{ep+1:3d}/{args.episodes}] [OK] history saved -> {hist_path.name}")
        else:
            print(f"[{ep+1:3d}/{args.episodes}] return={metrics['episode_return']:8.2f}, "
                  f"collision={metrics['collision_any']}, success={metrics['success_ownship']:.0f}")

    env.close()

    # Save per-episode CSV
    csv_path = seed_dir / "policy_eval_per_episode.csv"
    if per_episode_results:
        fieldnames = list(per_episode_results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_episode_results)
        print(f"\n[OK] Per-episode results saved to: {csv_path}")

    # OLD: if llm_logger is not None and len(llm_logger) > 0:
    #          llm_log_path = seed_dir / "llm_intent_log.csv"
    #          llm_logger.save(llm_log_path)
    # CHANGED: save both the intent log (generation) and the execution log
    # (proposed vs. applied actions + arbitration outcome) separately.
    if intent_service is not None and len(intent_service) > 0:
        intent_service.save(seed_dir / "llm_intent_log.csv")
    if execution_logger is not None and len(execution_logger) > 0:
        execution_logger.save(seed_dir / "execution_log.csv")

    # Aggregate results
    agg_results = {}
    for key in per_episode_results[0].keys():
        if key in ["episode_index", "episode_seed"]:
            continue
        values = [r[key] for r in per_episode_results if not np.isnan(r[key])]
        if values:
            agg_results[f"{key}_mean"] = float(np.mean(values))
            agg_results[f"{key}_std"] = float(np.std(values))
    
    # Find best-return episode
    best_ep_idx = np.argmax([r["episode_return"] for r in per_episode_results])
    best_ep = per_episode_results[best_ep_idx]
    best_seed = best_ep["episode_seed"]
    agg_results["best_return_episode_idx"] = int(best_ep_idx)
    agg_results["best_return_episode_seed"] = int(best_seed)
    agg_results["best_return_value"] = float(best_ep["episode_return"])

    # Attach eval configuration for reproducibility
    agg_results["eval_config"] = {
        "checkpoint":          str(args.checkpoint),
        "case":                int(args.case),
        "episodes":            int(args.episodes),
        "seed":                int(args.seed),
        "dt":                  float(args.dt),
        "sim_time":            float(args.sim_time),
        "route_len_nmi":       float(args.route_len_nmi),
        "desired_cross_x_nmi": float(args.desired_cross_x_nmi),
        "target_speed_mps":    float(args.target_speed_mps),
        "ownship_speed_mps":   float(args.ownship_speed_mps) if args.ownship_speed_mps is not None else None,
        "llm":                 bool(getattr(args, "llm", False)),
        "llm_rule_only":       bool(getattr(args, "llm_rule_only", False)),
        "llm_provider":        getattr(args, "llm_provider", None),
        "llm_interval":        float(getattr(args, "llm_interval", 30.0)),
        "llm_env_file":        getattr(args, "llm_env_file", None),
    }

    # Save summary
    summary_path = seed_dir / "policy_eval_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(agg_results, f, indent=2)
    print(f"[OK] Summary saved to: {summary_path}\n")
    
    # Print summary
    print(f"\n{'='*70}")
    print("AGGREGATE METRICS")
    print(f"{'='*70}")
    for key, val in sorted(agg_results.items()):
        if isinstance(val, dict):
            continue  # skip nested eval_config dict in console output
        print(f"{key:40s}: {val:10.3f}")
    print(f"{'='*70}\n")
    
    # Highlight best episode
    print(f"{'='*70}")
    print("BEST EPISODE FOR VISUALIZATION")
    print(f"{'='*70}")
    print(f"Episode Index:  {best_ep_idx}")
    print(f"Episode Seed:   {best_seed}")
    print(f"Return:         {best_ep['episode_return']:.2f}")
    print(f"Success:        {bool(best_ep['success_ownship'])}")
    print(f"Collision:      {bool(best_ep['collision_any'])}")
    print(f"Min DCPA (m):   {best_ep['min_dcpa_m_ownship']:.1f}")
    hist_pattern = f"case{args.case}_seed{best_seed}_ep{best_ep_idx:03d}.npz"
    print(f"Desired cross x (nmi): {args.desired_cross_x_nmi}")
    print(f"Target speed (m/s):    {args.target_speed_mps}")
    print(f"Ownship speed (m/s):   {args.ownship_speed_mps}")
    print(f"\nHistory file:   {hist_pattern}")
    print(f"{'='*70}\n")
    
    # Print batch_animate_eval command
    print("To animate this episode with batch_animate_eval:")
    print(f"  python -m src.visualizations.batch_animate_eval --eval_dir \"{seed_dir}\"")
    print()


def main():
    args = parse_args()
    evaluate_policy(args)


if __name__ == "__main__":
    main()
