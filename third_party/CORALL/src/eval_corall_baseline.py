"""
Evaluate CORALL rule-based baseline on Imazu cases and export episode metrics.

Goal:
- run CORALL baseline planner (LLM off / rule-based)
- compute the same metrics as PPO evaluation:
    * path length
    * collision flag
    * completion time
    * minimum DCPA
    * risk exposure integral
- save per-episode CSV + aggregate JSON

Usage:
python -m maritime_rl_pkg.maritime_rl.evaluate_corall_baseline ^
    --case 2 ^
    --episodes 20 ^
    --dt 0.2 ^
    --sim_time 300 ^
    --seed 0

Structure: 
- read Imazu case
- set up baseline route based on initialized state from case
- CORALL-style control loop:
    for each episode: (full run on one case)
        for each step:
            for each agent:
                select heading reference using CORALL waypoint planner + reactive avoidance
                keep speed at nominal (e.g. initialized speed from case)
            propagate dynamics with CORALL vessel model + controller
            compute pairwise safety metrics (DCPA, risk)
            accumulate episode metrics and check termination (collision or success)
- at each episode end compute evaluation metrics and save to CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


NMI = 1852.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=int, default=2, help="CORALL Imazu case number")
    p.add_argument("--episodes", type=int, default=20, help="Number of baseline eval episodes")
    p.add_argument("--dt", type=float, default=0.2, help="Simulation time step [s]")
    p.add_argument("--sim_time", type=float, default=300.0, help="Max sim time [s]")
    p.add_argument("--loa_m", type=float, default=175.0, help="Ship length overall [m]")
    p.add_argument("--route_len_nmi", type=float, default=40.0, help="Straight route length [nmi]")
    p.add_argument("--seed", type=int, default=0, help="Base seed")
    p.add_argument("--output_dir", type=str, default=None, help="Optional output dir")
    return p.parse_args()


def safe_mean(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def build_initial_states_from_case(case_number: int):
    """
    Mirror your PPO env initialization so baseline comparison is as fair as possible.
    """
    from utils.imazu_cases import get_obstacle_data

    Xob, Yob, Vob, psiob = get_obstacle_data(case_number)

    n_obstacles = len(Xob)
    n_agents = 1 + n_obstacles

    X_all = np.zeros((n_agents, 6), dtype=float)
    u_des_all = np.zeros(n_agents, dtype=float)

    # Ownship at origin, heading 0, speed matched to first traffic speed if available
    if len(Vob) > 0:
        u0 = float(Vob[0])
    else:
        u0 = 18.52 

    X_all[0, :] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, u0], dtype=float)
    u_des_all[0] = u0

    for j in range(n_obstacles):
        X_all[j + 1, :] = np.array([Xob[j], Yob[j], psiob[j], 0.0, 0.0, Vob[j]], dtype=float)
        u_des_all[j + 1] = float(Vob[j])

    return X_all, u_des_all


def build_waypoints(X_all: np.ndarray, route_len_nmi: float):
    """
    Same straight-line route construction used in MARL PPO env.
    """
    Xwpt_all: List[List[float]] = []
    Ywpt_all: List[List[float]] = []

    for k in range(X_all.shape[0]):
        x_m, y_m, psi = float(X_all[k, 0]), float(X_all[k, 1]), float(X_all[k, 2])
        x0_nmi = x_m / NMI
        y0_nmi = y_m / NMI

        x1_nmi = x0_nmi + route_len_nmi * np.cos(psi)
        y1_nmi = y0_nmi + route_len_nmi * np.sin(psi)

        Xwpt_all.append([x0_nmi, x1_nmi])
        Ywpt_all.append([y0_nmi, y1_nmi])

    return Xwpt_all, Ywpt_all


def pairwise_cpa_risk(X_all: np.ndarray, prev_X_all: np.ndarray, a: int, b: int, dt: float):
    from risk_assessment.cpa_calculations import cpa_calculations
    from risk_assessment.risk_calculations import risk_calculations

    xa, ya = float(X_all[a, 0]), float(X_all[a, 1])
    xa_prev, ya_prev = float(prev_X_all[a, 0]), float(prev_X_all[a, 1])

    xb, yb = float(X_all[b, 0]), float(X_all[b, 1])
    xb_prev, yb_prev = float(prev_X_all[b, 0]), float(prev_X_all[b, 1])

    dcpa, tcpa, vrel, alpha, psi_vrel = cpa_calculations(
        xa, ya, xa_prev, ya_prev,
        xb, yb, xb_prev, yb_prev,
        dt,
    )
    dist = float(np.hypot(xa - xb, ya - yb))
    risk = float(risk_calculations(dcpa, tcpa, dist, vrel))
    return float(dcpa), float(tcpa), dist, risk


def choose_baseline_heading(
    k: int,
    X_all: np.ndarray,
    prev_X_all: np.ndarray,
    Xwpt_all: List[List[float]],
    Ywpt_all: List[List[float]],
    i_wpt_all: np.ndarray,
    dt: float,
) -> float:
    """
    Baseline heading command.
    - follow CORALL waypoint planner / reactive avoidance decisions to select heading reference

    """
    from navigation.planning import waypoint_selection, planning
    from navigation.reactive_avoidance import reactive_avoidance

    x_m, y_m, psi_k = float(X_all[k, 0]), float(X_all[k, 1]), float(X_all[k, 2])
    x_nmi, y_nmi = x_m / NMI, y_m / NMI

    Xwpt_k = Xwpt_all[k]
    Ywpt_k = Ywpt_all[k]
    i_wpt_k = int(i_wpt_all[k])

    # Path planning and collision avoidance 
    i_wpt_k = waypoint_selection(Xwpt_k, Ywpt_k, x_nmi, y_nmi, i_wpt_k)
    i_wpt_all[k] = i_wpt_k

    # same waypoint-selection logic as CORALL
    psi_result = planning(Xwpt_k, Ywpt_k, x_nmi, y_nmi, i_wpt_k)
    psi_wp = float(psi_result) if psi_result is not None else 0.0

    # CORALL reactive avoidance (obstacle positions in nmi)
    x_ob_nmi = []
    y_ob_nmi = []

    for j in range(X_all.shape[0]):
        if j == k:
            continue
        x_ob_nmi.append(float(X_all[j, 0]) / NMI)
        y_ob_nmi.append(float(X_all[j, 1]) / NMI)

    psi_oa, w_B, w_R, distance_ob, bearing_ob = reactive_avoidance(x_ob_nmi, y_ob_nmi, x_nmi, y_nmi, psi_k, 0.0)

    # baseline CORALL behavior without LLM modification
    # Kdir stays +1, so only modify heading reference with obstacle avoidance if w_B > 0
    psi_ref = float(psi_wp + psi_oa)

    return psi_ref


def run_corall_baseline_episode(
    case_number: int,
    dt: float = 0.2,
    sim_time: float = 300.0,
    loa_m: float = 175.0,
    route_len_nmi: float = 40.0,
    seed: int = 0,
):
    """
    Run one deterministic CORALL-style rule-based baseline episode.
    """
    from dynamics.controller import controller
    from dynamics.actuator_modeling import actuator_modeling
    from dynamics.vessel_dynamics import vessel_dynamics
    from core.integration import integration

    rng = np.random.default_rng(seed)

    X_all, u_des_all = build_initial_states_from_case(case_number)
    prev_X_all = X_all.copy()

    n_agents = X_all.shape[0]
    agent_ids = [f"ship_{i}" for i in range(n_agents)]

    Xwpt_all, Ywpt_all = build_waypoints(X_all, route_len_nmi=route_len_nmi)
    i_wpt_all = np.array(
        [1 if len(Xwpt_all[k]) > 1 else 0 for k in range(n_agents)], dtype=int  
    )
    ui_psi1_all = np.zeros(n_agents, dtype=float)

    t = 0.0
    step_count = 0

    goal_radius = 2.0 * loa_m
    dcpa_safe = 4.0 * loa_m

    episode_metrics = {
        agent: {
            "path_length_m": 0.0,
            "min_dcpa_m": float("inf"),
            "risk_exposure": 0.0,
            "collision": 0,
            "success": 0,
            "completion_time_s": None,
        }
        for agent in agent_ids
    }

    while t < sim_time:
        prev_X_all = X_all.copy()

        # 1) choose baseline heading + keep nominal speed
        psi_ref = np.zeros(n_agents, dtype=float)
        u_cmd = np.zeros(n_agents, dtype=float)

        for k, agent in enumerate(agent_ids):
            psi_ref[k] = choose_baseline_heading(
                k=k,
                X_all=X_all,
                prev_X_all=prev_X_all,
                Xwpt_all=Xwpt_all,
                Ywpt_all=Ywpt_all,
                i_wpt_all=i_wpt_all,
                dt=dt,
            )
            u_cmd[k] = float(u_des_all[k])

        # 2) propagate each vessel using CORALL dynamics stack
        for k in range(n_agents):
            x_m, y_m, psi_k, r_k, b_k, u_k = X_all[k, :]
            ui_psi1_k = float(ui_psi1_all[k])

            tau_c, v_c, ui_psi1_k = controller(
                psi_ref[k], psi_k, r_k, u_cmd[k], b_k, ui_psi1_k, dt
            )
            ui_psi1_all[k] = ui_psi1_k

            tau_ac = actuator_modeling(tau_c, sat_amp_s=20)
            X_dot = vessel_dynamics(X_all[k, :], [tau_ac, v_c])
            X_all[k, :] = integration(X_all[k, :], X_dot, dt)

        # 3) compute pairwise safety values
        pair_dcpa = np.full((n_agents, n_agents), np.inf, dtype=float)
        pair_risk = np.zeros((n_agents, n_agents), dtype=float)
        pair_dist = np.full((n_agents, n_agents), np.inf, dtype=float)

        for a in range(n_agents):
            for b in range(n_agents):
                if a == b:
                    continue
                dcpa, tcpa, dist, risk = pairwise_cpa_risk(X_all, prev_X_all, a, b, dt)
                pair_dcpa[a, b] = dcpa
                pair_risk[a, b] = risk
                pair_dist[a, b] = dist

        # 4) accumulate metrics and termination
        any_collision = False
        any_success = False

        for k, agent in enumerate(agent_ids):
            x_k, y_k = float(X_all[k, 0]), float(X_all[k, 1])

            dx_step = float(X_all[k, 0] - prev_X_all[k, 0])
            dy_step = float(X_all[k, 1] - prev_X_all[k, 1])
            step_dist = float(np.hypot(dx_step, dy_step))
            episode_metrics[agent]["path_length_m"] += step_dist

            risks = pair_risk[k].copy()
            risks[k] = 0.0
            max_risk = float(np.max(risks))
            episode_metrics[agent]["risk_exposure"] += max_risk * dt

            dcpa_vals = pair_dcpa[k]
            dcpa_vals = dcpa_vals[np.isfinite(dcpa_vals)]
            min_dcpa = float(np.min(dcpa_vals)) if dcpa_vals.size else dcpa_safe
            episode_metrics[agent]["min_dcpa_m"] = min(
                episode_metrics[agent]["min_dcpa_m"], min_dcpa
            )

            finite_d = np.isfinite(pair_dist[k])
            min_dist = float(np.min(pair_dist[k][finite_d])) if np.any(finite_d) else np.inf

            collision = (min_dist < loa_m) or (min_dcpa < loa_m)
            if collision:
                episode_metrics[agent]["collision"] = 1
                any_collision = True

            Xwpt_k = Xwpt_all[k]
            Ywpt_k = Ywpt_all[k]
            i_wpt_k = int(i_wpt_all[k])

            wx_nmi = float(Xwpt_k[i_wpt_k])
            wy_nmi = float(Ywpt_k[i_wpt_k])
            wx_m, wy_m = wx_nmi * NMI, wy_nmi * NMI

            final_reached = (
                i_wpt_k >= len(Xwpt_k) - 1
                and float(np.hypot(wx_m - x_k, wy_m - y_k)) <= goal_radius
            )

            if final_reached:
                episode_metrics[agent]["success"] = 1
                if episode_metrics[agent]["completion_time_s"] is None:
                    episode_metrics[agent]["completion_time_s"] = t
                any_success = True

        step_count += 1
        t += dt

        # centralized stop, matching your PPO env convention
        if any_collision or any_success:
            break

    own = episode_metrics["ship_0"]

    per_agent_path = [episode_metrics[a]["path_length_m"] for a in agent_ids]
    per_agent_dcpa = [episode_metrics[a]["min_dcpa_m"] for a in agent_ids]
    per_agent_risk = [episode_metrics[a]["risk_exposure"] for a in agent_ids]
    per_agent_success = [episode_metrics[a]["success"] for a in agent_ids]
    per_agent_collision = [episode_metrics[a]["collision"] for a in agent_ids]
    per_agent_ct = [
        episode_metrics[a]["completion_time_s"]
        for a in agent_ids
        if episode_metrics[a]["completion_time_s"] is not None
    ]

    return {
        "episode_steps": step_count,
        "collision_any": float(any(per_agent_collision)),
        "success_rate_agents": float(np.mean(per_agent_success)),
        "path_length_m_mean": float(np.mean(per_agent_path)),
        "path_length_m_ownship": float(own["path_length_m"]),
        "min_dcpa_m_mean": float(np.mean(per_agent_dcpa)),
        "min_dcpa_m_ownship": float(own["min_dcpa_m"]),
        "risk_exposure_mean": float(np.mean(per_agent_risk)),
        "risk_exposure_ownship": float(own["risk_exposure"]),
        "completion_time_s_mean": safe_mean(per_agent_ct),
        "completion_time_s_ownship": float(own["completion_time_s"])
        if own["completion_time_s"] is not None else float("nan"),
    }


def main():
    args = parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if args.output_dir is None:
        output_dir = Path(f"corall_baseline_eval_case{args.case}_{timestamp}") / f"seed_{args.seed}"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ep in range(args.episodes):
        ep_seed = args.seed + ep

        row = run_corall_baseline_episode(
            case_number=args.case,
            dt=args.dt,
            sim_time=args.sim_time,
            loa_m=args.loa_m,
            route_len_nmi=args.route_len_nmi,
            seed=ep_seed,
        )
        row["episode_index"] = ep
        row["seed"] = ep_seed
        rows.append(row)

        print(
            f"[baseline ep {ep+1}/{args.episodes}] "
            f"collision_any={row['collision_any']}, "
            f"success_rate_agents={row['success_rate_agents']:.3f}, "
            f"path_length_ownship_m={row['path_length_ownship_m']:.3f}, "
            f"min_dcpa_ownship_m={row['min_dcpa_ownship_m']:.3f}, "
            f"risk_exposure_ownship={row['risk_exposure_ownship']:.3f}"
        )

    summary = {
        "case": args.case,
        "episodes": args.episodes,
        "seed_base": args.seed,
        "collision_rate": safe_mean([r["collision_any"] for r in rows]),
        "success_rate_agents_mean": safe_mean([r["success_rate_agents"] for r in rows]),
        "path_length_m_mean": safe_mean([r["path_length_m_mean"] for r in rows]),
        "path_length_ownship_m_mean": safe_mean([r["path_length_m_ownship"] for r in rows]),
        "min_dcpa_m_mean": safe_mean([r["min_dcpa_m_mean"] for r in rows]),
        "min_dcpa_ownship_m_mean": safe_mean([r["min_dcpa_m_ownship"] for r in rows]),
        "risk_exposure_mean": safe_mean([r["risk_exposure_mean"] for r in rows]),
        "risk_exposure_ownship_mean": safe_mean([r["risk_exposure_ownship"] for r in rows]),
        "completion_time_s_mean": safe_mean([r["completion_time_s_mean"] for r in rows]),
        "completion_time_s_ownship_mean": safe_mean([r["completion_time_s_ownship"] for r in rows]),
    }

    csv_path = output_dir / "corall_baseline_per_episode.csv"
    fieldnames = [
        "episode_index",
        "episode_seed",
        "episode_steps",
        "collision_any",
        "success_rate_agents",
        "path_length_m_mean",
        "path_length_m_ownship",
        "min_dcpa_m_mean",
        "min_dcpa_m_ownship",
        "risk_exposure_mean",
        "risk_exposure_ownship",
        "completion_time_s_mean",
        "completion_time_s_ownship",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    summary_path = output_dir / "corall_baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== CORALL Baseline Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\nSaved per-episode CSV to: {csv_path}")
    print(f"Saved summary JSON to: {summary_path}")


if __name__ == "__main__":
    main()