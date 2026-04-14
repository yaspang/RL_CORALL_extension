
"""
Creates a CORALLComparisonEnv that directly compares CORALL's scripted guidance and traffic propagation against an RL-controlled ownship, while keeping the same observation space 
and evaluation metrics for compatibility to compare against RL ownship results.

- CORALL-controlled ownship (ship_0) with internal guidance and dynamics propagation

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv

# ensure CORALL repository relative imports resolve
from .path_setup import ensure_paths
ensure_paths()

from utils.imazu_cases import get_obstacle_data
from navigation.planning import waypoint_selection, planning
from navigation.reactive_avoidance import reactive_avoidance
from navigation.obstacle_sim import obstacle_sim
from dynamics.controller import controller
from dynamics.actuator_modeling import actuator_modeling
from dynamics.vessel_dynamics import vessel_dynamics
from core.integration import integration
from risk_assessment.cpa_calculations import cpa_calculations
from risk_assessment.cpa_calculations_0speed import cpa_calculations_0speed
from risk_assessment.risk_calculations import risk_calculations

# Spatial optimization
from .spatial_optimization import AABBBroadPhase

NMI = 1852.0

@dataclass
class ObsNorm: 
    """Normalization constants for observation features"""

    # position scales (nmi)
    pos_scale_nmi: float = 2.0              # typical scenario size (CORALL)
    # speed / turn rate scales
    u_max: float = 10.0                       # m/s (match action decoder)
    r_scale: float = 0.25                     # rad/s (tunable -> keep within [-1, 1] with clip)
    b_scale: float = 5.0                      # __ bias 

    # CPA scaling 
    dcpa_scale_m: float = 400.0               # m, normalize DCPA
    tcpa_scale_s: float = 300.0               # s, normalize TCPA
    # clip range for safety
    clip: float = 1.0

class CORALLComparisonEnv(ParallelEnv):
    """
    Direct comparison environment for CORALL-vs-RL ownship evaluation.

    Purpose:
    - Keep CORALL-style scripted traffic propagation for target ships/obstacles.
    - Control only ownship (ship_0) through the helper `compute_corall_baseline_action()`.
    - Produce ownship-centric metrics compatible with current evaluation
      pipeline: path length, min DCPA, risk exposure, min separation, success,
      completion time, and goal progress.

    Design choice:
    - Only `ship_0` is an active agent in this version of PettingZoo API.
    - Obstacles are exogenous traffic propagated with CORALL's `obstacle_sim()`.
    - Observation still contains full traffic-relative geometry.

    """

    metadata = {"render_modes": ["human"], "name": "corall_comparison_env_v0"}


    def __init__(
            self,
            case_number: int,
            dt: float = 0.5,
            sim_time: float = 1950.0,
            render_mode: Optional[str] = None,
            u_min: float = 5.0,
            u_max: float = 10.0,
            loa_m: float = 30.0,
            bol_m: float = 16.0,
            obs_norm: ObsNorm = ObsNorm(),
            route_len_nmi: float = 2.0,
            goal_radius_m: float = 60.0,
            # spatial optimization (AABB broad-phase filtering for pairwise CPA)
            enable_aabb_filtering: bool = False,
            aabb_radius_m: float = 1500.0,
            seed: Optional[int] = None,
        ):
            super().__init__()

            self.case_number = int(case_number)
            self.dt = float(dt)
            self.sim_time = float(sim_time)
            self.render_mode = render_mode

            self.u_min = float(u_min)
            self.u_max = float(u_max)
            self.LOA_own = float(loa_m)
            self.BOL_own = float(bol_m)
            self.goal_radius_m = float(goal_radius_m)
            self.norm = obs_norm
            self.route_len_nmi = float(route_len_nmi)
            self.rng = np.random.default_rng(seed)
            
            # Spatial optimization
            self.enable_aabb_filtering = bool(enable_aabb_filtering)
            self.aabb_radius_m = float(aabb_radius_m)

            # CORALL case traffic
            Xob, Yob, Vob, psiob = get_obstacle_data(self.case_number)
            self.n_obstacles = len(Xob)

            # Only ownship is an active control agent for CORALL baseline comparison
            self.agents = ["ship_0"]
            self.possible_agents = self.agents[:]

            # Actions are ignored; env uses internal CORALL guidance (structural alignment)
            self._action_space = {
                "ship_0": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            }

            own_dim = 7
            per_other_dim = 8
            obs_dim = own_dim + self.n_obstacles * per_other_dim
            self._obs_space = {
                "ship_0": spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            }

            # State [x, y, psi, r, b, u] for ownship and each obstacle.
            self.n_agents_total = 1 + self.n_obstacles
            self.X_all = np.zeros((self.n_agents_total, 6), dtype=float)
            self.prev_X_all = self.X_all.copy()

            self.i_wpt = 1
            self.ui_psi1 = 0.0
            self.Xwpt: List[float] = []
            self.Ywpt: List[float] = []
            self.psi_route_ref = 0.0
            self.u_des = 0.0

            self.t = 0.0
            self.step_count = 0
            self.done = False

            # Scripted obstacle state caches (CORALL-style)
            self.Xob = np.array(Xob, dtype=float)
            self.Yob = np.array(Yob, dtype=float)
            self.Vob = np.array(Vob, dtype=float)
            self.psiob = np.array(psiob, dtype=float)
            self._case_cache = dict(
                Xob=self.Xob.copy(),
                Yob=self.Yob.copy(),
                Vob=self.Vob.copy(),
                psiob=self.psiob.copy(),
            )

            # Pairwise geometry for ownship vs traffic (kept in full matrix form so plotting / history tools remain compatible).
            self.pair_dcpa = np.full((self.n_agents_total, self.n_agents_total), np.inf, dtype=float)
            self.pair_tcpa = np.zeros((self.n_agents_total, self.n_agents_total), dtype=float)
            self.pair_dist = np.full((self.n_agents_total, self.n_agents_total), np.inf, dtype=float)
            self.pair_risk = np.zeros((self.n_agents_total, self.n_agents_total), dtype=float)

            self.episode_metrics: Dict[str, Dict[str, float]] = {}


    # ---------------------------------------------------------------------
    # PettingZoo API
    # ---------------------------------------------------------------------
    def observation_space(self, agent):
        return self._obs_space[agent]

    def action_space(self, agent):
        return self._action_space[agent]

    # ---------------------------------------------------------------------
    # Setup helpers
    # ---------------------------------------------------------------------
    def init_ownship(self) -> np.ndarray:
        """Initialize ownship consistent with existing env convention."""
        if len(self.Vob) > 0:
            u0 = float(np.clip(self.Vob[0], self.u_min, self.u_max))
        else:
            u0 = float(np.clip(0.5 * (self.u_min + self.u_max), self.u_min, self.u_max))

        self.u_des = u0
        self.ui_psi1 = 0.0
        self.i_wpt = 1
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, u0], dtype=float)

    def build_ownship_waypoints(self) -> Tuple[List[float], List[float]]:
        """Build ownship route waypoints consistent with CORALL case definition (straight line to goal at route_len_nmi distance)."""
        return [0.0, self.route_len_nmi], [0.0, 0.0]

    def refresh_route_heading_cache(self) -> None:
        if len(self.Xwpt) <= 1:
            self.psi_route_ref = float(self.X_all[0, 2])
            return

        i_wpt = min(int(self.i_wpt), len(self.Xwpt) - 1)
        x_prev_nmi = float(self.Xwpt[max(0, i_wpt - 1)])
        y_prev_nmi = float(self.Ywpt[max(0, i_wpt - 1)])
        x_next_nmi = float(self.Xwpt[i_wpt])
        y_next_nmi = float(self.Ywpt[i_wpt])

        dx_m = (x_next_nmi - x_prev_nmi) * NMI
        dy_m = (y_next_nmi - y_prev_nmi) * NMI

        # If waypoints are coincident, keep previous heading (e.g., from initial state or last valid route segment) to avoid discontinuity. 
        if abs(dx_m) < 1e-12 and abs(dy_m) < 1e-12:
            self.psi_route_ref = float(self.X_all[0, 2])
        else:
            self.psi_route_ref = float(np.arctan2(dy_m, dx_m))

    def reset_internal_state(self) -> None:
        self.t = 0.0
        self.step_count = 0
        self.done = False

        # reset scripted traffic to the original CORALL case definition
        self.Xob = self._case_cache["Xob"].copy()
        self.Yob = self._case_cache["Yob"].copy()
        self.Vob = self._case_cache["Vob"].copy()
        self.psiob = self._case_cache["psiob"].copy()

        self.X_all = np.zeros((self.n_agents_total, 6), dtype=float)
        self.X_all[0, :] = self.init_ownship()
        for j in range(self.n_obstacles):
            self.X_all[j + 1, :] = np.array([
                self.Xob[j], self.Yob[j], self.psiob[j], 0.0, 0.0, self.Vob[j]
            ], dtype=float)

        self.prev_X_all = self.X_all.copy()
        self.Xwpt, self.Ywpt = self.build_ownship_waypoints()
        self.refresh_route_heading_cache()
        self.update_pairwise_geometry_cache()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.reset_internal_state()

        self.episode_metrics = {
            "ship_0": {
                "path_length_m": 0.0,
                "min_dcpa_m": float("inf"),
                "min_tcpa_s": float("inf"),
                "risk_exposure": 0.0,
                "collision": 0,
                "success": 0,
                "completion_time_s": np.nan,
                "min_actual_sep_m": float("inf"),
                "near_miss": 0,
                "goal_progress": 0.0,
                "goal_passed": 0,
            }
        }

        observations = {"ship_0": self.get_observation()}
        infos = {"ship_0": {}}
        return observations, infos

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def compute_corall_guidance(self) -> Tuple[float, float]:
        """Compute CORALL heading and speed guidance for ownship (ignores action input).
        
        Returns:
            psi_ref: desired heading (rad)
            u_des: desired speed (m/s)
        """
        x_m, y_m, psi_k, _, _, _ = self.X_all[0, :]
        x_nmi = float(x_m / NMI)
        y_nmi = float(y_m / NMI)

        # Update waypoint and get planning heading
        self.i_wpt = waypoint_selection(self.Xwpt, self.Ywpt, x_nmi, y_nmi, int(self.i_wpt))
        psi_wp = planning(self.Xwpt, self.Ywpt, x_nmi, y_nmi, int(self.i_wpt))
        if psi_wp is None:
            psi_wp = float(self.psi_route_ref)

        # Apply reactive avoidance
        x_ob_nmi = [float(v / NMI) for v in self.Xob]
        y_ob_nmi = [float(v / NMI) for v in self.Yob]
        psi_oa, _, _, _, _ = reactive_avoidance(
            x_ob_nmi, y_ob_nmi, x_nmi, y_nmi, psi_k, self.t
        )

        psi_ref = float(psi_wp + psi_oa)
        return psi_ref, float(self.u_des)

    # ---------------------------------------------------------------------
    # Dynamics and traffic propagation
    # ---------------------------------------------------------------------
    def advance_ownship(self) -> None:
        """Advance ownship dynamics using CORALL guidance (ignores action input)."""
        psi_ref, u_cmd = self.compute_corall_guidance()
        x_m, y_m, psi_k, r_k, b_k, _ = self.X_all[0, :]

        tau_c, v_c, self.ui_psi1 = controller(
            psi_ref, psi_k, r_k, u_cmd, b_k, self.ui_psi1, self.dt
        )
        tau_ac = actuator_modeling(tau_c, sat_amp_s=20)
        X_dot = vessel_dynamics(self.X_all[0, :], [tau_ac, v_c])
        self.X_all[0, :] = integration(self.X_all[0, :], X_dot, self.dt)


    def propagate_scripted_traffic(self) -> None:
        """Propagate CORALL traffic with obstacle_sim(), keeping headings fixed."""
        Xob_new, Yob_new, _, _ = obstacle_sim(self.Xob, self.Yob, self.Vob, self.psiob, self.dt)
        self.Xob = np.asarray(Xob_new, dtype=float)
        self.Yob = np.asarray(Yob_new, dtype=float)

        for j in range(self.n_obstacles):
            self.X_all[j + 1, :] = np.array([
                self.Xob[j],
                self.Yob[j],
                self.psiob[j],
                0.0,
                0.0,
                self.Vob[j],
            ], dtype=float)

    # ---------------------------------------------------------------------
    # Geometry and observation
    # ---------------------------------------------------------------------
    def update_pairwise_geometry_cache(self) -> None:
        """
        Compute pairwise CPA / risk geometry for current state.
        - For ownship (agent 0): use velocity-based CPA (cpa_calculations_0speed) to handle active maneuvering
        - For obstacles: use projection-based CPA since they follow CORALL scripted trajectories
        - Optional: AABB broad-phase filtering to reduce expensive CPA calculations (recommended for 5+ agents)
        """
        self.pair_dcpa.fill(np.inf)
        self.pair_tcpa.fill(0.0)
        self.pair_dist.fill(np.inf)
        self.pair_risk.fill(0.0)

        # Get pairs to check: either all pairs (naive) or filtered by AABB (optimized)
        if self.enable_aabb_filtering and self.n_agents_total >= 5:
            # Broad-phase: only check pairs with overlapping AABBs
            pairs_to_check = AABBBroadPhase.get_overlapping_pairs(self.X_all, self.aabb_radius_m)
        else:
            # Naive: check all pairs
            pairs_to_check = [(a, b) for a in range(self.n_agents_total) for b in range(a + 1, self.n_agents_total)]

        # Narrow-phase: compute CPA for all checked pairs
        for a, b in pairs_to_check:
            xa, ya = float(self.X_all[a, 0]), float(self.X_all[a, 1])
            psi_a, u_a = float(self.X_all[a, 2]), float(self.X_all[a, 5])
            # Use current heading + speed for velocity estimate (more responsive to control inputs)
            vxa = u_a * np.cos(psi_a)
            vya = u_a * np.sin(psi_a)

            xb, yb = float(self.X_all[b, 0]), float(self.X_all[b, 1])
            psi_b, u_b = float(self.X_all[b, 2]), float(self.X_all[b, 5])
            # Use current heading + speed for velocity estimate
            vxb = u_b * np.cos(psi_b)
            vyb = u_b * np.sin(psi_b)

            dist = float(np.hypot(xa - xb, ya - yb))

            # Use velocity-based CPA for ownship (agent 0) to handle active maneuvering
            if a == 0:
                dcpa, tcpa, vrel, _, _ = cpa_calculations_0speed(
                    xa, ya, xb, yb,
                    vxa, vya, vxb, vyb,
                    dist,
                )
            else:
                # For non-ownship pairs, use projection-based
                xa_prev, ya_prev = float(self.prev_X_all[a, 0]), float(self.prev_X_all[a, 1])
                xb_prev, yb_prev = float(self.prev_X_all[b, 0]), float(self.prev_X_all[b, 1])
                dcpa, tcpa, vrel, _, _ = cpa_calculations(
                    xa, ya, xa_prev, ya_prev,
                    xb, yb, xb_prev, yb_prev,
                    self.dt,
                )
            risk = float(risk_calculations(dcpa, tcpa, dist, vrel))

            self.pair_dcpa[a, b] = self.pair_dcpa[b, a] = float(dcpa)
            self.pair_tcpa[a, b] = self.pair_tcpa[b, a] = float(tcpa)
            self.pair_dist[a, b] = self.pair_dist[b, a] = dist
            self.pair_risk[a, b] = self.pair_risk[b, a] = risk
        
        # For pairs NOT checked (filtered out by AABB), they keep their safe defaults (inf, 0.0)
        # so no additional resetting needed

    def own_state_features(self) -> np.ndarray:
        x_m, y_m, psi, r, b, u = self.X_all[0, :]
        feats = np.array([
            (x_m / NMI) / self.norm.pos_scale_nmi,
            (y_m / NMI) / self.norm.pos_scale_nmi,
            np.sin(psi),
            np.cos(psi),
            r / self.norm.r_scale,
            b / self.norm.b_scale,
            u / self.norm.u_max,
        ], dtype=np.float32)
        return np.clip(feats, -self.norm.clip, self.norm.clip)

    def get_observation(self) -> np.ndarray:
        own = self.own_state_features()
        xk_m, yk_m, psik = self.X_all[0, 0], self.X_all[0, 1], self.X_all[0, 2]

        per_other = []
        for j in range(1, self.n_agents_total):
            xj_m, yj_m, _ = self.X_all[j, 0], self.X_all[j, 1], self.X_all[j, 2]
            dx_nmi = (xj_m - xk_m) / NMI
            dy_nmi = (yj_m - yk_m) / NMI
            dist_nmi = float(np.hypot(dx_nmi, dy_nmi))

            bearing = float(np.arctan2(dy_nmi, dx_nmi))
            bearing_rel = self._wrap_angle(bearing - psik)
            sin_b, cos_b = np.sin(bearing_rel), np.cos(bearing_rel)

            dcpa_m = float(self.pair_dcpa[0, j])
            tcpa_s = float(self.pair_tcpa[0, j])
            risk = float(self.pair_risk[0, j])

            dcpa_n = np.clip(dcpa_m / self.norm.dcpa_scale_m, -self.norm.clip, self.norm.clip)
            tcpa_n = np.clip(tcpa_s / self.norm.tcpa_scale_s, -self.norm.clip, self.norm.clip)
            risk01 = float(np.clip(risk, 0.0, 1.0))
            risk_sym = 2.0 * risk01 - 1.0

            vec = np.array([
                dx_nmi / self.norm.pos_scale_nmi,
                dy_nmi / self.norm.pos_scale_nmi,
                dist_nmi / self.norm.pos_scale_nmi,
                sin_b,
                cos_b,
                dcpa_n,
                tcpa_n,
                np.clip(risk_sym, -self.norm.clip, self.norm.clip),
            ], dtype=np.float32)
            per_other.append(np.clip(vec, -self.norm.clip, self.norm.clip))

        obs = np.concatenate([own] + per_other, axis=0).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    # ---------------------------------------------------------------------
    # Metrics tracking (no reward shaping for baseline control)
    # ---------------------------------------------------------------------
    def route_progress(self) -> Tuple[float, float, float]:
        x0_m = float(self.Xwpt[0]) * NMI
        y0_m = float(self.Ywpt[0]) * NMI
        xg_m = float(self.Xwpt[-1]) * NMI
        yg_m = float(self.Ywpt[-1]) * NMI

        x_k = float(self.X_all[0, 0])
        y_k = float(self.X_all[0, 1])

        route_vec = np.array([xg_m - x0_m, yg_m - y0_m], dtype=float)
        pos_vec = np.array([x_k - x0_m, y_k - y0_m], dtype=float)
        route_len_m = float(np.linalg.norm(route_vec))

        if route_len_m < 1e-9:
            dist_to_goal_m = float(np.hypot(xg_m - x_k, yg_m - y_k))
            return 0.0, 1.0, dist_to_goal_m

        t_hat = route_vec / route_len_m
        progress_m = float(pos_vec @ t_hat)
        dist_to_goal_m = float(np.hypot(xg_m - x_k, yg_m - y_k))
        return progress_m, route_len_m, dist_to_goal_m

    def compute_dones(self):
        """Compute termination conditions and track evaluation metrics.
        
        Focuses on CORALL baseline evaluation: collision detection, goal achievement,
        and performance metrics. No learning-based reward shaping.
        """
        terminations: Dict[str, bool] = {"ship_0": False}
        truncations: Dict[str, bool] = {"ship_0": False}
        infos: Dict[str, dict] = {"ship_0": {}}

        agent = "ship_0"
        k = 0

        x_k, y_k = float(self.X_all[k, 0]), float(self.X_all[k, 1])
        wx_nmi, wy_nmi = float(self.Xwpt[int(self.i_wpt)]), float(self.Ywpt[int(self.i_wpt)])
        wx_m, wy_m = wx_nmi * NMI, wy_nmi * NMI

        # Track path length
        dx_step = float(x_k - self.prev_X_all[k, 0])
        dy_step = float(y_k - self.prev_X_all[k, 1])
        step_dist = float(np.hypot(dx_step, dy_step))
        self.episode_metrics[agent]["path_length_m"] += step_dist

        # Track risk exposure
        agent_risks = self.pair_risk[k].copy()
        agent_risks[k] = 0.0
        max_risk = float(np.max(agent_risks))
        infos[agent]["max_risk"] = max_risk
        self.episode_metrics[agent]["risk_exposure"] += max_risk * self.dt

        # Track minimum DCPA
        # NOTE: Use absolute value (like visualization does) since projection-based CPA can be negative
        # when ships diverge or are at exact CPA crossing. Filter to plausible encounter range (8 nmi)
        # and clip to 5 nmi max (visualization convention) to avoid extreme outliers from numerical error.
        # IMPORTANT: Only consider pairs in active encounter (current separation < 3 nmi) to avoid
        # numerical artifacts from initialization where CPA calculation gives near-zero values.
        dcpa_vals = self.pair_dcpa[k].copy()
        dist_vals = self.pair_dist[k].copy()
        
        # Filter: exclude self-pair, finite values, within 3 nmi min/8 nmi max (active encounter range)
        valid_idx = np.arange(len(dcpa_vals)) != k  # exclude self-pair
        # Only consider pairs in active encounter: current separation between 0 and 3 nmi
        in_encounter = (dist_vals > 0) & (dist_vals <= 3.0 * NMI)
        valid = valid_idx & np.isfinite(dcpa_vals) & in_encounter
        
        if np.any(valid):
            # Take absolute value (handles negative DCPA from diverging trajectories)
            # and clip to 5 nmi max like visualization does
            abs_dcpa = np.abs(dcpa_vals[valid])
            min_dcpa_abs = float(np.min(abs_dcpa))
            min_dcpa_abs = min(min_dcpa_abs, 5.0 * NMI)  # clip to 5 nmi = 9260m
        else:
            min_dcpa_abs = self.LOA_own * 4.0  # safe default if no active encounters
        
        self.episode_metrics[agent]["min_dcpa_m"] = min(self.episode_metrics[agent]["min_dcpa_m"], min_dcpa_abs)

        # Track minimum TCPA
        tcpa_vals = self.pair_tcpa[k]
        tcpa_vals = tcpa_vals[np.isfinite(tcpa_vals)]
        tcpa_vals = tcpa_vals[tcpa_vals > 0.0]  # Only positive TCPA values (future CPA)
        min_tcpa = float(np.min(tcpa_vals)) if tcpa_vals.size else float("inf")
        self.episode_metrics[agent]["min_tcpa_s"] = min(self.episode_metrics[agent]["min_tcpa_s"], min_tcpa)

        # Track collision and separation metrics
        finite_d = np.isfinite(self.pair_dist[k])
        min_dist = float(np.min(self.pair_dist[k][finite_d])) if np.any(finite_d) else np.inf
        collision = (min_dist < self.LOA_own)
        infos[agent]["min_dist"] = min_dist
        self.episode_metrics[agent]["min_actual_sep_m"] = min(self.episode_metrics[agent]["min_actual_sep_m"], min_dist)

        near_miss_threshold = self.LOA_own * 1.5
        if (min_dist >= self.LOA_own) and (min_dist < near_miss_threshold) and not collision:
            if self.episode_metrics[agent]["near_miss"] == 0:
                self.episode_metrics[agent]["near_miss"] = 1

        if collision:
            if self.episode_metrics[agent]["collision"] == 0:
                self.episode_metrics[agent]["collision"] = 1

        # Track goal progress and success
        dist_to_wp = float(np.hypot(wx_m - x_k, wy_m - y_k))
        progress_m, route_len_m, _ = self.route_progress()
        goal_progress = float(np.clip(progress_m / max(route_len_m, 1.0), 0.0, 1.0))
        self.episode_metrics[agent]["goal_progress"] = max(
            self.episode_metrics[agent]["goal_progress"], goal_progress
        )

        # success if reached final waypoint (uses built-in 200m waypoint-acceptance threshold)
        # IMPORTANT: Check that agent is actually NEAR the final waypoint, not just tracking it  
        # waypoint_selection increments index when within Circ (200m), but we need to verify proximity
        final_waypoint_reached_by_index = (int(self.i_wpt) >= len(self.Xwpt) - 1)
        
        if final_waypoint_reached_by_index and len(self.Xwpt) > 1:
            # Verify agent is actually close to the final waypoint
            xf_nmi = float(self.Xwpt[-1])
            yf_nmi = float(self.Ywpt[-1])
            x_nmi = x_k / NMI
            y_nmi = y_k / NMI
            dist_to_goal_nmi = np.hypot(xf_nmi - x_nmi, yf_nmi - y_nmi)
            final_reached = (dist_to_goal_nmi < 200.0 / 1852.0)  # ~200m acceptance radius
        else:
            final_reached = final_waypoint_reached_by_index
            
        if final_reached:
            infos[agent]["success"] = True
            self.episode_metrics[agent]["success"] = 1
            self.episode_metrics[agent]["goal_passed"] = 1
            if np.isnan(self.episode_metrics[agent]["completion_time_s"]):
                self.episode_metrics[agent]["completion_time_s"] = self.t
            terminations[agent] = True  # End episode on successful goal reach
        else:
            infos[agent]["success"] = False

        return terminations, truncations, infos

    # ---------------------------------------------------------------------
    # Step
    # ---------------------------------------------------------------------
    def step(self, actions: Dict[str, np.ndarray]):
        """Step environment using CORALL guidance (action parameter is ignored).
        
        Returns zero rewards since baseline control does not use reward signals.
        Metrics are tracked for evaluation purposes.
        """
        if "ship_0" not in actions:
            raise ValueError("CorallComparisonEnv.step() requires an action dict for ship_0 (value ignored)")

        self.prev_X_all = self.X_all.copy()

        # 1) Ownship guidance computed internally (reactive avoidance + waypoint planning)
        self.advance_ownship()

        # 2) Scripted CORALL traffic propagation
        self.propagate_scripted_traffic()

        # 3) Update geometry cache + compute done conditions & eval metrics
        self.update_pairwise_geometry_cache()
        terminations, truncations, infos = self.compute_dones()

        self.t += self.dt
        self.step_count += 1

        if self.t >= self.sim_time:
            truncations["ship_0"] = True
            self.done = True

        # Always include episode metrics in infos (whether done or not)
        infos["ship_0"]["episode_metrics"] = dict(self.episode_metrics["ship_0"])

        observations = {"ship_0": self.get_observation()}
        rewards = {"ship_0": 0.0}  # Baseline control does not use rewards
        return observations, rewards, terminations, truncations, infos
