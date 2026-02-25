"""
Multi-agent (PettingZoo ParallelEnv) for MAPPO training compatible with CORALL simulation base

Every vessel in each of the Imazu cases is an agent 
- agent 0 is "ownship"
- agents 1...K are former "obstacle ships" from CORALL get_obstacle_data(case_number)

State: 
    X = [x, y, psi, r, b, u] (m, m, rad, rad/s, ..., m/s)

Waypoints
- stored in nautifcal miles (nmi) like CORALL, but state in meters
- ownship waypoints are case-driven (simple defaults are provided)
- other ships get a straight-line far waypoint along initial heading
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional 

import numpy as np
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv

# ensure CORALL repository relative imports resolve
from maritime_rl.path_setup import ensure_paths
ensure_paths()

# importing CORALL core modules (from CORALL repo)
from utils.imazu_cases import get_obstacle_data
from navigation.planning import waypoint_selection, planning
from dynamics.controller import controller
from dynamics.actuator_modeling import actuator_modeling 
from dynamics.vessel_dynamics import vessel_dynamics
from core.integration import integration
from risk_assessment.cpa_calculations import cpa_calculations
from risk_assessment.risk_calculations import risk_calculations

NMI = 1852.0   # meters per nautical mile

@dataclass
class ObsNorm: 
    """Normalization constants for observation features"""

    # position scales (nmi)
    pos_scale_nmi: float = 40.0               # typical scenario size
    # speed / turn rate scales
    u_max: float = 15.0                       # m/s (match action decoder)
    r_scale: float = 0.25                     # rad/s (tunable -> keep within [-1, 1] with clip)
    b_scale: float = 5.0                      # __ bias 
    # CPA scaling 
    dcpa_scale_m: float = 400.0               # m, normalize DCPA
    tcpa_scale_s: float = 300.0               # s, used to normalize TCPA
    # clip range for safety
    clip: float = 1.0


class MultiShipParallelEnv(ParallelEnv):
    """
    CORALL case-driven multi-agent environment for MAPPO algorithm

    Observation (agent k)

        own_state_norm: 
            - x_nmi / pos_scale, y_nmi / pos_scale
            - sin(psi), cos(psi)
            - r / r_scale
            - b / b_scale
            - u / u_max
        
        per other ship j != k:
            - dx_nmi / pos_scale, dy_nmi / pos_scale
            - dist_nmi / pos_scale
            - sin(bearing), cos(bearing)
            - DCPA / dcpa_scale
            - TCPA / tcpa_scale
            - risk [0, 1] -> clipped

    All features clipped to [-clip, clip] for normalization
    """

    metadata = {"render_modes": ["human"], "name": "corall_multiship_mappo_v0"}

    def __init__(
            
        self, 
        case_number: int, 
        dt: float = 0.2, 
        sim_time: float = 300.0, 
        render_mode: Optional[str] = None, 
        # action discretization
        n_heading: int = 7, 
        n_speed: int = 5, 
        max_heading_change_deg: float = 25.0, 
        u_min: float = 5.0, 
        u_max: float = 15.0, 
        # ship geometry for collision logic 
        loa_m: float = 175.0,
        # observation normalization
        obs_norm: ObsNorm = ObsNorm(),
        # waypoint generation parameters
        ownship_default_route_nmi: Tuple[Tuple[float, ...], Tuple[float, ...]] = ((0.0, 40.0), (0.0, 0.0)), 
        other_ship_route_len_nmi: float = 40.0,
        seed: Optional[int] = None,
    ): 

        super().__init__()
        self.case_number = int(case_number)
        self.dt = float(dt)
        self.Ts = float(dt)
        self.sim_time = float(sim_time)
        self.render_mode = render_mode
    

        self.n_heading = int(n_heading)
        self.n_speed = int(n_speed)
    
        self.max_heading_change = np.deg2rad(float(max_heading_change_deg))
        self.u_min = float(u_min)
        self.u_max = float(u_max)

        self.LOA_own = float(loa_m)
        self.norm = obs_norm
    
        self.ownship_default_route_nmi = ownship_default_route_nmi
        self.other_ship_route_len_nmi = float(other_ship_route_len_nmi)
    
        self.rng = np.random.default_rng(seed)

        # Determine number of agents from CORALL case
        Xob, Yob, Vob, psiob = get_obstacle_data(self.case_number)
        self.n_obstacles = len(Xob)
        self.n_agents = 1 + self.n_obstacles
    
        self.agents = [f"ship_{i}" for i in range(self.n_agents)]
        self.possible_agents = self.agents[:]
    
        # Spaces: MultiDiscrete([heading_idx, speed_idx]) for each agent 
        self._action_space = {

            agent: spaces.MultiDiscrete([self.n_heading, self.n_speed]) for agent in self.agents
        }
    
        # Observation space is same for all agents, but depends on n_agents 
        own_dim = 7             # [x, y, sin(psi), cos(psi), r, b, u]
        per_other_dim = 8       # dx, dy, dist, sinB, cosB, dcpa, tcpa, risk -- B is rel. bearing of other ship to ownship
        obs_dim = own_dim + (self.n_agents - 1) * per_other_dim
        self._obs_space = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32) for agent in self.agents
        }
    
        # Internal buffers
        self.X_all = np.zeros((self.n_agents, 6), dtype=float)
        self.prev_X_all = self.X_all.copy()
        self.i_wpt_all = np.zeros(self.n_agents, dtype=int)
        self.ui_psi1_all = np.zeros(self.n_agents, dtype=float)
        self.Xwpt_all: List[List[float]] = []
        self.Ywpt_all: List[List[float]] = []
    
        self.t = 0.0
        self.step_count = 0.0
        self.done = False

        # Cache case data for reset
        self._case_cache = dict(Xob=np.array(Xob, dtype=float), 
                                Yob=np.array(Yob, dtype=float), 
                                Vob=np.array(Vob, dtype=float), 
                                psiob=np.array(psiob, dtype=float))
    

    # PettingZoo API
    def observation_space(self, agent):
        return self._obs_space[agent]
    
    def action_space(self, agent):
        return self._action_space[agent]
    

    # --------------------------------------
    # Initialization utilities
    # --------------------------------------

    def _init_from_case(self) -> np.ndarray:
        """Create X_all state from Imazu case defined. Agent 0 is ownship"""

        Xob = self._case_cache["Xob"]
        Yob = self._case_cache["Yob"]
        Vob = self._case_cache["Vob"]
        psiob = self._case_cache["psiob"]

        X_all = np.zeros((self.n_agents, 6), dtype=float)

        # Ownship default: at origin, heading, intiial surge at mid range (u_min)
        u0 = float(np.clip(0.5 * (self.u_min + self.u_max), self.u_min, self.u_max))
        X_all[0, :] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, u0], dtype=float)

        for j in range(self.n_obstacles):
            X_all[j+1, :] = np.array([Xob[j], Yob[j], psiob[j], 0.0, 0.0, Vob[j]], dtype=float)

        return X_all

    def _build_waypoints_from_states(self, X_all: np.ndarray) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Build waypoints in nmi (like in CORALL)
        
        Ownship: use case-based routing (start with default)
        Other-ships: keep initial course (straight-line for waypoints)

        """
        # List over ships -> list over consecutive waypoint coordinates

        Xwpt_all: List[List[float]] = []
        Ywpt_all: List[List[float]] = []

        # Ownship route 
        ## maybe expand this to a per-case table for experiment design 
        X_own, Y_own = self.ownship_default_route_nmi 
        Xwpt_all.append(list(X_own))
        Ywpt_all.append(list(Y_own))

        # Other ship routes: straight line
        for k in range(1, self.n_agents):
            x_m, y_m, psi = X_all[k, 0], X_all[k, 1], X_all[k, 2]
            x0 = x_m / NMI
            y0 = y_m / NMI
            x1 = x0 + self.other_ship_route_len_nmi * float(np.cos(psi))
            y1 = y0 + self.other_ship_route_len_nmi * float(np.sin(psi))
            Xwpt_all.append([x0, x1])
            Ywpt_all.append([y0, y1])

        return Xwpt_all, Ywpt_all


    def _reset_internal_state(self):
        self.t = 0.0
        self.step_count = 0
        self.done = False

        self.X_all = self._init_from_case()
        self.prev_X_all = self.X_all.copy()

        self.i_wpt_all = np.zeros(self.n_agents, dtype=int)
        self.ui_psi1_all = np.zeros(self.n_agents, dtype=float)

        self.Xwpt_all, self.Ywpt_all = self._build_waypoints_from_states(self.X_all)

    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_internal_state()

        observations = {agent: self._get_observation(k) for k, agent in enumerate(self.agents)}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]):
        """
        actions: dict {agent_name: MultiDiscrete([heading_idx, speed_idx])}
        
        """

        if self.done:
            observations, infos = self.reset()
            rewards = {agent: 0.0 for agent in self.agents}
            terminations = {agent: True for agent in self.agents}
            truncations = {agent: False for agent in self.agents}
            return observations, rewards, terminations, truncations, infos

        self.prev_X_all = self.X_all.copy()

        # 1) decode actions into (psi_ref, u_cmd)
        psi_ref = np.zeros(self.n_agents, dtype=float)
        u_cmd = np.zeros(self.n_agents, dtype=float)

        for k, agent in enumerate(self.agents):
            a = np.asarray(actions[agent], dtype=float)
            if a.size != 2:
                raise ValueError(f"Action for {agent} must be shape (2,), got {a.shape}")
            
            heading_idx = int(np.clip(a[0], 0, self.n_heading - 1))
            speed_idx = int(np.clip(a[1], 0, self.n_speed - 1))

            # normalize to [-1, 1] scale
            if self.n_heading > 1:
                delta_heading_norm = -1.0 + 2.0 * heading_idx / (self.n_heading - 1)
            else: 
                delta_heading_norm = 0.0
            
            if self.n_speed > 1:
                delta_speed_norm = -1.0 + 2.0 * speed_idx / (self.n_speed - 1)
            else:
                delta_speed_norm = 0.0
            
            delta_heading = delta_heading_norm * self.max_heading_change

            # map speed norm [-1, 1] to [u_min, u_max]
            u_cmd[k] = self.u_min + 0.5 * (delta_speed_norm + 1.0) * (self.u_max - self.u_min)

            # CORALL waypoint selection and planner format -> use coordinates in nmi
            Xwpt_k = self.Xwpt_all[k]
            Ywpt_k = self.Ywpt_all[k]
            i_wpt_k = int(self.i_wpt_all[k])

            x_m, y_m, psi_k, r_k, b_k, u_k = self.X_all[k, :]
            x_nmi, y_nmi = x_m / NMI, y_m / NMI

            i_wpt_k = waypoint_selection(Xwpt_k, Ywpt_k, x_nmi, y_nmi, i_wpt_k)
            self.i_wpt_all[k] = i_wpt_k

            psi_wp = planning(Xwpt_k, Ywpt_k, x_nmi, y_nmi, i_wpt_k)
            psi_ref[k] = float(psi_wp + delta_heading)
        
        # 2) advance dynamics for each ship 
        for k in range(self.n_agents):
            x_m, y_m, psi_k, r_k, b_k, u_k = self.X_all[k, :]

            ui_psi1_k = float(self.ui_psi1_all[k])
            tau_c, v_c, ui_psi1_k = controller(
                psi_ref[k], psi_k, r_k, u_cmd[k], b_k, ui_psi1_k, self.Ts
            )
            self.ui_psi1_all[k] = ui_psi1_k

            tau_ac = actuator_modeling(tau_c, sat_amp_s=20)
            inputs = [tau_ac, v_c]

            X_dot = vessel_dynamics(self.X_all[k, :], inputs)
            self.X_all[k, :] = integration(self.X_all[k, :], X_dot, self.dt)

        # 3) rewards / dones
        rewards, terminations, truncations, infos = self._compute_rewards_and_dones()

        # time / truncation handling
        self.t += self.dt
        self.step_count += 1

        if self.t >= self.sim_time:
            for agent in self.agents:
                truncations[agent] = True
            self.done = True
        
        observations = {agent: self._get_observation(k) for k, agent in enumerate(self.agents)}

        return observations, rewards, terminations, truncations, infos

    # --------------------------
    # Observations
    # --------------------------
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def _own_state_features(self, k: int) -> np.ndarray:
        x_m, y_m, psi, r, b, u = self.X_all[k, :]
        x_nmi = x_m / NMI
        y_nmi = y_m / NMI
    
        feats = np.array([

            x_nmi / self.norm.pos_scale_nmi,
            y_nmi / self.norm.pos_scale_nmi, 
            np.sin(psi), 
            np.cos(psi), 
            r / self.norm.r_scale,
            b / self.norm.b_scale, 
            u / self.norm.u_max, 

        ], dtype=np.float32)

        feats = np.clip(feats, -self.norm.clip, self.norm.clip)
        return feats

    def _pairwise_cpa_risk(self, a: int, b: int) -> Tuple[float, float, float, float]:
        """
        Compute DCPA (m), TCPA(s), dist(m), risk for pair (a, b) using CORALL cpa_calculations + risk_calculations functions
        - use current and previous positions

        """

        xa, ya = float(self.X_all[a, 0]), float(self.X_all[a, 1])
        xa_prev, ya_prev = float(self.prev_X_all[a, 0]), float(self.prev_X_all[a, 1])
        
        xb, yb = float(self.X_all[b, 0]), float(self.X_all[b, 1])
        xb_prev, yb_prev = float(self.prev_X_all[b, 0]), float(self.prev_X_all[b, 1])
        
        dcpa, tcpa, vrel, alpha, psi_vrel = cpa_calculations(
            xa, ya, xa_prev, ya_prev, 
            xb, yb, xb_prev, yb_prev, 
            self.dt
        )
        dist = float(np.hypot(xa - xb, ya - yb))
        risk = risk_calculations(dcpa, tcpa, dist, vrel)
        return float(dcpa), float(tcpa), float(dist), float(risk)
    
    def _get_observation(self, k: int) -> np.ndarray:
        own = self._own_state_features(k)
        xk_m, yk_m, psik = self.X_all[k, 0], self.X_all[k, 1], self.X_all[k, 2]

        per_other = []
        for j in range(self.n_agents):
            if j == k:
                continue
            
            xj_m, yj_m, psij = self.X_all[j, 0], self.X_all[j, 1], self.X_all[j, 2]
            dx_nmi = (xj_m - xk_m) / NMI
            dy_nmi = (yj_m - yk_m) / NMI
            dist_nmi = float(np.hypot(dx_nmi, dy_nmi))

            # use relative bearing in obs space
            bearing = float(np.arctan2(dy_nmi, dx_nmi))
            bearing_rel = self._wrap_angle(bearing - psik)           # relative bearing of other ship j heading to ownship k heading
            sin_b, cos_b = np.sin(bearing_rel), np.cos(bearing_rel)

            dcpa_m, tcpa_s, dist_m, risk = self._pairwise_cpa_risk(k, j)

            # normalize CPA features
            dcpa_n = np.clip(dcpa_m / self.norm.dcpa_scale_m, -self.norm.clip, self.norm.clip)
            tcpa_n = np.clip(tcpa_s / self.norm.tcpa_scale_s, -self.norm.clip, self.norm.clip)

            # risk is in [0, 1] from base function, clip to [0, 1] and then map to [-1, 1] range like other normalized values for symmetry
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

            vec = np.clip(vec, -self.norm.clip, self.norm.clip)
            per_other.append(vec)
        
        obs = np.concatenate([own] + per_other, axis=0).astype(np.float32)
        # safety: if NaNs appear, replace (possible if dynamics blow up )
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

        return obs

    # -------------------------------
    # Rewards / terminations 
    def _compute_rewards_and_dones(self):
        rewards: Dict[str, float] = {}
        terminations: Dict[str, bool] = {agent: False for agent in self.agents}
        truncations: Dict[str, bool] = {agent: False for agent in self.agents}
        infos: Dict[str, dict] = {agent: {} for agent in self.agents}

        # pairwise risk and DCPA across all agent
        pair_dcpa = np.full((self.n_agents, self.n_agents), np.inf, dtype=float)
        pair_risk = np.zeros((self.n_agents, self.n_agents), dtype=float)
        pair_dist = np.full((self.n_agents, self.n_agents), np.inf, dtype=float)

        for a in range(self.n_agents):
            for b in range(self.n_agents):
                if a == b:
                    continue
                dcpa, tcpa, dist, risk = self._pairwise_cpa_risk(a, b)
                pair_dcpa[a, b] = dcpa
                pair_risk[a, b] = risk
                pair_dist[a, b] = dist
        
        LOA = self.LOA_own
        dcpa_safe = LOA * 4.0
        goal_radius = LOA * 2.0

        for k, agent in enumerate(self.agents):
            # waypoint following: reward along-track movement, penalize cross-track
            Xwpt_k = self.Xwpt_all[k]
            Ywpt_k = self.Ywpt_all[k]
            i_wpt_k = int(self.i_wpt_all[k])

            x_k, y_k = float(self.X_all[k, 0]), float(self.X_all[k, 1])
            wx_nmi, wy_nmi = float(Xwpt_k[i_wpt_k]), float(Ywpt_k[i_wpt_k])
            wx_m, wy_m = wx_nmi * NMI, wy_nmi * NMI

            psi_des = float(np.arctan2(wy_m - y_k, wx_m - x_k))
            t_hat = np.array([np.cos(psi_des), np.sin(psi_des)], dtype=float)
            n_hat = np.array([-np.sin(psi_des), np.cos(psi_des)], dtype=float)

            dx_step = float(x_k - self.prev_X_all[k, 0])
            dy_step = float(y_k - self.prev_X_all[k, 1])
            dp = np.array([dx_step, dy_step], dtype=float)

            along = float(dp @ t_hat)
            cross = float(dp @ n_hat)

            step_scale = max(1e-6, self.u_max * self.dt)
            along_norm = float(np.clip(along / step_scale, -1.0, 1.0))
            cross_norm = float(np.clip(cross / step_scale, -1.0, 1.0))

            w_along = 1.0
            w_cross = -0.2

            r_along = w_along * along_norm
            r_cross = -0.2 * abs(cross_norm)

            # risk penalty: discourage large collision risk with any other vessel
            agent_risks = pair_risk[k].copy()
            agent_risks[k] = 0.0 
            max_risk = float(np.max(agent_risks))
            infos[agent]["max_risk"] = max_risk

            w_risk = -2.0
            r_risk = w_risk * max_risk

            # DCPA penalty: normalized on scale of smallest safe DCPA
            dcpa_vals = pair_dcpa[k]
            dcpa_vals = dcpa_vals[np.isfinite(dcpa_vals)]
            min_dcpa = float(np.min(dcpa_vals)) if dcpa_vals.size else dcpa_safe
            dcpa_def = max(0.0, dcpa_safe - min_dcpa)
            infos[agent]["min_dcpa"] = min_dcpa
            
            w_dcpa = -1.0
            r_dcpa = w_dcpa * (dcpa_def / dcpa_safe)

            # speed penalty for huge speed changes
            u_k = float(self.X_all[k, 5])
            cruising = 10.0

            w_speed = -0.05
            r_speed = w_speed * ((u_k - cruising) **2) / (cruising **2)

            # time penalty 
            r_time = -0.002
            infos[agent]["t"] = self.t

            total = r_along + r_cross + r_risk + r_dcpa + r_speed + r_time

            # collision penalty
            finite_d = np.isfinite(pair_dist[k])
            min_dist = float(np.min(pair_dist[k][finite_d])) if np.any(finite_d) else np.inf
            collision = (min_dist < LOA) or (min_dcpa < LOA)
            infos[agent]["min_dist"] = min_dist

            w_collision = -10.0

            if collision: 
                terminations[agent] = True
                infos[agent]["collision"] = True
                total += w_collision * (1.0 + max_risk + u_k / max(1.0, cruising))

            # sucess: reached final wp and within radius
            dist_to_wp = float(np.hypot(wx_m - x_k, wy_m - y_k))
            final_reached = (i_wpt_k >= len(Xwpt_k) - 1) and (dist_to_wp <= goal_radius)

            w_success = 20.0

            if final_reached:
                terminations[agent] = True
                infos[agent]["success"] = True
                total += w_success

            rewards[agent] = float(total)

        # end episode for all if any termination (centralized training stability)
        if any(terminations.values()):
            self.done = True
        
        return rewards, terminations, truncations, infos 


