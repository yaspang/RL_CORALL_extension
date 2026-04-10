"""
Multi-agent (PettingZoo ParallelEnv) for PPO training compatible with CORALL simulation base

Every vessel in each of the Imazu cases is an agent -> in the frame of CORALL...
- agent 0 is "ownship"
- agents 1...K are former "obstacle ships" from CORALL get_obstacle_data(case_number)

State: 
    X = [x, y, psi, r, b, u] (m, m, rad, rad/s, ..., m/s)

Waypoints
- stored in nautifcal miles (nmi) like CORALL, but state in meters (m)
- ownship waypoints are case-driven (simple default waypoints provided)
- other ships get a straight-line far waypoint along initial heading
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional 

import numpy as np
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv

# ensure CORALL repository relative imports resolve
from .path_setup import ensure_paths
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
from navigation.reactive_avoidance import reactive_avoidance

NMI = 1852.0   # meters per nautical mile

@dataclass
class ObsNorm: 
    """Normalization constants for observation features"""

    # position scales (nmi)
    pos_scale_nmi: float = 40.0               # typical scenario size (CORALL)
    # speed / turn rate scales
    u_max: float = 10.0                       # m/s (match action decoder)
    r_scale: float = 0.25                     # rad/s (tunable -> keep within [-1, 1] with clip)
    b_scale: float = 5.0                      # __ bias 

    # CPA scaling 
    dcpa_scale_m: float = 400.0               # m, normalize DCPA
    tcpa_scale_s: float = 300.0               # s, normalize TCPA
    # clip range for safety
    clip: float = 1.0


class MultiShipParallelEnv(ParallelEnv):
    """
    CORALL case-driven multi-agent environment for PPO algorithm

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

    All features clipped to [-clip, clip] for normalization of observation space
    """

    metadata = {"render_modes": ["human"], "name": "corall_multiship_mappo_v0"}

    def __init__(
            
        self, 
        case_number: int, 
        dt: float = 0.5, 
        sim_time: float = 10_000.0, 
        render_mode: Optional[str] = None, 
        # action discretization
        n_heading: int = 7, 
        n_speed: int = 5, 
        max_heading_change_deg: float = 25.0, 
        ## speed range selected according to typical traffic speeds (9.5m/s) in Imazu cases, but tunable
        u_min: float = 5.0, 
        u_max: float = 10.0, 
        # ship geometry for collision logic 
        loa_m: float = 30.0,
        # observation normalization
        obs_norm: ObsNorm = ObsNorm(),
        # waypoint generation parameter
        route_len_nmi: float = 40.0,
        seed: Optional[int] = None,
        # change mode of sim for baseline vs learning-based use
        traffic_mode: str = "policy", # "policy" or "baseline"
        ownship_mode: str = "policy", # "policy" or "baseline"
    ): 

        super().__init__()
        self.case_number = int(case_number)
        self.dt = float(dt)
        self.sim_time = float(sim_time)
        self.render_mode = render_mode
        self.traffic_mode = str(traffic_mode)
        self.ownship_mode = str(ownship_mode)
        self.n_heading = int(n_heading)
        self.n_speed = int(n_speed)
    
        self.max_heading_change = np.deg2rad(float(max_heading_change_deg))
        self.u_min = float(u_min)
        self.u_max = float(u_max)

        self.LOA_own = float(loa_m)
        self.norm = obs_norm
    
        self.route_len_nmi = float(route_len_nmi)
    
        self.rng = np.random.default_rng(seed)

        # Determine number of agents from CORALL case
        Xob, Yob, Vob, psiob = get_obstacle_data(self.case_number)
        self.n_obstacles = len(Xob)
        self.n_agents = 1 + self.n_obstacles
    
        self.agents = [f"ship_{i}" for i in range(self.n_agents)]
        self.possible_agents = self.agents[:]
    
        # Action Spaces: MultiDiscrete([heading_idx, speed_idx]) for each agent 
        self._action_space = {
            agent: spaces.MultiDiscrete([self.n_heading, self.n_speed]) for agent in self.agents
        }
    
        # Observation Spaces: same for all n_agents 
        own_dim = 7             # [x, y, sin(psi), cos(psi), r, b, u]
        per_other_dim = 8       # dx, dy, dist, sinB, cosB, dcpa, tcpa, risk -- B is rel. bearing of other ship to ownship
        obs_dim = own_dim + (self.n_agents - 1) * per_other_dim
        self._obs_space = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32) for agent in self.agents
        }
    
        # Internal buffers (to start with, will be populated properly in reset())
        self.X_all = np.zeros((self.n_agents, 6), dtype=float)
        self.prev_X_all = self.X_all.copy()
        self.i_wpt_all = np.zeros(self.n_agents, dtype=int)
        self.ui_psi1_all = np.zeros(self.n_agents, dtype=float)
        self.Xwpt_all: List[List[float]] = []
        self.Ywpt_all: List[List[float]] = []
        ## desired cruise / surge speed for each agent dependent on initialized state from Imazu case
        self.u_des_all = np.zeros(self.n_agents, dtype=float)         

        self.t = 0.0
        self.step_count = 0.0
        self.done = False

        # Cache case data for reset
        self._case_cache = dict(Xob=np.array(Xob, dtype=float), 
                                Yob=np.array(Yob, dtype=float), 
                                Vob=np.array(Vob, dtype=float), 
                                psiob=np.array(psiob, dtype=float))
        
        # Initialize dictionaries for evaluation metrics
        self.episode_metrics = {}
        self.prev_done_metrics = {}

        # Route / planning data cache
        ## routes are static, straight lines, so will compute downstream route-heading references once and reuse them 
        self.psi_route_ref_all = np.zeros(self.n_agents, dtype=float)

        # Pairwise geometry data cache for observations and rewards
        ## (initialize distances at np.inf so they are not initialized in "dangerous" range before dynamics update)
        self.pair_dcpa = np.full((self.n_agents, self.n_agents), np.inf, dtype=float)
        self.pair_tcpa = np.zeros((self.n_agents, self.n_agents), dtype=float)
        self.pair_dist = np.full((self.n_agents, self.n_agents), np.inf, dtype=float)
        self.pair_risk = np.zeros((self.n_agents, self.n_agents), dtype=float)
    
    # PettingZoo API
    def observation_space(self, agent):
        return self._obs_space[agent]
    
    def action_space(self, agent):
        return self._action_space[agent]
    

    # --------------------------------------
    # Initialization utilities
    # --------------------------------------

    def init_from_case(self) -> np.ndarray:
        """
        Create X_all state from Imazu case geometry 
        
        Agent 0 is ownship
        
        Agents 1...K are the former CORALL "obstacle" ships 

        State: [x, y, psi, r, b, u]
        Units: [m, m, rad, rad/s, ..., m/s] for dynamics, but waypoints in nmi for CORALL planner compatibility
        """

        Xob = self._case_cache["Xob"]
        Yob = self._case_cache["Yob"]
        Vob = self._case_cache["Vob"]
        psiob = self._case_cache["psiob"]

        X_all = np.zeros((self.n_agents, 6), dtype=float)

        # Ownship initial state
        ## keep CORALL encounter geometry but set ownship at origin, with zero heading
        ## set ownship speed as the same nominal traffic speed scale as other ships in the case (for more consistent dynamics across cases and easier learning)

        if len(Vob) > 0:
            u0 = float(np.clip(Vob[0], self.u_min, self.u_max))
        else:
            u0 = float(np.clip(0.5 * (self.u_min + self.u_max), self.u_min, self.u_max))
        
        self.u_des_all[0] = u0
        X_all[0, :] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, u0], dtype=float)

        # Other ship initial states
        for j in range(self.n_obstacles):
            u_j = float(np.clip(Vob[j], self.u_min, self.u_max))
            self.u_des_all[j + 1] = u_j
            X_all[j + 1, :] = np.array([Xob[j], Yob[j], psiob[j], 0.0, 0.0, u_j], dtype=float)

        return X_all

    def build_waypoints(self, X_all: np.ndarray) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Build navigation routes in nmi for ALL ships corresponding to initial scenario geometry set by Imazu case
        
        Each agent gets: 
            waypoint 0 = initial position
            waypoint 1 = far point in initial heading direction (straight line route for simplicity and CORALL compatibility)

        Keep route structure consistent across ownship and traffic vessels, while using geometry of selected case
        """
        # List over ships -> list over consecutive waypoint coordinates
        Xwpt_all: List[List[float]] = []
        Ywpt_all: List[List[float]] = []

        # Use the same route length parameter for every ship 
        route_len_nmi = float(self.route_len_nmi)

        for k in range(self.n_agents):
            x_m = float(X_all[k, 0])
            y_m = float(X_all[k, 1])
            psi = float(X_all[k, 2])

            x0_nmi = x_m / NMI
            y0_nmi = y_m / NMI

            x1_nmi = x0_nmi + route_len_nmi * float(np.cos(psi))
            y1_nmi = y0_nmi + route_len_nmi * float(np.sin(psi))

            Xwpt_all.append([x0_nmi, x1_nmi])
            Ywpt_all.append([y0_nmi, y1_nmi])

        return Xwpt_all, Ywpt_all

    def refresh_route_heading_cache(self) -> None:
        """
        Precompute downstream route-heading for each ship based on current active waypoint segment. 

        Current CORALL cases have simple straight line routes, so desired heading goal is effectively constant for whole episode. 
            - save computation time by caching route-heading references and only updating if active waypoint segment changes

        """

        for k in range(self.n_agents):
            i_wpt_k = int(self.i_wpt_all[k])
            Xwpt_k = self.Xwpt_all[k]
            Ywpt_k = self.Ywpt_all[k]

            if len(Xwpt_k) <= 1:
                self.psi_route_ref_all[k] = float(self.X_all[k, 2])   # if no waypoints or only 1 waypoint, use current heading as route reference
                continue
                
            i_wpt_k = min(i_wpt_k, len(Xwpt_k) - 1)   # safety check
            x_prev_nmi = float(Xwpt_k[max(0, i_wpt_k - 1 )])
            y_prev_nmi = float(Ywpt_k[max(0, i_wpt_k - 1 )])
            x_next_nmi = float(Xwpt_k[i_wpt_k])
            y_next_nmi = float(Ywpt_k[i_wpt_k])

            dx_m = (x_next_nmi - x_prev_nmi) * NMI
            dy_m = (y_next_nmi - y_prev_nmi) * NMI

            if abs(dx_m) < 1e-12 and abs(dy_m) < 1e-12: 
                self.psi_route_ref_all[k] = float(self.X_all[k, 2])   # if waypoints are effectively the same point, use current heading as route reference
            else: 
                self.psi_route_ref_all[k] = float(np.arctan2(dy_m, dx_m))

    def update_pairwise_geometry_cache(self) -> None:
        """
        Compute pairwise CPA / risk geometry exactly once for the current state and reuse it in observations and reward shaping
            - save computation time by caching pairwise geometry calculations since they are expensive to compute 
            - (involve relative geometry of both current and previous positions for each pair) 
        """

        for a in range(self.n_agents):
            xa, ya = float(self.X_all[a, 0]), float(self.X_all[a, 1])
            xa_prev, ya_prev = float(self.prev_X_all[a, 0]), float(self.prev_X_all[a, 1])

            for b in range(a + 1, self.n_agents):
                xb, yb = float(self.X_all[b, 0]), float(self.X_all[b, 1])
                xb_prev, yb_prev = float(self.prev_X_all[b, 0]), float(self.prev_X_all[b, 1])

                dcpa, tcpa, vrel, alpha, psi_vrel = cpa_calculations(
                    xa, ya, xa_prev, ya_prev,
                    xb, yb, xb_prev, yb_prev,
                    self.dt
                )

                dist = float(np.hypot(xa - xb, ya - yb))
                risk = float(risk_calculations(dcpa, tcpa, dist, vrel))

                self.pair_dcpa[a, b] = self.pair_dcpa[b, a] = float(dcpa)
                self.pair_tcpa[a, b] = self.pair_tcpa[b, a] = float(tcpa)
                self.pair_dist[a, b] = self.pair_dist[b, a] = dist
                self.pair_risk[a, b] = self.pair_risk[b, a] = risk

    # for baseline CORALL evaluation
    def heading_speed_to_discrete_action(self, desired_heading_delta, desired_u_cmd):
        # map heading delta to discrete index (nearest bin)
        h_vals = np.linspace(-1.0, 1.0, self.n_heading)
        s_vals = np.linspace(-1.0, 1.0, self.n_speed)

        h_norm = np.clip(desired_heading_delta / self.max_heading_change, -1.0, 1.0)
        h_idx = int(np.argmin(np.abs(h_vals - h_norm)))

        if self.u_max > self.u_min:
            s_norm = 2.0 * (desired_u_cmd - self.u_min) / (self.u_max - self.u_min) - 1.0
        else: 
            s_norm = 0.0

        s_norm = np.clip(s_norm, -1.0, 1.0)
        s_idx = int(np.argmin(np.abs(s_vals - s_norm)))

        return np.array([h_idx, s_idx], dtype=np.int64)
    
    def get_default_agent_action(self, agent_id: str):
        k = self.agents.index(agent_id)

        return self.heading_speed_to_discrete_action(
            desired_heading_delta=0.0, desired_u_cmd=self.u_des_all[k]
        )
        
    # tried to save time, but will delete 
    def compute_corall_baseline_action(self, agent_id="ship_0"):
            """
            CORALL baseline ownship action:
                psi_ref = psi_wp + Kdir psi_oa, with Kdir = +1 for baseline rule-based mode

            # 1) get current ship state
            # 2) run same CORALL-style waypoint selection + planning + reactive_avoidance
            # 3) form desired heading / speed command from CORALL 
            # 4) map to env discrete action
            """

            k = self.agents.index(agent_id)
            if k != 0: 
                return self.get_default_agent_action(agent_id)
        
            # ownship current state in meters / radians
            x_m = float(self.X_all[k, 0])
            y_m = float(self.X_all[k, 1])
            psi_k = float(self.X_all[k, 2])

            # convert ownship position to nmi for CORALL planner compatibility
            x_nmi = x_m / NMI
            y_nmi = y_m / NMI

            # current ownship waypoint index
            i_wpt = int(self.i_wpt_all[k])

            # 1) CORALL waypoint selection
            i_wpt = waypoint_selection(self.Xwpt_all[k], self.Ywpt_all[k], x_nmi, y_nmi, i_wpt)
            self.i_wpt_all[k] = i_wpt

            # 2) CORALL path planning 
            psi_wp = planning(self.Xwpt_all[k], self.Ywpt_all[k], x_nmi, y_nmi, i_wpt)

            # safety fallback
            if psi_wp is None: 
                psi_wp = float(self.psi_route_ref_all[k])   # if planner fails, fall back to route heading reference
            
            # 3) build obstacle positions in nmi for reactive avoidance
            x_ob_nmi = []
            y_ob_nmi = []

            for j in range(1, self.n_agents):
                x_ob_nmi.append(float(self.X_all[j, 0]) / NMI)
                y_ob_nmi.append(float(self.X_all[j, 1]) / NMI)

            # 4) CORALL reactive avoidance (using Kdir = +1 for baseline rule-based mode)
            psi_oa, w_B, w_R, distance_ob, bearing_ob = reactive_avoidance(
                x_ob_nmi, y_ob_nmi, x_nmi, y_nmi, psi_k, self.t
            )

            # 5) baseline CORALL using Kdir = +1 (direction to turn)
            psi_ref = float(psi_wp + psi_oa)

            # convert desired heading to env discrete action
            desired_heading_delta = self._wrap_angle(psi_ref - psi_k)

            # use nominal ownship cruise speed predefined
            desired_u_cmd = float(self.u_des_all[k])

            return self.heading_speed_to_discrete_action(desired_heading_delta=desired_heading_delta, desired_u_cmd=desired_u_cmd)


    def reset_internal_state(self):
        self.t = 0.0
        self.step_count = 0
        self.done = False

        self.X_all = self.init_from_case()
        self.prev_X_all = self.X_all.copy()

        self.ui_psi1_all = np.zeros(self.n_agents, dtype=float)

        self.Xwpt_all, self.Ywpt_all = self.build_waypoints(self.X_all)

        # make each ship immediately aim for downstream route instead of initial location 
        self.i_wpt_all = np.array(
            [1 if len(self.Xwpt_all[k]) > 1 else 0 for k in range(self.n_agents)], dtype=int
        )

        self.refresh_route_heading_cache()
        self.update_pairwise_geometry_cache()

    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.reset_internal_state()
        observations = {agent: self.get_observation(k) for k, agent in enumerate(self.agents)}
        infos = {agent: {} for agent in self.agents}

        # define metrics dictionary to store episode metrics for each agent, which can be used for logging and evaluation 
        # at the end of the episode in step() when done=True
        self.episode_metrics = {
            agent: {
                "path_length_m": 0.0,
                "min_dcpa_m": float("inf"),
                "risk_exposure": 0.0,
                "collision": 0,
                "success": 0,
                "completion_time_s": np.nan,
                "min_actual_sep_m": float("inf"),
                "near_miss": 0,
                "goal_progress": 0.0,
                "goal_passed": 0,
            } for agent in self.agents
        }

        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]):
        """
        actions: dict {agent_name: MultiDiscrete([heading_idx, speed_idx])}
        
        """

        # Reset at the START of step if previous episode ended
        # This ensures termination signals are NOT paired with reset observations

        # if previous episode ended, reset environment before processing new actions
        self.prev_X_all = self.X_all.copy()


        # 1) decode actions into (psi_ref, u_cmd)
        psi_ref = np.zeros(self.n_agents, dtype=float)
        u_cmd = np.zeros(self.n_agents, dtype=float)

        for k, agent in enumerate(self.agents):
            # skip agents that have already terminated (not in actions dict) - to handle agents dropping out in long episodes
            if agent not in actions:
                continue
                
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

            # CORALL waypoint selection and planner -> use coordinates in nmi 
            Xwpt_k = self.Xwpt_all[k]
            Ywpt_k = self.Ywpt_all[k]
            i_wpt_k = int(self.i_wpt_all[k])

            x_m, y_m, psi_k, r_k, b_k, u_k = self.X_all[k, :]
            x_nmi, y_nmi = x_m / NMI, y_m / NMI

            i_wpt_k = waypoint_selection(Xwpt_k, Ywpt_k, x_nmi, y_nmi, i_wpt_k)
            self.i_wpt_all[k] = i_wpt_k

            psi_wp = planning(Xwpt_k, Ywpt_k, x_nmi, y_nmi, i_wpt_k)

            # safety fallback if the planner returns None
            if psi_wp is None:
                psi_wp = float(self.psi_route_ref_all[k])
            
            psi_ref[k] = float(psi_wp + delta_heading)
        
        # 2) advance dynamics for each ship 
        for k in range(self.n_agents):
            x_m, y_m, psi_k, r_k, b_k, u_k = self.X_all[k, :]

            ui_psi1_k = float(self.ui_psi1_all[k])
            tau_c, v_c, ui_psi1_k = controller(
                psi_ref[k], psi_k, r_k, u_cmd[k], b_k, ui_psi1_k, self.dt
            )
            self.ui_psi1_all[k] = ui_psi1_k

            tau_ac = actuator_modeling(tau_c, sat_amp_s=20)
            inputs = [tau_ac, v_c]

            X_dot = vessel_dynamics(self.X_all[k, :], inputs)
            self.X_all[k, :] = integration(self.X_all[k, :], X_dot, self.dt)

        # 3) update reusable pairwise geometruy cache for the new state
        self.update_pairwise_geometry_cache()

        # 4) rewards / dones
        rewards, terminations, truncations, infos = self.compute_rewards_and_dones()

        # time / truncation handling
        self.t += self.dt
        self.step_count += 1

        if self.t >= self.sim_time:
            # Global time limit: mark ALL agents as truncated uniformly
            # This gives RLlib a clean episode boundary
            for agent in self.agents:
                truncations[agent] = True
            self.done = True

        # if episode ends, attach final episode metrics to infos for logging and evaluation
        if self.done:
            for agent in self.agents:
                infos[agent]["episode_metrics"] = dict(self.episode_metrics[agent])

        
        observations = {agent: self.get_observation(k) for k, agent in enumerate(self.agents)}

        return observations, rewards, terminations, truncations, infos

    # --------------------------
    # Observations
    # --------------------------

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def own_state_features(self, k: int) -> np.ndarray:
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

    
    def get_observation(self, k: int) -> np.ndarray:
        own = self.own_state_features(k)
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

            dcpa_m = float(self.pair_dcpa[k, j])
            tcpa_s = float(self.pair_tcpa[k, j])
            risk = float(self.pair_risk[k, j])

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

    def route_progress(self, k: int) -> Tuple[float, float, float]:
        """
        Progress of ship k along its start->goal route.
        Returns:
            progress_m: projected distance traveled along route from start
            route_len_m: total route length
            dist_to_goal_m: Euclidean distance to final waypoint
        """
        x0_m = float(self.Xwpt_all[k][0]) * NMI
        y0_m = float(self.Ywpt_all[k][0]) * NMI
        xg_m = float(self.Xwpt_all[k][-1]) * NMI
        yg_m = float(self.Ywpt_all[k][-1]) * NMI

        x_k = float(self.X_all[k, 0])
        y_k = float(self.X_all[k, 1])

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

    # -------------------------------
    # Rewards / terminations 
    # -------------------------------

    def compute_rewards_and_dones(self):
        rewards: Dict[str, float] = {}
        # RLlib's old API with PettingZoo wrapper cannot handle per-agent done states
        ## All agents must have uniform termination/truncation flags, where individual agent success/failure tracked in infos instead
        terminations: Dict[str, bool] = {agent: False for agent in self.agents}
        truncations: Dict[str, bool] = {agent: False for agent in self.agents}
        infos: Dict[str, dict] = {agent: {} for agent in self.agents}

        # pairwise risk and DCPA across all agents pre-computed and cached for this step
        pair_dcpa = self.pair_dcpa
        pair_risk = self.pair_risk
        pair_dist = self.pair_dist
        
        LOA = self.LOA_own
        dcpa_safe = LOA * 4.0
        goal_radius = 300.0

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
            step_dist = float(np.hypot(dx_step, dy_step))
            self.episode_metrics[agent]["path_length_m"] += step_dist

            along = float(dp @ t_hat)
            cross = float(dp @ n_hat)

            step_scale = max(1e-6, self.u_max * self.dt)
            along_norm = float(np.clip(along / step_scale, -1.0, 1.0))
            cross_norm = float(np.clip(cross / step_scale, -1.0, 1.0))

            w_along = 1.0
            w_cross = -0.2

            r_along = w_along * along_norm
            r_cross = w_cross * abs(cross_norm)

            # risk penalty: discourage large collision risk with any other vessel
            agent_risks = pair_risk[k].copy()
            agent_risks[k] = 0.0 
            max_risk = float(np.max(agent_risks))
            infos[agent]["max_risk"] = max_risk
            self.episode_metrics[agent]["risk_exposure"] += max_risk * self.dt

            w_risk = -2.0
            r_risk = w_risk * max_risk

            # DCPA penalty: normalized on scale of smallest safe DCPA (absoluate DCPA magnitude)
            dcpa_vals = pair_dcpa[k]
            dcpa_vals = dcpa_vals[np.isfinite(dcpa_vals)]

            if dcpa_vals.size: 
                min_dcpa_abs = float(np.min(np.abs(dcpa_vals)))
            else: 
                min_dcpa_abs = dcpa_safe

            dcpa_def = max(0.0, dcpa_safe - min_dcpa_abs)
            self.episode_metrics[agent]["min_dcpa_m"] = min(self.episode_metrics[agent]["min_dcpa_m"], min_dcpa_abs)

            # optional additional shaping or in place of risk penalty, but risk is more comprehensive 
            # r_dcpa = w_dcpa * (dcpa_def / dcpa_safe)

            # speed penalty for huge speed changes
            u_k = float(self.X_all[k, 5])
            u_des = float(self.u_des_all[k])

            # w_speed = -0.05
            # r_speed = w_speed * ((u_k - u_des) **2) / max(u_des **2, 1e-6)
            infos[agent]["u_des"] = u_des

            # time penalty 
            # r_time = -0.002
            infos[agent]["t"] = self.t

            #total = r_along + r_cross + r_risk + r_dcpa + r_speed + r_time
            total = r_along + r_cross + r_risk

            # collision penalty
            finite_d = np.isfinite(pair_dist[k])
            min_dist = float(np.min(pair_dist[k][finite_d])) if np.any(finite_d) else np.inf
            collision = (min_dist < LOA) 
            infos[agent]["min_dist"] = min_dist
            
            # track actual minimum separation (instantaneous)
            self.episode_metrics[agent]["min_actual_sep_m"] = min(self.episode_metrics[agent]["min_actual_sep_m"], min_dist)
            
            # track near-miss: unsafe distance but not collision (1.5x LOA threshold)
            near_miss_threshold = LOA * 1.5
            if (min_dist >= LOA) and (min_dist < near_miss_threshold) and not collision:
                if self.episode_metrics[agent]["near_miss"] == 0:
                    self.episode_metrics[agent]["near_miss"] = 1

            w_collision = -10.0

            # only apply the collision penalty at the moment of collision, not every step after
            ## scale by risk and speed to encourage risk mitigation and slowing down in high-risk situations (instead of just a flat penalty every step within collision distance)
            if collision: 
                if self.episode_metrics[agent]["collision"] == 0:
                    total += w_collision * (1.0 + max_risk + u_k / max(1.0, u_des))
                else: 
                    total = total 
                self.episode_metrics[agent]["collision"] = 1
                # terminations[agent] = True
                # infos[agent]["collision"] = True

            # success: reached final wp and within radius OR made sufficient route progress
            dist_to_wp = float(np.hypot(wx_m - x_k, wy_m - y_k))
            progress_m, route_len_m, _ = self.route_progress(k)
            goal_progress = float(np.clip(progress_m / max(route_len_m, 1.0), 0.0, 1.0))
            self.episode_metrics[agent]["goal_progress"] = max(self.episode_metrics[agent]["goal_progress"], goal_progress)
            
            # success if: reached final waypoint only
            # (progress tracked separately but doesn't count as success)
            final_reached = (i_wpt_k >= len(Xwpt_k) - 1) and (dist_to_wp <= goal_radius)

            w_success = 40.0

            if final_reached:
                # Track agent success in infos, don't mark as done
                # RLlib's old API needs uniform done flags for all agents
                infos[agent]["success"] = True
                total += w_success                
                self.episode_metrics[agent]["success"] = 1
                self.episode_metrics[agent]["goal_passed"] = 1

                if np.isnan(self.episode_metrics[agent]["completion_time_s"]):
                    self.episode_metrics[agent]["completion_time_s"] = self.t
            else:
                infos[agent]["success"] = False

            rewards[agent] = float(total)

        # NOTE: Episode only ends at global time limit
        # All agents continue running together until sim_time is exceeded
        # Individual agent completions tracked via infos["success"]
        
        
        return rewards, terminations, truncations, infos 


        