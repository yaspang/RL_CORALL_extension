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

# importing CORALL core modules (from repo)
from utils.imazu_cases import get_obstacle_data
from navigation.planning import waypoint_selection, planning
from dynamics.controller import controller
from dynamics.actuator_modeling import actuator_modeling 
from dynamics.vessel_dynamics import vessel_dynamics
from core.integration import integration
from risk_assessment.cpa_calculations import cpa_calculations
from risk_assessment.cpa_calculations_0speed import cpa_calculations_0speed
from risk_assessment.risk_calculations import risk_calculations
from navigation.reactive_avoidance import reactive_avoidance

# future: spatial optimization (trying to implement later to optimize CPA calculations for many agents - see maritime_rl_pkg/maritime_rl/spatial_optimization.py)
from .spatial_optimization import AABBBroadPhase

NMI = 1852.0   # meters per nautical mile

@dataclass
class ObsNorm: 
    """Observation feature bounds (not scaling)"""

    # Position bounds in meters (scenarios ~16km extent)
    pos_max_m: float = 15000.0
    
    # Speed bounds in m/s
    u_max: float = 15.0
    
    # Turn rate bounds in rad/s (typical ship rates)
    r_max: float = 0.5
    
    # Actuator bias bounds (typically normalized)
    b_max: float = 1.0


class MultiShipParallelEnv(ParallelEnv):
    """
    CORALL case-driven multi-agent environment for PPO algorithm

    Observation (agent k):

        own_state (6 features):
            - x (meters), y (meters)
            - sin(psi), cos(psi)
            - r (rad/s) - turn rate
            - u (m/s) - surge speed
            - b - actuator bias
        
        per other ship j != k (4 features each):
            - dx (meters) - relative position
            - dy (meters) - relative position
            - sin(bearing_rel), cos(bearing_rel) - relative bearing

    All features are continuous (no arbitrary discretization).
    """

    metadata = {"render_modes": ["human"], "name": "corall_multiship_mappo_v0"}

    def __init__(
            
        self, 
        case_number: int, 
        dt: float = 0.5, 
        sim_time: float = 1950.0, 
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
        route_len_nmi: float = 2.0,
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
        self.n_heading = int(n_heading)
        self.n_speed = int(n_speed)
    
        self.max_heading_change = np.deg2rad(float(max_heading_change_deg))
        self.u_min = float(u_min)
        self.u_max = float(u_max)

        self.LOA_own = float(loa_m)
        self.norm = obs_norm
    
        self.route_len_nmi = float(route_len_nmi)
        
        # future: spatial optimization
        self.enable_aabb_filtering = bool(enable_aabb_filtering)
        self.aabb_radius_m = float(aabb_radius_m)
    
        self.rng = np.random.default_rng(seed)

        # determine number of agents from CORALL case
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
        # Own state: [x, y, sin(psi), cos(psi), r, u, b]
        own_dim = 7
        # Per other agent: [dx, dy, sin(bearing_rel), cos(bearing_rel), du_rel]
        per_other_dim = 5
        obs_dim = own_dim + (self.n_agents - 1) * per_other_dim
        
        # Bounds for observation space
        low_bounds = np.array(
            [-self.norm.pos_max_m, -self.norm.pos_max_m, -1.0, -1.0, -self.norm.r_max, 0.0, -self.norm.b_max] +
            [-self.norm.pos_max_m, -self.norm.pos_max_m, -1.0, -1.0, -self.norm.u_max] * (self.n_agents - 1),
            dtype=np.float32
        )
        high_bounds = np.array(
            [self.norm.pos_max_m, self.norm.pos_max_m, 1.0, 1.0, self.norm.r_max, self.norm.u_max, self.norm.b_max] +
            [self.norm.pos_max_m, self.norm.pos_max_m, 1.0, 1.0, self.norm.u_max] * (self.n_agents - 1),
            dtype=np.float32
        )
        self._obs_space = {
            agent: spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32) for agent in self.agents
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
        self.current_actions = {}  # Store actions for reward shaping

        # Cache case data for reset
        self._case_cache = dict(Xob=np.array(Xob, dtype=float), 
                                Yob=np.array(Yob, dtype=float), 
                                Vob=np.array(Vob, dtype=float), 
                                psiob=np.array(psiob, dtype=float))
        
        # Initialize dictionaries for evaluation metrics
        self.episode_metrics = {}
        self.prev_done_metrics = {}
        
        # Track previous goal progress for delta-progress reward shaping
        self.prev_goal_progress_all = {}

        # Route / planning data cache
        ## routes are static, straight lines, so will compute downstream route-heading references once and reuse them 
        self.psi_route_ref_all = np.zeros(self.n_agents, dtype=float)

        # Pairwise geometry data cache for observations and rewards
        ## (initialize distances at np.inf so they are not initialized in "dangerous" range before dynamics update)
        self.pair_dcpa = np.full((self.n_agents, self.n_agents), np.inf, dtype=float)
        self.pair_tcpa = np.zeros((self.n_agents, self.n_agents), dtype=float)
        self.pair_dist = np.full((self.n_agents, self.n_agents), np.inf, dtype=float)
        self.pair_risk = np.zeros((self.n_agents, self.n_agents), dtype=float)
    

    # ---------------------------------------------------------------------
    # PettingZoo API
    # ---------------------------------------------------------------------
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
        
        CRITICAL: Obstacle headings are set to point toward intercept waypoints (not Imazu headings),
        ensuring they navigate toward collision courses when center actions are used.
        """

        Xob = self._case_cache["Xob"]
        Yob = self._case_cache["Yob"]
        Vob = self._case_cache["Vob"]
        psiob = self._case_cache["psiob"]

        X_all = np.zeros((self.n_agents, 6), dtype=float)

        # "Ownship" as expressed in CORALL initial state
        ## keep CORALL encounter geometry but set ownship at origin and its speed at same nominal traffic speed scale as other ships in the case 
        ## (for more consistent dynamics across cases and easier learning)

        if len(Vob) > 0:
            u0 = float(np.clip(Vob[0], self.u_min, self.u_max))
        else:
            u0 = float(np.clip(0.5 * (self.u_min + self.u_max), self.u_min, self.u_max))
        
        self.u_des_all[0] = u0
        X_all[0, :] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, u0], dtype=float)

        # Target ship initial states - apply per-case scaling to encounter difficulty
        # Per-case difficulty scaling: now unified at 1.0 since global compression scale (0.35)
        # is applied in imazu_cases.py to all cases, bringing obstacles to realistic close range.
        # All cases now uniformly compressed while preserving relative geometry and crossing angles.
        case_scales = {1: 1.0, 6: 1.0, 21: 1.0}
        scenario_scale = case_scales.get(self.case_number, 1.0)
        
        # Predicted intercept point for waypoint-based navigation (obstacles navigate here)
        route_len_nmi = float(self.route_len_nmi)
        intercept_x_nmi = route_len_nmi / 2.0
        
        for j in range(self.n_obstacles):
            u_j = float(np.clip(Vob[j], self.u_min, self.u_max))
            self.u_des_all[j + 1] = u_j
            
            # Apply scaling to obstacle positions (ownship remains at origin)
            x_obs_m = float(Xob[j] * scenario_scale)
            y_obs_m = float(Yob[j] * scenario_scale)
            
            # CRITICAL FIX: Set heading DIRECTLY TOWARD OWNSHIP (at origin 0,0)
            # This creates true head-on collision courses for realistic training
            # Ownship is always at origin, so vector is simply (0 - x_obs, 0 - y_obs)
            
            x_obs_nmi = x_obs_m / NMI
            y_obs_nmi = y_obs_m / NMI
            
            # Vector from obstacle to ownship (at origin)
            dx_to_ownship = 0.0 - x_obs_nmi  # -x_obs
            dy_to_ownship = 0.0 - y_obs_nmi  # -y_obs
            
            if abs(dx_to_ownship) < 1e-9 and abs(dy_to_ownship) < 1e-9:
                # Obstacle at same position as ownship (degenerate case)
                psi_init = 0.0
            else:
                # Heading that points directly at ownship
                psi_init = float(np.arctan2(dy_to_ownship, dx_to_ownship))
            
            X_all[j + 1, :] = np.array([x_obs_m, y_obs_m, psi_init, 0.0, 0.0, u_j], dtype=float)

        return X_all

    def build_waypoints(self, X_all: np.ndarray) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Build navigation routes in nmi for ALL ships corresponding to initial scenario geometry set by Imazu case
        
        Each agent gets: 
            waypoint 0 = initial position
            waypoint 1 = far point along intended route

        CRITICAL FIX (2026-04-16):
        - Ownship (agent 0): navigates forward along initial heading (collision avoidance target)
        - Obstacles (agents 1+): all navigate toward a common PREDICTED INTERCEPT POINT at scenario center
          This creates geometric convergence (CORALL-style collision geometry) where all ships converge
        """
        # List over ships -> list over consecutive waypoint coordinates
        Xwpt_all: List[List[float]] = []
        Ywpt_all: List[List[float]] = []

        # Use the same route length parameter for every ship 
        route_len_nmi = float(self.route_len_nmi)
        
        # PREDICTED INTERCEPT POINT: center of ownship's expected path
        # Ownship travels from (0,0) to (route_len_nmi, 0), so intercept at midpoint
        intercept_x_nmi = route_len_nmi / 2.0  # Center of ownship's forward path

        for k in range(self.n_agents):
            x_m = float(X_all[k, 0])
            y_m = float(X_all[k, 1])
            psi = float(X_all[k, 2])

            x0_nmi = x_m / NMI
            y0_nmi = y_m / NMI

            if k == 0:
                # Ownship (agent 0): navigate forward along initial heading (straight line forward)
                heading_to_use = psi
            else:
                # Obstacles (agents 1+): compute bearing toward PREDICTED INTERCEPT POINT
                # All obstacles converge at the same engagement zone (center)
                intercept_y_nmi = 0.0
                dx_to_intercept = intercept_x_nmi - x0_nmi
                dy_to_intercept = intercept_y_nmi - y0_nmi
                
                if abs(dx_to_intercept) < 1e-9 and abs(dy_to_intercept) < 1e-9:
                    # Obstacle already at intercept point, use initial heading
                    heading_to_use = psi
                else:
                    # Bearing toward predicted intercept location (creates geometric convergence)
                    heading_to_use = float(np.arctan2(dy_to_intercept, dx_to_intercept))
            
            x1_nmi = x0_nmi + route_len_nmi * float(np.cos(heading_to_use))
            y1_nmi = y0_nmi + route_len_nmi * float(np.sin(heading_to_use))

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
            - For ownship (agent 0): use velocity-based CPA (cpa_calculations_0speed) to handle active maneuvering
            - Uses current state (heading + speed) for velocity prediction -> more responsive to RL commands
            - For obstacles (agents 1+): use projection-based CPA (cpa_calculations) since they follow scripted trajectories
            - Optional: AABB broad-phase filtering to reduce expensive CPA calculations (recommended for 5+ agents)
        """

        # Get pairs to check: either all pairs (naive) or filtered by AABB (optimized)
        if self.enable_aabb_filtering and self.n_agents >= 5:
            # Broad-phase: only check pairs with overlapping AABBs
            pairs_to_check = AABBBroadPhase.get_overlapping_pairs(self.X_all, self.aabb_radius_m)
        else:
            # Naive: check all pairs
            pairs_to_check = [(a, b) for a in range(self.n_agents) for b in range(a + 1, self.n_agents)]

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

            # Use velocity-based CPA for ownship pairs (agent 0 is ownship)
            # This handles active maneuvering better than projection-based CPA
            if a == 0:
                dcpa, tcpa, vrel, alpha, psi_vrel = cpa_calculations_0speed(
                    xa, ya, xb, yb,
                    vxa, vya, vxb, vyb,
                    dist
                )
            else:
                # For non-ownship pairs, use projection-based (shouldn't occur in typical 1+obstacles setup)
                xa_prev, ya_prev = float(self.prev_X_all[a, 0]), float(self.prev_X_all[a, 1])
                xb_prev, yb_prev = float(self.prev_X_all[b, 0]), float(self.prev_X_all[b, 1])
                dcpa, tcpa, vrel, alpha, psi_vrel = cpa_calculations(
                    xa, ya, xa_prev, ya_prev,
                    xb, yb, xb_prev, yb_prev,
                    self.dt
                )

            risk = float(risk_calculations(dcpa, tcpa, dist, vrel))

            self.pair_dcpa[a, b] = self.pair_dcpa[b, a] = float(dcpa)
            self.pair_tcpa[a, b] = self.pair_tcpa[b, a] = float(tcpa)
            self.pair_dist[a, b] = self.pair_dist[b, a] = dist
            self.pair_risk[a, b] = self.pair_risk[b, a] = risk
        
        # For pairs NOT checked (filtered out by AABB), reset to safe defaults
        if self.enable_aabb_filtering and self.n_agents >= 5:
            for a in range(self.n_agents):
                for b in range(a + 1, self.n_agents):
                    if (a, b) not in pairs_to_check:
                        # Far away - set to safe defaults
                        self.pair_dcpa[a, b] = self.pair_dcpa[b, a] = np.inf
                        self.pair_tcpa[a, b] = self.pair_tcpa[b, a] = 0.0
                        self.pair_dist[a, b] = self.pair_dist[b, a] = np.inf
                        self.pair_risk[a, b] = self.pair_risk[b, a] = 0.0


    def reset_internal_state(self):
        """Reinitialize all internal state variables for a new episode"""
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
                "min_tcpa_s": float("inf"),
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
        
        # track which agents have reached their goal (to freeze metrics collection)
        self.agent_reached_goal = {agent: False for agent in self.agents}
        
        # track which agents were truncated in previous step (for handling final observations)
        self.agent_previously_truncated = {agent: False for agent in self.agents}
        
        # Initialize goal progress tracking for all agents (used in delta-progress reward)
        self.prev_goal_progress_all = {agent: 0.0 for agent in self.agents}

        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]):
        """
        actions: dict {agent_name: MultiDiscrete([heading_idx, speed_idx])}
        
        """

        # Store actions for potential reward shaping (e.g., action penalties)
        self.current_actions = actions

        # Reset at the START of step if previous episode ended
        # if previous episode ended, reset environment before processing new actions
        self.prev_X_all = self.X_all.copy()


        # 1) decode actions into (psi_ref, u_cmd)
        psi_ref = np.zeros(self.n_agents, dtype=float)
        u_cmd = np.zeros(self.n_agents, dtype=float)

        for k, agent in enumerate(self.agents):
            # skip agents that have already terminated (not in actions dict) - to handle agents dropping out in long episodes
            if agent not in actions:
                continue
            
            # CRITICAL: Only ownship (k=0) uses waypoint planning and RL actions
            # Obstacles (k>0) propagate forward with FIXED initial heading (obstacle_sim style)
            # This ensures true straight-line collision courses for training
            if k == 0:
                # OWNSHIP: Full waypoint planning + RL action
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
            else:
                # OBSTACLES: Use fixed initial heading (straight-line propagation)
                # This creates deterministic collision courses without replanning
                psi_ref[k] = self.X_all[k, 2]  # Use current heading (maintain)
                u_cmd[k] = self.X_all[k, 5]    # Use current speed (maintain)
        
        # 2) advance dynamics for each ship 
        for k in range(self.n_agents):
            x_m, y_m, psi_k, r_k, b_k, u_k = self.X_all[k, :]

            # ownship controlled by RL -> target obstacles according to fixed speed + heading by kinematics
            ## (Single-agent prelim results) -- WILL CHANGE upon build to MARL
            if k == 0:
                # OWNSHIP: Full dynamics with controller and actuator modeling
                ui_psi1_k = float(self.ui_psi1_all[k])
                tau_c, v_c, ui_psi1_k = controller(
                    psi_ref[k], psi_k, r_k, u_cmd[k], b_k, ui_psi1_k, self.dt
                )
                self.ui_psi1_all[k] = ui_psi1_k

                tau_ac = actuator_modeling(tau_c, sat_amp_s=20)
                inputs = [tau_ac, v_c]

                X_dot = vessel_dynamics(self.X_all[k, :], inputs)
                self.X_all[k, :] = integration(self.X_all[k, :], X_dot, self.dt)
            else:
                # OBSTACLES: Fixed heading + speed (pure kinematics, no controller)
                # This ensures deterministic straight-line collision courses
                psi = psi_ref[k]  # Fixed heading (set above)
                u = u_cmd[k]      # Fixed speed (set above)
                
                # Simple kinematic propagation: x_dot = u * cos(psi), y_dot = u * sin(psi)
                dx_m = u * np.cos(psi) * self.dt
                dy_m = u * np.sin(psi) * self.dt
                
                # Update position only, keep heading and speed constant
                self.X_all[k, 0] += dx_m  # X position
                self.X_all[k, 1] += dy_m  # Y position
                # psi (heading), r (rate), b (balance), u (speed) all remain unchanged

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

        # IMPORTANT: Handle observations for new API stack RLlib
        # RLlib needs ONE final observation for agents being truncated (for value function bootstrapping)
        # but NOT observations for agents already truncated in previous steps
        observations = {}
        for k, agent in enumerate(self.agents):
            is_truncated_this_step = truncations[agent] and not self.agent_previously_truncated[agent]
            is_not_done = not (truncations[agent] or terminations[agent])
            # Return obs for: agents that are active OR agents being truncated this step
            if is_not_done or is_truncated_this_step:
                observations[agent] = self.get_observation(k)
        
        # Filter rewards to match observations (only active agents + newly truncated)
        rewards = {
            agent: r for agent, r in rewards.items() 
            if not (truncations[agent] or terminations[agent]) or (truncations[agent] and not self.agent_previously_truncated[agent])
        }
        
        # Filter infos to match observations (RLlib expects alignment)
        infos = {
            agent: infos[agent] for agent in observations.keys()
        }
        
        # Update tracking for next step
        for agent in self.agents:
            if truncations[agent]:
                self.agent_previously_truncated[agent] = True

        return observations, rewards, terminations, truncations, infos

    # --------------------------
    # Observations
    # --------------------------

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def own_state_features(self, k: int) -> np.ndarray:
        x_m, y_m, psi, r, b, u = self.X_all[k, :]
    
        feats = np.array([
            x_m,
            y_m, 
            np.sin(psi), 
            np.cos(psi), 
            r,
            u,
            b,
        ], dtype=np.float32)

        return feats

    
    def get_observation(self, k: int) -> np.ndarray:
        own = self.own_state_features(k)
        xk_m, yk_m, psik, uk = self.X_all[k, 0], self.X_all[k, 1], self.X_all[k, 2], self.X_all[k, 5]

        per_other = []
        for j in range(self.n_agents):
            if j == k:
                continue
            
            xj_m, yj_m, psij, uj = self.X_all[j, 0], self.X_all[j, 1], self.X_all[j, 2], self.X_all[j, 5]
            dx_m = xj_m - xk_m
            dy_m = yj_m - yk_m

            # Relative bearing from ownship k to other ship j
            bearing = float(np.arctan2(dy_m, dx_m))
            bearing_rel = self._wrap_angle(bearing - psik)
            sin_b, cos_b = np.sin(bearing_rel), np.cos(bearing_rel)
            
            # Relative speed: speed of agent j relative to agent k
            du_rel = float(uj - uk)

            vec = np.array([
                dx_m,
                dy_m,
                sin_b,
                cos_b,
                du_rel,
            ], dtype=np.float32)

            per_other.append(vec)
        
        obs = np.concatenate([own] + per_other, axis=0).astype(np.float32)

        # Safety: if NaNs appear, replace with zero (shouldn't happen with continuous features)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

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
        pair_tcpa = self.pair_tcpa
        pair_risk = self.pair_risk
        pair_dist = self.pair_dist

        
        LOA = self.LOA_own
        dcpa_safe = LOA * 4.0
        goal_radius = LOA * 2.0

        # Agent-by-agent reward computation and metrics tracking
        ## Once an agent reaches its final waypoint, we FREEZE its metrics.
        ## This prevents spurious metric pollution from post-goal behavior (e.g., drift, loiter, etc.)
        
        for k, agent in enumerate(self.agents):
            # waypoint following: reward along-track movement, penalize cross-track
            Xwpt_k = self.Xwpt_all[k]
            Ywpt_k = self.Ywpt_all[k]
            i_wpt_k = int(self.i_wpt_all[k])

            x_k, y_k = float(self.X_all[k, 0]), float(self.X_all[k, 1])
            x_nmi, y_nmi = x_k / NMI, y_k / NMI
            wx_nmi, wy_nmi = float(Xwpt_k[i_wpt_k]), float(Ywpt_k[i_wpt_k])
            wx_m, wy_m = wx_nmi * NMI, wy_nmi * NMI

            # Check if agent already reached goal - if so, skip metrics collection for this agent
            agent_already_done = self.agent_reached_goal[agent]

            # === REWARD FUNCTION ===
            # 6 terms: waypoint progress + risk penalty + near-miss penalty + collision penalty + time penalty + success bonus
            total = 0.0

            if not agent_already_done:
                psi_des = float(np.arctan2(wy_m - y_k, wx_m - x_k))
                t_hat = np.array([np.cos(psi_des), np.sin(psi_des)], dtype=float)
                n_hat = np.array([-np.sin(psi_des), np.cos(psi_des)], dtype=float)

                dx_step = float(x_k - self.prev_X_all[k, 0])
                dy_step = float(y_k - self.prev_X_all[k, 1])
                dp = np.array([dx_step, dy_step], dtype=float)
                step_dist = float(np.hypot(dx_step, dy_step))
                self.episode_metrics[agent]["path_length_m"] += step_dist
                
                # ========== OPTION 1: Delta Progress Reward (Waypoint-Based) ==========
                # Reward based on progress towards goal (route completion), not just displacement
                progress_m, route_len_m, _ = self.route_progress(k)
                goal_progress = float(np.clip(progress_m / max(route_len_m, 1.0), 0.0, 1.0))
                
                # Delta progress: how much closer to goal this step?
                prev_progress = self.prev_goal_progress_all.get(agent, 0.0)
                delta_progress = goal_progress - prev_progress
                self.prev_goal_progress_all[agent] = goal_progress
                
                # SPARSE REWARD ONLY: Delta progress for milestone rewards
                # Avoid continuous progress reward - it competes with risk penalty and causes aggressive behavior
                w_along = 1.0
                r_along = w_along * delta_progress                
            else:
                # Agent already reached goal - no waypoint following reward
                r_along = 0.0
                r_cross = 0.0

            # risk penalty: discourage large collision risk with any other vessel
            agent_risks = pair_risk[k].copy()
            agent_risks[k] = 0.0 
            max_risk = float(np.max(agent_risks))
            infos[agent]["max_risk"] = max_risk
            
            w_risk = -100.0
            total += w_risk * max_risk
        

            # only update metrics if agent hasn't reached goal yet
            if not agent_already_done:
                self.episode_metrics[agent]["risk_exposure"] += max_risk * self.dt

            # TCPA tracking: record minimum TCPA over episode (only future encounters, positive TCPA)
            tcpa_vals = pair_tcpa[k]
            tcpa_vals_finite = tcpa_vals[np.isfinite(tcpa_vals)]
            tcpa_vals_positive = tcpa_vals_finite[tcpa_vals_finite > 0.0]  # Only future CPA (positive TCPA)
            
            if tcpa_vals_positive.size:
                min_tcpa_abs = float(np.min(tcpa_vals_positive))
            else:
                min_tcpa_abs = float("inf")
            
            if not agent_already_done:
                self.episode_metrics[agent]["min_tcpa_s"] = min(self.episode_metrics[agent]["min_tcpa_s"], min_tcpa_abs)

            infos[agent]["t"] = self.t

            # separation margin reward: bonus to maintain safe distance
            safe_dist_m = LOA * 3.0
            min_dist = float(np.min(pair_dist[k][np.isfinite(pair_dist[k])])) if np.any(np.isfinite(pair_dist[k])) else np.inf
            if min_dist > safe_dist_m: 
                # good distance maintained -> reward
                w_safe = 2.9
                r_separation = w_safe * (1.0 - (min_dist - safe_dist_m) / 5000.0)
                total += r_separation
            elif min_dist > LOA and min_dist <= safe_dist_m:
                # Warning zone: penalty scales with proximity
                w_warning = -15.0
                r_separation = w_warning * (1.0 - (min_dist - LOA) / safe_dist_m)
                total += r_separation

                        
            # Collision penalty: large negative reward for actual collision
            # If any collision occurred this step, apply heavy penalty 
            finite_d = np.isfinite(pair_dist[k])
            min_dist = float(np.min(pair_dist[k][finite_d])) if np.any(finite_d) else np.inf
            collision = (min_dist < LOA)
            if collision:
                if self.episode_metrics[agent]["collision"] == 0:
                    self.episode_metrics[agent]["collision"] = 1
            
            # COLLISION PENALTY: Large negative reward for actual collision
            if collision:
                w_collision = -200.0  # Massive penalty for collision
                total += w_collision
            
            
            # Time penalty: small per-step cost to encourage faster completion
            w_time = -0.01
            total += w_time
            

            # Success: reached final wp and within radius 
            dist_to_wp = float(np.hypot(wx_m - x_k, wy_m - y_k))
            progress_m, route_len_m, _ = self.route_progress(k)
            goal_progress = float(np.clip(progress_m / max(route_len_m, 1.0), 0.0, 1.0))
            
            # update goal progress only if agent hasn't reached goal yet
            if not agent_already_done:
                self.episode_metrics[agent]["goal_progress"] = max(self.episode_metrics[agent]["goal_progress"], goal_progress)
            
            # success if: reached final waypoint using waypoint_selection() built-in threshold (~200m from CORALL planning.py)
            final_waypoint_reached_by_index = (i_wpt_k >= len(Xwpt_k) - 1)
            
            if final_waypoint_reached_by_index and len(Xwpt_k) > 1:
                # Verify agent is actually close to the final waypoint
                xf_nmi = float(Xwpt_k[-1])
                yf_nmi = float(Ywpt_k[-1])
                dist_to_goal_nmi = np.hypot(xf_nmi - x_nmi, yf_nmi - y_nmi)
                final_reached = (dist_to_goal_nmi < 200.0 / 1852.0)  # ~200m acceptance radius (from CORALL)
            else:
                final_reached = final_waypoint_reached_by_index

            w_success = 50.0

            if final_reached:
                # Track agent success in infos
                infos[agent]["success"] = True
                total += w_success  # ENABLED: reward agents for reaching destination
                self.episode_metrics[agent]["success"] = 1
                self.episode_metrics[agent]["goal_passed"] = 1
                
                # Mark agent as reached goal - stop collecting metrics for this agent
                self.agent_reached_goal[agent] = True

                if np.isnan(self.episode_metrics[agent]["completion_time_s"]):
                    self.episode_metrics[agent]["completion_time_s"] = self.t
                
                # Mark agent as truncated (agent-level done) when reaching goal
                # Note: Use truncations, NOT terminations - terminations end the global episode for all agents
                # which breaks RLlib's trajectory handling in multi-agent environments
                truncations[agent] = True
            else:
                infos[agent]["success"] = False
            
            # Track collision metrics (for analysis, not reward shaping)
            finite_d = np.isfinite(pair_dist[k])
            min_dist = float(np.min(pair_dist[k][finite_d])) if np.any(finite_d) else np.inf
            collision = (min_dist < LOA) 
            infos[agent]["min_dist"] = min_dist
            
            if not agent_already_done:
                self.episode_metrics[agent]["min_actual_sep_m"] = min(self.episode_metrics[agent]["min_actual_sep_m"], min_dist)
                near_miss_threshold = LOA * 1.5
                if (min_dist >= LOA) and (min_dist < near_miss_threshold) and not collision:
                    if self.episode_metrics[agent]["near_miss"] == 0:
                        self.episode_metrics[agent]["near_miss"] = 1
            
        

            rewards[agent] = float(total)

        # Note: Episode only ends at global simulation time limit and all agents continue running together until sim_time is exceeded
        return rewards, terminations, truncations, infos 


        