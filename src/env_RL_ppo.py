"""
Multi-agent (PettingZoo ParallelEnv) for PPO training compatible with CORALL simulation base.
-> Current set-up is for single-agent training, but uses PettingZoo multi-agent interface to allow for future multi-agent training.

Every vessel in each of the Imazu cases is an agent -> in the frame of CORALL
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

# Ensure CORALL repository relative imports resolve
from .path_setup import ensure_paths
ensure_paths()

# Importing CORALL core modules (from repo)
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

NMI = 1852.0   # meters per nautical mile

@dataclass
class ObsNorm: 
    """Observation feature bounds"""
    # Position bounds in meters (arbitrarily large to avoid clipping, but not too large to avoid numerical issues)
    pos_max_m: float = 15000.0
    
    # Velocity bounds in m/s (typical ship speeds up to ~15 m/s)
    vel_max: float = 15.0
    
    # Turn rate bounds in rad/s (typical ship rates)
    r_max: float = 0.5
    
    # Actuator bias bounds (typically normalized)
    b_max: float = 1.0


class MultiShipParallelEnv(ParallelEnv):
    """
    CORALL multi-agent environment for PPO algorithm

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
            - du_x (meters / s) - relative velocity (x-component)
            - du_y (meters / s) - relative velocity (y-component)

    All features are continuous (no arbitrary discretization).
    """

    metadata = {"render_modes": ["human"], "name": "corall_multiship_ppo_v0"}

    def __init__(
            
        self, 
        case_number: int, 
        dt: float = 0.5, 
        sim_time: float = 900.0, 
        render_mode: Optional[str] = None, 
        # Heading discretization: n_heading bins, ±max_heading_change_deg offset from waypoint bearing
        n_heading: int = 7,
        max_heading_change_deg: float = 25.0,
        # Speed discretization: n_speed bins spanning 7.0–12.0 m/s absolute
        n_speed: int = 5,
        # Ship geometry for collision logic 
        loa_m: float = 30.0,
        # Observation normalization
        obs_norm: ObsNorm = ObsNorm(),
        # Waypoint generation parameter
        route_len_nmi: float = 2.0,

        seed: Optional[int] = None,
        
        # Geometry parameters for encounter placement and speeds
        desired_cross_x_nmi: float = 1.0,
        target_speed_mps: float = 10.0,
        ownship_speed_mps: Optional[float] = None,
    ): 

        super().__init__()
        self.case_number = int(case_number)
        self.dt = float(dt)
        self.sim_time = float(sim_time)
        self.render_mode = render_mode
        self.n_heading = int(n_heading)
        self.max_heading_change = np.deg2rad(float(max_heading_change_deg))
        # Speed discretization: n_speed bins from speed_min_frac to 1.0 of nominal
        self.n_speed = int(n_speed)
        self.speed_options_mps = np.linspace(7.0, 12.0, self.n_speed)  # [7.0, 8.25, 9.5, 10.75, 12.0] m/s

        self.LOA_own = float(loa_m)
        self.norm = obs_norm
    
        self.route_len_nmi = float(route_len_nmi)
    
        self.rng = np.random.default_rng(seed)

        # Determine number of ships present in env from CORALL case
        # Pass geometry parameters to customize encounter placement and speeds
        Xob, Yob, Vob, psiob = get_obstacle_data(
            self.case_number,
            desired_cross_x_nmi=desired_cross_x_nmi,
            target_speed_mps=target_speed_mps,
            ownship_speed_mps=ownship_speed_mps,
            synchronize_arrivals=True,
            min_speed_mps=7.0,
            max_speed_mps=12.0,
        )
        self.n_obstacles = len(Xob)
        self.n_agents = 1 + self.n_obstacles
        
        # Store obstacle data for later use in init_from_case()
        self._case_cache = {"Xob": Xob, "Yob": Yob, "Vob": Vob, "psiob": psiob}
        
        # Override ownship speed if specified, otherwise will inherit from first obstacle in init_from_case()
        self.ownship_speed_mps_override = ownship_speed_mps
        
        self.reward_weights = {
            'progress':  200.0,
            'collision': -10000.0,
        }
    
        self.agents = [f"ship_{i}" for i in range(self.n_agents)]
        self.possible_agents = self.agents[:]
    
        # Action Spaces: MultiDiscrete([n_heading, n_speed]) — heading + speed control
        self._action_space = {
            agent: spaces.MultiDiscrete([self.n_heading, self.n_speed]) for agent in self.agents
        }
    
        # Observation Spaces: same for all n_agents
        # Own state: [x, y, sin(psi), cos(psi), r, u_x, u_y, b, sin(goal_bearing), cos(goal_bearing), goal_distance_norm] (11 features)
            # u_x = u * cos(psi), u_y = u * sin(psi) for velocity components
            # goal_bearing = angle from ownship to final goal waypoint
            # goal_distance_norm = normalized distance to final goal (clipped to [-1, 1])
        own_dim = 11
        # Per other agent: [dx, dy, sin(bearing_rel), cos(bearing_rel), du_x, du_y] (6 features)
        per_other_dim = 6
        obs_dim = own_dim + (self.n_agents - 1) * per_other_dim
        
        # Bounds for observation space
        # Own state bounds: [x, y, sin(psi), cos(psi), r, u_x, u_y, b, sin(goal_bearing), cos(goal_bearing), goal_distance_norm]
        own_low = np.array([-self.norm.pos_max_m, -self.norm.pos_max_m, -1.0, -1.0, -self.norm.r_max, -self.norm.vel_max, -self.norm.vel_max, -self.norm.b_max, -1.0, -1.0, -1.0], dtype=np.float32)
        own_high = np.array([self.norm.pos_max_m, self.norm.pos_max_m, 1.0, 1.0, self.norm.r_max, self.norm.vel_max, self.norm.vel_max, self.norm.b_max, 1.0, 1.0, 1.0], dtype=np.float32)
        
        # Per-other bounds: [dx, dy, sin(bearing), cos(bearing), du_x, du_y]
        per_other_low = np.array([-self.norm.pos_max_m, -self.norm.pos_max_m, -1.0, -1.0, -self.norm.vel_max, -self.norm.vel_max], dtype=np.float32)
        per_other_high = np.array([self.norm.pos_max_m, self.norm.pos_max_m, 1.0, 1.0, self.norm.vel_max, self.norm.vel_max], dtype=np.float32)
        
        low_bounds = np.concatenate([own_low] + [per_other_low] * (self.n_agents - 1))
        high_bounds = np.concatenate([own_high] + [per_other_high] * (self.n_agents - 1))
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
        Obstacle headings use original Imazu angles; positions and speeds are scaled by imazu_cases.get_obstacle_data().
        """

        Xob = self._case_cache["Xob"]
        Yob = self._case_cache["Yob"]
        Vob = self._case_cache["Vob"]
        psiob = self._case_cache["psiob"]

        X_all = np.zeros((self.n_agents, 6), dtype=float)

        # "Ownship" as expressed in CORALL initial state
        ## keep CORALL encounter geometry but set ownship at origin and its speed at same nominal traffic speed scale as other ships in the case 
        ## (for more consistent dynamics across cases and easier learning)

        # Ownship uses first obstacle velocity as cruise speed (or default 9.5 m/s)
        if self.ownship_speed_mps_override is not None:
            # Use overridden ownship speed if provided
            u0 = float(self.ownship_speed_mps_override)
        elif len(Vob) > 0:
            # Otherwise inherit from first obstacle
            u0 = float(Vob[0])
        else:
            u0 = 9.5  # Default cruise speed in m/s
        
        self.u_des_all[0] = u0
        X_all[0, :] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, u0], dtype=float)

        # Predicted intercept point for waypoint-based navigation (obstacles navigate here)
        route_len_nmi = float(self.route_len_nmi)
        
        for j in range(self.n_obstacles):
            u_j = float(Vob[j])
            self.u_des_all[j + 1] = u_j

            x_obs_m = float(Xob[j])
            y_obs_m = float(Yob[j])

            # original imazu headings
            psi_init = float(psiob[j])

            X_all[j + 1, :] = np.array(
                [x_obs_m, y_obs_m, psi_init, 0.0, 0.0, u_j],
                dtype=float
            )

        return X_all

    def build_waypoints(self, X_all: np.ndarray) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Build navigation routes in nmi for ALL ships corresponding to initial scenario geometry set by Imazu case
        
        Each agent gets: 
            waypoint 0 = initial position
            waypoint 1 = far point along intended route

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

            # all ships use current / original heading
            heading = psi
            x1_nmi = x0_nmi + route_len_nmi * float(np.cos(heading))
            y1_nmi = y0_nmi + route_len_nmi * float(np.sin(heading))
            
            x1_nmi = x0_nmi + route_len_nmi * float(np.cos(heading))
            y1_nmi = y0_nmi + route_len_nmi * float(np.sin(heading))

            Xwpt_all.append([x0_nmi, x1_nmi])
            Ywpt_all.append([y0_nmi, y1_nmi])

        return Xwpt_all, Ywpt_all

    def refresh_route_heading_cache(self) -> None:
        """
        Precompute downstream route-heading for each ship based on current active waypoint segment. 

        Current CORALL cases have simple straight line routes, so desired heading goal is effectively constant for whole episode. 
            -> Save computation time by caching route-heading references and only updating if active waypoint segment changes

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
            - Optional add: AABB broad-phase filtering to reduce expensive CPA calculations (5+ agents)
        """
        pairs_to_check = [(a, b) for a in range(self.n_agents) for b in range(a + 1, self.n_agents)]

        # Compute CPA for all the ownship-target ship pairs in the the current state
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

            risk = float(risk_calculations(
                dcpa / 1852.0,    # meters → NMI
                tcpa / 3600.0,    # seconds → hours
                dist / 1852.0,    # meters → NMI
                vrel,
            ))

            self.pair_dcpa[a, b] = self.pair_dcpa[b, a] = float(dcpa)
            self.pair_tcpa[a, b] = self.pair_tcpa[b, a] = float(tcpa)
            self.pair_dist[a, b] = self.pair_dist[b, a] = dist
            self.pair_risk[a, b] = self.pair_risk[b, a] = risk


    def reset_internal_state(self):
        """Reinitialize all internal state variables for a new episode"""
        self.t = 0.0
        self.step_count = 0
        self.done = False

        self.X_all = self.init_from_case()
        self.prev_X_all = self.X_all.copy()

        self.ui_psi1_all = np.zeros(self.n_agents, dtype=float)

        self.Xwpt_all, self.Ywpt_all = self.build_waypoints(self.X_all)

        # Make each ship immediately aim for downstream route (second waypoint at start of episode)
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

        # Define metrics dictionary to store episode metrics for each agent, which can be used for logging and evaluation 
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
                "min_dcpa_m": float("inf"),
                "near_miss": 0,
                "goal_progress": 0.0,
                "goal_passed": 0,
            } for agent in self.agents
        }
        
        # Track which agents have reached their goal (to freeze metrics collection)
        self.agent_reached_goal = {agent: False for agent in self.agents}
        
        # Track which agents were truncated in previous step (for handling final observations)
        self.agent_previously_truncated = {agent: False for agent in self.agents}
        
        # Track which agents were terminated in previous step (for handling final observations)
        self.agent_previously_terminated = {agent: False for agent in self.agents}
        
        # Initialize goal progress tracking for all agents (used in delta-progress reward)
        self.prev_goal_progress_all = {agent: 0.0 for agent in self.agents}

        return observations, infos

    # -------------------------------
    # Step Function
    # -------------------------------
    

    def step(self, actions: Dict[str, np.ndarray]):
        """
        Step the environment forward by one time step using the provided actions.

        Arguments
            self: MultiShipParallelEnv instance
            actions: dict {agent_name: MultiDiscrete([heading_idx, speed_idx])}


        Returns
            observations: dict {agent_name: observation_array}
            rewards: dict {agent_name: reward_value}
            terminations: dict {agent_name: bool}
            truncations: dict {agent_name: bool}
            infos: dict {agent_name: info_dict}
        """

        # Store actions for potential reward shaping (e.g., action penalties)
        self.current_actions = actions

        # Reset at the START of step and reset environment if previous episode ended (before new action)
        self.prev_X_all = self.X_all.copy()

        # 1) Decode actions into (psi_ref, speed_cmd)
        psi_ref = np.zeros(self.n_agents, dtype=float)
        speed_cmd_all = self.u_des_all.copy()  # default: nominal speed for each agent
        heading_idx = self.n_heading // 2      # default if ship_0 not in actions
        speed_idx   = self.n_speed - 1         # default: max speed

        for k, agent in enumerate(self.agents):
            # Skip agents that have already terminated (not in actions dict) - to handle agents dropping out in long episodes
            if agent not in actions:
                continue
            
            # Only ownship (k=0) uses waypoint planning and RL actions
            # Obstacle vessels: (k>0) propagate forward with FIXED initial heading 
            if k == 0:
                # OWNSHIP: Full waypoint planning + RL action (heading + speed)
                action_arr = np.asarray(actions[agent], dtype=int).flatten()
                heading_idx = int(np.clip(action_arr[0], 0, self.n_heading - 1))
                speed_idx   = int(np.clip(action_arr[1], 0, self.n_speed - 1))

                # Commanded speed = absolute m/s from discrete speed bin
                speed_cmd_all[0] = self.speed_options_mps[speed_idx]

                # Normalize heading index to [-1, 1] scale
                if self.n_heading > 1:
                    delta_heading_norm = -1.0 + 2.0 * heading_idx / (self.n_heading - 1)
                else: 
                    delta_heading_norm = 0.0

                # Heading change action scaled to max_heading_change (radians)
                delta_heading = delta_heading_norm * self.max_heading_change

                # CORALL waypoint selection and planner (CORALL coordinates in nmi, but state is in meters)
                Xwpt_k = self.Xwpt_all[k]
                Ywpt_k = self.Ywpt_all[k]
                i_wpt_k = int(self.i_wpt_all[k])

                x_m, y_m, psi_k, r_k, b_k, u_k = self.X_all[k, :]
                x_nmi, y_nmi = x_m / NMI, y_m / NMI

                i_wpt_k = waypoint_selection(Xwpt_k, Ywpt_k, x_nmi, y_nmi, i_wpt_k)
                self.i_wpt_all[k] = i_wpt_k

                psi_wp = planning(Xwpt_k, Ywpt_k, x_nmi, y_nmi, i_wpt_k)

                # Safety fallback if the planner returns None
                if psi_wp is None:
                    psi_wp = float(self.psi_route_ref_all[k])
                
                psi_ref[k] = float(psi_wp + delta_heading)
            else:
                # OBSTACLE vessels: Use fixed initial heading (straight-line propagation)
                psi_ref[k] = self.X_all[k, 2]  # Use current heading (maintain)
        
        # 2) Advance dynamics for each ship 
        for k in range(self.n_agents):
            x_m, y_m, psi_k, r_k, b_k, u_k = self.X_all[k, :]

            # Ownship controlled by RL; obstacles propagate with fixed heading and speed
            if k == 0:
                # OWNSHIP: Apply commanded speed then full dynamics with heading controller
                self.X_all[0, 5] = speed_cmd_all[0]  # override surge speed with discrete command
                x_m, y_m, psi_k, r_k, b_k, u_k = self.X_all[k, :]
                ui_psi1_k = float(self.ui_psi1_all[k])
                # Use current surge speed (u_k) instead of speed command
                tau_c, v_c, ui_psi1_k = controller(
                    psi_ref[k], psi_k, r_k, u_k, b_k, ui_psi1_k, self.dt
                )
                self.ui_psi1_all[k] = ui_psi1_k

                tau_ac = actuator_modeling(tau_c, sat_amp_s=20)
                inputs = [tau_ac, v_c]

                X_dot = vessel_dynamics(self.X_all[k, :], inputs)
                self.X_all[k, :] = integration(self.X_all[k, :], X_dot, self.dt)
            else:
                # OBSTACLES: Fixed heading + speed (pure kinematics, no controller)
                    # This ensures deterministic straight-line collision courses
                psi = psi_ref[k]      # Fixed heading (set above)
                u = self.X_all[k, 5]  # Fixed speed (current surge speed)
                
                # Simple kinematic propagation: x_dot = u * cos(psi), y_dot = u * sin(psi)
                dx_m = u * np.cos(psi) * self.dt
                dy_m = u * np.sin(psi) * self.dt
                
                # Update position only, keep heading and speed constant
                self.X_all[k, 0] += dx_m  # X position
                self.X_all[k, 1] += dy_m  # Y position
                # psi (heading), r (rate), b (balance), u (speed) all remain unchanged

        # 3) Update reusable pairwise geometry cache for the new state (save time)
        self.update_pairwise_geometry_cache()

        # 4) Rewards / Dones
        rewards, terminations, truncations, infos = self.compute_rewards_and_dones()

        # Log ownship action info for speed-usage diagnostics
        if "ship_0" in infos:
            infos["ship_0"]["heading_idx"]   = int(heading_idx)
            infos["ship_0"]["speed_idx"]     = int(speed_idx)
            infos["ship_0"]["speed_cmd_mps"] = float(speed_cmd_all[0])
            infos["ship_0"]["speed_fraction"] = float(speed_cmd_all[0] / self.u_des_all[0]) if self.u_des_all[0] > 0 else 1.0

        # Time / truncation handling
        self.t += self.dt
        self.step_count += 1

        if self.t >= self.sim_time:
            # Global time limit: truncate all agents uniformly
            for agent in self.agents:
                truncations[agent] = True
            self.done = True

        # If episode ends, attach final episode metrics to infos for logging and evaluation
        if self.done:
            for agent in self.agents:
                infos[agent]["episode_metrics"] = dict(self.episode_metrics[agent])

        # Return one final observation for agents being terminated/truncated this step;
            # Skip agents already done in a prior step.
        observations = {}
        for k, agent in enumerate(self.agents):
            is_terminated_this_step = terminations[agent] and not self.agent_previously_terminated[agent]
            is_truncated_this_step = truncations[agent] and not self.agent_previously_truncated[agent]
            is_not_done = not (truncations[agent] or terminations[agent])
            # Return obs for: agents that are active OR agents being terminated/truncated this step
            if is_not_done or is_terminated_this_step or is_truncated_this_step:
                observations[agent] = self.get_observation(k)
        
        # Filter rewards to match observations (active + newly done agents only)
        rewards_filtered = {}
        for agent, r in rewards.items():
            is_terminated_this_step = terminations[agent] and not self.agent_previously_terminated[agent]
            is_truncated_this_step = truncations[agent] and not self.agent_previously_truncated[agent]
            is_not_done = not (truncations[agent] or terminations[agent])
            if is_not_done or is_terminated_this_step or is_truncated_this_step:
                rewards_filtered[agent] = r
        rewards = rewards_filtered
        
        # Filter infos to match observations
        infos = {
            agent: infos[agent] for agent in observations.keys()
        }
        
        # Update tracking for next step
        for agent in self.agents:
            if truncations[agent]:
                self.agent_previously_truncated[agent] = True
            if terminations[agent]:
                self.agent_previously_terminated[agent] = True

        return observations, rewards, terminations, truncations, infos

    # --------------------------
    # Observations
    # --------------------------

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def own_state_features(self, k: int) -> np.ndarray:
        """
        Handles normalization of own state features contributing to normalized observation space (necessary for learning stability)
        """
        x_m, y_m, psi, r, b, u = self.X_all[k, :]
        
        # Decompose surge speed into x and y velocity components
        u_x = u * np.cos(psi)
        u_y = u * np.sin(psi)
        
        # NORMALIZATION: Clip all raw values to [-1, 1] scale using normalization bounds
        x_norm = np.clip(x_m / self.norm.pos_max_m, -1.0, 1.0)
        y_norm = np.clip(y_m / self.norm.pos_max_m, -1.0, 1.0)
        r_norm = np.clip(r / self.norm.r_max, -1.0, 1.0)
        u_x_norm = np.clip(u_x / self.norm.vel_max, -1.0, 1.0)
        u_y_norm = np.clip(u_y / self.norm.vel_max, -1.0, 1.0)
        b_norm = np.clip(b / self.norm.b_max, -1.0, 1.0)
    
        feats = np.array([
            x_norm,
            y_norm, 
            np.sin(psi), 
            np.cos(psi), 
            r_norm,
            u_x_norm,
            u_y_norm,
            b_norm,
        ], dtype=np.float32)

        return feats

    
    def get_observation(self, k: int) -> np.ndarray:
        """
        Get observation for agent k of FIXED SIZE (11 own + 18 for max 3 obstacles = 29 dims).
        
        This ensures consistent observation size regardless of actual obstacle count.
        Unused obstacle slots are zero-filled but maintain consistent positions.
        
        Structure (always 29 dims):
        - Dims 0-7: own state (x, y, sin(psi), cos(psi), r, u_x, u_y, b) - normalized
        - Dims 8-10: goal information (sin(goal_bearing), cos(goal_bearing), goal_distance_norm) - normalized
        - Dims 11-16: obstacle 0 (dx, dy, sin(bearing), cos(bearing), du_x, du_y) - normalized
        - Dims 17-22: obstacle 1 (6 dims) - zero-padded if not present
        - Dims 23-28: obstacle 2 (6 dims) - zero-padded if not present
        """
        own = self.own_state_features(k)  # 8 dims
        xk_m, yk_m, psik, uk = self.X_all[k, 0], self.X_all[k, 1], self.X_all[k, 2], self.X_all[k, 5]
        
        # Own surge speed components
        uk_x = uk * np.cos(psik)
        uk_y = uk * np.sin(psik)
        
        # Calculate bearing from ownship to final goal waypoint
        xg_m = float(self.Xwpt_all[k][-1]) * NMI
        yg_m = float(self.Ywpt_all[k][-1]) * NMI
        goal_dx_m = xg_m - xk_m
        goal_dy_m = yg_m - yk_m
        goal_bearing = float(np.arctan2(goal_dy_m, goal_dx_m))
        goal_bearing_rel = self._wrap_angle(goal_bearing - psik)
        sin_goal_b = np.sin(goal_bearing_rel)
        cos_goal_b = np.cos(goal_bearing_rel)
        
        # Goal distance (normalized)
        goal_dist_m = float(np.hypot(goal_dx_m, goal_dy_m))
        goal_dist_norm = np.clip(goal_dist_m / self.norm.pos_max_m, -1.0, 1.0)
        
        goal_features = np.array([sin_goal_b, cos_goal_b, goal_dist_norm], dtype=np.float32)

        # Collect ALL obstacles (up to 3)
        per_other = []
        obstacle_idx = 0
        
        for j in range(self.n_agents):
            if j == k:
                continue
            
            # Only process up to 3 obstacles (curriculum supports max 4-ship = 3 obstacles)
            if obstacle_idx >= 3:
                break
            
            xj_m, yj_m, psij, uj = self.X_all[j, 0], self.X_all[j, 1], self.X_all[j, 2], self.X_all[j, 5]
            dx_m = xj_m - xk_m
            dy_m = yj_m - yk_m

            # Relative bearing from ownship k to other ship j
            bearing = float(np.arctan2(dy_m, dx_m))
            bearing_rel = self._wrap_angle(bearing - psik)
            sin_b, cos_b = np.sin(bearing_rel), np.cos(bearing_rel)
            
            # Relative velocity components
            uj_x = uj * np.cos(psij)
            uj_y = uj * np.sin(psij)
            du_x = uj_x - uk_x
            du_y = uj_y - uk_y
            
            # NORMALIZATION: Clip relative position and velocity to [-1, 1]
            dx_norm = np.clip(dx_m / self.norm.pos_max_m, -1.0, 1.0)
            dy_norm = np.clip(dy_m / self.norm.pos_max_m, -1.0, 1.0)
            du_x_norm = np.clip(du_x / self.norm.vel_max, -1.0, 1.0)
            du_y_norm = np.clip(du_y / self.norm.vel_max, -1.0, 1.0)

            vec = np.array([
                dx_norm,
                dy_norm,
                sin_b,
                cos_b,
                du_x_norm,
                du_y_norm,
            ], dtype=np.float32)

            per_other.append(vec)
            obstacle_idx += 1
        
        # Pad with zero obstacles to reach exactly 3 obstacles (18 dims)
        while obstacle_idx < 3:
            per_other.append(np.zeros(6, dtype=np.float32))
            obstacle_idx += 1
        
        # Concatenate: own (8 dims) + goal (3 dims) + obstacles (18 dims) = 29 dims
        obs = np.concatenate([own, goal_features] + per_other, axis=0).astype(np.float32)
        
        # Ensure exactly 29 dims
        assert len(obs) == 29, f"Observation size mismatch: expected 29, got {len(obs)}"

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
        # Per-agent termination/truncation is supported; success/failure details tracked in infos.
        terminations: Dict[str, bool] = {agent: False for agent in self.agents}
        truncations: Dict[str, bool] = {agent: False for agent in self.agents}
        infos: Dict[str, dict] = {agent: {} for agent in self.agents}

        # Pairwise risk and DCPA across all agents pre-computed and cached for this step
        pair_dcpa = self.pair_dcpa
        pair_tcpa = self.pair_tcpa
        pair_risk = self.pair_risk
        pair_dist = self.pair_dist

        
        LOA = self.LOA_own
        dcpa_safe = LOA * 4.0
        goal_radius = LOA * 2.0

        # Agent-by-agent reward computation and metrics tracking
            # Once an agent reaches its final waypoint, we FREEZE its metrics.
            # This prevents spurious metric pollution from post-goal behavior (e.g., drift, loiter, etc.)
        
        for k, agent in enumerate(self.agents):
            # Waypoint following: reward along-track movement, penalize cross-track
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
            # N4 terms: progress reward - bounded safety cost + success bonus + collision penalty
            total = 0.0

            # 1a) Progress reward: reward along-track movement towards goal, penalize cross-track movement
            if not agent_already_done:
                psi_des = float(np.arctan2(wy_m - y_k, wx_m - x_k))
                t_hat = np.array([np.cos(psi_des), np.sin(psi_des)], dtype=float)
                n_hat = np.array([-np.sin(psi_des), np.cos(psi_des)], dtype=float)

                dx_step = float(x_k - self.prev_X_all[k, 0])
                dy_step = float(y_k - self.prev_X_all[k, 1])
                dp = np.array([dx_step, dy_step], dtype=float)
                step_dist = float(np.hypot(dx_step, dy_step))
                self.episode_metrics[agent]["path_length_m"] += step_dist
                
                # Reward based on progress towards goal (route completion), not just displacement
                progress_m, route_len_m, _ = self.route_progress(k)
                goal_progress = float(np.clip(progress_m / max(route_len_m, 1.0), 0.0, 1.0))
                
                # Delta progress: how much closer to goal this step?
                prev_progress = self.prev_goal_progress_all.get(agent, 0.0)
                delta_progress = goal_progress - prev_progress
                self.prev_goal_progress_all[agent] = goal_progress

            else:
                # Agent already reached goal - no waypoint following reward
                delta_progress = 0.0


            # Risk tracking: record max risk experienced between two vessels during a given time step (for evaluation diagnostics)
            agent_risks = pair_risk[k].copy()
            agent_risks[k] = 0.0 
            max_risk = float(np.max(agent_risks))
            infos[agent]["max_risk"] = max_risk
            # Only update metrics if agent hasn't reached goal yet
            if not agent_already_done:
                self.episode_metrics[agent]["risk_exposure"] += max_risk * self.dt

            # TCPA tracking: record minimum TCPA over episode (only future encounters have positive TCPA)
            tcpa_vals = pair_tcpa[k]
            tcpa_vals_finite = tcpa_vals[np.isfinite(tcpa_vals)]
            tcpa_vals_positive = tcpa_vals_finite[tcpa_vals_finite > 0.0]  # Only future CPA (positive TCPA)
            
            if tcpa_vals_positive.size:
                min_tcpa_abs = float(np.min(tcpa_vals_positive))
            else:
                min_tcpa_abs = float("inf")
            
            if not agent_already_done:
                self.episode_metrics[agent]["min_tcpa_s"] = min(self.episode_metrics[agent]["min_tcpa_s"], min_tcpa_abs)

            # DCPA tracking: minimum CPA distance across all other agents 
            dcpa_vals = pair_dcpa[k].copy()
            dcpa_vals[k] = np.inf  # exclude self
            min_dcpa_abs = float(np.min(dcpa_vals[np.isfinite(dcpa_vals)])) if np.any(np.isfinite(dcpa_vals)) else float("inf")

            # Track episode-minimum DCPA: active encounter only (< 3 nmi), clipped at 5 nmi, numerical artifacts < 10m removed.
                # 3 nmi threshold matches compare_case_metrics.py so training and evaluation metrics are comparable.
            dist_vals_ep = pair_dist[k].copy()
            in_encounter_ep = (dist_vals_ep > 0) & (dist_vals_ep <= 3.0 * NMI)
            valid_ep = (np.arange(len(dcpa_vals)) != k) & np.isfinite(dcpa_vals) & in_encounter_ep
            if np.any(valid_ep):
                abs_dcpa_ep = np.abs(dcpa_vals[valid_ep])
                abs_dcpa_ep = abs_dcpa_ep[abs_dcpa_ep >= 10.0]  # filter numerical artifacts
                min_dcpa_ep = min(float(np.min(abs_dcpa_ep)), 5.0 * NMI) if len(abs_dcpa_ep) > 0 else LOA * 4.0
            else:
                min_dcpa_ep = LOA * 4.0  # no active encounter
            if not agent_already_done:
                self.episode_metrics[agent]["min_dcpa_m"] = min(
                    self.episode_metrics[agent]["min_dcpa_m"], min_dcpa_ep
                )

            infos[agent]["t"] = self.t

            # Compute min_dist (needed for safety cost and collision check below)
            min_dist = float(np.min(pair_dist[k][np.isfinite(pair_dist[k])])) if np.any(np.isfinite(pair_dist[k])) else np.inf

            # 1b) Adjust progress reward: conflict-scaled to reduce reward by 50% if in active encounter (to encourage avoidance)
            conflict_active = (
                min_dist < LOA * 5.0
                or (0.0 < min_tcpa_abs < 120.0 and min_dcpa_abs < LOA * 4.0)
            )
            progress_scale = 0.50 if conflict_active else 1.00
            r_progress = progress_scale * self.reward_weights["progress"] * delta_progress
            total += r_progress

            # 2) Bounded safety penalty (zero outside danger zone, capped per step) 
            safety_cost = 0.0
            danger_dist = LOA * 5.0   # ~150 m for a 30 m ship
            critical_dist = LOA * 2.0  # ~60 m

            if min_dist < danger_dist:
                frac = (danger_dist - min_dist) / danger_dist
                safety_cost += 0.5 * frac ** 2

            if min_dist < critical_dist:
                frac = (critical_dist - min_dist) / critical_dist
                safety_cost += 2.0 * frac ** 2

            if 0.0 < min_tcpa_abs < 120.0 and min_dcpa_abs < LOA * 4.0:
                dcpa_frac = float(np.clip((LOA * 4.0 - min_dcpa_abs) / (LOA * 4.0), 0.0, 1.0))
                tcpa_frac = float(np.clip((120.0 - min_tcpa_abs) / 120.0, 0.0, 1.0))
                safety_cost += 1.0 * dcpa_frac * tcpa_frac

            safety_cost = min(safety_cost, 3.0)
            total -= safety_cost

            infos[agent]["r_progress"] = float(r_progress)
            infos[agent]["safety_cost"] = float(safety_cost)
            infos[agent]["conflict_active"] = bool(conflict_active)
            infos[agent]["min_dcpa_abs"] = float(min_dcpa_abs)
            infos[agent]["min_tcpa_abs"] = float(min_tcpa_abs)

            # 3) Collision penalty: large negative reward for actual collision
                # Difficulty-aware: harder cases (4-ship) get more severe penalties
            
            # Check if collision first: hard blocker for success
            finite_d = np.isfinite(pair_dist[k])
            min_dist = float(np.min(pair_dist[k][finite_d])) if np.any(finite_d) else np.inf
            collision = (min_dist < LOA)

            infos[agent]["collision"] = bool(collision)

            if collision:
                if self.episode_metrics[agent]["collision"] == 0:
                    self.episode_metrics[agent]["collision"] = 1
                    self.episode_metrics[agent]["success"] = 0
                    self.episode_metrics[agent]["goal_passed"] = 0

                    # Per-agent termination: collision ends THIS agent's episode immediately
                    # (Other agents continue in multi-agent scenarios)
                    terminations[agent] = True
                    truncations[agent] = False 

                    if np.isnan(self.episode_metrics[agent]["completion_time_s"]):
                        self.episode_metrics[agent]["completion_time_s"] = self.t

                    # Mark agent as done to stop collecting metrics
                    self.agent_reached_goal[agent] = True
                 
                infos[agent]["success"] = False  
                infos[agent]["terminal_reason"] = "collision"
                infos[agent]["hard_collision_penalty"] = float(self.reward_weights['collision'])
                
                # Collision reward is exactly the hard penalty - this is a terminal failure signal for learning
                total = float(self.reward_weights['collision'])  # hard terminal collision penalty

            else:
                # No collision: now check if agent reached goal
                # Success: reached final wp and within radius 
                progress_m, route_len_m, _ = self.route_progress(k)
                goal_progress = float(np.clip(progress_m / max(route_len_m, 1.0), 0.0, 1.0))
                
                # Update goal progress only if agent hasn't reached goal yet
                if not agent_already_done:
                    self.episode_metrics[agent]["goal_progress"] = max(self.episode_metrics[agent]["goal_progress"], goal_progress)
                
                # Success if: reached final waypoint using waypoint_selection() built-in threshold (~200m from CORALL planning.py)
                final_waypoint_reached_by_index = (i_wpt_k >= len(Xwpt_k) - 1)
                infos[agent]["terminal_reason"] = None
                
                # Calculate distance to final goal (always, not just when at final waypoint)
                xf_nmi = float(Xwpt_k[-1])
                yf_nmi = float(Ywpt_k[-1])
                dist_to_goal_nmi = np.hypot(xf_nmi - x_nmi, yf_nmi - y_nmi)
                dist_to_goal_m = dist_to_goal_nmi * NMI
                threshold_nmi = 200.0 / 1852.0  # ~200m acceptance radius
                
                # Check if agent is close to FINAL goal, regardless of waypoint index
                close_to_goal = (dist_to_goal_nmi < threshold_nmi)
                
                if final_waypoint_reached_by_index and len(Xwpt_k) > 1:
                    final_reached = close_to_goal
                else:
                    final_reached = close_to_goal
                
                # 4) Graduated safe-success reward: bonus scales with episode-minimum separation.
                # Uses the tracked worst-case separation over the whole episode, not just at the goal.
                if final_reached and not collision:
                    episode_min_sep = min(
                        self.episode_metrics[agent]["min_actual_sep_m"],
                        min_dist,
                    )

                    if episode_min_sep >= LOA * 3.0:
                        # Clean success — wide clearance throughout
                        success_bonus = 1000.0
                        infos[agent]["success"] = True
                        infos[agent]["terminal_reason"] = "success"
                    elif episode_min_sep >= LOA * 2.0:
                        # Success, but came close at some point
                        success_bonus = 700.0
                        infos[agent]["success"] = True
                        infos[agent]["terminal_reason"] = "success"
                    else:
                        # Reached goal but had a near-miss during the episode
                        success_bonus = 300.0
                        infos[agent]["success"] = True
                        infos[agent]["terminal_reason"] = "success_near_miss"

                    total += success_bonus
                    infos[agent]["success_bonus"] = success_bonus
                    self.episode_metrics[agent]["success"] = 1
                    self.episode_metrics[agent]["goal_passed"] = 1

                    # Mark agent as reached goal - stop collecting metrics for this agent
                    self.agent_reached_goal[agent] = True

                    if np.isnan(self.episode_metrics[agent]["completion_time_s"]):
                        self.episode_metrics[agent]["completion_time_s"] = self.t

                    # Mark agent as truncated (agent-level done) when reaching goal
                    truncations[agent] = True

                else:
                    infos[agent]["success"] = False
                    infos[agent]["success_bonus"] = 0.0
            
            # Track collision metrics (for analysis, not reward shaping)
            finite_d = np.isfinite(pair_dist[k])
            min_dist = float(np.min(pair_dist[k][finite_d])) if np.any(finite_d) else np.inf
            collision = (min_dist < LOA) 

            w_collision = self.reward_weights['collision']
            infos[agent]["min_dist"] = min_dist
            
            if not agent_already_done:
                self.episode_metrics[agent]["min_actual_sep_m"] = min(self.episode_metrics[agent]["min_actual_sep_m"], min_dist)
                near_miss_threshold = LOA * 1.5
                if (min_dist >= LOA) and (min_dist < near_miss_threshold) and not collision:
                    if self.episode_metrics[agent]["near_miss"] == 0:
                        self.episode_metrics[agent]["near_miss"] = 1
            
        

            rewards[agent] = float(total)

        # Individual agents terminate on collision or success; sim_time truncates any still-running agents.
        return rewards, terminations, truncations, infos 


        