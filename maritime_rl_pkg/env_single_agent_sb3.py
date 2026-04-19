"""
Single-agent Gymnasium environment wrapper for Stable-Baselines3.

This module converts the multi-agent PettingZoo environment (MultiShipParallelEnv) into a
single-agent Gymnasium environment suitable for Stable-Baselines3 training.

ARCHITECTURE:
=============
- ownship (agent_0): RL-controlled agent
- obstacles (agents 1...K): scripted to maintain constant heading & speed
- Reward function: inherited from MultiShipParallelEnv (collision penalty + waypoint progress)
- Observation: only ownship state (position, heading, speed, other ships' relative positions)

PER-CASE GEOMETRY:
==================
- Case 1: scale=1.0 (original Imazu distances, loose encounters)
- Case 6: scale=0.75 (25% closer, medium difficulty)
- Case 21: scale=0.5 (50% closer, tight encounters)
- Waypoint distance: 2.0 NMI (3,704 m) ahead in initial heading direction
- Obstacle speeds: uniform 9.5 m/s (normalized across all Imazu cases)

USAGE:
======
Used by train_single_agent_sb3.py in the training loop. Instantiate with:
    env = SingleAgentOwnshipEnv(case_number=6, route_len_nmi=2.0, seed=0)
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

Compatible with Stable-Baselines3 PPO, SAC, DQN, and all standard Gymnasium RL algorithms.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .env_multi_agent_ppo import MultiShipParallelEnv
from navigation.reactive_avoidance import reactive_avoidance

NMI = 1852.0


class SingleAgentOwnshipEnv(gym.Env):
    """
    Single-agent Gymnasium wrapper around multi-agent PettingZoo environment.
    
    This wrapper extracts only the ownship (agent_0) from the multi-agent environment,
    making it compatible with Stable-Baselines3 single-agent trainers.
    
    KEY CONCEPTS:
    =============
    1. Ownship (agent_0): The trained RL agent
       - Receives RL actions (heading rate and speed discretization)
       - Gets observations: own state + relative positions of obstacles
       - Receives rewards from MultiShipParallelEnv reward function
    
    2. Obstacles (agents 1...K): Scripted (not learned)
       - Always maintain constant heading and speed (center actions)
       - Preserve original Imazu case headings and speeds
       - Uniform propagation: constant velocity at fixed heading
    
    3. Geometry (defined in MultiShipParallelEnv):
       - Ownship always starts at origin
       - Obstacles placed at original Imazu positions with original headings
       - No per-case scaling; full original geometry preserved
    
    4. Reward function (from MultiShipParallelEnv):
       - Collision penalty: -10 if any agent separation < LOA
       - Waypoint progress reward: +goal_progress per step
       - CPA-based risk penalization
       - Goal completion bonus
    
    args:
        case_number: CORALL case (1, 6, 21, etc.) - determines obstacle geometry
        dt: simulation timestep (seconds, default 0.5)
        sim_time: episode length (seconds, default 1950.0 ≈ 32.5 min)
        render_mode: visualization mode (not fully implemented)
        n_heading: action discretization for heading (7 bins, ±25°)
        n_speed: action discretization for speed (5 bins, 5-10 m/s)
        u_min, u_max: speed bounds for ownship and obstacles (m/s)
        loa_m: length of ownship for collision detection (30 m)
        route_len_nmi: waypoint distance along initial heading (2.0 NMI = 3704 m)
        seed: random seed for reproducibility
    """

    metadata = {"render_modes": ["human"], "name": "corall_ownship_sb3_v0"}

    def __init__(
        self,
        case_number: int,
        dt: float = 0.5,
        sim_time: float = 490.0,
        render_mode: Optional[str] = None,
        # action discretization
        n_heading: int = 7,
        #n_speed: int = 5,
        max_heading_change_deg: float = 25.0,
        #u_min: float = 5.0,
        #u_max: float = 10.0,
        loa_m: float = 30.0,
        route_len_nmi: float = 2.0,
        seed: Optional[int] = None,
        # Geometry parameters
        desired_cross_x_nmi: float = 1.0,
        target_speed_mps: float = 10.0,
        ownship_speed_mps: Optional[float] = None,
    ):
        super().__init__()
        
        self.case_number = int(case_number)
        self.dt = float(dt)
        self.sim_time = float(sim_time)
        self.render_mode = render_mode
        
        # Create the underlying multi-agent environment
        self.env_multi = MultiShipParallelEnv(
            case_number=case_number,
            dt=dt,
            sim_time=sim_time,
            render_mode=render_mode,
            n_heading=n_heading,
            max_heading_change_deg=max_heading_change_deg,
            loa_m=loa_m,
            route_len_nmi=route_len_nmi,
            seed=seed,
            desired_cross_x_nmi=desired_cross_x_nmi,
            target_speed_mps=target_speed_mps,
            ownship_speed_mps=ownship_speed_mps,
        )
        
        # Single-agent action/observation spaces (only for ownship)
        self.action_space = self.env_multi.action_space("ship_0")
        self.observation_space = self.env_multi.observation_space("ship_0")
        
        # Tracking
        self.last_obs_dict = {}
        self.last_reward_dict = {}
        self.last_done_dict = {}
        self.last_info_dict = {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment and return ownship observation."""
        obs_dict, info_dict = self.env_multi.reset(seed=seed, options=options)
        
        # Store for later
        self.last_obs_dict = obs_dict
        self.last_info_dict = info_dict
        
        # Return only ownship (agent_0) observation and info
        return obs_dict["ship_0"], info_dict.get("ship_0", {})

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step environment with RL action for ownship. 
        Obstacles use scripted "maintain heading/speed" actions.
        
        CONFIG EXPLANATION:
        ===================
        Action space (for ownship):
            - MultiDiscrete([7, 5]) where:
              - Heading index: 0-6 (left, center-left, ..., center, ..., right)
              - Speed index: 0-4 (slow, ..., fast)
            - Center indices (3, 2) correspond to "maintain current heading/speed"
        
        Obstacle scripting:
            - All obstacles (agents 1...K) use action [center_heading=3, center_speed=2]
            - This keeps them on deterministic collision course toward ownship
            - No exploration, purely scripted to test avoidance learning
        
        Returns: (obs, reward, terminated, truncated, info)
            - obs: ownship state + obstacles' relative positions (single-agent observation)
            - reward: ownship reward from collision/goal progress (scalar)
            - terminated: True if collision occurred
            - truncated: True if episode time limit reached
            - info: ownship info dict (metrics, etc.)
        """
        
        # Get current state of obstacles for reactive avoidance actions
        n_agents = self.env_multi.n_agents
        scripted_actions = {}
        
        # Ownship gets RL action (trained policy output)
        scripted_actions["ship_0"] = action
        
        # Obstacles use reactive avoidance: compute heading to avoid all vessels
        X_all = self.env_multi.X_all  # State: [x, y, psi, r, b, u]
        t = self.env_multi.t
        
        x_own = float(X_all[0, 0])
        y_own = float(X_all[0, 1])
        psi_own = float(X_all[0, 2])
        
        center_heading = (self.env_multi.n_heading - 1) // 2
        
        for k in range(1, n_agents):
            # Obstacles are PASSIVE (no reactive avoidance)
            # They maintain initial heading and speed from Imazu case geometry
            # This creates deterministic collision courses for testing avoidance learning
            
            # Center actions = maintain current heading and speed
            scripted_actions[f"ship_{k}"] = np.array([center_heading], dtype=np.int32)
        
        # Step multi-agent environment
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.env_multi.step(scripted_actions)
        
        # Store for later access (used by eval callbacks, metrics, etc.)
        self.last_obs_dict = obs_dict
        self.last_reward_dict = reward_dict
        self.last_done_dict = {**terminated_dict, **truncated_dict}
        self.last_info_dict = info_dict
        
        # Extract ownship-only outputs
        obs = obs_dict["ship_0"]
        reward = reward_dict["ship_0"]
        terminated = terminated_dict.get("ship_0", False)
        truncated = truncated_dict.get("ship_0", False)
        info = info_dict.get("ship_0", {})
        
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[Any]:
        """Render episode (if supported by underlying env)."""
        if hasattr(self.env_multi, 'render'):
            return self.env_multi.render()
        return None

    def close(self):
        """Close environment."""
        if hasattr(self.env_multi, 'close'):
            self.env_multi.close()

    # -----------------------------------------------
    # Helpers for compatibility / inspection
    # -----------------------------------------------
    
    @property
    def unwrapped(self):
        """Return the base environment (for inspection)."""
        return self

    def get_multi_agent_env(self):
        """Get underlying multi-agent environment (for debugging/inspection)."""
        return self.env_multi

    def get_ownship_metrics(self) -> Dict[str, Any]:
        """
        Extract ownship-only metrics from the multi-agent environment.
        Returns the episode metrics collected for ownship (agent_0).
        """
        if hasattr(self.env_multi, 'episode_metrics') and "ship_0" in self.env_multi.episode_metrics:
            return self.env_multi.episode_metrics["ship_0"]
        return {}

    def get_state(self) -> np.ndarray:
        """Get full state (for inspection/debugging)."""
        return self.env_multi.X_all.copy()

    def get_pairwise_geometry(self) -> Dict[str, np.ndarray]:
        """Get pairwise CPA/risk/distance caches."""
        return {
            "pair_dcpa": self.env_multi.pair_dcpa.copy(),
            "pair_tcpa": self.env_multi.pair_tcpa.copy(),
            "pair_dist": self.env_multi.pair_dist.copy(),
            "pair_risk": self.env_multi.pair_risk.copy(),
        }
