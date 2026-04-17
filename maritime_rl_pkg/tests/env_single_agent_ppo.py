"""
Single-agent (Gymnasium Env) for PPO training compatible with CORALL simulation base

Only agent 0 (ownship) is trained.
Agents 1...K (obstacle ships) use scripted/deterministic actions (maintain heading).

All ownship metrics are calculated identically to multi-agent version for fair comparison.

Wraps the multi-agent ParallelEnv for compatibility with single-agent PPO trainers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Import the multi-agent env
from ..env_multi_agent_ppo import MultiShipParallelEnv

NMI = 1852.0


class SingleAgentOwnshipEnv(gym.Env):
    """
    Single-agent wrapper around multi-agent env.
    
    - Only agent_0 (ownship) receives RL actions
    - Obstacles (agents 1+) maintain constant heading (scripted)
    - Observations/rewards/dones only for ownship
    - Compatible with standard Gymnasium and single-agent PPO trainers
    """

    metadata = {"render_modes": ["human"], "name": "corall_ownship_ppo_v0"}

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
        u_min: float = 5.0,
        u_max: float = 10.0,
        loa_m: float = 30.0,
        route_len_nmi: float = 2.0,
        seed: Optional[int] = None,
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
            n_speed=n_speed,
            max_heading_change_deg=max_heading_change_deg,
            u_min=u_min,
            u_max=u_max,
            loa_m=loa_m,
            route_len_nmi=route_len_nmi,
            seed=seed,
        )
        
        # Single-agent action/observation spaces (only for ownship)
        self.action_space = self.env_multi.action_space("ship_0")
        self.observation_space = self.env_multi.observation_space("ship_0")
        
        # Tracking
        self.last_obs_dict = {}
        self.last_reward_dict = {}
        self.last_done_dict = {}
        self.last_info_dict = {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment and return ownship observation."""
        obs_dict, info_dict = self.env_multi.reset(seed=seed, options=options)
        
        # Store for later
        self.last_obs_dict = obs_dict
        self.last_info_dict = info_dict
        
        # Return only ownship (agent_0) observation
        return obs_dict["ship_0"], info_dict.get("ship_0", {})

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step environment with RL action for ownship.
        Obstacles maintain constant heading (scripted).
        
        Returns: (obs, reward, terminated, truncated, info)
        """
        
        # Get current state of obstacles for scripted actions
        n_agents = self.env_multi.n_agents
        scripted_actions = {}
        
        # Ownship gets RL action
        scripted_actions["ship_0"] = action
        
        # Obstacles maintain constant heading: action = [heading_idx=center, speed_idx=center]
        # This corresponds to "maintain current heading and speed"
        center_heading = (self.env_multi.n_heading - 1) // 2
        center_speed = (self.env_multi.n_speed - 1) // 2
        
        for k in range(1, n_agents):
            # Scripted action: maintain heading and speed
            scripted_actions[f"ship_{k}"] = np.array([center_heading, center_speed], dtype=np.int32)
        
        # Step multi-agent environment
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.env_multi.step(scripted_actions)
        
        # Store for later access
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
