"""
Random-case environment wrapper for generalized policy training.

This wrapper randomizes case (1, 6, 21) and seed at each episode reset,
enabling training of a single policy that generalizes across difficulty levels
and random encounter geometries.

USAGE:
======
env = RandomCaseEnv(
    base_case=1,  # placeholder, ignored since case is randomized
    num_seeds=100,
    cases_to_train=[1, 6, 21],
)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)

EXPECTED BEHAVIOR:
==================
- Each reset() picks a random case (1, 6, 21) and random seed
- Policy learns to adapt to all three difficulty levels
- Produces a single checkpoint generalizing across all cases
- Randomized seeds prevent memorization of specific geometries
"""

from __future__ import annotations

from typing import Optional, Tuple, Any, List
import numpy as np
import gymnasium as gym

from .env_single_agent_sb3 import SingleAgentOwnshipEnv


class RandomCaseEnv(gym.Wrapper):
    """
    Gymnasium wrapper that randomizes case and seed at each reset.
    
    Wraps SingleAgentOwnshipEnv to provide curriculum-free multi-case learning.
    On each reset(), a random case (1, 6, 21) and random seed are selected,
    forcing the policy to generalize across different encounter difficulties.
    
    IMPORTANT: Handles variable observation sizes across cases by padding to
    max size (Case 21 with 3 obstacles = 22 dims). Ensures SB3's buffer can
    handle shape changes across resets.
    
    Args:
        cases_to_train: List of cases to sample from (default: [1, 6, 21])
        num_seeds: Range of seeds to sample from [0, num_seeds-1] (default: 100)
        dt: Simulation timestep in seconds (default: 0.5)
        sim_time: Episode length in seconds (default: 1950.0)
        n_heading: Heading action discretization bins (default: 7)
        n_speed: Speed action discretization bins (default: 5)
        max_heading_change_deg: Max heading change per action (default: 25.0)
        u_min, u_max: Speed bounds (default: 5.0, 10.0 m/s)
        loa_m: Ownship length for collision detection (default: 30.0 m)
        route_len_nmi: Waypoint distance (default: 2.0 NMI)
    """
    
    # Observation sizes per case (ownship=8 + obstacles*6)
    # Updated to match new observation space: [x, y, sin(psi), cos(psi), r, u_x, u_y, b] + [dx, dy, sin_b, cos_b, du_x, du_y]
    CASE_OBS_SIZES = {
        1: 8 + 1*6,   # 1 obstacle = 14
        6: 8 + 2*6,   # 2 obstacles = 20
        21: 8 + 3*6,  # 3 obstacles = 26
    }
    MAX_OBS_SIZE = 26  # Case 21
    
    def __init__(
        self,
        cases_to_train: List[int] = [1, 6, 21],
        num_seeds: int = 100,
        dt: float = 0.5,
        sim_time: float = 490.0,
        n_heading: int = 7,
        max_heading_change_deg: float = 25.0,
        loa_m: float = 30.0,
        route_len_nmi: float = 2.0,
        master_seed: Optional[int] = None,
    ):
        # Create initial environment (case doesn't matter, will be randomized at reset)
        base_env = SingleAgentOwnshipEnv(
            case_number=cases_to_train[0],
            dt=dt,
            sim_time=sim_time,
            n_heading=n_heading,
            max_heading_change_deg=max_heading_change_deg,
            loa_m=loa_m,
            route_len_nmi=route_len_nmi,
            seed=None,
        )
        super().__init__(base_env)
        
        self.cases_to_train = list(cases_to_train)
        self.num_seeds = int(num_seeds)
        self.dt = float(dt)
        self.sim_time = float(sim_time)
        self.n_heading = int(n_heading)
        self.max_heading_change_deg = float(max_heading_change_deg)
        self.loa_m = float(loa_m)
        self.route_len_nmi = float(route_len_nmi)
        
        # Master RNG for reproducible case/seed sequence
        self.rng = np.random.default_rng(master_seed)
        
        # Tracking for logging
        self.episode_count = 0
        self.current_case = cases_to_train[0]
        self.current_seed = 0
        
        # Override observation space to match max size (handles variable obs across cases)
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-1.5e4, high=1.5e4,
            shape=(self.MAX_OBS_SIZE,),
            dtype=np.float32
        )
    
    def _pad_observation(self, obs: np.ndarray, case: int) -> np.ndarray:
        """
        Pad observation to max size by zero-filling unused obstacle slots.
        
        For cases with fewer obstacles, pad with zeros to reach MAX_OBS_SIZE.
        Ownship features (first 7): always populated
        Obstacle features (5 each): zero-padded if fewer than 3 obstacles
        """
        target_size = self.MAX_OBS_SIZE
        obs_size = len(obs)
        
        if obs_size == target_size:
            return obs.astype(np.float32)
        
        # Pad with zeros
        padded = np.zeros(target_size, dtype=np.float32)
        padded[:obs_size] = obs
        return padded
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset environment with randomized case and seed.
        
        Uses master RNG for reproducible case/seed sequence.
        Creates new SingleAgentOwnshipEnv with random case/seed each time.
        Returns observation padded to consistent max size.
        """
        # Sample random case and seed using master RNG
        self.current_case = int(self.rng.choice(self.cases_to_train))
        self.current_seed = int(self.rng.integers(0, self.num_seeds))
        
        # Create new environment with random case/seed
        self.env = SingleAgentOwnshipEnv(
            case_number=self.current_case,
            dt=self.dt,
            sim_time=self.sim_time,
            n_heading=self.n_heading,
            max_heading_change_deg=self.max_heading_change_deg,
            loa_m=self.loa_m,
            route_len_nmi=self.route_len_nmi,
            seed=self.current_seed,
        )
        
        self.episode_count += 1
        
        # Reset and return observation (with padding)
        obs, info = self.env.reset(seed=self.current_seed)
        obs = self._pad_observation(obs, self.current_case)
        
        # Add case/seed info to info dict for logging
        info["case"] = self.current_case
        info["seed"] = self.current_seed
        info["episode"] = self.episode_count
        
        return obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Step environment (delegated to wrapped env). Returns padded observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Pad observation to consistent size
        obs = self._pad_observation(obs, self.current_case)
        
        # Add case/seed to info for logging
        info["case"] = self.current_case
        info["seed"] = self.current_seed
        
        return obs, reward, terminated, truncated, info
    
    def get_case_distribution(self) -> dict:
        """Get distribution of cases trained so far."""
        # This would require tracking, but for now just return info
        return {
            "total_episodes": self.episode_count,
            "cases_available": self.cases_to_train,
            "seed_range": f"[0, {self.num_seeds})",
        }
    
    @property
    def env_multi(self):
        """Access the wrapped MultiShipParallelEnv for state/history access."""
        if hasattr(self.env, 'env_multi'):
            return self.env.env_multi
        raise RuntimeError("env_multi not accessible: RandomCaseEnv.env may not be SingleAgentOwnshipEnv")
