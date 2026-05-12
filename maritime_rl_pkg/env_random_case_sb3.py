"""
Random-case environment wrapper for generalized policy training.

This wrapper randomizes cases and seeds at each episode reset,
enabling training of a single policy that generalizes across difficulty levels
and random encounter geometries.


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
    On each reset(), a random case (1, 6, 21, etc.) and random seed are selected,
    forcing the policy to generalize across different encounter difficulties.
    
    IMPORTANT: Handles variable observation sizes across cases by padding to
    max size. Ensures SB3's buffer can handle shape changes across resets.
    
    Args:
        cases_to_train: List of cases to sample from (default: [1, 6, 21])
        num_seeds: Range of seeds to sample from [0, num_seeds-1] (default: 100)
        dt: Simulation timestep in seconds (default: 0.5)
        sim_time: Episode length in seconds (default: 490.0)
        n_heading: Heading action discretization bins (default: 7)
        max_heading_change_deg: Max heading change per action (default: 25.0)
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
    MAX_OBS_SIZE = 29  # (own state) + 3 (goal bearing/distance) + 18 (3 obstacles × 6) = 29
    
    # Case groupings by agent count
    CASES_2SHIP = list(range(1, 5))      # Cases 1-4: 2-ship scenarios
    CASES_3SHIP = list(range(5, 12))     # Cases 5-11: 3-ship scenarios
    CASES_4SHIP = list(range(12, 23))    # Cases 12-22: 4-ship scenarios
    
    # Curriculum phases: (step_threshold, available_cases)
    # Simple hard switches: no blending, just unlock new cases at each threshold
    CURRICULUM_PHASES = [
        (0, CASES_2SHIP),                           # Phase 1: 2-ship cases only [1-4]
        (500000, CASES_2SHIP + CASES_3SHIP),       # Phase 2: 2-ship + 3-ship [1-11]
        (1000000, CASES_2SHIP + CASES_3SHIP + CASES_4SHIP),  # Phase 3: all scenarios [1-22]
    ]
    CURRICULUM_TRANSITION = 200000  # steps for smooth blending between phases
    
    def __init__(
        self,
        cases_to_train: List[int] = [1, 6, 21],
        num_seeds: int = 100,
        dt: float = 0.5,
        sim_time: float = 500.0,
        n_heading: int = 7,
        max_heading_change_deg: float = 25.0,
        loa_m: float = 30.0,
        route_len_nmi: float = 2.0,
        master_seed: Optional[int] = None,
        desired_cross_x_nmi: float = 1.0,
        target_speed_mps: float = 10.0,
        ownship_speed_mps: Optional[float] = None,
        enable_curriculum: bool = True,
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
            desired_cross_x_nmi=desired_cross_x_nmi,
            target_speed_mps=target_speed_mps,
            ownship_speed_mps=ownship_speed_mps,
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
        self.desired_cross_x_nmi = float(desired_cross_x_nmi)
        self.target_speed_mps = float(target_speed_mps)
        self.ownship_speed_mps = ownship_speed_mps if ownship_speed_mps is None else float(ownship_speed_mps)
        
        # Master RNG for reproducible case/seed sequence
        self.rng = np.random.default_rng(master_seed)
        
        # Curriculum learning
        self.enable_curriculum = enable_curriculum
        self.current_step = 0
        
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

    def _get_curriculum_available_cases(self) -> List[int]:
        """Get available cases at current training step based on curriculum."""
        if not self.enable_curriculum:
            return self.cases_to_train
        
        # Find current phase based on step thresholds
        current_phase_idx = 0
        for i, (threshold, _) in enumerate(self.CURRICULUM_PHASES):
            if self.current_step >= threshold:
                current_phase_idx = i
        
        # Get cases for current phase (hard switch, no blending)
        phase_cases = self.CURRICULUM_PHASES[current_phase_idx][1]
        
        # Filter by requested cases
        available = [c for c in phase_cases if c in self.cases_to_train]
        return available if available else self.cases_to_train
    
    def _get_case_sampling_weights(self) -> dict:
        """
        Weighted sampling with smooth curriculum progression.
        
        Phase 1 (0-500k): Uniform on 2-ship only
        Phase 2 (500k-1M): Blend 2-ship and 3-ship with smooth transition
        Phase 3 (1M-1.2M transition, then 1.2M+): Gradually blend in 4-ship cases
        
        Returns: dict mapping case -> weight (will be normalized to probabilities)
        """
        available = self._get_curriculum_available_cases()
        if not available:
            raise ValueError("No available cases for sampling")
        
        weights = {c: 0.0 for c in available}
        
        # Determine current phase
        current_phase_idx = 0
        for i, (threshold, _) in enumerate(self.CURRICULUM_PHASES):
            if self.current_step >= threshold:
                current_phase_idx = i
        
        # Count cases in each difficulty group
        avail_2ship = [c for c in available if c in self.CASES_2SHIP]
        avail_3ship = [c for c in available if c in self.CASES_3SHIP]
        avail_4ship = [c for c in available if c in self.CASES_4SHIP]
        
        if current_phase_idx == 0:  # Phase 1: 2-ship only
            if avail_2ship:
                for c in avail_2ship:
                    weights[c] = 1.0 / len(avail_2ship)
        
        elif current_phase_idx == 1:  # Phase 2: 2-ship + 3-ship with smooth blend
            # Compute transition progress for phase 2->3 boundary
            next_threshold = self.CURRICULUM_PHASES[2][0]  # Phase 3 threshold (1M)
            transition_start = next_threshold - self.CURRICULUM_TRANSITION  # 800k
            
            if self.current_step < transition_start:
                phase_progress = 0.0
            else:
                phase_progress = min(1.0, (self.current_step - transition_start) / self.CURRICULUM_TRANSITION)
            
            # Transition from 50/50 to 30/70
            pct_2ship = 0.5 - phase_progress * 0.2  # 0.5 -> 0.3
            pct_3ship = 0.5 + phase_progress * 0.2  # 0.5 -> 0.7
            
            total_cases = len(avail_2ship) + len(avail_3ship)
            if total_cases > 0:
                if avail_2ship:
                    weight_per_2ship = pct_2ship / len(avail_2ship)
                    for c in avail_2ship:
                        weights[c] = weight_per_2ship
                
                if avail_3ship:
                    weight_per_3ship = pct_3ship / len(avail_3ship)
                    for c in avail_3ship:
                        weights[c] = weight_per_3ship
        
        else:  # Phase 3: smooth introduction of 4-ship cases
            # Phase 3 transition: 1M to 1.2M steps (200k step transition window)
            phase3_start = self.CURRICULUM_PHASES[2][0]  # 1M
            transition_end = phase3_start + self.CURRICULUM_TRANSITION  # 1.2M
            
            if self.current_step < transition_end:
                # During transition window (1M to 1.2M): smooth ramp-up of 4-ship
                phase_progress = (self.current_step - phase3_start) / self.CURRICULUM_TRANSITION
            else:
                # After transition (1.2M+): full 4-ship emphasis
                phase_progress = 1.0
            
            # Start at 30% 2-ship / 70% 3-ship (end of phase 2)
            # Gradually transition to 20% 2-ship / 30% 3-ship / 50% 4-ship (smooth gradient to avoid forgetting)
            pct_2ship = 0.30 - phase_progress * 0.10      # 0.30 -> 0.20
            pct_3ship = 0.70 - phase_progress * 0.40      # 0.70 -> 0.30
            pct_4ship = phase_progress * 0.50             # 0.00 -> 0.50 (gradual ramp-up!)
            
            total_cases = len(avail_2ship) + len(avail_3ship) + len(avail_4ship)
            if total_cases > 0:
                if avail_2ship:
                    weight_per_2ship = pct_2ship / len(avail_2ship)
                    for c in avail_2ship:
                        weights[c] = weight_per_2ship
                
                if avail_3ship:
                    weight_per_3ship = pct_3ship / len(avail_3ship)
                    for c in avail_3ship:
                        weights[c] = weight_per_3ship
                
                if avail_4ship:
                    weight_per_4ship = pct_4ship / len(avail_4ship)
                    for c in avail_4ship:
                        weights[c] = weight_per_4ship
        
        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {c: w / total for c, w in weights.items()}
        else:
            weights = {c: 1.0 / len(available) for c in available}
        
        return weights

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
        Reset environment with curriculum-controlled case and random seed.
        
        Curriculum progressively unlocks harder cases with weighted sampling to prevent forgetting:
        - Phase 1: 2-ship uniform
        - Phase 2: 2-ship + 3-ship uniform
        - Phase 3: 4-ship focused (80%) with 2-ship/3-ship maintenance (5%/15%)
        """
        # Use provided seed or master RNG for case/seed sequence
        if seed is not None:
            local_rng = np.random.default_rng(seed)
        else:
            local_rng = self.rng

        # Get curriculum-controlled available cases
        available_cases = self._get_curriculum_available_cases()
        
        if not available_cases:
            raise ValueError("No available cases for sampling")
        
        # Sample case with weighted probabilities (prevents forgetting)
        case_weights = self._get_case_sampling_weights()
        case_probs = np.array([case_weights.get(c, 1.0 / len(available_cases)) for c in available_cases])
        
        # Ensure probabilities are valid
        prob_sum = case_probs.sum()
        if prob_sum > 0:
            case_probs = case_probs / prob_sum
        else:
            case_probs = np.ones(len(available_cases)) / len(available_cases)
        
        # Sample case weighted by difficulty
        self.current_case = int(local_rng.choice(available_cases, p=case_probs))
        self.current_seed = int(local_rng.integers(0, self.num_seeds))
        
        # Create new environment with random case/seed and geometry parameters
        self.env = SingleAgentOwnshipEnv(
            case_number=self.current_case,
            dt=self.dt,
            sim_time=self.sim_time,
            n_heading=self.n_heading,
            max_heading_change_deg=self.max_heading_change_deg,
            loa_m=self.loa_m,
            route_len_nmi=self.route_len_nmi,
            seed=self.current_seed,
            desired_cross_x_nmi=self.desired_cross_x_nmi,
            target_speed_mps=self.target_speed_mps,
            ownship_speed_mps=self.ownship_speed_mps,
        )

        self.episode_count += 1

        obs, info = self.env.reset(seed=self.current_seed)
        obs = self._pad_observation(obs, self.current_case)

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
    
    def update_step(self, step: int):
        """Update current training step (called from training callback)."""
        self.current_step = step
    
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
