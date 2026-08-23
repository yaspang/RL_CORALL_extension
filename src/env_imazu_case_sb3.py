"""
Imazu Case environment wrapper for generalized policy training and evaluation.

Randomizes which Imazu case and episode seed are used at each reset, enabling
a single PPO policy to train across all 22 fixed encounter geometries. Case
selection follows a curriculum that progressively unlocks 3-ship and 4-ship
cases, with smooth probability blending between phases.

Also used during evaluation: eval_generalized_policy_sb3.py wraps each fixed
evaluation case in ImazuCaseEnv to ensure the same 29-dim padded observation
space that the policy was trained with.
"""

from __future__ import annotations

from typing import Optional, Tuple, Any, List, Dict
import numpy as np
import gymnasium as gym
from pathlib import Path

from .env_single_agent_sb3 import SingleAgentOwnshipEnv


class ImazuCaseEnv(gym.Wrapper):
    """
    Gymnasium wrapper that samples a random Imazu case and seed at each reset.

    Wraps SingleAgentOwnshipEnv and pads observations to a fixed 29-dim vector
    so SB3's replay buffer stays consistent across cases with different obstacle
    counts (2-ship = 14 dims raw, 4-ship = 26 dims raw, all padded to 29).

    Case selection is curriculum-weighted: 2-ship cases dominate early training,
    with 3-ship and 4-ship cases blended in smoothly as training_step increases.
    Call update_step(training_step) from the training callback to drive this.

    Args:
        cases_to_train: Imazu case numbers to sample from (default: all 22)
        num_seeds: Episode seeds sampled from [0, num_seeds-1] (default: 100)
        dt: Simulation timestep in seconds (default: 0.5)
        sim_time: Episode horizon in seconds (default: 500.0)
        n_heading: Heading action bins (default: 7)
        n_speed: Speed action bins (default: 5)
        max_heading_change_deg: Max heading offset per step (default: 25.0 deg)
        loa_m: Ownship LOA for collision detection (default: 30.0 m)
        route_len_nmi: Waypoint route length (default: 2.0 nmi)
        enable_curriculum: Use curriculum-weighted case sampling (default: True)
    """
    
    MAX_OBS_SIZE = 29  # own(8) + goal(3) + 3 obstacles×6 = 29

    CASES_2SHIP = list(range(1, 5))
    CASES_3SHIP = list(range(5, 12))
    CASES_4SHIP = list(range(12, 23))

    CURRICULUM_PHASES = [
        (0,       CASES_2SHIP),
        (500000,  CASES_2SHIP + CASES_3SHIP),
        (1000000, CASES_2SHIP + CASES_3SHIP + CASES_4SHIP),
    ]
    CURRICULUM_TRANSITION = 200000  # steps for smooth blending between phases

    def __init__(
        self,
        cases_to_train: List[int] = [1, 6, 21],
        num_seeds: int = 100,
        dt: float = 0.5,
        sim_time: float = 500.0,
        n_heading: int = 7,
        n_speed: int = 5,
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
            n_speed=n_speed,
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
        self.n_speed = int(n_speed)
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
            pct_4ship = phase_progress * 0.50             # 0.00 -> 0.50 (gradual ramp-up)
            
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
        """Reset with curriculum-weighted case and random seed."""
        local_rng = np.random.default_rng(seed) if seed is not None else self.rng

        available_cases = self._get_curriculum_available_cases()
        if not available_cases:
            raise ValueError("No available cases for sampling")

        case_weights = self._get_case_sampling_weights()
        case_probs = np.array([case_weights.get(c, 1.0 / len(available_cases)) for c in available_cases])
        prob_sum = case_probs.sum()
        case_probs = case_probs / prob_sum if prob_sum > 0 else np.ones(len(available_cases)) / len(available_cases)

        self.current_case = int(local_rng.choice(available_cases, p=case_probs))
        self.current_seed = int(local_rng.integers(0, self.num_seeds))
        
        # Create new environment with random case/seed and geometry parameters
        self.env = SingleAgentOwnshipEnv(
            case_number=self.current_case,
            dt=self.dt,
            sim_time=self.sim_time,
            n_heading=self.n_heading,
            n_speed=self.n_speed,
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
        raise RuntimeError("env_multi not accessible: ImazuCaseEnv.env may not be SingleAgentOwnshipEnv")
