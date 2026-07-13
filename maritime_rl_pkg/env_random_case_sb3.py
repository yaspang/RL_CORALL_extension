"""
Random-case environment wrapper for generalized policy training.

This wrapper randomizes cases and seeds at each episode reset,
enabling training of a single policy that generalizes across difficulty levels
and random encounter geometries.

HARD-CASE OVERSAMPLING:
======================
Supports sensitive training with weighted case sampling:
  - 50% random curriculum cases (based on difficulty progression)
  - 30% hard Imazu cases + recent failures (cases 13, 18, 20, 21 + tracked failures)
  - 20% easy cases (prevent catastrophic forgetting)

Enable via: hard_case_oversampling=True
Update failures via: env.update_hard_case_pool(failures_dict)

"""

from __future__ import annotations

from typing import Optional, Tuple, Any, List, Dict, Set
import numpy as np
import gymnasium as gym
import json
from pathlib import Path

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
    
    # Hard-case oversampling: known trouble cases (Imazu scenarios)
    HARD_CASES = [13, 18, 20, 21]  # Problematic cases to oversample
    EASY_CASES = [1, 5, 12]        # Representative easy cases for anti-forgetting
    
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
        hard_case_oversampling: bool = False,
        hard_case_pool_file: Optional[str] = None,
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
        
        # Hard-case oversampling (failure replay)
        self.hard_case_oversampling = hard_case_oversampling
        self.hard_case_pool_file = hard_case_pool_file
        self.hard_case_pool: Dict[int, Set[int]] = {}  # {case: {seed1, seed2, ...}}
        self.failure_case_seed_pairs: Set[Tuple[int, int]] = set()  # (case, seed) pairs that failed
        
        # Load hard-case pool from file if provided
        if hard_case_oversampling and hard_case_pool_file:
            self._load_hard_case_pool(hard_case_pool_file)
        
        # Initialize hard case pool with known problem cases
        if hard_case_oversampling and not self.hard_case_pool:
            for case in self.HARD_CASES:
                if case in self.cases_to_train:
                    self.hard_case_pool[case] = set(range(self.num_seeds))  # All seeds initially available
        
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

    def _load_hard_case_pool(self, pool_file: str):
        """Load hard case pool from JSON file."""
        try:
            pool_path = Path(pool_file)
            if pool_path.exists():
                with open(pool_path, 'r') as f:
                    pool_data = json.load(f)
                # Convert list of dicts to {case: set(seeds)}
                self.hard_case_pool = {
                    int(case): set(pool_data.get(str(case), list(range(self.num_seeds))))
                    for case in self.HARD_CASES
                    if int(case) in self.cases_to_train
                }
                print(f"[HardCaseOversampling] Loaded pool from {pool_file}: {len(self.hard_case_pool)} cases")
        except Exception as e:
            print(f"[HardCaseOversampling] Failed to load pool from {pool_file}: {e}")
    
    def update_hard_case_pool(self, failures_dict: Dict[int, List[int]]):
        """
        Update hard case pool with newly failed cases/seeds.
        
        Args:
            failures_dict: {case: [seed1, seed2, ...]} of cases/seeds to add to hard pool
        """
        if not self.hard_case_oversampling:
            return
        
        for case, seeds in failures_dict.items():
            case = int(case)
            if case not in self.cases_to_train:
                continue
            
            if case not in self.hard_case_pool:
                self.hard_case_pool[case] = set()
            
            # Add failed seeds to the pool
            self.hard_case_pool[case].update(int(s) for s in seeds)
        
        print(f"[HardCaseOversampling] Updated pool with {len(failures_dict)} cases")
        if self.hard_case_pool_file:
            self._save_hard_case_pool(self.hard_case_pool_file)
    
    def _save_hard_case_pool(self, pool_file: str):
        """Save hard case pool to JSON file."""
        try:
            pool_data = {
                str(case): list(seeds)
                for case, seeds in self.hard_case_pool.items()
            }
            pool_path = Path(pool_file)
            pool_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pool_path, 'w') as f:
                json.dump(pool_data, f, indent=2)
        except Exception as e:
            print(f"[HardCaseOversampling] Failed to save pool to {pool_file}: {e}")

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
        
        With hard_case_oversampling:
          - 50% random curriculum cases
          - 30% hard Imazu cases + recent failures
          - 20% easy cases (anti-forgetting)
        
        Without oversampling:
          - Uniform or curriculum-weighted sampling from available cases
        """
        # Use provided seed or master RNG for case/seed sequence
        if seed is not None:
            local_rng = np.random.default_rng(seed)
        else:
            local_rng = self.rng

        # Select case using appropriate strategy
        if self.hard_case_oversampling:
            self.current_case, self.current_seed = self._sample_case_with_hard_oversampling(local_rng)
        else:
            # Original curriculum-based sampling
            available_cases = self._get_curriculum_available_cases()
            if not available_cases:
                raise ValueError("No available cases for sampling")
            
            case_weights = self._get_case_sampling_weights()
            case_probs = np.array([case_weights.get(c, 1.0 / len(available_cases)) for c in available_cases])
            prob_sum = case_probs.sum()
            if prob_sum > 0:
                case_probs = case_probs / prob_sum
            else:
                case_probs = np.ones(len(available_cases)) / len(available_cases)
            
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
    
    def _sample_case_with_hard_oversampling(self, local_rng: np.random.Generator) -> Tuple[int, int]:
        """
        Sample case and seed using stratified hard-case oversampling:
          - 50% from curriculum cases
          - 30% from hard cases (Imazu + tracked failures)
          - 20% from easy cases
        
        Returns: (case, seed)
        """
        # Get curriculum cases
        curriculum_cases = self._get_curriculum_available_cases()
        
        # Get hard cases (intersection of HARD_CASES with cases_to_train)
        hard_cases_available = [c for c in self.HARD_CASES if c in self.cases_to_train]
        
        # Add dynamically tracked failures to hard pool
        for case, seeds in self.hard_case_pool.items():
            if case in self.cases_to_train and case not in hard_cases_available:
                hard_cases_available.append(case)
        
        # Get easy cases
        easy_cases_available = [c for c in self.EASY_CASES if c in self.cases_to_train]
        
        # Stratified sampling: 50% curriculum, 30% hard, 20% easy
        pool_selector = local_rng.random()
        
        if pool_selector < 0.5:
            # 50%: Sample from curriculum
            if curriculum_cases:
                case = int(local_rng.choice(curriculum_cases))
            elif hard_cases_available:
                case = int(local_rng.choice(hard_cases_available))
            else:
                case = int(local_rng.choice(self.cases_to_train))
        
        elif pool_selector < 0.8:
            # 30%: Sample from hard cases
            if hard_cases_available:
                case = int(local_rng.choice(hard_cases_available))
            elif curriculum_cases:
                case = int(local_rng.choice(curriculum_cases))
            else:
                case = int(local_rng.choice(self.cases_to_train))
        
        else:
            # 20%: Sample from easy cases (anti-forgetting)
            if easy_cases_available:
                case = int(local_rng.choice(easy_cases_available))
            elif curriculum_cases:
                case = int(local_rng.choice(curriculum_cases))
            else:
                case = int(local_rng.choice(self.cases_to_train))
        
        # For hard cases with tracked failures, preferentially sample failed seeds
        if case in self.hard_case_pool and self.hard_case_pool[case]:
            # 70% chance to sample from failed seeds, 30% random
            if local_rng.random() < 0.7:
                seed = int(local_rng.choice(list(self.hard_case_pool[case])))
            else:
                seed = int(local_rng.integers(0, self.num_seeds))
        else:
            seed = int(local_rng.integers(0, self.num_seeds))
        
        return case, seed
        
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
