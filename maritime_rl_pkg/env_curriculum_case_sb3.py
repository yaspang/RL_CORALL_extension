"""
Curriculum-based case environment for progressive complexity learning.

This wrapper implements curriculum learning by progressively unlocking cases
based on training step count:
- Phase 1 (0-500k steps): Case 1 only (2-ship scenarios)
- Phase 2 (500k-1M steps): Cases 1 + 6 (2-ship + 3-ship)
- Phase 3 (1M+steps): Cases 1 + 6 + 21 (all scenarios 2/3/4-ship)

Smooth transitions with probabilistic blending prevent distribution shift.
"""

from typing import List
import numpy as np
import gymnasium as gym


class CurriculumCaseEnv(gym.Wrapper):
    """
    Curriculum learning wrapper that progressively unlocks cases by agent count.
    """

    # Curriculum phases: (step_threshold, available_cases)
    PHASES = [
        (0, [1]),              # Phase 1: 2-ship only
        (500000, [1, 6]),      # Phase 2: 2-ship + 3-ship
        (1000000, [1, 6, 21]), # Phase 3: all scenarios
    ]
    TRANSITION_WINDOW = 50000  # steps for smooth blending between phases

    # Observation sizes per case (ownship=8 + obstacles*6)
    CASE_OBS_SIZES = {
        1: 8 + 1*6,   # 1 obstacle = 14
        6: 8 + 2*6,   # 2 obstacles = 20
        21: 8 + 3*6,  # 3 obstacles = 26
    }
    MAX_OBS_SIZE = 29  # v8: 8 (own) + 3 (goal bearing/distance) + 18 (3 obstacles × 6)

    def __init__(
        self,
        env,
        cases_to_train: List[int] = [1, 6, 21],
        num_seeds: int = 100,
        master_seed: int = None,
    ):
        """
        Args:
            env: Base environment to wrap
            cases_to_train: Cases to make available in curriculum (default: all)
            num_seeds: Number of seeds per case (0-99)
            master_seed: For reproducible case/seed sequences
        """
        super().__init__(env)
        self.num_seeds = num_seeds
        self.rng = np.random.RandomState(master_seed)
        
        # Filter available cases
        self.cases_to_train = [c for c in cases_to_train if c in [1, 6, 21]]
        if not self.cases_to_train:
            self.cases_to_train = [1, 6, 21]
        
        self.current_step = 0
        self.current_case = self.cases_to_train[0]
        self.current_seed = 0
        self._sample_new_episode()

    def _get_available_cases(self, step: int) -> List[int]:
        """Get available cases at the given step based on curriculum."""
        # Find current and next phase
        current_phase_idx = 0
        for i, (threshold, _) in enumerate(self.PHASES):
            if step >= threshold:
                current_phase_idx = i
        
        if current_phase_idx >= len(self.PHASES) - 1:
            # Last phase
            phase_cases = self.PHASES[-1][1]
        else:
            # Check if in transition window
            next_threshold = self.PHASES[current_phase_idx + 1][0]
            if step >= next_threshold - self.TRANSITION_WINDOW:
                # Transition: blend both phases
                current_cases = self.PHASES[current_phase_idx][1]
                next_cases = self.PHASES[current_phase_idx + 1][1]
                phase_cases = list(set(current_cases + next_cases))
            else:
                phase_cases = self.PHASES[current_phase_idx][1]
        
        # Filter by requested cases
        return [c for c in phase_cases if c in self.cases_to_train]

    def _sample_new_episode(self):
        """Sample case and seed for next episode."""
        available = self._get_available_cases(self.current_step)
        if not available:
            available = self.cases_to_train
        
        self.current_case = self.rng.choice(available)
        self.current_seed = self.rng.randint(0, self.num_seeds)

    def _pad_observation(self, obs: np.ndarray) -> np.ndarray:
        """Pad variable-sized observation to MAX_OBS_SIZE."""
        if len(obs) >= self.MAX_OBS_SIZE:
            return obs[:self.MAX_OBS_SIZE]
        padded = np.zeros(self.MAX_OBS_SIZE, dtype=obs.dtype)
        padded[:len(obs)] = obs
        return padded

    def reset(self, **kwargs):
        """Reset with curriculum-selected case and seed."""
        self._sample_new_episode()
        obs, info = self.env.reset(
            case=self.current_case,
            seed=self.current_seed,
            **kwargs
        )
        # Pad observation to fixed size
        obs = self._pad_observation(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        # Pad observation to fixed size
        obs = self._pad_observation(obs)
        return obs, reward, terminated, truncated, info

    def update_step(self, step: int):
        """Called from training loop to update curriculum phase."""
        self.current_step = step
