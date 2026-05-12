"""
Reward normalization wrapper for multi-scenario training.

Normalizes rewards by scenario type (number of ships) so that:
- 2-ship, 3-ship, and 4-ship cases contribute equally to policy learning
- Raw reward magnitudes don't dominate training for any scenario
- Training convergence is smoother across heterogeneous difficulty levels

USAGE:
======
env = RandomCaseEnv(cases_to_train=[1, 6, 21], ...)
env = RewardNormalizerByShipCount(env)
env = Monitor(env)
env = EpisodeReturnTracker(env)

HOW IT WORKS:
=============
1. Tracks running statistics (mean, std) of episode returns per ship count
2. Normalizes individual rewards based on observed statistics
3. Uses standard reward normalization: (r - mean) / (std + 1e-8)
4. Adapts statistics online as training progresses
5. Outputs diagnostic info at episode ends

EXPECTED EFFECT:
================
- Smoother, more stable training across all scenario types
- Better convergence signal visible in per-scenario return curves
- Policy learns balanced strategies rather than specializing on easy cases
- Training plot should show less extreme variance
"""

from typing import Tuple, Optional, Dict
import numpy as np
import gymnasium as gym


class RewardNormalizerByShipCount(gym.Wrapper):
    """
    Wraps RandomCaseEnv to normalize rewards by scenario difficulty (ship count).
    
    Maintains running statistics of episode returns per ship count and normalizes
    individual step rewards based on these statistics.
    
    Attributes:
        ship_count_stats: Dict mapping ship count -> (mean, std, count) of episode returns
        case_to_ships: Dict mapping case number -> number of ships
        current_episode_return: Running sum of rewards in current episode
        normalize_rewards: Whether to apply normalization (for ablation studies)
    """
    
    # Mapping from case number to expected number of ships
    # Case 1: 2 ships, Case 6: 3 ships, Case 21: 4 ships
    CASE_TO_SHIPS = {
        1: 2,
        6: 3,
        21: 4,
    }
    
    def __init__(self, env, normalize_rewards: bool = True, verbose: bool = False):
        """
        Initialize reward normalizer.
        
        Args:
            env: RandomCaseEnv wrapper instance
            normalize_rewards: If True, apply normalization; if False, pass through
            verbose: If True, print statistics at episode boundaries
        """
        super().__init__(env)
        
        self.normalize_rewards = normalize_rewards
        self.verbose = verbose
        
        # Running statistics per ship count
        self.ship_count_stats: Dict[int, Dict[str, float]] = {
            2: {"mean": 0.0, "std": 1.0, "count": 0, "min": float('inf'), "max": float('-inf')},
            3: {"mean": 0.0, "std": 1.0, "count": 0, "min": float('inf'), "max": float('-inf')},
            4: {"mean": 0.0, "std": 1.0, "count": 0, "min": float('inf'), "max": float('-inf')},
        }
        
        # Episode tracking
        self.current_episode_return = 0.0
        self.episode_count = 0
        self.last_reset_info = {}
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment and prepare for new episode."""
        obs, info = self.env.reset(seed=seed, options=options)
        self.current_episode_return = 0.0
        self.last_reset_info = info.copy()
        return obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step environment with reward normalization.
        
        Returns:
            obs, normalized_reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track raw reward for statistics
        raw_reward = reward
        self.current_episode_return += raw_reward
        
        # Normalize reward based on current scenario
        if self.normalize_rewards:
            current_case = info.get("case", 1)
            ships = self.CASE_TO_SHIPS.get(current_case, 2)
            reward = self._normalize_reward(raw_reward, ships)
        
        # Episode done: update statistics
        if terminated or truncated:
            current_case = info.get("case", 1)
            ships = self.CASE_TO_SHIPS.get(current_case, 2)
            self._update_episode_statistics(self.current_episode_return, ships)
            
            if self.verbose:
                self._print_episode_summary(current_case, ships)
        
        info["raw_reward"] = raw_reward
        info["normalized_reward"] = reward
        info["episode_return_so_far"] = self.current_episode_return
        
        return obs, reward, terminated, truncated, info
    
    def _normalize_reward(self, reward: float, ship_count: int) -> float:
        """
        Normalize reward using running statistics for ship count.
        
        Normalization: z = (r - mean) / (std + eps)
        This centers rewards around 0 with std ~1 for each ship count.
        """
        stats = self.ship_count_stats[ship_count]
        mean = stats["mean"]
        std = max(stats["std"], 1e-8)  # Prevent division by zero
        
        normalized = (reward - mean) / std
        
        # Clip to reasonable range to prevent extreme values
        normalized = np.clip(normalized, -5.0, 5.0)
        
        return float(normalized)
    
    def _update_episode_statistics(self, episode_return: float, ship_count: int):
        """Update running mean/std for episode return of given ship count."""
        stats = self.ship_count_stats[ship_count]
        
        # Welford's online algorithm for mean/std
        n = stats["count"]
        old_mean = stats["mean"]
        old_std = stats["std"]
        
        # Update count
        n_new = n + 1
        stats["count"] = n_new
        
        # Update min/max
        stats["min"] = min(stats["min"], episode_return)
        stats["max"] = max(stats["max"], episode_return)
        
        # Update mean
        delta = episode_return - old_mean
        new_mean = old_mean + delta / n_new
        stats["mean"] = new_mean
        
        # Update std (variance)
        if n_new > 1:
            # Simplified: track rolling std over all returns
            # For more accuracy, consider implementing full Welford variance update
            all_returns = getattr(self, f'_returns_{ship_count}', [])
            if not hasattr(self, f'_returns_{ship_count}'):
                setattr(self, f'_returns_{ship_count}', all_returns)
            
            all_returns.append(episode_return)
            if len(all_returns) > 0:
                stats["std"] = float(np.std(all_returns)) if len(all_returns) > 1 else 1.0
        
        self.episode_count += 1
    
    def _print_episode_summary(self, case: int, ship_count: int):
        """Print episode summary with statistics."""
        stats = self.ship_count_stats[ship_count]
        n_episodes = stats["count"]
        
        print(
            f"[Episode {self.episode_count:4d}] Case {case} ({ship_count} ships) | "
            f"Return: {self.current_episode_return:10.1f} | "
            f"Mean return: {stats['mean']:10.1f} | "
            f"Std: {stats['std']:8.1f} | "
            f"Episodes: {n_episodes:3d}"
        )
    
    def get_normalization_stats(self) -> Dict[int, Dict[str, float]]:
        """Get current normalization statistics."""
        return {
            ships: {
                "mean": stats["mean"],
                "std": stats["std"],
                "count": stats["count"],
                "min": stats["min"],
                "max": stats["max"],
            }
            for ships, stats in self.ship_count_stats.items()
        }
    
    def print_normalization_stats(self):
        """Print current normalization statistics."""
        print("\n" + "=" * 80)
        print("REWARD NORMALIZATION STATISTICS")
        print("=" * 80)
        
        for ships in [2, 3, 4]:
            stats = self.ship_count_stats[ships]
            n = stats["count"]
            print(f"\n{ships}-ship scenarios ({n} episodes):")
            print(f"  Mean return:     {stats['mean']:10.1f}")
            print(f"  Std return:      {stats['std']:10.1f}")
            print(f"  Min return:      {stats['min']:10.1f}")
            print(f"  Max return:      {stats['max']:10.1f}")
            if n > 0:
                print(f"  Range:           {stats['max'] - stats['min']:10.1f}")
        
        print("\n" + "=" * 80 + "\n")
