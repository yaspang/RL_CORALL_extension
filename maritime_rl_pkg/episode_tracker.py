"""
Episode return tracking wrapper for direct access to episode statistics.

This wrapper captures episode returns directly as episodes complete,
bypassing SB3's logger/Monitor complexity.
"""

import gymnasium as gym
import numpy as np
from collections import deque
from typing import Any
import threading


class EpisodeReturnTracker(gym.Wrapper):
    """
    Wrapper that tracks episode returns directly.
    
    Stores completed episode returns in a thread-safe queue that can be
    accessed by training callbacks.
    """
    
    # Imazu case → ship count mapping
    _CASE_SHIPS = {
        **{c: 2 for c in range(1, 5)},     # Cases 1-4: 2 ships
        **{c: 3 for c in range(5, 12)},     # Cases 5-11: 3 ships
        **{c: 4 for c in range(12, 24)},    # Cases 12-23: 4 ships
    }
    
    def __init__(self, env: gym.Env, max_episodes: int = 1000):
        super().__init__(env)
        self.episode_returns = deque(maxlen=max_episodes)
        self.episode_lengths = deque(maxlen=max_episodes)
        # Per-ship-count tracking
        self.returns_by_ships = {2: deque(maxlen=500), 3: deque(maxlen=500), 4: deque(maxlen=500)}
        self.episode_reward = 0.0
        self.episode_length = 0
        self._current_case = None
        self.lock = threading.Lock()
    
    def reset(self, **kwargs):
        """Reset the environment and episode tracking."""
        obs, info = self.env.reset(**kwargs)
        self.episode_reward = 0.0
        self.episode_length = 0
        self._current_case = info.get("case", None)
        return obs, info
    
    def step(self, action):
        """Step the environment and track episode return."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_reward += reward
        self.episode_length += 1
        
        done = terminated or truncated
        if done:
            with self.lock:
                self.episode_returns.append(self.episode_reward)
                self.episode_lengths.append(self.episode_length)
                # Track per-ship-count
                case = info.get("case", self._current_case)
                if case is not None:
                    n_ships = self._CASE_SHIPS.get(int(case), None)
                    if n_ships in self.returns_by_ships:
                        self.returns_by_ships[n_ships].append(self.episode_reward)
            
            self.episode_reward = 0.0
            self.episode_length = 0
        
        return obs, reward, terminated, truncated, info
    
    def get_mean_return(self, window: int = 50) -> float:
        """Get mean return of recent episodes (windowed)."""
        with self.lock:
            if len(self.episode_returns) == 0:
                return 0.0
            recent = list(self.episode_returns)[-window:]
            return float(np.mean(recent))
    
    def get_mean_return_by_ships(self, window: int = 20) -> dict[int, float]:
        """Get windowed mean return per ship count (2, 3, 4)."""
        with self.lock:
            result = {}
            for n_ships, q in self.returns_by_ships.items():
                if len(q) > 0:
                    recent = list(q)[-window:]
                    result[n_ships] = float(np.mean(recent))
            return result
    
    def get_recent_returns(self, n: int | None = None) -> list[float]:
        """
        Get recent episode returns.
        
        Args:
            n: Number of recent episodes to return. If None, return all.
        
        Returns:
            List of recent episode returns.
        """
        with self.lock:
            returns_list = list(self.episode_returns)
        
        if n is None:
            return returns_list
        
        return returns_list[-n:] if len(returns_list) > 0 else []
    
    def clear_returns(self):
        """Clear all tracked returns."""
        with self.lock:
            self.episode_returns.clear()
            self.episode_lengths.clear()
