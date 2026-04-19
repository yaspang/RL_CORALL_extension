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
    
    Usage:
        env = RandomCaseEnv(...)
        env = EpisodeReturnTracker(env)
        
        # During training, access returns:
        recent_returns = list(env.episode_returns)
        mean_return = np.mean(recent_returns) if recent_returns else 0.0
    """
    
    def __init__(self, env: gym.Env, max_episodes: int = 1000):
        """
        Initialize the episode return tracker.
        
        Args:
            env: The environment to wrap
            max_episodes: Maximum number of episodes to track in memory
        """
        super().__init__(env)
        self.episode_returns = deque(maxlen=max_episodes)
        self.episode_lengths = deque(maxlen=max_episodes)
        self.episode_reward = 0.0
        self.episode_length = 0
        self.lock = threading.Lock()
    
    def reset(self, **kwargs):
        """Reset the environment and episode tracking."""
        obs, info = self.env.reset(**kwargs)
        self.episode_reward = 0.0
        self.episode_length = 0
        return obs, info
    
    def step(self, action):
        """
        Step the environment and track episode return.
        
        Returns:
            obs, reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track episode progress
        self.episode_reward += reward
        self.episode_length += 1
        
        # If episode is done, record the return
        done = terminated or truncated
        if done:
            with self.lock:
                self.episode_returns.append(self.episode_reward)
                self.episode_lengths.append(self.episode_length)
            
            # Reset for next episode
            self.episode_reward = 0.0
            self.episode_length = 0
        
        return obs, reward, terminated, truncated, info
    
    def get_mean_return(self) -> float:
        """Get mean return of tracked episodes."""
        with self.lock:
            if len(self.episode_returns) == 0:
                return 0.0
            return float(np.mean(list(self.episode_returns)))
    
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
