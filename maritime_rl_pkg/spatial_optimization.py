"""
Spatial optimization utilities for multi-agent environments.

Provides AABB (axis-aligned bounding box) broad-phase filtering for efficient
pairwise collision detection across many agents.

Apply in increasingly dense environments for multi-agent control 
"""

import numpy as np


class AABBBroadPhase:
    """
    AABB-based broad-phase spatial filtering for pairwise CPA calculations.
    
    Reduces number of expensive narrow-phase CPA computations by filtering pairs
    using axis-aligned bounding boxes. Only overlapping pairs proceed to full CPA.
    
    Typically 1.5-3x faster for 5+ agents, negligible for 2-3 agents.
    """
    
    @staticmethod
    def check_aabb_overlap(x1, y1, r1, x2, y2, r2):
        """
        Check if two AABBs (axis-aligned bounding boxes) overlap.
        
        Each agent gets a bounding box extending from (x - r, y - r) to (x + r, y + r)
        where r is a "collision radius" for look-ahead distance.
        
        Args:
            x1, y1: Position of agent 1 (meters)
            r1: AABB radius for agent 1 (meters)
            x2, y2: Position of agent 2 (meters)
            r2: AABB radius for agent 2 (meters)
        
        Returns:
            bool: True if AABBs overlap on both x and y axes, False otherwise
        """
        # AABB bounds
        x1_min, x1_max = x1 - r1, x1 + r1
        y1_min, y1_max = y1 - r1, y1 + r1
        
        x2_min, x2_max = x2 - r2, x2 + r2
        y2_min, y2_max = y2 - r2, y2 + r2
        
        # Check overlap on both axes
        x_overlap = (x1_min <= x2_max) and (x2_min <= x1_max)
        y_overlap = (y1_min <= y2_max) and (y2_min <= y1_max)
        
        return x_overlap and y_overlap
    
    @staticmethod
    def get_overlapping_pairs(X_all, radius_m):
        """
        Get list of agent pairs whose AABBs overlap.
        
        Args:
            X_all: Array of shape (n_agents, 6) with state [x, y, psi, r, b, u]
            radius_m: Collision radius for each AABB (e.g., 1500m for broad-phase cutoff)
        
        Returns:
            List of (a, b) tuples where a < b, representing overlapping AABB pairs
        
        Example:
            >>> X_all = np.array([[0, 0, 0, 0, 0, 5], [1000, 0, 0, 0, 0, 5]])  # 1000m apart
            >>> pairs = AABBBroadPhase.get_overlapping_pairs(X_all, radius_m=1500)
            >>> len(pairs)
            1  # Pair (0, 1) overlaps
        """
        n_agents = X_all.shape[0]
        overlapping_pairs = []
        
        for a in range(n_agents):
            xa, ya = X_all[a, 0], X_all[a, 1]
            for b in range(a + 1, n_agents):
                xb, yb = X_all[b, 0], X_all[b, 1]
                
                if AABBBroadPhase.check_aabb_overlap(xa, ya, radius_m, xb, yb, radius_m):
                    overlapping_pairs.append((a, b))
        
        return overlapping_pairs
