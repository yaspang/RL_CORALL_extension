"""
Stub for AABB broad-phase spatial optimization.

The full implementation is a future optimization for large multi-agent scenarios
(5+ agents). For the current training setup (2-4 agents), enable_aabb_filtering
defaults to False so this stub is never called.
"""


class AABBBroadPhase:
    """Axis-Aligned Bounding Box broad-phase collision filter (stub)."""

    @staticmethod
    def get_overlapping_pairs(X_all, radius_m):
        """Return all agent pairs (naive fallback — replace with spatial index later)."""
        n = X_all.shape[0]
        return [(a, b) for a in range(n) for b in range(a + 1, n)]
