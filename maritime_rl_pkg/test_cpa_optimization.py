"""
Test AABB broad-phase filtering vs direct CPA calculation.

Compare performance of:
1. Naive: All pairs -> CPA calculation
2. AABB broad-phase: Filter by spatial bounds first -> CPA only for overlapping pairs
3. Measure timing for different agent counts (2-10 agents)
"""

import numpy as np
import time
from pathlib import Path

# ensure CORALL repository relative imports resolve
from .path_setup import ensure_paths
ensure_paths()

from risk_assessment.cpa_calculations_0speed import cpa_calculations_0speed
from risk_assessment.risk_calculations import risk_calculations

NMI = 1852.0
results = []

class AABBBroadPhase:
    """AABB-based broad-phase spatial filtering for pairwise CPA calculations."""
    
    @staticmethod
    def check_aabb_overlap(x1, y1, r1, x2, y2, r2):
        """
        Check if two AABBs (axis-aligned bounding boxes) overlap.
        
        Each agent gets a bounding box extending from (x - r, y - r) to (x + r, y + r)
        where r is a "collision radius" (e.g., influence distance).
        
        Returns: True if AABBs overlap, False otherwise
        """
        # AABB 1
        x1_min, x1_max = x1 - r1, x1 + r1
        y1_min, y1_max = y1 - r1, y1 + r1
        
        # AABB 2
        x2_min, x2_max = x2 - r2, x2 + r2
        y2_min, y2_max = y2 - r2, y2 + r2
        
        # Check overlap on both axes
        x_overlap = (x1_min <= x2_max) and (x2_min <= x1_max)
        y_overlap = (y1_min <= y2_max) and (y2_min <= y1_max)
        
        return x_overlap and y_overlap
    
    @staticmethod
    def get_pairs_within_aabb(X_all, radius_m):
        """
        Get list of agent pairs whose AABBs overlap.
        
        Args:
            X_all: Array of shape (n_agents, 6) with [x, y, psi, r, b, u]
            radius_m: Collision radius for each AABB (e.g., 1000m = search distance)
        
        Returns:
            List of (a, b) pairs where a < b
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


def compute_all_pairs_naive(X_all, prev_X_all, dt=0.5):
    """
    Naive approach: compute CPA for all pairs without filtering.
    
    Returns: dict of {(a, b): (dcpa, tcpa)}
    """
    n_agents = X_all.shape[0]
    results = {}
    
    for a in range(n_agents):
        xa, ya = X_all[a, 0], X_all[a, 1]
        psi_a, u_a = X_all[a, 2], X_all[a, 5]
        vxa = u_a * np.cos(psi_a)
        vya = u_a * np.sin(psi_a)
        
        for b in range(a + 1, n_agents):
            xb, yb = X_all[b, 0], X_all[b, 1]
            psi_b, u_b = X_all[b, 2], X_all[b, 5]
            vxb = u_b * np.cos(psi_b)
            vyb = u_b * np.sin(psi_b)
            
            dist = np.hypot(xa - xb, ya - yb)
            
            # Use velocity-based CPA
            dcpa, tcpa, vrel, alpha, psi_vrel = cpa_calculations_0speed(
                xa, ya, xb, yb,
                vxa, vya, vxb, vyb,
                dist
            )
            
            results[(a, b)] = (dcpa, tcpa)
    
    return results


def compute_all_pairs_aabb(X_all, prev_X_all, dt=0.5, aabb_radius_m=1500.0):
    """
    Optimized approach: AABB broad-phase filtering before CPA.
    
    Args:
        aabb_radius_m: Collision detection radius (e.g., 1500m = ~0.8 nmi)
    
    Returns: dict of {(a, b): (dcpa, tcpa)}
    """
    n_agents = X_all.shape[0]
    results = {}
    
    # Broad-phase: get potentially colliding pairs
    overlapping_pairs = AABBBroadPhase.get_pairs_within_aabb(X_all, aabb_radius_m)
    
    # Narrow-phase: compute CPA only for overlapping pairs
    for a, b in overlapping_pairs:
        xa, ya = X_all[a, 0], X_all[a, 1]
        psi_a, u_a = X_all[a, 2], X_all[a, 5]
        vxa = u_a * np.cos(psi_a)
        vya = u_a * np.sin(psi_a)
        
        xb, yb = X_all[b, 0], X_all[b, 1]
        psi_b, u_b = X_all[b, 2], X_all[b, 5]
        vxb = u_b * np.cos(psi_b)
        vyb = u_b * np.sin(psi_b)
        
        dist = np.hypot(xa - xb, ya - yb)
        
        # Use velocity-based CPA
        dcpa, tcpa, vrel, alpha, psi_vrel = cpa_calculations_0speed(
            xa, ya, xb, yb,
            vxa, vya, vxb, vyb,
            dist
        )
        
        results[(a, b)] = (dcpa, tcpa)
    
    return results


def generate_random_scenario(n_agents, extent_m=5000.0, speed_range=(5.0, 10.0), seed=42):
    """Generate random agent positions and velocities for testing."""
    rng = np.random.default_rng(seed)
    
    X_all = np.zeros((n_agents, 6), dtype=float)
    
    for k in range(n_agents):
        x = rng.uniform(-extent_m, extent_m)
        y = rng.uniform(-extent_m, extent_m)
        psi = rng.uniform(-np.pi, np.pi)
        u = rng.uniform(*speed_range)
        
        X_all[k] = [x, y, psi, 0.0, 0.0, u]
    
    return X_all


def benchmark_approach(compute_func, X_all, prev_X_all, n_iterations=100, **kwargs):
    """Benchmark a CPA computation approach."""
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        results = compute_func(X_all, prev_X_all, **kwargs)
    t_elapsed = time.perf_counter() - t0
    
    return t_elapsed, results


def main():
    print("=" * 80)
    print("BENCHMARK: CPA Optimization Test (AABB Broad-Phase Filtering)")
    print("=" * 80)
    
    # Test across different agent counts
    agent_counts = [2, 3, 5, 10, 15]
    n_iterations = 100
    aabb_radius_m = 1500.0  # ~0.81 nmi
    
    print(f"\nRunning {n_iterations} iterations per test...\n")
    print(f"AABB Collision Radius: {aabb_radius_m:.0f} m ({aabb_radius_m / NMI:.2f} nmi)\n")
    
    results_summary = []
    
    for n_agents in agent_counts:
        print(f"Testing with {n_agents} agents ({(n_agents * (n_agents - 1) // 2)} potential pairs):")
        
        # Generate scenario
        X_all = generate_random_scenario(n_agents)
        prev_X_all = X_all.copy()
        
        # Benchmark naive approach
        t_naive, results_naive = benchmark_approach(
            compute_all_pairs_naive, X_all, prev_X_all, n_iterations
        )
        
        # Benchmark AABB approach
        t_aabb, results_aabb = benchmark_approach(
            compute_all_pairs_aabb, X_all, prev_X_all, n_iterations,
            aabb_radius_m=aabb_radius_m
        )
        
        # Count how many pairs were filtered
        n_potential_pairs = n_agents * (n_agents - 1) // 2
        n_checked_pairs = len(results_aabb)
        pairs_filtered = n_potential_pairs - n_checked_pairs
        filter_ratio = (pairs_filtered / n_potential_pairs * 100) if n_potential_pairs > 0 else 0
        
        speedup = t_naive / t_aabb if t_aabb > 0 else float('inf')
        time_per_iter_naive_us = (t_naive / n_iterations) * 1e6
        time_per_iter_aabb_us = (t_aabb / n_iterations) * 1e6
        
        print(f"  Naive:       {t_naive:.4f} s ({time_per_iter_naive_us:.2f} µs/iter)")
        print(f"  AABB:        {t_aabb:.4f} s ({time_per_iter_aabb_us:.2f} µs/iter)")
        print(f"  Speedup:     {speedup:.2f}x")
        print(f"  Pairs filtered: {n_checked_pairs}/{n_potential_pairs} checked ({filter_ratio:.1f}% skipped)")
        print()
        
        results_summary.append({
            'n_agents': n_agents,
            'n_potential_pairs': n_potential_pairs,
            'n_checked_pairs': n_checked_pairs,
            'filter_ratio': filter_ratio,
            't_naive': t_naive,
            't_aabb': t_aabb,
            'speedup': speedup,
        })
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Agents':<8} {'Pairs':<12} {'Checked':<10} {'Filtered %':<12} {'Speedup':<10}")
    print("-" * 80)
    for r in results_summary:
        print(
            f"{r['n_agents']:<8} "
            f"{r['n_potential_pairs']:<12} "
            f"{r['n_checked_pairs']:<10} "
            f"{r['filter_ratio']:<12.1f} "
            f"{r['speedup']:<10.2f}x"
        )
    
    print("\n" + "=" * 80)
    print("OBSERVATIONS:")
    print("=" * 80)
    avg_speedup = np.mean([r['speedup'] for r in results_summary])
    avg_filter = np.mean([r['filter_ratio'] for r in results_summary])
    print(f"• Average speedup: {avg_speedup:.2f}x")
    print(f"• Average filter efficiency: {avg_filter:.1f}% of pairs skipped")
    
    if avg_speedup > 1.2:
        print("✓ AABB optimization is WORTHWHILE - provides consistent speedup")
    elif avg_speedup > 1.05:
        print("~ AABB optimization is MARGINAL - small speedup, overhead minimal")
    else:
        print("✗ AABB optimization is NOT WORTHWHILE - overhead outweighs benefit")
    
    print("\nCONCLUSION:")
    print("-" * 80)
    if n_agents <= 3:
        print("Current scenario (1-3 agents): Stick with naive approach for simplicity.")
        print("  - Only 1-3 pairs, AABB overhead not justified")
    elif n_agents <= 10:
        print("Medium scenario (5-10 agents): AABB becomes beneficial.")
        print("  - Filter ratio increases, general speedup 1.5-3x typical")
    else:
        print("Large scenario (10+ agents): AABB is ESSENTIAL.")
        print("  - Quadratic pair growth, heavy filtering needed for real-time")


if __name__ == "__main__":
    main()
