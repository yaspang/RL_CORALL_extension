#!/usr/bin/env python
"""Verify waypoint generation and collision scenario logic for Case 6."""

import sys
sys.path.insert(0, 'third_party/CORALL/src')

import numpy as np
from maritime_rl_pkg.env_multi_agent_ppo import MultiShipParallelEnv

# Create environment
env = MultiShipParallelEnv(
    case_number=6,
    route_len_nmi=2.0,
    sim_time=490.0,
)

obs, info = env.reset()

print("=" * 80)
print("CASE 6 VERIFICATION: Waypoints and Collision Scenario")
print("=" * 80)
print()

# Get state
X_all = env.X_all
Xwpt_all = env.Xwpt_all
Ywpt_all = env.Ywpt_all

NMI = 1852.0

print("INITIAL POSITIONS AND HEADINGS:")
print()
for k in range(env.n_agents):
    x_m, y_m = X_all[k, 0], X_all[k, 1]
    sin_psi, cos_psi = X_all[k, 2], X_all[k, 3]
    u = X_all[k, 5]
    
    psi = np.arctan2(sin_psi, cos_psi)
    psi_deg = np.degrees(psi)
    
    label = "Ownship" if k == 0 else f"Target {k-1}"
    print(f"{label}:")
    print(f"  Start: ({x_m:.1f}, {y_m:.1f}) m = ({x_m/NMI:.3f}, {y_m/NMI:.3f}) nmi")
    print(f"  Heading: {psi_deg:.1f} deg")
    print(f"  Speed: {u:.1f} m/s")
    
    x0_nmi, y0_nmi = Xwpt_all[k][0], Ywpt_all[k][0]
    x1_nmi, y1_nmi = Xwpt_all[k][1], Ywpt_all[k][1]
    
    print(f"  Waypoint start (nmi): ({x0_nmi:.3f}, {y0_nmi:.3f})")
    print(f"  Waypoint goal (nmi): ({x1_nmi:.3f}, {y1_nmi:.3f})")
    
    x1_m, y1_m = x1_nmi * NMI, y1_nmi * NMI
    dist_nmi = np.sqrt((x1_nmi - x0_nmi)**2 + (y1_nmi - y0_nmi)**2)
    print(f"  Goal in meters: ({x1_m:.1f}, {y1_m:.1f})")
    print(f"  Route distance: {dist_nmi:.3f} nmi")
    print()

print("=" * 80)
print("COLLISION SCENARIO ANALYSIS:")
print("=" * 80)
print()

# Simulate motion for first 60 seconds
dt = 1.0
max_t = 60
positions_over_time = {k: [] for k in range(env.n_agents)}

for t in range(max_t):
    for k in range(env.n_agents):
        x_m, y_m = X_all[k, 0], X_all[k, 1]
        sin_psi, cos_psi = X_all[k, 2], X_all[k, 3]
        u = X_all[k, 5]
        psi = np.arctan2(sin_psi, cos_psi)
        
        # Simple kinematics: move forward at constant heading and speed
        x_new = x_m + u * np.cos(psi) * dt
        y_new = y_m + u * np.sin(psi) * dt
        
        positions_over_time[k].append((x_new, y_new))

print("Position evolution (every 10 seconds):")
print()
for t in [0, 10, 20, 30, 40, 50, 60]:
    print(f"t = {t} s:")
    for k in range(env.n_agents):
        if t < len(positions_over_time[k]):
            x, y = positions_over_time[k][t]
            label = "Ownship" if k == 0 else f"Target {k-1}"
            print(f"  {label}: ({x/NMI:7.3f}, {y/NMI:7.3f}) nmi = ({x:8.1f}, {y:8.1f}) m")
    print()

# Calculate distances at key times
print("=" * 80)
print("INTER-AGENT DISTANCES:")
print("=" * 80)
print()

for t in [0, 10, 20, 30, 40, 50, 60]:
    print(f"t = {t} s:")
    
    if t >= len(positions_over_time[0]):
        break
        
    for i in range(env.n_agents):
        for j in range(i+1, env.n_agents):
            xi, yi = positions_over_time[i][t]
            xj, yj = positions_over_time[j][t]
            
            dist_m = np.sqrt((xj - xi)**2 + (yj - yi)**2)
            dist_nmi = dist_m / NMI
            
            label_i = "Own" if i == 0 else f"Tgt{i}"
            label_j = "Own" if j == 0 else f"Tgt{j}"
            
            collision_threshold_m = 30.0  # ~LOA
            status = "COLLISION!" if dist_m < collision_threshold_m else ""
            
            print(f"  {label_i}-{label_j}: {dist_m:7.1f} m ({dist_nmi:6.3f} nmi) {status}")
    print()

print("=" * 80)
print("REWARD FUNCTION VERIFICATION:")
print("=" * 80)
print()

# Check if reward function encourages correct behaviors
print("Per-step reward components (example):")
print()
print("1. Waypoint following reward (r_along):")
print("   - Should be positive when moving toward waypoint")
print("   - Normalized by route distance")
print()
print("2. Risk penalty (r_risk):")
print("   - Should penalize collision risk")
print("   - Weight: -2.0")
print()
print("3. Collision penalty (r_collision):")
print("   - Should penalize actual collisions")  
print("   - Weight: -10.0")
print()
print("Question: Are these rewards sufficient to encourage")
print("safe waypoint following vs. converging to targets?")
print()

# Check by running one step through reward computation
print("=" * 80)
print("OBSERVATION SPACE CHECK:")
print("=" * 80)
print()

obs_own = obs["ship_0"]
print(f"Ownship observation: {obs_own.shape} features")
print(f"  Own state (0-6): {obs_own[0:7]}")
print(f"  Waypoint info (7-10): {obs_own[7:11]}")
print(f"    - Distance to goal (norm): {obs_own[7]:.4f}")
print(f"    - sin(bearing to goal): {obs_own[8]:.4f}")
print(f"    - cos(bearing to goal): {obs_own[9]:.4f}")
print(f"    - Goal progress: {obs_own[10]:.4f}")
print(f"  Per-other features (11+): {obs_own[11:]}")
print()

print("CONCLUSION:")
print("-" * 80)
print("Check items:")
print("1. Do waypoint endpoints make geometric sense?")
print("2. Do agents' paths actually intersect?")
print("3. Are distances between agents decreasing early on?")
print("4. Is waypoint distance info correctly in observation?")
