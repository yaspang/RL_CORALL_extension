import numpy as np
from pathlib import Path

# Load case 1 episode 0
ep_file = Path('corall_baseline_case1_20260419-202941/seed_0/episode_histories/baseline_case1_seed0_ep000.npz')
if ep_file.exists():
    data = np.load(ep_file, allow_pickle=True)
    X_all = np.asarray(data['X_all'], dtype=float)
    
    NMI = 1852.0
    
    # Get initial and final positions
    x_init_m = X_all[0, 0, 0]
    y_init_m = X_all[0, 0, 1]
    x_final_m = X_all[-1, 0, 0]
    y_final_m = X_all[-1, 0, 1]
    
    # Route: from (0, 0) to (2 NMI, 0) = (0, 0) to (3704m, 0)
    expected_goal_x_m = 2.0 * NMI
    expected_goal_y_m = 0.0
    
    print(f"Route: (0, 0) → ({expected_goal_x_m:.1f}m, {expected_goal_y_m:.1f}m) [2 NMI straight]")
    print(f"\nInitial position: ({x_init_m:.1f}m, {y_init_m:.1f}m)")
    print(f"Final position:   ({x_final_m:.1f}m, {y_final_m:.1f}m)")
    print(f"Distance from goal: {np.hypot(expected_goal_x_m - x_final_m, expected_goal_y_m - y_final_m):.1f}m")
    print(f"Distance from goal (acceptance radius): {np.hypot(expected_goal_x_m - x_final_m, expected_goal_y_m - y_final_m) / NMI * 1852:.1f}m (threshold: 200m)")
    
    # Calculate progress
    route_vec = np.array([expected_goal_x_m, expected_goal_y_m])
    pos_vec = np.array([x_final_m, y_final_m])
    route_len_m = np.linalg.norm(route_vec)
    progress_m = pos_vec @ (route_vec / route_len_m)
    progress_pct = (progress_m / route_len_m) * 100
    
    print(f"\nProgress: {progress_m:.1f}m / {route_len_m:.1f}m = {progress_pct:.1f}%")
    print(f"\nShould agent reach goal? dist_to_goal={(np.hypot(expected_goal_x_m - x_final_m, expected_goal_y_m - y_final_m)):.1f}m, threshold=200m → {'YES' if np.hypot(expected_goal_x_m - x_final_m, expected_goal_y_m - y_final_m) < 200 else 'NO'}")
else:
    print('Episode file not found')
