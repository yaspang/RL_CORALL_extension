import numpy as np
from pathlib import Path

ep_dir = Path('corall_baseline_case1_20260419-204446/seed_0/episode_histories')
ep_files = sorted(ep_dir.glob('*.npz'))[:1]

if ep_files:
    data = np.load(ep_files[0], allow_pickle=True)
    dist = np.asarray(data['pair_dist'], dtype=float)
    X_all = np.asarray(data['X_all'], dtype=float)
    
    print(f"Loaded: {ep_files[0].name}")
    print(f"Shape: pair_dist {dist.shape}, X_all {X_all.shape}\n")
    
    min_sep = np.nanmin(dist[:, 0, 1:])
    max_sep = np.nanmax(dist[:, 0, 1:])
    mean_sep = np.nanmean(dist[:, 0, 1:])
    
    threshold_3nmi = 3.0 * 1852.0
    active_timesteps = np.sum((dist[:, 0, 1:] > 0) & (dist[:, 0, 1:] <= threshold_3nmi))
    total_valid = np.sum(np.isfinite(dist[:, 0, 1:]))
    
    print(f"Ownship-to-obstacles distances:")
    print(f"  Min: {min_sep/1852.0:.3f} NMI ({min_sep:.0f}m)")
    print(f"  Max: {max_sep/1852.0:.3f} NMI ({max_sep:.0f}m)")
    print(f"  Mean: {mean_sep/1852.0:.3f} NMI ({mean_sep:.0f}m)")
    print(f"  Timesteps in [0, 3NMI]: {active_timesteps} / {total_valid}")
    if active_timesteps == 0:
        print(f"  → ISSUE FOUND: No active encounters! min_dcpa_m stays at inf → becomes NaN")
else:
    print('No episode files found')
