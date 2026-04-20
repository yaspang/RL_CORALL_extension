import numpy as np
from pathlib import Path

# Load case 1 episode history
ep_file = Path('corall_baseline_case1_20260419-202941/seed_0/episode_histories/baseline_case1_seed0_ep000.npz')
if ep_file.exists():
    data = np.load(ep_file, allow_pickle=True)
    dcpa = np.asarray(data['pair_dcpa'], dtype=float)
    dist = np.asarray(data['pair_dist'], dtype=float)
    X_all = np.asarray(data['X_all'], dtype=float)
    
    print(f'Shape of pair_dcpa: {dcpa.shape}')
    print(f'Shape of pair_dist: {dist.shape}')
    print(f'Shape of X_all: {X_all.shape}')
    print(f'Min DCPA (entire array): {np.nanmin(dcpa):.10e}')
    print(f'Min separation distance: {np.nanmin(dist):.2f}')
    print(f'Number of agents: {X_all.shape[1]}')
    print(f'\nInitial positions (meters):')
    print(X_all[0, :, :2])
    print(f'\nDCPA stats: min={np.nanmin(dcpa):.2e}, max={np.nanmax(dcpa):.2e}, mean={np.nanmean(dcpa):.2e}')
    
    # Check if pair_dcpa has NaNs
    nan_count = np.isnan(dcpa).sum()
    print(f'Number of NaN values in DCPA: {nan_count} / {dcpa.size}')
else:
    print('Episode file not found')
