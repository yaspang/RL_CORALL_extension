import numpy as np
from pathlib import Path

ep_dir = Path('corall_baseline_case1_20260419-204446/seed_0/episode_histories')
ep_files = sorted(ep_dir.glob('*.npz'))[:1]

if ep_files:
    data = np.load(ep_files[0], allow_pickle=True)
    dcpa = np.asarray(data['pair_dcpa'], dtype=float)
    
    print(f"pair_dcpa shape: {dcpa.shape}")
    print(f"pair_dcpa dtype: {dcpa.dtype}")
    print(f"All finite? {np.all(np.isfinite(dcpa))}")
    print(f"All NaN? {np.all(np.isnan(dcpa))}")
    print(f"All inf? {np.all(np.isinf(dcpa))}")
    
    valid_dcpa = dcpa[np.isfinite(dcpa)]
    print(f"\nValid DCPA values: {len(valid_dcpa)}")
    if len(valid_dcpa) > 0:
        print(f"  Min: {np.min(valid_dcpa):.1f} m ({np.min(valid_dcpa)/1852.0:.3f} NMI)")
        print(f"  Max: {np.max(valid_dcpa):.1f} m ({np.max(valid_dcpa)/1852.0:.3f} NMI)")
        print(f"  Mean: {np.mean(valid_dcpa):.1f} m ({np.mean(valid_dcpa)/1852.0:.3f} NMI)")
else:
    print('No episode files found')
