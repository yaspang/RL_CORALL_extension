import numpy as np
import json
from pathlib import Path

# Load the per-episode CSV
csv_file = Path('corall_baseline_case1_20260419-202941/seed_0/policy_eval_per_episode.csv')
if csv_file.exists():
    import pandas as pd
    df = pd.read_csv(csv_file)
    
    print(f"Total episodes: {len(df)}")
    print(f"\nmin_dcpa_m_ownship values (first 10 episodes):")
    print(df['min_dcpa_m_ownship'].head(10).tolist())
    print(f"\nmin_dcpa_m_ownship stats:")
    print(f"  Min: {df['min_dcpa_m_ownship'].min():.10e}")
    print(f"  Max: {df['min_dcpa_m_ownship'].max():.2f}")
    print(f"  Mean: {df['min_dcpa_m_ownship'].mean():.10e}")
    print(f"  # of inf values: {(df['min_dcpa_m_ownship'] == np.inf).sum()}")
    print(f"  # of nan values: {df['min_dcpa_m_ownship'].isna().sum()}")
    
    # Check what safe_mean does
    vals = df['min_dcpa_m_ownship'].tolist()
    filtered_vals = [v for v in vals if not np.isnan(v) and not np.isinf(v)]
    print(f"\nAfter filtering inf/nan: {len(filtered_vals)} values")
    if filtered_vals:
        print(f"  Mean of filtered: {np.mean(filtered_vals):.10e}")
        print(f"  Min of filtered: {np.min(filtered_vals):.2f}")
    
    # Check summary
    json_file = Path('corall_baseline_case1_20260419-202941/seed_0/policy_eval_summary.json')
    if json_file.exists():
        with open(json_file) as f:
            summary = json.load(f)
        print(f"\nFrom summary JSON:")
        print(f"  min_dcpa_m_mean: {summary.get('min_dcpa_m_mean', 'N/A'):.10e}")
else:
    print("CSV file not found")
