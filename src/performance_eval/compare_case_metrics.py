"""
Compare RL policy vs CORALL baseline across all 22 Imazu cases using bar charts and scaling plots.

Usage
-----
  python -m src.performance_eval.compare_case_metrics \\
      --base_dir "1_Baseline_Results_900s_100ep" \\
      --rl_dir "GENERALIZED_SB3_YYYYMMDD-HHMMSS" \\
      --rl_step 850000 \\
      --case_numbers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 \\
      --output_dir "1_comparison_results_850k"

Key arguments
-------------
  --base_dir PATH          Directory containing corall_baseline_case* folders
  --rl_dir PATH            Training directory containing eval_cp*_case* folders
  --rl_step INT            Checkpoint step to compare (filters eval_cp<step>_case* dirs)
  --case_numbers INT [..] Cases to compare (default: 1 6 21)
  --output_dir PATH        Output directory for charts and CSV (default: comparison_results)

Generated figures (01–11)
-----------------------
  01 Min separation by case      05 Collision rate
  02 Path length + efficiency    06 Success rate
  03 Completion time             07 Summary by complexity (2/3/4-ship)
  04 Risk exposure               08 Separation scaling
                                 09 Scaling analysis (2×2 grid)
                                 10 Per-case DCPA/Risk timeseries
                                 11 All-cases timeseries grid

Generated CSVs
--------------
  case_metrics.csv             Per-case aggregate metrics
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NMI = 1852.0

# Colour palette for target ships TS1–TS4 (same colour per TS across Baseline and RL;
# linestyle '-' = Baseline, '--' = RL distinguishes the two systems).
_TS_PALETTE = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']


def load_episode_histories(eval_dir: Path) -> List[Dict]:
    """Load all NPZ episode histories from eval directory"""
    ep_dir = eval_dir / "seed_0" / "episode_histories"
    if not ep_dir.exists():
        return []
    
    histories = []
    for npz_file in sorted(ep_dir.glob("*.npz")):
        try:
            data = np.load(npz_file, allow_pickle=True)
            t_arr = np.asarray(data['t'], dtype=float)
            X_all_arr = np.asarray(data['X_all'], dtype=float)
            pair_dist_arr = np.asarray(data['pair_dist'], dtype=float)
            
            # Try to load pair_risk, set to None if not available
            pair_risk_arr = None
            if 'pair_risk' in data.files:
                pair_risk_arr = np.asarray(data['pair_risk'], dtype=float)
            
            # Try to load collision flag
            collision_val = 0
            if 'collision' in data.files:
                collision_val = int(data['collision'].item() if hasattr(data['collision'], 'item') else data['collision'])
            
            # Extract scalars from 0-d arrays
            case_val = data['case'].item() if hasattr(data['case'], 'item') else data['case']
            seed_val = data['seed'].item() if hasattr(data['seed'], 'item') else data['seed']
            baseline_val = data['baseline'].item() if hasattr(data['baseline'], 'item') else data['baseline']
            wx_val = data['final_waypoint_x_nmi'].item() if hasattr(data['final_waypoint_x_nmi'], 'item') else data['final_waypoint_x_nmi']
            wy_val = data['final_waypoint_y_nmi'].item() if hasattr(data['final_waypoint_y_nmi'], 'item') else data['final_waypoint_y_nmi']
            
            hist_dict = {
                'filename': npz_file.name,
                't': t_arr,
                'X_all': X_all_arr,
                'pair_dist': pair_dist_arr,
                'pair_risk': pair_risk_arr,
                'case': int(case_val),
                'seed': int(seed_val),
                'baseline': bool(baseline_val),
                'collision': int(collision_val),
                'n_agents': X_all_arr.shape[1],
                'final_waypoint': (float(wx_val), float(wy_val)),
            }
            histories.append(hist_dict)
        except Exception as e:
            print(f"  Warning: Failed to load {npz_file.name}: {e}")
    
    return histories


def load_one_episode_history(eval_dir: Path) -> Optional[Dict]:
    """Load the first NPZ from seed_0/episode_histories/. Returns None if unavailable."""
    ep_dir = eval_dir / "seed_0" / "episode_histories"
    if not ep_dir.exists():
        return None
    npz_files = sorted(ep_dir.glob("*.npz"))
    if not npz_files:
        return None
    try:
        data = np.load(npz_files[0], allow_pickle=True)
        X_all = np.asarray(data['X_all'], dtype=float)
        return {
            't':         np.asarray(data['t'],         dtype=float),
            'pair_dcpa': np.asarray(data['pair_dcpa'], dtype=float),
            'pair_dist': np.asarray(data['pair_dist'], dtype=float),
            'pair_tcpa': np.asarray(data['pair_tcpa'], dtype=float),
            'pair_risk': np.asarray(data['pair_risk'], dtype=float),
            'n_agents':  X_all.shape[1],
        }
    except Exception as e:
        print(f"  Warning: could not load history from {ep_dir}: {e}")
        return None


def summarize_csv(eval_dir: Path) -> dict:
    """Aggregate per-episode CSV metrics for an eval directory (reads seed_0/policy_eval_per_episode.csv)."""
    csv_path = eval_dir / "seed_0" / "policy_eval_per_episode.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    return {
        "collision_rate":     df["collision_any"].mean()             if "collision_any"            in df.columns else np.nan,
        "success_rate":       df["success_ownship"].mean()           if "success_ownship"          in df.columns else np.nan,
        "min_sep_m":          df["min_actual_sep_m_ownship"].mean()  if "min_actual_sep_m_ownship" in df.columns else np.nan,
        "min_sep_m_std":      df["min_actual_sep_m_ownship"].std()   if "min_actual_sep_m_ownship" in df.columns else np.nan,
        "risk_exposure":      df["risk_exposure_ownship"].mean()     if "risk_exposure_ownship"    in df.columns else np.nan,
        "risk_exposure_std":  df["risk_exposure_ownship"].std()      if "risk_exposure_ownship"    in df.columns else np.nan,
        "path_length_m":      df["path_length_m_ownship"].mean()     if "path_length_m_ownship"    in df.columns else np.nan,
        "path_length_m_std":  df["path_length_m_ownship"].std()      if "path_length_m_ownship"    in df.columns else np.nan,
        "time_s":             df["completion_time_s_ownship"].mean() if "completion_time_s_ownship" in df.columns else np.nan,
        "time_s_std":         df["completion_time_s_ownship"].std()  if "completion_time_s_ownship" in df.columns else np.nan,
        "near_miss_rate":     df["near_miss_ownship"].mean()         if "near_miss_ownship"        in df.columns else (
                              df["near_miss_any"].mean()             if "near_miss_any"            in df.columns else np.nan),
        "n_episodes":         len(df),
    }


def compute_distance_traveled(X_all: np.ndarray, agent_idx: int = 0) -> float:
    """
    Compute total distance traveled by agent as sum of Euclidean distances
    between consecutive positions (in meters).
    
    Args:
        X_all: Shape (timesteps, n_agents, state_dim) - positions in meters
        agent_idx: Index of agent (default 0 for ownship)
    
    Returns:
        Total distance in meters
    """
    positions = X_all[:, agent_idx, :2]  # (timesteps, 2) in meters
    displacements = np.diff(positions, axis=0)  # (timesteps-1, 2)
    distances = np.linalg.norm(displacements, axis=1)  # (timesteps-1,)
    total_distance_m = np.sum(distances)
    return total_distance_m


def compute_total_time(t: np.ndarray) -> float:
    """Get total episode duration in seconds"""
    return float(t[-1] - t[0])


def compute_min_dcpa(pair_dcpa: np.ndarray, own_idx: int = 0) -> float:
    """
    Compute minimum DCPA across all targets for ownship.
    
    Args:
        pair_dcpa: Shape (timesteps, n_agents, n_agents)
        own_idx: Index of ownship (default 0)
    
    Returns:
        Minimum DCPA in nautical miles
    """
    # Get ownship's DCPA values across all targets
    own_dcpa = pair_dcpa[:, own_idx, :]
    # Exclude self (diagonal)
    own_dcpa_others = own_dcpa[:, np.arange(own_dcpa.shape[1]) != own_idx]
    # Get minimum across all targets and time
    min_dcpa_m = np.nanmin(own_dcpa_others)
    min_dcpa_nmi = min_dcpa_m / NMI if not np.isnan(min_dcpa_m) else np.nan
    return min_dcpa_nmi


def compute_min_separation(pair_dist: np.ndarray, own_idx: int = 0) -> float:
    """
    Compute minimum separation distance between ownship and any other vessel.
    
    Args:
        pair_dist: Shape (timesteps, n_agents, n_agents) - distances in meters
        own_idx: Index of ownship (default 0)
    
    Returns:
        Minimum separation in nautical miles
    """
    # Get ownship's distances to all other vessels
    own_dist = pair_dist[:, own_idx, :]
    # Exclude self (diagonal)
    own_dist_others = own_dist[:, np.arange(own_dist.shape[1]) != own_idx]
    # Get minimum across all targets and time
    min_sep_m = np.nanmin(own_dist_others)
    min_sep_nmi = min_sep_m / NMI if not np.isnan(min_sep_m) else np.nan
    return min_sep_nmi


def compute_time_weighted_risk(pair_risk: np.ndarray, t: np.ndarray, own_idx: int = 0) -> float:
    """
    Compute time-weighted risk (area under risk curve).
    
    Args:
        pair_risk: Shape (timesteps, n_agents, n_agents)
        t: Time array (timesteps,)
        own_idx: Index of ownship (default 0)
    
    Returns:
        Time-weighted risk exposure (dimensionless)
    """
    # Get ownship's max risk across all targets at each timestep
    own_risk = pair_risk[:, own_idx, :]
    own_risk_others = own_risk[:, np.arange(own_risk.shape[1]) != own_idx]
    max_risk_per_t = np.nanmax(own_risk_others, axis=1)
    
    # Compute area under curve using trapezoidal integration (manual)
    dt = np.diff(t)
    avg_risk = (max_risk_per_t[:-1] + max_risk_per_t[1:]) / 2
    time_weighted_risk = np.sum(avg_risk * dt)
    return float(time_weighted_risk)


def get_n_agents_from_case(case_num: int) -> int:
    """Map case number to number of agents"""
    if case_num <= 4:
        return 2
    elif case_num <= 11:
        return 3
    else:  # case_num > 11
        return 4


def aggregate_case_metrics(
    case_num: int,
    baseline_dirs: List[Path],
    rl_dirs: List[Path],
) -> Dict | None:
    """
    Aggregate metrics for a single case across all episodes.
    
    Returns dict with keys: case, n_agents, n_additional_agents,
                            baseline_min_sep_nmi, rl_min_sep_nmi,
                            baseline_dist_m, rl_dist_m,
                            baseline_time_s, rl_time_s,
                            baseline_near_miss_rate, rl_near_miss_rate,
                            n_episodes_baseline, n_episodes_rl
    """
    # Find directories for this case (use regex to match exact case number)
    case_pattern = re.compile(rf"case{case_num}(?:\D|$)")
    baseline_for_case = [d for d in baseline_dirs if case_pattern.search(d.name)]
    rl_for_case = [d for d in rl_dirs if case_pattern.search(d.name)]
    
    if not baseline_for_case or not rl_for_case:
        return None
    
    baseline_dir = baseline_for_case[0]
    rl_dir = rl_for_case[0]
    
    # Load histories (optional - may not exist for baseline)
    baseline_hists = load_episode_histories(baseline_dir)
    rl_hists = load_episode_histories(rl_dir)
    
    # Get number of agents from histories or from case mapping
    if baseline_hists:
        n_agents = baseline_hists[0]['n_agents']
    elif rl_hists:
        n_agents = rl_hists[0]['n_agents']
    else:
        n_agents = get_n_agents_from_case(case_num)
    
    n_additional_agents = n_agents - 1  # Exclude ownship
    
    # Compute metrics for baseline (if histories exist)
    baseline_min_sep_vals = []
    baseline_risk_vals = []
    baseline_distances = []
    baseline_times = []
    
    if baseline_hists:
        for hist in baseline_hists:
            try:
                min_sep = compute_min_separation(hist['pair_dist'])
                dist = compute_distance_traveled(hist['X_all'])
                time = compute_total_time(hist['t'])
                
                # Compute risk only if available
                risk = np.nan
                if hist['pair_risk'] is not None:
                    risk = compute_time_weighted_risk(hist['pair_risk'], hist['t'])
                
                if not np.isnan(min_sep):
                    baseline_min_sep_vals.append(min_sep)
                if not np.isnan(risk):
                    baseline_risk_vals.append(risk)
                baseline_distances.append(dist)
                baseline_times.append(time)
            except Exception as e:
                print(f"    Warning: Failed to compute baseline metrics: {e}")
    else:
        print(f"  No baseline histories found for case {case_num} (will use CSV-only metrics)")
    
    # Compute metrics for RL (if histories exist)
    rl_min_sep_vals = []
    rl_risk_vals = []
    rl_distances = []
    rl_times = []
    
    if rl_hists:
        for hist in rl_hists:
            try:
                min_sep = compute_min_separation(hist['pair_dist'])
                dist = compute_distance_traveled(hist['X_all'])
                time = compute_total_time(hist['t'])
                
                # Compute risk only if available
                risk = np.nan
                if hist['pair_risk'] is not None:
                    risk = compute_time_weighted_risk(hist['pair_risk'], hist['t'])
                
                if not np.isnan(min_sep):
                    rl_min_sep_vals.append(min_sep)
                if not np.isnan(risk):
                    rl_risk_vals.append(risk)
                rl_distances.append(dist)
                rl_times.append(time)
            except Exception as e:
                print(f"    Warning: Failed to compute RL metrics: {e}")
    else:
        print(f"  No RL histories found for case {case_num} (will use CSV-only metrics)")
    
    # Average and std metrics
    baseline_min_sep_avg = np.mean(baseline_min_sep_vals) if baseline_min_sep_vals else np.nan
    rl_min_sep_avg = np.mean(rl_min_sep_vals) if rl_min_sep_vals else np.nan
    baseline_min_sep_std = np.std(baseline_min_sep_vals, ddof=1) if len(baseline_min_sep_vals) > 1 else np.nan
    rl_min_sep_std = np.std(rl_min_sep_vals, ddof=1) if len(rl_min_sep_vals) > 1 else np.nan

    baseline_risk_avg = np.mean(baseline_risk_vals) if baseline_risk_vals else np.nan
    rl_risk_avg = np.mean(rl_risk_vals) if rl_risk_vals else np.nan
    baseline_risk_std = np.std(baseline_risk_vals, ddof=1) if len(baseline_risk_vals) > 1 else np.nan
    rl_risk_std = np.std(rl_risk_vals, ddof=1) if len(rl_risk_vals) > 1 else np.nan

    baseline_dist_avg = np.mean(baseline_distances) if baseline_distances else np.nan
    rl_dist_avg = np.mean(rl_distances) if rl_distances else np.nan
    baseline_dist_std = np.std(baseline_distances, ddof=1) if len(baseline_distances) > 1 else np.nan
    rl_dist_std = np.std(rl_distances, ddof=1) if len(rl_distances) > 1 else np.nan

    baseline_time_avg = np.mean(baseline_times) if baseline_times else np.nan
    rl_time_avg = np.mean(rl_times) if rl_times else np.nan
    baseline_time_std = np.std(baseline_times, ddof=1) if len(baseline_times) > 1 else np.nan
    rl_time_std = np.std(rl_times, ddof=1) if len(rl_times) > 1 else np.nan

    # ===== CSV SUMMARIES ARE AUTHORITATIVE — ALWAYS PREFER OVER NPZ-COMPUTED VALUES =====
    # policy_eval_per_episode.csv covers all 100 episodes; NPZ histories may be a single saved
    # episode (n_episodes_baseline=1), so scalar metrics must come from the CSV.
    baseline_csv_summary = summarize_csv(baseline_dir)
    rl_csv_summary = summarize_csv(rl_dir)

    baseline_n_episodes_csv = 0
    baseline_collision_rate = np.nan
    baseline_success_rate = np.nan
    baseline_near_miss_rate = np.nan

    if baseline_csv_summary:
        baseline_n_episodes_csv = int(baseline_csv_summary.get("n_episodes", 0))
        baseline_min_sep_avg   = baseline_csv_summary.get("min_sep_m", np.nan) / NMI
        baseline_min_sep_std   = baseline_csv_summary.get("min_sep_m_std", np.nan) / NMI
        baseline_risk_avg      = baseline_csv_summary.get("risk_exposure", np.nan)
        baseline_risk_std      = baseline_csv_summary.get("risk_exposure_std", np.nan)
        baseline_dist_avg      = baseline_csv_summary.get("path_length_m", np.nan)
        baseline_dist_std      = baseline_csv_summary.get("path_length_m_std", np.nan)
        baseline_time_avg      = baseline_csv_summary.get("time_s", np.nan)
        baseline_time_std      = baseline_csv_summary.get("time_s_std", np.nan)
        baseline_collision_rate = baseline_csv_summary.get("collision_rate", np.nan)
        baseline_success_rate  = baseline_csv_summary.get("success_rate", np.nan)
        baseline_near_miss_rate = baseline_csv_summary.get("near_miss_rate", np.nan)

    rl_n_episodes_csv = 0
    rl_collision_rate = np.nan
    rl_success_rate = np.nan
    rl_near_miss_rate = np.nan

    if rl_csv_summary:
        rl_n_episodes_csv = int(rl_csv_summary.get("n_episodes", 0))
        rl_min_sep_avg   = rl_csv_summary.get("min_sep_m", np.nan) / NMI
        rl_min_sep_std   = rl_csv_summary.get("min_sep_m_std", np.nan) / NMI
        rl_risk_avg      = rl_csv_summary.get("risk_exposure", np.nan)
        rl_risk_std      = rl_csv_summary.get("risk_exposure_std", np.nan)
        rl_dist_avg      = rl_csv_summary.get("path_length_m", np.nan)
        rl_dist_std      = rl_csv_summary.get("path_length_m_std", np.nan)
        rl_time_avg      = rl_csv_summary.get("time_s", np.nan)
        rl_time_std      = rl_csv_summary.get("time_s_std", np.nan)
        rl_collision_rate = rl_csv_summary.get("collision_rate", np.nan)
        rl_success_rate  = rl_csv_summary.get("success_rate", np.nan)
        rl_near_miss_rate = rl_csv_summary.get("near_miss_rate", np.nan)

    return {
        'case': case_num,
        'n_agents': n_agents,
        'n_additional_agents': n_additional_agents,
        'baseline_min_sep_nmi': baseline_min_sep_avg,
        'baseline_min_sep_nmi_std': baseline_min_sep_std,
        'rl_min_sep_nmi': rl_min_sep_avg,
        'rl_min_sep_nmi_std': rl_min_sep_std,
        'baseline_risk_exposure': baseline_risk_avg,
        'baseline_risk_exposure_std': baseline_risk_std,
        'rl_risk_exposure': rl_risk_avg,
        'rl_risk_exposure_std': rl_risk_std,
        'baseline_dist_m': baseline_dist_avg,
        'baseline_dist_m_std': baseline_dist_std,
        'rl_dist_m': rl_dist_avg,
        'rl_dist_m_std': rl_dist_std,
        'baseline_time_s': baseline_time_avg,
        'baseline_time_s_std': baseline_time_std,
        'rl_time_s': rl_time_avg,
        'rl_time_s_std': rl_time_std,
        'baseline_collision_rate': baseline_collision_rate,
        'rl_collision_rate': rl_collision_rate,
        'baseline_success_rate': baseline_success_rate,
        'rl_success_rate': rl_success_rate,
        'baseline_near_miss_rate': baseline_near_miss_rate,
        'rl_near_miss_rate': rl_near_miss_rate,
        'n_episodes_baseline': baseline_n_episodes_csv if baseline_n_episodes_csv > 0 else len(baseline_distances),
        'n_episodes_rl': rl_n_episodes_csv if rl_n_episodes_csv > 0 else len(rl_distances),
    }


def find_eval_directories(
    base_dir: Path,
    case_numbers: List[int],
    rl_dir: Path = None,
    rl_step: int = None,
) -> Tuple[List[Path], List[Path]]:
    """Find baseline and RL evaluation directories for given cases.

    Args:
        base_dir:     Root directory searched for baseline dirs (and legacy RL dirs).
        case_numbers: Cases to include.
        rl_dir:       If provided, scan this directory for eval_cp*/policy_eval* RL dirs.
        rl_step:      If provided with rl_dir, only include dirs whose step number matches.
    """
    baseline_dirs = []
    rl_dirs = []

    for item in base_dir.iterdir():
        if not item.is_dir():
            continue

        # Match baseline directories (corall_baseline_case* pattern)
        if "corall_baseline_case" in item.name:
            match = re.search(r"corall_baseline_case(\d+)", item.name)
            if match and int(match.group(1)) in case_numbers:
                baseline_dirs.append(item)
        # Also support old baseline_case* pattern
        elif item.name.startswith("baseline_case"):
            match = re.search(r"baseline_case(\d+)", item.name)
            if match and int(match.group(1)) in case_numbers:
                baseline_dirs.append(item)

        # Legacy RL eval dirs: policy_eval*sb3* inside base_dir
        elif "policy_eval" in item.name and "sb3" in item.name:
            match = re.search(r"case(\d+)", item.name)
            if match and int(match.group(1)) in case_numbers:
                rl_dirs.append(item)

        # Ranker-style eval_cp* dirs directly inside base_dir (when no dedicated rl_dir given)
        elif rl_dir is None and item.name.startswith("eval_cp") and "case" in item.name:
            match = re.search(r"case(\d+)", item.name)
            if match and int(match.group(1)) in case_numbers:
                if rl_step is not None:
                    step_match = re.search(r"eval_cp(\d+)_", item.name)
                    if not step_match or int(step_match.group(1)) != rl_step:
                        continue
                rl_dirs.append(item)

    # If a dedicated RL directory is provided, scan it for eval_cp* dirs
    if rl_dir is not None and rl_dir.exists():
        for item in rl_dir.iterdir():
            if not item.is_dir():
                continue
            if item.name.startswith("eval_cp") and "case" in item.name:
                match = re.search(r"case(\d+)", item.name)
                if match and int(match.group(1)) in case_numbers:
                    if rl_step is not None:
                        step_match = re.search(r"eval_cp(\d+)_", item.name)
                        if not step_match or int(step_match.group(1)) != rl_step:
                            continue
                    rl_dirs.append(item)

    return baseline_dirs, rl_dirs


def create_bar_chart_separation(metrics_df: pd.DataFrame, output_path: Path):
    """Chart 1: Min Separation Distance by case (RL vs Baseline)"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(metrics_df))
    width = 0.35
    
    # Shorter labels to avoid overlap
    case_labels = [f"C{int(c)}\n({int(n)}s)" 
                   for c, n in zip(metrics_df['case'], metrics_df['n_agents'])]
    
    # Convert from nmi to meters for consistency
    baseline_sep_m = metrics_df['baseline_min_sep_nmi'] * NMI
    rl_sep_m = metrics_df['rl_min_sep_nmi'] * NMI
    baseline_sep_std_m = metrics_df['baseline_min_sep_nmi_std'] * NMI
    rl_sep_std_m = metrics_df['rl_min_sep_nmi_std'] * NMI
    
    # Replace NaN with 0 for error bars
    baseline_sep_std_m = np.nan_to_num(baseline_sep_std_m, nan=0.0)
    rl_sep_std_m = np.nan_to_num(rl_sep_std_m, nan=0.0)
    
    ax.bar(x - width/2, baseline_sep_m, width, label='Baseline (Rule-Based)', color='#1f77b4', yerr=baseline_sep_std_m, capsize=5, error_kw={'linewidth': 2})
    ax.bar(x + width/2, rl_sep_m, width, label='RL Policy (Learning-Based)', color='#ff7f0e', yerr=rl_sep_std_m, capsize=5, error_kw={'linewidth': 2})
    
    ax.set_xlabel('Case', fontsize=14, fontweight='bold')
    ax.set_ylabel('Minimum Separation Distance (m)', fontsize=14, fontweight='bold')
    ax.set_title('Minimum Separation Distance: Baseline vs RL Policy', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=9, rotation=0, ha='center')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='y', labelsize=12)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def create_bar_chart_distance(metrics_df: pd.DataFrame, output_path: Path):
    """Chart 2: Total path length by case with efficiency %"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(metrics_df))
    width = 0.35
    
    case_labels = [f"C{int(c)}\n({int(n)}s)" 
                   for c, n in zip(metrics_df['case'], metrics_df['n_agents'])]
    
    baseline_dist = metrics_df['baseline_dist_m']
    rl_dist = metrics_df['rl_dist_m']
    baseline_dist_std = metrics_df.get('baseline_dist_m_std', pd.Series([np.nan]*len(metrics_df)))
    rl_dist_std = metrics_df.get('rl_dist_m_std', pd.Series([np.nan]*len(metrics_df)))
    
    baseline_dist_std = np.nan_to_num(baseline_dist_std, nan=0.0)
    rl_dist_std = np.nan_to_num(rl_dist_std, nan=0.0)
    
    bars1 = ax.bar(x - width/2, baseline_dist, width, label='Baseline (Rule-Based)', color='#1f77b4', yerr=baseline_dist_std, capsize=5, error_kw={'linewidth': 2})
    bars2 = ax.bar(x + width/2, rl_dist, width, label='RL Policy (Learning-Based)', color='#ff7f0e', yerr=rl_dist_std, capsize=5, error_kw={'linewidth': 2})
    
    # Add efficiency labels on bars
    for i, (b_dist, r_dist) in enumerate(zip(baseline_dist, rl_dist)):
        efficiency_pct = ((b_dist - r_dist) / b_dist) * 100
        ax.text(i + width/2, r_dist + 50, 
               f'{efficiency_pct:+.1f}%', ha='center', fontsize=11, fontweight='bold',
               color='darkred' if efficiency_pct > 0 else 'darkblue')
    
    ax.set_xlabel('Case', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Path Length (meters)', fontsize=14, fontweight='bold')
    ax.set_title('Total Path Length Traveled (Ownship): Baseline vs RL Policy', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=9, rotation=0, ha='center')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='y', labelsize=12)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def create_bar_chart_time(metrics_df: pd.DataFrame, output_path: Path):
    """Chart 3: Total time traveled by case with efficiency %"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(metrics_df))
    width = 0.35
    
    case_labels = [f"C{int(c)}\n({int(n)}s)" 
                   for c, n in zip(metrics_df['case'], metrics_df['n_agents'])]
    
    baseline_time = metrics_df['baseline_time_s']
    rl_time = metrics_df['rl_time_s']
    baseline_time_std = metrics_df.get('baseline_time_s_std', pd.Series([np.nan]*len(metrics_df)))
    rl_time_std = metrics_df.get('rl_time_s_std', pd.Series([np.nan]*len(metrics_df)))
    
    baseline_time_std = np.nan_to_num(baseline_time_std, nan=0.0)
    rl_time_std = np.nan_to_num(rl_time_std, nan=0.0)
    
    bars1 = ax.bar(x - width/2, baseline_time, width, label='Baseline (Rule-Based)', color='#1f77b4', yerr=baseline_time_std, capsize=5, error_kw={'linewidth': 2})
    bars2 = ax.bar(x + width/2, rl_time, width, label='RL Policy (Learning-Based)', color='#ff7f0e', yerr=rl_time_std, capsize=5, error_kw={'linewidth': 2})
    
    # Add efficiency labels on bars
    for i, (b_time, r_time) in enumerate(zip(baseline_time, rl_time)):
        efficiency_pct = ((b_time - r_time) / b_time) * 100
        ax.text(i + width/2, r_time + 3, 
               f'{efficiency_pct:+.1f}%', ha='center', fontsize=11, fontweight='bold',
               color='darkred' if efficiency_pct > 0 else 'darkblue')
    
    ax.set_xlabel('Case', fontsize=14, fontweight='bold')
    ax.set_ylabel('Episode Duration (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Time to Goal: Baseline vs RL Policy', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=9, rotation=0, ha='center')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def create_bar_chart_risk_exposure(metrics_df: pd.DataFrame, output_path: Path):
    """Chart 4: Risk Exposure (time-weighted maximum risk) by case"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(metrics_df))
    width = 0.35
    
    case_labels = [f"C{int(c)}\n({int(n)}s)" 
                   for c, n in zip(metrics_df['case'], metrics_df['n_agents'])]
    
    baseline_risk = metrics_df['baseline_risk_exposure']
    rl_risk = metrics_df['rl_risk_exposure']
    baseline_risk_std = metrics_df.get('baseline_risk_exposure_std', pd.Series([np.nan]*len(metrics_df)))
    rl_risk_std = metrics_df.get('rl_risk_exposure_std', pd.Series([np.nan]*len(metrics_df)))
    
    baseline_risk_std = np.nan_to_num(baseline_risk_std, nan=0.0)
    rl_risk_std = np.nan_to_num(rl_risk_std, nan=0.0)
    
    bars1 = ax.bar(x - width/2, baseline_risk, width, label='Baseline (Rule-Based)', color='#1f77b4', yerr=baseline_risk_std, capsize=5, error_kw={'linewidth': 2})
    bars2 = ax.bar(x + width/2, rl_risk, width, label='RL Policy (Learning-Based)', color='#ff7f0e', yerr=rl_risk_std, capsize=5, error_kw={'linewidth': 2})
    
    # Add difference labels on bars
    for i, (b_risk, r_risk) in enumerate(zip(baseline_risk, rl_risk)):
        if b_risk > 0:
            risk_reduction_pct = ((b_risk - r_risk) / b_risk) * 100
            ax.text(i + width/2, r_risk + 0.05 * baseline_risk.max(), 
                   f'{risk_reduction_pct:+.1f}%', ha='center', fontsize=11, fontweight='bold',
                   color='darkred' if risk_reduction_pct > 0 else 'darkblue')
    
    ax.set_xlabel('Case', fontsize=14, fontweight='bold')
    ax.set_ylabel('Risk Exposure (Time-Weighted Max Risk)', fontsize=14, fontweight='bold')
    ax.set_title('Risk Exposure Over Time: Baseline vs RL Policy (Lower is Better)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=9, rotation=0, ha='center')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='y', labelsize=12)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def create_bar_chart_collision_rate(metrics_df: pd.DataFrame, output_path: Path):
    """Chart: Collision Rate (%) by case"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(metrics_df))
    width = 0.35
    
    case_labels = [f"C{int(c)}\n({int(n)}s)" 
                   for c, n in zip(metrics_df['case'], metrics_df['n_agents'])]
    
    baseline_collision_pct = metrics_df['baseline_collision_rate'] * 100
    rl_collision_pct = metrics_df['rl_collision_rate'] * 100
    
    bars1 = ax.bar(x - width/2, baseline_collision_pct, width, label='Baseline', color='#1f77b4')
    bars2 = ax.bar(x + width/2, rl_collision_pct, width, label='RL Policy', color='#ff7f0e')
    
    # Add collision counts on bars
    for i, (b_rate, r_rate) in enumerate(zip(baseline_collision_pct, rl_collision_pct)):
        ax.text(i - width/2, b_rate + 1, f'{b_rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
        ax.text(i + width/2, r_rate + 1, f'{r_rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Case', fontsize=14, fontweight='bold')
    ax.set_ylabel('Collision Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Collision Rate: Baseline vs RL Policy (Lower is Better)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=9, rotation=0, ha='center')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def create_bar_chart_success_rate(metrics_df: pd.DataFrame, output_path: Path):
    """Chart: Success Rate (%) by case - complement of collision rate"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(metrics_df))
    width = 0.35
    
    case_labels = [f"C{int(c)}\n({int(n)}s)" 
                   for c, n in zip(metrics_df['case'], metrics_df['n_agents'])]
    
    # Use actual success_rate from CSV if available; fall back to 1 - collision_rate
    baseline_success_pct = metrics_df['baseline_success_rate'].fillna(
        1 - metrics_df['baseline_collision_rate']
    ) * 100
    rl_success_pct = metrics_df['rl_success_rate'].fillna(
        1 - metrics_df['rl_collision_rate']
    ) * 100
    
    bars1 = ax.bar(x - width/2, baseline_success_pct, width, label='Baseline', color='#2ca02c')
    bars2 = ax.bar(x + width/2, rl_success_pct, width, label='RL Policy', color='#1f77b4')
    
    # Add success rates on bars with checkmarks/crosses for visibility
    for i, (b_rate, r_rate) in enumerate(zip(baseline_success_pct, rl_success_pct)):
        ax.text(i - width/2, b_rate + 1.5, f'{b_rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
        ax.text(i + width/2, r_rate + 1.5, f'{r_rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Case', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Success Rate: Baseline vs RL Policy (Higher is Better) - Guaranteed Safe Passage', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=9, rotation=0, ha='center')
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.4, linewidth=1.5, label='Perfect Success')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='y', labelsize=12)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def create_summary_guaranteed_success(metrics_df: pd.DataFrame, output_path: Path):
    """
    Chart: Aggregated Success/Collision Metrics across agent complexities
    Shows average success rate for 2-ship, 3-ship, 4-ship cases to demonstrate
    guaranteed collision avoidance across increasing complexity.
    """
    # Categorize cases by number of agents
    cat_2ship = metrics_df[metrics_df['n_agents'] == 2]
    cat_3ship = metrics_df[metrics_df['n_agents'] == 3]
    cat_4ship = metrics_df[metrics_df['n_agents'] == 4]
    
    categories = []
    baseline_success = []
    rl_success = []
    baseline_collision = []
    rl_collision = []
    
    # 2-ship scenarios
    if len(cat_2ship) > 0:
        categories.append("2-Ship\n(Simple)")
        baseline_success.append(cat_2ship['baseline_success_rate'].fillna(1 - cat_2ship['baseline_collision_rate']).mean() * 100)
        rl_success.append(cat_2ship['rl_success_rate'].fillna(1 - cat_2ship['rl_collision_rate']).mean() * 100)
        baseline_collision.append(cat_2ship['baseline_collision_rate'].mean() * 100)
        rl_collision.append(cat_2ship['rl_collision_rate'].mean() * 100)
    
    # 3-ship scenarios
    if len(cat_3ship) > 0:
        categories.append("3-Ship\n(Moderate)")
        baseline_success.append(cat_3ship['baseline_success_rate'].fillna(1 - cat_3ship['baseline_collision_rate']).mean() * 100)
        rl_success.append(cat_3ship['rl_success_rate'].fillna(1 - cat_3ship['rl_collision_rate']).mean() * 100)
        baseline_collision.append(cat_3ship['baseline_collision_rate'].mean() * 100)
        rl_collision.append(cat_3ship['rl_collision_rate'].mean() * 100)
    
    # 4-ship scenarios
    if len(cat_4ship) > 0:
        categories.append("4-Ship\n(Complex)")
        baseline_success.append(cat_4ship['baseline_success_rate'].fillna(1 - cat_4ship['baseline_collision_rate']).mean() * 100)
        rl_success.append(cat_4ship['rl_success_rate'].fillna(1 - cat_4ship['rl_collision_rate']).mean() * 100)
        baseline_collision.append(cat_4ship['baseline_collision_rate'].mean() * 100)
        rl_collision.append(cat_4ship['rl_collision_rate'].mean() * 100)
    
    # Overall average
    categories.append("Overall\nAverage")
    baseline_success.append(metrics_df['baseline_success_rate'].fillna(1 - metrics_df['baseline_collision_rate']).mean() * 100)
    rl_success.append(metrics_df['rl_success_rate'].fillna(1 - metrics_df['rl_collision_rate']).mean() * 100)
    baseline_collision.append(metrics_df['baseline_collision_rate'].mean() * 100)
    rl_collision.append(metrics_df['rl_collision_rate'].mean() * 100)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    x = np.arange(len(categories))
    width = 0.35
    
    # ===== Subplot 1: Success Rate =====
    bars1 = ax1.bar(x - width/2, baseline_success, width, label='Baseline', 
                    color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, rl_success, width, label='RL Policy', 
                    color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on success rate bars
    for i, (b, r) in enumerate(zip(baseline_success, rl_success)):
        ax1.text(i - width/2, b + 1.5, f'{b:.1f}%', ha='center', fontsize=12, fontweight='bold')
        ax1.text(i + width/2, r + 1.5, f'{r:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Success Rate Across Complexity Levels\n(Guaranteed Collision Avoidance)', 
                  fontsize=15, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Perfect Safety')
    ax1.legend(fontsize=12, loc='lower right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='y', labelsize=11)
    
    # ===== Subplot 2: Collision Rate =====
    bars3 = ax2.bar(x - width/2, baseline_collision, width, label='Baseline', 
                    color='#d62728', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars4 = ax2.bar(x + width/2, rl_collision, width, label='RL Policy', 
                    color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on collision rate bars
    for i, (b, r) in enumerate(zip(baseline_collision, rl_collision)):
        if b > 0.5:
            ax2.text(i - width/2, b + 1.5, f'{b:.1f}%', ha='center', fontsize=12, fontweight='bold')
        if r > 0.5:
            ax2.text(i + width/2, r + 1.5, f'{r:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Collision Rate (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Collision Rate Across Complexity Levels\n(Lower is Better)', 
                  fontsize=15, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)  # Full 0-100% scale for accurate percentage representation
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Zero Collisions')
    ax2.legend(fontsize=12, loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='y', labelsize=11)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def create_scaling_chart_separation(metrics_df: pd.DataFrame, output_path: Path):
    """Chart 5: Min Separation Distance vs number of ships in environment (agent count)"""
    # Group by total number of agents
    grouped = metrics_df.groupby('n_agents')
    
    fig, ax = plt.subplots(figsize=(12, 7.5))
    
    agent_counts = sorted(metrics_df['n_agents'].unique())
    baseline_sep_by_agents = []
    rl_sep_by_agents = []
    
    x_positions = []
    labels = []
    
    for i, agent_count in enumerate(agent_counts):
        group = metrics_df[metrics_df['n_agents'] == agent_count]
        
        # Convert from nmi to meters for consistency
        baseline_avg = group['baseline_min_sep_nmi'].mean() * NMI
        rl_avg = group['rl_min_sep_nmi'].mean() * NMI
        
        baseline_sep_by_agents.append(baseline_avg)
        rl_sep_by_agents.append(rl_avg)
        x_positions.append(i)
        labels.append(f"{int(agent_count)} ships")
    
    width = 0.35
    ax.bar(np.array(x_positions) - width/2, baseline_sep_by_agents, width, 
           label='Baseline (avg)', color='#1f77b4')
    ax.bar(np.array(x_positions) + width/2, rl_sep_by_agents, width, 
           label='RL Policy (avg)', color='#ff7f0e')
    
    ax.set_xlabel('Total Ships in Environment', fontsize=15, fontweight='bold')
    ax.set_ylabel('Minimum Separation Distance (m)', fontsize=15, fontweight='bold')
    ax.set_title('Minimum Separation Distance Scaling: Impact of Ships in Environment', fontsize=17, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='y', labelsize=12)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def create_scaling_line_charts(metrics_df: pd.DataFrame, output_path: Path):
    """Chart: 2-row grid (2 efficiency plots on top, 3 colav/risk plots on bottom)"""
    # Group by number of agents
    grouped_data = []
    for agent_count in sorted(metrics_df['n_agents'].unique()):
        group = metrics_df[metrics_df['n_agents'] == agent_count]
        # Means
        baseline_sep = group['baseline_min_sep_nmi'].mean() * NMI  # meters
        rl_sep = group['rl_min_sep_nmi'].mean() * NMI
        baseline_risk = group['baseline_risk_exposure'].mean()
        rl_risk = group['rl_risk_exposure'].mean()
        baseline_dist = group['baseline_dist_m'].mean()
        rl_dist = group['rl_dist_m'].mean()
        baseline_time = group['baseline_time_s'].mean()
        rl_time = group['rl_time_s'].mean()
        baseline_collision = group['baseline_collision_rate'].mean()
        rl_collision = group['rl_collision_rate'].mean()
        # Efficiencies
        sep_efficiency = ((rl_sep - baseline_sep) / baseline_sep * 100) if baseline_sep > 0 else 0
        # Suppress risk % when baseline is near zero (avoids absurd values like -27000%)
        risk_efficiency = ((baseline_risk - rl_risk) / baseline_risk * 100) if baseline_risk > 0.1 else np.nan
        dist_efficiency = ((baseline_dist - rl_dist) / baseline_dist * 100) if baseline_dist > 0 else 0
        time_efficiency = ((baseline_time - rl_time) / baseline_time * 100) if baseline_time > 0 else 0
        collision_efficiency = ((baseline_collision - rl_collision) / max(baseline_collision, 0.001) * 100)

        # STDs (use std, fallback to nan if not present)
        baseline_sep_std = group['baseline_min_sep_nmi_std'].mean() * NMI if 'baseline_min_sep_nmi_std' in group else np.nan
        rl_sep_std = group['rl_min_sep_nmi_std'].mean() * NMI if 'rl_min_sep_nmi_std' in group else np.nan
        baseline_risk_std = group['baseline_risk_exposure_std'].mean() if 'baseline_risk_exposure_std' in group else np.nan
        rl_risk_std = group['rl_risk_exposure_std'].mean() if 'rl_risk_exposure_std' in group else np.nan
        baseline_dist_std = group['baseline_dist_m_std'].mean() if 'baseline_dist_m_std' in group else np.nan
        rl_dist_std = group['rl_dist_m_std'].mean() if 'rl_dist_m_std' in group else np.nan
        baseline_time_std = group['baseline_time_s_std'].mean() if 'baseline_time_s_std' in group else np.nan
        rl_time_std = group['rl_time_s_std'].mean() if 'rl_time_s_std' in group else np.nan

        grouped_data.append({
            'n_agents': agent_count,
            'baseline_sep': baseline_sep,
            'rl_sep': rl_sep,
            'sep_efficiency': sep_efficiency,
            'baseline_risk': baseline_risk,
            'rl_risk': rl_risk,
            'risk_efficiency': risk_efficiency,
            'baseline_dist': baseline_dist,
            'rl_dist': rl_dist,
            'dist_efficiency': dist_efficiency,
            'baseline_time': baseline_time,
            'rl_time': rl_time,
            'time_efficiency': time_efficiency,
            'baseline_collision': baseline_collision,
            'rl_collision': rl_collision,
            'collision_efficiency': collision_efficiency,
            'baseline_sep_std': baseline_sep_std,
            'rl_sep_std': rl_sep_std,
            'baseline_risk_std': baseline_risk_std,
            'rl_risk_std': rl_risk_std,
            'baseline_dist_std': baseline_dist_std,
            'rl_dist_std': rl_dist_std,
            'baseline_time_std': baseline_time_std,
            'rl_time_std': rl_time_std,
        })
    
    grouped_df = pd.DataFrame(grouped_data)
    
    # Convert pandas Series to numpy arrays for matplotlib compatibility
    x = np.array(grouped_df['n_agents'].values, dtype=float)
    baseline_sep_arr = np.array(grouped_df['baseline_sep'].values, dtype=float)
    rl_sep_arr = np.array(grouped_df['rl_sep'].values, dtype=float)
    baseline_risk_arr = np.array(grouped_df['baseline_risk'].values, dtype=float)
    rl_risk_arr = np.array(grouped_df['rl_risk'].values, dtype=float)
    baseline_dist_arr = np.array(grouped_df['baseline_dist'].values, dtype=float)
    rl_dist_arr = np.array(grouped_df['rl_dist'].values, dtype=float)
    baseline_time_arr = np.array(grouped_df['baseline_time'].values, dtype=float)
    rl_time_arr = np.array(grouped_df['rl_time'].values, dtype=float)
    baseline_collision_arr = np.array(grouped_df['baseline_collision'].values, dtype=float) * 100
    rl_collision_arr = np.array(grouped_df['rl_collision'].values, dtype=float) * 100
    # std arrays
    baseline_sep_std_arr = np.array(grouped_df['baseline_sep_std'].values, dtype=float)
    rl_sep_std_arr = np.array(grouped_df['rl_sep_std'].values, dtype=float)
    baseline_risk_std_arr = np.array(grouped_df['baseline_risk_std'].values, dtype=float)
    rl_risk_std_arr = np.array(grouped_df['rl_risk_std'].values, dtype=float)
    baseline_dist_std_arr = np.array(grouped_df['baseline_dist_std'].values, dtype=float)
    rl_dist_std_arr = np.array(grouped_df['rl_dist_std'].values, dtype=float)
    baseline_time_std_arr = np.array(grouped_df['baseline_time_std'].values, dtype=float)
    rl_time_std_arr = np.array(grouped_df['rl_time_std'].values, dtype=float)
    
    # Create figure with space for legend above subplots
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.12, right=0.95, top=0.82, bottom=0.08, hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    axs = np.array([[ax1, ax2], [ax3, ax4]])
    
    # ===== Panel 1 (Top Left): Total Path Length =====
    ax = axs[0, 0]
    base_line, = ax.plot(x, baseline_dist_arr, 'o-', linewidth=1.5, markersize=5, color='#1f77b4')
    rl_line, = ax.plot(x, rl_dist_arr, 's-', linewidth=1.5, markersize=5, color='#ff7f0e')
    if np.isfinite(baseline_dist_std_arr).any():
        ax.fill_between(x, baseline_dist_arr - baseline_dist_std_arr, baseline_dist_arr + baseline_dist_std_arr, color='#1f77b4', alpha=0.18)
    if np.isfinite(rl_dist_std_arr).any():
        ax.fill_between(x, rl_dist_arr - rl_dist_std_arr, rl_dist_arr + rl_dist_std_arr, color='#ff7f0e', alpha=0.18)
    for i, (xi, eff) in enumerate(zip(x, grouped_df['dist_efficiency'])):
        color = 'green' if eff > 0 else 'red'
        symbol = '+' if eff > 0 else ''
        ax.annotate(f'{symbol}{eff:.1f}%', xy=(xi, rl_dist_arr[i]), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=10, fontweight='bold', color=color, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    ax.set_xlabel('Total Ships in Environment', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Path Length (m)', fontsize=13, fontweight='bold')
    ax.set_title('Total Path Length', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(a)}' for a in x], fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3)
    
    # ===== Panel 2 (Top Right): Total Time Travelled =====
    ax = axs[0, 1]
    base_line, = ax.plot(x, baseline_time_arr, 'o-', linewidth=1.5, markersize=5, color='#1f77b4')
    rl_line, = ax.plot(x, rl_time_arr, 's-', linewidth=1.5, markersize=5, color='#ff7f0e')
    if np.isfinite(baseline_time_std_arr).any():
        ax.fill_between(x, baseline_time_arr - baseline_time_std_arr, baseline_time_arr + baseline_time_std_arr, color='#1f77b4', alpha=0.18)
    if np.isfinite(rl_time_std_arr).any():
        ax.fill_between(x, rl_time_arr - rl_time_std_arr, rl_time_arr + rl_time_std_arr, color='#ff7f0e', alpha=0.18)
    for i, (xi, eff) in enumerate(zip(x, grouped_df['time_efficiency'])):
        color = 'green' if eff > 0 else 'red'
        symbol = '+' if eff > 0 else ''
        ax.annotate(f'{symbol}{eff:.1f}%', xy=(xi, rl_time_arr[i]), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=11, fontweight='bold', color=color, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    ax.set_xlabel('Total Ships in Environment', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Time Travelled (s)', fontsize=13, fontweight='bold')
    ax.set_title('Total Time Travelled', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(a)}' for a in x], fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3)
    
    # ===== Panel 3 (Bottom Left): Min Separation Distance =====
    ax = axs[1, 0]
    base_line, = ax.plot(x, baseline_sep_arr, 'o-', linewidth=1.5, markersize=5, color='#1f77b4')
    rl_line, = ax.plot(x, rl_sep_arr, 's-', linewidth=1.5, markersize=5, color='#ff7f0e')
    if np.isfinite(baseline_sep_std_arr).any():
        ax.fill_between(x, baseline_sep_arr - baseline_sep_std_arr, baseline_sep_arr + baseline_sep_std_arr, color='#1f77b4', alpha=0.18)
    if np.isfinite(rl_sep_std_arr).any():
        ax.fill_between(x, rl_sep_arr - rl_sep_std_arr, rl_sep_arr + rl_sep_std_arr, color='#ff7f0e', alpha=0.18)
    desired_sep = 90.0
    sep_line = ax.axhline(y=desired_sep, color='red', linestyle='--', linewidth=1.5, label=f'Desired Min Sep (3×LOA = {desired_sep:.0f}m)', alpha=0.7)
    for i, (xi, eff) in enumerate(zip(x, grouped_df['sep_efficiency'])):
        color = 'green' if eff > 0 else 'red'
        symbol = '+' if eff > 0 else ''
        ax.annotate(f'{symbol}{eff:.1f}%', xy=(xi, rl_sep_arr[i]), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=11, fontweight='bold', color=color, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    ax.set_xlabel('Total Ships in Environment', fontsize=13, fontweight='bold')
    ax.set_ylabel('Minimum Separation Distance (m)', fontsize=13, fontweight='bold')
    ax.set_title('Min Separation Distance', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(a)}' for a in x], fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(handles=[sep_line], fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    # ===== Panel 4 (Bottom Right): Risk Exposure =====
    ax = axs[1, 1]
    base_line, = ax.plot(x, baseline_risk_arr, 'o-', linewidth=1.5, markersize=5, color='#1f77b4')
    rl_line, = ax.plot(x, rl_risk_arr, 's-', linewidth=1.5, markersize=5, color='#ff7f0e')
    if np.isfinite(baseline_risk_std_arr).any():
        ax.fill_between(x, baseline_risk_arr - baseline_risk_std_arr, baseline_risk_arr + baseline_risk_std_arr, color='#1f77b4', alpha=0.18)
    if np.isfinite(rl_risk_std_arr).any():
        ax.fill_between(x, rl_risk_arr - rl_risk_std_arr, rl_risk_arr + rl_risk_std_arr, color='#ff7f0e', alpha=0.18)
    for i, (xi, eff) in enumerate(zip(x, grouped_df['risk_efficiency'])):
        if np.isnan(eff):
            continue
        color = 'green' if eff > 0 else 'red'
        symbol = '+' if eff > 0 else ''
        ax.annotate(f'{symbol}{eff:.1f}%', xy=(xi, rl_risk_arr[i]), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=11, fontweight='bold', color=color, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    ax.set_xlabel('Total Ships in Environment', fontsize=13, fontweight='bold')
    ax.set_ylabel('Risk Exposure (time-weighted)', fontsize=13, fontweight='bold')
    ax.set_title('Risk Exposure', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(a)}' for a in x], fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add title at the top
    fig.suptitle('Scaling Analysis: RL vs Baseline Performance across Ship Count', fontsize=17, fontweight='bold', y=0.91)
    
    # Add legend below title, above subplots
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', marker='o', linestyle='-', linewidth=2, markersize=6, label='Baseline Ownship'),
        Patch(facecolor='#1f77b4', alpha=0.18, label='Baseline ±1 std'),
        Line2D([0], [0], color='#ff7f0e', marker='s', linestyle='-', linewidth=2, markersize=6, label='RL Ownship'),
        Patch(facecolor='#ff7f0e', alpha=0.18, label='RL ±1 std'),
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.88), ncol=4, fontsize=13, frameon=True, facecolor='white', edgecolor='black')
    
    fig.savefig(output_path, dpi=400, format='png')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


_ENCOUNTER_DIST_NMI = 3.0  # same threshold used in env_RL_ppo.py DCPA tracking


def _fill_encounter_axes(
    ax_dcpa, ax_r, ax_tcpa, ax_risk,
    hist: Dict, n_ts: int,
    system_label: str, lw: float, ls: str,
    tcpa_clip: float = 1500.0,
):
    """
    Plot DCPA / Range / TCPA / Risk from ownship (index 0) vs each target ship
    into four axes.  Colour distinguishes TS; linestyle distinguishes the system.

    DCPA and TCPA are masked to NaN outside the 3 nmi encounter zone (same
    threshold as env_multi_agent_ppo.py) so ill-defined CPA geometry when
    ships are far apart does not produce spikes.  Range (R) and Risk are
    shown for the full episode.

    Units:
    - DCPA in meters
    - Range in meters
    - TCPA in seconds
    - Risk in [0, 1]
    """
    t = hist['t']
    own = 0
    for i in range(n_ts):
        k = i + 1
        color = _TS_PALETTE[i % len(_TS_PALETTE)]
        ts_tag = f" TS{i+1}" if n_ts > 1 else ""
        lbl = f"{system_label}{ts_tag}"

        dist_m    = hist['pair_dist'][:, own, k]
        r_m       = dist_m
        risk_k    = hist['pair_risk'][:, own, k]

        # Mask DCPA and TCPA outside the encounter zone (dist > 3 nmi)
        in_encounter = dist_m <= (_ENCOUNTER_DIST_NMI * NMI)
        dcpa_raw  = hist['pair_dcpa'][:, own, k]
        tcpa_raw  = np.clip(hist['pair_tcpa'][:, own, k], -tcpa_clip, tcpa_clip)
        dcpa_m    = np.where(in_encounter, dcpa_raw,  np.nan)
        tcpa_s    = np.where(in_encounter, tcpa_raw,  np.nan)

        ax_dcpa.plot(t, dcpa_m, color=color, lw=lw, ls=ls, label=lbl)
        ax_r.plot(   t, r_m,    color=color, lw=lw, ls=ls, label=lbl)
        ax_tcpa.plot(t, tcpa_s,  color=color, lw=lw, ls=ls, label=lbl)
        ax_risk.plot(t, risk_k,  color=color, lw=lw, ls=ls, label=lbl)


def plot_timeseries_per_case(
    baseline_dirs: List[Path],
    rl_dirs: List[Path],
    case_numbers: List[int],
    output_dir: Path,
) -> int:
    """
    Per-case 2x2 time-series comparison: DCPA (m), Range (m), TCPA (s), Risk.
    Baseline = solid lines, RL = dashed lines.  Colours distinguish target ships.
    Saves 10_timeseries_case{N:02d}.png for each case with at least one NPZ.
    Returns number of figures saved.
    """
    from matplotlib.lines import Line2D

    n_saved = 0
    for case_num in sorted(case_numbers):
        pat = re.compile(rf"case{case_num}(?:\D|$)")
        bl_match = [d for d in baseline_dirs if pat.search(d.name)]
        rl_match = [d for d in rl_dirs       if pat.search(d.name)]

        bl_hist = load_one_episode_history(bl_match[0]) if bl_match else None
        rl_hist = load_one_episode_history(rl_match[0]) if rl_match else None

        if bl_hist is None and rl_hist is None:
            continue

        n_agents = (bl_hist or rl_hist)['n_agents']
        n_ts     = n_agents - 1

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        ax_dcpa, ax_r    = axes[0]
        ax_tcpa, ax_risk = axes[1]

        if bl_hist is not None:
            _fill_encounter_axes(ax_dcpa, ax_r, ax_tcpa, ax_risk,
                                  bl_hist, n_ts, 'Baseline', lw=1.5, ls='-')
        if rl_hist is not None:
            _fill_encounter_axes(ax_dcpa, ax_r, ax_tcpa, ax_risk,
                                  rl_hist, n_ts, 'RL', lw=1.5, ls='--')

        ax_dcpa.set(ylabel='DCPA (m)', xlabel='Time (s)')
        ax_dcpa.set_ylim(bottom=0)
        ax_r.set(ylabel='Range (m)', xlabel='Time (s)')
        ax_r.set_ylim(bottom=0)
        ax_tcpa.set(ylabel='TCPA (s)', xlabel='Time (s)')
        ax_tcpa.axhline(0, color='k', lw=0.6, ls=':')
        ax_risk.set(ylabel='Risk', xlabel='Time (s)', ylim=(0, 1.05))

        # Increase axis label readability (+2 sizing request).
        ax_dcpa.xaxis.label.set_size(12)
        ax_dcpa.yaxis.label.set_size(12)
        ax_r.xaxis.label.set_size(12)
        ax_r.yaxis.label.set_size(12)
        ax_tcpa.xaxis.label.set_size(12)
        ax_tcpa.yaxis.label.set_size(12)
        ax_risk.xaxis.label.set_size(12)
        ax_risk.yaxis.label.set_size(12)

        for ax in axes.flat:
            ax.grid(alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7, loc='best')

        # System + TS colour legend
        legend_handles = [
            Line2D([0], [0], color='k', ls='-',  lw=1.5, label='Baseline'),
            Line2D([0], [0], color='k', ls='--', lw=1.5, label='RL Policy'),
        ]
        if n_ts > 1:
            for i in range(n_ts):
                legend_handles.append(
                    Line2D([0], [0], color=_TS_PALETTE[i], lw=1.5, label=f'TS{i+1}')
                )
        fig.legend(handles=legend_handles, loc='upper center',
                   ncol=min(6, 2 + n_ts), bbox_to_anchor=(0.5, 1.01), fontsize=9)

        fig.suptitle(
            f"Case {case_num} — Encounter Metrics: Baseline vs RL (Ownship Perspective)",
            y=1.06, fontsize=12, fontweight='bold'
        )
        fig.tight_layout()

        out_path = output_dir / f"10_timeseries_case{case_num:02d}.png"
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        n_saved += 1
        print(f"  Saved: {out_path.name}")

    return n_saved


def plot_timeseries_grid_all(
    baseline_dirs: List[Path],
    rl_dirs: List[Path],
    case_numbers: List[int],
    output_path: Path,
):
    """
    All-cases overview: 4 cases per row, nested 2x2 sub-panels
    (DCPA/Range/TCPA/Risk).
    Saves 11_timeseries_grid_all_cases.png.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D

    sorted_cases = sorted(case_numbers)
    n_cases = len(sorted_cases)
    n_cols  = 4
    n_rows  = math.ceil(n_cases / n_cols)

    fig = plt.figure(figsize=(n_cols * 5.8, n_rows * 5.8))
    # Keep title/legend inside figure bounds and reserve explicit top margin,
    # avoiding large blank area when saving with bbox_inches='tight'.
    outer_gs = gridspec.GridSpec(
        n_rows, n_cols,
        left=0.035, right=0.995, top=0.88, bottom=0.055,
        hspace=0.64, wspace=0.35,
        figure=fig,
    )

    any_data = False
    for idx, case_num in enumerate(sorted_cases):
        r = idx // n_cols
        c = idx % n_cols
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer_gs[r, c], hspace=0.82, wspace=0.32
        )
        ax_dcpa = fig.add_subplot(inner[0, 0])
        ax_r    = fig.add_subplot(inner[0, 1])
        ax_tcpa = fig.add_subplot(inner[1, 0])
        ax_risk = fig.add_subplot(inner[1, 1])

        pat = re.compile(rf"case{case_num}(?:\D|$)")
        bl_match = [d for d in baseline_dirs if pat.search(d.name)]
        rl_match = [d for d in rl_dirs       if pat.search(d.name)]

        bl_hist = load_one_episode_history(bl_match[0]) if bl_match else None
        rl_hist = load_one_episode_history(rl_match[0]) if rl_match else None

        n_agents = get_n_agents_from_case(case_num)
        if bl_hist:  n_agents = bl_hist['n_agents']
        elif rl_hist: n_agents = rl_hist['n_agents']
        n_ts = n_agents - 1

        if bl_hist is not None:
            _fill_encounter_axes(ax_dcpa, ax_r, ax_tcpa, ax_risk,
                                  bl_hist, n_ts, 'BL', lw=1.0, ls='-')
            any_data = True
        if rl_hist is not None:
            _fill_encounter_axes(ax_dcpa, ax_r, ax_tcpa, ax_risk,
                                  rl_hist, n_ts, 'RL', lw=1.0, ls='--')
            any_data = True

        # Center case label over the entire 2x2 quad (instead of only over DCPA axis).
        quad_bbox = outer_gs[r, c].get_position(fig)
        case_x = (quad_bbox.x0 + quad_bbox.x1) * 0.5
        case_y = quad_bbox.y1 + 0.014
        fig.text(
            case_x,
            case_y,
            f"Case {case_num}",
            ha='center', va='bottom', fontsize=21, fontweight='bold'
        )
        fig.text(
            case_x,
            case_y - 0.013,
            f"({n_agents} ships)",
            ha='center', va='bottom', fontsize=13, fontweight='normal'
        )

        # Bold all subplot metric titles for consistency.
        # Double metric-title sizes for DCPA/Range/TCPA/Risk.
        ax_dcpa.set_title('DCPA (m)', fontsize=17, fontweight='bold', pad=2)
        ax_r.set_title('Range (m)', fontsize=17, fontweight='bold', pad=2)
        ax_tcpa.set_title('TCPA (s)', fontsize=17, fontweight='bold', pad=2)
        ax_risk.set_title('Risk', fontsize=17, fontweight='bold', pad=2)

        # Restore readable axis labels.
        # Increase x/y label font sizes by +2.
        ax_dcpa.set_ylabel('DCPA (m)', fontsize=9.5)
        ax_r.set_ylabel('Range (m)', fontsize=9.5)
        ax_tcpa.set_ylabel('TCPA (s)', fontsize=9.5)
        ax_risk.set_ylabel('Risk', fontsize=9.5)

        for ax in [ax_dcpa, ax_r, ax_tcpa, ax_risk]:
            ax.tick_params(labelsize=6.5)
            ax.grid(alpha=0.2)

        # Show x-axis labels on BOTH rows (requested), while larger hspace above
        # prevents overlap with second-row titles.
        ax_dcpa.set_xlabel('Time (s)', fontsize=9.5, labelpad=1)
        ax_r.set_xlabel('Time (s)', fontsize=9.5, labelpad=1)
        ax_tcpa.set_xlabel('Time (s)', fontsize=9.5, labelpad=1)
        ax_risk.set_xlabel('Time (s)', fontsize=9.5, labelpad=1)
        ax_dcpa.set_ylim(bottom=0)
        ax_r.set_ylim(bottom=0)
        ax_tcpa.axhline(0, color='k', lw=0.5, ls=':')
        ax_risk.set_ylim(0, 1.05)

    # Global legend
    legend_handles = [
        Line2D([0], [0], color='k', ls='-',  lw=1.5, label='Baseline'),
        Line2D([0], [0], color='k', ls='--', lw=1.5, label='RL Policy'),
    ]
    for i in range(3):
        legend_handles.append(
            Line2D([0], [0], color=_TS_PALETTE[i], lw=1.5, label=f'TS{i+1}')
        )
    fig.legend(handles=legend_handles, loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 0.945), fontsize=29)
    fig.suptitle(
        "Plot of parameter evolution in all 22 Imazu Test Cases",
        y=0.992, fontsize=53, fontweight='bold'
    )

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    if any_data:
        print(f"  Saved: {output_path.name}")
    else:
        print(f"  No NPZ history data found — {output_path.name} skipped")


def main():
    parser = argparse.ArgumentParser(description="Compare RL vs Baseline metrics across cases")
    parser.add_argument('--base_dir', type=Path, default=Path('.'),
                       help='Base directory searched for baseline dirs (corall_baseline_case*)')
    parser.add_argument('--rl_dir', type=Path, default=None,
                       help='Training directory with eval_cp*_case* subdirs '
                            '(e.g. GENERALIZED_SB3_20260716-130847). '
                            'If omitted, eval_cp* dirs inside --base_dir are used.')
    parser.add_argument('--rl_step', type=int, default=None,
                       help='Checkpoint step to compare (e.g. 850000). '
                            'Filters eval_cp<step>_case* dirs in --rl_dir.')
    parser.add_argument('--case_numbers', nargs='+', type=int, default=[1, 6, 21],
                       help='Case numbers to compare')
    parser.add_argument('--output_dir', type=Path, default=Path('comparison_results'),
                       help='Output directory for charts and CSV')

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rl_src = args.rl_dir or args.base_dir
    step_str = f" (step {args.rl_step})" if args.rl_step else ""
    print(f"\nSearching for baseline dirs in: {args.base_dir}")
    print(f"Searching for RL eval dirs in:  {rl_src}{step_str}")
    baseline_dirs, rl_dirs = find_eval_directories(
        args.base_dir, args.case_numbers, rl_dir=args.rl_dir, rl_step=args.rl_step
    )
    
    print(f"Found {len(baseline_dirs)} baseline and {len(rl_dirs)} RL eval directories")
    
    # Aggregate metrics for each case
    all_metrics = []
    for case_num in sorted(args.case_numbers):
        print(f"\nProcessing Case {case_num}...")
        metrics = aggregate_case_metrics(case_num, baseline_dirs, rl_dirs)
        if metrics:
            all_metrics.append(metrics)
            print(f"  ✓ Case {case_num}: {metrics['n_agents']} agents")
            print(f"    Baseline: min_sep={metrics['baseline_min_sep_nmi']:7.3f} nmi, "
                  f"risk={metrics['baseline_risk_exposure']:7.2f}, "
                  f"dist={metrics['baseline_dist_m']:7.1f} m, time={metrics['baseline_time_s']:7.1f} s")
            print(f"    RL:       min_sep={metrics['rl_min_sep_nmi']:7.3f} nmi, "
                  f"risk={metrics['rl_risk_exposure']:7.2f}, "
                  f"dist={metrics['rl_dist_m']:7.1f} m, time={metrics['rl_time_s']:7.1f} s")
    
    if not all_metrics:
        print("\n❌ No metrics computed. Check directory structure.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Save to CSV
    csv_path = args.output_dir / 'case_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved metrics to: {csv_path.name}")
    
    # Generate charts
    print("\nGenerating bar charts...")
    create_bar_chart_separation(df, args.output_dir / '01_min_separation_by_case.png')
    create_bar_chart_distance(df, args.output_dir / '02_path_length_by_case.png')
    create_bar_chart_time(df, args.output_dir / '03_time_by_case.png')
    create_bar_chart_risk_exposure(df, args.output_dir / '04_risk_exposure_by_case.png')
    create_bar_chart_collision_rate(df, args.output_dir / '05_collision_rate_by_case.png')
    create_bar_chart_success_rate(df, args.output_dir / '06_success_rate_by_case.png')
    create_summary_guaranteed_success(df, args.output_dir / '07_guaranteed_success_summary.png')
    create_scaling_chart_separation(df, args.output_dir / '08_separation_scaling_by_ships.png')
    
    print("\nGenerating scaling line charts...")
    create_scaling_line_charts(df, args.output_dir / '09_scaling_analysis_lines.png')

    print("\nGenerating time-series encounter plots...")
    n_ts_plots = plot_timeseries_per_case(
        baseline_dirs, rl_dirs, args.case_numbers, args.output_dir
    )
    if n_ts_plots > 0:
        plot_timeseries_grid_all(
            baseline_dirs, rl_dirs, args.case_numbers,
            args.output_dir / '11_timeseries_grid_all_cases.png'
        )
    else:
        print("  No episode histories found — run RL eval with --save_first_history")
        print("  and ensure baseline episode_histories/ dirs exist.")

    # Print summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    for _, row in df.iterrows():
        case = int(row['case'])
        agents = int(row['n_agents'])
        sep_pct = ((row['baseline_min_sep_nmi'] - row['rl_min_sep_nmi']) / row['baseline_min_sep_nmi']) * 100 if row['baseline_min_sep_nmi'] > 0 else 0
        risk_pct = ((row['baseline_risk_exposure'] - row['rl_risk_exposure']) / row['baseline_risk_exposure']) * 100 if row['baseline_risk_exposure'] > 0 else 0
        dist_pct = ((row['baseline_dist_m'] - row['rl_dist_m']) / row['baseline_dist_m']) * 100
        time_pct = ((row['baseline_time_s'] - row['rl_time_s']) / row['baseline_time_s']) * 100
        
        baseline_success = row.get('baseline_success_rate')
        if pd.isna(baseline_success) or baseline_success is None:
            baseline_success = (1 - row['baseline_collision_rate']) * 100
        else:
            baseline_success = baseline_success * 100
            
        rl_success = row.get('rl_success_rate')
        if pd.isna(rl_success) or rl_success is None:
            rl_success = (1 - row['rl_collision_rate']) * 100
        else:
            rl_success = rl_success * 100
        
        print(f"\nCase {case} ({agents} total ships):")
        print(f"  Min Separation: Baseline {row['baseline_min_sep_nmi']:7.3f} nmi vs RL {row['rl_min_sep_nmi']:7.3f} nmi ({sep_pct:+6.1f}%)")
        print(f"  Risk Exposure:  Baseline {row['baseline_risk_exposure']:7.2f}    vs RL {row['rl_risk_exposure']:7.2f}    ({risk_pct:+6.1f}%)")
        print(f"  Distance:       Baseline {row['baseline_dist_m']:7.1f} m  vs RL {row['rl_dist_m']:7.1f} m  ({dist_pct:+6.1f}%)")
        print(f"  Time:           Baseline {row['baseline_time_s']:7.1f} s  vs RL {row['rl_time_s']:7.1f} s  ({time_pct:+6.1f}%)")
        print(f"  Collision Rate: Baseline {row['baseline_collision_rate']*100:6.1f}% vs RL {row['rl_collision_rate']*100:6.1f}%")
        print(f"  Success Rate:   Baseline {baseline_success:6.1f}% vs RL {rl_success:6.1f}% (FROM CSV)")
    
    print("\n" + "="*70)
    print(f"All results saved to: {args.output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
