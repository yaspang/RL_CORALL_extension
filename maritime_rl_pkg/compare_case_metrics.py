"""
Compare RL policy vs baseline across multiple metrics using bar charts:
1. Min Separation Distance by case (RL vs Baseline)
2. Total path length by case with efficiency %
3. Total time traveled by case with efficiency %
4. Risk Exposure (time-weighted risk) by case
5. Scaling graph: Min Separation Distance vs number of ships (agents) in environment
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NMI = 1852.0


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


def aggregate_case_metrics(
    case_num: int,
    baseline_dirs: List[Path],
    rl_dirs: List[Path],
) -> Dict | None:
    """
    Aggregate metrics for a single case across all episodes.
    
    Returns dict with keys: case, n_agents, n_additional_agents,
                            baseline_min_dcpa, rl_min_dcpa,
                            baseline_dist_m, rl_dist_m,
                            baseline_time_s, rl_time_s,
                            n_episodes_baseline, n_episodes_rl
    """
    # Find directories for this case
    baseline_for_case = [d for d in baseline_dirs if f"case{case_num}" in d.name]
    rl_for_case = [d for d in rl_dirs if f"case{case_num}" in d.name]
    
    if not baseline_for_case or not rl_for_case:
        return None
    
    baseline_dir = baseline_for_case[0]
    rl_dir = rl_for_case[0]
    
    # Load histories
    baseline_hists = load_episode_histories(baseline_dir)
    rl_hists = load_episode_histories(rl_dir)
    
    if not baseline_hists or not rl_hists:
        print(f"  No histories found for case {case_num}")
        return None
    
    # Get number of agents
    n_agents = baseline_hists[0]['n_agents']
    n_additional_agents = n_agents - 1  # Exclude ownship
    
    # Compute metrics for baseline
    baseline_min_sep_vals = []
    baseline_risk_vals = []
    baseline_distances = []
    baseline_times = []
    baseline_collisions = []
    
    for hist in baseline_hists:
        try:
            min_sep = compute_min_separation(hist['pair_dist'])
            dist = compute_distance_traveled(hist['X_all'])
            time = compute_total_time(hist['t'])
            collision = hist['collision']
            
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
            baseline_collisions.append(collision)
        except Exception as e:
            print(f"    Warning: Failed to compute baseline metrics: {e}")
    
    # Compute metrics for RL
    rl_min_sep_vals = []
    rl_risk_vals = []
    rl_distances = []
    rl_times = []
    rl_collisions = []
    
    for hist in rl_hists:
        try:
            min_sep = compute_min_separation(hist['pair_dist'])
            dist = compute_distance_traveled(hist['X_all'])
            time = compute_total_time(hist['t'])
            collision = hist['collision']
            
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
            rl_collisions.append(collision)
        except Exception as e:
            print(f"    Warning: Failed to compute RL metrics: {e}")
    
    # Average metrics
    baseline_min_sep_avg = np.mean(baseline_min_sep_vals) if baseline_min_sep_vals else np.nan
    rl_min_sep_avg = np.mean(rl_min_sep_vals) if rl_min_sep_vals else np.nan
    
    baseline_risk_avg = np.mean(baseline_risk_vals) if baseline_risk_vals else np.nan
    rl_risk_avg = np.mean(rl_risk_vals) if rl_risk_vals else np.nan
    
    baseline_dist_avg = np.mean(baseline_distances) if baseline_distances else np.nan
    rl_dist_avg = np.mean(rl_distances) if rl_distances else np.nan
    
    baseline_time_avg = np.mean(baseline_times) if baseline_times else np.nan
    rl_time_avg = np.mean(rl_times) if rl_times else np.nan
    
    baseline_collision_rate = np.mean(baseline_collisions) if baseline_collisions else np.nan
    rl_collision_rate = np.mean(rl_collisions) if rl_collisions else np.nan
    
    return {
        'case': case_num,
        'n_agents': n_agents,
        'n_additional_agents': n_additional_agents,
        'baseline_min_sep_nmi': baseline_min_sep_avg,
        'rl_min_sep_nmi': rl_min_sep_avg,
        'baseline_risk_exposure': baseline_risk_avg,
        'rl_risk_exposure': rl_risk_avg,
        'baseline_dist_m': baseline_dist_avg,
        'rl_dist_m': rl_dist_avg,
        'baseline_time_s': baseline_time_avg,
        'rl_time_s': rl_time_avg,
        'baseline_collision_rate': baseline_collision_rate,
        'rl_collision_rate': rl_collision_rate,
        'n_episodes_baseline': len(baseline_distances),
        'n_episodes_rl': len(rl_distances),
    }


def find_eval_directories(base_dir: Path, case_numbers: List[int]) -> Tuple[List[Path], List[Path]]:
    """Find baseline and RL evaluation directories for given cases"""
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
        
        # Match RL evaluation directories
        elif "policy_eval" in item.name and "sb3" in item.name:
            match = re.search(r"case(\d+)", item.name)
            if match and int(match.group(1)) in case_numbers:
                rl_dirs.append(item)
    
    return baseline_dirs, rl_dirs


def create_bar_chart_separation(metrics_df: pd.DataFrame, output_path: Path):
    """Chart 1: Min Separation Distance by case (RL vs Baseline)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics_df))
    width = 0.35
    
    case_labels = [f"Case {int(c)} ({int(n)} agents)" 
                   for c, n in zip(metrics_df['case'], metrics_df['n_agents'])]
    
    # Convert from nmi to meters for consistency
    baseline_sep_m = metrics_df['baseline_min_sep_nmi'] * NMI
    rl_sep_m = metrics_df['rl_min_sep_nmi'] * NMI
    
    ax.bar(x - width/2, baseline_sep_m, width, label='Baseline', color='black')
    ax.bar(x + width/2, rl_sep_m, width, label='RL Policy', color='purple')
    
    ax.set_xlabel('Case', fontsize=12, fontweight='bold')
    ax.set_ylabel('Minimum Separation Distance (m)', fontsize=12, fontweight='bold')
    ax.set_title('Minimum Separation Distance: Baseline vs RL Policy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def create_bar_chart_distance(metrics_df: pd.DataFrame, output_path: Path):
    """Chart 2: Total path length by case with efficiency %"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics_df))
    width = 0.35
    
    case_labels = [f"Case {int(c)} ({int(n)} agents)" 
                   for c, n in zip(metrics_df['case'], metrics_df['n_agents'])]
    
    baseline_dist = metrics_df['baseline_dist_m']
    rl_dist = metrics_df['rl_dist_m']
    
    bars1 = ax.bar(x - width/2, baseline_dist, width, label='Baseline', color='black')
    bars2 = ax.bar(x + width/2, rl_dist, width, label='RL Policy', color='purple')
    
    # Add efficiency labels on bars
    for i, (b_dist, r_dist) in enumerate(zip(baseline_dist, rl_dist)):
        efficiency_pct = ((b_dist - r_dist) / b_dist) * 100
        ax.text(i + width/2, r_dist + 50, 
               f'{efficiency_pct:+.1f}%', ha='center', fontsize=9, fontweight='bold',
               color='darkred' if efficiency_pct > 0 else 'darkblue')
    
    ax.set_xlabel('Case', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Path Length (meters)', fontsize=12, fontweight='bold')
    ax.set_title('Total Path Length Traveled (Ownship): Baseline vs RL Policy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def create_bar_chart_time(metrics_df: pd.DataFrame, output_path: Path):
    """Chart 3: Total time traveled by case with efficiency %"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics_df))
    width = 0.35
    
    case_labels = [f"Case {int(c)} ({int(n)} agents)" 
                   for c, n in zip(metrics_df['case'], metrics_df['n_agents'])]
    
    baseline_time = metrics_df['baseline_time_s']
    rl_time = metrics_df['rl_time_s']
    
    bars1 = ax.bar(x - width/2, baseline_time, width, label='Baseline', color='black')
    bars2 = ax.bar(x + width/2, rl_time, width, label='RL Policy', color='purple')
    
    # Add efficiency labels on bars
    for i, (b_time, r_time) in enumerate(zip(baseline_time, rl_time)):
        efficiency_pct = ((b_time - r_time) / b_time) * 100
        ax.text(i + width/2, r_time + 3, 
               f'{efficiency_pct:+.1f}%', ha='center', fontsize=9, fontweight='bold',
               color='darkred' if efficiency_pct > 0 else 'darkblue')
    
    ax.set_xlabel('Case', fontsize=12, fontweight='bold')
    ax.set_ylabel('Episode Duration (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Time to Goal: Baseline vs RL Policy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def create_bar_chart_risk_exposure(metrics_df: pd.DataFrame, output_path: Path):
    """Chart 4: Risk Exposure (time-weighted risk) by case"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics_df))
    width = 0.35
    
    case_labels = [f"Case {int(c)} ({int(n)} agents)" 
                   for c, n in zip(metrics_df['case'], metrics_df['n_agents'])]
    
    baseline_risk = metrics_df['baseline_risk_exposure']
    rl_risk = metrics_df['rl_risk_exposure']
    
    bars1 = ax.bar(x - width/2, baseline_risk, width, label='Baseline', color='black')
    bars2 = ax.bar(x + width/2, rl_risk, width, label='RL Policy', color='purple')
    
    # Add difference labels on bars
    for i, (b_risk, r_risk) in enumerate(zip(baseline_risk, rl_risk)):
        if b_risk > 0:
            risk_reduction_pct = ((b_risk - r_risk) / b_risk) * 100
            ax.text(i + width/2, r_risk + 0.05 * baseline_risk.max(), 
                   f'{risk_reduction_pct:+.1f}%', ha='center', fontsize=9, fontweight='bold',
                   color='darkred' if risk_reduction_pct > 0 else 'darkblue')
    
    ax.set_xlabel('Case', fontsize=12, fontweight='bold')
    ax.set_ylabel('Risk Exposure (time-weighted)', fontsize=12, fontweight='bold')
    ax.set_title('Risk Exposure: Baseline vs RL Policy (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def create_bar_chart_collision_rate(metrics_df: pd.DataFrame, output_path: Path):
    """Chart: Collision Rate (%) by case"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics_df))
    width = 0.35
    
    case_labels = [f"Case {int(c)} ({int(n)} agents)" 
                   for c, n in zip(metrics_df['case'], metrics_df['n_agents'])]
    
    baseline_collision_pct = metrics_df['baseline_collision_rate'] * 100
    rl_collision_pct = metrics_df['rl_collision_rate'] * 100
    
    bars1 = ax.bar(x - width/2, baseline_collision_pct, width, label='Baseline', color='black')
    bars2 = ax.bar(x + width/2, rl_collision_pct, width, label='RL Policy', color='purple')
    
    # Add collision counts on bars
    for i, (b_rate, r_rate) in enumerate(zip(baseline_collision_pct, rl_collision_pct)):
        ax.text(i - width/2, b_rate + 1, f'{b_rate:.1f}%', ha='center', fontsize=9, fontweight='bold')
        ax.text(i + width/2, r_rate + 1, f'{r_rate:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Case', fontsize=12, fontweight='bold')
    ax.set_ylabel('Collision Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Collision Rate: Baseline vs RL Policy (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def create_scaling_chart_separation(metrics_df: pd.DataFrame, output_path: Path):
    """Chart 5: Min Separation Distance vs number of ships in environment (agent count)"""
    # Group by total number of agents
    grouped = metrics_df.groupby('n_agents')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
           label='Baseline (avg)', color='black')
    ax.bar(np.array(x_positions) + width/2, rl_sep_by_agents, width, 
           label='RL Policy (avg)', color='purple')
    
    ax.set_xlabel('Total Ships in Environment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Minimum Separation Distance (m)', fontsize=12, fontweight='bold')
    ax.set_title('Minimum Separation Distance Scaling: Impact of Ships in Environment', fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def create_scaling_line_charts(metrics_df: pd.DataFrame, output_path: Path):
    """Chart: 2-row grid (2 efficiency plots on top, 3 colav/risk plots on bottom)"""
    # Group by number of agents
    grouped_data = []
    for agent_count in sorted(metrics_df['n_agents'].unique()):
        group = metrics_df[metrics_df['n_agents'] == agent_count]
        
        baseline_sep = group['baseline_min_sep_nmi'].mean() * NMI  # Convert to meters
        rl_sep = group['rl_min_sep_nmi'].mean() * NMI  # Convert to meters
        # For separation: HIGHER is better, so if RL is higher (positive efficiency) = green
        sep_efficiency = ((rl_sep - baseline_sep) / baseline_sep * 100) if baseline_sep > 0 else 0
        
        baseline_risk = group['baseline_risk_exposure'].mean()
        rl_risk = group['rl_risk_exposure'].mean()
        # For risk: LOWER is better, so if RL is lower (positive efficiency) = green
        risk_efficiency = ((baseline_risk - rl_risk) / baseline_risk * 100) if baseline_risk > 0 else 0
        
        baseline_dist = group['baseline_dist_m'].mean()
        rl_dist = group['rl_dist_m'].mean()
        # For distance: LOWER is better, so if RL is lower (positive efficiency) = green
        dist_efficiency = ((baseline_dist - rl_dist) / baseline_dist * 100)
        
        baseline_time = group['baseline_time_s'].mean()
        rl_time = group['rl_time_s'].mean()
        # For time: LOWER is better, so if RL is lower (positive efficiency) = green
        time_efficiency = ((baseline_time - rl_time) / baseline_time * 100)
        
        baseline_collision = group['baseline_collision_rate'].mean()
        rl_collision = group['rl_collision_rate'].mean()
        # For collision rate: LOWER is better, so if RL is lower (positive efficiency) = green
        collision_efficiency = ((baseline_collision - rl_collision) / max(baseline_collision, 0.001) * 100)
        
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
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # ===== TOP ROW: EFFICIENCY METRICS =====
    
    # ===== Panel 1 (Top Left): Total Path Length Efficiency =====
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, baseline_dist_arr, 'o-', linewidth=2.5, markersize=8, 
           label='Baseline', color='black')
    ax.plot(x, rl_dist_arr, 's-', linewidth=2.5, markersize=8, 
           label='RL Policy', color='purple')
    
    # Add efficiency percentages (GREEN for shorter path, RED for longer)
    for i, (xi, eff) in enumerate(zip(x, grouped_df['dist_efficiency'])):
        color = 'green' if eff > 0 else 'red'
        symbol = '+' if eff > 0 else ''
        ax.text(xi, rl_dist_arr[i] - 80, 
               f'{symbol}{eff:.1f}%', ha='center', fontsize=10, fontweight='bold',
               color=color, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Total Ships in Environment', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Path Length (m)', fontsize=11, fontweight='bold')
    ax.set_title('Total Path Length', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(a)}' for a in x])
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # ===== Panel 2 (Top Right): Total Time Travelled Efficiency =====
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(x, baseline_time_arr, 'o-', linewidth=2.5, markersize=8, 
           label='Baseline', color='black')
    ax.plot(x, rl_time_arr, 's-', linewidth=2.5, markersize=8, 
           label='RL Policy', color='purple')
    
    # Add efficiency percentages (GREEN for shorter time, RED for longer time - more time is worse)
    for i, (xi, eff) in enumerate(zip(x, grouped_df['time_efficiency'])):
        color = 'green' if eff > 0 else 'red'
        symbol = '+' if eff > 0 else ''
        ax.text(xi, rl_time_arr[i] + 10, 
               f'{symbol}{eff:.1f}%', ha='center', fontsize=10, fontweight='bold',
               color=color, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Total Ships in Environment', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Time Travelled (s)', fontsize=11, fontweight='bold')
    ax.set_title('Total Time Travelled', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(a)}' for a in x])
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Blank panel for symmetry
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    
    # ===== BOTTOM ROW: COLLISION AVOIDANCE / RISK METRICS =====
    
    # ===== Panel 3 (Bottom Left): Min Separation Distance =====
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x, baseline_sep_arr, 'o-', linewidth=2.5, markersize=8, 
           label='Baseline', color='black')
    ax.plot(x, rl_sep_arr, 's-', linewidth=2.5, markersize=8, 
           label='RL Policy', color='purple')
    
    # Add desired min separation distance line (3 * LOA = 90m)
    desired_sep = 90.0  # 3 * LOA (LOA=30m default)
    ax.axhline(y=desired_sep, color='red', linestyle='--', linewidth=2.0, 
              label=f'Desired Min Sep (3×LOA = {desired_sep:.0f}m)', alpha=0.7)
    
    # Add efficiency percentages (GREEN for higher separation = safer, RED for lower)
    for i, (xi, eff) in enumerate(zip(x, grouped_df['sep_efficiency'])):
        color = 'green' if eff > 0 else 'red'
        symbol = '+' if eff > 0 else ''
        ax.text(xi, rl_sep_arr[i] - 50, 
               f'{symbol}{eff:.1f}%', ha='center', fontsize=10, fontweight='bold',
               color=color, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Total Ships in Environment', fontsize=11, fontweight='bold')
    ax.set_ylabel('Minimum Separation Distance (m)', fontsize=11, fontweight='bold')
    ax.set_title('Min Separation Distance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(a)}' for a in x])
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # ===== Panel 4 (Bottom Center): Risk Exposure =====
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x, baseline_risk_arr, 'o-', linewidth=2.5, markersize=8, 
           label='Baseline', color='black')
    ax.plot(x, rl_risk_arr, 's-', linewidth=2.5, markersize=8, 
           label='RL Policy', color='purple')
    
    # Add efficiency percentages (GREEN for lower risk, RED for higher)
    for i, (xi, eff) in enumerate(zip(x, grouped_df['risk_efficiency'])):
        color = 'green' if eff > 0 else 'red'
        symbol = '+' if eff > 0 else ''
        ax.text(xi, rl_risk_arr[i] + 0.03 * baseline_risk_arr.max(), 
               f'{symbol}{eff:.1f}%', ha='center', fontsize=10, fontweight='bold',
               color=color, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Total Ships in Environment', fontsize=11, fontweight='bold')
    ax.set_ylabel('Risk Exposure (time-weighted)', fontsize=11, fontweight='bold')
    ax.set_title('Risk Exposure', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(a)}' for a in x])
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # ===== Panel 5 (Bottom Right): Collision Rate =====
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(x, baseline_collision_arr, 'o-', linewidth=2.5, markersize=8, 
           label='Baseline', color='black')
    ax.plot(x, rl_collision_arr, 's-', linewidth=2.5, markersize=8, 
           label='RL Policy', color='purple')
    
    # Add efficiency percentages (GREEN for lower collision rate, RED for higher)
    for i, (xi, eff) in enumerate(zip(x, grouped_df['collision_efficiency'])):
        color = 'green' if eff > 0 else 'red'
        symbol = '+' if eff > 0 else ''
        ax.text(xi, rl_collision_arr[i] + 5, 
               f'{symbol}{eff:.1f}%', ha='center', fontsize=10, fontweight='bold',
               color=color, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Total Ships in Environment', fontsize=11, fontweight='bold')
    ax.set_ylabel('Collision Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Collision Rate', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(a)}' for a in x])
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Scaling Analysis: RL vs Baseline Performance across Ship Count\n(Top: Efficiency | Bottom: Collision Avoidance & Risk)', 
                fontsize=14, fontweight='bold', y=0.997)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Compare RL vs Baseline metrics across cases")
    parser.add_argument('--base_dir', type=Path, default=Path('.'), 
                       help='Base directory containing eval directories')
    parser.add_argument('--case_numbers', nargs='+', type=int, default=[1, 6, 21],
                       help='Case numbers to compare')
    parser.add_argument('--output_dir', type=Path, default=Path('comparison_results'),
                       help='Output directory for charts and CSV')
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSearching for evaluation directories in {args.base_dir}")
    baseline_dirs, rl_dirs = find_eval_directories(args.base_dir, args.case_numbers)
    
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
    create_scaling_chart_separation(df, args.output_dir / '06_separation_scaling_by_ships.png')
    
    print("\nGenerating scaling line charts...")
    create_scaling_line_charts(df, args.output_dir / '07_scaling_analysis_lines.png')
    
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
        
        print(f"\nCase {case} ({agents} total ships):")
        print(f"  Min Separation: Baseline {row['baseline_min_sep_nmi']:7.3f} nmi vs RL {row['rl_min_sep_nmi']:7.3f} nmi ({sep_pct:+6.1f}%)")
        print(f"  Risk Exposure:  Baseline {row['baseline_risk_exposure']:7.2f}    vs RL {row['rl_risk_exposure']:7.2f}    ({risk_pct:+6.1f}%)")
        print(f"  Distance:       Baseline {row['baseline_dist_m']:7.1f} m  vs RL {row['rl_dist_m']:7.1f} m  ({dist_pct:+6.1f}%)")
        print(f"  Time:           Baseline {row['baseline_time_s']:7.1f} s  vs RL {row['rl_time_s']:7.1f} s  ({time_pct:+6.1f}%)")
        print(f"  Collision Rate: Baseline {row['baseline_collision_rate']*100:6.1f}% vs RL {row['rl_collision_rate']*100:6.1f}%")
    
    print("\n" + "="*70)
    print(f"All results saved to: {args.output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
