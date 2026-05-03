"""
Plot risk exposure comparison between new RL policy and baseline CORALL.
Validates whether the updated reward function (w_risk=-10) is achieving desired safety improvements.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re


def extract_case_number(dirname):
    """Extract case number from directory name."""
    match = re.search(r'case(\d+)_', dirname)
    if match:
        return int(match.group(1))
    return None


def load_risk_exposure(base_dir, case_numbers):
    """Load risk exposure metrics from evaluation directories."""
    
    baseline_risks = {}
    rl_risks = {}
    
    for case in case_numbers:
        # Find baseline directory for this case
        baseline_pattern = f"corall_baseline_case{case}_*"
        baseline_dirs = glob.glob(str(Path(base_dir) / baseline_pattern))
        
        if baseline_dirs:
            baseline_dir = baseline_dirs[0]  # Take most recent
            summary_path = Path(baseline_dir) / "seed_0" / "policy_eval_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    data = json.load(f)
                    baseline_risks[case] = data.get('risk_exposure_ownship_mean', np.nan)
        
        # Find RL directory for this case
        rl_pattern = f"policy_eval_generalized_sb3_case{case}_*"
        rl_dirs = glob.glob(str(Path(base_dir) / rl_pattern))
        
        if rl_dirs:
            rl_dir = rl_dirs[0]  # Take most recent
            summary_path = Path(rl_dir) / "seed_0" / "policy_eval_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    data = json.load(f)
                    rl_risks[case] = data.get('risk_exposure_ownship_mean', np.nan)
    
    return baseline_risks, rl_risks


def plot_risk_exposure(baseline_risks, rl_risks, output_dir="."):
    """Create risk exposure comparison plot."""
    
    # Extract cases and sort
    cases = sorted(set(list(baseline_risks.keys()) + list(rl_risks.keys())))
    
    baseline_values = np.array([baseline_risks.get(c, np.nan) for c in cases])
    rl_values = np.array([rl_risks.get(c, np.nan) for c in cases])
    
    # Compute risk reduction
    valid_mask = ~(np.isnan(baseline_values) | np.isnan(rl_values))
    risk_reduction = np.full_like(rl_values, np.nan)
    risk_reduction[valid_mask] = (1 - rl_values[valid_mask] / baseline_values[valid_mask]) * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(cases))
    width = 0.35
    
    # Plot bars
    ax.bar(x - width/2, baseline_values, width, label='Baseline CORALL', 
           color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1.5)
    ax.bar(x + width/2, rl_values, width, label='RL Policy (w_risk=-10)', 
           color='darkorange', alpha=0.8, edgecolor='orangered', linewidth=1.5)
    
    # Add value labels and reduction percentages
    for i, (case, baseline, rl) in enumerate(zip(cases, baseline_values, rl_values)):
        # Baseline label
        if not np.isnan(baseline):
            ax.text(i - width/2, baseline + 5, f'{baseline:.0f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # RL label
        if not np.isnan(rl):
            ax.text(i + width/2, rl + 5, f'{rl:.0f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Risk reduction percentage
        if not np.isnan(risk_reduction[i]):
            reduction = risk_reduction[i]
            color = 'green' if reduction > 0 else 'red'
            symbol = '↓' if reduction > 0 else '↑'
            ax.text(i, max(baseline, rl) + 40, f'{symbol} {abs(reduction):.1f}%',
                   ha='center', va='bottom', fontsize=8, fontweight='bold', color=color)
    
    # Reference line for safety threshold (typical maritime)
    ax.axhline(y=300, color='red', linestyle='--', alpha=0.5, linewidth=2, 
               label='Safety Threshold (~300 m·s)')
    
    ax.set_xlabel('Case Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('Risk Exposure (m·s)', fontsize=13, fontweight='bold')
    ax.set_title('Risk Exposure Comparison: RL Policy vs Baseline CORALL\n(New Reward Function Validation)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    output_path = Path(output_dir) / "risk_exposure_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Risk exposure plot saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("RISK EXPOSURE ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Case':<6} {'Baseline':<12} {'RL Policy':<12} {'Reduction':<12} {'Status':<15}")
    print("-" * 60)
    
    for i, case in enumerate(cases):
        baseline = baseline_values[i]
        rl = rl_values[i]
        reduction = risk_reduction[i]
        
        if np.isnan(baseline) or np.isnan(rl):
            status = "⚠️  MISSING"
        elif reduction < -10:  # RL increased risk significantly
            status = "🔴 WORSE"
        elif reduction < 0:  # RL slightly worse
            status = "🟡 SLIGHTLY WORSE"
        elif reduction < 10:  # Small improvement
            status = "🟠 MINIMAL"
        elif reduction < 30:  # Moderate improvement
            status = "🟡 MODERATE"
        elif reduction < 50:  # Good improvement
            status = "🟢 GOOD"
        else:  # Excellent improvement
            status = "🟢 EXCELLENT"
        
        print(f"{case:<6} {baseline:<12.1f} {rl:<12.1f} {reduction:<12.1f}% {status:<15}")
    
    # Overall statistics
    valid_mask = ~np.isnan(risk_reduction)
    if np.any(valid_mask):
        mean_reduction = np.mean(risk_reduction[valid_mask])
        print("-" * 60)
        print(f"{'OVERALL':<6} {np.nanmean(baseline_values):<12.1f} {np.nanmean(rl_values):<12.1f} {mean_reduction:<12.1f}%")
        
        print("\n" + "=" * 80)
        print("INTERPRETATION")
        print("=" * 80)
        
        improved_cases = np.sum(risk_reduction[valid_mask] > 0)
        degraded_cases = np.sum(risk_reduction[valid_mask] < 0)
        
        print(f"\n✓ Improved cases: {improved_cases} out of {np.sum(valid_mask)}")
        print(f"✗ Degraded cases: {degraded_cases} out of {np.sum(valid_mask)}")
        print(f"\nMean risk reduction: {mean_reduction:.1f}%")
        
        if mean_reduction > 20:
            print("✓ VERDICT: New reward function showing GOOD safety improvement!")
        elif mean_reduction > 5:
            print("🟡 VERDICT: Moderate improvement - may need further tuning")
        elif mean_reduction > -5:
            print("🟡 VERDICT: Minimal change - reward function may be insufficient")
        else:
            print("✗ VERDICT: Risk INCREASED - reward function needs significant adjustment")
        
        # Case-by-case analysis
        print("\nDETAILED ANALYSIS BY SHIP COUNT:")
        
        # Group by ship count
        two_ship_cases = [c for c in cases if c <= 9]
        three_ship_cases = [c for c in cases if 10 <= c <= 17]
        four_ship_cases = [c for c in cases if c >= 18]
        
        for group_name, group_cases in [("2-ship (cases 1-9)", two_ship_cases), 
                                        ("3-ship (cases 10-17)", three_ship_cases),
                                        ("4-ship (cases 18-22)", four_ship_cases)]:
            group_reductions = [risk_reduction[cases.index(c)] for c in group_cases if c in cases]
            if group_reductions:
                valid_reductions = [r for r in group_reductions if not np.isnan(r)]
                if valid_reductions:
                    mean_group_reduction = np.mean(valid_reductions)
                    print(f"\n{group_name}: {mean_group_reduction:.1f}% avg reduction")
                    if mean_group_reduction > 20:
                        print(f"  ✓ Good performance on {group_name.split()[0].lower()}-ship cases")
                    elif mean_group_reduction < 0:
                        print(f"  ⚠️  CONCERN: Risk INCREASED on {group_name.split()[0].lower()}-ship cases!")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    import sys
    
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    
    # Cases to evaluate
    case_numbers = list(range(1, 23))  # 1-22
    
    print("Loading risk exposure metrics...")
    baseline_risks, rl_risks = load_risk_exposure(base_dir, case_numbers)
    
    print(f"Loaded {len(baseline_risks)} baseline cases and {len(rl_risks)} RL policy cases\n")
    
    plot_risk_exposure(baseline_risks, rl_risks, output_dir)
