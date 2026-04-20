"""
Generate trajectory overlay plots (full + encounter detail) for selected cases.
Compares best RL episode vs matching baseline episode.

Usage:
    python generate_trajectory_overlays.py --cases 1 10 18 --output_dir Visualizations/trajectory_overlays
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from maritime_rl_pkg.path_setup import ensure_paths
ensure_paths()

from maritime_rl_pkg.episode_overlay_tools import (
    plot_full_trajectory_overlay,
    plot_encounter_detail_clean,
    plot_ownship_threat_profile,
    plot_stacked_trajectory_overlay,
)

BASE_DIR = Path(__file__).parent


def find_case_dir(base: Path, case_num: int, pattern_prefix: str) -> Path | None:
    """Find directory matching exact case number."""
    case_re = re.compile(rf"case{case_num}(?:\D|$)")
    for d in sorted(base.iterdir()):
        if d.is_dir() and pattern_prefix in d.name and case_re.search(d.name):
            return d
    return None


def find_best_episode_file(eval_dir: Path) -> Path | None:
    """Find the best episode NPZ file based on CSV returns."""
    seed_dir = eval_dir / "seed_0"
    csv_file = seed_dir / "policy_eval_per_episode.csv"
    hist_dir = seed_dir / "episode_histories"

    if not csv_file.exists() or not hist_dir.exists():
        return None

    with open(csv_file) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    # Parse returns
    episodes = []
    for row in rows:
        ret = float(row.get("episode_return") or row.get("episode_return_ownship", "0"))
        idx = int(row.get("episode_index", 0))
        seed = int(row.get("episode_seed", 0))
        episodes.append((ret, idx, seed))

    best_ret, best_idx, best_seed = max(episodes, key=lambda x: x[0])

    # Find matching NPZ file
    ep_token = f"_ep{best_idx:03d}"
    seed_token = f"_seed{best_seed}"
    for npz in sorted(hist_dir.glob("*.npz")):
        if ep_token in npz.stem and seed_token in npz.stem:
            return npz

    # Fallback: match just episode index
    for npz in sorted(hist_dir.glob("*.npz")):
        if ep_token in npz.stem:
            return npz

    return None


def find_matching_baseline_file(baseline_dir: Path, rl_seed: int) -> Path | None:
    """Find baseline episode with matching seed, or first available."""
    hist_dir = baseline_dir / "seed_0" / "episode_histories"
    if not hist_dir.exists():
        return None

    # Try matching seed
    seed_token = f"_seed{rl_seed}"
    for npz in sorted(hist_dir.glob("*.npz")):
        if seed_token in npz.stem:
            return npz

    # Fallback: first episode
    npzs = sorted(hist_dir.glob("*.npz"))
    return npzs[0] if npzs else None


def load_npz_as_dict(npz_path: Path) -> dict:
    """Load NPZ file and convert to dict compatible with episode_overlay_tools."""
    data = np.load(npz_path, allow_pickle=True)
    hist = {}
    for key in data.keys():
        val = data[key]
        if key in ("case", "seed"):
            hist[key] = int(val.item() if hasattr(val, "item") else val[0])
        elif key in ("baseline", "checkpoint"):
            hist[key] = str(val[0]) if len(val) > 0 else ""
        elif key in ("final_waypoint_x_nmi", "final_waypoint_y_nmi"):
            v = val[0] if len(val) > 0 else None
            hist[key] = float(v) if v is not None else None
        else:
            hist[key] = val.tolist()
    return hist


def main():
    parser = argparse.ArgumentParser(description="Generate trajectory overlay plots")
    parser.add_argument("--cases", nargs="+", type=int, default=[1, 10, 18],
                        help="Case numbers to generate overlays for")
    parser.add_argument("--output_dir", type=Path, default=Path("Visualizations/trajectory_overlays"),
                        help="Output directory for plots")
    parser.add_argument("--base_dir", type=Path, default=None,
                        help="Base directory containing eval/baseline dirs (default: project root)")
    args = parser.parse_args()

    search_dir = args.base_dir if args.base_dir else BASE_DIR
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stacked_data = []  # collect (bl_hist, rl_hist, case_num) for stacked figure

    for case_num in args.cases:
        print(f"\n{'='*60}")
        print(f"Case {case_num}")
        print(f"{'='*60}")

        # Find directories
        rl_dir = find_case_dir(search_dir, case_num, "policy_eval")
        bl_dir = find_case_dir(search_dir, case_num, "corall_baseline_case")

        if not rl_dir:
            print(f"  ERROR: No RL eval directory found for case {case_num}")
            continue
        if not bl_dir:
            print(f"  ERROR: No baseline directory found for case {case_num}")
            continue

        print(f"  RL dir:       {rl_dir.name}")
        print(f"  Baseline dir: {bl_dir.name}")

        # Find best RL episode
        rl_file = find_best_episode_file(rl_dir)
        if not rl_file:
            print(f"  ERROR: No best RL episode found")
            continue

        # Extract seed from RL filename for matching
        seed_match = re.search(r"_seed(\d+)", rl_file.stem)
        rl_seed = int(seed_match.group(1)) if seed_match else 0

        # Find matching baseline episode
        bl_file = find_matching_baseline_file(bl_dir, rl_seed)
        if not bl_file:
            print(f"  ERROR: No baseline episode found")
            continue

        print(f"  RL episode:       {rl_file.name}")
        print(f"  Baseline episode: {bl_file.name}")

        # Load histories
        rl_hist = load_npz_as_dict(rl_file)
        bl_hist = load_npz_as_dict(bl_file)

        n_agents = np.asarray(bl_hist["X_all"]).shape[1]
        print(f"  Agents: {n_agents} total ships")

        # Generate full trajectory overlay
        full_path = args.output_dir / f"case{case_num}_full_trajectory.png"
        print(f"  Generating full trajectory overlay...")
        plot_full_trajectory_overlay(bl_hist, rl_hist, full_path)
        print(f"    Saved: {full_path.name}")

        # Generate encounter detail
        encounter_path = args.output_dir / f"case{case_num}_encounter_detail.png"
        print(f"  Generating encounter detail...")
        plot_encounter_detail_clean(bl_hist, rl_hist, encounter_path)
        print(f"    Saved: {encounter_path.name}")

        # Generate threat profile (CPA panel)
        threat_path = args.output_dir / f"case{case_num}_threat_profile.png"
        print(f"  Generating threat profile...")
        plot_ownship_threat_profile(bl_hist, rl_hist, threat_path)
        print(f"    Saved: {threat_path.name}")

        # Collect data for stacked figure
        stacked_data.append((bl_hist, rl_hist, case_num))

    # Generate stacked trajectory figure if we have multiple cases
    if len(stacked_data) >= 2:
        stacked_path = args.output_dir / "trajectory_comparison_stacked.png"
        print(f"\nGenerating stacked trajectory figure ({len(stacked_data)} cases)...")
        plot_stacked_trajectory_overlay(stacked_data, stacked_path)
        print(f"  Saved: {stacked_path.name}")

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
