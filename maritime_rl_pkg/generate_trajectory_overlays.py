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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from maritime_rl_pkg.path_setup import ensure_paths
ensure_paths()

from maritime_rl_pkg.episode_overlay_tools import (
    animate_ship,
    plot_full_trajectory_overlay,
    plot_encounter_detail_clean,
    plot_ownship_threat_profile,
    plot_stacked_trajectory_overlay,
    to_numpy_history,
)

BASE_DIR = Path(__file__).parent
NMI = 1852.0


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

    if not hist_dir.exists():
        return None

    npz_files = sorted(hist_dir.glob("*.npz"))
    if not npz_files:
        return None

    # If CSV is unavailable, fall back to first available NPZ.
    if not csv_file.exists():
        return npz_files[0]

    with open(csv_file) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return npz_files[0]

    # Parse returns
    episodes = []
    for row in rows:
        try:
            ret = float(row.get("episode_return") or row.get("episode_return_ownship", "0"))
            idx = int(row.get("episode_index", 0))
            seed = int(row.get("episode_seed", 0))
            episodes.append((ret, idx, seed))
        except Exception:
            continue

    if not episodes:
        return npz_files[0]

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

    # Final fallback to first available history.
    return npz_files[0]


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


def find_rl_eval_dir(base: Path, case_num: int, rl_step: int | None = None) -> Path | None:
    """
    Find an RL evaluation directory for a case.

    Looks for directories containing case token and a seed_0/episode_histories
    subdirectory. Prefers names containing 'eval_cp' or 'policy_eval'.
    """
    case_re = re.compile(rf"case{case_num}(?:\D|$)")

    # First pass: direct children (fast path)
    direct_candidates = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        if not case_re.search(d.name):
            continue
        if (d / "seed_0" / "episode_histories").exists():
            direct_candidates.append(d)

    # Second pass: recursive fallback
    if not direct_candidates:
        recursive_candidates = []
        for d in sorted(base.rglob("*")):
            if not d.is_dir():
                continue
            if not case_re.search(d.name):
                continue
            if (d / "seed_0" / "episode_histories").exists():
                recursive_candidates.append(d)
        direct_candidates = recursive_candidates

    if not direct_candidates:
        return None

    # Optional step filtering for eval_cp<step>_case* naming.
    if rl_step is not None:
        step_pat = re.compile(r"eval_cp(\d+)_")
        step_filtered = []
        for d in direct_candidates:
            m = step_pat.search(d.name)
            if m and int(m.group(1)) == rl_step:
                step_filtered.append(d)
        direct_candidates = step_filtered
        if not direct_candidates:
            return None

    # Prefer eval-style directory names.
    preferred = [d for d in direct_candidates if ("eval_cp" in d.name or "policy_eval" in d.name)]
    return preferred[0] if preferred else direct_candidates[0]


def plot_rl_trajectories_grid(
    rl_base_dir: Path,
    case_numbers: list[int],
    output_path: Path,
    rl_step: int | None = None,
) -> tuple[int, int]:
    """
    Create a single large multi-panel figure with RL trajectories for each case.

    Each case panel shows ownship (OS) and target ship trajectories (TS1..TS3).
    This mode does not require baseline histories.
    """
    cases = sorted(case_numbers)
    n_cols = 4
    n_rows = int(np.ceil(len(cases) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.2, n_rows * 3.2))
    axes = np.array(axes).reshape(n_rows, n_cols)

    # RL-only style: ownship blue, all targets grey.
    os_color = "#1f77b4"
    ts_color = "#b0b7c3"

    n_found = 0
    for idx, case_num in enumerate(cases):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]

        rl_dir = find_rl_eval_dir(rl_base_dir, case_num, rl_step=rl_step)
        if rl_dir is None:
            ax.set_title(f"Case {case_num}\n(not found)", fontsize=10)
            ax.axis("off")
            continue

        rl_file = find_best_episode_file(rl_dir)
        if rl_file is None:
            ax.set_title(f"Case {case_num}\n(no history)", fontsize=10)
            ax.axis("off")
            continue

        hist = load_npz_as_dict(rl_file)
        X_all = np.asarray(hist["X_all"], dtype=float)
        n_agents = X_all.shape[1]

        # Ownship trajectory (agent 0)
        x0 = X_all[:, 0, 0] / NMI
        y0 = X_all[:, 0, 1] / NMI
        ax.plot(x0, y0, color=os_color, lw=1.5)

        # Target trajectories
        for j in range(1, n_agents):
            xj = X_all[:, j, 0] / NMI
            yj = X_all[:, j, 1] / NMI
            ax.plot(xj, yj, color=ts_color, lw=1.3)

        ax.set_title(f"Case {case_num}", fontsize=10)
        ax.set_xlabel("X (nmi)", fontsize=8)
        ax.set_ylabel("Y (nmi)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        n_found += 1

    # Turn off unused axes in the final row.
    for idx in range(len(cases), n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis("off")

    legend_handles = [
        Line2D([0], [0], color=ts_color, lw=2, label="Target ship(s)"),
        Line2D([0], [0], color=os_color, lw=2, label="RL ownship"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.02), fontsize=11)
    fig.suptitle("RL Ownship + Target Trajectories Across Imazu Cases", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.98])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return n_found, len(cases)


def plot_stacked_rl_only_trajectory_overlay(
    case_data: list[tuple[dict, int]],
    save_path: Path,
) -> Path:
    """
    Stacked RL-only trajectory overlay with one subplot per case.

    Ownship is blue, all target ships are grey. This is intended as a more
    readable companion figure to the baseline-vs-RL stacked comparison.
    """
    import matplotlib.gridspec as gridspec

    n_cases = len(case_data)
    n_cols = 4
    n_rows = int(np.ceil(n_cases / n_cols))
    fig = plt.figure(figsize=(n_cols * 5.0, n_rows * 4.35))
    outer_gs = gridspec.GridSpec(
        n_rows, n_cols,
        left=0.03, right=0.995, top=0.84, bottom=0.055,
        hspace=0.62, wspace=0.28,
        figure=fig,
    )

    ownship_color = "#ff7f0e"
    target_color = "#b9c0ca"
    start_color = "gold"
    goal_color = "darkgreen"
    ship_icon_interval_s = 60.0
    ship_scale = 1.08
    loa_m = 30.0
    beam_m = 16.0

    for row, (rl_hist, case_num) in enumerate(case_data):
        r = row // n_cols
        c = row % n_cols
        ax = fig.add_subplot(outer_gs[r, c])
        _, Xr, _, _, _, _ = to_numpy_history(rl_hist)
        n_agents = Xr.shape[1]

        # Target trajectories, kept grey and low-emphasis.
        for j in range(1, n_agents):
            lbl = "Target ship(s)" if (row == 0 and j == 1) else None
            ax.plot(
                Xr[:, j, 0],
                Xr[:, j, 1],
                "-",
                linewidth=1.15,
                alpha=0.55,
                color=target_color,
                label=lbl,
                zorder=1,
            )

        # Ownship track.
        xr0, yr0 = Xr[:, 0, 0], Xr[:, 0, 1]
        lbl_rl = "RL policy" if row == 0 else None
        ax.plot(xr0, yr0, "-", color=ownship_color, linewidth=1.8, label=lbl_rl, zorder=3)

        # Ship icons along the trajectories for visual alignment with the comparison figure.
        max_time = max(1.0, float(Xr.shape[0] - 1))
        icon_times = np.arange(ship_icon_interval_s, max_time, ship_icon_interval_s)
        for t_icon in icon_times:
            idx_icon = int(np.clip(round(t_icon), 0, Xr.shape[0] - 1))
            for j in range(1, n_agents):
                p = animate_ship(
                    float(Xr[idx_icon, j, 0]),
                    float(Xr[idx_icon, j, 1]),
                    float(Xr[idx_icon, j, 2]),
                    loa_m * ship_scale,
                    beam_m * ship_scale,
                    cpa=0.0,
                    color=target_color,
                    ax=ax,
                )
                p.set_alpha(0.18)
            animate_ship(
                float(Xr[idx_icon, 0, 0]),
                float(Xr[idx_icon, 0, 1]),
                float(Xr[idx_icon, 0, 2]),
                loa_m * ship_scale,
                beam_m * ship_scale,
                cpa=0.0,
                color=ownship_color,
                ax=ax,
            )

        # Start / goal markers.
        lbl_start = "Ownship start" if row == 0 else None
        lbl_goal = "Goal" if row == 0 else None
        ax.scatter(xr0[0], yr0[0], s=40, color=start_color, label=lbl_start,
                   zorder=6, edgecolors="black", linewidth=0.5)
        ax.scatter(xr0[-1], yr0[-1], marker="X", s=50, color=goal_color,
                   zorder=6, label=lbl_goal)

        # Case label centered above each panel, like the compare_case_metrics grid.
        quad_bbox = outer_gs[r, c].get_position(fig)
        case_x = (quad_bbox.x0 + quad_bbox.x1) * 0.5
        case_y = quad_bbox.y1 + 0.01
        fig.text(
            case_x,
            case_y,
            f"Case {case_num}",
            ha="center", va="bottom", fontsize=22, fontweight="bold"
        )
        fig.text(
            case_x,
            case_y - 0.020,
            f"({n_agents} ships)",
            ha="center", va="bottom", fontsize=11, fontweight="normal"
        )

        ax.set_ylabel("Y position (m)", fontsize=13, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=10)

        # Bounds fit the trajectories tightly but preserve the subplot aspect.
        all_x = [Xr[:, j, 0] for j in range(n_agents)]
        all_y = [Xr[:, j, 1] for j in range(n_agents)]
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)

        xpad = max(150, 0.08 * max(1.0, np.ptp(all_x)))
        ypad = max(150, 0.12 * max(1.0, np.ptp(all_y)))
        xmin, xmax = float(np.min(all_x) - xpad), float(np.max(all_x) + xpad)
        ymin, ymax = float(np.min(all_y) - ypad), float(np.max(all_y) + ypad)

        # Preserve the visual balance of each panel inside the grid.
        ax_aspect = 4.35 / 5.0
        x_range = xmax - xmin
        y_range = ymax - ymin
        if y_range / x_range < ax_aspect:
            needed = ax_aspect * x_range
            mid_y = (ymin + ymax) / 2.0
            ymin, ymax = mid_y - needed / 2.0, mid_y + needed / 2.0
        else:
            needed = y_range / ax_aspect
            mid_x = (xmin + xmax) / 2.0
            xmin, xmax = mid_x - needed / 2.0, mid_x + needed / 2.0

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        if r == n_rows - 1:
            ax.set_xlabel("X position (m)", fontsize=13, fontweight="bold")

    handles = [
        Line2D([0], [0], color=target_color, lw=1.5, alpha=0.8, label="Target ship(s)"),
        Line2D([0], [0], color=ownship_color, lw=1.8, label="RL policy"),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=start_color,
               markeredgecolor='black', markersize=6, label="Ownship start"),
        Line2D([0], [0], marker='X', color='none', markerfacecolor=goal_color,
               markeredgecolor=goal_color, markersize=7, label="Goal"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4,
               fontsize=13, framealpha=0.95, bbox_to_anchor=(0.5, 0.965),
               prop={'weight': 'bold'}, edgecolor='black', fancybox=True)

    fig.suptitle(
        "RL Ownship Trajectories with Target Ships Across All 22 Imazu Cases",
        y=0.995, fontsize=44, fontweight="bold"
    )

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
    plt.close(fig)
    return Path(save_path)


def main():
    parser = argparse.ArgumentParser(description="Generate trajectory overlay plots")
    parser.add_argument("--cases", nargs="+", type=int, default=[1, 10, 18],
                        help="Case numbers to generate overlays for")
    parser.add_argument("--output_dir", type=Path, default=Path("Visualizations/trajectory_overlays"),
                        help="Output directory for plots")
    parser.add_argument("--base_dir", type=Path, default=Path('.'),
                        help="Base directory containing eval/baseline dirs (default: project root)")
    parser.add_argument("--rl_dir", type=Path, default=None,
                        help="Directory containing RL eval case folders (defaults to --base_dir)")
    parser.add_argument("--rl_step", type=int, default=None,
                        help="Optional RL step filter for eval_cp<step>_case* folders")
    parser.add_argument("--rl_only_grid", action="store_true",
                        help="Generate one large RL-only multi-case trajectory figure (no baseline required)")
    parser.add_argument("--grid_output_name", type=str, default="rl_all_cases_trajectory_grid.png",
                        help="Output filename for --rl_only_grid mode")
    args = parser.parse_args()

    search_dir = args.base_dir if args.base_dir else Path('.')  # Use current directory as default
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.rl_only_grid:
        rl_search_dir = args.rl_dir if args.rl_dir is not None else search_dir
        output_path = args.output_dir / args.grid_output_name
        print(f"Generating RL-only trajectory grid from: {rl_search_dir}")
        n_found, n_total = plot_rl_trajectories_grid(
            rl_search_dir,
            args.cases,
            output_path,
            rl_step=args.rl_step,
        )
        print(f"Saved: {output_path}")
        print(f"Cases plotted: {n_found}/{n_total}")
        return

    stacked_data = []  # collect (bl_hist, rl_hist, case_num) for stacked figure
    rl_only_data = []  # collect (rl_hist, case_num) for RL-only stacked figure

    for case_num in args.cases:
        print(f"\n{'='*60}")
        print(f"Case {case_num}")
        print(f"{'='*60}")

        # Find directories
        rl_search_dir = args.rl_dir if args.rl_dir is not None else search_dir
        rl_dir = find_rl_eval_dir(rl_search_dir, case_num, rl_step=args.rl_step)
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
        rl_only_data.append((rl_hist, case_num))

    # Generate stacked trajectory figure if we have multiple cases
    if len(stacked_data) >= 2:
        stacked_path = args.output_dir / "trajectory_comparison_stacked.png"
        print(f"\nGenerating stacked trajectory figure ({len(stacked_data)} cases)...")
        plot_stacked_trajectory_overlay(stacked_data, stacked_path)
        print(f"  Saved: {stacked_path.name}")

    if rl_only_data:
        rl_only_path = args.output_dir / "trajectory_rl_only_stacked.png"
        print(f"\nGenerating RL-only stacked trajectory figure ({len(rl_only_data)} cases)...")
        plot_stacked_rl_only_trajectory_overlay(rl_only_data, rl_only_path)
        print(f"  Saved: {rl_only_path.name}")

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()