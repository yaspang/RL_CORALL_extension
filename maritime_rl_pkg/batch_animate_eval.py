"""
Batch animate all episodes from a policy evaluation directory.

Usage:
python -m maritime_rl_pkg.batch_animate_eval \
  --eval_dir "C:\path\to\policy_eval_case6_...\seed_0" \
  --output_dir "C:\path\to\policy_eval_case6_...\seed_0\animations" \
  --case 6 --ship_scale 2.0 --fps 20 --stride 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .path_setup import ensure_paths
ensure_paths()

from .episode_overlay_tools import load_episode_history
from visualization.rendering import animate_ship

NMI = 1852.0

def find_episode_file(hist_dir: Path, episode_index: int, episode_seed: int | None = None) -> Path | None:
    """
    Map evaluation episode_index / episode_seed to the correct saved history file.

    Expected filename pattern:
        case{case}_seed{seed}_ep{episode_index:03d}.npz
    """
    candidates = sorted(hist_dir.glob("*.npz"))

    ep_token = f"_ep{episode_index:03d}"
    matches = [p for p in candidates if ep_token in p.stem]

    if episode_seed is not None:
        seed_token = f"_seed{episode_seed}"
        seeded = [p for p in matches if seed_token in p.stem]
        if seeded:
            return seeded[0]

    if matches:
        return matches[0]

    return None

def compute_bounds_nmi(X_all: np.ndarray, pad_nmi: float = 0.8):
    """Compute plot bounds from episode trajectory."""
    x_all = X_all[:, :, 0] / NMI
    y_all = X_all[:, :, 1] / NMI

    xmin = float(np.min(x_all))
    xmax = float(np.max(x_all))
    ymin = float(np.min(y_all))
    ymax = float(np.max(y_all))

    if abs(xmax - xmin) < 1.0:
        xmin -= 0.5
        xmax += 0.5
    if abs(ymax - ymin) < 1.0:
        ymin -= 0.5
        ymax += 0.5

    return xmin - pad_nmi, xmax + pad_nmi, ymin - pad_nmi, ymax + pad_nmi


def animate_single_episode(
    history_path: Path,
    output_path: Path,
    case: int | None = None, 
    ship_scale: float = 6.0,
    fps: int = 20,
    stride: int = 4,
):
    """Animate a single episode and save to GIF (aligned with animate_episode_history.py style)."""
    print(f"  Loading: {history_path.name}")
    
    # Load episode history
    history = load_episode_history(str(history_path))
    t = np.asarray(history.get("t", []), dtype=float)
    X_all = np.asarray(history["X_all"], dtype=float)
    pair_dist = np.asarray(history.get("pair_dist", []), dtype=float)
    
    n_steps, n_agents, _ = X_all.shape
    own_idx = 0  # Ownship is always agent 0
    
    case_num = history.get("case", case or "?")
    seed_num = history.get("seed", "?")
    checkpoint = history.get("checkpoint", "trained policy")
    final_waypoint_x_nmi = history.get("final_waypoint_x_nmi")
    final_waypoint_y_nmi = history.get("final_waypoint_y_nmi")
    
    # Compute plot bounds
    xmin, xmax, ymin, ymax = compute_bounds_nmi(X_all, pad_nmi=0.8)
    
    # Trail length for trajectory visualization
    trail_seconds = 120.0
    trail_steps = max(2, int(trail_seconds / max(1e-9, (t[1] - t[0])))) if len(t) > 1 else 50
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 8))
    
    def draw_frame(frame_idx):
        ax.clear()
        artists = []
        
        s = frame_idx * stride
        s = min(s, n_steps - 1)
        i0 = max(0, s - trail_steps)
        
        # Set plot properties
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("X (nmi)", fontsize=11)
        ax.set_ylabel("Y (nmi)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        
        # TARGET SHIPS: dotted blue trails
        for j in range(1, n_agents):
            xj = X_all[i0:s + 1, j, 0] / NMI
            yj = X_all[i0:s + 1, j, 1] / NMI
            line, = ax.plot(xj, yj, color="tab:blue", linewidth=2.0, alpha=0.7, 
                           linestyle=":", label="Target ship trail" if j == 1 else "")
            artists.append(line)
            
            # Render target ship icon
            x_now = X_all[s, j, 0] / NMI
            y_now = X_all[s, j, 1] / NMI
            psi_now = float(X_all[s, j, 2])
            
            ship_artists = animate_ship(
                x_now, y_now, psi_now,
                (30.0 / NMI) * ship_scale,
                (16.0 / NMI) * ship_scale,
                cpa=0.0,
                color="tab:blue",
                ax=ax,
            )
            if ship_artists is not None:
                if isinstance(ship_artists, list):
                    artists.extend(ship_artists)
                else:
                    artists.append(ship_artists)
        
        # OWNSHIP: dashed purple trail
        xo = X_all[i0:s + 1, own_idx, 0] / NMI
        yo = X_all[i0:s + 1, own_idx, 1] / NMI
        line, = ax.plot(xo, yo, color="#ff7f0e", linewidth=2.8, alpha=0.85, 
                       linestyle="--", label="RL ownship trail")
        artists.append(line)
        
        # Render ownship icon
        x_now = X_all[s, own_idx, 0] / NMI
        y_now = X_all[s, own_idx, 1] / NMI
        psi_now = float(X_all[s, own_idx, 2])
        
        ship_artists = animate_ship(
            x_now, y_now, psi_now,
            (30.0 / NMI) * ship_scale,
            (16.0 / NMI) * ship_scale,
            cpa=0.0,
            color="#ff7f0e",
            ax=ax,
        )
        if ship_artists is not None:
            if isinstance(ship_artists, list):
                artists.extend(ship_artists)
            else:
                artists.append(ship_artists)
        
        # Start marker
        scatter1 = ax.scatter(
            X_all[0, own_idx, 0] / NMI,
            X_all[0, own_idx, 1] / NMI,
            s=80,
            color="orange",
            zorder=5,
            label="Start",
        )
        artists.append(scatter1)
        
        # Plot final waypoint target if available
        if final_waypoint_x_nmi is not None and final_waypoint_y_nmi is not None:
            # Plot waypoint center marker only
            scatter_wp = ax.scatter(
                final_waypoint_x_nmi,
                final_waypoint_y_nmi,
                s=150,
                marker="X",
                color="darkgreen",
                zorder=6,
            )
            artists.append(scatter_wp)
        
        # End marker (no legend label)
        scatter2 = ax.scatter(
            X_all[-1, own_idx, 0] / NMI,
            X_all[-1, own_idx, 1] / NMI,
            s=70,
            marker="o",
            color="#ff7f0e",
            alpha=0.6,
            edgecolors="black",
            linewidth=1.0,
            zorder=5,
        )
        artists.append(scatter2)
        
        # Current min separation
        if len(pair_dist) > s and own_idx < len(pair_dist[s]):
            drow = pair_dist[s, own_idx].copy()
            drow[own_idx] = np.inf
            min_sep_m = float(np.min(drow)) if np.any(np.isfinite(drow)) else np.nan
        else:
            min_sep_m = np.nan
        
        # Title
        title = f"Case {case_num} | Seed {seed_num} | t = {t[s] if len(t) > s else 0:.1f} s"
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        # Info box
        txt = (
            f"Episode: {history_path.stem}\n"
            f"Min separation: {min_sep_m / NMI:.2f} nmi ({min_sep_m:.0f} m)"
        )
        text_obj = ax.text(
            0.02, 0.98, txt,
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )
        artists.append(text_obj)
        
        # Legend
        legend = ax.legend(loc="best")
        if legend is not None:
            artists.append(legend)
        
        return artists
    
    n_frames = int(np.ceil(n_steps / stride))
    anim = FuncAnimation(fig, draw_frame, frames=n_frames, interval=1000 / fps, repeat=False)
    
    # Save GIF
    print(f"  Saving: {output_path.name}")
    writer = PillowWriter(fps=fps)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)


def batch_animate_episodes(
    eval_dir: str,
    output_dir: str,
    case: Optional[int] = None,
    ship_scale: float = 2.0,
    fps: int = 20,
    stride: int = 4,
):
    """Animate the best and worst episodes from an evaluation directory."""
    eval_path = Path(eval_dir)
    hist_dir = eval_path / "episode_histories"
    summary_file = eval_path / "policy_eval_summary.json"

    # If files not at root level, search for seed_* subdirectories
    if not hist_dir.exists() or not summary_file.exists():
        seed_dirs = sorted([p for p in eval_path.glob("seed_*") if p.is_dir()])
        found = False
        for seed_dir in seed_dirs:
            hist_dir_alt = seed_dir / "episode_histories"
            summary_file_alt = seed_dir / "policy_eval_summary.json"
            if hist_dir_alt.exists() and summary_file_alt.exists():
                print(f"Found files in seed subdirectory, using: {seed_dir}")
                hist_dir = hist_dir_alt
                summary_file = summary_file_alt
                eval_path = seed_dir
                found = True
                break
        if not found:
            print(f"ERROR: could not find valid seed_* evaluation directory under {eval_path}")
            return

    if not hist_dir.exists():
        print(f"ERROR: episode_histories directory not found at {hist_dir}")
        return

    if not summary_file.exists():
        print(f"ERROR: policy_eval_summary.json not found at {summary_file}")
        return

    with open(summary_file, "r") as f:
        summary = json.load(f)

    csv_file = eval_path / "policy_eval_per_episode.csv"
    per_episode_metrics = []

    if csv_file.exists():
        import csv
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # support both RL and baseline CSV naming
                    if "episode_return" in row:
                        ep_return = float(row["episode_return"])
                    elif "episode_return_ownship" in row:
                        ep_return = float(row["episode_return_ownship"])
                    else:
                        ep_return = 0.0

                    per_episode_metrics.append({
                        "episode_index": int(row.get("episode_index", 0)),
                        "episode_seed": int(row.get("episode_seed", 0)),
                        "episode_return": ep_return,
                    })
                except (ValueError, KeyError):
                    pass

    # Prefer CSV because it carries both episode_index and episode_seed explicitly
    if per_episode_metrics:
        best_row = max(per_episode_metrics, key=lambda r: r["episode_return"])
        worst_row = min(per_episode_metrics, key=lambda r: r["episode_return"])

        best_idx = int(best_row["episode_index"])
        best_seed = int(best_row["episode_seed"])
        best_return = float(best_row["episode_return"])

        worst_idx = int(worst_row["episode_index"])
        worst_seed = int(worst_row["episode_seed"])
        worst_return = float(worst_row["episode_return"])
    else:
        # Fallback to summary-only behavior
        best_idx = int(summary.get("best_return_episode_idx", 0))
        best_seed = int(summary.get("best_return_episode_seed", 0))
        best_return = float(summary.get("best_return_value", 0.0))

        # Without CSV, worst is harder to identify robustly
        worst_idx = 0
        worst_seed = 0
        worst_return = 0.0

    episode_files = sorted(hist_dir.glob("*.npz"))
    print(f"Found {len(episode_files)} episodes total")

    print(f"\nEpisode Return Range:")
    print(f"  Best:  Episode {best_idx:03d} with return = {best_return:10.3f}")
    print(f"  Worst: Episode {worst_idx:03d} with return = {worst_return:10.3f}")
    print(f"  Delta: {best_return - worst_return:10.3f}\n")

    best_file = find_episode_file(hist_dir, best_idx, best_seed)
    worst_file = find_episode_file(hist_dir, worst_idx, worst_seed)

    if best_file is None:
        print(f"ERROR: Could not find history file for best episode idx={best_idx}, seed={best_seed}")
        return
    if worst_file is None:
        print(f"ERROR: Could not find history file for worst episode idx={worst_idx}, seed={worst_seed}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Animate best
    best_gif = output_path / f"{best_file.stem}_BEST.gif"
    print(f"Animating best episode: {best_file.stem}")
    try:
        animate_single_episode(
            best_file,
            best_gif,
            case=case,
            ship_scale=ship_scale,
            fps=fps,
            stride=stride,
        )
        print(f"  ✓ Saved: {best_gif.name}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Animate worst
    worst_gif = output_path / f"{worst_file.stem}_WORST.gif"
    print(f"\nAnimating worst episode: {worst_file.stem}")
    try:
        animate_single_episode(
            worst_file,
            worst_gif,
            case=case,
            ship_scale=ship_scale,
            fps=fps,
            stride=stride,
        )
        print(f"  ✓ Saved: {worst_gif.name}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print(f"\nAnimations saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch animate evaluation episodes to GIFs"
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Path to policy_eval output directory (with episode_histories subdirectory)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for GIFs. Defaults to eval_dir/animations",
    )
    parser.add_argument(
        "--case",
        type=int,
        default=None,
        help="Case number (for reference)",
    )
    parser.add_argument(
        "--ship_scale",
        type=float,
        default=2.0,
        help="Scale factor for ship rendering",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for GIF",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Stride for downsampling frames",
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or str(Path(args.eval_dir) / "animations")
    
    batch_animate_episodes(
        args.eval_dir,
        output_dir,
        case=args.case,
        ship_scale=args.ship_scale,
        fps=args.fps,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
