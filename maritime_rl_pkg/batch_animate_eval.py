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
        line, = ax.plot(xo, yo, color="purple", linewidth=2.8, alpha=0.85, 
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
            color="purple",
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
            color="purple",
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
    """Animate ONLY the best episode from an evaluation directory (highest return)."""
    eval_path = Path(eval_dir)
    hist_dir = eval_path / "episode_histories"
    summary_file = eval_path / "policy_eval_summary.json"
    
    # If files not at root level, check seed_0 subdirectory (baseline eval structure)
    if not hist_dir.exists() or not summary_file.exists():
        seed_0_path = eval_path / "seed_0"
        hist_dir_alt = seed_0_path / "episode_histories"
        summary_file_alt = seed_0_path / "policy_eval_summary.json"
        
        if hist_dir_alt.exists() and summary_file_alt.exists():
            print(f"Found files in seed_0 subdirectory, using: {seed_0_path}")
            hist_dir = hist_dir_alt
            summary_file = summary_file_alt
    
    if not hist_dir.exists():
        print(f"ERROR: episode_histories directory not found at {hist_dir}")
        return
    
    # Load evaluation summary to find best episode
    if not summary_file.exists():
        print(f"ERROR: policy_eval_summary.json not found at {summary_file}")
        return
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # First, try to get best episode from summary fields (SB3 eval format)
    best_idx = summary.get("best_return_episode_idx")
    best_return = summary.get("best_return_value")
    
    # If not found in summary, try to get per_episode_metrics from summary (policy eval format)
    if best_idx is None or best_return is None:
        per_episode_metrics = summary.get("per_episode_metrics", [])
        
        # If not found, try to load from CSV file (baseline eval format)
        if not per_episode_metrics:
            csv_file = eval_path.parent / "policy_eval_per_episode.csv" if "seed_0" in str(eval_path) else eval_path / "policy_eval_per_episode.csv"
            # Check seed_0 subdirectory if needed
            if not csv_file.exists():
                csv_file = eval_path / "seed_0" / "policy_eval_per_episode.csv"
            
            if csv_file.exists():
                import csv
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            per_episode_metrics.append({
                                "episode_index": int(row.get("episode_index", 0)),
                                "episode_return": float(row.get("episode_return_ownship", 0.0))
                            })
                        except (ValueError, KeyError):
                            pass
        
        if not per_episode_metrics:
            print("ERROR: No best_return_episode_idx in summary, no per_episode_metrics, and no CSV file found")
            return
        
        # Find episode with best return
        best_idx = 0
        best_return = per_episode_metrics[0]["episode_return"]
        for i, metrics in enumerate(per_episode_metrics):
            if metrics["episode_return"] > best_return:
                best_return = metrics["episode_return"]
                best_idx = i
        
        print(f"Found {len(per_episode_metrics)} episodes total")
    else:
        # Count total episodes from files
        episode_files = sorted(hist_dir.glob("*.npz"))
        print(f"Found {len(episode_files)} episodes total")
    
    print(f"Best episode: Episode {best_idx:03d} with return = {best_return:.3f}")
    
    # Find corresponding npz file for best episode
    episode_files = sorted(hist_dir.glob("*.npz"))
    if best_idx >= len(episode_files):
        print(f"ERROR: Best episode index {best_idx} exceeds number of files {len(episode_files)}")
        return
    
    ep_file = episode_files[best_idx]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Animate only the best episode
    output_gif = output_path / f"{ep_file.stem}_BEST.gif"
    print(f"Animating best episode: {ep_file.stem}")
    
    try:
        animate_single_episode(
            ep_file,
            output_gif,
            case=case,
            ship_scale=ship_scale,
            fps=fps,
            stride=stride,
        )
        print(f"✓ Best episode animation saved to: {output_gif.name}")
    except Exception as e:
        print(f"✗ Error animating best episode: {e}")
    
    print(f"\nAnimation saved to: {output_path}")


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
