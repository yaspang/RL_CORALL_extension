"""
Animate episode history from eval_trained_policy_with_hist.py output.
"""
#python -m maritime_rl_pkg.maritime_rl.animate_episode_history ^
  #--history "C:\path\to\policy_eval_case8_...\seed_0\episode_histories\trained_case8_seed0_ep000.json.npz" ^
  #--output "C:\path\to\policy_eval_case8_...\seed_0\episode_replay.gif"


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .path_setup import ensure_paths
ensure_paths()

from .episode_overlay_tools import load_episode_history
from visualization.rendering import animate_ship

NMI = 1852.0


def compute_bounds_nmi(X_all: np.ndarray, pad_nmi: float = 0.8):
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


def animate_history(
    history_path: str | Path,
    output_path: str | Path,
    trail_seconds: float = 120.0,
    ship_scale: float = 6.0,
    fps: int = 10,
    stride: int = 2,
):
    hist = load_episode_history(history_path)

    t = np.asarray(hist["t"], dtype=float)
    X_all = np.asarray(hist["X_all"], dtype=float)
    pair_dist = np.asarray(hist["pair_dist"], dtype=float)

    case = hist.get("case", "?")
    seed = hist.get("seed", "?")
    checkpoint = hist.get("checkpoint", "")

    n_steps, n_agents, _ = X_all.shape
    own_idx = 0

    xmin, xmax, ymin, ymax = compute_bounds_nmi(X_all)

    fig, ax = plt.subplots(figsize=(11, 8))

    trail_steps = max(2, int(trail_seconds / max(1e-9, (t[1] - t[0])))) if len(t) > 1 else 50

    def draw_frame(frame_idx: int):
        ax.clear()
        artists = []

        s = frame_idx * stride
        s = min(s, n_steps - 1)

        i0 = max(0, s - trail_steps)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("X (nmi)")
        ax.set_ylabel("Y (nmi)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        # target ships
        for j in range(1, n_agents):
            xj = X_all[i0:s + 1, j, 0] / NMI
            yj = X_all[i0:s + 1, j, 1] / NMI
            line, = ax.plot(xj, yj, color="tab:blue", linewidth=1.8, alpha=0.8)
            artists.append(line)

            x_now = X_all[s, j, 0] / NMI
            y_now = X_all[s, j, 1] / NMI
            psi_now = float(X_all[s, j, 2])

            ship_artists = animate_ship(
                x_now,
                y_now,
                psi_now,
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

        # ownship trail
        xo = X_all[i0:s + 1, own_idx, 0] / NMI
        yo = X_all[i0:s + 1, own_idx, 1] / NMI
        line, = ax.plot(xo, yo, color="purple", linewidth=2.4, alpha=0.95, label="RL ownship")
        artists.append(line)

        x_now = X_all[s, own_idx, 0] / NMI
        y_now = X_all[s, own_idx, 1] / NMI
        psi_now = float(X_all[s, own_idx, 2])

        ship_artists = animate_ship(
            x_now,
            y_now,
            psi_now,
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

        # start marker
        scatter1 = ax.scatter(
            X_all[0, own_idx, 0] / NMI,
            X_all[0, own_idx, 1] / NMI,
            s=80,
            color="orange",
            zorder=5,
            label="Start",
        )
        artists.append(scatter1)

        # goal marker from ownship final waypoint proxy: final ownship route end
        scatter2 = ax.scatter(
            X_all[-1, own_idx, 0] / NMI,
            X_all[-1, own_idx, 1] / NMI,
            s=90,
            marker="*",
            color="green",
            zorder=5,
            label="End",
        )
        artists.append(scatter2)

        # current min separation
        drow = pair_dist[s, own_idx].copy()
        if own_idx < len(drow):
            drow[own_idx] = np.inf
        min_sep_m = float(np.min(drow)) if np.any(np.isfinite(drow)) else np.nan

        title = f"Case {case} | seed {seed} | t = {t[s]:.1f} s"
        ax.set_title(title, fontsize=14, fontweight="bold")

        txt = (
            f"Checkpoint: {Path(checkpoint).name if checkpoint else 'trained policy'}\n"
            f"Min sep now: {min_sep_m / NMI:.2f} nmi"
        )
        text_obj = ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )
        artists.append(text_obj)

        legend = ax.legend(loc="best")
        if legend is not None:
            artists.append(legend)

        return artists

    n_frames = int(np.ceil(n_steps / stride))
    anim = FuncAnimation(fig, draw_frame, frames=n_frames, interval=1000 / fps, repeat=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".gif":
        anim.save(output_path, writer=PillowWriter(fps=fps))
    else:
        anim.save(output_path, fps=fps)

    plt.close(fig)
    return output_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--trail_seconds", type=float, default=120.0)
    p.add_argument("--ship_scale", type=float, default=6.0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--stride", type=int, default=2)
    args = p.parse_args()

    out = animate_history(
        history_path=args.history,
        output_path=args.output,
        trail_seconds=args.trail_seconds,
        ship_scale=args.ship_scale,
        fps=args.fps,
        stride=args.stride,
    )
    print(f"Saved animation to: {out}")


if __name__ == "__main__":
    main()