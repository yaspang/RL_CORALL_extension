import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from matplotlib import scale
import matplotlib.pyplot as plt
import numpy as np

# ensure CORALL repository relative imports resolve
from .path_setup import ensure_paths
ensure_paths()

# local rendering helper from CORALL
from visualization.rendering import animate_ship

NMI = 1852.0

def load_ep_history(path: str | Path) -> dict: 
    """ Load episode history from JSON file saved by `save_episode_history`"""
    with open(path, "r") as f: 
        return json.load(f)

def to_numpy_history(hist: dict): 
    t = np.asarray(hist["t"], dtype=float)
    X_all = np.asarray(hist["X_all"], dtype=float)
    pair_risk = np.asarray(hist["pair_risk"], dtype=float)
    pair_dcpa = np.asarray(hist["pair_dcpa"], dtype=float)
    pair_dist = np.asarray(hist["pair_dist"], dtype=float)
    pair_tcpa = np.asarray(hist["pair_tcpa"], dtype=float)
    return t, X_all, pair_risk, pair_dcpa, pair_dist, pair_tcpa

def xy_nmi(X_all: np.ndarray, agent_idx: int) -> Tuple[np.ndarray, np.ndarray]: 
    """Extract x/y position in nmi from X_all history array"""
    x = X_all[:, agent_idx, 0] / NMI
    y = X_all[:, agent_idx, 1] / NMI
    return x, y

def closest_approach_index(pair_dist: np.ndarray, own_idx: int = 0, target_idx: int = 1) -> int:
    """
    Return the step index of ownship's closest approach to any other vessel
    """
    d = pair_dist[:, own_idx, target_idx]
    return int(np.argmin(d))

def series_max_risk(pair_risk: np.ndarray, own_idx: int = 0) -> np.ndarray:
    """
    Return the time series of ownship's maximum pairwise risk across all other vessels
    """
    # exclude self-risk on the diagonal 
    vals = pair_risk[:, own_idx, :].copy()
    vals[:, own_idx] = 0.0
    return np.nanmax(vals, axis=1)

def series_min_abs_dcpa(
    pair_dcpa: np.ndarray,
    pair_dist: np.ndarray,
    own_idx: int = 0,
    encounter_dist_nmi: float = 8.0,
    dcpa_clip_nmi: float = 5.0,
) -> np.ndarray:
    """
    Return ownship minimum absolute DCPA over time, but only when another ship
    is within an encounter-relevant range. Far-away values are masked.
    """
    dcpa_out = []

    encounter_dist_m = encounter_dist_nmi * NMI
    dcpa_clip_m = dcpa_clip_nmi * NMI

    n_steps = pair_dcpa.shape[0]
    for s in range(n_steps):
        dcpa_row = np.asarray(pair_dcpa[s, own_idx], dtype=float).copy()
        dist_row = np.asarray(pair_dist[s, own_idx], dtype=float).copy()

        if own_idx < len(dcpa_row):
            dcpa_row[own_idx] = np.nan
            dist_row[own_idx] = np.nan

        valid = np.isfinite(dcpa_row) & np.isfinite(dist_row) & (dist_row <= encounter_dist_m)

        if not np.any(valid):
            dcpa_out.append(np.nan)
            continue

        v = np.min(np.abs(dcpa_row[valid]))
        v = min(v, dcpa_clip_m)
        dcpa_out.append(v / NMI)

    return np.asarray(dcpa_out, dtype=float)


def compute_target_series(pair_data: np.ndarray, own_idx: int = 0) -> Dict[str, np.ndarray]:
    """
    Compute time series for each target ship (all ships except ownship).
    Returns dict with keys like 'TS1', 'TS2', etc. mapping to time series arrays.
    """
    n_steps = pair_data.shape[0]
    n_agents = pair_data.shape[1]
    
    target_series = {}
    for j in range(n_agents):
        if j == own_idx:
            continue
        key = f"TS{j}"
        target_series[key] = pair_data[:, own_idx, j].astype(float)
    
    return target_series


def plot_ownship_cpa_panel_baseline_rl(
    baseline_hist: dict,
    rl_hist: dict,
    save_path: str | Path,
    own_idx: int = 0,
) -> Path:
    """
    Plot CORALL-style CPA analysis panel: DCPA, Risk, TCPA, Range for baseline vs RL.
    4 subplots for each metric, showing evolution over time for all target ships.
    Requires pair_tcpa data to be saved in episode history.
    """
    tb, _, risk_b, dcpa_b, dist_b, tcpa_b = to_numpy_history(baseline_hist)
    tr, _, risk_r, dcpa_r, dist_r, tcpa_r = to_numpy_history(rl_hist)
    
    # Convert distance from m to nmi
    dist_b_nmi = dist_b / NMI
    dist_r_nmi = dist_r / NMI
    
    # Extract per-target series
    dcpa_b_targets = compute_target_series(dcpa_b, own_idx=own_idx)
    dcpa_r_targets = compute_target_series(dcpa_r, own_idx=own_idx)
    risk_b_targets = compute_target_series(risk_b, own_idx=own_idx)
    risk_r_targets = compute_target_series(risk_r, own_idx=own_idx)
    tcpa_b_targets = compute_target_series(tcpa_b, own_idx=own_idx)
    tcpa_r_targets = compute_target_series(tcpa_r, own_idx=own_idx)
    dist_b_targets = compute_target_series(dist_b_nmi, own_idx=own_idx)
    dist_r_targets = compute_target_series(dist_r_nmi, own_idx=own_idx)
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Ownship CPA Analysis | Case {baseline_hist.get('case', '?')} | Seed {baseline_hist.get('seed', '?')}",
                 fontsize=14, fontweight="bold")
    
    # 1. DCPA (top-left)
    ax = axes[0, 0]
    for key, ts in dcpa_b_targets.items():
        ax.plot(tb, np.abs(ts) / NMI, "--", linewidth=1.5, alpha=0.7, label=key)
    for key, ts in dcpa_r_targets.items():
        ax.plot(tr, np.abs(ts) / NMI, "-", linewidth=1.5, alpha=0.7, label=f"{key}_RL")
    ax.set_ylabel("DCPA (nmi)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Distance to Closest Point of Approach")
    
    # 2. Risk (top-right)
    ax = axes[0, 1]
    for key, ts in risk_b_targets.items():
        ax.plot(tb, ts, "--", linewidth=1.5, alpha=0.7, label=key)
    for key, ts in risk_r_targets.items():
        ax.plot(tr, ts, "-", linewidth=1.5, alpha=0.7, label=f"{key}_RL")
    ax.set_ylabel("Risk")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Collision Risk")
    ax.set_ylim([0, 1.0])
    
    # 3. TCPA (bottom-left) - now uses actual pair_tcpa data
    ax = axes[1, 0]
    for key, ts in tcpa_b_targets.items():
        ax.plot(tb, ts, "--", linewidth=1.5, alpha=0.7, label=key)
    for key, ts in tcpa_r_targets.items():
        ax.plot(tr, ts, "-", linewidth=1.5, alpha=0.7, label=f"{key}_RL")
    ax.set_ylabel("TCPA (s)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Time to Closest Point of Approach")
    
    # 4. Summary Risk (bottom-right)
    ax = axes[1, 1]
    max_risk_b = series_max_risk(risk_b, own_idx=own_idx)
    max_risk_r = series_max_risk(risk_r, own_idx=own_idx)
    ax.plot(tb, max_risk_b, "--", linewidth=2, label="Baseline (max risk)")
    ax.plot(tr, max_risk_r, "-", linewidth=2, label="RL (max risk)")
    ax.set_ylabel("Max Pairwise Risk")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Aggregate Ownship Risk")
    ax.set_ylim([0, 1.0])
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return Path(save_path)


def plot_ownship_cpa_panel(
    baseline_hist: Dict,
    rl_hist: Dict,
    save_path: str | Path,
    own_idx: int = 0,
    target_indices: Optional[List[int]] = None,
    t_max: Optional[float] = None,
    dcpa_ylim: Optional[Tuple[float, float]] = None,
    range_ylim: Optional[Tuple[float, float]] = None,
    tcpa_ylim: Optional[Tuple[float, float]] = None,
    risk_ylim: Optional[Tuple[float, float]] = None,
) -> Path:
    """
    CORALL-style CPA/risk panel comparing baseline vs RL.
    Shows DCPA, Range, Risk per target ship with baseline (--black) vs RL (-purple).
    
    Args:
        dcpa_ylim: Y-axis limits for DCPA panel [min, max]. If None, auto-scales.
        range_ylim: Y-axis limits for Range panel [min, max]. If None, auto-scales.
        tcpa_ylim: Y-axis limits for TCPA panel [min, max]. If None, auto-scales.
        risk_ylim: Y-axis limits for Risk panel [min, max]. If None, auto-scales.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    tb, X_allb, pair_riskb, pair_dcpab, pair_distb, pair_tcpab = to_numpy_history(baseline_hist)
    tr, X_allr, pair_riskr, pair_dcpar, pair_distr, pair_tcpar = to_numpy_history(rl_hist)

    n_agents = X_allb.shape[1]

    if target_indices is None:
        target_indices = [j for j in range(n_agents) if j != own_idx]

    if t_max is not None:
        maskb = tb <= t_max
        maskr = tr <= t_max
    else:
        maskb = np.ones_like(tb, dtype=bool)
        maskr = np.ones_like(tr, dtype=bool)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Plot each target ship's metrics
    for j in target_indices:
        # Baseline
        dcpa_jb = series_dcpa_target_filtered(pair_dcpab, pair_distb, own_idx, j, encounter_dist_nmi=8.0)
        range_jb = series_range_target(pair_distb, own_idx, j)
        risk_jb = series_risk_target(pair_riskb, own_idx, j)
        
        # RL
        dcpa_jr = series_dcpa_target_filtered(pair_dcpar, pair_distr, own_idx, j, encounter_dist_nmi=8.0)
        range_jr = series_range_target(pair_distr, own_idx, j)
        risk_jr = series_risk_target(pair_riskr, own_idx, j)

        # DCPA (top-left)
        axs[0, 0].plot(tb[maskb], dcpa_jb[maskb], "--", linewidth=1.5, alpha=0.7, color="black", label=f"TS{j} baseline")
        axs[0, 0].plot(tr[maskr], dcpa_jr[maskr], "-", linewidth=1.5, alpha=0.7, color="purple", label=f"TS{j} RL")
        
        # Range (top-right)
        axs[0, 1].plot(tb[maskb], range_jb[maskb], "--", linewidth=1.5, alpha=0.7, color="black", label=f"TS{j} baseline")
        axs[0, 1].plot(tr[maskr], range_jr[maskr], "-", linewidth=1.5, alpha=0.7, color="purple", label=f"TS{j} RL")
        
        # Risk (bottom-right)
        axs[1, 1].plot(tb[maskb], risk_jb[maskb], "--", linewidth=1.5, alpha=0.7, color="black", label=f"TS{j} baseline")
        axs[1, 1].plot(tr[maskr], risk_jr[maskr], "-", linewidth=1.5, alpha=0.7, color="purple", label=f"TS{j} RL")
        
        # TCPA (bottom-left)
        tcpa_jb = series_tcpa_target(pair_tcpab, own_idx, j)
        tcpa_jr = series_tcpa_target(pair_tcpar, own_idx, j)
        axs[1, 0].plot(tb[maskb], tcpa_jb[maskb], "--", linewidth=1.5, alpha=0.7, color="black", label=f"TS{j} baseline")
        axs[1, 0].plot(tr[maskr], tcpa_jr[maskr], "-", linewidth=1.5, alpha=0.7, color="purple", label=f"TS{j} RL")

    axs[0, 0].set_ylabel("DCPA (nmi)")
    axs[0, 0].set_title("Distance to Closest Point of Approach")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=8, loc="best")
    if dcpa_ylim is not None:
        axs[0, 0].set_ylim(dcpa_ylim)

    axs[0, 1].set_ylabel("Range (nmi)")
    axs[0, 1].set_title("Range (Distance)")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(fontsize=8, loc="best")
    if range_ylim is not None:
        axs[0, 1].set_ylim(range_ylim)

    # TCPA panel (bottom-left)
    axs[1, 0].set_ylabel("TCPA (s)")
    axs[1, 0].set_title("Time to Closest Point of Approach")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(fontsize=8, loc="best")
    if tcpa_ylim is not None:
        axs[1, 0].set_ylim(tcpa_ylim)

    axs[1, 1].set_ylabel("Risk")
    axs[1, 1].set_title("Collision Risk")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend(fontsize=8, loc="best")
    if risk_ylim is not None:
        axs[1, 1].set_ylim(risk_ylim)

    for ax in [axs[0, 0], axs[0, 1], axs[1, 1]]:
        ax.set_xlabel("Time (s)")

    case = baseline_hist.get("case", None)
    seed = baseline_hist.get("seed", None)
    title = "Ownship CPA Analysis: Baseline vs RL"
    if case is not None:
        title += f" | Case {case}"
    if seed is not None:
        title += f" | Seed {seed}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    fig.tight_layout(rect=(0, 0, 1.0, 0.96))
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_full_trajectory_overlay(
    baseline_hist: dict,
    rl_hist: dict,
    save_path: str | Path,
    show_goal: bool = True,
    show_all_targets: bool = True,
    target_alpha: float = 0.7,
    target_linewidth: float = 2.8,
    show_target_labels: bool = False,
) -> Path:
    """
    Clean full-route comparison:
    - ownship baseline vs RL
    - optional nominal/reference trajectories for all target ships
    - start/end markers
    - optional goal marker
    - NO CPA markers on the full plot
    
    Args:
        target_alpha: Alpha transparency for target ships (default 0.7, more opaque for visibility)
        target_linewidth: Line width for target ships (default 2.8, thicker for prominence)
        show_target_labels: Whether to show "Target starts" label (default False for cleaner plot)
    """

    _, Xb, _, _, _, _ = to_numpy_history(baseline_hist)
    _, Xr, _, _, _, _ = to_numpy_history(rl_hist)

    xb0, yb0 = xy_nmi(Xb, 0)
    xr0, yr0 = xy_nmi(Xr, 0)

    n_agents = Xb.shape[1]

    fig, ax = plt.subplots(figsize=(13, 8))

    # --------------------------------------------------
    # Plot all target ships as nominal straight-line paths
    # using baseline history initial state for consistency
    # --------------------------------------------------
    if show_all_targets and n_agents > 1:
        # estimate route length from overall x-span of ownship tracks
        ownship_span_guess = max(
            np.ptp(xb0) if len(xb0) > 1 else 0.0,
            np.ptp(xr0) if len(xr0) > 1 else 0.0,
            1.0,
        )

        for j in range(1, n_agents):
            xt0 = Xb[0, j, 0] / NMI
            yt0 = Xb[0, j, 1] / NMI
            psi_t = float(Xb[0, j, 2])

            # build straight nominal path from initial heading
            xt_line = np.linspace(xt0, xt0 + ownship_span_guess * np.cos(psi_t), 100)
            yt_line = np.linspace(yt0, yt0 + ownship_span_guess * np.sin(psi_t), 100)

            ax.plot(
                xt_line,
                yt_line,
                linestyle="-",
                linewidth=3.5,
                alpha=target_alpha,
                color="tab:cyan",
                label="Target ships (nominal paths)" if j == 1 else None,
                zorder=1,
            )

            ax.scatter(
                xt0,
                yt0,
                marker="s",
                s=35,
                alpha=min(0.9, target_alpha + 0.2),
                color="cyan",
                label="Target starts" if (j == 1 and show_target_labels) else None,
                zorder=2,
            )

    # -----------------------
    # Ownship tracks
    # -----------------------
    ax.plot(xb0, yb0, "--", color="black", linewidth=2.5, label="CORALL baseline ownship", zorder=3)
    ax.plot(xr0, yr0, "--", color="purple", linewidth=2.5, label="RL policy ownship", zorder=3)

    # Start marker
    ax.scatter(xb0[0], yb0[0], s=120, color="orange", label="Start", zorder=4)

    # End markers
    ax.scatter(xb0[-1], yb0[-1], marker="*", s=160, color="orange", label="Baseline end", zorder=4)
    ax.scatter(xr0[-1], yr0[-1], marker="*", s=160, color="green", label="RL end", zorder=4)

    # Goal marker
    if show_goal:
        # use baseline end x as a simple display proxy for goal vicinity
        goal_x = max(xb0[-1], xr0[-1])
        ax.scatter(goal_x, 0.0, marker="D", s=80, color="tab:red", label="Goal vicinity", zorder=4)

    ax.set_title(
        f"Baseline vs RL trajectory | Imazu case {baseline_hist.get('case', '?')} | seed {baseline_hist.get('seed', '?')}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("X position (nmi)")
    ax.set_ylabel("Y position (nmi)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_aspect("auto")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # bounds based on ownship and target nominal lines
    all_x = [xb0, xr0]
    all_y = [yb0, yr0]

    if show_all_targets and n_agents > 1:
        for j in range(1, n_agents):
            xt0 = Xb[0, j, 0] / NMI
            yt0 = Xb[0, j, 1] / NMI
            psi_t = float(Xb[0, j, 2])
            ownship_span_guess = max(
                np.ptp(xb0) if len(xb0) > 1 else 0.0,
                np.ptp(xr0) if len(xr0) > 1 else 0.0,
                1.0,
            )
            xt_line = np.linspace(xt0, xt0 + ownship_span_guess * np.cos(psi_t), 100)
            yt_line = np.linspace(yt0, yt0 + ownship_span_guess * np.sin(psi_t), 100)
            all_x.append(xt_line)
            all_y.append(yt_line)

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    xpad = max(0.3, 0.08 * max(1.0, np.ptp(all_x)))
    ypad = max(0.15, 0.15 * max(0.5, np.ptp(all_y)))
    ax.set_xlim(np.min(all_x) - xpad, np.max(all_x) + xpad)
    ax.set_ylim(np.min(all_y) - ypad, np.max(all_y) + ypad)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return Path(save_path)


def plot_encounter_detail_clean(
    baseline_hist: dict,
    rl_hist: dict,
    save_path: str | Path,
    target_idx: Optional[int] = None,
    zoom_margin_nmi: float = 0.5,
    window_steps: int = 200,
) -> Path:
    
    tb, Xb, _, _, dist_b, _ = to_numpy_history(baseline_hist)
    tr, Xr, _, _, dist_r, _ = to_numpy_history(rl_hist)

    if target_idx is None:
        target_idx_b = closest_target_index(dist_b, own_idx=0)
        target_idx_r = closest_target_index(dist_r, own_idx=0)
    else:
        target_idx_b = int(target_idx)
        target_idx_r = int(target_idx)

    ib = closest_approach_index(dist_b, own_idx=0, target_idx=target_idx_b)
    ir = closest_approach_index(dist_r, own_idx=0, target_idx=target_idx_r)

    xb0, yb0 = xy_nmi(Xb, 0)
    xr0, yr0 = xy_nmi(Xr, 0)

    xbt, ybt = xy_nmi(Xb, target_idx_b)
    xrt, yrt = xy_nmi(Xr, target_idx_r)

    # Zoom window around each run's own CPA
    i0_b = max(0, ib - window_steps)
    i1_b = min(len(tb), ib + window_steps + 1)
    i0_r = max(0, ir - window_steps)
    i1_r = min(len(tr), ir + window_steps + 1)

    xb_cpa, yb_cpa = xb0[ib], yb0[ib]
    xr_cpa, yr_cpa = xr0[ir], yr0[ir]

    min_sep_b = dist_b[ib, 0, target_idx_b] / NMI
    min_sep_r = dist_r[ir, 0, target_idx_r] / NMI

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.plot(xb0[i0_b:i1_b], yb0[i0_b:i1_b], "--", color="black", linewidth=2.5, label="Baseline ownship")
    ax.plot(xr0[i0_r:i1_r], yr0[i0_r:i1_r], "-", color="purple", linewidth=2.5, label="RL ownship")

    ax.plot(xbt[i0_b:i1_b], ybt[i0_b:i1_b], color="tab:blue", linewidth=2.0,
            label=f"Baseline target ship {target_idx_b}")
    ax.plot(xrt[i0_r:i1_r], yrt[i0_r:i1_r], color="tab:cyan", linewidth=2.0,
            label=f"RL target ship {target_idx_r}")

    ax.scatter(xb_cpa, yb_cpa, marker="X", s=180, color="red", label="Baseline CPA")
    ax.scatter(xr_cpa, yr_cpa, marker="X", s=180, color="orange", label="RL CPA")

    ax.plot([xbt[ib], xb_cpa], [ybt[ib], yb_cpa], "-", color="red", linewidth=1.8, alpha=0.9)
    ax.plot([xrt[ir], xr_cpa], [yrt[ir], yr_cpa], "-", color="orange", linewidth=1.8, alpha=0.9)

    dcpa_b_nmi = np.abs(baseline_hist["pair_dcpa"][ib][0][target_idx_b]) / NMI
    dcpa_r_nmi = np.abs(rl_hist["pair_dcpa"][ir][0][target_idx_r]) / NMI

    text = (
        f"Baseline min sep: {min_sep_b:.2f} nmi\n"
        f"Baseline |DCPA| at CPA: {dcpa_b_nmi:.2f} nmi\n"
        f"RL min sep: {min_sep_r:.2f} nmi\n"
        f"RL |DCPA| at CPA: {dcpa_r_nmi:.2f} nmi"
    )

    # Zoom bounds centered on CPA region
    xs = np.array([xb_cpa, xr_cpa, xbt[ib], xrt[ir]])
    ys = np.array([yb_cpa, yr_cpa, ybt[ib], yrt[ir]])

    xc, yc = np.mean(xs), np.mean(ys)
    xmin, xmax = xc - 1.5, xc + 1.5
    ymin, ymax = yc - 1.5, yc + 1.5

    ax.set_xlim(float(xmin), float(xmax))
    ax.set_ylim(float(ymin), float(ymax))
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(
        f"Encounter detail | Imazu case {baseline_hist.get('case', '?')} | seed {baseline_hist.get('seed', '?')}",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("X position (nmi)")
    ax.set_ylabel("Y position (nmi)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return Path(save_path)


def plot_risk_timeseries(
    baseline_hist: dict,
    rl_hist: dict,
    save_path: str | Path,
) -> Path:
    tb, _, risk_b, _, _, _ = to_numpy_history(baseline_hist)
    tr, _, risk_r, _, _, _ = to_numpy_history(rl_hist)

    max_risk_b = series_max_risk(risk_b, own_idx=0)
    max_risk_r = series_max_risk(risk_r, own_idx=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tb, max_risk_b, "--", linewidth=2.5, label="Baseline max pairwise risk")
    ax.plot(tr, max_risk_r, linewidth=2.5, label="RL max pairwise risk")

    ib = int(np.nanargmax(max_risk_b))
    ir = int(np.nanargmax(max_risk_r))
    ax.scatter(tb[ib], max_risk_b[ib], s=70, label="Baseline peak")
    ax.scatter(tr[ir], max_risk_r[ir], s=70, label="RL peak")

    ax.set_title("Ownship max pairwise risk over time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Max pairwise risk")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return Path(save_path)


def plot_min_dcpa_timeseries(
    baseline_hist: dict,
    rl_hist: dict,
    save_path: str | Path,
    encounter_dist_nmi: float = 8.0,
    dcpa_clip_nmi: float = 5.0,
) -> Path:
    
    tb, _, _, dcpa_b, dist_b, _ = to_numpy_history(baseline_hist)
    tr, _, _, dcpa_r, dist_r, _ = to_numpy_history(rl_hist)

    min_dcpa_b = series_min_abs_dcpa(
        dcpa_b, dist_b, own_idx=0,
        encounter_dist_nmi=encounter_dist_nmi,
        dcpa_clip_nmi=dcpa_clip_nmi,
    )
    min_dcpa_r = series_min_abs_dcpa(
        dcpa_r, dist_r, own_idx=0,
        encounter_dist_nmi=encounter_dist_nmi,
        dcpa_clip_nmi=dcpa_clip_nmi,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tb, min_dcpa_b, "--", linewidth=2.5, label="Baseline min |DCPA|")
    ax.plot(tr, min_dcpa_r, linewidth=2.5, label="RL min |DCPA|")

    ib = int(np.nanargmin(min_dcpa_b))
    ir = int(np.nanargmin(min_dcpa_r))
    ax.scatter(tb[ib], min_dcpa_b[ib], s=70, label="Baseline minimum")
    ax.scatter(tr[ir], min_dcpa_r[ir], s=70, label="RL minimum")

    ax.set_title("Ownship minimum absolute DCPA over time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Min |DCPA| (nmi)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return Path(save_path)
    
def ensure_array_list(x):
    """
    Ensure the input is a list of numpy arrays. If it's a single numpy array, wrap it in a list.
    """
    return [np.asarray(v) for v in x]

def save_episode_history(history: Dict, output_path: str | Path) -> Path: 
    """
    Save episode trajectory / state history to a compressed NPZ file for later analysis / visualization.

    Arrays are saved with explicit keys so they can be loaded later for plotting
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path, 
        t = np.asarray(history["t"], dtype=float), 
        X_all = np.asarray(history["X_all"], dtype=float),
        pair_risk = np.asarray(history["pair_risk"], dtype=float), 
        pair_dcpa = np.asarray(history["pair_dcpa"], dtype=float),
        pair_dist = np.asarray(history["pair_dist"], dtype=float),
        pair_tcpa = np.asarray(history["pair_tcpa"], dtype=float),
        agents = np.asarray(history["agents"], dtype=object), 
        case = np.asarray([history.get("case", -1)], dtype=int), 
        seed = np.asarray([history.get("seed", -1)], dtype=int), 
        baseline = np.asarray([history.get("baseline", "")], dtype=object), 
        checkpoint = np.asarray([history.get("checkpoint", "")], dtype=object),
    )

    return output_path

def load_episode_history(path: str | Path) -> Dict:
    """
    Load episode trajectory / state history from a compressed NPZ file saved by `save_episode_history`.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    history = {
        "t": data["t"],
        "X_all": data["X_all"],
        "pair_risk": data["pair_risk"],
        "pair_dcpa": data["pair_dcpa"],
        "pair_dist": data["pair_dist"],
        "pair_tcpa": data["pair_tcpa"] if "pair_tcpa" in data else None,
        "agents": list(data["agents"]) if "agents" in data else None,
        "case": int(data["case"][0]) if "case" in data else None,
        "seed": int(data["seed"][0]) if "seed" in data else None,
        "baseline": str(data["baseline"][0]) if "baseline" in data else None,
        "checkpoint": str(data["checkpoint"][0]) if "checkpoint" in data else None,
    }

    return history

def moving_average(x: np.ndarray, window: int = 51) -> np.ndarray:
    """Simple moving average smoothing for 1D array."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or window <= 1:
        return x
    window = min(window, len(x))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


    
def risk_series(pair_risk: np.ndarray, own_idx: int = 0) -> np.ndarray:
    """
    Return the risk series of ownship's interactions with other vessels
    """

    out = []
    for s in range(pair_risk.shape[0]):
        row = np.asarray(pair_risk[s, own_idx], dtype=float).copy()

        if own_idx < row.shape[0]:
            row[own_idx] = 0.0
        out.append(float(np.max(row))) if row.size else 0.0

    return np.asarray(out, dtype=float)
    
def dcpa_series(pair_dcpa: np.ndarray, own_idx: int = 0) -> np.ndarray:
    """
    Return the DCPA series of ownship's interactions with other vessels
    """

    out = []
    for s in range(pair_dcpa.shape[0]):
        row = np.asarray(pair_dcpa[s, own_idx], dtype=float)
        row = row[np.isfinite(row)]  # filter out non-finite values
        out.append(float(np.min(np.abs(row))) if row.size else np.nan)
    return np.asarray(out, dtype=float)

def series_dcpa_target(pair_dcpa: np.ndarray, own_idx: int, target_idx: int) -> np.ndarray:
    """
    Ownship DCPA relative to one specific target ship over time.
    Returns |DCPA| in nmi.
    """
    vals = np.abs(np.asarray(pair_dcpa[:, own_idx, target_idx], dtype=float)) / NMI
    vals[~np.isfinite(vals)] = np.nan
    return vals


def series_dcpa_target_filtered(
    pair_dcpa: np.ndarray,
    pair_dist: np.ndarray,
    own_idx: int,
    target_idx: int,
    encounter_dist_nmi: float = 8.0,
    dcpa_clip_nmi: float = 5.0,
) -> np.ndarray:
    """
    Ownship DCPA relative to specific target ship, filtered to only meaningful encounters.
    Masks values where target is beyond encounter_dist_nmi.
    Returns |DCPA| in nmi, clipped to dcpa_clip_nmi.
    """
    dcpa_vals = np.abs(np.asarray(pair_dcpa[:, own_idx, target_idx], dtype=float)) / NMI
    dist_vals = np.asarray(pair_dist[:, own_idx, target_idx], dtype=float) / NMI
    
    # Mask out encounters beyond encounter distance threshold
    mask = dist_vals <= encounter_dist_nmi
    dcpa_vals = np.where(mask, dcpa_vals, np.nan)
    
    # Clip to meaningful range
    dcpa_vals = np.minimum(dcpa_vals, dcpa_clip_nmi)
    
    return dcpa_vals


def series_tcpa_target(pair_tcpa: np.ndarray, own_idx: int, target_idx: int) -> np.ndarray:
    """
    Ownship TCPA relative to one specific target ship over time.
    Returns TCPA in seconds.
    """
    vals = np.asarray(pair_tcpa[:, own_idx, target_idx], dtype=float)
    vals[~np.isfinite(vals)] = np.nan
    return vals


def series_range_target(pair_dist: np.ndarray, own_idx: int, target_idx: int) -> np.ndarray:
    """
    Ownship range to one specific target ship over time.
    Returns range in nmi.
    """
    vals = np.asarray(pair_dist[:, own_idx, target_idx], dtype=float) / NMI
    vals[~np.isfinite(vals)] = np.nan
    return vals


def series_risk_target(pair_risk: np.ndarray, own_idx: int, target_idx: int) -> np.ndarray:
    """
    Ownship pairwise risk relative to one specific target ship over time.
    """
    vals = np.asarray(pair_risk[:, own_idx, target_idx], dtype=float)
    vals[~np.isfinite(vals)] = np.nan
    return vals


def pick_snapshot_indices(n_steps: int, n_snapshots: int = 2) -> np.ndarray: 
    if n_steps <= 0: 
        return np.asarray([], dtype=int)
    
    if n_steps <= n_snapshots:
        return np.arange(n_steps, dtype=float)
    
    return np.unique(np.linspace(0, n_steps - 1, n_snapshots).astype(int))

def draw_ship_snapshot(ax, x_m, y_m, psi, color, loa_m = 30.0, bol_m = 16.0, scale=1.0):
    """
    Draw a ship snapshot on the given axis at the specified position and orientation.
    """
    x_nmi = x_m / NMI
    y_nmi = y_m / NMI
    loa_nmi = loa_m / NMI * scale
    bol_nmi = bol_m / NMI * scale

    if animate_ship is not None:
        animate_ship(x_nmi, y_nmi, psi, loa_nmi, bol_nmi, cpa=0.0, color=color, ax=ax)
    else: 
        ax.scatter([x_nmi], [y_nmi], color=color, s=50, marker='s', zorder=4)

def plot_episode_overlay(
        baseline_history: Dict, 
        trained_history: Dict,
        output_path: str | Path, 
        own_idx: int = 0,
        title: Optional[str] = None,
        show_snapshots: bool = False,
        n_snapshots: int = 2,
        loa_m: float = 30.0,
        bol_m: float = 16.0,
) -> Path: 
    
    """
    Static CORALL-style overlay plot for one baseline rollout vs. one trained policy rollout, showing trajectories, risk, DCPA, and snapshots of ship positions at key moments.
        - ship tracks plotted in nmi 
        - obstacle tracks shown in gray
        - ownship baseline vs RL highlighted
        - closest approach points marked
        - optional ship silhouettes at evenly spaced time intervals

    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    Xb = np.asarray(baseline_history["X_all"], dtype=float)
    Xt = np.asarray(trained_history["X_all"], dtype=float)

    nb, na_b, _ = Xb.shape
    nt, na_t, _ = Xt.shape

    if na_b != na_t: 
        raise ValueError(f"Agent-count mismatch: baseline has {na_b} agents but trained policy has {na_t} agents")

    n_agents = na_b

    fig, ax = plt.subplots(figsize=(10, 8))

    # background obstacle tracks in from baseline geometry
    for j in range(1, n_agents):
        xb = Xb[:, j, 0] / NMI
        yb = Xb[:, j, 1] / NMI
        ax.plot(xb, yb, color="0.75", linewidth=1.5, alpha=0.9, zorder=1)
        ax.scatter(xb[0], yb[0], color="0.55", s=20, marker='s', zorder=2)  # start
    
    target_idx = closest_target_index(np.asarray(baseline_history["pair_dist"], dtype=float), own_idx=own_idx)
    xbt = Xb[:, target_idx, 0] / NMI
    ybt = Xb[:, target_idx, 1] / NMI
    ax.plot(xbt, ybt, color="tab:cyan", linewidth=2.0, alpha=0.9, label=f"Target ship {target_idx}", zorder=2)

    # ownship tracks
    xbo = Xb[:, own_idx, 0] / NMI
    ybo = Xb[:, own_idx, 1] / NMI
    xto = Xt[:, own_idx, 0] / NMI
    yto = Xt[:, own_idx, 1] / NMI

    ax.plot(xbo, ybo, linestyle="--", linewidth=2.6, color="black", label="CORALL baseline ownship", zorder=3)
    ax.plot(xto, yto, linestyle="--", linewidth=2.8, color="purple", label="RL policy ownship", zorder=3)

    # start and end markers
    ax.scatter(xbo[0], ybo[0], color="black", s=80, marker='o', zorder=4, label="Start")  # baseline start
    ax.scatter(xbo[-1], ybo[-1], color="black", s=80, marker='*', zorder=4, label="Baseline End")  # baseline end
    ax.scatter(xto[-1], yto[-1], color="purple", s=80, marker='*', zorder=4, label="RL End")  # RL end

    # closest approach markers for ownship
    ib = closest_approach_index(np.asarray(baseline_history["pair_dist"], dtype=float), own_idx=own_idx)
    it = closest_approach_index(np.asarray(trained_history["pair_dist"], dtype=float), own_idx=own_idx)
    ax.scatter(xbo[ib], ybo[ib], color="red", s=100, marker='X', zorder=5, label="Baseline Closest Approach")
    ax.scatter(xto[it], yto[it], color="orange", s=100, marker='X', zorder=5, label="RL Closest Approach")

    if show_snapshots: 
        for idx in pick_snapshot_indices(nb, n_snapshots):
            draw_ship_snapshot(ax, Xb[idx, own_idx, 0], Xb[idx, own_idx, 1], Xb[idx, own_idx, 2], color=[0.1, 0.1, 0.1], loa_m=loa_m, bol_m=bol_m, scale=1.2)
        for idx in pick_snapshot_indices(nt, n_snapshots):
            draw_ship_snapshot(ax, Xt[idx, own_idx, 0], Xt[idx, own_idx, 1], Xt[idx, own_idx, 2], color=[0.55, 0.1, 0.55], loa_m=loa_m, bol_m=bol_m, scale=1.2)

    ax.set_xlabel("X position (nmi)")
    ax.set_ylabel("Y position (nmi)")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.25)

    case = baseline_history.get("case", None)
    seed = baseline_history.get("seed", None)

    if title is None: 
        title = f"Baseline vs RL overlay"

        if case is not None: 
            title += f" | Imazu case {case}"
        if seed is not None:
            title += f" | seed {seed}"
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best")

    # autocscale with margin from both runs 
    all_x = np.concatenate([Xb[:, :, 0].ravel(), Xt[:, :, 0].ravel()]) / NMI
    all_y = np.concatenate([Xb[:, :, 1].ravel(), Xt[:, :, 1].ravel()]) / NMI
    xpad = max(0.5, 0.08 * max(1.0, np.ptp(all_x)))
    ypad = max(0.5, 0.08 * max(1.0, np.ptp(all_y)))
    ax.set_xlim(np.min(all_x) - xpad, np.max(all_x) + xpad)
    ax.set_ylim(np.min(all_y) - ypad, np.max(all_y) + ypad)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def plot_risk_dcpa_timeseries(
        baseline_history: Dict, 
        trained_history: Dict,
        output_path: str | Path, 
        own_idx: int = 0,
        dcpa_clip_nmi: float = 5.0,
        smooth_window: int = 51
    ) -> Path: 

    """

    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tb = np.asarray(baseline_history["t"], dtype=float)
    tt = np.asarray(trained_history["t"], dtype=float)
    
    risk_b = risk_series(np.asarray(baseline_history["pair_risk"], dtype=float), own_idx=own_idx)
    risk_t = risk_series(np.asarray(trained_history["pair_risk"], dtype=float), own_idx=own_idx)
    # smooth risk for readibility
    risk_b = moving_average(risk_b, window=smooth_window)
    risk_t = moving_average(risk_t, window=smooth_window)

    # DCPA comes from env in m -> convert to nmi for plotting, and clip to focus on meaningful range of values
    dcpa_b = dcpa_series(np.asarray(baseline_history["pair_dcpa"], dtype=float), own_idx=own_idx) / NMI
    dcpa_t = dcpa_series(np.asarray(trained_history["pair_dcpa"], dtype=float), own_idx=own_idx) / NMI
    # remove far distance noise
    dcpa_b = np.clip(dcpa_b, 0, 5) # 0-5nmi is meaningful DCPA
    dcpa_t = np.clip(dcpa_t, 0, 5) # 0-5nmi is meaningful DCPA

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(tb, risk_b, linestyle="--", linewidth=2.2, color="black", label="Baseline max Risk", zorder=3)
    ax1.plot(tt, risk_t, linestyle="-", linewidth=2.2, color="purple", label="RL max Risk", zorder=3)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Max pairwise risk")
    ax1.grid(True, alpha=0.25)
    # focus x-axis on encounter window
    ax1.set_xlim(0, 2000)

    ax2 = ax1.twinx()
    ax2.plot(tb, dcpa_b, linestyle="--", linewidth=2.0, color="tab:red", label="Baseline min |DCPA|", zorder=3)
    ax2.plot(tt, dcpa_t, linestyle="-", linewidth=2.0, color="tab:orange", label="RL min |DCPA|", zorder=3)
    ax2.set_ylabel("Min |DCPA| (nmi)")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [str(ln.get_label()) for ln in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=9)
    ax1.set_title("Ownship risk and closest approach history", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path

def closest_target_index(pair_dist: np.ndarray, own_idx: int = 0) -> int:
    """
    Return the other-ship index corresponding to ownship's closest approach
    over the whole episode, using the minimum separation across time.
    """
    pair_dist = np.asarray(pair_dist, dtype=float)
    n_steps, n_agents, _ = pair_dist.shape

    best_j = None
    best_d = np.inf

    for j in range(n_agents):
        if j == own_idx:
            continue
        d = pair_dist[:, own_idx, j]
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        dmin = float(np.min(d))
        if dmin < best_d:
            best_d = dmin
            best_j = j

    if best_j is None:
        return 1 if n_agents > 1 else 0
    return int(best_j)


def encounter_window_indices(pair_dist: np.ndarray, own_idx: int = 0, window_steps: int = 400) -> tuple[int, int, int]:
    """
    Center a plotting window around ownship's closest-approach instant.
    """
    idx = closest_approach_index(pair_dist, own_idx=own_idx)
    n_steps = pair_dist.shape[0]
    i0 = max(0, idx - window_steps)
    i1 = min(n_steps, idx + window_steps + 1)
    return i0, i1, idx


def compute_zoom_bounds(X: np.ndarray, agents_to_include: list[int], i0: int, i1: int, pad_nmi: float = 0.8):
    """
    Compute x/y bounds in nmi for a subset of agents and time window.
    """
    xs = []
    ys = []
    for j in agents_to_include:
        xs.append(X[i0:i1, j, 0] / NMI)
        ys.append(X[i0:i1, j, 1] / NMI)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))

    # ensure non-degenerate bounds
    if abs(xmax - xmin) < 1.0:
        xmin -= 0.5
        xmax += 0.5
    if abs(ymax - ymin) < 1.0:
        ymin -= 0.5
        ymax += 0.5

    return xmin - pad_nmi, xmax + pad_nmi, ymin - pad_nmi, ymax + pad_nmi

def plot_encounter_overlay(
        baseline_history: Dict,
        trained_history: Dict,
        output_path: str | Path,
        own_idx: int = 0,
        target_idx: Optional[int] = None,
        window_steps: int = 400,
        loa_m: float = 30.0,
        bol_m: float = 16.0,
        ship_scale: float = 5.0, # plot bigger ships for visiblity 
) -> Path:
    """
    Zoomed overlay centered on closest approach.
    Shows baseline ownship, RL ownship, and the selected target ship.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    Xb = np.asarray(baseline_history["X_all"], dtype=float)
    Xt = np.asarray(trained_history["X_all"], dtype=float)
    pair_dist_b = np.asarray(baseline_history["pair_dist"], dtype=float)

    if target_idx is None:
        target_idx = closest_target_index(pair_dist_b, own_idx=own_idx)

    i0, i1, ic = encounter_window_indices(pair_dist_b, own_idx=own_idx, window_steps=window_steps)

    fig, ax = plt.subplots(figsize=(9, 8))

    # trajectories in zoom window
    xbo = Xb[i0:i1, own_idx, 0] / NMI
    ybo = Xb[i0:i1, own_idx, 1] / NMI
    xto = Xt[i0:i1, own_idx, 0] / NMI
    yto = Xt[i0:i1, own_idx, 1] / NMI

    xbt = Xb[i0:i1, target_idx, 0] / NMI
    ybt = Xb[i0:i1, target_idx, 1] / NMI

    ax.plot(xbo, ybo, "--", color="black", linewidth=2.8, label="Baseline ownship", zorder=3)
    ax.plot(xto, yto, "-", color="purple", linewidth=2.8, label="RL ownship", zorder=3)
    ax.plot(xbt, ybt, "-", color="tab:blue", linewidth=2.2, alpha=0.85, label=f"Target ship {target_idx}", zorder=2)

    # closest-approach markers
    ib = closest_approach_index(pair_dist_b, own_idx=own_idx)
    it = closest_approach_index(np.asarray(trained_history["pair_dist"], dtype=float), own_idx=own_idx)

    # baseline: target point at baseline CPA instant
    ax.scatter(
        Xb[ib, target_idx, 0] / NMI,
        Xb[ib, target_idx, 1] / NMI,
        color="tab:blue",
        s=80,
        marker="o",
        zorder=5,
        label="Target at baseline CPA"
    )
    ax.plot(
        [Xb[ib, own_idx, 0] / NMI, Xb[ib, target_idx, 0] / NMI],
        [Xb[ib, own_idx, 1] / NMI, Xb[ib, target_idx, 1] / NMI],
        color="red",
        linewidth=1.5,
        alpha=0.8,
        zorder=4,
    )

    # RL: target point at RL CPA instant
    ax.scatter(
        Xt[it, target_idx, 0] / NMI,
        Xt[it, target_idx, 1] / NMI,
        color="tab:blue",
        s=80,
        marker="s",
        zorder=5,
        label="Target at RL CPA"
    )
    ax.plot(
        [Xt[it, own_idx, 0] / NMI, Xt[it, target_idx, 0] / NMI],
        [Xt[it, own_idx, 1] / NMI, Xt[it, target_idx, 1] / NMI],
        color="orange",
        linewidth=1.5,
        alpha=0.8,
        zorder=4,
    )

    ax.scatter(Xb[ib, own_idx, 0] / NMI, Xb[ib, own_idx, 1] / NMI,
               color="red", s=120, marker="X", zorder=5, label="Baseline closest approach")
    ax.scatter(Xt[it, own_idx, 0] / NMI, Xt[it, own_idx, 1] / NMI,
               color="orange", s=120, marker="X", zorder=5, label="RL closest approach")

    baseline_cpa_nmi = np.min(np.asarray(baseline_history["pair_dist"])[:, own_idx, target_idx]) / NMI
    rl_cpa_nmi = np.min(np.asarray(trained_history["pair_dist"])[:, own_idx, target_idx]) / NMI

    ax.text(
        0.02, 0.98,
        f"Baseline min sep: {baseline_cpa_nmi:.2f} nmi\nRL min sep: {rl_cpa_nmi:.2f} nmi",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )
    
    # large ship graphics at 3 key times: before / near / after
    snapshot_ids = np.unique(np.array([
        max(i0, ic - window_steps // 2),
        ic,
        min(i1 - 1, ic + window_steps // 2)
    ], dtype=int))

    for idx in snapshot_ids:
        # baseline ownship
        draw_ship_snapshot(
            ax, Xb[idx, own_idx, 0], Xb[idx, own_idx, 1], Xb[idx, own_idx, 2],
            color=[0.1, 0.1, 0.1], loa_m=loa_m, bol_m=bol_m, scale=ship_scale
        )
        # RL ownship
        draw_ship_snapshot(
            ax, Xt[idx, own_idx, 0], Xt[idx, own_idx, 1], Xt[idx, own_idx, 2],
            color=[0.55, 0.1, 0.55], loa_m=loa_m, bol_m=bol_m, scale=ship_scale
        )
        # target ship from baseline geometry
        draw_ship_snapshot(
            ax, Xb[idx, target_idx, 0], Xb[idx, target_idx, 1], Xb[idx, target_idx, 2],
            color=[0.2, 0.45, 0.9], loa_m=loa_m, bol_m=bol_m, scale=ship_scale
        )

    xmin, xmax, ymin, ymax = compute_zoom_bounds(
        Xb, agents_to_include=[own_idx, target_idx], i0=i0, i1=i1, pad_nmi=0.8
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    case = baseline_history.get("case", None)
    seed = baseline_history.get("seed", None)
    title = f"Encounter detail"
    if case is not None:
        title += f" | Imazu case {case}"
    if seed is not None:
        title += f" | seed {seed}"

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X position (nmi)")
    ax.set_ylabel("Y position (nmi)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path

def make_overlay_figure_set(
        baseline_history_path: str | Path,
        trained_history_path: str | Path,
        output_dir: str | Path,
        own_idx: int = 0,
        dcpa_ylim: Optional[Tuple[float, float]] = None,
        range_ylim: Optional[Tuple[float, float]] = None,
        tcpa_ylim: Optional[Tuple[float, float]] = None,
        risk_ylim: Optional[Tuple[float, float]] = None,
) -> Tuple[Path, Path, Path]: 
    """
    Generate a set of 3 overlay figures for a given pair of baseline vs. trained policy episode histories, and save to output directory.
    Returns: (full_overlay, encounter_detail, cpa_panel)
    
    Args:
        dcpa_ylim: Y-axis limits for DCPA panel [min, max]. If None, auto-scales.
        range_ylim: Y-axis limits for Range panel [min, max]. If None, auto-scales.
        tcpa_ylim: Y-axis limits for TCPA panel [min, max]. If None, auto-scales.
        risk_ylim: Y-axis limits for Risk panel [min, max]. If None, auto-scales.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_history = load_episode_history(baseline_history_path)
    trained_history = load_episode_history(trained_history_path)

    # overlay trajectory figures auto-pick relevant closest target experiencing
    full_overlay = plot_full_trajectory_overlay(
        baseline_history,
        trained_history,
        output_dir / "full_trajectory_overlay.png"
    )
    encounter_detail = plot_encounter_detail_clean(
        baseline_history,
        trained_history,
        output_dir / "encounter_detail_zoom.png",
        target_idx=None,
    )
    cpa_panel = plot_ownship_cpa_panel(
        baseline_history,
        trained_history,
        output_dir / "cpa_analysis_panel.png",
        own_idx=own_idx,
        dcpa_ylim=dcpa_ylim,
        range_ylim=range_ylim,
        tcpa_ylim=tcpa_ylim,
        risk_ylim=risk_ylim,
    )

    return full_overlay, encounter_detail, cpa_panel

if __name__ == "__main__":
   import argparse
   
   p = argparse.ArgumentParser(description="Create CORALL-style overlay figures from saved baseline/RL histories.")
   p.add_argument("--baseline_history", type=str, required=True)
   p.add_argument("--trained_history", type=str, required=True)
   p.add_argument("--output_dir", type=str, required=True)
   p.add_argument("--own_idx", type=int, default=0)
   p.add_argument("--dcpa_ylim", type=float, nargs=2, default=None, help="Y-axis limits for DCPA [min max]")
   p.add_argument("--range_ylim", type=float, nargs=2, default=None, help="Y-axis limits for Range [min max]")
   p.add_argument("--tcpa_ylim", type=float, nargs=2, default=None, help="Y-axis limits for TCPA [min max]")
   p.add_argument("--risk_ylim", type=float, nargs=2, default=None, help="Y-axis limits for Risk [min max]")
   args = p.parse_args()
   
   full_overlay_path, encounter_detail_path, cpa_panel_path = make_overlay_figure_set(
        args.baseline_history,
        args.trained_history,
        args.output_dir,
        own_idx=args.own_idx,
        dcpa_ylim=tuple(args.dcpa_ylim) if args.dcpa_ylim else None,
        range_ylim=tuple(args.range_ylim) if args.range_ylim else None,
        tcpa_ylim=tuple(args.tcpa_ylim) if args.tcpa_ylim else None,
        risk_ylim=tuple(args.risk_ylim) if args.risk_ylim else None,
        )
   
   print(f"Saved full trajectory overlay to: {full_overlay_path}")
   print(f"Saved encounter detail (zoom) to: {encounter_detail_path}")
   print(f"Saved CPA analysis panel to: {cpa_panel_path}")

