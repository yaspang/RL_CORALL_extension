"""
Generate synchronized LLM-guided execution plots to verify LLM intent integration.

Panel A: Ownship + target ship trajectory
Panel B: LLM intent and arbitration timeline
  - Shaded spans: intent_applied / rl_fallback / emergency_override
  - Solid line: applied heading at each control step
  - Dashed step line: LLM parsed K_dir between query points
  - Diamond markers at query points:
      green  = valid & correct
      orange = valid but incorrect
      red    = invalid / provider failure

Usage
-----
  # Single episode:
  python -m src.llm_integration.eval_LLM_intent \\
      --eval_dir "results_llmapi_case1_interval10/seed_0"

  # One figure per episode in episode_histories/:
  python -m src.llm_integration.eval_LLM_intent \\
      --eval_dir "results_llmapi_case1_interval10/seed_0" \\
      --all_episodes

Key arguments
-------------
  --eval_dir PATH          Seed dir with llm_intent_log.csv (required)
  --all_episodes           Generate one PNG per NPZ in episode_histories/
  --npz_path PATH          Explicit NPZ to plot (single-episode mode)
  --output PATH            Output PNG path (single-episode mode)

Requires
--------
  execution_log.csv in the seed dir (produced by eval_generalized_policy_sb3 --llm)
  results_llm_reliability/llm_call_scores.csv for correctness diamond colouring
"""
      --eval_dir "results_llmapi_case1_interval10_robust/seed_0"

    # Generate one figure per NPZ episode in episode_histories/:
    python -m src.llm_integration.eval_LLM_intent \\
        --eval_dir "results_llmapi_case1_interval10_robust/seed_0" \\
        --all_episodes

Requires execution_log.csv in the seed dir (produced by eval_generalized_policy_sb3.py).
If results_llm_reliability/llm_call_scores.csv exists, diamond correctness colouring is used;
otherwise only valid/invalid is shown.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

NMI = 1852.0

_ARBI_COLORS = {
    "intent_applied":     "#a8d8a4",  # saturated green
    "rl_fallback":        "#d0d0d0",  # visible gray
    "emergency_override": "#f4a4a4",  # saturated pink-red
}
_ARBI_LABELS = {
    "intent_applied":     "Intent applied",
    "rl_fallback":        "RL fallback",
    "emergency_override": "Emergency",
}
CENTER_IDX = 3  # heading action index for no-turn

# Shared visual style — mirrors generate_trajectory_overlays.py
_COLOR_OS       = "#ff7f0e"
_COLOR_TS       = "#b9c0ca"
_ICON_INTERVAL_S = 60.0
_LOA_M          = 30.0
_BEAM_M         = 16.0
_SHIP_SCALE     = 1.08

try:
    from ..episode_overlay_tools import animate_ship as _animate_ship
except Exception:
    _animate_ship = None



def _resolve_seed_dir(eval_dir: Path) -> Path:
    """Resolve eval_dir to the seed directory that contains llm_intent_log.csv."""
    eval_dir = eval_dir.resolve()
    direct = eval_dir / "llm_intent_log.csv"
    nested = eval_dir / "seed_0" / "llm_intent_log.csv"

    if direct.exists():
        return eval_dir
    if nested.exists():
        return eval_dir / "seed_0"

    raise FileNotFoundError(
        f"Could not find llm_intent_log.csv in {eval_dir} or {eval_dir / 'seed_0'}"
    )


def _find_episode_npz(seed_dir: Path, npz_path: Optional[Path]) -> Path:
    """Pick episode NPZ explicitly or default to first file in episode_histories."""
    if npz_path is not None:
        p = npz_path.resolve()
        if not p.exists():
            raise FileNotFoundError(f"NPZ file not found: {p}")
        return p

    hist_dir = seed_dir / "episode_histories"
    if not hist_dir.exists():
        raise FileNotFoundError(f"episode_histories not found in {seed_dir}")

    npzs = sorted(hist_dir.glob("*.npz"))
    if not npzs:
        raise FileNotFoundError(f"No NPZ files found in {hist_dir}")
    return npzs[0]


def _find_all_episode_npz(seed_dir: Path) -> List[Path]:
    """Return all NPZ files in episode_histories sorted by filename."""
    hist_dir = seed_dir / "episode_histories"
    if not hist_dir.exists():
        raise FileNotFoundError(f"episode_histories not found in {seed_dir}")
    npzs = sorted(hist_dir.glob("*.npz"))
    if not npzs:
        raise FileNotFoundError(f"No NPZ files found in {hist_dir}")
    return npzs


def _load_npz_history(npz_path: Path) -> Dict:
    """Load NPZ history arrays and normalize scalar-ish metadata fields."""
    data = np.load(npz_path, allow_pickle=True)

    hist: Dict = {}
    for key in data.files:
        val = data[key]
        if key in ("case", "seed"):
            hist[key] = int(np.asarray(val).ravel()[0])
        elif key in ("baseline", "checkpoint"):
            hist[key] = str(np.asarray(val).ravel()[0]) if np.asarray(val).size > 0 else ""
        elif key in ("final_waypoint_x_nmi", "final_waypoint_y_nmi"):
            arr = np.asarray(val).ravel()
            hist[key] = None if arr.size == 0 else float(arr[0])
        else:
            hist[key] = np.asarray(val)

    required = ("t", "X_all", "pair_dist", "pair_dcpa", "pair_tcpa", "pair_risk")
    missing = [k for k in required if k not in hist]
    if missing:
        raise KeyError(f"Missing keys in NPZ: {missing}")

    return hist


def _read_llm_log(csv_path: Path) -> List[Dict]:
    """Read llm_intent_log.csv as typed rows."""
    rows: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "episode_idx":    int(float(row.get("episode_idx", "0"))),
                    "episode_seed":   int(float(row.get("episode_seed", "0"))),
                    "time_s": float(row.get("time_s", "nan")),
                    "target_id": int(float(row.get("target_id", "0"))),
                    # CSV stores nmi; convert to meters for plotting consistency.
                    "range_m": float(row.get("range_nmi", "nan")) * NMI,
                    "abs_dcpa_m": float(row.get("abs_dcpa_nmi", "nan")) * NMI,
                    "tcpa_s": float(row.get("tcpa_s", "nan")),
                    "risk": float(row.get("risk", "nan")),
                    "colreg_rule": row.get("colreg_rule", ""),
                    "encounter_phase": row.get("encounter_phase", ""),
                    "llm_call_status": row.get("llm_call_status", ""),
                    "llm_response": row.get("llm_response", ""),
                    "parsed_kdir": int(float(row.get("parsed_kdir", "0"))),
                }
            )
    if not rows:
        raise ValueError(f"No rows in {csv_path}")
    return rows


def _extract_action_text(response: str) -> str:
    """Extract concise action string from the LLM response text."""
    s = response.strip()
    lower = s.lower()
    if "action:" in lower:
        idx = lower.find("action:")
        action_part = s[idx + len("action:"):].strip()
        lower_action = action_part.lower()
        if "explanation:" in lower_action:
            j = lower_action.find("explanation:")
            action_part = action_part[:j].strip()
        return action_part

    # Fallback to compact heuristic if Action: is not present
    lower = s.lower()
    if "starboard" in lower:
        return "Give-way turn starboard"
    if "port" in lower:
        return "Give-way turn port"
    if "stand on" in lower:
        return "Stand on, no action"
    if "continue" in lower:
        return "Continue current maneuver"
    return "(unparsed action)"


def _aggregate_primary_llm_rows(rows: List[Dict]) -> List[Dict]:
    """
    For each LLM call time, keep one representative row.

    Selection rule:
    - highest risk row at that time
    - tie-break by smaller range
    """
    by_t: Dict[float, List[Dict]] = {}
    for row in rows:
        by_t.setdefault(row["time_s"], []).append(row)

    out: List[Dict] = []
    for t in sorted(by_t.keys()):
        group = by_t[t]
        group_sorted = sorted(
            group,
            key=lambda r: (
                -(r["risk"] if np.isfinite(r["risk"]) else -1e9),
                (r["range_m"] if np.isfinite(r["range_m"]) else 1e9),
            ),
        )
        best = dict(group_sorted[0])
        best["action_text"] = _extract_action_text(best["llm_response"])
        out.append(best)
    return out


def _first_all_target_post_cpa_time(rows: List[Dict]) -> Optional[float]:
    """
    Return the first LLM-call time where all active-target rows are post-CPA.
    """
    by_t: Dict[float, List[Dict]] = {}
    for row in rows:
        by_t.setdefault(float(row["time_s"]), []).append(row)

    for t in sorted(by_t.keys()):
        group = by_t[t]
        phases = [(r.get("encounter_phase", "") or "").lower() for r in group]
        if phases and all(p == "post_cpa" for p in phases):
            return t
    return None


# ── Stage-2 helpers: execution log + call scoring ─────────────────────────────

def _episode_idx_from_npz(npz_path: Path) -> int:
    """Extract episode_idx from filenames like case1_seed0_ep000.npz."""
    m = re.search(r"ep(\d+)", npz_path.stem)
    return int(m.group(1)) if m else 0


def _load_exec_log(seed_dir: Path, episode_seed: int, episode_idx: int) -> List[Dict]:
    """Load execution_log.csv rows for a single episode, sorted by time_s."""
    p = seed_dir / "execution_log.csv"
    if not p.exists():
        return []
    rows: List[Dict] = []
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(float(row["episode_seed"])) != episode_seed:
                continue
            if int(float(row["episode_idx"])) != episode_idx:
                continue
            rows.append({
                "time_s":           float(row["time_s"]),
                "proposed_heading": int(float(row["proposed_heading_idx"])),
                "applied_heading":  int(float(row["applied_heading_idx"])),
                "arbitration_mode": row.get("arbitration_mode") or "rl_fallback",
            })
    return sorted(rows, key=lambda r: r["time_s"])


def _load_call_scores_for_episode(
    seed_dir: Path,
    episode_seed: int,
    episode_idx: int,
    fallback_llm_rows: List[Dict],
) -> List[Dict]:
    """
    Return per-call scoring dicts for one episode.

    Uses results_llm_reliability/llm_call_scores.csv when available.
    Falls back to a reduced view built from llm_intent_log rows.
    Each returned dict has: time_s, parsed_kdir, expected_kdir, valid_call, is_correct.
    """
    scores_path = seed_dir / "results_llm_reliability" / "llm_call_scores.csv"
    if scores_path.exists():
        raw: List[Dict] = []
        with open(scores_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(float(row.get("episode_seed", episode_seed))) != episode_seed:
                    continue
                if int(float(row.get("episode_idx", episode_idx))) != episode_idx:
                    continue
                exp_raw = row.get("expected_single", "").strip()
                raw.append({
                    "time_s":          float(row["time_s"]),
                    "parsed_kdir":     int(float(row.get("parsed_kdir", "0"))) if row.get("parsed_kdir") else 0,
                    "expected_kdir":   int(float(exp_raw)) if exp_raw not in ("", "None", "nan") else None,
                    "valid_call":      int(float(row.get("valid_call", "0"))),
                    "is_correct":      int(float(row.get("is_correct", "0"))) if row.get("is_correct") not in ("", "None") else None,
                    "colreg_rule":     row.get("colreg_rule", ""),
                    "encounter_phase": row.get("encounter_phase", ""),
                })
        # Deduplicate per time_s: keep the first scored entry
        by_t: Dict[float, List[Dict]] = {}
        for r in raw:
            by_t.setdefault(r["time_s"], []).append(r)
        deduped: List[Dict] = []
        for t in sorted(by_t.keys()):
            grp = by_t[t]
            scored = [r for r in grp if r["is_correct"] is not None]
            deduped.append(scored[0] if scored else grp[0])
        return deduped

    # Fallback: derive from llm_intent_log rows filtered to this episode
    ep_rows = [
        r for r in fallback_llm_rows
        if r["episode_seed"] == episode_seed and r["episode_idx"] == episode_idx
    ]
    # Aggregate to primary row per time (highest risk, same as _aggregate_primary_llm_rows)
    primary = _aggregate_primary_llm_rows(ep_rows)
    return [
        {
            "time_s":          r["time_s"],
            "parsed_kdir":     r["parsed_kdir"],
            "expected_kdir":   None,
            "valid_call":      1 if r.get("llm_call_status", "").lower() == "success" else 0,
            "is_correct":      None,
            "colreg_rule":     r.get("colreg_rule", ""),
            "encounter_phase": r.get("encounter_phase", ""),
        }
        for r in primary
    ]


def _plot_panel_b_timeline(
    ax,
    exec_rows: List[Dict],
    call_rows: List[Dict],
    post_cpa_start_t: Optional[float],
) -> None:
    """
    Panel B: four overlaid layers on a shared direction axis (−1 / 0 / +1).

    Layer 1 (background): arbitration mode spans coloured by mode.
    Layer 2 (solid line):  applied heading direction at every control step.
    Layer 3 (dashed line): LLM parsed intent between query points.
    Layer 4 (diamonds):    COLREG expected action at each LLM query time,
                           coloured green/orange/red by correctness.
    """
    from matplotlib.lines import Line2D

    if not exec_rows and not call_rows:
        ax.text(0.5, 0.5, "No execution or LLM data for this episode.",
                ha="center", transform=ax.transAxes)
        return

    # ── Layer 1: arbitration mode spans ──────────────────────────────────────
    if exec_rows:
        t_end = exec_rows[-1]["time_s"] + 2.0
        run_mode  = exec_rows[0]["arbitration_mode"]
        run_start = exec_rows[0]["time_s"]
        spans: List[Tuple[float, float, str]] = []
        for r in exec_rows[1:]:
            if r["arbitration_mode"] != run_mode:
                spans.append((run_start, r["time_s"], run_mode))
                run_mode  = r["arbitration_mode"]
                run_start = r["time_s"]
        spans.append((run_start, t_end, run_mode))

        for (t0, t1, mode) in spans:
            color = _ARBI_COLORS.get(mode, "#ffffff")
            ax.axvspan(t0, t1, color=color, alpha=0.40, linewidth=0, zorder=0)

        # Compact mode label at top of each span
        for (t0, t1, mode) in spans:
            label = _ARBI_LABELS.get(mode, mode)
            ax.text(
                (t0 + t1) / 2, 1.0, label,
                ha="center", va="bottom", fontsize=7, color="#444444",
                style="italic", transform=ax.get_xaxis_transform(), clip_on=True,
            )

    # ── Layer 2: applied heading direction (solid continuous) ────────────────
    if exec_rows:
        t_exec = np.array([r["time_s"] for r in exec_rows])
        applied_dir = np.array(
            [np.sign(CENTER_IDX - r["applied_heading"]) for r in exec_rows],
            dtype=float,
        )
        ax.step(t_exec, applied_dir, where="post", color="#1f77b4",
                lw=2.2, ls="-", label="Applied action", zorder=3)

    # ── Layer 3: LLM parsed intent (dashed step) ─────────────────────────────
    if call_rows:
        t_llm   = np.array([r["time_s"]     for r in call_rows])
        kdir_lm = np.array([r["parsed_kdir"] for r in call_rows], dtype=float)
        ax.step(t_llm, kdir_lm, where="post", color="#333333",
                lw=1.5, ls="--", label="LLM parsed intent", zorder=4)

    # ── Layer 4: diamonds at LLM query times ─────────────────────────────────
    if call_rows:
        # Marker shapes differ for grayscale distinguishability:
        #   correct → ◆ D,  baseline-inconsistent → ● o,  invalid → ✕ X
        for r in call_rows:
            t     = r["time_s"]
            y_pos = r["expected_kdir"] if r["expected_kdir"] is not None else r["parsed_kdir"]
            vc      = r["valid_call"]
            correct = r["is_correct"]

            if not vc:
                color, mkr = "#d62728", "X"   # invalid / provider failure
            elif correct is None:
                color, mkr = "#2ca02c", "D"   # no scoring info: treat as valid
            elif correct:
                color, mkr = "#2ca02c", "D"   # correct vs. baseline
            else:
                color, mkr = "#ff7f0e", "o"   # baseline-inconsistent

            ax.scatter(t, y_pos, s=90, marker=mkr, color=color,
                       edgecolors="white", linewidths=0.8, zorder=5)

    # ── Faint verticals at query times ───────────────────────────────────────
    for r in call_rows:
        ax.axvline(r["time_s"], color="gray", lw=0.5, alpha=0.2, zorder=1)

    # ── Post-CPA boundary ────────────────────────────────────────────────────
    if post_cpa_start_t is not None:
        ax.axvline(post_cpa_start_t, color="#666666", lw=1.2, ls="--", alpha=0.9, zorder=4,
                   label="All-target post-CPA")

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color="#333333", lw=1.5, ls="--", label="LLM parsed intent"),
        Line2D([0], [0], color="#1f77b4", lw=2.2, ls="-",  label="Applied action"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#2ca02c",
               markeredgecolor="white", markersize=8, label="LLM correct vs. baseline"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#ff7f0e",
               markeredgecolor="white", markersize=8, label="Valid, baseline-inconsistent"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#d62728",
               markeredgecolor="white", markersize=8, label="Invalid / failure"),
        mpatches.Patch(facecolor=_ARBI_COLORS["intent_applied"],     alpha=0.7, label="Arb: intent applied"),
        mpatches.Patch(facecolor=_ARBI_COLORS["rl_fallback"],        alpha=0.7, label="Arb: RL fallback"),
        mpatches.Patch(facecolor=_ARBI_COLORS["emergency_override"], alpha=0.7, label="Arb: emergency"),
    ]
    if post_cpa_start_t is not None:
        legend_handles.append(
            Line2D([0], [0], color="#666666", lw=1.2, ls="--", label="All-target post-CPA")
        )

    ax.legend(handles=legend_handles, loc="best", fontsize=8, ncol=2, framealpha=0.88)
    ax.set_title("B. LLM Intent and Controller-Execution Timeline",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Maneuver direction")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Port (\u22121)", "Maintain (0)", "Starboard (+1)"])
    ax.set_ylim(-1.6, 1.5)
    ax.margins(x=0.01)
    ax.grid(alpha=0.20)


def _closest_target_metrics_over_time(hist: Dict, own_idx: int = 0) -> Dict[str, np.ndarray]:
    """Build encounter metrics vs time for the closest target at each step."""
    t = np.asarray(hist["t"], dtype=float)
    pair_dist = np.asarray(hist["pair_dist"], dtype=float)
    pair_dcpa = np.asarray(hist["pair_dcpa"], dtype=float)
    pair_tcpa = np.asarray(hist["pair_tcpa"], dtype=float)
    pair_risk = np.asarray(hist["pair_risk"], dtype=float)

    n_steps = t.shape[0]
    range_m = np.full(n_steps, np.nan, dtype=float)
    abs_dcpa_m = np.full(n_steps, np.nan, dtype=float)
    tcpa_s = np.full(n_steps, np.nan, dtype=float)
    risk = np.full(n_steps, np.nan, dtype=float)

    for s in range(n_steps):
        drow = pair_dist[s, own_idx].astype(float).copy()
        drow[own_idx] = np.inf
        if not np.any(np.isfinite(drow)):
            continue
        j = int(np.nanargmin(drow))
        range_m[s] = drow[j]
        abs_dcpa_m[s] = abs(float(pair_dcpa[s, own_idx, j]))
        tcpa_s[s] = float(pair_tcpa[s, own_idx, j])
        risk[s] = float(pair_risk[s, own_idx, j])

    return {
        "t": t,
        "range_m": range_m,
        "abs_dcpa_m": abs_dcpa_m,
        "tcpa_s": tcpa_s,
        "risk": risk,
    }


def _plot_panel_a(ax, hist: Dict):
    """Panel A: trajectory overlay — style matches generate_trajectory_overlays.py."""
    X_all  = np.asarray(hist["X_all"], dtype=float)
    t_arr  = np.asarray(hist["t"],    dtype=float)
    n_agents = X_all.shape[1]

    # Target paths — gray, low emphasis
    for j in range(1, n_agents):
        xj, yj = X_all[:, j, 0], X_all[:, j, 1]
        ax.plot(xj, yj, color=_COLOR_TS, lw=1.3, alpha=0.70,
                label="Target ship(s)" if j == 1 else None, zorder=1)

    # Ownship path — orange solid
    x0, y0 = X_all[:, 0, 0], X_all[:, 0, 1]
    ax.plot(x0, y0, color=_COLOR_OS, lw=1.8, zorder=3, label="Ownship")

    # Ship icons at regular time intervals
    if _animate_ship is not None and t_arr[-1] > _ICON_INTERVAL_S:
        icon_times = np.arange(_ICON_INTERVAL_S, t_arr[-1], _ICON_INTERVAL_S)
        for t_icon in icon_times:
            idx = int(np.argmin(np.abs(t_arr - t_icon)))
            for j in range(1, n_agents):
                p = _animate_ship(
                    float(X_all[idx, j, 0]), float(X_all[idx, j, 1]),
                    float(X_all[idx, j, 2]),
                    _LOA_M * _SHIP_SCALE, _BEAM_M * _SHIP_SCALE,
                    cpa=0.0, color=_COLOR_TS, ax=ax,
                )
                if p is not None:
                    p.set_alpha(0.25)
            _animate_ship(
                float(X_all[idx, 0, 0]), float(X_all[idx, 0, 1]),
                float(X_all[idx, 0, 2]),
                _LOA_M * _SHIP_SCALE, _BEAM_M * _SHIP_SCALE,
                cpa=0.0, color=_COLOR_OS, ax=ax,
            )

    # Start / goal markers
    ax.scatter(x0[0], y0[0], s=50, color="gold", edgecolors="black",
               linewidth=0.5, zorder=6, label="Ownship start")
    ax.scatter(x0[-1], y0[-1], s=55, marker="X", color="darkgreen",
               zorder=6, label="Goal")

    fx = hist.get("final_waypoint_x_nmi", None)
    fy = hist.get("final_waypoint_y_nmi", None)
    if fx is not None and fy is not None and np.isfinite(fx) and np.isfinite(fy):
        wx, wy = float(fx) * NMI, float(fy) * NMI
        ax.scatter(wx, wy, s=120, marker="*", color="darkgreen", zorder=6, label="Waypoint")
        # 200 m success zone drawn in data coordinates
        theta = np.linspace(0, 2 * np.pi, 180)
        ax.plot(wx + 200.0 * np.cos(theta), wy + 200.0 * np.sin(theta),
                color="red", lw=1.2, ls=":", zorder=5, label="200 m success zone")
    else:
        # fallback: draw zone around ownship final position
        theta = np.linspace(0, 2 * np.pi, 180)
        ax.plot(x0[-1] + 200.0 * np.cos(theta), y0[-1] + 200.0 * np.sin(theta),
                color="red", lw=1.2, ls=":", zorder=5, label="200 m success zone")

    ax.set_title("A. Representative Vessel Trajectory", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    # datalim: keeps axes panel size fixed (consistent height) while keeping ship icon geometry correct
    ax.set_aspect("equal", adjustable="datalim")


def _plot_panel_b(ax, series: Dict, llm_times: List[float]):
    """Panel B: time-series encounter state for closest target over time."""
    t = series["t"]
    ax.plot(t, series["range_m"], label="Range (m)", color="#1f77b4", lw=1.8)
    ax.plot(t, series["abs_dcpa_m"], label="|DCPA| (m)", color="#d62728", lw=1.8)
    ax.plot(t, series["tcpa_s"] / 100.0, label="TCPA / 100 (s)", color="#9467bd", lw=1.6, ls="--")

    # Vertical markers at LLM query times
    for tt in llm_times:
        ax.axvline(tt, color="gray", lw=0.7, alpha=0.3)

    ax.set_title("B. Encounter-State Time Series (Closest Target)", fontsize=19.2, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Meters / scaled seconds")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)


def _plot_panel_c(ax, primary_rows: List[Dict], post_cpa_start_t: Optional[float]):
    """Panel C: Clean LLM/rule intent timeline (no full response text annotations)."""
    from matplotlib.lines import Line2D

    times = np.array([r["time_s"] for r in primary_rows], dtype=float)
    kdir = np.array([r["parsed_kdir"] for r in primary_rows], dtype=int)
    # Main transition line as a step plot.
    ax.step(times, kdir, where="post", color="black", lw=1.4, alpha=0.9)

    # Rule-category marker style map (used only while approaching).
    rule_style = {
        # Open diamond is easier to read against black step line.
        "13": {"color": "#8c564b", "marker": "D", "label": "Rule 13 (Overtaking)", "open": True},
        "14": {"color": "#d62728", "marker": "D", "label": "Rule 14 (Head-on)"},
        "15.1": {"color": "#ff7f0e", "marker": "^", "label": "Rule 15.1 (Crossing give-way)"},
        "15.2": {"color": "#1f77b4", "marker": "v", "label": "Rule 15.2 (Crossing stand-on)"},
        "other": {"color": "#9467bd", "marker": "o", "label": "Other rule"},
    }

    def _rule_key(rule_text: str) -> str:
        txt = (rule_text or "").lower()
        if "15.1" in txt:
            return "15.1"
        if "15.2" in txt:
            return "15.2"
        if "14" in txt:
            return "14"
        if "13" in txt:
            return "13"
        return "other"

    # Keep track of legend entries we have already shown.
    shown_labels = set()

    for r in primary_rows:
        t = float(r["time_s"])
        y = int(r["parsed_kdir"])
        phase = (r.get("encounter_phase", "") or "").lower()
        status = (r.get("llm_call_status", "") or "").lower()

        if phase == "post_cpa":
            color = "#7f7f7f"
            marker = "o" if status == "success" else "x"
            legend_label = "Post-CPA"
            marker_kwargs = {"facecolors": color, "edgecolors": color}
        else:
            k = _rule_key(r.get("colreg_rule", ""))
            style = rule_style[k]
            color = style["color"]
            marker = style["marker"] if status == "success" else "x"
            legend_label = style["label"]
            if status == "success" and style.get("open", False):
                marker_kwargs = {"facecolors": "none", "edgecolors": color, "linewidths": 1.3}
            else:
                marker_kwargs = {"facecolors": color, "edgecolors": color}

        label = legend_label if legend_label not in shown_labels else None
        if marker == "x":
            ax.scatter(t, y, s=70, color=color, marker=marker, zorder=4, label=label)
        else:
            ax.scatter(t, y, s=70, marker=marker, zorder=4, label=label, **marker_kwargs)
        if label is not None:
            shown_labels.add(legend_label)

    # Mark transition into all-target post-CPA region.
    if post_cpa_start_t is not None:
        ax.axvline(post_cpa_start_t, color="#666666", lw=1.2, ls="--", alpha=0.9)

    # Legend-only semantics (no text labels on points).
    custom_handles = [
        Line2D([0], [0], color="black", lw=1.4, label="parsed_kdir (step)"),
        Line2D([0], [0], marker="D", markerfacecolor="none", markeredgecolor="#8c564b", color="none", markersize=7, label="Rule 13"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#d62728", markeredgecolor="#d62728", markersize=7, label="Rule 14"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#ff7f0e", markeredgecolor="#ff7f0e", markersize=7, label="Rule 15.1"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="#1f77b4", markeredgecolor="#1f77b4", markersize=7, label="Rule 15.2"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#7f7f7f", markeredgecolor="#7f7f7f", markersize=7, label="Post-CPA"),
        Line2D([0], [0], color="#666666", lw=1.2, ls="--", label="All-target post-CPA begins"),
    ]

    ax.set_title("C. LLM / Rule Intent Timeline", fontsize=19.2, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Parsed maneuver intent")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Port (-1)", "Stand-on (0)", "Starboard (+1)"])
    ax.grid(alpha=0.25)
    ax.legend(handles=custom_handles, loc="best", fontsize=8)


def _format_time_value(t: float) -> str:
    """Format time compactly for table windows."""
    if abs(t - round(t)) < 1e-6:
        return f"{int(round(t))}"
    return f"{t:.1f}"


def _intent_text_from_row(row: Dict) -> str:
    """Map one timeline row to a concise interpretation phrase."""
    phase = (row.get("encounter_phase", "") or "").lower()
    kdir = int(row.get("parsed_kdir", 0))
    rule = (row.get("colreg_rule", "") or "").lower()

    if phase == "post_cpa":
        return "Post-CPA, no new maneuver"
    if kdir == 1:
        return "Give-way / starboard maneuver"
    if kdir == -1:
        return "Give-way / port maneuver"
    if "15.2" in rule:
        return "Crossing stand-on / maintain course"
    if "13" in rule or "14" in rule:
        return "Initial overtaking / crossing assessment"
    return "Stand-on / no new maneuver"


def _rule_text_from_row(row: Dict) -> str:
    """Return concise 'Rule #: encounter type' text for the summary table."""
    phase = (row.get("encounter_phase", "") or "").lower()
    if phase == "post_cpa":
        return "Post-CPA"

    rule = (row.get("colreg_rule", "") or "").strip()
    if not rule or rule == "N/A":
        return "Rule ?: Unknown encounter"
    return rule


def _executed_action_from_row(row: Dict) -> str:
    """Return concise action label from parsed maneuver intent."""
    kdir = int(row.get("parsed_kdir", 0))
    if kdir > 0:
        return "Turn starboard"
    if kdir < 0:
        return "Turn port"
    return "Stand on / no new maneuver"


def _build_intent_summary_rows(primary_rows: List[Dict]) -> List[List[str]]:
    """
    Build collapsed time-window rows for Panel D table.

    Columns:
    - Time window
    - LLM / rule interpretation
    - Parsed intent
    - Executed action
    """
    if not primary_rows:
        return []

    rows = sorted(primary_rows, key=lambda r: float(r["time_s"]))

    segments: List[List[str]] = []
    seg_start_idx = 0

    def _row_signature(r: Dict) -> tuple:
        # Collapse by intent state and high-level interpretation text.
        return (int(r.get("parsed_kdir", 0)), _intent_text_from_row(r))

    for i in range(1, len(rows)):
        if _row_signature(rows[i]) != _row_signature(rows[seg_start_idx]):
            start = float(rows[seg_start_idx]["time_s"])
            end = float(rows[i]["time_s"])
            parsed = int(rows[seg_start_idx].get("parsed_kdir", 0))
            segments.append([
                f"{_format_time_value(start)}-{_format_time_value(end)} s",
                _rule_text_from_row(rows[seg_start_idx]),
                f"{parsed:+d}" if parsed != 0 else "0",
                _executed_action_from_row(rows[seg_start_idx]),
            ])
            seg_start_idx = i

    # Final open-ended segment
    start = float(rows[seg_start_idx]["time_s"])
    parsed = int(rows[seg_start_idx].get("parsed_kdir", 0))
    segments.append([
        f"{_format_time_value(start)}+ s",
        _rule_text_from_row(rows[seg_start_idx]),
        f"{parsed:+d}" if parsed != 0 else "0",
        _executed_action_from_row(rows[seg_start_idx]),
    ])

    return segments


def _plot_panel_d_table(ax, summary_rows: List[List[str]]):
    """Panel D: compact table of time-window intent summary."""
    ax.axis("off")
    ax.set_title("D. Time-window intent summary", fontsize=19.2, fontweight="bold", loc="center", pad=8)

    if not summary_rows:
        ax.text(0.01, 0.65, "No LLM intent rows available.", fontsize=10)
        return

    table = ax.table(
        cellText=summary_rows,
        colLabels=["Time window", "Rule: encounter type", "Parsed intent", "Executed action"],
        colLoc="left",
        cellLoc="left",
        colWidths=[0.18, 0.36, 0.14, 0.27],
        bbox=[0.01, 0.02, 0.98, 0.90],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)

    # Header emphasis + subtle row lines for readability.
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f2f2")
        cell.set_edgecolor("#d9d9d9")
        cell.set_linewidth(0.5)


def make_llm_intent_confirmation_plot(
    seed_dir: Path,
    npz_path: Optional[Path],
    output_path: Path,
) -> Path:
    """Create 2-panel LLM-guided closed-loop execution figure."""
    csv_path = seed_dir / "llm_intent_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"llm_intent_log.csv not found: {csv_path}")

    episode_npz  = _find_episode_npz(seed_dir, npz_path)
    hist         = _load_npz_history(episode_npz)
    episode_idx  = _episode_idx_from_npz(episode_npz)
    episode_seed = int(hist.get("seed", 0))

    all_llm_rows  = _read_llm_log(csv_path)
    # Filter to this episode for the existing per-panel helpers
    ep_llm_rows   = [
        r for r in all_llm_rows
        if r["episode_seed"] == episode_seed and r["episode_idx"] == episode_idx
    ]
    primary_rows  = _aggregate_primary_llm_rows(ep_llm_rows)
    post_cpa_t    = _first_all_target_post_cpa_time(ep_llm_rows)

    exec_rows  = _load_exec_log(seed_dir, episode_seed, episode_idx)
    call_rows  = _load_call_scores_for_episode(seed_dir, episode_seed, episode_idx, all_llm_rows)

    case_num = hist.get("case", "?")
    seed     = hist.get("seed", "?")

    fig = plt.figure(figsize=(14, 11))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.2], hspace=0.38)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    _plot_panel_a(ax_a, hist)
    _plot_panel_b_timeline(ax_b, exec_rows, call_rows, post_cpa_t)

    fig.suptitle(
        f"LLM-Guided Closed-Loop Execution \u2014 Case {case_num}, Episode {episode_idx}",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(left=0.09, right=0.96, top=0.92, bottom=0.07, hspace=0.40)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a synchronized trajectory + intent confirmation figure from eval outputs"
    )
    p.add_argument(
        "--eval_dir",
        type=Path,
        required=True,
        help="Path to seed dir (contains llm_intent_log.csv) or parent eval dir (contains seed_0)",
    )
    p.add_argument(
        "--npz_path",
        type=Path,
        default=None,
        help="Optional explicit NPZ history file to plot. Defaults to first file in episode_histories/",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path for single-episode mode. "
            "Defaults to <seed_dir>/results_eval_intent/llm_intent_confirmation.png"
        ),
    )
    p.add_argument(
        "--all_episodes",
        action="store_true",
        help=(
            "Generate one confirmation PNG for each NPZ in episode_histories/. "
            "Outputs to <seed_dir>/results_eval_intent/."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_dir = _resolve_seed_dir(args.eval_dir)
    results_dir = seed_dir / "results_eval_intent"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.all_episodes:
        if args.npz_path is not None:
            raise ValueError("--all_episodes cannot be combined with --npz_path")
        if args.output is not None:
            raise ValueError("--all_episodes cannot be combined with --output")

        npz_files = _find_all_episode_npz(seed_dir)
        for npz in npz_files:
            out_name = f"llm_intent_confirmation_{npz.stem}.png"
            out_path = results_dir / out_name
            out = make_llm_intent_confirmation_plot(
                seed_dir=seed_dir,
                npz_path=npz,
                output_path=out_path,
            )
            print(f"Saved: {out}")
        print(f"Generated {len(npz_files)} file(s) in: {results_dir}")
        return

    output_path = args.output or (results_dir / "llm_intent_confirmation.png")
    out = make_llm_intent_confirmation_plot(
        seed_dir=seed_dir,
        npz_path=args.npz_path,
        output_path=output_path,
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
