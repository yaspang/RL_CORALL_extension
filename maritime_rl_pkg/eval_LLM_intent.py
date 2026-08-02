"""
Create a synchronized Stage-1 "trajectory + intent" confirmation plot.

Panel A: Trajectory / overlay (ownship + target ship paths)
Panel B: Encounter-state time series (Range, DCPA, TCPA, Risk)
Panel C: LLM/rule-intent timeline (parsed_kdir + rule + phase)

Usage:
    python -m maritime_rl_pkg.eval_LLM_intent \
      --eval_dir "stage1_llm_case7_real_api_interval10/seed_0"

        # Generate one figure per NPZ episode in episode_histories/
        python -m maritime_rl_pkg.eval_LLM_intent \
            --eval_dir "stage1_llm_case7_real_api_interval10/seed_0" \
            --all_episodes

The script accepts either:
- a seed directory containing llm_intent_log.csv, or
- a parent eval directory containing seed_0/llm_intent_log.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

NMI = 1852.0


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
    """Panel A: trajectory overlay for ownship and all target ships."""
    X_all = np.asarray(hist["X_all"], dtype=float)
    n_agents = X_all.shape[1]

    # Ownship path
    x0 = X_all[:, 0, 0]
    y0 = X_all[:, 0, 1]
    ax.plot(x0, y0, color="#ff7f0e", lw=2.3, ls="--", label="Ownship")

    # Target paths
    for j in range(1, n_agents):
        xj = X_all[:, j, 0]
        yj = X_all[:, j, 1]
        ax.plot(xj, yj, color="#1f77b4", lw=1.4, alpha=0.75, label="Target" if j == 1 else None)
        # Target start/end markers: blue circle for start, blue x for finish.
        ax.scatter(xj[0], yj[0], s=28, color="#1f77b4", marker="o", zorder=4,
                   label="Target start" if j == 1 else None)
        ax.scatter(xj[-1], yj[-1], s=38, color="#1f77b4", marker="x", linewidths=1.4,
                   zorder=4, label="Target end" if j == 1 else None)

    # Start / end / waypoint markers
    ax.scatter(x0[0], y0[0], s=80, color="orange", label="Start", zorder=4)
    ax.scatter(x0[-1], y0[-1], s=85, marker="o", color="darkgreen", edgecolors="black", zorder=5, label="End")

    fx = hist.get("final_waypoint_x_nmi", None)
    fy = hist.get("final_waypoint_y_nmi", None)
    if fx is not None and fy is not None and np.isfinite(fx) and np.isfinite(fy):
        # Waypoint metadata is stored in nmi; convert to meters for this plot.
        ax.scatter(float(fx) * NMI, float(fy) * NMI, s=140, marker="X", color="darkgreen", zorder=6, label="Target endpoint")

    ax.set_title("A. Trajectory / Overlay", fontsize=19.2, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal", adjustable="box")


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
    """Create 3-panel synchronized confirmation plot."""
    csv_path = seed_dir / "llm_intent_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"llm_intent_log.csv not found: {csv_path}")

    episode_npz = _find_episode_npz(seed_dir, npz_path)
    hist = _load_npz_history(episode_npz)
    rows = _read_llm_log(csv_path)
    primary_rows = _aggregate_primary_llm_rows(rows)
    post_cpa_start_t = _first_all_target_post_cpa_time(rows)
    series = _closest_target_metrics_over_time(hist)
    llm_times = [r["time_s"] for r in primary_rows]
    summary_rows = _build_intent_summary_rows(primary_rows)

    case_num = hist.get("case", "?")
    seed = hist.get("seed", "?")

    fig = plt.figure(figsize=(14, 17))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1.0, 1.15, 0.95], hspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[3, 0])

    _plot_panel_a(ax_a, hist)
    _plot_panel_b(ax_b, series, llm_times)
    _plot_panel_c(ax_c, primary_rows, post_cpa_start_t)
    _plot_panel_d_table(ax_d, summary_rows)

    fig.suptitle(
        f"Stage-1 LLM Intent Confirmation | Case {case_num} | Seed {seed}",
        fontsize=30,
        fontweight="bold",
        y=0.985,
    )

    # Keep all panels visually centered and aligned in the output figure.
    fig.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.05)

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
