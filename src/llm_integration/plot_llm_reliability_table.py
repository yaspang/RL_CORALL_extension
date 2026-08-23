"""
Generate reliability summary table and stacked bar chart from LLM evaluation results.

Usage
-----
  python -m src.llm_integration.plot_llm_reliability_table \\
      --eval_dirs \\
          "results_llmapi_case1_interval10/seed_0" \\
          "results_llmapi_case6_interval10/seed_0" \\
          "results_llmapi_case18_interval10/seed_0" \\
      --case_labels 1 6 18 \\
      --output_dir "1_Visualizations/llm_reliability_summary"

Key arguments
-------------
  --eval_dirs PATH [...]   One or more seed dirs (or parent dirs with seed_0/) (required)
  --case_labels STR [...]  Labels for each eval dir (default: derived from dir name)
  --output_dir STR         Output directory for figures

Requires
--------
  Each seed_dir must contain results_llm_reliability/llm_reliability_summary.json
  (produced by eval_llm_reliability.py) and policy_eval_summary.json.

Outputs
-------
  llm_reliability_table.png     IEEE-style table: Valid Intent / Strict Acc. /
                                  Missed / Unnecessary / Collision / Success
  llm_reliability_stacked_bar.png  Valid Intent vs No Valid Intent per case
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── colour palette (consistent with existing overlay plots) ───────────────────
_COL_VALID_CORRECT   = "#2ca02c"   # green
_COL_VALID_INCORRECT = "#ff7f0e"   # orange
_COL_INVALID         = "#d62728"   # red
_COL_FALLBACK        = "#9467bd"   # purple


def _pct(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    return f"{v * 100:.1f}%"


def _f1(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    return f"{v:.3f}"


def _load_reliability(seed_dir: Path) -> Dict:
    p = seed_dir / "results_llm_reliability" / "llm_reliability_summary.json"
    if not p.exists():
        raise FileNotFoundError(
            f"llm_reliability_summary.json not found at {p}\n"
            "Run eval_llm_reliability first."
        )
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _load_policy(seed_dir: Path) -> Dict:
    p = seed_dir / "policy_eval_summary.json"
    if not p.exists():
        raise FileNotFoundError(f"policy_eval_summary.json not found at {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _resolve(eval_dir: str) -> Path:
    p = Path(eval_dir)
    # accept parent dir that contains seed_0/
    if (p / "seed_0" / "policy_eval_summary.json").exists():
        return p / "seed_0"
    return p


# ── Figure 1: IEEE-style table ────────────────────────────────────────────────

def _make_table_figure(rows: List[Dict], output_path: Path) -> None:
    col_labels = [
        "Case",
        "Valid Intent",
        "Strict Acc.",
        "Missed",
        "Unnecessary",
        "Collision",
        "Success",
    ]
    cell_data = []
    for row in rows:
        cell_data.append([
            str(row["case"]),
            _pct(row["valid_intent_rate"]),
            _pct(row["strict_action_accuracy"]),
            _pct(row["missed_action_rate"]),
            _pct(row["unnecessary_maneuver_rate"]),
            _pct(row["collision_rate"]),
            _pct(row["success_rate"]),
        ])

    # Overall aggregates row
    n_calls_total = sum(r["n_calls"] for r in rows)
    n_strict_total = sum(r["strict_subset_calls"] for r in rows)

    def _wavg_calls(key):
        total = sum(r["n_calls"] for r in rows)
        return sum(r[key] * r["n_calls"] for r in rows) / total if total else float("nan")

    def _wavg_strict(key):
        total = sum(r["strict_subset_calls"] for r in rows)
        return sum(r[key] * r["strict_subset_calls"] for r in rows) / total if total else float("nan")

    def _avg(key):
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    cell_data.append([
        "Overall",
        _pct(_wavg_calls("valid_intent_rate")),
        _pct(_wavg_strict("strict_action_accuracy")),
        _pct(_wavg_strict("missed_action_rate")),
        _pct(_wavg_strict("unnecessary_maneuver_rate")),
        _pct(_avg("collision_rate")),
        _pct(_avg("success_rate")),
    ])

    n_rows = len(cell_data)
    n_cols = len(col_labels)

    fig_w = max(10, n_cols * 1.45)
    fig_h = 0.55 * (n_rows + 1) + 0.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.6)

    # Header styling
    for c in range(n_cols):
        cell = tbl[0, c]
        cell.set_facecolor("#2c4a6e")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternating row shading; bold the Overall row
    for r in range(1, n_rows + 1):
        is_overall = r == n_rows
        for c in range(n_cols):
            cell = tbl[r, c]
            if is_overall:
                cell.set_facecolor("#d8e4f0")
                cell.set_text_props(fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#f0f4f8")
            else:
                cell.set_facecolor("white")

    fig.suptitle(
        "LLM Integration Reliability and Closed-Loop Performance",
        fontsize=13, fontweight="bold", y=0.98,
    )
    footnote = (
        f"Valid Intent: successful LLM response parsed into K\u2091\u1d35\u2523 \u2208 \u007b\u22121, 0, +1\u007d.  "
        f"Strict Acc. computed over valid intents with a high-confidence single-label COLREGs reference "
        f"(n\u209b\u209c\u1d63\u1d4f\u209c = {n_strict_total:,} of {n_calls_total:,} total calls).  "
        "Collision and Success are policy-level metrics across 100 episodes."
    )
    fig.text(0.5, 0.01, footnote, ha="center", fontsize=8, color="#444444",
             style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Table saved: {output_path}")


# ── Figure 2: stacked outcome bar chart ───────────────────────────────────────

def _make_stacked_bar(rows: List[Dict], output_path: Path) -> None:
    """
    Each bar = one case.  Two stacks: Valid Intent / No Valid Intent = 100%.
    """
    labels = [str(r["case"]) for r in rows]
    n = len(rows)

    valid_intent   = np.array([r["valid_intent_rate"]       for r in rows])
    no_valid_intent = 1.0 - valid_intent

    x = np.arange(n)
    width = 0.5

    fig, ax = plt.subplots(figsize=(max(5, n * 1.8), 5))

    ax.bar(x, valid_intent,    width, color=_COL_VALID_CORRECT, label="Valid Intent")
    ax.bar(x, no_valid_intent, width, bottom=valid_intent,
           color=_COL_INVALID, label="No Valid Intent")

    for i in range(n):
        if valid_intent[i] > 0.05:
            ax.text(x[i], valid_intent[i] / 2, f"{valid_intent[i]*100:.1f}%",
                    ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        if no_valid_intent[i] > 0.04:
            ax.text(x[i], valid_intent[i] + no_valid_intent[i] / 2,
                    f"{no_valid_intent[i]*100:.1f}%",
                    ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Case {l}" for l in labels], fontsize=11)
    ax.set_ylabel("Fraction of LLM calls", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.set_title("LLM Pipeline Outcome Composition per Case",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Stacked bar saved: {output_path}")


# ── Data assembly ─────────────────────────────────────────────────────────────

def _build_row(seed_dir: Path, case_label: str) -> Dict:
    rel = _load_reliability(seed_dir)
    pol = _load_policy(seed_dir)

    n_calls        = int(rel["n_calls"])
    strict_calls   = int(rel["strict_subset_calls"])
    strict_correct = int(round(rel["strict_action_accuracy"] * strict_calls)) if strict_calls else 0

    valid_intent_rate = float(rel.get("valid_intent_rate", rel.get("valid_call_rate", float("nan"))))

    # Stacked bar fractions relative to n_calls
    invalid_frac = max(0.0, 1.0 - valid_intent_rate)

    collision_rate = float(pol.get("collision_any_mean", float("nan")))
    success_rate   = float(pol.get("success_ownship_mean", float("nan")))

    return {
        "case":                    case_label,
        "n_calls":                 n_calls,
        "strict_subset_calls":     strict_calls,
        "valid_intent_rate":       valid_intent_rate,
        "strict_action_accuracy":  float(rel["strict_action_accuracy"]),
        "strict_macro_f1":         float(rel["strict_macro_f1"]),
        "missed_action_rate":      float(rel["missed_action_rate"]),
        "unnecessary_maneuver_rate": float(rel["unnecessary_maneuver_rate"]),
        "collision_rate":          collision_rate,
        "success_rate":            success_rate,
        "invalid_frac":            invalid_frac,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate IEEE-style reliability table and stacked bar chart"
    )
    p.add_argument(
        "--eval_dirs",
        nargs="+",
        required=True,
        metavar="DIR",
        help="One or more seed_dirs (or parent eval dirs containing seed_0/)",
    )
    p.add_argument(
        "--case_labels",
        nargs="*",
        default=None,
        metavar="LABEL",
        help="Case labels to use in the plot (default: derived from dir name)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="1_Visualizations/llm_reliability_summary",
        help="Directory for output figures",
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_dirs = [_resolve(d) for d in args.eval_dirs]

    if args.case_labels:
        if len(args.case_labels) != len(eval_dirs):
            p.error("--case_labels must have the same count as --eval_dirs")
        case_labels = args.case_labels
    else:
        # auto-derive from directory name: pick the number after "case" if present
        import re
        case_labels = []
        for d in eval_dirs:
            m = re.search(r"case(\d+)", str(d))
            case_labels.append(m.group(1) if m else d.parent.name)

    rows = []
    for seed_dir, label in zip(eval_dirs, case_labels):
        print(f"Loading case {label} from {seed_dir} …")
        rows.append(_build_row(seed_dir, label))

    _make_table_figure(rows, out_dir / "llm_reliability_table.png")
    _make_stacked_bar(rows, out_dir / "llm_reliability_stacked_bar.png")

    n_calls_total = sum(r["n_calls"] for r in rows)
    n_strict = sum(r["strict_subset_calls"] for r in rows)
    valid_intent_total = sum(int(round(r["valid_intent_rate"] * r["n_calls"])) for r in rows)
    strict_correct_total = sum(
        int(round(r["strict_action_accuracy"] * r["strict_subset_calls"])) for r in rows
    )
    overall_valid_intent = valid_intent_total / n_calls_total if n_calls_total else float("nan")
    overall_strict_acc = strict_correct_total / n_strict if n_strict else float("nan")

    print(
        f"\nAcross {len(rows)} case(s): {n_calls_total:,} LLM calls  |  "
        f"{overall_valid_intent*100:.1f}% valid intent  |  "
        f"{overall_strict_acc*100:.1f}% strict accuracy"
    )
    print(f"Figures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
