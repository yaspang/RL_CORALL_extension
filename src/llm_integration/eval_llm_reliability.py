"""
Score LLM recommendation reliability from llm_intent_log.csv.

Usage
-----
  python -m src.llm_integration.eval_llm_reliability \\
      --eval_dir "results_llmapi_case1_interval10/seed_0"

Key arguments
-------------
  --eval_dir PATH          Seed dir with llm_intent_log.csv, or parent dir with seed_0/
  --output_dir PATH        Output dir (default: <seed_dir>/results_llm_reliability)
  --no_action_risk_threshold FLOAT
                           Risk below which no-rule encounters score as maintain-course
                           (default: 0.20)

Outputs
-------
  llm_call_scores.csv          Per-call scores with valid_intent, is_correct flags
  llm_reliability_summary.json  Aggregate: valid_intent_rate, strict_action_accuracy,
                                 missed_action_rate, unnecessary_maneuver_rate, etc.
  llm_confusion_matrix.csv      3x3 confusion counts on strict high-confidence subset
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _resolve_seed_dir(eval_dir: Path) -> Path:
    """Resolve to the seed directory that contains llm_intent_log.csv."""
    eval_dir = Path(eval_dir)
    direct = eval_dir / "llm_intent_log.csv"
    nested = eval_dir / "seed_0" / "llm_intent_log.csv"

    if direct.exists():
        return eval_dir
    if nested.exists():
        return eval_dir / "seed_0"

    raise FileNotFoundError(
        f"Could not find llm_intent_log.csv in {eval_dir} or {eval_dir / 'seed_0'}"
    )


def _to_float(row: Dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def _to_int(row: Dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return default


def _read_llm_rows(csv_path: Path) -> List[Dict]:
    """Read llm_intent_log.csv into typed dictionaries."""
    rows: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "episode_idx": _to_int(r, "episode_idx", -1),
                    "episode_seed": _to_int(r, "episode_seed", -1),
                    "step": _to_int(r, "step", -1),
                    "time_s": _to_float(r, "time_s"),
                    "target_id": _to_int(r, "target_id", -1),
                    "n_active_vessels": _to_int(r, "n_active_vessels", 0),
                    "range_nmi": _to_float(r, "range_nmi"),
                    "abs_dcpa_nmi": _to_float(r, "abs_dcpa_nmi"),
                    "tcpa_s": _to_float(r, "tcpa_s"),
                    "relative_bearing_deg": _to_float(r, "relative_bearing_deg"),
                    "risk": _to_float(r, "risk"),
                    "colreg_rule": (r.get("colreg_rule", "") or "").strip(),
                    "encounter_phase": (r.get("encounter_phase", "") or "").strip().lower(),
                    "llm_call_status": (r.get("llm_call_status", "") or "").strip().lower(),
                    "parsed_kdir": _to_int(r, "parsed_kdir", 0),
                    "parsed_kdir_valid": _to_int(r, "parsed_kdir_valid", -1),
                    "parse_reason": (r.get("parse_reason", "") or "").strip().lower(),
                    "llm_response": (r.get("llm_response", "") or "").strip(),
                }
            )

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def _rule_key(rule_text: str) -> str:
    """Normalize rule labels into compact keys."""
    s = (rule_text or "").strip().lower()
    if "15.1" in s:
        return "15.1"
    if "15.2" in s:
        return "15.2"
    if "rule 14" in s or "14" == s:
        return "14"
    if "rule 13" in s or "13" == s:
        return "13"
    return "other"


def _collapse_to_calls(rows: List[Dict]) -> List[Dict]:
    """
    Collapse target rows to one row per actual LLM call.

    Group key uses episode + step + time to avoid conflating calls across episodes.
    Within each call, keep the highest-risk row, tie-broken by shortest range.
    """
    by_call: Dict[Tuple[int, int, int, float], List[Dict]] = defaultdict(list)
    for r in rows:
        key = (
            int(r["episode_idx"]),
            int(r["episode_seed"]),
            int(r["step"]),
            float(r["time_s"]),
        )
        by_call[key].append(r)

    call_rows: List[Dict] = []
    for key in sorted(by_call.keys(), key=lambda k: (k[0], k[2], k[3], k[1])):
        group = by_call[key]
        group_sorted = sorted(
            group,
            key=lambda r: (
                -(r["risk"] if np.isfinite(r["risk"]) else -1e9),
                (r["range_nmi"] if np.isfinite(r["range_nmi"]) else 1e9),
            ),
        )
        best = dict(group_sorted[0])
        best["call_size"] = len(group)
        call_rows.append(best)
    return call_rows


def _expected_action_set(
    row: Dict,
    no_action_risk_threshold: float,
) -> Tuple[Optional[List[int]], str, str]:
    """
    Return (acceptable_kdir_values, baseline_reason, confidence_tier).

    confidence_tier: high | medium | low | none
    """
    phase = (row.get("encounter_phase", "") or "").lower()
    tcpa = float(row.get("tcpa_s", float("nan")))
    risk = float(row.get("risk", float("nan")))
    rule = _rule_key(str(row.get("colreg_rule", "")))
    rel_bearing = float(row.get("relative_bearing_deg", float("nan")))

    # Post-CPA should maintain/stop initiating new avoidance.
    if phase == "post_cpa" or (np.isfinite(tcpa) and tcpa <= 0.0):
        return [0], "post_cpa_or_tcpa_nonpositive", "high"

    if rule == "14":
        return [1], "rule14_headon", "high"

    if rule == "15.1":
        return [1], "rule15_1_crossing_giveway", "high"

    if rule == "15.2":
        return [0], "rule15_2_crossing_standon", "high"

    if rule == "13":
        # Rule 13 may require contextual role determination.
        # Proxy logic using ownship-relative bearing only (limited confidence).
        rb = abs(rel_bearing) if np.isfinite(rel_bearing) else float("nan")
        if np.isfinite(rb) and rb >= 112.5:
            return [0], "rule13_proxy_being_overtaken", "medium"
        if np.isfinite(rb) and rb <= 67.5:
            return [-1, 1], "rule13_proxy_overtaking_target", "medium"
        return [-1, 0, 1], "rule13_ambiguous", "low"

    # For weak/non-specific rule labels, only assert maintain when low risk.
    if np.isfinite(risk) and risk < no_action_risk_threshold:
        return [0], "low_risk_non_threatening", "medium"

    return None, "insufficient_geometry_label", "none"


def _confusion_frame(strict_rows: List[Dict]) -> List[Dict]:
    labels = [-1, 0, 1]
    counts = {(e, p): 0 for e in labels for p in labels}

    for r in strict_rows:
        exp = int(r["expected_single"])
        pred = int(r["parsed_kdir"])
        if exp in labels and pred in labels:
            counts[(exp, pred)] += 1

    out: List[Dict] = []
    for exp in labels:
        row = {"expected": exp, "pred_-1": counts[(exp, -1)], "pred_0": counts[(exp, 0)], "pred_1": counts[(exp, 1)]}
        out.append(row)
    return out


def _macro_f1(strict_rows: List[Dict]) -> float:
    labels = [-1, 0, 1]
    f1s: List[float] = []

    exp = np.array([int(r["expected_single"]) for r in strict_rows], dtype=int)
    pred = np.array([int(r["parsed_kdir"]) for r in strict_rows], dtype=int)

    for c in labels:
        tp = int(np.sum((exp == c) & (pred == c)))
        fp = int(np.sum((exp != c) & (pred == c)))
        fn = int(np.sum((exp == c) & (pred != c)))

        if tp == 0 and fp == 0 and fn == 0:
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1s.append(f1)

    if not f1s:
        return float("nan")
    return float(np.mean(f1s))


def evaluate_llm_reliability(
    eval_dir: Path,
    output_dir: Optional[Path] = None,
    no_action_risk_threshold: float = 0.20,
) -> Dict:
    seed_dir = _resolve_seed_dir(eval_dir)
    csv_path = seed_dir / "llm_intent_log.csv"
    rows = _read_llm_rows(csv_path)
    call_rows = _collapse_to_calls(rows)

    scored_rows: List[Dict] = []
    for row in call_rows:
        status = str(row.get("llm_call_status", "")).lower()
        parsed = int(row.get("parsed_kdir", 0))
        # Require both valid flag and a legal K_dir value; fallback for old logs without the flag.
        parsed_valid_field = int(row.get("parsed_kdir_valid", -1))
        parse_ok = (
            bool(parsed_valid_field == 1) and parsed in (-1, 0, 1)
        ) if parsed_valid_field in (0, 1) else (parsed in (-1, 0, 1))
        valid_call = status == "success"

        expected_set, baseline_reason, confidence = _expected_action_set(
            row, no_action_risk_threshold=no_action_risk_threshold
        )

        valid_intent = valid_call and parse_ok
        is_scored = valid_intent and expected_set is not None
        is_correct = bool(is_scored and expected_set is not None and parsed in expected_set)

        strict_single = bool(
            is_scored and confidence == "high" and expected_set is not None and len(expected_set) == 1
        )
        expected_single = int(expected_set[0]) if (strict_single and expected_set is not None) else None

        scored_rows.append(
            {
                "episode_idx": int(row["episode_idx"]),
                "episode_seed": int(row["episode_seed"]),
                "step": int(row["step"]),
                "time_s": float(row["time_s"]),
                "target_id_primary": int(row["target_id"]),
                "call_size": int(row.get("call_size", 1)),
                "status": status,
                "parse_ok": int(parse_ok),
                "parse_reason": str(row.get("parse_reason", "")),
                "valid_call": int(valid_call),
                "rule_key": _rule_key(str(row.get("colreg_rule", ""))),
                "colreg_rule": str(row.get("colreg_rule", "")),
                "encounter_phase": str(row.get("encounter_phase", "")),
                "risk": float(row.get("risk", float("nan"))),
                "tcpa_s": float(row.get("tcpa_s", float("nan"))),
                "abs_dcpa_nmi": float(row.get("abs_dcpa_nmi", float("nan"))),
                "range_nmi": float(row.get("range_nmi", float("nan"))),
                "relative_bearing_deg": float(row.get("relative_bearing_deg", float("nan"))),
                "parsed_kdir": parsed,
                "expected_set": "" if expected_set is None else "{" + ",".join(str(v) for v in expected_set) + "}",
                "expected_single": "" if expected_single is None else int(expected_single),
                "baseline_reason": baseline_reason,
                "confidence": confidence,
                "valid_intent": int(valid_intent),
                "is_scored": int(is_scored),
                "is_correct": int(is_correct),
                "strict_single": int(strict_single),
            }
        )

    out_dir = output_dir or (seed_dir / "results_llm_reliability")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save row-wise results.
    call_scores_path = out_dir / "llm_call_scores.csv"
    with open(call_scores_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scored_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scored_rows)

    # Aggregate metrics.
    n_calls = len(scored_rows)
    valid_calls = [r for r in scored_rows if int(r["valid_call"]) == 1]
    parse_ok_calls = [r for r in scored_rows if int(r["parse_ok"]) == 1]
    valid_intent_calls = [r for r in scored_rows if int(r["valid_intent"]) == 1]
    scored_calls = [r for r in scored_rows if int(r["is_scored"]) == 1]
    strict_calls = [r for r in scored_rows if int(r["strict_single"]) == 1]

    strict_correct = sum(int(r["is_correct"]) for r in strict_calls)
    strict_acc = strict_correct / len(strict_calls) if strict_calls else float("nan")

    macro_f1 = _macro_f1(strict_calls)

    # Safety-flavored error rates on strict subset.
    # expected +1 means avoidance expected.
    wrong_side = 0
    missed_action = 0
    unnecessary = 0
    post_cpa_violation = 0

    rule_correct = Counter()
    rule_total = Counter()

    for r in strict_calls:
        exp = int(r["expected_single"])
        pred = int(r["parsed_kdir"])
        reason = str(r["baseline_reason"])

        rk = str(r["rule_key"])
        rule_total[rk] += 1
        if pred == exp:
            rule_correct[rk] += 1

        if reason == "post_cpa_or_tcpa_nonpositive" and pred != 0:
            post_cpa_violation += 1

        if exp == 1 and pred == -1:
            wrong_side += 1
        if exp == 1 and pred == 0:
            missed_action += 1
        if exp == 0 and pred != 0:
            unnecessary += 1

    # Episode-level error rate on strict subset.
    by_ep = defaultdict(list)
    for r in strict_calls:
        ep_key = (int(r["episode_idx"]), int(r["episode_seed"]))
        by_ep[ep_key].append(r)

    episode_error_count = 0
    for ep_rows in by_ep.values():
        if any(int(r["is_correct"]) == 0 for r in ep_rows):
            episode_error_count += 1

    episode_error_rate = (
        episode_error_count / len(by_ep) if by_ep else float("nan")
    )

    per_rule_accuracy = {}
    for rk in sorted(rule_total.keys()):
        per_rule_accuracy[rk] = {
            "n": int(rule_total[rk]),
            "accuracy": (rule_correct[rk] / rule_total[rk]) if rule_total[rk] else float("nan"),
        }

    parse_reason_counts = Counter(str(r.get("parse_reason", "") or "") for r in scored_rows)

    summary = {
        "eval_dir": str(eval_dir),
        "seed_dir": str(seed_dir),
        "source_csv": str(csv_path),
        "n_total_rows": len(rows),
        "n_calls": n_calls,
        "valid_call_rate": (len(valid_calls) / n_calls) if n_calls else float("nan"),
        "parse_success_rate": (len(parse_ok_calls) / n_calls) if n_calls else float("nan"),
        "valid_intent_rate": (len(valid_intent_calls) / n_calls) if n_calls else float("nan"),
        "scored_call_rate": (len(scored_calls) / n_calls) if n_calls else float("nan"),
        "maneuver_consistency_rate": (sum(int(r["is_correct"]) for r in scored_calls) / len(scored_calls)) if scored_calls else float("nan"),
        "strict_subset_calls": len(strict_calls),
        "strict_action_accuracy": strict_acc,
        "strict_macro_f1": macro_f1,
        "post_cpa_violation_rate": (post_cpa_violation / len(strict_calls)) if strict_calls else float("nan"),
        "wrong_side_rate": (wrong_side / len(strict_calls)) if strict_calls else float("nan"),
        "missed_action_rate": (missed_action / len(strict_calls)) if strict_calls else float("nan"),
        "unnecessary_maneuver_rate": (unnecessary / len(strict_calls)) if strict_calls else float("nan"),
        "episode_error_rate": episode_error_rate,
        "per_rule_accuracy": per_rule_accuracy,
        "parse_reason_counts": dict(sorted(parse_reason_counts.items())),
        "notes": {
            "strict_subset_definition": "valid_call & parse_ok & high-confidence single-label baseline",
            "call_collapse": "one row per (episode_idx, episode_seed, step, time_s), selecting highest-risk target",
            "rule13_handling": "proxy/ambiguous handling due to limited overtaking-role observability in current logs",
        },
    }

    summary_path = out_dir / "llm_reliability_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    confusion_rows = _confusion_frame(strict_calls)
    cm_path = out_dir / "llm_confusion_matrix.csv"
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["expected", "pred_-1", "pred_0", "pred_1"])
        writer.writeheader()
        writer.writerows(confusion_rows)

    print(f"Saved: {call_scores_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {cm_path}")
    print("-")
    print(f"Calls: {n_calls}")
    print(f"Valid-call rate: {summary['valid_call_rate']:.3f}" if n_calls else "Valid-call rate: n/a")
    print(f"Parse success rate: {summary['parse_success_rate']:.3f}" if n_calls else "Parse success rate: n/a")
    print(f"Valid intent rate: {summary['valid_intent_rate']:.3f}" if n_calls else "Valid intent rate: n/a")
    if scored_calls:
        print(f"Maneuver consistency rate: {summary['maneuver_consistency_rate']:.3f} ({sum(int(r['is_correct']) for r in scored_calls)}/{len(scored_calls)} scored calls)")
    if strict_calls:
        print(f"Strict accuracy: {strict_acc:.3f} ({strict_correct}/{len(strict_calls)})")
        print(f"Strict macro-F1: {macro_f1:.3f}" if np.isfinite(macro_f1) else "Strict macro-F1: n/a")
    else:
        print("Strict accuracy: n/a (no strict calls)")

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Stage-1 LLM recommendation reliability from llm_intent_log.csv"
    )
    p.add_argument(
        "--eval_dir",
        type=Path,
        required=True,
        help="Path to seed dir (contains llm_intent_log.csv) or parent eval dir (contains seed_0)",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional output dir. Default: <seed_dir>/results_llm_reliability",
    )
    p.add_argument(
        "--no_action_risk_threshold",
        type=float,
        default=0.20,
        help="Risk threshold below which 'no applicable rule' is scored as maintain-course",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_llm_reliability(
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
        no_action_risk_threshold=args.no_action_risk_threshold,
    )


if __name__ == "__main__":
    main()
