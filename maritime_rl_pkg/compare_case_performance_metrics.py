"""
Compare performance metrics across multiple cases and runs for baseline and RL policies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# Helpers
# -----------------------------

def safe_float(value) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return x


def nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr)) if arr.size else float('nan')


def nanstd(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanstd(arr, ddof=0)) if arr.size else float('nan')


def first_existing(d: Dict, keys: List[str], default=float('nan')):
    for key in keys:
        if key in d:
            return d[key]
    return default


def normalize_method_label(raw: str) -> str:
    raw_l = (raw or '').strip().lower()
    if raw_l in {'baseline', 'corall', 'corall_rule_based', 'corall_reactive_avoidance_with_waypoint_planning'}:
        return 'baseline'
    if raw_l in {'rl', 'trained', 'policy', 'trained_policy'}:
        return 'rl'
    return raw_l or 'unknown'


# -----------------------------
# File discovery
# -----------------------------

def iter_candidate_dirs(paths: Iterable[str]) -> Iterable[Path]:
    for p in paths:
        path = Path(p).expanduser().resolve()
        if not path.exists():
            print(f'[warning] path not found: {path}')
            continue
        yield path


def discover_summary_files(root: Path) -> List[Path]:
    patterns = [
        '**/corall_baseline_eval_summary.json',
        '**/policy_eval_summary.json',
        '**/policy_eval_summary_VIS.json',
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    return sorted(set(files))


def discover_csv_files(root: Path) -> List[Path]:
    patterns = [
        '**/corall_baseline_eval_per_episode.csv',
        '**/policy_eval_per_episode.csv',
        '**/policy_eval_per_episode_VIS.csv',
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    return sorted(set(files))


# -----------------------------
# Parsing summary JSON and CSV
# -----------------------------

def infer_method_from_summary(path: Path, summary: Dict, explicit_method: Optional[str]) -> str:
    if explicit_method:
        return explicit_method
    if 'checkpoint' in summary:
        return 'rl'
    if 'baseline_type' in summary or 'baseline' in summary:
        return 'baseline'
    name = path.name.lower()
    if 'baseline' in name:
        return 'baseline'
    return 'rl'


def record_from_summary(path: Path, explicit_method: Optional[str] = None) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    method = infer_method_from_summary(path, summary, explicit_method)
    case = int(first_existing(summary, ['case'], default=-1))
    episodes = int(first_existing(summary, ['episodes'], default=-1))
    seed_base = int(first_existing(summary, ['seed_base'], default=-1))

    record = {
        'source_file': str(path),
        'source_kind': 'summary_json',
        'method': method,
        'case': case,
        'episodes': episodes,
        'seed_base': seed_base,
        'run_name': path.parent.parent.name if path.parent.parent else path.parent.name,
        'success_rate': safe_float(first_existing(summary, [
            'success_rate_ownship_mean', 'success_rate', 'success_rate_agents_mean'
        ])),
        'collision_rate': safe_float(first_existing(summary, ['collision_rate'])),
        'path_length_m': safe_float(first_existing(summary, [
            'path_length_m_ownship_mean', 'path_length_m_mean'
        ])),
        'min_dcpa_m': safe_float(first_existing(summary, [
            'min_dcpa_m_ownship_mean', 'min_dcpa_m_mean'
        ])),
        'min_tcpa_s': safe_float(first_existing(summary, [
            'min_tcpa_s_ownship_mean', 'min_tcpa_s_mean'
        ])),
        'risk_exposure': safe_float(first_existing(summary, [
            'risk_exposure_ownship_mean', 'risk_exposure_mean'
        ])),
        'min_actual_sep_m': safe_float(first_existing(summary, [
            'min_actual_sep_m_ownship_mean', 'min_actual_sep_m_mean'
        ])),
        'near_miss_rate': safe_float(first_existing(summary, ['near_miss_rate'])),
        'goal_progress': safe_float(first_existing(summary, [
            'goal_progress_ownship_mean', 'goal_progress_mean'
        ])),
        'completion_time_s': safe_float(first_existing(summary, [
            'completion_time_s_ownship_mean', 'completion_time_s_mean'
        ])),
        'episode_return_mean': safe_float(first_existing(summary, [
            'episode_return_ownship_mean', 'episode_return_mean'
        ])),
    }
    return record


def infer_method_from_csv(path: Path, explicit_method: Optional[str]) -> str:
    if explicit_method:
        return explicit_method
    lower = path.name.lower()
    if 'baseline' in lower:
        return 'baseline'
    return 'rl'


def record_from_csv(path: Path, explicit_method: Optional[str] = None) -> Optional[Dict]:
    rows: List[Dict] = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return None

    method = infer_method_from_csv(path, explicit_method)

    def col_mean(colnames: List[str]) -> float:
        values = []
        for row in rows:
            for c in colnames:
                if c in row and row[c] not in {'', None}:
                    values.append(safe_float(row[c]))
                    break
        return nanmean(values)

    first_row = rows[0]
    # infer case from folder name if needed, e.g. policy_eval_case6_* or corall_baseline_case6_*
    case = -1
    for part in path.parts[::-1]:
        lowered = part.lower()
        if 'case' in lowered:
            digits = ''.join(ch for ch in lowered if ch.isdigit())
            if digits:
                case = int(digits)
                break

    seed_base = safe_float(first_row.get('episode_seed', float('nan')))
    record = {
        'source_file': str(path),
        'source_kind': 'per_episode_csv',
        'method': method,
        'case': case,
        'episodes': len(rows),
        'seed_base': int(seed_base) if np.isfinite(seed_base) else -1,
        'run_name': path.parent.parent.name if path.parent.parent else path.parent.name,
        'success_rate': col_mean(['success_ownship', 'success_rate_agents']),
        'collision_rate': col_mean(['collision_any']),
        'path_length_m': col_mean(['path_length_m_ownship', 'path_length_m_mean']),
        'min_dcpa_m': col_mean(['min_dcpa_m_ownship', 'min_dcpa_m_mean']),
        'min_tcpa_s': col_mean(['min_tcpa_s_ownship', 'min_tcpa_s_mean']),
        'risk_exposure': col_mean(['risk_exposure_ownship', 'risk_exposure_mean']),
        'min_actual_sep_m': col_mean(['min_actual_sep_m_ownship', 'min_actual_sep_m_mean']),
        'near_miss_rate': col_mean(['near_miss_any']),
        'goal_progress': col_mean(['goal_progress_ownship', 'goal_progress_mean']),
        'completion_time_s': col_mean(['completion_time_s_ownship', 'completion_time_s_mean']),
        'episode_return_mean': col_mean(['episode_return_ownship', 'episode_return_mean']),
    }
    return record


# -----------------------------
# Aggregation
# -----------------------------

METRICS = [
    'success_rate',
    'collision_rate',
    'path_length_m',
    'min_dcpa_m',
    'min_tcpa_s',
    'risk_exposure',
    'min_actual_sep_m',
    'near_miss_rate',
    'goal_progress',
    'completion_time_s',
    'episode_return_mean',
]

HIGHER_IS_BETTER = {
    'success_rate': True,
    'collision_rate': False,
    'path_length_m': False,
    'min_dcpa_m': True,
    'min_tcpa_s': True,
    'risk_exposure': False,
    'min_actual_sep_m': True,
    'near_miss_rate': False,
    'goal_progress': True,
    'completion_time_s': False,
    'episode_return_mean': True,
}


def deduplicate_records(records: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for rec in records:
        key = (rec['method'], rec['case'], rec['run_name'])
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def aggregate_records(records: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for rec in records:
        grouped[(rec['method'], rec['case'])].append(rec)

    agg_rows: List[Dict] = []
    for (method, case), group in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        row: dict[str, str | int | float] = {
            'method': method,
            'case': case,
            'n_runs': len(group),
            'episodes_mean': nanmean(rec['episodes'] for rec in group),
        }
        for metric in METRICS:
            vals = [safe_float(rec.get(metric, float('nan'))) for rec in group]
            row[f'{metric}_mean'] = nanmean(vals)
            row[f'{metric}_std'] = nanstd(vals)
        agg_rows.append(row)
    return agg_rows


def build_improvement_rows(agg_rows: List[Dict]) -> List[Dict]:
    by_case: Dict[int, Dict[str, Dict]] = defaultdict(dict)
    for row in agg_rows:
        by_case[int(row['case'])][normalize_method_label(str(row['method']))] = row

    out = []
    for case in sorted(by_case):
        pair = by_case[case]
        if 'baseline' not in pair or 'rl' not in pair:
            continue
        base = pair['baseline']
        rl = pair['rl']
        row: dict[str, int | float] = {'case': case}
        for metric in METRICS:
            b = safe_float(base.get(f'{metric}_mean', float('nan')))
            r = safe_float(rl.get(f'{metric}_mean', float('nan')))
            diff = r - b if np.isfinite(r) and np.isfinite(b) else float('nan')
            if np.isfinite(r) and np.isfinite(b) and abs(b) > 1e-12:
                pct = 100.0 * diff / abs(b)
            else:
                pct = float('nan')
            # positive score means RL better, regardless of metric direction
            signed_gain = diff if HIGHER_IS_BETTER[metric] else -diff
            row[f'{metric}_baseline'] = b
            row[f'{metric}_rl'] = r
            row[f'{metric}_diff_rl_minus_baseline'] = diff
            row[f'{metric}_pct_change_from_baseline'] = pct
            row[f'{metric}_signed_gain'] = signed_gain
        out.append(row)
    return out


# -----------------------------
# Output writers
# -----------------------------

def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write('')
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_by_case(agg_rows: List[Dict], metric: str, output_dir: Path) -> Optional[Path]:
    cases = sorted({int(r['case']) for r in agg_rows})
    baseline_means, baseline_stds = [], []
    rl_means, rl_stds = [], []

    lookup = {(normalize_method_label(r['method']), int(r['case'])): r for r in agg_rows}

    for case in cases:
        b = lookup.get(('baseline', case))
        r = lookup.get(('rl', case))
        baseline_means.append(safe_float(b.get(f'{metric}_mean', float('nan'))) if b else float('nan'))
        baseline_stds.append(safe_float(b.get(f'{metric}_std', float('nan'))) if b else float('nan'))
        rl_means.append(safe_float(r.get(f'{metric}_mean', float('nan'))) if r else float('nan'))
        rl_stds.append(safe_float(r.get(f'{metric}_std', float('nan'))) if r else float('nan'))

    if not np.any(np.isfinite(np.asarray(baseline_means + rl_means, dtype=float))):
        return None

    x = np.arange(len(cases), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, baseline_means, width, yerr=baseline_stds, capsize=4, label='Baseline')
    ax.bar(x + width / 2, rl_means, width, yerr=rl_stds, capsize=4, label='RL')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Case {c}' for c in cases])
    ax.set_ylabel(metric)
    ax.set_title(f'{metric}: baseline vs RL by case')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = output_dir / f'{metric}_by_case.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path


def write_markdown_summary(path: Path, agg_rows: List[Dict], improvement_rows: List[Dict]) -> None:
    lines = ['# Case comparison summary', '']
    if not agg_rows:
        lines.append('No records found.')
    else:
        lines.append('## Aggregated runs')
        lines.append('')
        lines.append('| Case | Method | Runs | Success | Collision | Min DCPA (m) | Risk exposure | Goal progress | Completion time (s) |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
        for row in sorted(agg_rows, key=lambda r: (r['case'], r['method'])):
            lines.append(
                f"| {int(row['case'])} | {row['method']} | {int(row['n_runs'])} | "
                f"{row['success_rate_mean']:.3f} | {row['collision_rate_mean']:.3f} | "
                f"{row['min_dcpa_m_mean']:.1f} | {row['risk_exposure_mean']:.3f} | "
                f"{row['goal_progress_mean']:.3f} | {row['completion_time_s_mean']:.1f} |"
            )
        lines.append('')

    if improvement_rows:
        lines.append('## RL minus baseline by case')
        lines.append('')
        lines.append('| Case | Δ Success | Δ Collision | Δ Min DCPA (m) | Δ Risk exposure | Δ Goal progress | Δ Completion time (s) |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|')
        for row in improvement_rows:
            lines.append(
                f"| {int(row['case'])} | {row['success_rate_diff_rl_minus_baseline']:.3f} | "
                f"{row['collision_rate_diff_rl_minus_baseline']:.3f} | "
                f"{row['min_dcpa_m_diff_rl_minus_baseline']:.1f} | "
                f"{row['risk_exposure_diff_rl_minus_baseline']:.3f} | "
                f"{row['goal_progress_diff_rl_minus_baseline']:.3f} | "
                f"{row['completion_time_s_diff_rl_minus_baseline']:.1f} |"
            )
    path.write_text('\n'.join(lines), encoding='utf-8')


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Aggregate baseline and trained-policy evaluation outputs across Imazu cases.'
    )
    p.add_argument('--baseline_dirs', nargs='*', default=[], help='One or more baseline output folders or parent directories.')
    p.add_argument('--rl_dirs', nargs='*', default=[], help='One or more RL evaluation output folders or parent directories.')
    p.add_argument('--output_dir', required=True, help='Directory to write comparison tables and plots.')
    p.add_argument('--prefer_csv', action='store_true', help='Use per-episode CSV files even if summary JSON exists.')
    return p.parse_args()


def collect_records(paths: List[str], method: str, prefer_csv: bool) -> List[Dict]:
    records: List[Dict] = []
    for root in iter_candidate_dirs(paths):
        summary_files = discover_summary_files(root)
        csv_files = discover_csv_files(root)

        if prefer_csv or not summary_files:
            for csv_path in csv_files:
                rec = record_from_csv(csv_path, explicit_method=method)
                if rec is not None:
                    records.append(rec)
        else:
            for summary_path in summary_files:
                records.append(record_from_summary(summary_path, explicit_method=method))

    return deduplicate_records(records)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_records = collect_records(args.baseline_dirs, 'baseline', args.prefer_csv)
    rl_records = collect_records(args.rl_dirs, 'rl', args.prefer_csv)
    all_records = sorted(baseline_records + rl_records, key=lambda r: (r['case'], r['method'], r['run_name']))

    if not all_records:
        raise SystemExit('No evaluation records found. Check the input directories and filenames.')

    agg_rows = aggregate_records(all_records)
    improvement_rows = build_improvement_rows(agg_rows)

    write_csv(out_dir / 'all_runs_tidy.csv', all_records)
    write_csv(out_dir / 'case_method_summary.csv', agg_rows)
    write_csv(out_dir / 'case_rl_vs_baseline_deltas.csv', improvement_rows)
    write_markdown_summary(out_dir / 'comparison_summary.md', agg_rows, improvement_rows)

    plot_dir = out_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plotted = []
    for metric in METRICS:
        p = plot_metric_by_case(agg_rows, metric, plot_dir)
        if p is not None:
            plotted.append(p)

    print('\n=== Comparison outputs saved ===')
    print(f'Tidy run table:       {out_dir / "all_runs_tidy.csv"}')
    print(f'Case summary table:   {out_dir / "case_method_summary.csv"}')
    print(f'RL-baseline deltas:   {out_dir / "case_rl_vs_baseline_deltas.csv"}')
    print(f'Markdown summary:     {out_dir / "comparison_summary.md"}')
    if plotted:
        print('Plots:')
        for p in plotted:
            print(f'  - {p}')


if __name__ == '__main__':
    main()
