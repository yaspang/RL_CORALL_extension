"""
Execution Logger — Records what the RL policy proposed, what was
actually applied after arbitration, and the observed encounter context.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm_intent_service import IntentCommand, NMI


class ExecutionLogger:
    def __init__(self):
        self._records: List[Dict] = []

    def log(
        self,
        step: int,
        time_s: float,
        episode_idx: int,
        episode_seed: int,
        proposed_action,
        applied_action,
        intent: Optional[IntentCommand],
        arbitration: Dict[str, Any],
        multi_env=None,
    ) -> None:
        record: Dict[str, Any] = {
            "episode_idx":          episode_idx,
            "episode_seed":         episode_seed,
            "step":                 step,
            "time_s":               round(float(time_s), 2),
            "proposed_heading_idx": int(proposed_action[0]),
            "proposed_speed_idx":   int(proposed_action[1]) if len(proposed_action) > 1 else None,
            "applied_heading_idx":  int(applied_action[0]),
            "applied_speed_idx":    int(applied_action[1]) if len(applied_action) > 1 else None,
            "arbitration_mode":     arbitration.get("mode"),
            "arbitration_reason":   arbitration.get("reason"),
            "intent_command_id":    intent.command_id if intent is not None else None,
            "intent_mode":          intent.mode if intent is not None else None,
            "intent_kdir":          intent.kdir if intent is not None else None,
            "intent_valid":         intent.valid if intent is not None else None,
            "intent_rule":          intent.rule if intent is not None else None,
            "intent_target_id":     intent.target_id if intent is not None else None,
        }

        if multi_env is not None and intent is not None and intent.target_id is not None:
            k = intent.target_id
            record["target_range_nmi"] = round(float(multi_env.pair_dist[0, k]) / NMI, 4)
            record["target_dcpa_nmi"]  = round(abs(float(multi_env.pair_dcpa[0, k])) / NMI, 4)
            record["target_tcpa_s"]    = round(float(multi_env.pair_tcpa[0, k]), 2)
            record["target_risk"]      = round(float(multi_env.pair_risk[0, k]), 4)

        self._records.append(record)

    def get_records(self) -> List[Dict]:
        return self._records

    def save(self, csv_path: Path) -> None:
        if not self._records:
            print("[ExecutionLogger] No records to save.")
            return
        fieldnames = list(self._records[0].keys())
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._records)
        print(f"[ExecutionLogger] Execution log saved: {csv_path} ({len(self._records)} records)")

    def __len__(self) -> int:
        return len(self._records)
