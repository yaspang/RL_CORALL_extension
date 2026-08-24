"""
Intent Action Arbiter — Validates/constrains the RL action using the
active LLM intent before it reaches env.step() (item 4: apply intent before step).

Order of authority (item 6):
  1. Emergency collision-avoidance supervisor (deterministic, always wins)
  2. Validated COLREG/LLM intent (constrains, doesn't replace, the RL action)
  3. RL policy (used as-is when no valid/applicable intent exists)
  4. Safe fallback == RL policy's own proposed action (never force kdir=0)

Heading action space is MultiDiscrete([n_heading, n_speed]) (env_RL_ppo.py).
delta_heading_norm = -1.0 + 2.0*heading_idx/(n_heading-1) uses the standard CCW-positive
math convention, so a negative heading offset is a starboard (clockwise) turn:
  - kdir=+1 (starboard) -> heading_idx BELOW center
  - kdir=-1 (port)      -> heading_idx ABOVE center
  - center index        -> no route-heading offset
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .llm_intent_service import IntentCommand, NMI


class IntentActionArbiter:
    def __init__(
        self,
        n_heading: int,
        minimum_turn_bins: int = 1,
        emergency_range_nmi: float = 0.15,
        emergency_risk: float = 0.9,
    ):
        self.n_heading = int(n_heading)
        self.center = (self.n_heading - 1) // 2
        self.minimum_turn_bins = int(minimum_turn_bins)
        self.emergency_range_nmi = float(emergency_range_nmi)
        self.emergency_risk = float(emergency_risk)
        self.previous_heading_index = self.center

    def reset_episode(self) -> None:
        """Reset carried-over heading state so episode n+1 doesn't inherit episode n's state."""
        self.previous_heading_index = self.center

    def apply(
        self,
        proposed_action,
        intent: Optional[IntentCommand],
        multi_env,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        applied = np.asarray(proposed_action, dtype=np.int32).copy()

        # ── Priority 1: deterministic emergency supervisor ─────────
        emergency = self._check_emergency(multi_env)
        if emergency is not None:
            target_bearing_deg, min_dist_nmi, max_risk = emergency
            # Turn away from the nearest/highest-risk target regardless of COLREG.
            applied[0] = 0 if target_bearing_deg < 0 else self.n_heading - 1
            if applied.shape[0] > 1:
                applied[1] = 0  # minimum speed bin
            self.previous_heading_index = int(applied[0])
            return applied, {
                "mode": "emergency_override",
                "reason": f"min_range={min_dist_nmi:.3f}nmi max_risk={max_risk:.3f}",
            }

        # ── Priority 3/4: no usable intent -> RL action stands as-is ─────────
        if intent is None or not intent.valid:
            self.previous_heading_index = int(applied[0])
            return applied, {"mode": "rl_fallback", "reason": "no_valid_intent"}

        # ── Safety-supervisor vetoes of an otherwise-valid intent (item 6) ───
        reject_reason = self._veto_reason(intent, multi_env)
        if reject_reason is not None:
            self.previous_heading_index = int(applied[0])
            return applied, {"mode": "rl_fallback", "reason": reject_reason}

        # ── Priority 2: apply the validated intent as an action constraint ───
        if intent.mode == "INITIATE_AVOIDANCE":
            if intent.kdir == +1:
                max_allowed = self.center - self.minimum_turn_bins
                applied[0] = min(int(applied[0]), max_allowed)
            elif intent.kdir == -1:
                min_allowed = self.center + self.minimum_turn_bins
                applied[0] = max(int(applied[0]), min_allowed)
        elif intent.mode == "CONTINUE_AVOIDANCE":
            # Preserve the previously applied avoidance heading rather than letting
            # a fresh kdir=0-like read undo an in-progress maneuver.
            applied[0] = self.previous_heading_index
        elif intent.mode == "STAND_ON":
            applied[0] = self.center
        elif intent.mode == "RESUME_ROUTE":
            applied[0] = self._move_one_bin_toward(int(applied[0]), self.center)

        self.previous_heading_index = int(applied[0])
        return applied, {
            "mode": "intent_applied",
            "intent_mode": intent.mode,
            "kdir": intent.kdir,
            "command_id": intent.command_id,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _move_one_bin_toward(self, current: int, target: int) -> int:
        if current < target:
            return current + 1
        if current > target:
            return current - 1
        return current

    def _check_emergency(self, multi_env) -> Optional[Tuple[float, float, float]]:
        """Return (bearing_deg_of_worst_target, min_range_nmi, max_risk) or None."""
        pair_dist = multi_env.pair_dist[0]
        pair_risk = multi_env.pair_risk[0]
        finite = np.isfinite(pair_dist)
        finite[0] = False  # exclude self
        if not np.any(finite):
            return None

        min_dist_nmi = float(np.min(pair_dist[finite])) / NMI
        max_risk = float(np.max(pair_risk[finite]))
        if min_dist_nmi >= self.emergency_range_nmi and max_risk < self.emergency_risk:
            return None

        k = int(np.argmin(np.where(finite, pair_dist, np.inf)))
        X_all = multi_env.X_all
        x_own, y_own, psi_own = float(X_all[0, 0]), float(X_all[0, 1]), float(X_all[0, 2])
        x_ts, y_ts = float(X_all[k, 0]), float(X_all[k, 1])
        los = np.arctan2(y_ts - y_own, x_ts - x_own)
        bearing_deg = float(np.degrees((psi_own - los + np.pi) % (2 * np.pi) - np.pi))
        return bearing_deg, min_dist_nmi, max_risk

    def _veto_reason(self, intent: IntentCommand, multi_env) -> Optional[str]:
        if intent.target_id is not None:
            k = intent.target_id
            dist_nmi = float(multi_env.pair_dist[0, k]) / NMI
            if not np.isfinite(dist_nmi) or dist_nmi > 3.0:
                return "target_inactive"

            if intent.mode == "INITIATE_AVOIDANCE":
                tcpa_s = float(multi_env.pair_tcpa[0, k])
                if tcpa_s <= 0:
                    return "new_maneuver_after_cpa"

            # Port requested during a clear give-way encounter is a COLREG violation.
            if intent.kdir == -1 and intent.rule in (
                "Rule 14 (Head-on)", "Rule 15.1 (Crossing give-way)",
            ):
                return "port_in_give_way_rule"

            # A different, currently higher-risk target should take priority.
            pair_risk = multi_env.pair_risk[0]
            intent_risk = float(pair_risk[k]) if k < pair_risk.shape[0] else 0.0
            other_risks = np.delete(pair_risk, [0, k]) if pair_risk.shape[0] > k else pair_risk
            if other_risks.size > 0 and float(np.max(other_risks)) > intent_risk:
                return "conflicts_with_higher_risk_target"

        return None
