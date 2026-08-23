"""
LLM Intent Service — Stage 2: intent generation, separated from logging/arbitration.

# CHANGED (new file): extracted the "generate an intent" responsibilities out of
# LLMIntentLogger (llm_intent_logger.py), which used to extract geometry, run the
# rule classifier, build the prompt, call the API, parse the answer, AND write CSV
# rows all in one method. This service only creates/updates IntentCommand objects;
# src.llm_integration.intent_arbiter applies them to RL actions and
# src.llm_integration.execution_logger records what was requested vs. applied.

Usage
-----
    from src.llm_integration.llm_intent_service import LLMIntentService

    intent_service = LLMIntentService(use_llm=args.llm, provider=args.llm_provider,
                                       interval_s=args.llm_interval, dt=args.dt,
                                       env_file=args.llm_env_file)

    # Once per control step, before env.step():
    intent_service.maybe_request_update(step, time_s, episode_idx, seed, multi_env)
    active_intent = intent_service.get_active_intent(time_s, multi_env)

    # After all episodes:
    intent_service.save(seed_dir / "llm_intent_log.csv")
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np

from ..path_setup import ensure_paths  # CHANGED: moved into LLM_integration/ subpackage
ensure_paths()

# Reuse the existing provider bootstrap, prompt builder, and response parser
# instead of duplicating them (item 1: separate generation from logging, not
# from the CORALL-specific parsing/prompting logic that already works).
from .llm_intent_logger import (
    NMI,
    _ENCOUNTER_DIST_NMI,
    _COLREG_NAMES,
    _LLM_AVAILABLE,
    _COLREG_CLASSIFY_AVAILABLE,
    MultiLLMCOLREGSInterpreter,
    VesselState,
    _colreg_classify,
    _extract_kdir,
    _is_error_response,
    build_llm_prompt,
    load_env_file,
)

IntentMode = Literal[
    "INITIATE_AVOIDANCE",
    "CONTINUE_AVOIDANCE",
    "STAND_ON",
    "RESUME_ROUTE",
]


@dataclass(frozen=True)
class IntentCommand:
    command_id: str
    source_time_s: float
    issued_time_s: float
    target_id: Optional[int]
    rule: str
    mode: IntentMode
    kdir: int              # +1 starboard, -1 port, 0 neutral
    valid: bool
    confidence: float
    valid_for_s: float
    rationale: str


class LLMIntentService:
    """
    Generates :class:`IntentCommand` objects at each decision interval and keeps
    the most recent one available between calls (item 3: persist intent between
    LLM calls, since the LLM is queried far less often than the control loop).
    """

    _command_counter = itertools.count(1)

    def __init__(
        self,
        use_llm: bool = False,
        provider: Optional[str] = None,
        interval_s: float = 30.0,
        dt: float = 0.5,
        env_file: Optional[str] = None,
    ):
        self.use_llm: bool = use_llm and _LLM_AVAILABLE
        self.interval_s: float = float(interval_s)
        self.interval_steps: int = max(1, int(round(interval_s / dt)))
        self._interpreter: Optional[MultiLLMCOLREGSInterpreter] = None
        self._active_intent: Optional[IntentCommand] = None
        self._records: List[Dict] = []

        if self.use_llm:
            load_env_file(env_file)
            try:
                self._interpreter = MultiLLMCOLREGSInterpreter(provider=provider)
                print(f"[LLMIntentService] LLM provider ready: {provider or 'auto'}")
            except Exception as exc:
                print(f"[LLMIntentService] WARNING: could not init LLM provider — {exc}")
                self.use_llm = False
        else:
            if not _LLM_AVAILABLE and use_llm:
                print("[LLMIntentService] WARNING: 'decision_making.multi_llm_decision' "
                      "not importable; LLM intent generation disabled.")
            if not _COLREG_CLASSIFY_AVAILABLE:
                print("[LLMIntentService] WARNING: 'decision_making.decision_making' "
                      "not importable; COLREG rule classification will be skipped.")

        rule_status = "available" if _COLREG_CLASSIFY_AVAILABLE else "unavailable"
        llm_status = "enabled" if self.use_llm else "disabled (rule-based only)"
        print(
            f"[LLMIntentService] init — interval={interval_s}s "
            f"({self.interval_steps} steps), "
            f"COLREG rule-classify={rule_status}, LLM={llm_status}"
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def reset_episode(self) -> None:
        """Clear the persisted intent so episode n+1 doesn't inherit episode n's state."""
        self._active_intent = None

    def should_query(self, step: int) -> bool:
        """Return True when ``step`` falls on a decision-interval boundary."""
        return (step % self.interval_steps) == 0

    def maybe_request_update(
        self,
        step: int,
        time_s: float,
        episode_idx: int,
        episode_seed: int,
        multi_env,
    ) -> Optional[IntentCommand]:
        """Query the LLM (if it's an interval boundary) and refresh the active intent."""
        if not self.should_query(step):
            return self._active_intent

        active_feats, vessel_states = self._collect_encounter_geometry(multi_env)
        if not active_feats:
            return self._active_intent

        primary_feat = max(active_feats, key=lambda feat: feat["risk"])
        llm_response, llm_call_status, parsed_kdir, parsed_ok, parse_reason = (
            self._call_llm(active_feats)
        )
        mode = self._derive_mode(primary_feat, llm_call_status, parsed_kdir, parsed_ok)
        valid = bool(llm_call_status == "success" and parsed_ok)

        command = IntentCommand(
            command_id=f"intent-{next(self._command_counter)}",
            source_time_s=float(time_s),
            issued_time_s=float(time_s),   # synchronous call: issued == source
            target_id=int(primary_feat["target_id"]),
            rule=str(primary_feat["colreg_rule"]),
            mode=mode,
            kdir=int(parsed_kdir),
            valid=valid,
            confidence=1.0 if valid else 0.0,
            valid_for_s=self.interval_s,
            rationale=str(llm_response),
        )
        self._active_intent = command

        for feat in active_feats:
            self._records.append({
                "episode_idx":       episode_idx,
                "episode_seed":      episode_seed,
                "step":              step,
                "time_s":            round(time_s, 2),
                "n_active_vessels":  len(active_feats),
                **feat,
                "llm_call_status":   llm_call_status,
                "llm_response":      llm_response,
                "parsed_kdir":       parsed_kdir,
                "parsed_kdir_valid": int(parsed_ok),
                "parse_reason":      parse_reason,
                "command_id":        command.command_id,
                "intent_mode":       command.mode,
                "intent_valid_for_s": command.valid_for_s,
            })

        return command

    def get_active_intent(self, time_s: float, multi_env=None) -> Optional[IntentCommand]:
        """
        Return the current intent if it is still valid, else ``None``.

        Rejects (returns ``None``) when:
          - no intent has been issued yet,
          - the intent is stale (item 8's staleness formula: age > valid_for_s),
          - the intent was never valid (unparseable / failed LLM call),
          - (when ``multi_env`` is given) its target is no longer within
            encounter range, or its CPA has already passed for an avoidance mode.
        """
        intent = self._active_intent
        if intent is None or not intent.valid:
            return None

        intent_age = float(time_s) - intent.source_time_s
        if intent_age > intent.valid_for_s:
            return None  # stale — item 8

        if multi_env is not None and intent.target_id is not None:
            k = intent.target_id
            dist_nmi = float(multi_env.pair_dist[0, k]) / NMI
            if not np.isfinite(dist_nmi) or dist_nmi > _ENCOUNTER_DIST_NMI:
                return None  # target no longer active — item 3 / item 6
            if intent.mode in ("INITIATE_AVOIDANCE", "CONTINUE_AVOIDANCE"):
                tcpa_s = float(multi_env.pair_tcpa[0, k])
                if tcpa_s <= 0:
                    return None  # CPA passed — item 3

        return intent

    def latest_valid_intent(self, current_time_s: float) -> Optional[IntentCommand]:
        """Signature-compatible convenience wrapper (no target-liveness checks)."""
        return self.get_active_intent(time_s=current_time_s, multi_env=None)

    def get_records(self) -> List[Dict]:
        return self._records

    def save(self, csv_path: Path) -> None:
        if not self._records:
            print("[LLMIntentService] No records to save.")
            return
        import csv
        fieldnames = list(self._records[0].keys())
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._records)
        print(f"[LLMIntentService] Intent log saved: {csv_path} ({len(self._records)} records)")

    def __len__(self) -> int:
        return len(self._records)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _collect_encounter_geometry(self, multi_env):
        """Same Pass-1 geometry extraction previously inlined in LLMIntentLogger.log_step."""
        X_all     = multi_env.X_all
        pair_dist = multi_env.pair_dist
        pair_dcpa = multi_env.pair_dcpa
        pair_tcpa = multi_env.pair_tcpa
        pair_risk = multi_env.pair_risk

        n_agents = X_all.shape[0]
        own      = 0
        x_own    = float(X_all[own, 0])
        y_own    = float(X_all[own, 1])
        psi_own  = float(X_all[own, 2])
        u_own    = float(X_all[own, 5])

        active_feats: List[Dict] = []
        vessel_states: List = []

        for k in range(1, n_agents):
            dist_m   = float(pair_dist[own, k])
            dist_nmi = dist_m / NMI
            if not np.isfinite(dist_m) or dist_nmi > _ENCOUNTER_DIST_NMI:
                continue

            x_ts   = float(X_all[k, 0])
            y_ts   = float(X_all[k, 1])
            psi_ts = float(X_all[k, 2])
            u_ts   = float(X_all[k, 5])

            dcpa_nmi = abs(float(pair_dcpa[own, k])) / NMI
            tcpa_s   = float(pair_tcpa[own, k])
            risk_k   = float(pair_risk[own, k])

            colreg_rule  = "N/A"
            rel_bear_deg = float("nan")
            if _COLREG_CLASSIFY_AVAILABLE and _colreg_classify is not None:
                try:
                    v_rel = float(np.hypot(
                        u_own * np.cos(psi_own) - u_ts * np.cos(psi_ts),
                        u_own * np.sin(psi_own) - u_ts * np.sin(psi_ts),
                    ))
                    colreg_no, _, _, rel_bearing_arr = _colreg_classify(
                        x_own, y_own, psi_own,
                        [x_ts], [y_ts], [psi_ts],
                        v_rel, u_own, [risk_k],
                    )
                    rel_bear_deg = float(np.degrees(rel_bearing_arr[0]))
                    colreg_rule  = _COLREG_NAMES.get(colreg_no, f"Rule {colreg_no}")
                except Exception:
                    los          = np.arctan2(y_ts - y_own, x_ts - x_own)
                    rel_bear_deg = float(np.degrees(
                        (psi_own - los + np.pi) % (2 * np.pi) - np.pi
                    ))
            else:
                los          = np.arctan2(y_ts - y_own, x_ts - x_own)
                rel_bear_deg = float(np.degrees(
                    (psi_own - los + np.pi) % (2 * np.pi) - np.pi
                ))

            active_feats.append({
                "target_id":            k,
                "range_nmi":            round(dist_nmi, 4),
                "abs_dcpa_nmi":         round(dcpa_nmi, 4),
                "tcpa_s":               round(tcpa_s, 2),
                "relative_bearing_deg": round(rel_bear_deg, 2),
                "risk":                 round(risk_k, 4),
                "colreg_rule":          colreg_rule,
                "encounter_phase":      "approaching" if tcpa_s > 0 else "post_cpa",
            })

            if self.use_llm and _LLM_AVAILABLE and VesselState is not None:
                vessel_states.append(VesselState(
                    risk=risk_k, distance=dist_nmi, bearing=rel_bear_deg,
                    dcpa=dcpa_nmi, tcpa=tcpa_s,
                ))

        return active_feats, vessel_states

    def _call_llm(self, active_feats: List[Dict]):
        """Same Pass-2 single multi-vessel LLM call previously inlined in log_step."""
        if not (self.use_llm and self._interpreter is not None and _LLM_AVAILABLE):
            return "N/A (LLM disabled)", "disabled", 0, False, "llm_disabled"

        try:
            prompt = build_llm_prompt(active_feats)
            provider = getattr(self._interpreter, "provider", None)
            if provider is None:
                return "No LLM provider available", "no_provider", 0, False, "no_provider"

            raw_response = provider.generate_response(prompt)
            provider_name = getattr(self._interpreter, "provider_name", "LLM").upper()
            llm_response = f"[{provider_name}] {raw_response}"

            if _is_error_response(llm_response):
                status = (
                    "no_provider"
                    if "no llm provider" in llm_response.lower() or "no provider" in llm_response.lower()
                    else "error"
                )
                return llm_response, status, 0, False, "provider_error_response"

            parsed_kdir, parsed_ok, parse_reason = _extract_kdir(llm_response)
            return llm_response, "success", parsed_kdir, parsed_ok, parse_reason
        except Exception as exc:
            return f"LLM error: {exc}", "error", 0, False, "provider_exception"

    def _derive_mode(self, primary_feat: Dict, llm_call_status: str,
                      parsed_kdir: int, parsed_ok: bool) -> IntentMode:
        """
        Map the free-text parser output to a mode + direction (item 2) instead of
        collapsing everything to ``kdir=0``, which conflated "stand on", "resume
        route", and "unrecognized" into a single indistinguishable value.
        """
        if llm_call_status != "success" or not parsed_ok:
            return "STAND_ON"

        prev = self._active_intent
        if primary_feat["encounter_phase"] == "post_cpa":
            if prev is not None and prev.target_id == primary_feat["target_id"] \
                    and prev.mode in ("INITIATE_AVOIDANCE", "CONTINUE_AVOIDANCE"):
                return "RESUME_ROUTE"
            return "STAND_ON"

        if parsed_kdir == 0:
            return "STAND_ON"

        if prev is not None and prev.target_id == primary_feat["target_id"] \
                and prev.mode in ("INITIATE_AVOIDANCE", "CONTINUE_AVOIDANCE") \
                and prev.kdir == parsed_kdir:
            return "CONTINUE_AVOIDANCE"
        return "INITIATE_AVOIDANCE"
