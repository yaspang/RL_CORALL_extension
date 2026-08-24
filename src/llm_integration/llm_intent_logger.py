"""
LLM Intent Logger for RL evaluation pipeline — Stage 1 LLM eval: logging only.

At each LLM decision interval the logger:
  1. Computes per-TS encounter geometry from the live multi-agent environment.
  2. Applies the CORALL rule-based COLREGs classifier (fast, no API required).
  3. Optionally queries the CORALL LLM layer for a COLREGs action recommendation.

The RL policy is NOT affected — this module is purely observational.

Logged fields (one row per active TS encounter per decision interval):
  episode_idx, episode_seed, step, time_s, target_id,
  range_nmi, abs_dcpa_nmi, tcpa_s, relative_bearing_deg, risk,
    colreg_rule, llm_response, parsed_kdir, parsed_kdir_valid

Usage
-----
# In evaluate_policy():
from src.llm_integration.llm_intent_logger import LLMIntentLogger

logger = LLMIntentLogger(use_llm=args.llm, provider=args.llm_provider,
                         interval_s=args.llm_interval)

# In run_one_episode() after env.step():
if logger is not None and logger.should_log(step):
    multi_env = env.env_multi if hasattr(env, 'env_multi') else env.env.env_multi
    logger.log_step(step, float(multi_env.t), episode_idx, seed, multi_env)

# After all episodes:
logger.save(seed_dir / "llm_intent_log.csv")
"""
from __future__ import annotations

import os
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure third_party/CORALL/src is on sys.path before importing CORALL modules,
# making this file self-contained regardless of how the caller bootstrapped paths.
from ..path_setup import ensure_paths  # CHANGED: moved into LLM_integration/ subpackage
ensure_paths()

# ── Constants ─────────────────────────────────────────────────────────────────
NMI: float = 1852.0
_ENCOUNTER_DIST_NMI: float = 3.0   # only log encounters within this range

# Human-readable names for COLREGs rules returned by decision_making()
_COLREG_NAMES: Dict = {
    0:    "No rule",
    13:   "Rule 13 (Overtaking)",
    14:   "Rule 14 (Head-on)",
    15.1: "Rule 15.1 (Crossing give-way)",
    15.2: "Rule 15.2 (Crossing stand-on)",
}

# ── Optional CORALL imports ───────────────────────────────────────────────────
try:
    from decision_making.multi_llm_decision import MultiLLMCOLREGSInterpreter, VesselState
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False
    MultiLLMCOLREGSInterpreter = None  # type: ignore
    VesselState = None                 # type: ignore

try:
    from decision_making.decision_making import decision_making as _colreg_classify
    _COLREG_CLASSIFY_AVAILABLE = True
except ImportError:
    _COLREG_CLASSIFY_AVAILABLE = False
    _colreg_classify = None  # type: ignore


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_kdir(response: str) -> Tuple[int, bool, str]:
    """
    Parse LLM text response to a steering direction integer.

    Returns
    -------
    +1  starboard (give-way turn starboard)
    -1  port      (give-way turn port)
     0  stand on / maintain

        bool flag indicates whether a known action phrase was recognized.
        parse_reason provides an explicit parser outcome label:
            - action_stand_on_like
            - action_starboard
            - action_port
            - action_unrecognized
    """
    r = response.lower()

    # Prefer the explicit Action field if present; ignore explanation text.
    action_text = r
    if "action:" in r:
        action_text = r.split("action:", 1)[1]
        if "explanation:" in action_text:
            action_text = action_text.split("explanation:", 1)[0]

    # Neutral actions must win before checking port/starboard words.
    if (
        "stand on" in action_text
        or "no action" in action_text
        or "continue current" in action_text
        or "maintain" in action_text
    ):
        return 0, True, "action_stand_on_like"

    if "starboard" in action_text:
        return 1, True, "action_starboard"
    if "port" in action_text:
        return -1, True, "action_port"
    return 0, False, "action_unrecognized"


def _is_error_response(response_text: str) -> bool:
    """Return True when the provider returned an error string instead of a decision."""
    response_lower = response_text.lower()
    error_markers = (
        "openai error",
        "claude error",
        "no llm provider",
        "no provider",
        "insufficient_quota",
        "rate_limit",
        "error code:",
        "not available",
    )
    return any(marker in response_lower for marker in error_markers)



# (Stage 2) can build the same prompt without needing an LLMIntentLogger instance.
def build_llm_prompt(active_feats: List[Dict]) -> str:
    """
    Build a CORALL-aligned prompt that anchors the LLM to the rule-based
    classifier and explicitly discourages new avoidance maneuvers after CPA.
    """
    if not active_feats:
        return "No vessels detected - maintain course and speed."

    primary_feat = max(active_feats, key=lambda feat: feat["risk"])
    vessels_text = "\n".join(
        (
            f"- TS{feat['target_id']}: phase={feat['encounter_phase']}, "
            f"range={feat['range_nmi']:.4f} nmi, "
            f"abs_dcpa={feat['abs_dcpa_nmi']:.4f} nmi, "
            f"tcpa={feat['tcpa_s']:.2f} s, "
            f"rel_bearing={feat['relative_bearing_deg']:.2f} deg, "
            f"risk={feat['risk']:.4f}, rule_hint={feat['colreg_rule']}"
        )
        for feat in active_feats
    )

    return f"""You are the decision-making system for an autonomous ship navigating according to COLREGs rules. Your task is to make immediate COLREGs-compliant decisions. Make a COLREGs-compliant decision from the encounter geometry below.

Rule classifier hint for the primary vessel: {primary_feat['colreg_rule']}.
Use this hint unless the numeric encounter data clearly contradicts it.

Relative bearing convention:
- positive relative bearing = target on starboard side
- negative relative bearing = target on port side
- 0 deg = dead ahead
- ±180 deg = dead 

Encounter-phase guidance:
- approaching = TCPA > 0
- post_cpa = TCPA <= 0
- If TCPA is negative or phase is post_cpa, do not initiate a new avoidance maneuver.
  Prefer Continue current maneuver or Stand on, no action unless there is an immediate safety reason.

Active vessels:
{vessels_text}

Respond exactly in this format:
Rule N (description), Action: [Stand on, no action / Give-way, turn to starboard / Give-way, turn to port / Continue current maneuver], Explanation: concise rationale based on the geometry and COLREGs.
"""


def load_env_file(env_path: Optional[str] = None) -> bool:
    """
    Load API keys from a .env file into os.environ (non-overwriting).

    Searches in order:
      1. ``env_path`` if supplied
      2. ``<workspace-root>/third_party/CORALL/.env``
      3. ``.env`` in current working directory
    """
    candidates: List[Optional[str]] = []
    if env_path:
        candidates.append(env_path)
    candidates += [
        str(Path(__file__).parent.parent / "third_party" / "CORALL" / ".env"),
        ".env",
    ]
    for p in candidates:
        if p and Path(p).exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = val
            return True
    return False


# ── Main class ────────────────────────────────────────────────────────────────

class LLMIntentLogger:
    """
    Collects COLREGs intent records during RL evaluation (Stage 1 eval: logging only).

    Parameters
    ----------
    use_llm : bool
        If *True*, query the LLM API for an action recommendation at each
        decision interval.  If *False* (default), only the fast rule-based
        COLREGs classifier is logged — no API calls or keys required.
    provider : str or None
        ``'openai'`` | ``'claude'`` | ``None`` (reads ``LLM_PROVIDER`` env var
        or falls back to first available provider).
    interval_s : float
        Log / LLM-query interval in **simulation seconds** (default 30 s).
        At dt=0.5 s this corresponds to every 60 steps.
    dt : float
        Simulation timestep in seconds (default 0.5).
    env_file : str or None
        Explicit path to a ``.env`` file for API keys.  When ``None`` the
        logger searches for ``third_party/CORALL/.env`` automatically.

    Notes
    -----
    • LLM API calls take ~1–5 s each.  With ``interval_s=30`` and a 300 s
      episode that is ~10 calls per episode.  For large evaluation runs
      (≥ 10 episodes) consider either ``interval_s=60`` or ``--episodes 1``.
    • All logging is append-only; records survive across multiple episodes as
      long as the same ``LLMIntentLogger`` instance is reused.
    """

    def __init__(
        self,
        use_llm: bool = False,
        provider: Optional[str] = None,
        interval_s: float = 30.0,
        dt: float = 0.5,
        env_file: Optional[str] = None,
    ):
        self.use_llm: bool = use_llm and _LLM_AVAILABLE
        self.interval_steps: int = max(1, int(round(interval_s / dt)))
        self._records: List[Dict] = []
        self._interpreter: Any = None

        if self.use_llm:
            load_env_file(env_file)
            try:
                self._interpreter = MultiLLMCOLREGSInterpreter(provider=provider or "openai")
                print(f"[LLMIntentLogger] LLM provider ready: {provider or 'auto'}")
            except Exception as exc:
                print(f"[LLMIntentLogger] WARNING: could not init LLM provider — {exc}")
                self.use_llm = False
        else:
            if not _LLM_AVAILABLE and use_llm:
                print("[LLMIntentLogger] WARNING: 'decision_making.multi_llm_decision' "
                      "not importable; LLM logging disabled.")
            if not _COLREG_CLASSIFY_AVAILABLE:
                print("[LLMIntentLogger] WARNING: 'decision_making.decision_making' "
                      "not importable; COLREG rule classification will be skipped.")

        rule_status = "available" if _COLREG_CLASSIFY_AVAILABLE else "unavailable"
        llm_status = "enabled" if self.use_llm else "disabled (rule-based only)"
        print(
            f"[LLMIntentLogger] init — interval={interval_s}s "
            f"({self.interval_steps} steps), "
            f"COLREG rule-classify={rule_status}, LLM={llm_status}"
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def should_log(self, step: int) -> bool:
        """Return True when ``step`` falls on a logging interval boundary."""
        return (step % self.interval_steps) == 0

    def _build_llm_prompt(self, active_feats: List[Dict]) -> str:
        """Build the CORALL-aligned prompt (delegates to module-level helper so
        llm_intent_service.py can reuse it without an LLMIntentLogger instance)."""
        return build_llm_prompt(active_feats)

    def log_step(
        self,
        step: int,
        time_s: float,
        episode_idx: int,
        episode_seed: int,
        multi_env,
    ) -> int:
        """
        Collect per-TS encounter geometry and append one record per active vessel.

        Mirrors the CORALL design: all active vessels within ``_ENCOUNTER_DIST_NMI``
        are collected first, then a **single** LLM call is made with all of them
        (``MultiLLMCOLREGSInterpreter.make_decision(vessels)``).  The shared LLM
        response and ``llm_call_status`` are written to each per-target row.

        CSV columns added by this method
        ---------------------------------
        n_active_vessels : int
            How many target ships were within encounter range at this interval.
        llm_call_status : str
            ``'disabled'``   — LLM mode off (rule-based only).
            ``'success'``    — API call returned a real response.
            ``'no_provider'``— No API key / provider configured.
            ``'error'``      — Exception during API call.

        Returns
        -------
        int
            Number of records appended (equals number of active vessels).
        """
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

        # ── Pass 1: collect all active vessel features + VesselState objects ─
        # Mirrors run_colm(): build list of VesselState for every active TS,
        # pass them all to make_decision() in one call (multi-vessel design).
        active_feats: List[Dict] = []
        vessel_states: List      = []

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

            # Rule-based COLREGs classification (fast, no API needed)
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
                # approaching = ships still closing (tcpa > 0); post_cpa = CPA passed
                # For Stage 2: only issue new avoidance commands when approaching.
                "encounter_phase":      "approaching" if tcpa_s > 0 else "post_cpa",
            })

            if self.use_llm and _LLM_AVAILABLE and VesselState is not None:
                vessel_states.append(VesselState(
                    risk=risk_k,
                    distance=dist_nmi,
                    bearing=rel_bear_deg,
                    dcpa=dcpa_nmi,
                    tcpa=tcpa_s,
                ))

        if not active_feats:
            return 0

        n_active = len(active_feats)

        # ── Pass 2: single multi-vessel LLM call (mirrors CORALL run_colm) ──
        # One API call per decision interval with ALL active vessels at once.
        llm_response    = "N/A (LLM disabled)"
        parsed_kdir     = 0
        parsed_kdir_valid = 0
        parse_reason = "llm_disabled"
        llm_call_status = "disabled"

        if self.use_llm and self._interpreter is not None and _LLM_AVAILABLE:
            try:
                prompt = self._build_llm_prompt(active_feats)

                provider = getattr(self._interpreter, "provider", None)
                if provider is None:
                    llm_response = "No LLM provider available"
                    llm_call_status = "no_provider"
                    parsed_kdir = 0
                    parsed_kdir_valid = 0
                    parse_reason = "no_provider"
                else:
                    raw_response = provider.generate_response(prompt)
                    provider_name = getattr(self._interpreter, "provider_name", "LLM").upper()
                    llm_response = f"[{provider_name}] {raw_response}"

                    if _is_error_response(llm_response):
                        llm_call_status = (
                            "no_provider"
                            if "no llm provider" in llm_response.lower() or "no provider" in llm_response.lower()
                            else "error"
                        )
                        parsed_kdir = 0
                        parsed_kdir_valid = 0
                        parse_reason = "provider_error_response"
                    else:
                        llm_call_status = "success"
                        parsed_kdir, parsed_ok, parse_reason = _extract_kdir(llm_response)
                        parsed_kdir_valid = 1 if parsed_ok else 0
            except Exception as exc:
                llm_response    = f"LLM error: {exc}"
                llm_call_status = "error"
                parsed_kdir     = 0
                parsed_kdir_valid = 0
                parse_reason = "provider_exception"
        elif self.use_llm:
            # use_llm requested but interpreter has no provider (no API key)
            llm_call_status = "no_provider"
            parsed_kdir_valid = 0
            parse_reason = "no_provider"

        # ── Pass 3: write one CSV row per active target, shared LLM result ──
        for feat in active_feats:
            self._records.append({
                "episode_idx":      episode_idx,
                "episode_seed":     episode_seed,
                "step":             step,
                "time_s":           round(time_s, 2),
                "n_active_vessels": n_active,
                **feat,
                "llm_call_status":  llm_call_status,
                "llm_response":     llm_response,
                "parsed_kdir":      parsed_kdir,
                "parsed_kdir_valid": parsed_kdir_valid,
                "parse_reason":     parse_reason,
            })

        return n_active

    def get_records(self) -> List[Dict]:
        """Return all accumulated log records."""
        return self._records

    def save(self, csv_path: Path) -> None:
        """Write all accumulated records to *csv_path*."""
        if not self._records:
            print("[LLMIntentLogger] No records to save.")
            return
        fieldnames = list(self._records[0].keys())
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._records)
        print(
            f"[LLMIntentLogger] Intent log saved: {csv_path}  "
            f"({len(self._records)} records)"
        )

    def reset(self) -> None:
        """Clear accumulated records (does not affect interpreter state)."""
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"LLMIntentLogger(use_llm={self.use_llm}, "
            f"interval_steps={self.interval_steps}, "
            f"records={len(self._records)})"
        )
