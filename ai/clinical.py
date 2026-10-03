"""How the clinical record (FinchNode, Contract D) changes monitoring.

One place for every record-driven rule, used by the backend (to build Contract D),
the pipeline, and the dashboard (to show what the record changed):

1. Risk tier sets alert sensitivity: higher risk alerts on a smaller rise, sooner.
2. A recorded irregular rhythm (atrial fibrillation, flutter, arrhythmia) switches signal
   processing to irregular-rhythm mode, so the rhythm itself isn't mistaken for noise.
3. Rate-controlling medication (e.g. metoprolol) selects the recovery comparison group
   (handled in ai.recovery) and gives the wording context (ai.explainer).
"""

from __future__ import annotations

import ai.config as ai_cfg
from ai.contracts import Config, PatientContext

# deviation_trigger is in SDs of the resting baseline; persist_s is how long it must last.
TIER_SENSITIVITY = {
    "high": {"deviation_trigger": 2.0, "persist_s": 30.0},
    "medium": {"deviation_trigger": 2.5, "persist_s": 45.0},
    "low": {"deviation_trigger": 3.0, "persist_s": 60.0},
}
IRREGULAR_RHYTHM_TERMS = ("atrial fibrillation", "atrial flutter", "arrhythmia", "irregular heart")
# With an irregular rhythm, window-to-window HR swings ~4-6 bpm on its own, so the
# baseline spread gets a wider floor; otherwise ordinary AF variation reads as deviation.
IRREGULAR_BASELINE_MIN_STD = 6.0
# Clinic readings aren't home resting conditions, so until the glasses measure the real
# usual rate, the clinic value only counts with a wider spread floor (a bigger rise).
PROVISIONAL_MIN_STD = 8.0
RATE_CONTROL_TERMS = (
    "metoprolol", "bisoprolol", "carvedilol", "atenolol", "propranolol", "nebivolol",
    "diltiazem", "verapamil", "digoxin", "amiodarone", "ivabradine",
)


def tuned_config(base: Config, risk_tier: str) -> Config:
    """Contract D thresholds for this tier; min_quality and cooldown stay as configured."""
    return base.model_copy(update=TIER_SENSITIVITY.get(risk_tier, {}))


def irregular_rhythm_expected(context: PatientContext) -> bool:
    return any(term in c.lower() for c in context.conditions for term in IRREGULAR_RHYTHM_TERMS)


def baseline_min_std(context: PatientContext) -> float:
    return IRREGULAR_BASELINE_MIN_STD if irregular_rhythm_expected(context) else ai_cfg.BASELINE_MIN_STD


def rate_control_medications(context: PatientContext) -> list[str]:
    return [m for m in context.medications if any(term in m.lower() for term in RATE_CONTROL_TERMS)]


def _short_med(name: str) -> str:
    words = [w for w in name.split() if w.isalpha() and w.lower() not in ("hr", "mg", "oral", "tablet", "extended", "release")]
    hit = next((w for w in words if any(t in w.lower() for t in RATE_CONTROL_TERMS)), None)
    return (hit or name).lower()


def effects(context: PatientContext) -> list[str]:
    """Plain-English list of what the record changes, for the dashboard."""
    cfg = context.config
    # The baseline spread is floored at BASELINE_MIN_STD bpm, so this is the smallest rise that counts.
    min_rise = round(cfg.deviation_trigger * baseline_min_std(context))
    out = [
        f"{context.risk_tier.capitalize()} risk: alerts if your resting heart rate stays "
        f"{min_rise}+ bpm above your usual for {cfg.persist_s:g} seconds."
    ]
    rhythm = [c for c in context.conditions if any(t in c.lower() for t in IRREGULAR_RHYTHM_TERMS)]
    if rhythm:
        out.append(f"{rhythm[0]}: an irregular beat is expected, so it isn't treated as a bad signal.")
    if context.clinic_resting_hr is not None:
        out.append(
            f"Clinic heart rate {context.clinic_resting_hr:.0f}: used as your starting point until the "
            "glasses learn your usual rate (about a minute of sitting still)."
        )
    meds = rate_control_medications(context)
    if meds:
        out.append(f"On {_short_med(meds[0])}: recovery after activity is compared with people on similar medication.")
    return out
