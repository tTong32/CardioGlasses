"""Alert decision: rules choose the level, the explainer writes the words.

A Reading only counts as evidence when the signal is trusted (HR present, quality >=
min_quality, signal_status "ok"), the wearer is resting, and a baseline exists. Then:

- Elevated means deviation >= deviation_trigger (in SDs of the resting baseline).
- Elevated HR right after exertion is expected. It is "explained" while a recovery episode
  is running with a normal (or not-yet-known) verdict, or within POST_EXERTION_GRACE_S of
  the last movement. Explained elevation never raises the level.
- A slow recovery verdict raises to monitor, a very slow one to notify.
- Unexplained elevation: monitor after MONITOR_AFTER_S, notify after persist_s, escalate
  after 3 x persist_s if deviation is also >= 2 x trigger.
- Back to normal only after deviation < 1 SD for DEVIATION_NORMAL_DURATION_S.
- Untrusted, moving, or uncalibrated readings hold the current level; they never raise it.
- After returning to normal, the same or a lower level can't be raised again within
  cooldown_s (a higher one can). Alerts are emitted only when the level changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import ai.config as cfg
from ai.contracts import Alert, Level, PatientContext, Reading

POST_EXERTION_GRACE_S = 60.0
EVIDENCE_HOLD_S = 10.0  # the return-to-normal timer survives untrusted readings this long
MONITOR_AFTER_S = 10.0
ESCALATE_PERSIST_MULTIPLIER = 3.0
SEVERITY = {"normal": 0, "monitor": 1, "notify": 2, "escalate": 3}


@dataclass
class Context:
    """What the pipeline knows beyond the Reading itself."""

    recovering: bool = False  # a recovery episode is active
    live_verdict: Optional[str] = None  # verdict of the active episode only
    seconds_since_moving: Optional[float] = None


@dataclass
class Target:
    level: Level
    kind: str  # see explainer.KINDS
    confidence: float


class DecisionEngine:
    def __init__(self, context: PatientContext, explainer=None):
        self.context = context
        self.explainer = explainer
        self.level: Level = "normal"
        self._normal_since: Optional[int] = None
        self._episode_peak = 0  # highest severity since the last return to normal
        self._cooldown_until: Optional[int] = None
        self._cooldown_cap = 0
        self._last_evidence_t: Optional[int] = None
        self.last_kind: Optional[str] = None
        self._held_artifact = False  # one monitor/notify already held for a messy signal this episode

    def trusted(self, reading: Reading) -> bool:
        return (
            reading.hr is not None
            and reading.quality is not None
            and reading.quality >= self.context.config.min_quality
            and reading.signal_status in (None, "ok")
        )

    def _target(self, reading: Reading, extra: Context) -> Optional[Target]:
        """The level this reading argues for, or None to hold."""
        cfg_p = self.context.config
        if not self.trusted(reading) or reading.activity != "resting" or reading.deviation is None:
            return None
        dev = reading.deviation
        persist = reading.persist_s or 0.0
        quality = reading.quality or 0.0
        if dev < cfg.DEVIATION_NORMAL_THRESHOLD:
            return Target("normal", "back_to_normal", round(quality, 2))
        if dev < cfg_p.deviation_trigger:
            return None

        sustained = persist >= ESCALATE_PERSIST_MULTIPLIER * cfg_p.persist_s
        far = dev >= cfg.DEVIATION_ESCALATE_MULTIPLIER * cfg_p.deviation_trigger
        if extra.recovering and extra.live_verdict == "very_slow":
            return Target("escalate" if sustained and far else "notify", "slow_recovery", round(quality * 0.85, 2))
        if extra.recovering and extra.live_verdict == "slow":
            return Target("monitor", "slow_recovery", round(quality * 0.8, 2))
        explained = extra.recovering or (
            extra.seconds_since_moving is not None and extra.seconds_since_moving < POST_EXERTION_GRACE_S
        )
        if explained:
            return None

        evidence = min(1.0, 0.5 + 0.5 * persist / cfg_p.persist_s)
        confidence = round(quality * evidence, 2)
        if sustained and far:
            level: Level = "escalate"
        elif persist >= cfg_p.persist_s:
            level = "notify"
        elif persist >= MONITOR_AFTER_S:
            level = "monitor"
        else:
            return None
        if SEVERITY[level] >= SEVERITY["notify"] and confidence < cfg.CONFIDENCE_MIN_FOR_ALERT:
            level = "monitor"  # unsure: watch, don't alarm
        return Target(level, "unexplained_elevation", confidence)

    def update(self, reading: Reading, extra: Optional[Context] = None) -> Optional[Alert]:
        """Feed one Reading. Returns an Alert only when the level changes."""
        extra = extra or Context()
        evidence = self.trusted(reading) and reading.activity == "resting" and reading.deviation is not None
        if evidence:
            self._last_evidence_t = reading.t
        elif self._last_evidence_t is None or (reading.t - self._last_evidence_t) / 1000.0 > EVIDENCE_HOLD_S:
            self._normal_since = None  # too long without evidence: start the count again
        target = self._target(reading, extra)
        if target is None:
            if evidence:
                self._normal_since = None  # trusted but between normal and elevated
            return None

        if target.level == "normal":
            if self.level == "normal":
                return None
            if self._normal_since is None:
                self._normal_since = reading.t
            if (reading.t - self._normal_since) / 1000.0 < cfg.DEVIATION_NORMAL_DURATION_S:
                return None
            self._cooldown_until = reading.t + int(self.context.config.cooldown_s * 1000)
            self._cooldown_cap = self._episode_peak
            self._episode_peak = 0
            return self._change(target, reading)

        self._normal_since = None
        if SEVERITY[target.level] <= SEVERITY[self.level]:
            return None  # only raise here; lowering goes through "normal"
        in_cooldown = self._cooldown_until is not None and reading.t < self._cooldown_until
        if self.level == "normal" and in_cooldown and SEVERITY[target.level] <= self._cooldown_cap:
            return None
        self._episode_peak = max(self._episode_peak, SEVERITY[target.level])
        return self._change(target, reading)

    def _signal_check(self, target: Target, reading: Reading):
        """Second opinion for an alert that asks for attention. Normal has nothing to cross-check."""
        if self.explainer is None or target.level == "normal":
            return None
        return self.explainer.check_signal(reading)

    def _hold_for_artifact(self, target: Target, check) -> bool:
        """One cycle only, and only when SIGNAL_CHECK_DELAY is on. Escalate always goes out."""
        return bool(
            check is not None
            and check.verdict == "artifact"
            and target.level in ("monitor", "notify")
            and not self._held_artifact
            and cfg.signal_check_may_delay()
        )

    def _change(self, target: Target, reading: Reading) -> Optional[Alert]:
        check = self._signal_check(target, reading)
        if self._hold_for_artifact(target, check):
            self._held_artifact = True
            return None
        self.level = target.level
        self.last_kind = target.kind
        if target.level == "normal":
            self._held_artifact = False
        if self.explainer is not None:
            words = self.explainer.explain(target.level, target.kind, reading, self.context)
        else:
            from ai.explainer import template

            words = template(target.level, target.kind, reading, self.context)
        return Alert(
            t=reading.t,
            level=target.level,
            confidence=target.confidence,
            headline=words.headline,
            reason=words.reason,
            voice_text=words.voice_text,
            next_step=words.next_step,
            reading=reading,
            signal_check=None if check is None else check.verdict,
            signal_check_note=None if check is None else check.note,
        )


_engines: dict[str, DecisionEngine] = {}


def evaluate(reading: Reading, context: PatientContext) -> Optional[Alert]:
    """Stateless-looking wrapper kept for the original API (one engine per patient).

    It has no recovery context, so prefer DecisionEngine via ai.pipeline.Pipeline.
    """
    engine = _engines.setdefault(context.patient_id, DecisionEngine(context))
    return engine.update(reading)
