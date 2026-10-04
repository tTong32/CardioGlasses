"""Alert wording. The rules pick the level; this only writes the words.

Gemini writes `reason` and `voice_text` from the facts we give it. Headline and next step
come from a fixed set so the action is never invented. If Gemini is unavailable, slow,
or returns something unusable, the templated wording is used instead.
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel

import ai.config as cfg
from ai.clinical import rate_control_medications
from ai.contracts import Level, PatientContext, Reading

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
log = logging.getLogger(__name__)

# Why the level changed; decided by the rules, passed to the wording.
KINDS = ("unexplained_elevation", "slow_recovery", "back_to_normal")

HEADLINES = {
    ("monitor", "unexplained_elevation"): "Heart rate a little high at rest",
    ("notify", "unexplained_elevation"): "Elevated heart rate at rest",
    ("escalate", "unexplained_elevation"): "Heart rate has stayed high",
    ("monitor", "slow_recovery"): "Slower recovery than usual",
    ("notify", "slow_recovery"): "Heart rate slow to settle",
    ("escalate", "slow_recovery"): "Heart rate still high after activity",
    ("normal", "back_to_normal"): "Back within your usual range",
}
NEXT_STEPS = {
    "normal": "No action needed",
    "monitor": "Stay seated and breathe normally",
    "notify": "Sit down and check your phone",
    "escalate": "Sit down and contact your care team",
}
# The spoken instruction is fixed per level; Gemini only describes what was measured.
VOICE_ACTIONS = {
    "normal": "I'll keep monitoring.",
    "monitor": "Stay seated and rest for now.",
    "notify": "Please sit down and check your phone.",
    "escalate": "Please sit down and contact your care team.",
}
CARDIAC_TERMS = ("atrial fibrillation", "heart failure", "coronary", "arrhythmia", "cardiomyopathy", "hypertension")
GEMINI_DEADLINE_MS = 10_000  # the API's minimum allowed deadline
FORBIDDEN = re.compile(r"heart attack|cardiac arrest|stroke|diagnos|emergency|911|you have (a|an) ", re.I)

SYSTEM_PROMPT = """You write short messages for CardioGlasses, a wellness monitor worn by an older adult with heart disease.
The monitoring rules have already chosen the alert level; you only describe what was measured, in plain, calm English.
Rules:
- Never diagnose or name a medical event (no "heart attack", "arrhythmia episode", "stroke").
- Never give instructions or advice; the next step is fixed and added separately.
- Never say what a medication does or how it affects them. You may mention one relevant condition or medication from the record only as context ("with your atrial fibrillation in mind").
- reason: 1-2 sentences, at most 45 words, second person, for the phone screen. Include the current and usual heart rate and how long it has lasted when given. When signal_caveat is present, work it in as one short clause about the measurement.
- voice_text: ONE short spoken sentence of at most {max_words} words saying what changed. No instructions, no decimals. Do not repeat the signal caveat.
"""


class Wording(BaseModel):
    reason: str
    voice_text: str


class SignalVerdict(BaseModel):
    artifact_suspected: bool
    why: str


@dataclass
class Explanation:
    headline: str
    reason: str
    voice_text: str
    next_step: str
    source: str  # "gemini", "cache", or "template"


@dataclass
class SignalCheck:
    """A second opinion on the pulse. The alert is sent either way."""

    verdict: str  # "clean", "artifact", or "unavailable"
    note: str


LOCAL_CLEAN = "Pulse looks steady."
CHECK_UNAVAILABLE = "Signal check didn't run. The alert still stands."
CHECK_PROMPT = """You cross-check one heart-rate alert for CardioGlasses. The monitoring rules have already chosen the alert. You only say whether the pulse measurement looks too messy to trust.
Rules:
- artifact_suspected is true only when the quality score is modest or signal_caveat describes clipping, missed beats, or a measurement problem.
- A strong quality score with no caveat is not an artifact.
- why: at most 12 words, plain English. No diagnosis, no advice, no instructions.
"""


def relevant_history(context: PatientContext) -> tuple[list[str], list[str]]:
    conditions = [c for c in context.conditions if any(t in c.lower() for t in CARDIAC_TERMS)]
    meds = rate_control_medications(context)
    return conditions[:2], meds[:1]


def _history_clause(context: PatientContext) -> str:
    conditions, _ = relevant_history(context)
    if not conditions:
        return ""
    return f" Your record includes {' and '.join(c.lower() for c in conditions)}, so this is worth a check."


def template(level: Level, kind: str, reading: Reading, context: PatientContext) -> Explanation:
    hr = round(reading.hr) if reading.hr is not None else None
    usual = round(reading.baseline_hr) if reading.baseline_hr is not None else None
    held = int(reading.persist_s or 0)
    now = f"{hr} bpm" if hr is not None else "higher than usual"
    usual_text = f"your usual resting rate of {usual}" if usual is not None else "your usual resting rate"
    history = _history_clause(context)
    if kind == "back_to_normal":
        reason = f"Your heart rate is {now}, close to {usual_text}. Monitoring continues."
        voice = "Your heart rate is back within your usual range. I'll keep monitoring."
    elif kind == "slow_recovery":
        reason = f"After activity your heart rate is settling more slowly than expected; it's {now} compared with {usual_text}.{history}"
        voice = {
            "monitor": "Your heart rate is settling a little slowly. Stay seated and rest.",
            "notify": "Your heart rate is taking a while to settle. Please sit down and check your phone.",
            "escalate": "Your heart rate is still high after activity. Please sit down and contact your care team.",
        }[level]
    else:
        if level == "monitor":
            reason = f"Your heart rate is {now} while you're sitting still, above {usual_text}. I'm watching to see if it settles."
            voice = "Your heart rate is a little high while resting. I'm keeping an eye on it."
        else:
            reason = f"Your heart rate has stayed around {now} for {held} seconds while resting, well above {usual_text}.{history}"
            voice = (
                "Your heart rate has stayed high while you're resting. Please sit down and check your phone."
                if level == "notify"
                else "Your heart rate has stayed high for a while. Please sit down and contact your care team."
            )
    return Explanation(HEADLINES[(level, kind)], _include_note(reason, reading), voice, NEXT_STEPS[level], "template")


def _include_note(reason: str, reading: Reading) -> str:
    """Keep a measurement caveat in the screen text when the model leaves it out."""
    note = reading.signal_note
    if not note:
        return reason
    if note[:24].lower() in reason.lower():
        return reason
    return f"{reason.rstrip()} {note}"


OBSERVATION_MAX_WORDS = 16  # plus the fixed action sentence stays within EXPLAINER_MAX_VOICE_WORDS
ADVICE = re.compile(r"\b(please|let's|let us|take a|sit down|call|contact|you should|try to)\b", re.I)


def _valid(wording: Wording) -> bool:
    words = len(wording.voice_text.split())
    return (
        0 < words <= OBSERVATION_MAX_WORDS
        and not ADVICE.search(wording.voice_text)
        and 0 < len(wording.reason.split()) <= 60
        and not FORBIDDEN.search(wording.reason + " " + wording.voice_text)
    )


class Explainer:
    """Gemini wording with a cache and a template fallback. Never raises."""

    def __init__(self, use_llm: bool = True, model: Optional[str] = None, timeout_s: Optional[float] = None):
        self.model = model or os.environ.get("GEMINI_MODEL") or cfg.EXPLAINER_MODEL
        self.timeout_s = timeout_s if timeout_s is not None else cfg.EXPLAINER_TIMEOUT_S
        self._cache: dict[tuple, Wording] = {}
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="explainer")
        self._client = None
        api_key = os.environ.get("GEMINI_API_KEY")
        if use_llm and api_key:
            try:
                from google import genai
                from google.genai import types  # noqa: F401  (load now, not inside the first alert's budget)

                self._client = genai.Client(api_key=api_key)
            except Exception as exc:  # missing package, bad key format
                log.warning("Gemini unavailable, using templates: %s", exc)

    @property
    def uses_llm(self) -> bool:
        return self._client is not None

    def explain(self, level: Level, kind: str, reading: Reading, context: PatientContext) -> Explanation:
        fallback = template(level, kind, reading, context)
        if self._client is None:
            return fallback
        bucket = cfg.EXPLAINER_HR_BUCKET
        hr_key = None if reading.hr is None else int(reading.hr // bucket) * bucket
        key = (level, kind, hr_key, reading.recovery_verdict, int((reading.persist_s or 0) // 30))
        if key in self._cache:
            wording, source = self._cache[key], "cache"
        else:
            # Gemini's minimum request deadline is 10 s; we won't hold an alert that long.
            future = self._pool.submit(self._ask, level, kind, reading, context)
            try:
                wording = future.result(timeout=self.timeout_s)
            except FutureTimeout:
                log.warning("Gemini slower than %.0fs, using template (late answer will be cached)", self.timeout_s)
                future.add_done_callback(lambda f: f.result() and self._cache.setdefault(key, f.result()))
                return fallback
            if wording is None:
                return fallback
            self._cache[key] = wording
            source = "gemini"
        voice = f"{wording.voice_text.rstrip()} {VOICE_ACTIONS[level]}"
        return Explanation(fallback.headline, _include_note(wording.reason, reading), voice, fallback.next_step, source)

    def check_signal(self, reading: Reading) -> SignalCheck:
        """Say whether this alert's pulse looks messy. Never raises. A strong clean window skips the network."""
        quality = reading.quality or 0.0
        if quality >= cfg.SIGNAL_CHECK_STRONG and not reading.signal_note:
            return SignalCheck("clean", LOCAL_CLEAN)
        if self._client is None:
            return SignalCheck("unavailable", CHECK_UNAVAILABLE)
        future = self._pool.submit(self._ask_signal, reading)
        try:
            verdict = future.result(timeout=cfg.SIGNAL_CHECK_TIMEOUT_S)
        except FutureTimeout:
            log.warning("Signal check slower than %.1fs; the alert still stands", cfg.SIGNAL_CHECK_TIMEOUT_S)
            future.add_done_callback(lambda done: done.exception())
            return SignalCheck("unavailable", CHECK_UNAVAILABLE)
        if verdict is None:
            return SignalCheck("unavailable", CHECK_UNAVAILABLE)
        note = verdict.why.strip().rstrip(".") + "."
        return SignalCheck("artifact" if verdict.artifact_suspected else "clean", note)

    def _ask(self, level: Level, kind: str, reading: Reading, context: PatientContext) -> Optional[Wording]:
        from google.genai import types

        conditions, meds = relevant_history(context)
        facts = {
            "alert_level": level,
            "why": kind.replace("_", " "),
            "heart_rate_bpm": None if reading.hr is None else round(reading.hr),
            "usual_resting_bpm": None if reading.baseline_hr is None else round(reading.baseline_hr),
            "seconds_elevated_while_resting": int(reading.persist_s or 0),
            "activity": reading.activity,
            "recovery_after_activity": reading.recovery_verdict,
            "age": context.age,
            "relevant_conditions": conditions,
            "rate_affecting_medication": meds,
            "next_step_shown_on_screen": NEXT_STEPS[level],
            "signal_caveat": reading.signal_note,
        }
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=f"Write the message for these measurements:\n{facts}",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT.format(max_words=OBSERVATION_MAX_WORDS),
                    response_mime_type="application/json",
                    response_schema=Wording,
                    temperature=0.4,
                    max_output_tokens=400,
                    http_options=types.HttpOptions(timeout=GEMINI_DEADLINE_MS),
                ),
            )
            wording = response.parsed
        except Exception as exc:  # timeout, quota, network, schema
            log.warning("Gemini wording failed, using template: %s", exc)
            return None
        if not isinstance(wording, Wording) or not _valid(wording):
            log.warning("Gemini wording rejected, using template: %r", wording)
            return None
        return wording

    def _ask_signal(self, reading: Reading) -> Optional[SignalVerdict]:
        from google.genai import types

        facts = {
            "quality_0_to_1": reading.quality,
            "signal_caveat": reading.signal_note,
            "heart_rate_bpm": None if reading.hr is None else round(reading.hr),
            "usual_resting_bpm": None if reading.baseline_hr is None else round(reading.baseline_hr),
            "activity": reading.activity,
            "seconds_elevated_while_resting": int(reading.persist_s or 0),
        }
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=f"Is this pulse measurement too messy to trust?\n{facts}",
                config=types.GenerateContentConfig(
                    system_instruction=CHECK_PROMPT,
                    response_mime_type="application/json",
                    response_schema=SignalVerdict,
                    temperature=0.0,
                    max_output_tokens=120,
                    http_options=types.HttpOptions(timeout=GEMINI_DEADLINE_MS),
                ),
            )
            verdict = response.parsed
        except Exception as exc:
            log.warning("Signal check failed; the alert still stands: %s", exc)
            return None
        if not isinstance(verdict, SignalVerdict) or not _valid_why(verdict.why):
            log.warning("Signal check rejected: %r", verdict)
            return None
        return verdict


def _valid_why(text: str) -> bool:
    words = text.split()
    return 0 < len(words) <= 16 and not ADVICE.search(text) and not FORBIDDEN.search(text)
