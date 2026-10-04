"""Shared Contract A–D models. The backend imports this module.

Nullable fields are `Optional[X]` with no default: a missing key is rejected, and an
explicit null is stored. Serialization keeps those nulls (they are not omitted).
"""

from typing import Literal, Optional

from pydantic import BaseModel

Level = Literal["normal", "monitor", "notify", "escalate"]
Activity = Literal["resting", "moving"]
SignalStatus = Literal["ok", "poor", "offline"]
RecoveryVerdict = Literal["normal", "slow", "very_slow"]


class Sample(BaseModel):
    """Contract A. One PPG + IMU sample.

    `red` is the optional second PPG channel the original contract note left open
    ("2nd PPG channel optional"): the MAX30102 reports both `ir` (-> `ppg`) and `red`.
    Defaults to None so Contract A's exact 8-key wire format still validates unchanged.
    """

    t: int
    ppg: Optional[int]
    red: Optional[int] = None
    ax: Optional[float]
    ay: Optional[float]
    az: Optional[float]
    gx: Optional[float]
    gy: Optional[float]
    gz: Optional[float]


class Reading(BaseModel):
    """Contract B. A ~2 s summary posted to the backend."""

    t: int
    hr: Optional[float]
    ibi_ms: Optional[list[int]]
    activity: Optional[Activity]
    quality: Optional[float]
    baseline_hr: Optional[float]
    deviation: Optional[float]
    persist_s: Optional[float]
    recovery_tau_s: Optional[float]
    hr_drop_60s: Optional[float]
    recovery_ratio: Optional[float]
    recovery_percentile: Optional[float]
    recovery_verdict: Optional[RecoveryVerdict]
    signal_status: Optional[SignalStatus]
    # Why this window is hard to trust, in plain words, or null when it looks clean.
    # Optional so older senders still validate.
    signal_note: Optional[str] = None


class Config(BaseModel):
    """Thresholds inside PatientContext."""

    min_quality: float
    persist_s: float
    deviation_trigger: float
    cooldown_s: float


class PatientContext(BaseModel):
    """Contract D. Loaded at startup."""

    patient_id: str
    age: int
    conditions: list[str]
    medications: list[str]
    risk_tier: str
    config: Config
    # Median of recent clinic heart-rate readings (bpm), or null. Optional so older senders
    # still validate; the backend always emits it. A starting point before calibration.
    clinic_resting_hr: Optional[float] = None


class Alert(BaseModel):
    """Contract C. Emitted only when the level changes."""

    t: int
    level: Level
    confidence: Optional[float]
    headline: Optional[str]
    reason: Optional[str]
    voice_text: Optional[str]
    next_step: Optional[str]
    reading: Reading
    # Second opinion on the pulse, shown on the alert. Never required to send the alert.
    # "clean", "artifact", or "unavailable". Optional so older senders still validate.
    signal_check: Optional[str] = None
    signal_check_note: Optional[str] = None
