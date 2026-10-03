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
    """Contract A. One PPG + IMU sample."""

    t: int
    ppg: Optional[int]
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
