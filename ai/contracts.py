"""Shared Contract A–D models. The backend imports this module.

Nullable fields are `X | None` with no default: a missing key is rejected, and an
explicit null is stored. Serialization keeps those nulls (they are not omitted).
"""

from typing import Literal

from pydantic import BaseModel

Level = Literal["normal", "monitor", "notify", "escalate"]
Activity = Literal["resting", "moving"]


class Sample(BaseModel):
    """Contract A. One PPG + IMU sample."""

    t: int
    ppg: int | None
    ax: float | None
    ay: float | None
    az: float | None
    gx: float | None
    gy: float | None
    gz: float | None


class Reading(BaseModel):
    """Contract B. A ~2 s summary posted to the backend."""

    t: int
    hr: float | None
    ibi_ms: list[int] | None
    activity: Activity | None
    quality: float | None
    baseline_hr: float | None
    deviation: float | None
    persist_s: float | None


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
    confidence: float | None
    headline: str | None
    reason: str | None
    voice_text: str | None
    next_step: str | None
    reading: Reading
