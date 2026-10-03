"""FinchNode health record → Contract D PatientContext.

The public demo API needs no key. `patient-demo-polypharmacy` is a 78-year-old with
atrial fibrillation and heart failure on metoprolol, which fits our high-risk user.
"""

from __future__ import annotations

import os
from datetime import date
from statistics import median

import requests

from ai.contracts import Config, PatientContext

DEMO_BASE_URL = "https://api.finchnode.com/demo/v1"
DEFAULT_PATIENT_ID = "patient-demo-polypharmacy"

CARDIAC_TERMS = (
    "atrial fibrillation",
    "heart failure",
    "coronary",
    "myocardial",
    "arrhythmia",
    "cardiomyopathy",
    "angina",
    "stroke",
)
RISK_TERMS = ("hypertension", "diabetes", "kidney", "hyperlipidemia")


def age_from_birth_date(birth_date: str, today: date | None = None) -> int:
    born = date.fromisoformat(birth_date[:10])
    today = today or date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def risk_tier(age: int, conditions: list[str]) -> str:
    """High for cardiac disease or age 75+, medium for other risk factors or age 65+."""
    text = " ".join(conditions).lower()
    if age >= 75 or any(term in text for term in CARDIAC_TERMS):
        return "high"
    if age >= 65 or any(term in text for term in RISK_TERMS):
        return "medium"
    return "low"


def _active_names(items: list[dict]) -> list[str]:
    names: list[str] = []
    for item in items or []:
        name = (item.get("name") or "").strip()
        if name and item.get("status", "active") == "active" and name not in names:
            names.append(name)
    return names


def clinic_resting_hr(vitals: list[dict], last_n: int = 10) -> float | None:
    """Median of the most recent clinic heart-rate readings, or None if there are none."""
    readings = []
    for v in vitals or []:
        if (v.get("name") or "").strip().lower() != "heart rate":
            continue
        try:
            bpm = float(v.get("value"))
        except (TypeError, ValueError):
            continue
        if 30 <= bpm <= 200:
            readings.append((v.get("date") or "", bpm))
    if not readings:
        return None
    recent = sorted(readings)[-last_n:]
    return round(float(median(bpm for _, bpm in recent)), 1)


def to_patient_context(record: dict, config: Config, today: date | None = None) -> PatientContext:
    """Map a FinchNode `/users/{subject}/records` response to Contract D."""
    data = record.get("data") or {}
    demographics = data.get("demographics") or {}
    birth_date = demographics.get("birthDate")
    if not birth_date:
        raise ValueError("FinchNode record has no birthDate")
    age = age_from_birth_date(birth_date, today)
    conditions = _active_names(data.get("conditions"))
    return PatientContext(
        patient_id=record.get("id") or "unknown",
        age=age,
        conditions=conditions,
        medications=_active_names(data.get("medications")),
        risk_tier=risk_tier(age, conditions),
        config=config,
        clinic_resting_hr=clinic_resting_hr(data.get("vitals")),
    )


def fetch_record(base_url: str, patient_id: str, api_key: str | None = None, timeout: float = 8) -> dict:
    headers = {}
    # The demo API is open; only send the key to a non-demo host.
    if api_key and "/demo/" not in base_url:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.get(
        f"{base_url.rstrip('/')}/users/{patient_id}/records",
        params={"categories": "demographics,conditions,medications,vitals"},
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def fetch_patient_context(config: Config) -> PatientContext:
    """Fetch the configured patient from FinchNode. Raises on any failure."""
    record = fetch_record(
        os.environ.get("FINCHNODE_BASE_URL") or DEMO_BASE_URL,
        os.environ.get("FINCHNODE_PATIENT_ID") or DEFAULT_PATIENT_ID,
        os.environ.get("FINCHNODE_API_KEY") or None,
    )
    return to_patient_context(record, config)
