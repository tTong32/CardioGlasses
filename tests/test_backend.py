"""Backend tests. No network: FinchNode and ElevenLabs are stubbed."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from ai.contracts import Config, PatientContext
from backend import finchnode, main, voice

CONFIG = Config(min_quality=0.6, persist_s=30, deviation_trigger=2.0, cooldown_s=120)

RECORD = {
    "id": "patient-demo-polypharmacy",
    "data": {
        "demographics": {"birthDate": "1948-03-02"},
        "conditions": [
            {"name": "Atrial fibrillation", "status": "active"},
            {"name": "Heart failure", "status": "active"},
            {"name": "Old fracture", "status": "resolved"},
        ],
        "medications": [
            {"name": "metoprolol 50 MG", "status": "active"},
            {"name": "metoprolol 50 MG", "status": "active"},
            {"name": "amoxicillin", "status": "completed"},
        ],
    },
}

READING = {
    "t": 1760000002000, "hr": 104, "ibi_ms": [580, 570], "activity": "resting",
    "quality": 0.86, "baseline_hr": 68, "deviation": 2.6, "persist_s": 48,
    "recovery_tau_s": None, "hr_drop_60s": None, "recovery_ratio": None,
    "recovery_percentile": None, "recovery_verdict": None, "signal_status": "ok",
}
ALERT = {
    "t": 1760000050000, "level": "notify", "confidence": 0.8,
    "headline": "Elevated heart rate at rest", "reason": "Test reason",
    "voice_text": "Please sit down and check your phone.", "next_step": "Sit down and review",
    "reading": READING,
}


# ---------- FinchNode mapping ----------

def test_maps_finchnode_record_to_contract_d():
    context = finchnode.to_patient_context(RECORD, CONFIG, today=date(2026, 10, 3))
    assert context.patient_id == "patient-demo-polypharmacy"
    assert context.age == 78
    assert context.conditions == ["Atrial fibrillation", "Heart failure"]
    assert context.medications == ["metoprolol 50 MG"]
    assert context.risk_tier == "high"
    assert context.config == CONFIG


def test_age_counts_birthday_not_yet_reached():
    assert finchnode.age_from_birth_date("1948-12-31", today=date(2026, 10, 3)) == 77


@pytest.mark.parametrize(
    ("age", "conditions", "tier"),
    [
        (40, ["Atrial fibrillation"], "high"),
        (80, [], "high"),
        (50, ["Essential hypertension"], "medium"),
        (66, [], "medium"),
        (30, ["Asthma"], "low"),
    ],
)
def test_risk_tier(age, conditions, tier):
    assert finchnode.risk_tier(age, conditions) == tier


def test_missing_birth_date_is_rejected():
    with pytest.raises(ValueError):
        finchnode.to_patient_context({"id": "x", "data": {"demographics": {}}}, CONFIG)


# ---------- App fixtures ----------

@pytest.fixture
def app_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'app.db'}")
    monkeypatch.setenv("FINCHNODE_ENABLED", "1")
    monkeypatch.setattr(main, "PATIENT_CACHE_PATH", tmp_path / "finchnode_cache.json")
    monkeypatch.setattr(voice, "AUDIO_DIR", tmp_path / "audio")
    (tmp_path / "audio").mkdir()
    return tmp_path


def finchnode_ok(monkeypatch):
    monkeypatch.setattr(
        finchnode, "fetch_patient_context",
        lambda config: finchnode.to_patient_context(RECORD, config, today=date(2026, 10, 3)),
    )


def finchnode_down(monkeypatch):
    def fail(config):
        raise ConnectionError("no network")
    monkeypatch.setattr(finchnode, "fetch_patient_context", fail)


# ---------- Patient source ----------

def test_patient_comes_from_finchnode_and_is_cached(app_env, monkeypatch):
    finchnode_ok(monkeypatch)
    with TestClient(main.app) as client:
        assert client.get("/patient").json()["patient_id"] == "patient-demo-polypharmacy"
        assert client.get("/status").json()["patient_source"] == "finchnode"
    assert (app_env / "finchnode_cache.json").is_file()


def test_patient_falls_back_to_cache_then_local(app_env, monkeypatch):
    finchnode_ok(monkeypatch)
    with TestClient(main.app):
        pass  # writes the cache
    finchnode_down(monkeypatch)
    with TestClient(main.app) as client:
        status = client.get("/status").json()
        assert status["patient_source"] == "cache"
        assert "no network" in status["patient_error"]
        assert client.get("/patient").json()["patient_id"] == "patient-demo-polypharmacy"

    (app_env / "finchnode_cache.json").unlink()
    with TestClient(main.app) as client:
        assert client.get("/status").json()["patient_source"] == "local"
        PatientContext.model_validate(client.get("/patient").json())


def test_finchnode_can_be_disabled(app_env, monkeypatch):
    monkeypatch.setenv("FINCHNODE_ENABLED", "0")
    finchnode_ok(monkeypatch)
    with TestClient(main.app) as client:
        assert client.get("/status").json()["patient_source"] == "local"


# ---------- Readings, alerts, voice ----------

def test_reading_is_stored_and_pushed(app_env, monkeypatch):
    finchnode_down(monkeypatch)
    with TestClient(main.app) as client, client.websocket_connect("/ws") as ws:
        assert client.post("/readings", json=READING).status_code == 200
        assert ws.receive_json() == {"type": "reading", "data": READING}
        assert client.get("/readings?limit=1").json() == [READING]


def test_reading_with_missing_field_is_rejected(app_env, monkeypatch):
    finchnode_down(monkeypatch)
    bad = {k: v for k, v in READING.items() if k != "quality"}
    with TestClient(main.app) as client:
        assert client.post("/readings", json=bad).status_code == 422


def test_alert_carries_elevenlabs_audio(app_env, monkeypatch):
    finchnode_down(monkeypatch)
    calls = []
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setattr(voice, "synthesize", lambda text, timeout=8: calls.append(text) or b"mp3")
    with TestClient(main.app) as client, client.websocket_connect("/ws") as ws:
        client.post("/alerts", json=ALERT)
        msg = ws.receive_json()
        assert msg["type"] == "alert"
        assert msg["data"] == ALERT
        assert msg["audio_source"] == "elevenlabs"
        assert client.get(msg["audio_url"]).content == b"mp3"

        client.post("/alerts", json=ALERT)
        assert ws.receive_json()["audio_source"] == "cache"
    assert calls == [ALERT["voice_text"]]
    assert client.get("/alerts").json()[0] == ALERT


def test_alert_uses_fallback_clip_when_elevenlabs_fails(app_env, monkeypatch):
    finchnode_down(monkeypatch)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    (app_env / "audio" / "fallback_notify.mp3").write_bytes(b"fallback")
    with TestClient(main.app) as client, client.websocket_connect("/ws") as ws:
        client.post("/alerts", json=ALERT)
        msg = ws.receive_json()
        assert msg["audio_source"] == "fallback"
        assert msg["audio_url"] == "/audio/fallback_notify.mp3"
        assert client.get("/status").json()["fallback_audio_levels"] == ["notify"]


def test_alert_without_any_audio_still_broadcasts(app_env, monkeypatch):
    finchnode_down(monkeypatch)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with TestClient(main.app) as client, client.websocket_connect("/ws") as ws:
        client.post("/alerts", json=ALERT)
        msg = ws.receive_json()
        assert msg["audio_url"] is None
        assert msg["audio_source"] == "none"


def test_dashboard_is_served(app_env, monkeypatch):
    finchnode_down(monkeypatch)
    with TestClient(main.app) as client:
        assert "CardioGlasses" in client.get("/").text
