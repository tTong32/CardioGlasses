"""Backend tests. No network: FinchNode and ElevenLabs are stubbed."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from ai.contracts import Config, PatientContext
from backend import caregiver, finchnode, main, voice

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
    "signal_note": None,
}
ALERT = {
    "t": 1760000050000, "level": "notify", "confidence": 0.8,
    "headline": "Elevated heart rate at rest", "reason": "Test reason",
    "voice_text": "Please sit down and check your phone.", "next_step": "Sit down and review",
    "reading": READING,
    "signal_check": None, "signal_check_note": None,
}


def next_alert(ws):
    """Skip check-in and caregiver messages; return the next alert message."""
    while True:
        msg = ws.receive_json()
        if msg["type"] == "alert":
            return msg


# ---------- FinchNode mapping ----------

def test_maps_finchnode_record_to_contract_d():
    context = finchnode.to_patient_context(RECORD, CONFIG, today=date(2026, 10, 3))
    assert context.patient_id == "patient-demo-polypharmacy"
    assert context.age == 78
    assert context.conditions == ["Atrial fibrillation", "Heart failure"]
    assert context.medications == ["metoprolol 50 MG"]
    assert context.risk_tier == "high"
    assert context.config == CONFIG


def test_clinic_resting_hr_is_median_of_recent_heart_rates():
    vitals = [{"name": "Heart rate", "value": str(v), "date": f"2024-0{i + 1}-01"} for i, v in enumerate([71, 77, 84, 68, 65])]
    vitals += [{"name": "Body weight", "value": "70", "date": "2024-01-01"}, {"name": "Heart rate", "value": "n/a", "date": "x"}]
    assert finchnode.clinic_resting_hr(vitals) == 71.0
    assert finchnode.clinic_resting_hr([]) is None
    record = {**RECORD, "data": {**RECORD["data"], "vitals": vitals}}
    assert finchnode.to_patient_context(record, CONFIG, today=date(2026, 10, 3)).clinic_resting_hr == 71.0
    assert finchnode.to_patient_context(RECORD, CONFIG, today=date(2026, 10, 3)).clinic_resting_hr is None


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


def test_record_shapes_thresholds_and_status(app_env, monkeypatch):
    finchnode_ok(monkeypatch)  # AF + heart failure, age 78 -> high risk
    with TestClient(main.app) as client:
        config = client.get("/patient").json()["config"]
        assert (config["deviation_trigger"], config["persist_s"]) == (2.0, 30.0)
        effects = client.get("/status").json()["record_effects"]
        assert effects[0].startswith("High risk:")
        assert any(line.startswith("Atrial fibrillation:") for line in effects)
        assert any(line.startswith("On metoprolol:") for line in effects)


def test_lower_risk_record_needs_more_evidence(app_env, monkeypatch):
    medium = {
        "id": "patient-demo-001",
        "data": {
            "demographics": {"birthDate": "1975-01-01"},
            "conditions": [{"name": "Essential hypertension", "status": "active"}],
            "medications": [],
        },
    }
    monkeypatch.setattr(
        finchnode, "fetch_patient_context",
        lambda config: finchnode.to_patient_context(medium, config, today=date(2026, 10, 3)),
    )
    with TestClient(main.app) as client:
        patient = client.get("/patient").json()
        assert patient["risk_tier"] == "medium"
        assert (patient["config"]["deviation_trigger"], patient["config"]["persist_s"]) == (2.5, 45.0)
        # A cached record keeps its tier's thresholds when FinchNode is down.
    finchnode_down(monkeypatch)
    with TestClient(main.app) as client:
        assert client.get("/patient").json()["config"]["persist_s"] == 45.0


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
        msg = next_alert(ws)
        assert msg["type"] == "alert"
        assert msg["data"] == ALERT
        assert msg["audio_source"] == "elevenlabs"
        assert client.get(msg["audio_url"]).content == b"mp3"

        client.post("/alerts", json=ALERT)
        assert next_alert(ws)["audio_source"] == "cache"
    assert calls == [ALERT["voice_text"]]
    assert client.get("/alerts").json()[0] == ALERT


def test_alert_uses_fallback_clip_when_elevenlabs_fails(app_env, monkeypatch):
    finchnode_down(monkeypatch)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    (app_env / "audio" / "fallback_notify.mp3").write_bytes(b"fallback")
    with TestClient(main.app) as client, client.websocket_connect("/ws") as ws:
        client.post("/alerts", json=ALERT)
        msg = next_alert(ws)
        assert msg["audio_source"] == "fallback"
        assert msg["audio_url"] == "/audio/fallback_notify.mp3"
        assert client.get("/status").json()["fallback_audio_levels"] == ["notify"]


def test_alert_without_any_audio_still_broadcasts(app_env, monkeypatch):
    finchnode_down(monkeypatch)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with TestClient(main.app) as client, client.websocket_connect("/ws") as ws:
        client.post("/alerts", json=ALERT)
        msg = next_alert(ws)
        assert msg["audio_url"] is None
        assert msg["audio_source"] == "none"


def test_dashboard_is_served(app_env, monkeypatch):
    finchnode_down(monkeypatch)
    with TestClient(main.app) as client:
        assert "CardioGlasses" in client.get("/").text


# ---------- Caregiver safety net ----------

NORMAL = {**ALERT, "level": "normal", "headline": "Back within your usual range", "next_step": "No action needed"}


@pytest.fixture
def safety_env(app_env, monkeypatch):
    finchnode_down(monkeypatch)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setenv("PATIENT_NAME", "Harriet")
    monkeypatch.setenv("CAREGIVER_NAME", "Alex")
    monkeypatch.setenv("CHECKIN_TIMEOUT_S", "0.3")
    sent = []
    monkeypatch.setattr(caregiver, "send_imessage", lambda text: sent.append(text) or ("imessage", "guid-1"))
    main.safety.current = None
    main.safety.events.clear()
    return sent


def wait_for_status(client, status, timeout=3.0):
    import time
    end = time.time() + timeout
    while time.time() < end:
        current = client.get("/checkin").json()["current"]
        if current and current["status"] == status:
            return current
        time.sleep(0.05)
    raise AssertionError(f"check-in never reached {status}: {client.get('/checkin').json()}")


def test_notify_starts_a_checkin_and_ok_closes_it(safety_env):
    with TestClient(main.app) as client:
        client.post("/alerts", json=ALERT)
        current = client.get("/checkin").json()["current"]
        assert current["status"] == "waiting" and current["level"] == "notify"
        reply = client.post(f"/checkin/{current['id']}/respond", json={"answer": "ok"})
        assert reply.json()["status"] == "ok"
        snap = client.get("/checkin").json()
        assert [e["kind"] for e in snap["events"]] == ["ok", "started"]
        assert snap["patient_name"] == "Harriet" and snap["caregiver_name"] == "Alex"
    assert safety_env == []  # no message when they answer


def test_no_answer_texts_the_caregiver(safety_env):
    with TestClient(main.app) as client:
        client.post("/alerts", json=ALERT)
        wait_for_status(client, "no_response")
        event = client.get("/checkin").json()["events"][0]
        assert event["kind"] == "no_response" and event["channel"] == "imessage" and event["urgent"]
    assert len(safety_env) == 1
    text = safety_env[0]
    assert "Harriet hasn't responded" in text and "104 bpm" in text and "usual 68" in text and "48 seconds" in text
    assert "asked to sit down and review" in text


def test_need_help_texts_immediately(safety_env, monkeypatch):
    monkeypatch.setenv("CHECKIN_TIMEOUT_S", "60")
    with TestClient(main.app) as client:
        client.post("/alerts", json=ALERT)
        checkin_id = client.get("/checkin").json()["current"]["id"]
        assert client.post(f"/checkin/{checkin_id}/respond", json={"answer": "help"}).json()["status"] == "help"
        assert client.post(f"/checkin/{checkin_id}/respond", json={"answer": "ok"}).status_code == 409
        assert client.post("/checkin/nope/respond", json={"answer": "ok"}).status_code == 404
        snap = client.post("/checkin/ack").json()
        assert snap["current"]["acknowledged"] and snap["events"][0]["kind"] == "acknowledged"
    assert len(safety_env) == 1 and 'tapped "I need help"' in safety_env[0]


def test_recovery_before_the_deadline_closes_quietly(safety_env, monkeypatch):
    monkeypatch.setenv("CHECKIN_TIMEOUT_S", "60")
    with TestClient(main.app) as client:
        client.post("/alerts", json=ALERT)
        client.post("/alerts", json=NORMAL)
        assert client.get("/checkin").json()["current"]["status"] == "resolved"
    assert safety_env == []


def test_escalate_upgrades_a_waiting_checkin(safety_env, monkeypatch):
    monkeypatch.setenv("CHECKIN_TIMEOUT_S", "60")
    with TestClient(main.app) as client:
        client.post("/alerts", json=ALERT)
        first = client.get("/checkin").json()["current"]
        client.post("/alerts", json={**ALERT, "level": "escalate"})
        current = client.get("/checkin").json()["current"]
        assert current["id"] == first["id"] and current["level"] == "escalate"


def test_checkin_is_pushed_to_screens(safety_env, monkeypatch):
    monkeypatch.setenv("CHECKIN_TIMEOUT_S", "60")
    with TestClient(main.app) as client, client.websocket_connect("/ws") as ws:
        client.post("/alerts", json=ALERT)
        kinds = [ws.receive_json()["type"] for _ in range(3)]
        assert kinds == ["alert", "caregiver", "checkin"]


def test_imessage_needs_configuration(monkeypatch):
    for key in ("CAREGIVER_PHONE", "PHOTON_IMESSAGE_ADDRESS", "PHOTON_IMESSAGE_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    assert caregiver.send_imessage("hi") == ("screen", "iMessage not configured")


def test_photon_sender_reports_errors_as_json(monkeypatch):
    import json, shutil, subprocess
    if shutil.which("node") is None or not (caregiver.SENDER.parent / "node_modules").is_dir():
        pytest.skip("node or tools/imessage dependencies not installed")
    env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("PHOTON_")}
    done = subprocess.run(["node", str(caregiver.SENDER), "+15550001111"], input="hello",
                          capture_output=True, text=True, timeout=30, env=env, cwd=caregiver.SENDER.parent)
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["ok"] is False and "PHOTON_IMESSAGE_ADDRESS" in result["error"]


def test_caregiver_text_retells_the_step_in_third_person():
    from ai.contracts import Alert
    alert = Alert.model_validate({**ALERT, "next_step": "Sit down and check your phone"})
    assert caregiver.reading_summary(alert, "Harriet").endswith("Harriet was asked to sit down and check their phone.")


def test_only_actionable_alerts_are_spoken(app_env, monkeypatch):
    finchnode_down(monkeypatch)
    calls = []
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setattr(voice, "synthesize", lambda text, timeout=8: calls.append(text) or b"mp3")
    with TestClient(main.app) as client, client.websocket_connect("/ws") as ws:
        for level in ("monitor", "normal"):
            client.post("/alerts", json={**ALERT, "level": level})
            msg = next_alert(ws)
            assert msg["speak"] is False and msg["audio_url"] is None
        client.post("/alerts", json=ALERT)
        assert next_alert(ws)["speak"] is True
    assert calls == [ALERT["voice_text"]]
