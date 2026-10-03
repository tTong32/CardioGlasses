"""Tests for the demo simulator and the serial checker."""

import json

import pytest

from ai.contracts import Config, PatientContext
from tools import demo_sim
from tools.serial_check import Checker, write_csv

CONTEXT = PatientContext(
    patient_id="patient-demo-polypharmacy",
    age=78,
    conditions=["Atrial fibrillation", "Heart failure", "Hypothyroidism"],
    medications=["metoprolol"],
    risk_tier="high",
    config=Config(min_quality=0.6, persist_s=30, deviation_trigger=2.0, cooldown_s=120),
)


def levels(scenario):
    return [(e.offset_s, e.alert.level) for e in demo_sim.build_timeline(scenario, CONTEXT) if e.alert]


# ---------- Demo simulator ----------

def test_elevated_story_escalates_in_order_then_recovers():
    story = levels("elevated")
    assert [level for _, level in story] == ["monitor", "notify", "normal"]
    monitor_at, notify_at, normal_at = (t for t, _ in story)
    assert 100 <= monitor_at < notify_at <= monitor_at + 32
    assert normal_at > notify_at


def test_notify_only_after_persist_threshold():
    notify = next(e for e in demo_sim.build_timeline("elevated", CONTEXT) if e.alert and e.alert.level == "notify")
    reading = notify.alert.reading
    assert reading.persist_s >= CONTEXT.config.persist_s
    assert reading.activity == "resting"
    assert reading.quality >= CONTEXT.config.min_quality
    assert reading.deviation >= CONTEXT.config.deviation_trigger


def test_escalate_scenario_reaches_escalate():
    assert "escalate" in [level for _, level in levels("escalate")]


@pytest.mark.parametrize("scenario", ["poor-signal", "normal", "dropout"])
def test_quiet_scenarios_never_alert(scenario):
    assert levels(scenario) == []


def test_poor_signal_is_reported_honestly():
    poor = [e.reading for e in demo_sim.build_timeline("poor-signal", CONTEXT) if e.reading and e.reading.signal_status == "poor"]
    assert poor and all(r.quality < CONTEXT.config.min_quality for r in poor)


def test_dropout_sends_nothing_for_a_while():
    gaps = [e for e in demo_sim.build_timeline("dropout", CONTEXT) if e.reading is None]
    assert len(gaps) * demo_sim.STEP_S >= 10


def test_moving_never_counts_toward_persist():
    for e in demo_sim.build_timeline("elevated", CONTEXT):
        if e.reading and e.reading.activity == "moving":
            assert e.reading.persist_s == 0


def test_alert_mentions_cardiac_history():
    notify = next(e.alert for e in demo_sim.build_timeline("elevated", CONTEXT) if e.alert and e.alert.level == "notify")
    assert "atrial fibrillation" in notify.reason
    assert "hypothyroidism" not in notify.reason


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError):
        demo_sim.build_timeline("nope", CONTEXT)


# ---------- Serial checker ----------

def line(t, ppg=50000, ax=0.02, ay=-0.98, az=0.10, **extra):
    payload = {"t": t, "ppg": ppg, "ax": ax, "ay": ay, "az": az, "gx": 0.1, "gy": 0.0, "gz": 0.0}
    payload.update(extra)
    return json.dumps(payload)


def good_stream(n=200, hz=50, t0=1_760_000_000_000):
    return [line(t0 + i * 1000 // hz, ppg=50000 + (i % 25) * 100) for i in range(n)]


def test_clean_stream_passes():
    checker = Checker(expected_hz=50)
    for text in good_stream():
        checker.feed(text)
    assert checker.problems() == []
    assert checker.accel_units() == "g"
    assert checker.rate_hz() == pytest.approx(50, rel=0.02)


def test_bad_lines_are_counted_not_fatal():
    checker = Checker(expected_hz=50)
    for text in good_stream(50) + ["garbage", '{"t": 1}', line(1, ppg="x")]:
        checker.feed(text)
    assert checker.bad_lines == 3
    assert len(checker.samples) == 50
    assert any("missing keys" in example for example in checker.bad_examples)


def test_detects_wrong_rate_gaps_and_units():
    checker = Checker(expected_hz=50)
    t0 = 1_760_000_000_000
    for i in range(100):  # 100 Hz in m/s^2, with one big gap
        t = t0 + i * 10 + (500 if i > 50 else 0)
        checker.feed(line(t, ppg=50000 + (i % 25) * 100, ax=0.2, ay=-9.8, az=1.0))
    issues = " ".join(checker.problems())
    assert "sample rate" in issues
    assert "gaps" in issues
    assert "m/s^2" in issues


def test_flags_flat_ppg_and_uptime_timestamps():
    checker = Checker(expected_hz=50)
    for i in range(100):
        checker.feed(line(i * 20, ppg=50000))
    issues = " ".join(checker.problems())
    assert "flat" in issues
    assert "uptime" in issues


def test_nulls_are_allowed_but_counted():
    checker = Checker(expected_hz=50)
    checker.feed(line(1_760_000_000_000, ppg=None))
    assert checker.nulls == 1 and len(checker.samples) == 1


def test_csv_round_trips_through_replay(tmp_path):
    from ai.replay import load_csv

    checker = Checker(expected_hz=50)
    for text in good_stream(20) + [line(1_760_000_000_400, ppg=None)]:
        checker.feed(text)
    path = tmp_path / "rec.csv"
    write_csv(checker.samples, path)
    assert load_csv(path) == checker.samples
