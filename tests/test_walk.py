"""Activity coaching (backend/walk.py) and the heart-rate zone rule (ai/clinical.py)."""

from __future__ import annotations

import asyncio

import pytest

from ai import clinical
from ai.contracts import Config, PatientContext, Reading
from backend import walk
from tools import demo_sim

CONTEXT = PatientContext(
    patient_id="p", age=78, conditions=["Atrial fibrillation"], medications=["24 HR metoprolol succinate 50 MG"],
    risk_tier="high", config=Config(min_quality=0.6, persist_s=30, deviation_trigger=2.0, cooldown_s=120),
    clinic_resting_hr=73.5,
)


def reading(t: int, hr: float | None, activity: str = "moving", status: str = "ok", baseline: float | None = 68.0) -> Reading:
    return Reading(
        t=t, hr=hr, ibi_ms=None, activity=activity, quality=0.9, baseline_hr=baseline, deviation=None, persist_s=0,
        recovery_tau_s=None, hr_drop_60s=None, recovery_ratio=None, recovery_percentile=None,
        recovery_verdict=None, signal_status=status,
    )


class Harness:
    def __init__(self, monkeypatch, context=CONTEXT):
        self.spoken: list[tuple[str, str]] = []
        self.pushed: list[dict] = []
        self.notes: list[str] = []
        self.clock = [1_000_000]
        monkeypatch.setattr(walk, "now_ms", lambda: self.clock[0])

        async def speak(text, kind):
            self.spoken.append((kind, text))

        async def broadcast(msg):
            self.pushed.append(msg)

        async def note(text):
            self.notes.append(text)

        self.coach = walk.Coach(broadcast, speak, lambda: context, lambda: "Harriet", note=note)

    async def feed(self, seconds: float, hr: float, activity: str = "moving", status: str = "ok"):
        steps = int(seconds / 2)
        for _ in range(steps):
            self.clock[0] += 2000
            await self.coach.on_reading(reading(self.clock[0], hr, activity, status))


# ---------- zone rule ----------

def test_zone_on_rate_control_medication():
    assert clinical.walk_zone(CONTEXT, 68) == (88, 98, "usual 68 + 20 to 30, because of metoprolol")


def test_zone_without_rate_control_uses_heart_rate_reserve():
    low, high, note = clinical.walk_zone(CONTEXT.model_copy(update={"medications": [], "age": 40}), 62)
    assert (low, high) == (109, 133) and "reserve" in note


def test_zone_falls_back_to_clinic_heart_rate():
    assert clinical.walk_zone(CONTEXT, None)[:2] == (94, 104)  # 73.5 + 20..30


# ---------- coaching ----------

def test_start_announces_the_zone_from_the_learned_baseline(monkeypatch):
    h = Harness(monkeypatch)

    async def go():
        await h.coach.on_reading(reading(h.clock[0], 70, "resting", baseline=68))
        state = await h.coach.start()
        assert (state.zone_low, state.zone_high) == (88, 98)
        assert state.mode == "exercise" and state.source == "manual" and state.phase == "walking"
    asyncio.run(go())
    assert h.spoken[0][0] == "start" and "between 88 and 98" in h.spoken[0][1]


def test_speaks_only_when_above_the_zone_and_once_back_in(monkeypatch):
    h = Harness(monkeypatch)

    async def go():
        await h.coach.on_reading(reading(h.clock[0], 70, "resting", baseline=68))  # learned usual: zone 88-98
        await h.coach.start()
        await h.feed(20, 92)        # in zone: silent
        await h.feed(4, 104)        # above, but not for long enough yet
        assert [k for k, _ in h.spoken] == ["start"]
        await h.feed(10, 106)       # above for 6 s+: one "slow down"
        await h.feed(10, 107)       # still above: no repeat inside the cooldown
        await h.feed(6, 93)         # back in zone: one "good"
        await h.feed(20, 92)        # stays quiet
    asyncio.run(go())
    kinds = [k for k, _ in h.spoken]
    assert kinds == ["start", "above", "back"]
    assert "above your zone of 98" in h.spoken[1][1]


def test_untrusted_readings_never_trigger_a_tip(monkeypatch):
    h = Harness(monkeypatch)

    async def go():
        await h.coach.start()
        await h.feed(30, 130, status="poor")
    asyncio.run(go())
    assert [k for k, _ in h.spoken] == ["start"]


def test_recovery_check_reports_the_one_minute_drop(monkeypatch):
    monkeypatch.setenv("WALK_RECOVERY_S", "0.05")
    h = Harness(monkeypatch)

    async def go():
        await h.coach.start()
        await h.feed(30, 110)
        await h.coach.stop()
        assert h.coach.state.hr_at_stop == 110
        await h.feed(6, 84, "resting")
        h.clock[0] = h.coach.state.recovery_due
        await asyncio.sleep(0.2)
    asyncio.run(go())
    s = h.coach.state
    assert s.phase == "done" and s.drop_1min == 26 and s.recovery_ok
    assert "dropped 26 beats" in h.spoken[-1][1] and "healthy" in h.spoken[-1][1]
    assert h.notes and "Recovery: healthy" in h.notes[0]


def test_slow_recovery_is_named_gently(monkeypatch):
    monkeypatch.setenv("WALK_RECOVERY_S", "0.05")
    h = Harness(monkeypatch)

    async def go():
        await h.coach.start()
        await h.feed(30, 110)
        await h.coach.stop()
        await h.feed(6, 102, "resting")
        h.clock[0] = h.coach.state.recovery_due
        await asyncio.sleep(0.2)
    asyncio.run(go())
    assert h.coach.state.recovery_ok is False
    assert "slower than expected" in h.spoken[-1][1] and "care team" in h.spoken[-1][1]


def test_stop_without_a_walk_does_nothing(monkeypatch):
    h = Harness(monkeypatch)
    state = asyncio.run(h.coach.stop())
    assert state.phase == "idle" and h.spoken == []


# ---------- quiet activity, auto-detect, cooldown ----------
# feed(N) steps every 2 s, so a fresh hold covers N - 2 seconds (the first sample starts the clock).

def test_auto_detect_starts_quiet_and_never_speaks(monkeypatch):
    monkeypatch.setenv("ACTIVITY_DETECT_S", "8")
    h = Harness(monkeypatch)

    async def go():
        await h.feed(20, 110, "resting")  # high, but not moving
        await h.feed(20, 78, "moving")  # moving, but only baseline + 10, which is not above it
        assert h.coach.state.phase == "idle" and h.spoken == []
        await h.feed(8, 90, "moving")  # 6 s of the 8 s
        assert h.coach.state.phase == "idle"
        await h.feed(4, 90, "moving")  # crosses 8 s
        assert h.coach.state.mode == "quiet" and h.coach.state.source == "auto"
        await h.feed(20, 140, "moving")  # still quiet, and still silent, however high the rate goes
    asyncio.run(go())
    assert h.spoken == []
    assert any(m["type"] == "activity" and m["data"]["mode"] == "quiet" for m in h.pushed)
    assert any(m["type"] == "walk" and m["data"]["source"] == "auto" for m in h.pushed)


def test_quiet_to_exercise_announces_the_zone_without_restarting(monkeypatch):
    monkeypatch.setenv("ACTIVITY_DETECT_S", "8")
    h = Harness(monkeypatch)

    async def go():
        assert (await h.coach.to_exercise()).phase == "idle"
        await h.feed(10, 90, "moving")
        started = h.coach.state.started_at
        state = await h.coach.to_exercise()
        assert state.mode == "exercise" and state.source == "auto" and state.started_at == started
        assert (state.zone_low, state.zone_high) == (88, 98)
        again = await h.coach.to_exercise()
        assert again.started_at == started
    asyncio.run(go())
    assert [k for k, _ in h.spoken] == ["start"]
    assert "between 88 and 98" in h.spoken[0][1]


def test_stop_in_quiet_is_silent(monkeypatch):
    monkeypatch.setenv("ACTIVITY_DETECT_S", "8")
    h = Harness(monkeypatch)

    async def go():
        await h.feed(10, 90, "moving")
        state = await h.coach.stop()
        assert state.phase == "idle" and state.mode is None
    asyncio.run(go())
    assert h.spoken == []


def test_auto_quiet_ends_after_resting(monkeypatch):
    monkeypatch.setenv("ACTIVITY_DETECT_S", "8")
    h = Harness(monkeypatch)

    async def go():
        await h.feed(10, 90, "moving")
        await h.feed(40, 70, "resting")  # 38 s, not yet a minute
        assert h.coach.state.mode == "quiet"
        await h.feed(4, 90, "moving")  # still out and about; the rest streak resets
        assert h.coach.state.mode == "quiet"
        await h.feed(40, 70, "resting")
        assert h.coach.state.mode == "quiet"
        await h.feed(24, 70, "resting")  # the rest streak reaches 60 s
        assert h.coach.state.phase == "idle"
    asyncio.run(go())
    assert h.spoken == []


def test_promoting_to_exercise_does_not_end_when_they_sit_down(monkeypatch):
    monkeypatch.setenv("ACTIVITY_DETECT_S", "8")
    h = Harness(monkeypatch)

    async def go():
        await h.feed(10, 90, "moving")
        await h.coach.to_exercise()
        await h.feed(70, 70, "resting")
        assert h.coach.state.phase == "walking" and h.coach.state.mode == "exercise"
    asyncio.run(go())
    assert [k for k, _ in h.spoken] == ["start"]


def test_cooldown_blocks_auto_until_hr_settles_then_allows_it(monkeypatch):
    monkeypatch.setenv("ACTIVITY_DETECT_S", "8")
    h = Harness(monkeypatch)

    async def go():
        await h.feed(10, 90, "moving")
        await h.coach.stop()
        await h.feed(10, 90, "moving")  # would qualify, but the cooldown is on
        assert h.coach.state.phase == "idle"
        await h.feed(40, 74, "resting")  # resting, but still baseline + 6
        await h.feed(10, 90, "moving")
        assert h.coach.state.phase == "idle"
        await h.feed(32, 73, "resting")  # 30 s at baseline + 5
        assert h.coach.state.phase == "idle"
        await h.feed(10, 90, "moving")
        assert h.coach.state.mode == "quiet" and h.coach.state.source == "auto"
    asyncio.run(go())
    assert h.spoken == []


def test_cooldown_cap_allows_auto_when_hr_never_settles(monkeypatch):
    monkeypatch.setenv("ACTIVITY_DETECT_S", "8")
    h = Harness(monkeypatch)

    async def go():
        await h.feed(10, 100, "moving")
        await h.coach.stop()
        h.clock[0] += 600_000  # 10 minutes, and the rate is still high
        await h.feed(10, 100, "moving")
        assert h.coach.state.mode == "quiet"
    asyncio.run(go())
    assert h.spoken == []


def test_manual_start_ignores_cooldown(monkeypatch):
    monkeypatch.setenv("ACTIVITY_DETECT_S", "8")
    h = Harness(monkeypatch)

    async def go():
        await h.feed(10, 90, "moving")
        await h.coach.stop()
        state = await h.coach.start()
        assert state.mode == "exercise" and state.source == "manual" and state.phase == "walking"
    asyncio.run(go())
    assert h.spoken[0][0] == "start" and "between 88 and 98" in h.spoken[0][1]


def test_weak_signal_readings_are_ignored(monkeypatch):
    monkeypatch.setenv("ACTIVITY_DETECT_S", "8")
    h = Harness(monkeypatch)

    async def go():
        await h.feed(30, 120, "moving", status="poor")
        assert h.coach.state.phase == "idle"
        await h.feed(6, 100, "moving")  # 4 s toward the 8 s
        await h.feed(30, 100, "moving", status="poor")  # must not finish the streak or reset it
        assert h.coach.state.phase == "idle"
        await h.feed(4, 100, "moving")  # +2 s of trusted movement → 6 s
        assert h.coach.state.phase == "idle"
        await h.feed(2, 100, "moving")  # +2 s → quiet
        assert h.coach.state.mode == "quiet"
        await h.feed(80, 65, "resting", status="poor")  # must not end the activity
        assert h.coach.state.mode == "quiet"
        await h.coach.stop()
        await h.feed(40, 70, "resting", status="poor")  # must not clear the cooldown
        await h.feed(10, 100, "moving")
        assert h.coach.state.phase == "idle"
    asyncio.run(go())
    assert h.spoken == []


def test_stop_in_exercise_runs_recovery(monkeypatch):
    monkeypatch.setenv("WALK_RECOVERY_S", "0.05")
    h = Harness(monkeypatch)

    async def go():
        await h.coach.on_reading(reading(h.clock[0], 70, "resting", baseline=68))
        await h.coach.start()
        assert h.coach.state.mode == "exercise"
        await h.feed(12, 110)
        await h.coach.stop()
        assert h.coach.state.phase == "recovering"
        assert any(k == "stop" for k, _ in h.spoken)
        await h.feed(6, 84, "resting")
        h.clock[0] = h.coach.state.recovery_due
        await asyncio.sleep(0.2)
    asyncio.run(go())
    assert h.coach.state.phase == "done" and h.coach.state.recovery_ok
    assert "healthy" in h.spoken[-1][1]


def test_demo_auto_story_starts_a_quiet_activity(monkeypatch):
    monkeypatch.setenv("ACTIVITY_DETECT_S", "120")
    h = Harness(monkeypatch)
    events = demo_sim.build_timeline("auto", CONTEXT)
    moving = [e.reading for e in events if e.reading and e.reading.activity == "moving"]
    assert moving and all(r.hr > r.baseline_hr + 10 for r in moving)
    assert (moving[-1].t - moving[0].t) / 1000 >= 120
    assert [e.action for e in events if e.action] == []
    assert [e for e in events if e.alert] == []

    async def go():
        for event in events:
            if event.reading is None:
                continue
            h.clock[0] = event.reading.t
            await h.coach.on_reading(event.reading)
        assert h.coach.state.mode == "quiet" and h.coach.state.source == "auto"
    asyncio.run(go())
    assert h.spoken == []


# ---------- demo simulator stories ----------

def test_demo_walk_story_presses_the_buttons_and_stays_quiet():
    events = demo_sim.build_timeline("walk", CONTEXT)
    assert [(e.offset_s, e.action) for e in events if e.action] == [(30, "activity_start"), (160, "activity_stop")]
    assert [e for e in events if e.alert] == []
    peak = max(e.reading.hr for e in events if e.reading)
    assert peak > 100  # goes above the 88-98 zone at some point
    exercise = demo_sim.build_timeline("exercise", CONTEXT)
    assert [(e.offset_s, e.action) for e in exercise if e.action] == [(30, "activity_start"), (160, "activity_stop")]
