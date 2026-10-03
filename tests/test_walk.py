"""Guided walk coaching (backend/walk.py) and the walk zone rule (ai/clinical.py)."""

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


# ---------- demo simulator walk story ----------

def test_demo_walk_story_presses_the_buttons_and_stays_quiet():
    events = demo_sim.build_timeline("walk", CONTEXT)
    assert [(e.offset_s, e.action) for e in events if e.action] == [(30, "walk_start"), (160, "walk_stop")]
    assert [e for e in events if e.alert] == []
    peak = max(e.reading.hr for e in events if e.reading)
    assert peak > 100  # goes above the 88-98 zone at some point
