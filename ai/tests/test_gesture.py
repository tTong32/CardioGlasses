"""Nod / shake detection from the glasses' gyroscope."""

from __future__ import annotations

import math
import random

from ai.contracts import Sample
from ai.gesture import GestureDetector


def stream(seconds: float, gx=lambda t: 0.0, gy=lambda t: 0.0, gz=lambda t: 0.0, t0: int = 0, noise: float = 2.0, seed: int = 1):
    rng = random.Random(seed)
    out = []
    for i in range(int(seconds * 50)):
        t = i / 50
        out.append(Sample(t=t0 + int(t * 1000), ppg=50000, ax=0.0, ay=-0.98, az=0.1,
                          gx=gx(t) + rng.gauss(0, noise), gy=gy(t) + rng.gauss(0, noise), gz=gz(t) + rng.gauss(0, noise)))
    return out


def burst(amplitude: float, hz: float, start: float, cycles: float):
    end = start + cycles / hz
    return lambda t: amplitude * math.sin(2 * math.pi * hz * (t - start)) if start <= t < end else 0.0


def detect(samples):
    detector = GestureDetector()
    return [g for s in samples if (g := detector.update(s))]


def test_nod_is_detected_once():
    assert detect(stream(3, gx=burst(130, 2.5, 1.0, 2))) == ["nod"]


def test_shake_is_detected_once():
    assert detect(stream(3, gy=burst(150, 3.0, 1.0, 2))) == ["shake"]


def test_a_single_small_nod_counts():
    assert detect(stream(3, gx=burst(80, 2.0, 1.0, 1))) == ["nod"]


def test_still_head_and_walking_bob_are_ignored():
    assert detect(stream(5)) == []
    walking = lambda t: 30 * math.sin(2 * math.pi * 1.8 * t)  # head bob while walking
    assert detect(stream(10, gx=walking, gy=lambda t: 0.5 * walking(t))) == []


def test_one_way_turn_is_not_a_gesture():
    # Looking to one side and staying there: one swing, no return.
    assert detect(stream(3, gy=lambda t: 120 if 1.0 <= t < 1.25 else 0.0)) == []


def test_diagonal_wobble_is_ambiguous_and_ignored():
    wobble = burst(120, 2.5, 1.0, 2)
    assert detect(stream(3, gx=wobble, gy=wobble)) == []


def test_axes_can_be_remapped(monkeypatch):
    monkeypatch.setenv("GESTURE_NOD_AXIS", "gz")
    assert detect(stream(3, gz=burst(130, 2.5, 1.0, 2))) == ["nod"]


# ---------- answering a check-in from the pipeline ----------

class FakeBackend:
    def __init__(self):
        self.current = None
        self.answers = []
        self.now = 0.0

    def get(self, url, timeout=None):
        current = self.current
        return type("R", (), {"json": lambda _self: {"current": current}})()

    def post(self, url, json=None, timeout=None):
        self.answers.append((url.rsplit("/", 2)[-2], json["answer"]))
        self.current = None

    def clock(self):
        return self.now


def run_listener(backend, samples, open_at_s=None):
    from ai.pipeline import CheckinListener

    listener = CheckinListener("http://x", get=backend.get, post=backend.post, clock=backend.clock)
    answers = []
    for s in samples:
        backend.now = s.t / 1000
        if open_at_s is not None and backend.now >= open_at_s and backend.current is None and not backend.answers:
            backend.current = {"id": "c1", "status": "waiting"}
        if (a := listener.feed(s)):
            answers.append(a)
    return answers


def test_nod_answers_a_waiting_checkin():
    backend = FakeBackend()
    assert run_listener(backend, stream(4, gx=burst(130, 2.5, 2.0, 2)), open_at_s=0.5) == ["ok"]
    assert backend.answers == [("c1", "ok")]


def test_shake_asks_for_help():
    backend = FakeBackend()
    assert run_listener(backend, stream(4, gy=burst(150, 3.0, 2.0, 2)), open_at_s=0.5) == ["help"]


def test_gestures_without_a_checkin_do_nothing():
    backend = FakeBackend()
    assert run_listener(backend, stream(4, gx=burst(130, 2.5, 2.0, 2))) == []
    assert backend.answers == []


def test_a_nod_before_the_question_does_not_count():
    backend = FakeBackend()
    # Nod at 0.5-1.3 s, question opens at 2.0 s, no movement afterwards.
    assert run_listener(backend, stream(4, gx=burst(130, 2.5, 0.5, 2)), open_at_s=2.0) == []
