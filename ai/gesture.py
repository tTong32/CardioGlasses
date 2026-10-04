"""Head gestures from the glasses' gyroscope: a nod ("I'm OK") or a shake ("I need help").

Only used while an "Are you OK?" check-in is waiting, so ordinary head movement can't
answer anything. A gesture is two or more strong swings in opposite directions on one
rotation axis within a short window, clearly stronger than any rotation on the other axis.

Axes follow Contract A (gx, gy, gz in deg/s). With the sensor's x axis pointing to the
wearer's right and y up, a nod rotates about x and a shake about y. If the board is mounted
differently, set GESTURE_NOD_AXIS / GESTURE_SHAKE_AXIS (e.g. "gz") in .env.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Literal, Optional

from ai.contracts import Sample

Gesture = Literal["nod", "shake"]

SWING_DPS = 60.0  # a deliberate nod peaks well above this; walking head-bob stays around 20-40
RELEASE_DPS = 25.0  # a swing ends when rotation drops back below this
WINDOW_S = 1.6  # the swings of one gesture must fall inside this window
DOMINANCE = 1.8  # gesture axis peak must beat the other axis by this factor
REFRACTORY_S = 2.0  # ignore further swings this long after a detection


def _axes() -> tuple[str, str]:
    nod = os.environ.get("GESTURE_NOD_AXIS") or "gx"
    shake = os.environ.get("GESTURE_SHAKE_AXIS") or "gy"
    return nod, shake


class GestureDetector:
    def __init__(self) -> None:
        self.nod_axis, self.shake_axis = _axes()
        self.reset()

    def reset(self) -> None:
        # Per axis: the swing in progress (sign, peak) and finished swings (t_ms, sign, peak).
        self._current: dict[str, Optional[list]] = {self.nod_axis: None, self.shake_axis: None}
        self._swings: dict[str, deque] = {self.nod_axis: deque(), self.shake_axis: deque()}
        self._quiet_until = 0

    def update(self, sample: Sample) -> Optional[Gesture]:
        """Feed one sample; returns "nod" or "shake" when one just completed."""
        t = sample.t
        for axis in (self.nod_axis, self.shake_axis):
            value = getattr(sample, axis, None)
            if value is None:
                continue
            self._track(axis, t, float(value))
        if t < self._quiet_until:
            return None
        for gesture, axis, other in (("nod", self.nod_axis, self.shake_axis), ("shake", self.shake_axis, self.nod_axis)):
            if self._is_gesture(axis, other, t):
                self._quiet_until = t + int(REFRACTORY_S * 1000)
                self._swings[self.nod_axis].clear()
                self._swings[self.shake_axis].clear()
                return gesture  # type: ignore[return-value]
        return None

    def _track(self, axis: str, t: int, value: float) -> None:
        current = self._current[axis]
        if current is None:
            if abs(value) >= SWING_DPS:
                self._current[axis] = [1 if value > 0 else -1, abs(value)]
            return
        sign, peak = current
        if value * sign > 0 and abs(value) > RELEASE_DPS:
            current[1] = max(peak, abs(value))
            return
        self._swings[axis].append((t, sign, peak))  # swing finished (fell back or reversed)
        self._current[axis] = None
        if abs(value) >= SWING_DPS:  # reversed straight into a new swing
            self._current[axis] = [1 if value > 0 else -1, abs(value)]
        cutoff = t - int(WINDOW_S * 1000)
        while self._swings[axis] and self._swings[axis][0][0] < cutoff:
            self._swings[axis].popleft()

    def _is_gesture(self, axis: str, other: str, t: int) -> bool:
        cutoff = t - int(WINDOW_S * 1000)
        swings = [s for s in self._swings[axis] if s[0] >= cutoff]
        if len(swings) < 2 or len({sign for _, sign, _ in swings}) < 2:
            return False
        peak = max(p for _, _, p in swings)
        other_peak = max((p for tt, _, p in self._swings[other] if tt >= cutoff), default=0.0)
        current_other = self._current[other]
        if current_other is not None:
            other_peak = max(other_peak, current_other[1])
        return peak >= DOMINANCE * other_peak
