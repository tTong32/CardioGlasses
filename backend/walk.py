"""Guided walk: live voice coaching against a heart-rate zone from the record, then a
one-minute recovery check.

It runs on the Readings the backend already receives, so it works with the glasses, a
replay, or the demo simulator. It speaks only when something needs saying: start, staying
above the zone, back in the zone, and the recovery result.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Awaitable, Callable, Optional

from ai.clinical import walk_zone
from ai.contracts import PatientContext, Reading

ABOVE_FOR_S = 6.0  # above the zone this long before saying "slow down"
TIP_COOLDOWN_S = 30.0  # don't repeat the same tip sooner than this
HEALTHY_DROP_BPM = 12  # a 1-minute drop of 12 bpm or less is the classic "slow recovery" cutoff


def now_ms() -> int:
    return int(time.time() * 1000)


def recovery_wait_s() -> float:
    return float(os.environ.get("WALK_RECOVERY_S") or 60)


@dataclass
class WalkState:
    phase: str = "idle"  # idle | walking | recovering | done
    started_at: Optional[int] = None
    ended_at: Optional[int] = None
    zone_low: Optional[int] = None
    zone_high: Optional[int] = None
    zone_note: str = ""
    hr_at_stop: Optional[float] = None
    recovery_due: Optional[int] = None
    drop_1min: Optional[float] = None
    recovery_ok: Optional[bool] = None
    last_tip: Optional[str] = None
    peak_hr: Optional[float] = None
    minutes: Optional[float] = None
    recent_hr: list = field(default_factory=list)  # (t, hr) of trusted readings, newest last

    def public(self) -> dict:
        data = asdict(self)
        data.pop("recent_hr")
        data["recovery_remaining_s"] = (
            max(0.0, (self.recovery_due - now_ms()) / 1000) if self.phase == "recovering" and self.recovery_due else 0.0
        )
        return data


Speak = Callable[[str, str], Awaitable[None]]  # (text, kind)
Broadcast = Callable[[dict], Awaitable[None]]
Note = Callable[[str], Awaitable[None]]


class Coach:
    def __init__(self, broadcast: Broadcast, speak: Speak, context: Callable[[], PatientContext],
                 name: Callable[[], str], note: Optional[Note] = None):
        self.broadcast = broadcast
        self.speak = speak
        self.context = context
        self.name = name
        self.note = note
        self.state = WalkState()
        self._last_resting: Optional[float] = None
        self._above_since: Optional[int] = None
        self._was_above = False
        self._last_tip_at: dict[str, int] = {}
        self._timer: Optional[asyncio.Task] = None

    # ---------- inputs ----------
    async def on_reading(self, r: Reading) -> None:
        trusted = r.hr is not None and r.signal_status in (None, "ok")
        if r.baseline_hr is not None:
            self._last_resting = r.baseline_hr
        if not trusted:
            return
        s = self.state
        s.recent_hr.append((r.t, r.hr))
        del s.recent_hr[:-15]
        if s.phase != "walking":
            return
        s.peak_hr = max(s.peak_hr or 0, r.hr)
        if r.hr > s.zone_high:
            self._above_since = self._above_since or now_ms()
            if (now_ms() - self._above_since) / 1000 >= ABOVE_FOR_S and await self._tip(
                "above", f"You're at {round(r.hr)}, above your zone of {s.zone_high}. Slow down a little."
            ):
                self._was_above = True
        else:
            self._above_since = None
            if self._was_above and r.hr <= s.zone_high - 2:
                self._was_above = False
                await self._tip("back", "Good, you're back in your zone. Keep this pace.", force=True)
        await self._push()

    async def start(self) -> WalkState:
        if self.state.phase == "walking":
            return self.state
        self._cancel_timer()
        low, high, note = walk_zone(self.context(), self._last_resting)
        self.state = WalkState(phase="walking", started_at=now_ms(), zone_low=low, zone_high=high, zone_note=note,
                               recent_hr=self.state.recent_hr)
        self._above_since, self._was_above, self._last_tip_at = None, False, {}
        await self._push()
        await self.speak(f"Walk started. Try to keep your heart rate between {low} and {high}. "
                         "I'll tell you if you go above it.", "start")
        return self.state

    async def stop(self) -> WalkState:
        s = self.state
        if s.phase != "walking":
            return s
        s.phase = "recovering"
        s.ended_at = now_ms()
        s.minutes = round((s.ended_at - s.started_at) / 60000, 1)
        s.hr_at_stop = self._hr_now()
        s.recovery_due = s.ended_at + int(recovery_wait_s() * 1000)
        await self._push()
        await self.speak("Walk finished. Stand or sit still for a minute, and I'll check how your heart recovers.", "stop")
        self._timer = asyncio.create_task(self._recovery_check(s.started_at))
        return s

    # ---------- internals ----------
    def _hr_now(self, window_ms: int = 6000) -> Optional[float]:
        recent = self.state.recent_hr
        if not recent:
            return None
        newest = recent[-1][0]
        return float(median(hr for t, hr in recent if newest - t <= window_ms))

    async def _recovery_check(self, walk_id: int) -> None:
        try:
            await asyncio.sleep(max(0.0, (self.state.recovery_due - now_ms()) / 1000))
        except asyncio.CancelledError:
            return
        s = self.state
        if s.phase != "recovering" or s.started_at != walk_id:
            return
        hr_now = self._hr_now()
        s.phase = "done"
        if s.hr_at_stop is None or hr_now is None:
            await self._push()
            await self.speak("I couldn't get a clear reading to check your recovery this time.", "recovery")
            return
        s.drop_1min = round(s.hr_at_stop - hr_now, 1)
        s.recovery_ok = s.drop_1min > HEALTHY_DROP_BPM
        drop = round(s.drop_1min)
        if s.recovery_ok:
            text = f"Your heart rate dropped {drop} beats in the first minute. That's a healthy recovery. Nice walk."
        else:
            text = (f"Your heart rate dropped {max(drop, 0)} beats in the first minute, which is slower than expected. "
                    "Take it easy, and mention it to your care team if it keeps happening.")
        await self._push()
        await self.speak(text, "recovery")
        if self.note is not None:
            verdict = "healthy" if s.recovery_ok else "slower than expected"
            await self.note(f"{self.name()} finished a {s.minutes:g}-minute walk. Recovery: {verdict} "
                            f"(down {max(drop, 0)} bpm in the first minute).")

    async def _tip(self, kind: str, text: str, force: bool = False) -> bool:
        last = self._last_tip_at.get(kind)
        if not force and last is not None and (now_ms() - last) / 1000 < TIP_COOLDOWN_S:
            return False
        self._last_tip_at[kind] = now_ms()
        self.state.last_tip = text
        await self.speak(text, kind)
        return True

    def _cancel_timer(self) -> None:
        if self._timer is not None and not self._timer.done():
            self._timer.cancel()
        self._timer = None

    async def _push(self) -> None:
        await self.broadcast({"type": "walk", "data": self.state.public()})
