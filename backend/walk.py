"""Activity: quiet movement detected from the glasses, or a coached exercise.

A manual start (or "make it exercise") coaches a heart-rate zone from the record and,
on stop, checks the one-minute recovery. Movement that keeps the heart rate above the
usual resting rate starts a quiet activity instead: no voice at all. It speaks only in
exercise mode, and only when something needs saying: the zone, staying above it, back
in the zone, and the recovery result.

`phase` stays "walking" while an activity is underway so older clients keep working.
`mode` is "quiet" or "exercise"; `source` is "auto" or "manual".
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
RAISED_BPM = 10  # auto-detect: heart rate must be above the usual resting rate by more than this
SETTLED_BPM = 5  # cooldown lifts after resting within the usual rate plus this


def now_ms() -> int:
    return int(time.time() * 1000)


def recovery_wait_s() -> float:
    return float(os.environ.get("WALK_RECOVERY_S") or 60)


def activity_detect_s() -> float:
    """How long movement with a raised heart rate must last before a quiet activity starts."""
    return float(os.environ.get("ACTIVITY_DETECT_S") or 120)


def quiet_rest_s() -> float:
    """An auto quiet activity ends on its own after this long at rest."""
    return float(os.environ.get("ACTIVITY_REST_S") or 60)


def settle_s() -> float:
    """Trusted rest near the usual rate for this long lifts the auto-detect cooldown."""
    return float(os.environ.get("ACTIVITY_SETTLE_S") or 30)


def cooldown_cap_s() -> float:
    """Safety cap: auto-detect is allowed again this long after a stop, even if HR hasn't settled."""
    return float(os.environ.get("ACTIVITY_COOLDOWN_CAP_S") or 600)


class Hold:
    """How long a condition has been true, counting trusted readings only.

    An untrusted reading freezes the clock: it does not add time and it does not reset
    the streak. A trusted reading where the condition is false resets it.
    """

    def __init__(self) -> None:
        self._since: Optional[int] = None
        self._last: Optional[int] = None
        self._skipped = 0
        self._frozen = False

    def reset(self) -> None:
        self._since = None
        self._last = None
        self._skipped = 0
        self._frozen = False

    def freeze(self) -> None:
        if self._since is not None:
            self._frozen = True

    def advance(self, t: int, on: bool) -> float:
        if not on:
            self.reset()
            return 0.0
        if self._since is None:
            self._since = t
            self._last = t
            self._frozen = False
            self._skipped = 0
            return 0.0
        if self._frozen and self._last is not None:
            self._skipped += max(0, t - self._last)
            self._frozen = False
        self._last = t
        return max(0.0, (t - self._since - self._skipped) / 1000.0)


@dataclass
class WalkState:
    phase: str = "idle"  # idle | walking | recovering | done
    mode: Optional[str] = None  # quiet | exercise
    source: Optional[str] = None  # auto | manual
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
        self._was_above = False
        self._last_tip_at: dict[str, int] = {}
        self._timer: Optional[asyncio.Task] = None
        self._above = Hold()
        self._move = Hold()
        self._rest = Hold()
        self._settle = Hold()
        self._cooldown_at: Optional[int] = None

    # ---------- inputs ----------
    async def on_reading(self, r: Reading) -> None:
        trusted = r.hr is not None and r.signal_status in (None, "ok")
        if not trusted:
            self._above.freeze()
            self._move.freeze()
            self._rest.freeze()
            self._settle.freeze()
            return
        if r.baseline_hr is not None:
            self._last_resting = r.baseline_hr
        s = self.state
        s.recent_hr.append((r.t, r.hr))
        del s.recent_hr[:-15]
        if s.phase == "walking" and s.mode == "exercise":
            await self._coach_zone(r)
            await self._push()
            return
        if s.phase == "walking" and s.mode == "quiet":
            if s.source == "auto" and self._rest.advance(r.t, r.activity == "resting") >= quiet_rest_s():
                await self._finish_quiet()
                return
            await self._push()
            return
        self._watch_settle(r)
        if s.phase != "recovering":
            await self._watch_auto(r)

    async def start(self) -> WalkState:
        """Manual start: an exercise. A quiet activity is promoted instead of restarted."""
        s = self.state
        if s.phase == "walking" and s.mode == "quiet":
            return await self.to_exercise()
        if s.phase == "walking":
            return s
        return await self._begin_exercise("manual")

    async def to_exercise(self) -> WalkState:
        """Switch a quiet activity to exercise without restarting the clock."""
        s = self.state
        if s.phase != "walking" or s.mode != "quiet":
            return s
        low, high, note = self._zone()
        s.mode = "exercise"
        s.zone_low, s.zone_high, s.zone_note = low, high, note
        self._above.reset()
        self._was_above, self._last_tip_at = False, {}
        self._rest.reset()
        await self._push()
        await self._announce(low, high)
        return s

    async def stop(self) -> WalkState:
        s = self.state
        if s.phase != "walking":
            return s
        if s.mode == "quiet":
            await self._finish_quiet()
            return self.state
        s.phase = "recovering"
        s.ended_at = now_ms()
        s.minutes = round((s.ended_at - s.started_at) / 60000, 1)
        s.hr_at_stop = self._hr_now()
        s.recovery_due = s.ended_at + int(recovery_wait_s() * 1000)
        self._arm_cooldown()
        await self._push()
        await self.speak("Exercise finished. Stand or sit still for a minute, and I'll check how your heart recovers.", "stop")
        self._timer = asyncio.create_task(self._recovery_check(s.started_at))
        return s

    async def end_recovery(self) -> WalkState:
        """Skip the recovery check: back to idle, nothing spoken or logged."""
        if self.state.phase != "recovering":
            return self.state
        self._cancel_timer()
        self.state = WalkState(recent_hr=self.state.recent_hr)
        await self._push()
        return self.state

    # ---------- internals ----------
    def _baseline(self, r: Reading) -> Optional[float]:
        return r.baseline_hr if r.baseline_hr is not None else self._last_resting

    def _zone(self) -> tuple:
        return walk_zone(self.context(), self._last_resting)

    def _raised(self, r: Reading) -> bool:
        baseline = self._baseline(r)
        return baseline is not None and r.hr > baseline + RAISED_BPM

    def _settled(self, r: Reading) -> bool:
        baseline = self._baseline(r)
        return r.activity == "resting" and baseline is not None and r.hr <= baseline + SETTLED_BPM

    def _cooldown_blocking(self) -> bool:
        if self._cooldown_at is None:
            return False
        if (now_ms() - self._cooldown_at) / 1000.0 >= cooldown_cap_s():
            self._cooldown_at = None
            self._settle.reset()
            return False
        return True

    def _arm_cooldown(self) -> None:
        self._cooldown_at = now_ms()
        self._move.reset()
        self._rest.reset()
        self._settle.reset()

    def _watch_settle(self, r: Reading) -> None:
        if not self._cooldown_blocking():
            return
        if self._settle.advance(r.t, self._settled(r)) >= settle_s():
            self._cooldown_at = None
            self._settle.reset()

    async def _watch_auto(self, r: Reading) -> None:
        held = self._move.advance(r.t, r.activity == "moving" and self._raised(r))
        if held >= activity_detect_s() and not self._cooldown_blocking():
            await self._begin_quiet()

    async def _begin_quiet(self) -> None:
        self._cancel_timer()
        self._move.reset()
        self._rest.reset()
        self._settle.reset()
        self._cooldown_at = None
        self._above.reset()
        self._was_above, self._last_tip_at = False, {}
        self.state = WalkState(
            phase="walking", mode="quiet", source="auto", started_at=now_ms(), recent_hr=self.state.recent_hr,
        )
        await self._push()

    async def _finish_quiet(self) -> None:
        self._cancel_timer()
        recent = self.state.recent_hr
        self.state = WalkState(recent_hr=recent)
        self._above.reset()
        self._arm_cooldown()
        await self._push()

    async def _begin_exercise(self, source: str) -> WalkState:
        self._cancel_timer()
        self._cooldown_at = None
        self._move.reset()
        self._rest.reset()
        self._settle.reset()
        low, high, note = self._zone()
        self.state = WalkState(
            phase="walking", mode="exercise", source=source, started_at=now_ms(),
            zone_low=low, zone_high=high, zone_note=note, recent_hr=self.state.recent_hr,
        )
        self._above.reset()
        self._was_above, self._last_tip_at = False, {}
        await self._push()
        await self._announce(low, high)
        return self.state

    async def _announce(self, low: int, high: int) -> None:
        await self.speak(
            f"Exercise started. Try to keep your heart rate between {low} and {high}. "
            "I'll tell you if you go above it.",
            "start",
        )

    def _hr_now(self, window_ms: int = 6000) -> Optional[float]:
        recent = self.state.recent_hr
        if not recent:
            return None
        newest = recent[-1][0]
        return float(median(hr for t, hr in recent if newest - t <= window_ms))

    async def _coach_zone(self, r: Reading) -> None:
        s = self.state
        if s.zone_high is None:
            return
        s.peak_hr = max(s.peak_hr or 0, r.hr)
        above = r.hr > s.zone_high
        held = self._above.advance(r.t, above)
        if above:
            if held >= ABOVE_FOR_S and await self._tip(
                "above", f"You're at {round(r.hr)}, above your zone of {s.zone_high}. Slow down a little."
            ):
                self._was_above = True
        elif self._was_above and r.hr <= s.zone_high - 2:
            self._was_above = False
            await self._tip("back", "Good, you're back in your zone. Keep this pace.", force=True)

    async def _recovery_check(self, activity_id: int) -> None:
        try:
            await asyncio.sleep(max(0.0, (self.state.recovery_due - now_ms()) / 1000))
        except asyncio.CancelledError:
            return
        s = self.state
        if s.phase != "recovering" or s.started_at != activity_id:
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
            text = f"Your heart rate dropped {drop} beats in the first minute. That's a healthy recovery. Nice work."
        else:
            text = (f"Your heart rate dropped {max(drop, 0)} beats in the first minute, which is slower than expected. "
                    "Take it easy, and mention it to your care team if it keeps happening.")
        await self._push()
        await self.speak(text, "recovery")
        if self.note is not None:
            verdict = "healthy" if s.recovery_ok else "slower than expected"
            await self.note(f"{self.name()} finished a {s.minutes:g}-minute exercise. Recovery: {verdict} "
                            f"(down {max(drop, 0)} bpm in the first minute).")

    async def _tip(self, kind: str, text: str, force: bool = False) -> bool:
        if self.state.mode != "exercise":
            return False
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
        data = self.state.public()
        await self.broadcast({"type": "activity", "data": data})
        await self.broadcast({"type": "walk", "data": data})
