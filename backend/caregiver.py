"""Caregiver safety net.

When an alert reaches notify or escalate, the wearer gets an "Are you OK?" check-in with a
countdown. "I'm OK" closes it. "I need help", or no answer before the deadline, notifies the
caregiver: on the caregiver screen always, and by iMessage (Photon) when configured. If the
heart rate returns to normal first, the check-in closes quietly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

from dotenv import load_dotenv

from ai.contracts import Alert

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
SENDER = ROOT / "tools" / "imessage" / "send.mjs"
CHECKIN_LEVELS = ("notify", "escalate")
SEVERITY = {"normal": 0, "monitor": 1, "notify": 2, "escalate": 3}
MAX_EVENTS = 50

log = logging.getLogger(__name__)


def now_ms() -> int:
    return int(time.time() * 1000)


def timeout_s() -> float:
    return float(os.environ.get("CHECKIN_TIMEOUT_S") or 60)


def caregiver_name() -> str:
    return os.environ.get("CAREGIVER_NAME") or "your caregiver"


def imessage_configured() -> bool:
    return all(os.environ.get(k) for k in ("CAREGIVER_PHONE", "PHOTON_IMESSAGE_ADDRESS", "PHOTON_IMESSAGE_TOKEN"))


def send_imessage(text: str) -> tuple[str, Optional[str]]:
    """Send `text` to CAREGIVER_PHONE through Photon. Returns (channel, detail)."""
    if not imessage_configured():
        return "screen", "iMessage not configured"
    node = shutil.which("node")
    if node is None:
        return "imessage_failed", "node is not installed"
    try:
        done = subprocess.run(
            [node, str(SENDER), os.environ["CAREGIVER_PHONE"]],
            input=text, capture_output=True, text=True, timeout=20, cwd=SENDER.parent,
        )
        result = json.loads((done.stdout or "{}").strip().splitlines()[-1])
    except (subprocess.TimeoutExpired, ValueError, IndexError) as exc:
        return "imessage_failed", str(exc)
    if result.get("ok"):
        return "imessage", result.get("id") or result.get("guid")
    return "imessage_failed", result.get("error") or done.stderr.strip()[:200]


def reading_summary(alert: Alert, name: str) -> str:
    r = alert.reading
    parts = []
    if r.hr is not None:
        held = int(r.persist_s or 0)
        lasted = f" for {held} seconds" if held >= 2 else ""
        where = "at rest" if r.activity == "resting" else "while moving" if r.activity == "moving" else ""
        parts.append(f"{name}'s heart rate has been around {round(r.hr)} bpm {where}{lasted}".replace("  ", " "))
        if r.baseline_hr is not None:
            parts[-1] += f" (usual {round(r.baseline_hr)})"
        parts[-1] += "."
    if alert.next_step:
        step = re.sub(r"\byour\b", "their", alert.next_step.lower())  # the patient's instruction, retold
        parts.append(f"{name} was asked to {step}.")
    return " ".join(parts)


@dataclass
class CheckIn:
    id: str
    level: str
    started_at: int
    deadline: int
    alert: dict
    status: str = "waiting"  # waiting | ok | help | no_response | resolved
    ended_at: Optional[int] = None
    acknowledged: bool = False

    def public(self) -> dict:
        data = asdict(self)
        data["remaining_s"] = max(0.0, (self.deadline - now_ms()) / 1000) if self.status == "waiting" else 0.0
        return data


@dataclass
class CaregiverEvent:
    t: int
    kind: str  # started | ok | help | no_response | resolved | acknowledged
    text: str
    channel: str = "screen"  # screen | imessage | imessage_failed
    detail: Optional[str] = None
    urgent: bool = False


Broadcast = Callable[[dict], Awaitable[None]]


@dataclass
class SafetyNet:
    broadcast: Broadcast
    patient_name: Callable[[], str]
    current: Optional[CheckIn] = None
    events: list = field(default_factory=list)
    _timer: Optional[asyncio.Task] = None

    def snapshot(self) -> dict:
        return {
            "current": self.current.public() if self.current else None,
            "events": [asdict(e) for e in self.events],
            "patient_name": self.patient_name(),
            "caregiver_name": caregiver_name(),
            "imessage_configured": imessage_configured(),
            "timeout_s": timeout_s(),
        }

    async def on_alert(self, alert: Alert) -> None:
        waiting = self.current is not None and self.current.status == "waiting"
        if alert.level in CHECKIN_LEVELS:
            if waiting:
                if SEVERITY[alert.level] > SEVERITY[self.current.level]:
                    self.current.level = alert.level
                    self.current.alert = alert.model_dump(mode="json")
                    await self._push_checkin()
                return
            self._cancel_timer()
            started = now_ms()
            self.current = CheckIn(
                id=uuid.uuid4().hex[:12], level=alert.level, started_at=started,
                deadline=started + int(timeout_s() * 1000), alert=alert.model_dump(mode="json"),
            )
            name = self.patient_name()
            await self._record(CaregiverEvent(
                now_ms(), "started", f"Checking on {name}: {alert.headline or 'heart rate alert'}. Waiting for a reply."))
            await self._push_checkin()
            self._timer = asyncio.create_task(self._expire(self.current.id))
        elif alert.level == "normal" and waiting:
            await self._close("resolved", f"{self.patient_name()}'s heart rate is back to normal. No action needed.")

    async def respond(self, checkin_id: str, answer: str) -> CheckIn:
        if self.current is None or self.current.id != checkin_id:
            raise KeyError(checkin_id)
        if self.current.status != "waiting":
            raise ValueError(self.current.status)
        name = self.patient_name()
        if answer == "ok":
            await self._close("ok", f"{name} says they're OK.")
        else:
            alert = Alert.model_validate(self.current.alert)
            text = f"CardioGlasses: {name} tapped \"I need help\". {reading_summary(alert, name)}"
            await self._close("help", text, notify=True)
        return self.current

    async def note(self, text: str) -> None:
        """A non-urgent line for the caregiver's activity log (e.g. a finished walk)."""
        await self._record(CaregiverEvent(now_ms(), "note", text))

    async def acknowledge(self) -> None:
        if self.current is not None and self.current.status in ("help", "no_response") and not self.current.acknowledged:
            self.current.acknowledged = True
            await self._record(CaregiverEvent(now_ms(), "acknowledged", f"{caregiver_name().capitalize()} is on it."))
            await self._push_checkin()

    async def _expire(self, checkin_id: str) -> None:
        try:
            await asyncio.sleep(max(0.0, (self.current.deadline - now_ms()) / 1000))
        except asyncio.CancelledError:
            return
        if self.current is None or self.current.id != checkin_id or self.current.status != "waiting":
            return
        name = self.patient_name()
        alert = Alert.model_validate(self.current.alert)
        text = f"CardioGlasses: {name} hasn't responded to a check-in. {reading_summary(alert, name)}"
        await self._close("no_response", text, notify=True)

    async def _close(self, status: str, text: str, notify: bool = False) -> None:
        if self._timer is not None and self._timer is not asyncio.current_task():
            self._cancel_timer()
        self.current.status = status
        self.current.ended_at = now_ms()
        channel, detail = ("screen", None)
        if notify:
            channel, detail = await asyncio.to_thread(send_imessage, text)
            if channel == "imessage_failed":
                log.warning("Caregiver iMessage failed: %s", detail)
        await self._record(CaregiverEvent(now_ms(), status, text, channel, detail, urgent=notify))
        await self._push_checkin()

    def _cancel_timer(self) -> None:
        if self._timer is not None and not self._timer.done():
            self._timer.cancel()
        self._timer = None

    async def _record(self, event: CaregiverEvent) -> None:
        self.events.insert(0, event)
        del self.events[MAX_EVENTS:]
        await self.broadcast({"type": "caregiver", "data": asdict(event)})

    async def _push_checkin(self) -> None:
        await self.broadcast({"type": "checkin", "data": self.current.public()})
