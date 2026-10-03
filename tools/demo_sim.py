"""Demo simulator: stands in for the AI pipeline and posts scripted Readings and Alerts.

Use it to rehearse the demo (dashboard + voice) before the AI is wired up, and as the
fallback if the glasses or AI fail on the day. Alerts follow the planned rules
(quality OK + resting + deviation >= trigger for persist_s) so the story matches.

    python -m tools.demo_sim                       # elevated-at-rest story, real time
    python -m tools.demo_sim --scenario poor-signal --speed 3
    python -m tools.demo_sim --list
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

from ai.contracts import Alert, Level, PatientContext, Reading

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

STEP_S = 2
BASELINE_HR = 68.0
BASELINE_SD = 8.0  # typical day-to-day resting spread; keeps deviation in plausible SD units
RECOVERY_TAU_S = 35.0
CARDIAC_TERMS = ("atrial fibrillation", "heart failure", "coronary", "arrhythmia", "cardiomyopathy")

SCENARIOS = {
    "elevated": "Rest, light exertion, then HR stays high while seated -> monitor -> notify -> recovery -> normal",
    "escalate": "Like elevated, but HR stays high long enough to escalate",
    "poor-signal": "HR goes high but the signal is poor -> no alert, dashboard says poor signal",
    "dropout": "Normal rest with a 16 s gap in data -> dashboard shows sensor offline",
    "normal": "Quiet rest only; HR stays near baseline, no alerts",
    "walk": "Guided walk: presses Walk, drifts above the zone (voice: slow down), back in zone, ends, healthy recovery",
}


@dataclass
class Phase:
    end_s: float
    activity: str
    hr_from: float
    hr_to: float
    shape: str = "linear"  # "linear" or "recovery" (exponential toward hr_to)
    quality: float = 0.9
    signal_status: str = "ok"
    send: bool = True
    quiet: bool = False  # expected (e.g. cooling down after a walk): never counts toward an alert


def phases_for(scenario: str) -> list[Phase]:
    rest = Phase(60, "resting", BASELINE_HR, BASELINE_HR)
    exert = Phase(100, "moving", BASELINE_HR, 112)
    if scenario == "walk":
        return [
            Phase(30, "resting", BASELINE_HR, BASELINE_HR),
            Phase(110, "moving", BASELINE_HR + 2, 110),  # pace picks up; crosses the zone top
            Phase(160, "moving", 110, 94),  # slows down after the voice tip
            Phase(250, "resting", 94, BASELINE_HR + 2, shape="recovery", quiet=True),
        ]
    if scenario == "normal":
        return [Phase(180, "resting", BASELINE_HR, BASELINE_HR)]
    if scenario == "dropout":
        return [
            rest,
            Phase(76, "resting", BASELINE_HR, BASELINE_HR, quality=0.0, signal_status="offline", send=False),
            Phase(140, "resting", BASELINE_HR, BASELINE_HR),
        ]
    if scenario == "poor-signal":
        return [
            rest,
            exert,
            Phase(170, "resting", 108, 104, quality=0.3, signal_status="poor"),
            # Signal stays poor until HR is back under the trigger, so no alert fires on recovery.
            Phase(200, "resting", 104, 80, quality=0.3, signal_status="poor"),
            Phase(260, "resting", 80, BASELINE_HR + 2, shape="recovery"),
        ]
    high_until = 230 if scenario == "escalate" else 150
    return [
        rest,
        exert,
        Phase(high_until, "resting", 108, 104),
        Phase(high_until + 90, "resting", 104, BASELINE_HR + 2, shape="recovery"),
        Phase(high_until + 120, "resting", BASELINE_HR + 2, BASELINE_HR),
    ]


def _phase_at(phases: list[Phase], s: float) -> tuple[Phase, float]:
    start = 0.0
    for phase in phases:
        if s < phase.end_s:
            return phase, start
        start = phase.end_s
    return phases[-1], start


def _hr(phase: Phase, start: float, s: float) -> float:
    elapsed = s - start
    if phase.shape == "recovery":
        return phase.hr_to + (phase.hr_from - phase.hr_to) * math.exp(-elapsed / RECOVERY_TAU_S)
    frac = elapsed / max(phase.end_s - start, 1e-9)
    return phase.hr_from + (phase.hr_to - phase.hr_from) * frac


def _cardiac(context: PatientContext) -> list[str]:
    return [c for c in context.conditions if any(term in c.lower() for term in CARDIAC_TERMS)]


def make_alert(level: Level, reading: Reading, context: PatientContext) -> Alert:
    """Templated wording until the LLM explainer exists."""
    hr = round(reading.hr or 0)
    usual = round(reading.baseline_hr or BASELINE_HR)
    history = _cardiac(context)
    history_text = f" Your record includes {' and '.join(c.lower() for c in history[:2])}, so this is worth a check." if history else ""
    persisted = round(reading.persist_s or 0)
    texts = {
        "monitor": (
            0.6,
            "Heart rate a little high at rest",
            f"Your heart rate is {hr} bpm while you're sitting still; your usual resting rate is {usual}. I'm watching to see if it settles.",
            "I'm keeping a closer eye on your heart rate. No action needed yet.",
            "Stay seated and breathe normally",
        ),
        "notify": (
            0.8,
            "Elevated heart rate at rest",
            f"Your heart rate has stayed around {hr} bpm for {persisted} seconds while resting, well above your usual {usual}.{history_text}",
            "Your heart rate has stayed high while you're resting. Please sit down and check your phone.",
            "Sit down and review",
        ),
        "escalate": (
            0.85,
            "Heart rate has stayed high",
            f"Your heart rate has been around {hr} bpm for {persisted} seconds at rest, far above your usual {usual}.{history_text} Consider contacting your care team.",
            "Your heart rate has stayed high for a while. Please sit down, check your phone, and consider contacting your care team.",
            "Contact your care team",
        ),
        "normal": (
            0.9,
            "Back within your usual range",
            f"Your heart rate is {hr} bpm, close to your usual resting rate of {usual}. Monitoring continues.",
            "Your heart rate is back within your usual range. I'll keep monitoring.",
            "No action needed",
        ),
    }
    confidence, headline, reason, voice_text, next_step = texts[level]
    return Alert(
        t=reading.t, level=level, confidence=confidence, headline=headline, reason=reason,
        voice_text=voice_text, next_step=next_step, reading=reading,
    )


@dataclass
class Event:
    offset_s: float
    reading: Reading | None  # None = sensor dropout (nothing sent)
    alert: Alert | None
    action: str | None = None  # "walk_start" / "walk_stop": press the dashboard's Walk button


WALK_ACTIONS = {"walk": {30: "walk_start", 160: "walk_stop"}}


def build_timeline(scenario: str, context: PatientContext, t0_ms: int = 0, seed: int = 7) -> list[Event]:
    """Every 2 s: a Reading, plus an Alert when the level changes. Pure, so it's testable."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {', '.join(SCENARIOS)}")
    rng = random.Random(seed)
    cfg = context.config
    phases = phases_for(scenario)
    total = phases[-1].end_s
    events: list[Event] = []
    level: Level = "normal"
    persist = 0.0
    peak_hr: float | None = None
    recovery_start: float | None = None
    s = STEP_S
    while s <= total:
        phase, start = _phase_at(phases, s - 1e-9)
        if not phase.send:
            events.append(Event(s, None, None))
            s += STEP_S
            continue
        hr = _hr(phase, start, s) + rng.gauss(0, 1.2)
        deviation = (hr - BASELINE_HR) / BASELINE_SD
        quality = max(0.0, min(1.0, phase.quality + rng.gauss(0, 0.03)))
        trusted = quality >= cfg.min_quality and phase.signal_status == "ok"
        if trusted and phase.activity == "resting" and not phase.quiet and deviation >= cfg.deviation_trigger:
            persist += STEP_S
        else:
            persist = 0.0

        recovery = {}
        if phase.shape == "recovery":
            if recovery_start is None:
                recovery_start, peak_hr = s, phase.hr_from
            drop_60 = (peak_hr or hr) - hr if s - recovery_start >= 60 else None
            recovery = {
                "recovery_tau_s": RECOVERY_TAU_S,
                "hr_drop_60s": round(drop_60, 1) if drop_60 is not None else None,
                "recovery_ratio": 1.0,
                "recovery_percentile": 45.0,
                "recovery_verdict": "normal",
            }
        ibi = round(60000 / hr)
        reading = Reading(
            t=t0_ms + int(s * 1000),
            hr=round(hr, 1),
            ibi_ms=[ibi + rng.randint(-15, 15) for _ in range(3)],
            activity=phase.activity,
            quality=round(quality, 2),
            baseline_hr=BASELINE_HR,
            deviation=round(deviation, 2),
            persist_s=persist,
            recovery_tau_s=recovery.get("recovery_tau_s"),
            hr_drop_60s=recovery.get("hr_drop_60s"),
            recovery_ratio=recovery.get("recovery_ratio"),
            recovery_percentile=recovery.get("recovery_percentile"),
            recovery_verdict=recovery.get("recovery_verdict"),
            signal_status=phase.signal_status,
        )

        new_level: Level = level
        if persist >= 3 * cfg.persist_s:
            new_level = "escalate"
        elif persist >= cfg.persist_s:
            new_level = "notify" if level != "escalate" else level
        elif persist > 0 and level == "normal":
            new_level = "monitor"
        elif trusted and deviation < 1.0:
            new_level = "normal"
        alert = make_alert(new_level, reading, context) if new_level != level else None
        level = new_level
        events.append(Event(s, reading, alert, WALK_ACTIONS.get(scenario, {}).get(int(s))))
        s += STEP_S
    return events


def _post(base_url: str, path: str, model) -> None:
    response = requests.post(f"{base_url}{path}", json=model.model_dump(mode="json"), timeout=15)
    response.raise_for_status()


def run(scenario: str, base_url: str, speed: float, loop: bool) -> None:
    base_url = base_url.rstrip("/")
    context = PatientContext.model_validate(requests.get(f"{base_url}/patient", timeout=5).json())
    print(f"Scenario '{scenario}' at {speed}x for patient {context.patient_id}. Ctrl+C to stop.")
    while True:
        started = time.monotonic()
        for event in build_timeline(scenario, context, seed=random.randrange(1 << 30)):
            delay = event.offset_s / speed - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
            if event.action:
                path = "/walk/start" if event.action == "walk_start" else "/walk/stop"
                requests.post(f"{base_url}{path}", timeout=15).raise_for_status()
                print(f"{event.offset_s:5.0f}s  -> {event.action.replace('_', ' ')}")
            if event.reading is None:
                print(f"{event.offset_s:5.0f}s  (sensor dropout, nothing sent)")
                continue
            # Wall-clock time so the dashboard's "updated" and offline checks are honest.
            now_ms = int(time.time() * 1000)
            reading = event.reading.model_copy(update={"t": now_ms})
            _post(base_url, "/readings", reading)
            line = f"{event.offset_s:5.0f}s  hr {reading.hr:5.1f}  {reading.activity:7}  dev {reading.deviation:4.1f}  persist {reading.persist_s:3.0f}s"
            if event.alert is not None:
                alert = event.alert.model_copy(update={"t": now_ms, "reading": reading})
                _post(base_url, "/alerts", alert)
                line += f"  -> ALERT {alert.level.upper()}: {alert.headline}"
            print(line, flush=True)
        if not loop:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Post scripted Readings and Alerts to the backend.")
    parser.add_argument("--scenario", default="elevated", choices=sorted(SCENARIOS))
    parser.add_argument("--speed", type=float, default=1.0, help="Playback multiplier (3 = three times faster)")
    parser.add_argument("--loop", action="store_true", help="Repeat until stopped")
    parser.add_argument("--base-url", default=os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--list", action="store_true", help="Describe the scenarios and exit")
    args = parser.parse_args()
    if args.list:
        for name, text in SCENARIOS.items():
            print(f"{name:12} {text}")
        return
    if args.speed <= 0:
        parser.error("--speed must be > 0")
    try:
        run(args.scenario, args.base_url, args.speed, args.loop)
    except KeyboardInterrupt:
        print("stopped")
    except requests.ConnectionError:
        parser.exit(1, f"Can't reach the backend at {args.base_url}. Is uvicorn running?\n")


if __name__ == "__main__":
    main()
