"""Samples in, Readings (and Alerts) out.

Every STEP_S of sample time the last WINDOW_DURATION_S of samples is analysed:
PPG -> HR, beat intervals, quality; IMU -> resting/moving; HR + activity -> baseline,
deviation, persistence, recovery; then the decision engine may emit an Alert.

    python -m ai.pipeline --scenario slow_recovery --speed 10     # synthetic, posts to backend
    python -m ai.pipeline data/rec_rest.csv --speed 5             # a recording
    python -m ai.pipeline --scenario elevated_rest --dry-run      # print only, no backend
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

import requests
from dotenv import load_dotenv

import ai.config as cfg
from ai.activity import ActivityDetector
from ai.baseline import BaselineTracker
from ai.clinical import baseline_min_std, irregular_rhythm_expected, tuned_config
from ai.contracts import Alert, PatientContext, Reading, Sample
from ai.decision import POST_EXERTION_GRACE_S, Context, DecisionEngine
from ai.explainer import Explainer
from ai.processing import analyze_ppg, classify_activity
from ai.recovery import RecoveryModel, RecoveryOutput
from ai.replay import SCENARIOS, generate_scenario, iter_samples, load_csv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
CUTOFFS_PATH = ROOT / "ai" / "model" / "recovery_cutoffs.json"
STEP_S = 2.0
MIN_WINDOW_S = 6.0  # no Readings until the window holds this much data
# A low-quality window is no evidence either way: persistence holds through it and only
# resets after this long without a trusted reading.
EVIDENCE_HOLD_S = 10.0

log = logging.getLogger(__name__)


@dataclass
class Step:
    reading: Reading
    alert: Optional[Alert]


class Pipeline:
    """Stateful sample-to-Reading pipeline for one wearer."""

    def __init__(self, context: PatientContext, explainer: Optional[Explainer] = None):
        self.context = context
        self.window: deque[Sample] = deque()
        self.activity = ActivityDetector()
        self.baseline = BaselineTracker()
        self.recovery = RecoveryModel(cutoffs_path=str(CUTOFFS_PATH), medications=context.medications)
        self.engine = DecisionEngine(context, explainer)
        self.irregular_rhythm = irregular_rhythm_expected(context)
        self.min_std = baseline_min_std(context)
        self._next_emit_ms: Optional[int] = None
        self._last_emit_ms: Optional[int] = None
        self._last_sample_ms: Optional[int] = None
        self._last_moving_ms: Optional[int] = None
        self._accel_seen = False
        self._persist_s = 0.0
        self._untrusted_since_ms: Optional[int] = None
        self._shown_recovery = RecoveryOutput()

    def push(self, sample: Sample) -> Optional[Step]:
        """Add one sample. Returns a Step every STEP_S of sample time, else None."""
        if self._last_sample_ms is not None and sample.t <= self._last_sample_ms:
            return None  # duplicate or out of order
        if (
            self._last_sample_ms is not None
            and (sample.t - self._last_sample_ms) / 1000.0 > cfg.SIGNAL_OFFLINE_TIMEOUT_S
        ):
            self.window.clear()  # data gap: don't analyse across it
        self._last_sample_ms = sample.t

        self.window.append(sample)
        cutoff = sample.t - int(cfg.WINDOW_DURATION_S * 1000)
        while self.window and self.window[0].t < cutoff:
            self.window.popleft()
        if sample.ax is not None and sample.ay is not None and sample.az is not None:
            self.activity.update(sample.ax, sample.ay, sample.az)
            self._accel_seen = True

        if self._next_emit_ms is None:
            self._next_emit_ms = sample.t + int(STEP_S * 1000)
        if sample.t < self._next_emit_ms:
            return None
        self._next_emit_ms = max(self._next_emit_ms + int(STEP_S * 1000), sample.t + 1)
        if (self.window[-1].t - self.window[0].t) / 1000.0 < MIN_WINDOW_S:
            return None
        return self._emit(sample.t)

    def _emit(self, t: int) -> Step:
        cfg_p = self.context.config
        window = list(self.window)
        ppg = analyze_ppg(window, irregular_rhythm=self.irregular_rhythm)
        activity = self.activity.current_activity if self._accel_seen else classify_activity(window)
        if activity == "moving":
            self._last_moving_ms = t
        since_moving = None if self._last_moving_ms is None else (t - self._last_moving_ms) / 1000.0
        in_grace = since_moving is not None and since_moving < POST_EXERTION_GRACE_S

        hr = ppg.hr
        trusted = hr is not None and ppg.quality >= cfg_p.min_quality
        if "missing" in ppg.reason:
            signal_status = "offline"
        else:
            signal_status = "ok" if trusted else "poor"

        # Deviation uses the baseline as it was before this reading.
        base_hr, base_sd = self.baseline.baseline_hr, self.baseline.baseline_std
        deviation = None
        if hr is not None and base_hr is not None:
            deviation = (hr - base_hr) / max(base_sd or 0.0, self.min_std)

        # HR is already withheld below HR_MIN_QUALITY (where it stops being accurate), so the
        # recovery model gets it even when motion keeps quality under min_quality.
        rec = self.recovery.update(t, hr, activity or "resting", base_hr, base_sd)
        recovering = rec.episode_state == "active"
        if rec.recovery_tau_s is not None:
            self._shown_recovery = rec
        elif activity == "moving":
            self._shown_recovery = RecoveryOutput()

        # Only clean, calm, resting HR feeds the baseline, so an episode can't become "normal".
        if activity == "moving":
            self.baseline.update(t, hr, "moving")
        elif (
            trusted
            and not recovering
            and not in_grace
            and (deviation is None or deviation < cfg.DEVIATION_NORMAL_THRESHOLD)
        ):
            self.baseline.update(t, hr, "resting")

        step_s = STEP_S if self._last_emit_ms is None else (t - self._last_emit_ms) / 1000.0
        self._last_emit_ms = t
        if activity == "moving":
            self._persist_s, self._untrusted_since_ms = 0.0, None
        elif trusted and deviation is not None:
            self._untrusted_since_ms = None
            self._persist_s = self._persist_s + step_s if deviation >= cfg_p.deviation_trigger else 0.0
        else:  # resting but no trustworthy evidence: hold, unless it has gone on too long
            if self._untrusted_since_ms is None:
                self._untrusted_since_ms = t
            if (t - self._untrusted_since_ms) / 1000.0 > EVIDENCE_HOLD_S:
                self._persist_s = 0.0

        shown = self._shown_recovery
        reading = Reading(
            t=t,
            hr=hr,
            ibi_ms=ppg.ibi_ms[-5:] or None,
            activity=activity,
            quality=ppg.quality,
            baseline_hr=None if base_hr is None else round(base_hr, 1),
            deviation=None if deviation is None else round(deviation, 2),
            persist_s=round(self._persist_s, 1),
            recovery_tau_s=None if shown.recovery_tau_s is None else round(shown.recovery_tau_s, 1),
            hr_drop_60s=None if shown.hr_drop_60s is None else round(shown.hr_drop_60s, 1),
            recovery_ratio=None if shown.recovery_ratio is None else round(shown.recovery_ratio, 2),
            recovery_percentile=None if shown.recovery_percentile is None else round(shown.recovery_percentile, 1),
            recovery_verdict=shown.recovery_verdict,
            signal_status=signal_status,
        )
        alert = self.engine.update(
            reading,
            Context(
                recovering=recovering,
                live_verdict=rec.recovery_verdict if recovering else None,
                seconds_since_moving=since_moving,
            ),
        )
        return Step(reading, alert)

    def run(self, samples: Iterable[Sample]) -> Iterator[Step]:
        for sample in samples:
            step = self.push(sample)
            if step is not None:
                yield step


# ---------- CLI: replay into the backend ----------

def post_model(base_url: str, path: str, model: Union[Reading, Alert]) -> None:
    response = requests.post(f"{base_url.rstrip('/')}{path}", json=model.model_dump(mode="json"), timeout=15)
    response.raise_for_status()


def fetch_context(base_url: str) -> PatientContext:
    response = requests.get(f"{base_url.rstrip('/')}/patient", timeout=5)
    response.raise_for_status()
    return PatientContext.model_validate(response.json())


def offline_context() -> PatientContext:
    """The last FinchNode patient the backend cached, else the checked-in demo patient."""
    local = PatientContext.model_validate_json((ROOT / "data" / "patient.json").read_text(encoding="utf-8"))
    cached = ROOT / "data" / "finchnode_cache.json"
    record = PatientContext.model_validate_json(cached.read_text(encoding="utf-8")) if cached.is_file() else local
    return record.model_copy(update={"config": tuned_config(local.config, record.risk_tier)})


def load_samples(csv_path: Optional[str], scenario: Optional[str]) -> list[Sample]:
    if scenario:
        return generate_scenario(SCENARIOS[scenario], add_noise_artifacts=(scenario == "noisy"))
    return load_csv(Path(csv_path))


def describe(step: Step, offset_s: float) -> str:
    r = step.reading
    hr = "  --" if r.hr is None else f"{r.hr:5.1f}"
    dev = "  --" if r.deviation is None else f"{r.deviation:4.1f}"
    base = " --" if r.baseline_hr is None else f"{r.baseline_hr:3.0f}"
    line = (
        f"{offset_s:6.0f}s hr {hr} q {r.quality:.2f} {r.signal_status:7} {str(r.activity):7} "
        f"base {base} dev {dev} persist {r.persist_s:4.0f}s rec {r.recovery_verdict or '-'}"
    )
    if step.alert is not None:
        line += f"\n        -> ALERT {step.alert.level.upper()}: {step.alert.headline} | {step.alert.voice_text}"
    return line


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI pipeline on a recording or a synthetic scenario.")
    parser.add_argument("csv", nargs="?", help="Contract A CSV to replay")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), help="Generate a synthetic scenario instead")
    parser.add_argument("--speed", "--speedup", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--base-url", default=os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--dry-run", action="store_true", help="Print Readings and Alerts; don't post")
    parser.add_argument("--no-llm", action="store_true", help="Use templated wording instead of Gemini")
    parser.add_argument("--keep-time", action="store_true", help="Post sample timestamps instead of wall-clock time")
    args = parser.parse_args()
    if not args.csv and not args.scenario:
        parser.error("give a CSV path or --scenario")
    if args.speed <= 0:
        parser.error("--speed must be > 0")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.dry_run:
        context = offline_context()
    else:
        try:
            context = fetch_context(args.base_url)
        except requests.ConnectionError:
            parser.exit(1, f"Can't reach the backend at {args.base_url}. Is uvicorn running? (or use --dry-run)\n")
    explainer = Explainer(use_llm=not args.no_llm)
    print(f"Patient {context.patient_id} - wording: {'Gemini ' + explainer.model if explainer.uses_llm else 'templates'}")

    samples = load_samples(args.csv, args.scenario)
    pipeline = Pipeline(context, explainer)
    t0 = samples[0].t if samples else 0
    paced = iter_samples(samples, speed=args.speed) if not args.dry_run else iter(samples)
    try:
        for step in pipeline.run(paced):
            print(describe(step, (step.reading.t - t0) / 1000.0), flush=True)
            if args.dry_run:
                continue
            reading, alert = step.reading, step.alert
            if not args.keep_time:  # wall-clock time so the dashboard's freshness checks work
                now = int(time.time() * 1000)
                reading = reading.model_copy(update={"t": now})
                alert = None if alert is None else alert.model_copy(update={"t": now, "reading": reading})
            post_model(args.base_url, "/readings", reading)
            if alert is not None:
                post_model(args.base_url, "/alerts", alert)
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
