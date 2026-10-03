"""Samples in, Reading out, POST /readings. Feature stubs still return null."""

import argparse
import os
from pathlib import Path
from typing import Optional, Union

import requests
from dotenv import load_dotenv

from ai.baseline import BaselineTracker
from ai.activity import ActivityDetector
from ai.contracts import Alert, PatientContext, Reading, Sample
from ai.decision import evaluate
from ai.processing import classify_activity, compute_hr, signal_quality
from ai.replay import iter_samples

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def build_reading(
    samples: list[Sample],
    activity_detector: ActivityDetector,
    baseline_tracker: BaselineTracker
) -> Reading:
    """Build a Contract B reading using live activity and baseline tracking."""
    # Process all samples through activity detector
    activity_output = None
    for sample in samples:
        if sample.ax is not None and sample.ay is not None and sample.az is not None:
            activity_output = activity_detector.update(sample.ax, sample.ay, sample.az)

    # Get HR estimate
    hr = compute_hr(samples)

    # Update baseline tracker with HR and activity
    baseline_output = None
    if activity_output is not None and hr is not None:
        baseline_output = baseline_tracker.update(
            samples[-1].t,
            hr,
            activity_output.activity
        )

    # Build reading
    activity = activity_output.activity if activity_output else classify_activity(samples)
    baseline_hr = baseline_output.baseline_hr if baseline_output else None
    deviation = (hr - baseline_hr) if (hr is not None and baseline_hr is not None) else None

    return Reading(
        t=samples[-1].t,
        hr=hr,
        ibi_ms=None,
        activity=activity,
        quality=signal_quality(samples),
        baseline_hr=baseline_hr,
        deviation=deviation,
        persist_s=None,
    )


def post_model(base_url: str, path: str, model: Union[Reading, Alert]) -> None:
    response = requests.post(
        f"{base_url.rstrip('/')}{path}",
        json=model.model_dump(mode="json"),
        timeout=5,
    )
    response.raise_for_status()


def fetch_context(base_url: str) -> PatientContext:
    response = requests.get(f"{base_url.rstrip('/')}/patient", timeout=5)
    response.raise_for_status()
    return PatientContext.model_validate(response.json())


def run(csv_path: str, base_url: str, speedup: float = 1.0) -> None:
    """Replay `csv_path` and POST a Reading about every 2 s of sample time."""
    context = fetch_context(base_url)

    # Initialize stateful detectors
    activity_detector = ActivityDetector()
    baseline_tracker = BaselineTracker()

    batch: list[Sample] = []
    emit_at: Optional[int] = None
    for sample in iter_samples(csv_path, speedup=speedup):
        batch.append(sample)
        if emit_at is None:
            emit_at = sample.t + 2000
        if sample.t < emit_at:
            continue
        reading = build_reading(batch, activity_detector, baseline_tracker)
        post_model(base_url, "/readings", reading)
        alert = evaluate(reading, context)
        if alert is not None:
            post_model(base_url, "/alerts", alert)
        batch = []
        emit_at = sample.t + 2000


def main() -> None:
    parser = argparse.ArgumentParser(description="POST replayed readings to the backend.")
    parser.add_argument("csv", nargs="?", default=str(ROOT / "data" / "sample_rest.csv"))
    parser.add_argument("--speedup", type=float, default=1.0)
    parser.add_argument("--base-url", default=os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"))
    args = parser.parse_args()
    run(args.csv, base_url=args.base_url, speedup=args.speedup)


if __name__ == "__main__":
    main()
