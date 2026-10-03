"""Samples in, Reading out, POST /readings. Feature stubs still return null."""

import argparse
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from ai.baseline import resting_hr_stats
from ai.contracts import Alert, PatientContext, Reading, Sample
from ai.decision import evaluate
from ai.processing import classify_activity, compute_hr, signal_quality
from ai.replay import iter_samples

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def build_reading(samples: list[Sample]) -> Reading:
    """Build a Contract B reading. TODO: fill hr, activity, quality, and deviation."""
    hr = compute_hr(samples)
    mean_hr, _sd = resting_hr_stats([hr] if hr is not None else [])
    return Reading(
        t=samples[-1].t,
        hr=hr,
        ibi_ms=None,
        activity=classify_activity(samples),
        quality=signal_quality(samples),
        baseline_hr=mean_hr,
        deviation=None,
        persist_s=None,
    )


def post_model(base_url: str, path: str, model: Reading | Alert) -> None:
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
    batch: list[Sample] = []
    emit_at: int | None = None
    for sample in iter_samples(csv_path, speedup=speedup):
        batch.append(sample)
        if emit_at is None:
            emit_at = sample.t + 2000
        if sample.t < emit_at:
            continue
        reading = build_reading(batch)
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
