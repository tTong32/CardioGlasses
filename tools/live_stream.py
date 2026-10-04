"""Live hardware stream: read serial → AI pipeline → backend.

Connect the glasses via USB, set SERIAL_PORT in .env, then:

    python -m tools.live_stream                    # start streaming
    python -m tools.live_stream --port COM5        # override port
    python -m tools.live_stream --no-llm           # template wording (faster, no API key)
    python -m tools.live_stream --dry-run          # print only, don't POST

The serial port must send Contract A JSON lines (one sample per line) at the configured
sample rate (default 50 Hz). The pipeline emits Readings every 2 seconds and POSTs them
to the backend, along with any Alerts the decision engine generates.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import requests
import serial
from dotenv import load_dotenv
from pydantic import ValidationError

from ai.contracts import PatientContext, Sample
from ai.explainer import Explainer
from ai.pipeline import CONTEXT_REFRESH_S, Pipeline, describe, fetch_context, offline_context, post_model

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

log = logging.getLogger(__name__)


def read_serial_loop(
    pipeline: Pipeline,
    port: str,
    baud: int,
    base_url: str,
    dry_run: bool,
    explainer: Explainer,
) -> None:
    """Read serial port, feed pipeline, POST to backend."""
    with serial.Serial(port, baud, timeout=1) as conn:
        conn.reset_input_buffer()
        conn.readline()  # drop likely partial first line

        t0_ms: int | None = None
        bad_lines = 0
        last_context_refresh = time.monotonic()

        print(f"Streaming from {port} at {baud} baud. Ctrl+C to stop.\n", flush=True)

        while True:
            raw = conn.readline()
            if not raw:
                continue

            # Parse JSON line → Sample
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
                sample = Sample.model_validate(payload)
            except (ValueError, ValidationError) as exc:
                bad_lines += 1
                if bad_lines <= 3:
                    log.warning("Bad line: %s | %s", line[:80], str(exc).splitlines()[0])
                continue

            # Feed to pipeline
            step = pipeline.push(sample)
            if step is None:
                continue

            # Track offset for display
            if t0_ms is None:
                t0_ms = step.reading.t
            offset_s = (step.reading.t - t0_ms) / 1000.0

            print(describe(step, offset_s), flush=True)

            if dry_run:
                continue

            # Refresh patient context every 30s
            if time.monotonic() - last_context_refresh >= CONTEXT_REFRESH_S:
                last_context_refresh = time.monotonic()
                try:
                    fresh = fetch_context(base_url)
                    if fresh != pipeline.context:
                        pipeline.update_context(fresh)
                        print(f"        (patient record updated: {fresh.patient_id}, {fresh.risk_tier} risk)", flush=True)
                except requests.RequestException:
                    pass  # keep current context

            # POST to backend with wall-clock time
            reading = step.reading.model_copy(update={"t": int(time.time() * 1000)})
            alert = step.alert
            if alert is not None:
                alert = alert.model_copy(update={"t": reading.t, "reading": reading})

            try:
                post_model(base_url, "/readings", reading)
                if alert is not None:
                    post_model(base_url, "/alerts", alert)
            except requests.RequestException as exc:
                log.error("POST failed: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream live hardware data through AI pipeline to backend.")
    parser.add_argument("--port", default=os.environ.get("SERIAL_PORT"), help="Serial port (or set SERIAL_PORT in .env)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--base-url", default=os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--dry-run", action="store_true", help="Print only, don't POST to backend")
    parser.add_argument("--no-llm", action="store_true", help="Use template wording (no Gemini API key needed)")
    args = parser.parse_args()

    if not args.port:
        parser.error("Serial port required: --port COM5 or set SERIAL_PORT in .env")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    # Fetch patient context
    if args.dry_run:
        context = offline_context()
    else:
        try:
            context = fetch_context(args.base_url)
        except requests.ConnectionError:
            parser.exit(1, f"Can't reach backend at {args.base_url}. Is uvicorn running? (or use --dry-run)\n")

    explainer = Explainer(use_llm=not args.no_llm)
    print(f"Patient: {context.patient_id} ({context.risk_tier} risk)")
    print(f"Wording: {'Gemini ' + explainer.model if explainer.uses_llm else 'templates'}")

    pipeline = Pipeline(context, explainer)

    try:
        read_serial_loop(pipeline, args.port, args.baud, args.base_url, args.dry_run, explainer)
    except KeyboardInterrupt:
        print("\nstopped")
    except serial.SerialException as exc:
        parser.exit(1, f"Serial error: {exc}\n")


if __name__ == "__main__":
    main()
