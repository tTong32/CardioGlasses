"""Live CardioGlasses BLE bridge: glasses -> AI pipeline -> backend (and Contract A JSON).

    python -m hw.bridge.live_bridge                       # stream, feed the pipeline, POST to backend
    python -m hw.bridge.live_bridge --dry-run              # stream + print, don't POST
    python -m hw.bridge.live_bridge --no-llm               # template wording (no Gemini key needed)
    python -m hw.bridge.live_bridge --seconds 300          # stop after 5 min
    python -m hw.bridge.live_bridge --out data/live_rest.jsonl

What this does, per hw/README.md's pipeline spec:
  1. Connects to the "CardioGlasses" BLE peripheral and subscribes to its notify
     characteristic (connecting is what wakes the sensors; see hw/firmware/src/main.cpp).
  2. Unpacks each 24-byte record (<III6h>: idx, ir, red, ax, ay, az, gx, gy, gz),
     4 per notification.
  3. Converts units: accel raw/8192 -> g, gyro raw/65.5 -> deg/s.
  4. Timestamps t = (wall clock at the FIRST packet) + idx * 10. Never the per-packet
     arrival time -- BLE delivers in bursts, which would corrupt beat intervals.
  5. Feeds each Sample to ai.pipeline.Pipeline (same one tools/live_stream.py uses for a
     literal serial Contract A stream) and POSTs any Reading/Alert to the backend, exactly
     like tools/live_stream.py does -- this is that script's BLE equivalent.
  6. Also writes one Contract A JSON object per sample to --out, so every run is a
     replayable recording (A-02), and to stdout when --dry-run (so it stays pipeable
     into whatever the eventual delivery method turns out to be -- see hw/README.md).
  7. `ppg` is the raw IR count, unscaled (0-262143, 18-bit -- see ai/config.py's
     PPG_SENSOR_MAX). `red` rides along as the optional second channel.

A status line (samples, rate, dropped, bad, latest accel) prints to stderr once a
second so it doesn't pollute --dry-run's JSON on stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import struct
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, welch

import requests
from bleak import BleakClient, BleakScanner
from dotenv import load_dotenv

from ai.contracts import Reading, Sample
from ai.explainer import Explainer
from ai.pipeline import CONTEXT_REFRESH_S, Pipeline, describe, fetch_context, offline_context, post_model

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

log = logging.getLogger(__name__)

DEVICE_NAME = "CardioGlasses"
DATA_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
SAMPLE = struct.Struct("<III6h")  # idx, ir, red, ax, ay, az, gx, gy, gz
ACC_LSB = 8192.0   # +-4 g
GYR_LSB = 65.5     # +-500 deg/s
SAMPLE_PERIOD_MS = 10  # firmware's own 100 Hz clock (ai.config.SAMPLE_RATE_HZ is 50; see hw/README.md)
BPM_EVERY_S = 2        # the console prints a BPM number this often, always
BPM_WINDOW_S = 10      # how much recent IR signal each BPM number is computed from


def bpm_reading(hr: float) -> Reading:
    """A Contract B Reading carrying only the continuous BPM. Everything else is null."""
    return Reading(
        t=int(time.time() * 1000), hr=round(hr, 1), ibi_ms=None, activity=None, quality=None,
        baseline_hr=None, deviation=None, persist_s=None, recovery_tau_s=None, hr_drop_60s=None,
        recovery_ratio=None, recovery_percentile=None, recovery_verdict=None, signal_status=None,
    )


def raw_bpm(ir_window) -> Optional[float]:
    """Heart rate from the dominant pulse frequency in the IR window. No quality gate:
    always a number once there's enough data. Expect it to wander when the signal is poor."""
    x = np.asarray(ir_window, dtype=float)
    if len(x) < 6 * (1000 // SAMPLE_PERIOD_MS):
        return None
    x = x - x.mean()
    b, a = butter(3, [0.7, 3.0], btype="band", fs=1000 / SAMPLE_PERIOD_MS)
    y = filtfilt(b, a, x)
    f, P = welch(y, fs=1000 / SAMPLE_PERIOD_MS, nperseg=min(len(y), 800), nfft=2**14)
    band = (f >= 0.7) & (f <= 3.0)
    return float(60 * f[band][np.argmax(P[band])])


def default_out_path() -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    return ROOT / "data" / f"live_{stamp}.jsonl"


async def run(
    seconds: Optional[float],
    out_path: Path,
    pipeline: Optional[Pipeline],
    base_url: str,
    dry_run: bool,
) -> None:
    print(f"Scanning for {DEVICE_NAME}...", file=sys.stderr)
    dev = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=15)
    if dev is None:
        raise SystemExit("Device not found. Is it powered and advertising?")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = out_path.open("w", encoding="utf-8")

    state = {
        "first_packet_ms": None,  # anchor: wall clock at the first packet, per hw/README.md rule 6
        "last_idx": None,
        "n_samples": 0,
        "dropped": 0,
        "bad_packets": 0,
        "last_accel_g": None,
        "t0_display": None,
        "ir_window": deque(maxlen=BPM_WINDOW_S * (1000 // SAMPLE_PERIOD_MS)),
        "last_context_refresh": time.monotonic(),
    }

    def on_notify(_, data: bytearray) -> None:
        if len(data) % SAMPLE.size:
            state["bad_packets"] += 1  # truncated packet (MTU too small)
            return
        if state["first_packet_ms"] is None:
            state["first_packet_ms"] = int(time.time() * 1000)

        for off in range(0, len(data), SAMPLE.size):
            idx, ir, red, ax, ay, az, gx, gy, gz = SAMPLE.unpack_from(data, off)
            last = state["last_idx"]
            if last is not None and idx != last + 1:
                state["dropped"] += max(0, idx - last - 1)
            state["last_idx"] = idx
            state["n_samples"] += 1
            state["ir_window"].append(ir)

            accel_g = (round(ax / ACC_LSB, 4), round(ay / ACC_LSB, 4), round(az / ACC_LSB, 4))
            state["last_accel_g"] = accel_g

            sample_dict = {
                "t": state["first_packet_ms"] + idx * SAMPLE_PERIOD_MS,
                "ppg": ir,                 # raw IR counts, unscaled (0-262143)
                "red": red,                # optional second PPG channel
                "ax": accel_g[0], "ay": accel_g[1], "az": accel_g[2],
                "gx": round(gx / GYR_LSB, 2), "gy": round(gy / GYR_LSB, 2), "gz": round(gz / GYR_LSB, 2),
            }
            line = json.dumps(sample_dict)
            out_f.write(line + "\n")        # replay file for this run (A-02)
            if dry_run and pipeline is None:
                print(line, flush=True)     # bare Contract A stream, no pipeline attached

            if pipeline is None:
                continue

            step = pipeline.push(Sample.model_validate(sample_dict))
            if step is None:
                continue

            if state["t0_display"] is None:
                state["t0_display"] = step.reading.t
            offset_s = (step.reading.t - state["t0_display"]) / 1000.0
            print(describe(step, offset_s), flush=True)

            if dry_run:
                continue

            if time.monotonic() - state["last_context_refresh"] >= CONTEXT_REFRESH_S:
                state["last_context_refresh"] = time.monotonic()
                try:
                    fresh = fetch_context(base_url)
                    if fresh != pipeline.context:
                        pipeline.update_context(fresh)
                        print(f"        (patient record updated: {fresh.patient_id}, {fresh.risk_tier} risk)", flush=True)
                except requests.RequestException:
                    pass  # keep current context

            reading = step.reading.model_copy(update={"t": int(time.time() * 1000)})
            alert = step.alert
            if alert is not None:
                alert = alert.model_copy(update={"t": reading.t, "reading": reading})
            try:
                if alert is not None:
                    post_model(base_url, "/alerts", alert)
            except requests.RequestException as exc:
                log.error("POST failed: %s", exc)

    async with BleakClient(dev) as client:
        print(f"Connected to {dev.address}. Saving to {out_path}. Ctrl+C to stop.", file=sys.stderr)
        await client.start_notify(DATA_UUID, on_notify)
        start = time.time()
        try:
            while seconds is None or time.time() - start < seconds:
                await asyncio.sleep(BPM_EVERY_S)
                el = time.time() - start
                bpm_now = raw_bpm(state["ir_window"])
                bpm = "BPM  --  (collecting)" if bpm_now is None else f"BPM {bpm_now:5.1f}"
                if bpm_now is not None and not dry_run:
                    try:
                        post_model(base_url, "/readings", bpm_reading(bpm_now))
                    except requests.RequestException as exc:
                        log.error("POST failed: %s", exc)
                a = state["last_accel_g"]
                accel = f"  accel=({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f})g" if a else ""
                print(
                    f"  {el:4.0f}s  {bpm}  |  samples={state['n_samples']:6d}  "
                    f"rate={state['n_samples'] / el if el else 0:5.1f} Hz  "
                    f"dropped={state['dropped']}  bad={state['bad_packets']}{accel}",
                    file=sys.stderr,
                )
        except KeyboardInterrupt:
            pass
        finally:
            await client.stop_notify(DATA_UUID)

    out_f.close()
    print(
        f"Stopped. {state['n_samples']} samples, {state['dropped']} dropped, "
        f"{state['bad_packets']} bad packets. Saved to {out_path}",
        file=sys.stderr,
    )


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=None, help="stop after this many seconds (default: run until Ctrl+C)")
    ap.add_argument("--out", type=Path, default=None, help="replay JSON-lines file to save (default: data/live_<timestamp>.jsonl)")
    ap.add_argument("--base-url", default=os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--dry-run", action="store_true", help="Run the pipeline and print Readings/Alerts; don't POST to the backend")
    ap.add_argument("--no-pipeline", action="store_true", help="Just save/print the raw Contract A stream; skip HR/activity/decision entirely")
    ap.add_argument("--no-llm", action="store_true", help="Use template wording (no Gemini API key needed)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    pipeline = None
    if not args.no_pipeline:
        if args.dry_run:
            context = offline_context()
        else:
            try:
                context = fetch_context(args.base_url)
            except requests.ConnectionError:
                ap.exit(1, f"Can't reach backend at {args.base_url}. Is uvicorn running? (or use --dry-run / --no-pipeline)\n")
        explainer = Explainer(use_llm=not args.no_llm)
        print(f"Patient: {context.patient_id} ({context.risk_tier} risk)", file=sys.stderr)
        print(f"Wording: {'Gemini ' + explainer.model if explainer.uses_llm else 'templates'}", file=sys.stderr)
        pipeline = Pipeline(context, explainer, hardware=True)  # no irregular-rhythm scoring on real glasses

    asyncio.run(run(args.seconds, args.out or default_out_path(), pipeline, args.base_url, args.dry_run))


if __name__ == "__main__":
    main()
