"""Convert a raw ble_logger.py recording (hw/bridge/recordings/) into Contract A.

hw/bridge/ble_logger.py (the prototype logger) writes a session as:

    idx, t_dev_ms, t_host_ms, ir, red[, ax, ay, az, gx, gy, gz]

That's a device-clock timestamp plus IMU columns missing entirely on older,
PPG-only recordings. Contract A (CONTRACTS.md) is one JSON object per sample:

    {"t": <unix ms>, "ppg": <int>, "red": <int|null>, "ax": .., .., "gz": ..}

`ppg` is the raw IR count, unscaled (0-262143, 18-bit -- ai/config.py's
PPG_SENSOR_MIN/MAX matches that range). `red`, the optional second PPG channel
(the build tracker's own Contract A note left this open: "2nd PPG channel
optional"), rides along -- ai/contracts.py's Sample has it as Optional[int] = None,
so it's additive and doesn't break Contract A's original 8-key shape.

unknown/missing = null, never a dropped key. This writes that JSON-lines file
(the literal wire format -- what hw/bridge/live_bridge.py emits live, and what
tools/serial_check.py --from-file checks) *and* the Contract A CSV that
ai/replay.py / ai/pipeline.py read (ai.replay.load_csv has no JSON-line reader
yet, and its strict 8-column check means the CSV stays `red`-free).

Usage:
    python -m hw.bridge.convert_firmware_csv hw/bridge/recordings/ear_motion1.csv --out data/rec_rest
    python -m hw.bridge.convert_firmware_csv hw/bridge/recordings/imu_test.csv --out data/rec_moving --channel red

    python -m ai.pipeline data/rec_rest.csv --speed 5
    python -m tools.serial_check --from-file data/rec_rest.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

COLUMNS = ["t", "ppg", "ax", "ay", "az", "gx", "gy", "gz"]  # Contract A CSV, exact match required by ai.replay.load_csv
JSON_KEYS = ["t", "ppg", "red", "ax", "ay", "az", "gx", "gy", "gz"]  # + optional second channel
IMU_KEYS = ["ax", "ay", "az", "gx", "gy", "gz"]


def convert(path: Path, channel: str, origin_ms: int) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        has_imu = {"ax", "gx"}.issubset(fields)
        has_red = "red" in fields
        if channel not in fields:
            raise ValueError(f"{path} has no '{channel}' column (has: {sorted(fields)})")

        rows = []
        for row in reader:
            t_dev_ms = int(float(row["t_dev_ms"]))
            ppg_val = row.get(channel)
            if ppg_val is None or ppg_val.strip() == "":
                continue  # no PPG value for this row at all, skip
            rows.append({
                "t": origin_ms + t_dev_ms,
                "ppg": int(float(ppg_val)),  # raw IR counts, unscaled
                "red": int(float(row["red"])) if has_red and row.get("red") not in (None, "") else None,
                "ax": float(row["ax"]) if has_imu and row.get("ax") not in (None, "") else None,
                "ay": float(row["ay"]) if has_imu and row.get("ay") not in (None, "") else None,
                "az": float(row["az"]) if has_imu and row.get("az") not in (None, "") else None,
                "gx": float(row["gx"]) if has_imu and row.get("gx") not in (None, "") else None,
                "gy": float(row["gy"]) if has_imu and row.get("gy") not in (None, "") else None,
                "gz": float(row["gz"]) if has_imu and row.get("gz") not in (None, "") else None,
            })

        if not has_imu:
            print(f"Note: {path.name} has no IMU columns; writing null for {', '.join(IMU_KEYS)}.")
        return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    """Contract A CSV -- what ai/replay.py and ai/pipeline.py read."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for r in rows:
            writer.writerow([r[c] if r[c] is not None else "" for c in COLUMNS])


def write_jsonl(rows: list[dict], out_path: Path) -> None:
    """Contract A JSON lines -- the literal wire format hw/bridge/live_bridge.py emits live."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in JSON_KEYS}) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_in", type=Path, help="raw hw/bridge/ble_logger.py recording")
    ap.add_argument("--out", type=Path, required=True,
                     help="output path stem (e.g. data/rec_rest); writes <stem>.csv and <stem>.jsonl")
    ap.add_argument("--channel", choices=["ir", "red"], default="ir",
                     help="which PPG channel becomes 'ppg' (default: ir, matches plot_ppg.py)")
    ap.add_argument("--origin-ms", type=int, default=None,
                     help="Unix ms added to the device clock (default: now minus the recording's span)")
    args = ap.parse_args(argv)

    # Default origin: make the *last* sample land near "now", so a freshly
    # converted recording doesn't look stale, and 't' stays Unix ms per Contract A.
    if args.origin_ms is None:
        with args.csv_in.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            last_dev_ms = 0
            for row in reader:
                last_dev_ms = int(float(row["t_dev_ms"]))
        args.origin_ms = int(time.time() * 1000) - last_dev_ms

    rows = convert(args.csv_in, args.channel, args.origin_ms)
    if not rows:
        raise SystemExit(f"No usable rows found in {args.csv_in}")

    stem = args.out.with_suffix("")
    csv_path, jsonl_path = stem.with_suffix(".csv"), stem.with_suffix(".jsonl")
    write_csv(rows, csv_path)
    write_jsonl(rows, jsonl_path)

    span_s = (rows[-1]["t"] - rows[0]["t"]) / 1000
    print(f"Wrote {len(rows)} samples ({span_s:.1f}s) from {args.csv_in.name}:")
    print(f"  {csv_path}   (Contract A CSV, for ai.replay / ai.pipeline)")
    print(f"  {jsonl_path}  (Contract A JSON lines + 'red', for tools.serial_check --from-file / live_bridge parity)")


if __name__ == "__main__":
    main()
