"""Receive CardioGlasses BLE stream (PPG + IMU) and log to CSV.

Usage:  python ble_logger.py --seconds 60 --out ear_still.csv
CSV columns: idx, t_dev_ms, t_host_ms, ir, red, ax, ay, az (g), gx, gy, gz (deg/s)
"""
import argparse
import asyncio
import csv
import struct
import time

from bleak import BleakClient, BleakScanner

DEVICE_NAME = "CardioGlasses"
DATA_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
FS = 100                         # Hz, must match firmware
SAMPLE = struct.Struct("<III6h")  # idx, ir, red, ax, ay, az, gx, gy, gz
ACC_LSB = 8192.0                 # +-4 g
GYR_LSB = 65.5                   # +-500 dps


async def main(seconds: float, out_path: str) -> None:
    print(f"Scanning for {DEVICE_NAME}...")
    dev = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=15)
    if dev is None:
        raise SystemExit("Device not found. Is it powered and advertising?")

    rows = []
    stats = {"last_idx": None, "dropped": 0, "bad_packets": 0}

    def on_notify(_, data: bytearray) -> None:
        if len(data) % SAMPLE.size:
            stats["bad_packets"] += 1      # truncated packet (MTU too small)
        host_ms = int(time.time() * 1000)
        for off in range(0, len(data) - SAMPLE.size + 1, SAMPLE.size):
            idx, ir, red, ax, ay, az, gx, gy, gz = SAMPLE.unpack_from(data, off)
            last = stats["last_idx"]
            if last is not None and idx != last + 1:
                stats["dropped"] += max(0, idx - last - 1)
            stats["last_idx"] = idx
            rows.append((idx, idx * 1000 // FS, host_ms, ir, red,
                         round(ax / ACC_LSB, 4), round(ay / ACC_LSB, 4),
                         round(az / ACC_LSB, 4), round(gx / GYR_LSB, 2),
                         round(gy / GYR_LSB, 2), round(gz / GYR_LSB, 2)))

    async with BleakClient(dev) as client:
        print(f"Connected to {dev.address}. Recording {seconds:.0f} s...")
        await client.start_notify(DATA_UUID, on_notify)
        start = time.time()
        while time.time() - start < seconds:
            await asyncio.sleep(1)
            el = time.time() - start
            last = rows[-1] if rows else None
            imu = (f"  acc=({last[5]:+.2f},{last[6]:+.2f},{last[7]:+.2f})g"
                   if last else "")
            print(f"  {el:4.0f}s  samples={len(rows):6d}  "
                  f"rate={len(rows)/el:5.1f} Hz  dropped={stats['dropped']}"
                  f"  bad={stats['bad_packets']}{imu}")
        await client.stop_notify(DATA_UUID)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "t_dev_ms", "t_host_ms", "ir", "red",
                    "ax", "ay", "az", "gx", "gy", "gz"])
        w.writerows(rows)
    print(f"Saved {len(rows)} samples to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--out", default="recording.csv")
    args = ap.parse_args()
    asyncio.run(main(args.seconds, args.out))