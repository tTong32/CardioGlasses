"""Replay Contract A CSVs in real time, and write the two sample recordings."""

import argparse
import csv
import math
import random
import time
from pathlib import Path

from ai.contracts import Sample

COLUMNS = ["t", "ppg", "ax", "ay", "az", "gx", "gy", "gz"]
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def generate_recording(path: Path, bpm: float, seed: int, t0_ms: int = 1_760_000_000_000) -> None:
    """Write ~60 s of 50 Hz samples. PPG is a noisy sine; the IMU is near still."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rate = 50
    count = 60 * rate
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for i in range(count):
            seconds = i / rate
            ppg = 50000 + 2500 * math.sin(2 * math.pi * (bpm / 60) * seconds)
            ppg += rng.gauss(0, 80)
            writer.writerow(
                [
                    t0_ms + i * 20,
                    int(round(ppg)),
                    f"{rng.gauss(0.02, 0.005):.4f}",
                    f"{rng.gauss(-0.98, 0.005):.4f}",
                    f"{rng.gauss(0.10, 0.005):.4f}",
                    f"{rng.gauss(0.0, 0.02):.4f}",
                    f"{rng.gauss(0.0, 0.02):.4f}",
                    f"{rng.gauss(0.0, 0.02):.4f}",
                ]
            )


def write_sample_recordings() -> None:
    """Write data/sample_rest.csv (70 bpm) and data/sample_elevated.csv (100 bpm)."""
    generate_recording(DATA_DIR / "sample_rest.csv", bpm=70, seed=1)
    generate_recording(DATA_DIR / "sample_elevated.csv", bpm=100, seed=2)


def _cell(key: str, raw: str | None):
    text = (raw or "").strip()
    if text == "" or text.lower() == "null":
        return None
    if key == "t":
        return int(text)
    if key == "ppg":
        return int(float(text))
    return float(text)


def load_samples(path: str | Path) -> list[Sample]:
    """Read a Contract A CSV. Blank or 'null' cells become null."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names = [name.strip() for name in (reader.fieldnames or [])]
        if names != COLUMNS:
            raise ValueError(f"{path} must have columns: {','.join(COLUMNS)}")
        return [
            Sample.model_validate({key: _cell(key, row.get(key)) for key in COLUMNS})
            for row in reader
        ]


def iter_samples(path: str | Path, speedup: float = 1.0):
    """Yield samples, pacing sleeps so timestamps play at `speedup` × real time."""
    if speedup <= 0:
        raise ValueError("speedup must be > 0")
    samples = load_samples(path)
    if not samples:
        return
    origin = samples[0].t
    started = time.perf_counter()
    for sample in samples:
        delay = (sample.t - origin) / 1000 / speedup - (time.perf_counter() - started)
        if delay > 0:
            time.sleep(delay)
        yield sample


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay a Contract A CSV.")
    parser.add_argument("csv", nargs="?", help="CSV path")
    parser.add_argument("--speedup", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Write data/sample_rest.csv and data/sample_elevated.csv",
    )
    args = parser.parse_args(argv)
    if args.generate:
        write_sample_recordings()
        print(f"wrote {DATA_DIR / 'sample_rest.csv'}")
        print(f"wrote {DATA_DIR / 'sample_elevated.csv'}")
    if args.csv:
        if args.speedup <= 0:
            parser.error("--speedup must be > 0")
        for sample in iter_samples(args.csv, speedup=args.speedup):
            print(sample.model_dump_json(), flush=True)
    elif not args.generate:
        parser.error("provide a CSV path or --generate")


if __name__ == "__main__":
    main()
