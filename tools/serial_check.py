"""Check (and optionally record) the glasses' USB serial stream against Contract A.

    python -m tools.serial_check --list                          # show serial ports
    python -m tools.serial_check --port COM5                     # live check, Ctrl+C for summary
    python -m tools.serial_check --port COM5 --seconds 120 --out data/rec_rest.csv
    python -m tools.serial_check --from-file capture.txt         # check saved JSON lines

Reports the real sample rate, gaps, bad lines, accelerometer units (g vs m/s^2), and
whether the PPG looks alive. --out writes the Contract A CSV that replay reads.
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from ai.contracts import Sample

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

COLUMNS = ["t", "ppg", "ax", "ay", "az", "gx", "gy", "gz"]
EXPECTED_HZ = float(os.environ.get("SAMPLE_RATE_HZ") or 50)


@dataclass
class Checker:
    expected_hz: float = EXPECTED_HZ
    samples: list[Sample] = field(default_factory=list)
    bad_lines: int = 0
    bad_examples: list[str] = field(default_factory=list)
    gaps: int = 0
    out_of_order: int = 0
    nulls: int = 0

    def feed(self, line: str) -> Sample | None:
        """Validate one line. Returns the Sample, or None if the line is bad."""
        text = line.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
            missing = [key for key in COLUMNS if key not in payload]
            if missing:
                raise ValueError(f"missing keys {missing} (send null, don't omit)")
            sample = Sample.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            self.bad_lines += 1
            if len(self.bad_examples) < 3:
                self.bad_examples.append(f"{text[:80]!r}: {str(exc).splitlines()[0]}")
            return None
        if self.samples:
            dt = sample.t - self.samples[-1].t
            if dt <= 0:
                self.out_of_order += 1
            elif dt > 2.5 * 1000 / self.expected_hz:
                self.gaps += 1
        if any(getattr(sample, key) is None for key in COLUMNS[1:]):
            self.nulls += 1
        self.samples.append(sample)
        return sample

    def rate_hz(self) -> float | None:
        if len(self.samples) < 2:
            return None
        span_ms = self.samples[-1].t - self.samples[0].t
        return (len(self.samples) - 1) * 1000 / span_ms if span_ms > 0 else None

    def accel_units(self) -> str | None:
        """'g' if gravity reads ~1, 'm/s^2' if ~9.8. Contract A uses g."""
        mags = [
            math.sqrt(s.ax**2 + s.ay**2 + s.az**2)
            for s in self.samples
            if s.ax is not None and s.ay is not None and s.az is not None
        ]
        if len(mags) < 10:
            return None
        mean = statistics.fmean(mags)
        if 0.7 <= mean <= 1.3:
            return "g"
        if 7.0 <= mean <= 12.0:
            return "m/s^2"
        return f"unknown (|a| ~ {mean:.2f})"

    def ppg_spread(self) -> float | None:
        values = [s.ppg for s in self.samples[-250:] if s.ppg is not None]
        return statistics.pstdev(values) if len(values) >= 10 else None

    def problems(self) -> list[str]:
        issues = []
        total = len(self.samples) + self.bad_lines
        if total and self.bad_lines / total > 0.01:
            issues.append(f"{self.bad_lines}/{total} lines invalid (target < 1%)")
        rate = self.rate_hz()
        if rate is not None and abs(rate - self.expected_hz) / self.expected_hz > 0.1:
            issues.append(f"sample rate {rate:.1f} Hz, expected ~{self.expected_hz:.0f} Hz")
        if self.gaps:
            issues.append(f"{self.gaps} gaps longer than {2.5 * 1000 / self.expected_hz:.0f} ms")
        if self.out_of_order:
            issues.append(f"{self.out_of_order} timestamps not increasing")
        units = self.accel_units()
        if units and units != "g":
            issues.append(f"accelerometer looks like {units}; Contract A expects g (gravity ~ 1.0)")
        spread = self.ppg_spread()
        if spread is not None and spread < 10:
            issues.append(f"PPG nearly flat (std {spread:.1f}); is the sensor touching skin?")
        if self.samples and self.samples[0].t < 1_000_000_000_000:
            issues.append("t looks like device uptime, not Unix ms; the laptop may need to stamp time")
        return issues

    def status_line(self) -> str:
        rate = self.rate_hz()
        spread = self.ppg_spread()
        return (
            f"samples {len(self.samples):6}  rate {rate or 0:5.1f} Hz  bad {self.bad_lines}  gaps {self.gaps}  "
            f"accel {self.accel_units() or '?'}  ppg std {spread if spread is not None else 0:.0f}"
        )

    def summary(self) -> str:
        lines = ["", "Summary", "  " + self.status_line()]
        if self.nulls:
            lines.append(f"  {self.nulls} samples had null fields")
        for example in self.bad_examples:
            lines.append(f"  bad line: {example}")
        issues = self.problems()
        lines.append("  PASS: stream matches Contract A" if self.samples and not issues else "  Problems:")
        lines.extend(f"   - {issue}" for issue in issues)
        if not self.samples:
            lines.append("   - no valid samples received")
        return "\n".join(lines)


def write_csv(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for s in samples:
            writer.writerow(["" if getattr(s, key) is None else getattr(s, key) for key in COLUMNS])


def list_ports() -> None:
    from serial.tools import list_ports as lp

    ports = list(lp.comports())
    if not ports:
        print("No serial ports found. Is the board plugged in (and a data cable, not charge-only)?")
    for port in ports:
        print(f"{port.device:10} {port.description}")


def read_serial(checker: Checker, port: str, baud: int, seconds: float | None) -> None:
    import serial

    with serial.Serial(port, baud, timeout=1) as conn:
        conn.reset_input_buffer()
        conn.readline()  # drop a likely partial first line
        started = last_print = time.monotonic()
        while seconds is None or time.monotonic() - started < seconds:
            raw = conn.readline()
            if raw:
                checker.feed(raw.decode("utf-8", errors="replace"))
            if time.monotonic() - last_print >= 1:
                print(checker.status_line(), flush=True)
                last_print = time.monotonic()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the glasses' serial stream against Contract A.")
    parser.add_argument("--port", default=os.environ.get("SERIAL_PORT") or None)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seconds", type=float, help="Stop after this many seconds")
    parser.add_argument("--out", type=Path, help="Write valid samples to this Contract A CSV")
    parser.add_argument("--from-file", type=Path, help="Check a file of JSON lines instead of a port")
    parser.add_argument("--list", action="store_true", help="List serial ports and exit")
    args = parser.parse_args()

    if args.list:
        list_ports()
        return
    checker = Checker()
    try:
        if args.from_file:
            for line in args.from_file.read_text(encoding="utf-8").splitlines():
                checker.feed(line)
        elif args.port:
            print(f"Reading {args.port} at {args.baud} baud. Ctrl+C to stop.")
            read_serial(checker, args.port, args.baud, args.seconds)
        else:
            parser.error("give --port (or set SERIAL_PORT in .env), --from-file, or --list")
    except KeyboardInterrupt:
        pass
    print(checker.summary())
    if args.out and checker.samples:
        write_csv(checker.samples, args.out)
        print(f"wrote {len(checker.samples)} samples to {args.out}")
    sys.exit(0 if checker.samples and not checker.problems() else 1)


if __name__ == "__main__":
    main()
