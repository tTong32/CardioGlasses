"""Synthetic PPG + IMU generator and CSV player for Contract A."""

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from ai.contracts import Sample
import ai.config as cfg

COLUMNS = ["t", "ppg", "ax", "ay", "az", "gx", "gy", "gz"]


# ============================================================================
# Pulse shape synthesis
# ============================================================================


def _pulse_shape(phase: float) -> float:
    """
    Generate a single cardiac pulse using sum of two Gaussians.
    phase in [0, 1] represents one beat cycle.
    Returns normalized amplitude (0–1).
    """
    # Primary systolic peak: fast rise, positioned at phase ~0.3
    systolic = np.exp(-((phase - 0.3) ** 2) / (2 * 0.04**2))

    # Dicrotic notch: smaller, later bump at phase ~0.5
    dicrotic = 0.25 * np.exp(-((phase - 0.5) ** 2) / (2 * 0.06**2))

    return systolic + dicrotic


def _generate_beat_times(
    duration_s: float,
    hr_curve_fn,
    seed: int,
) -> list[float]:
    """
    Generate beat times with HRV jitter.
    hr_curve_fn(t_s) returns the instantaneous HR in bpm.
    Returns list of beat times in seconds.
    """
    rng = random.Random(seed)
    beats = []
    t = 0.0

    while t < duration_s:
        hr_bpm = hr_curve_fn(t)
        mean_ibi_s = 60.0 / hr_bpm
        # Add 3% jitter
        jitter = rng.gauss(1.0, cfg.HRV_JITTER_PERCENT / 100.0)
        ibi_s = mean_ibi_s * jitter
        t += ibi_s
        if t < duration_s:
            beats.append(t)

    return beats


def _generate_ppg_signal(
    duration_s: float,
    beat_times: list[float],
    rng: random.Random,
    drift_seed: int,
    motion_fn=None,
) -> np.ndarray:
    """
    Generate synthetic PPG at 50 Hz.
    motion_fn(t_s) returns motion artifact amplitude (0–1).
    Returns array of PPG values in sensor counts.
    """
    rate = cfg.SAMPLE_RATE_HZ
    n_samples = int(duration_s * rate)
    t_samples = np.arange(n_samples) / rate

    # Initialize with DC level
    ppg = np.full(n_samples, cfg.PPG_DC_LEVEL, dtype=float)

    # Add slow baseline drift
    drift_rng = np.random.default_rng(drift_seed)
    drift_freq = rng.uniform(0.1, 0.3)
    drift_phase = rng.uniform(0, 2 * np.pi)
    drift = cfg.PPG_DRIFT_AMPLITUDE * np.sin(2 * np.pi * drift_freq * t_samples + drift_phase)
    ppg += drift

    # Add cardiac pulses
    if beat_times:
        for i, t_sample in enumerate(t_samples):
            # Find surrounding beats
            phase = 0.0
            for j, beat_t in enumerate(beat_times):
                if beat_t > t_sample:
                    if j > 0:
                        prev_beat = beat_times[j - 1]
                        next_beat = beat_t
                        ibi = next_beat - prev_beat
                        phase = (t_sample - prev_beat) / ibi
                    break
            else:
                # After last beat
                if beat_times:
                    last_ibi = beat_times[-1] - beat_times[-2] if len(beat_times) > 1 else 60.0 / 70.0
                    phase = (t_sample - beat_times[-1]) / last_ibi

            if 0 <= phase <= 1:
                ppg[i] += cfg.PPG_PULSE_AMPLITUDE * _pulse_shape(phase)

    # Add motion artifacts if provided
    if motion_fn is not None:
        for i, t_sample in enumerate(t_samples):
            motion_level = motion_fn(t_sample)
            if motion_level > 0:
                artifact = motion_level * cfg.MOTION_ARTIFACT_AMPLITUDE * np.sin(
                    2 * np.pi * cfg.MOTION_ARTIFACT_FREQ_HZ * t_sample + rng.uniform(0, 2 * np.pi)
                )
                ppg[i] += artifact

    # Add Gaussian noise
    noise = drift_rng.normal(0, cfg.PPG_NOISE_STD, n_samples)
    ppg += noise

    # Clip to sensor range
    ppg = np.clip(ppg, cfg.PPG_SENSOR_MIN, cfg.PPG_SENSOR_MAX)

    return ppg


def _generate_imu_signal(
    duration_s: float,
    rng: random.Random,
    motion_fn=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic IMU at 50 Hz.
    motion_fn(t_s) returns motion level (0–1).
    Returns (ax, ay, az, gx, gy, gz) arrays.
    """
    rate = cfg.SAMPLE_RATE_HZ
    n_samples = int(duration_s * rate)
    t_samples = np.arange(n_samples) / rate

    # Initialize with rest values
    ax = np.zeros(n_samples)
    ay = np.full(n_samples, cfg.IMU_GRAVITY_Y)
    az = np.zeros(n_samples)
    gx = np.zeros(n_samples)
    gy = np.zeros(n_samples)
    gz = np.zeros(n_samples)

    # Add noise
    for arr in [ax, ay, az]:
        arr += rng.gauss(0, cfg.IMU_REST_ACCEL_NOISE) * np.random.default_rng(rng.randint(0, 1e9)).normal(0, 1, n_samples)

    for arr in [gx, gy, gz]:
        arr += rng.gauss(0, cfg.IMU_REST_GYRO_NOISE) * np.random.default_rng(rng.randint(0, 1e9)).normal(0, 1, n_samples)

    # Add movement
    if motion_fn is not None:
        for i, t_sample in enumerate(t_samples):
            motion_level = motion_fn(t_sample)
            if motion_level > 0:
                # Sinusoidal motion on all axes
                phase = 2 * np.pi * cfg.IMU_MOVING_FREQ_HZ * t_sample
                ax[i] += motion_level * cfg.IMU_MOVING_ACCEL_AMPLITUDE * np.sin(phase)
                ay[i] += motion_level * cfg.IMU_MOVING_ACCEL_AMPLITUDE * np.sin(phase + 0.5)
                az[i] += motion_level * cfg.IMU_MOVING_ACCEL_AMPLITUDE * np.sin(phase + 1.0)

                gx[i] += motion_level * cfg.IMU_MOVING_GYRO_AMPLITUDE * np.sin(phase + 1.5)
                gy[i] += motion_level * cfg.IMU_MOVING_GYRO_AMPLITUDE * np.sin(phase + 2.0)
                gz[i] += motion_level * cfg.IMU_MOVING_GYRO_AMPLITUDE * np.sin(phase + 2.5)

    return ax, ay, az, gx, gy, gz


# ============================================================================
# Scenario definitions
# ============================================================================


class Scenario:
    """A named test scenario with HR and motion curves."""

    def __init__(self, name: str, duration_s: float, hr_fn, motion_fn=None, seed: int = 42):
        self.name = name
        self.duration_s = duration_s
        self.hr_fn = hr_fn
        self.motion_fn = motion_fn or (lambda t: 0.0)
        self.seed = seed


def _constant_hr(bpm: float):
    return lambda t: bpm


def _linear_ramp(t_start: float, t_end: float, hr_start: float, hr_end: float):
    def fn(t):
        if t < t_start:
            return hr_start
        elif t > t_end:
            return hr_end
        else:
            progress = (t - t_start) / (t_end - t_start)
            return hr_start + progress * (hr_end - hr_start)
    return fn


def _exponential_recovery(t_start: float, hr_start: float, hr_baseline: float, tau_s: float):
    def fn(t):
        if t < t_start:
            return hr_start
        else:
            dt = t - t_start
            return hr_baseline + (hr_start - hr_baseline) * np.exp(-dt / tau_s)
    return fn


def _piecewise(*functions):
    """Combine multiple (t_end, fn) pairs."""
    def combined(t):
        for t_boundary, fn in functions:
            if t < t_boundary:
                return fn(t)
        return functions[-1][1](t)
    return combined


def _motion_window(t_start: float, t_end: float):
    return lambda t: 1.0 if t_start <= t <= t_end else 0.0


# Define all scenarios
SCENARIOS = {
    "rest": Scenario(
        name="rest",
        duration_s=180,
        hr_fn=_constant_hr(68),
        seed=1,
    ),

    "moving": Scenario(
        name="moving",
        duration_s=210,
        hr_fn=_piecewise(
            (60, _constant_hr(68)),
            (150, _linear_ramp(60, 150, 68, 105)),
            (210, _exponential_recovery(150, 105, 68, 30)),
        ),
        motion_fn=_motion_window(60, 150),
        seed=2,
    ),

    "normal_recovery": Scenario(
        name="normal_recovery",
        duration_s=330,
        hr_fn=_piecewise(
            (90, _constant_hr(68)),
            (150, _linear_ramp(90, 150, 68, 110)),
            (330, _exponential_recovery(150, 110, 68, 30)),
        ),
        motion_fn=_motion_window(90, 150),
        seed=3,
    ),

    "slow_recovery": Scenario(
        name="slow_recovery",
        duration_s=330,
        hr_fn=_piecewise(
            (90, _constant_hr(68)),
            (150, _linear_ramp(90, 150, 68, 110)),
            (330, _exponential_recovery(150, 110, 68, 150)),
        ),
        motion_fn=_motion_window(90, 150),
        seed=4,
    ),

    "noisy": Scenario(
        name="noisy",
        duration_s=120,
        hr_fn=_constant_hr(68),
        seed=5,
    ),

    "calibration_then_slow": Scenario(
        name="calibration_then_slow",
        duration_s=510,
        hr_fn=_piecewise(
            (90, _constant_hr(68)),
            (150, _linear_ramp(90, 150, 68, 105)),
            (270, _exponential_recovery(150, 105, 68, 30)),
            (330, _constant_hr(68)),
            (390, _linear_ramp(330, 390, 68, 110)),
            (510, _exponential_recovery(390, 110, 68, 150)),
        ),
        motion_fn=lambda t: 1.0 if (90 <= t <= 150) or (330 <= t <= 390) else 0.0,
        seed=6,
    ),
}


def generate_scenario(scenario: Scenario, add_noise_artifacts: bool = False) -> list[Sample]:
    """
    Generate synthetic samples for a scenario.
    If add_noise_artifacts=True (for 'noisy'), add heavy noise, dropouts, clipping.
    """
    rng = random.Random(scenario.seed)

    # Generate beat times
    beat_times = _generate_beat_times(scenario.duration_s, scenario.hr_fn, scenario.seed)

    # Generate PPG
    ppg = _generate_ppg_signal(
        scenario.duration_s,
        beat_times,
        rng,
        drift_seed=scenario.seed,
        motion_fn=scenario.motion_fn,
    )

    # Generate IMU
    ax, ay, az, gx, gy, gz = _generate_imu_signal(
        scenario.duration_s,
        rng,
        motion_fn=scenario.motion_fn,
    )

    # Add heavy artifacts for noisy scenario
    if add_noise_artifacts:
        n_samples = len(ppg)
        noise_rng = np.random.default_rng(scenario.seed + 100)

        # Heavy noise
        ppg += noise_rng.normal(0, 300, n_samples)

        # Random dropouts (flat at DC level)
        dropout_prob = 0.05
        dropout_mask = noise_rng.random(n_samples) < dropout_prob
        ppg[dropout_mask] = cfg.PPG_DC_LEVEL

        # Occasional clipping
        clip_prob = 0.03
        clip_high = noise_rng.random(n_samples) < clip_prob
        ppg[clip_high] = cfg.PPG_SENSOR_MAX

        ppg = np.clip(ppg, cfg.PPG_SENSOR_MIN, cfg.PPG_SENSOR_MAX)

    # Build samples
    rate = cfg.SAMPLE_RATE_HZ
    n_samples = int(scenario.duration_s * rate)
    t0_ms = 1_760_000_000_000

    samples = []
    for i in range(n_samples):
        samples.append(
            Sample(
                t=t0_ms + int(i * 1000 / rate),
                ppg=int(round(ppg[i])),
                ax=round(ax[i], 4),
                ay=round(ay[i], 4),
                az=round(az[i], 4),
                gx=round(gx[i], 4),
                gy=round(gy[i], 4),
                gz=round(gz[i], 4),
            )
        )

    return samples


# ============================================================================
# CSV I/O
# ============================================================================


def save_csv(samples: list[Sample], path: Path) -> None:
    """Write samples to a Contract A CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for s in samples:
            writer.writerow([
                s.t,
                s.ppg,
                s.ax,
                s.ay,
                s.az,
                s.gx,
                s.gy,
                s.gz,
            ])


def load_csv(path: Path) -> list[Sample]:
    """Read a Contract A CSV. Blank or 'null' cells become None."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        names = [name.strip() for name in (reader.fieldnames or [])]
        if names != COLUMNS:
            raise ValueError(f"{path} must have columns: {','.join(COLUMNS)}")

        samples = []
        for row in reader:
            data = {}
            for key in COLUMNS:
                val = (row.get(key) or "").strip()
                if val == "" or val.lower() == "null":
                    data[key] = None
                elif key == "t":
                    data[key] = int(val)
                elif key == "ppg":
                    data[key] = int(float(val))
                else:
                    data[key] = float(val)

            # Drop bad samples silently
            if data["t"] is None:
                continue
            if data["ppg"] is not None and (math.isnan(data["ppg"]) or abs(data["ppg"]) > 1e9):
                continue

            samples.append(Sample.model_validate(data))

        # Sort by time and drop duplicates/backwards
        samples.sort(key=lambda s: s.t)
        seen = set()
        filtered = []
        for s in samples:
            if s.t not in seen:
                seen.add(s.t)
                filtered.append(s)

        return filtered


def iter_samples(samples: list[Sample], speed: float = 1.0) -> Iterator[Sample]:
    """Yield samples with real-time pacing at speed × real time."""
    if not samples:
        return

    origin = samples[0].t
    started = time.perf_counter()

    for sample in samples:
        elapsed_ms = sample.t - origin
        target_s = elapsed_ms / 1000.0 / speed
        actual_s = time.perf_counter() - started
        delay = target_s - actual_s

        if delay > 0:
            time.sleep(delay)

        yield sample


# ============================================================================
# CLI
# ============================================================================


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate or replay Contract A data.")
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        help="Generate a scenario",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default 1.0 = real time)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Save generated scenario to CSV",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Replay a CSV file",
    )
    parser.add_argument(
        "--serial-format",
        action="store_true",
        help="Output one JSON line per sample (matches hardware)",
    )

    args = parser.parse_args(argv)

    if args.speed <= 0:
        parser.error("--speed must be > 0")

    samples = None

    # Generate scenario
    if args.scenario:
        scenario = SCENARIOS[args.scenario]
        add_noise = (args.scenario == "noisy")
        samples = generate_scenario(scenario, add_noise_artifacts=add_noise)

        if args.out:
            save_csv(samples, args.out)
            print(f"Saved {len(samples)} samples to {args.out}", file=sys.stderr)

    # Load CSV
    elif args.csv:
        samples = load_csv(args.csv)
        print(f"Loaded {len(samples)} samples from {args.csv}", file=sys.stderr)

    else:
        parser.error("Provide --scenario or --csv")

    # Output samples
    if samples:
        if args.serial_format:
            for sample in iter_samples(samples, speed=args.speed):
                print(sample.model_dump_json(), flush=True)
        else:
            # Just dump all samples immediately for testing
            for sample in samples:
                print(sample.model_dump_json(), flush=True)


if __name__ == "__main__":
    main()
