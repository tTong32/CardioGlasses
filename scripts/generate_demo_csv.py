#!/usr/bin/env python3
"""Generate realistic demo CSV with proper HR variability and recovery phase."""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent


def generate_realistic_ppg(duration_s: float, target_hr, rate_hz: int = 100) -> np.ndarray:
    """Generate PPG signal with realistic pulse waveforms and beat-to-beat variability.

    Args:
        duration_s: Duration in seconds
        target_hr: Target heart rate in BPM (can be float or array for varying HR)
        rate_hz: Sampling rate in Hz

    Returns:
        PPG signal array with realistic cardiac waveforms
    """
    num_samples = int(duration_s * rate_hz)

    # If target_hr is a single value, make it constant
    if isinstance(target_hr, (int, float)):
        hr_over_time = np.full(num_samples, target_hr)
    else:
        hr_over_time = target_hr

    # Generate beat times with HRV
    beat_times = []
    current_time = 0.0
    sample_idx = 0

    while current_time < duration_s and sample_idx < num_samples:
        beat_times.append(current_time)

        # Get target HR at this time point
        current_hr = hr_over_time[min(sample_idx, num_samples - 1)]
        mean_ibi = 60.0 / current_hr  # seconds per beat

        # Add heart rate variability (HRV): ±3-5% beat-to-beat variation is normal
        hrv_std = 0.04 * mean_ibi  # 4% standard deviation
        ibi = np.random.normal(mean_ibi, hrv_std)
        ibi = max(0.4, min(1.2, ibi))  # Clamp to reasonable range (50-150 BPM)

        current_time += ibi
        sample_idx = int(current_time * rate_hz)

    # Create PPG signal with realistic pulse waveforms
    # PPG pulse shape: rapid upstroke, slower decay, dicrotic notch
    signal = np.zeros(num_samples)
    baseline = 50000
    pulse_amplitude = 3500  # Realistic amplitude for good signal quality

    for beat_time in beat_times:
        beat_idx = int(beat_time * rate_hz)

        # Pulse waveform parameters
        upstroke_samples = int(0.10 * rate_hz)  # 100ms upstroke
        decay_samples = int(0.25 * rate_hz)     # 250ms decay
        dicrotic_delay = int(0.20 * rate_hz)    # Dicrotic notch at 200ms

        # Systolic upstroke (rapid)
        for i in range(upstroke_samples):
            idx = beat_idx + i
            if idx < num_samples:
                signal[idx] += pulse_amplitude * (i / upstroke_samples) ** 2

        # Diastolic decay (slower)
        for i in range(decay_samples):
            idx = beat_idx + upstroke_samples + i
            if idx < num_samples:
                decay_factor = np.exp(-3 * i / decay_samples)
                signal[idx] += pulse_amplitude * decay_factor

        # Dicrotic notch (small secondary peak)
        idx = beat_idx + dicrotic_delay
        if idx < num_samples:
            signal[idx:idx+5] += pulse_amplitude * 0.15

    # Add baseline and realistic noise
    noise = np.random.normal(0, 80, num_samples)  # Physiological noise
    signal = baseline + signal + noise

    return signal.astype(int)


def generate_imu_data(duration_s: float, rate_hz: int, activity: str) -> tuple:
    """Generate realistic IMU data for different activity states.

    Args:
        duration_s: Duration in seconds
        rate_hz: Sampling rate in Hz
        activity: "resting" or "walking"

    Returns:
        Tuple of (ax, ay, az, gx, gy, gz) arrays
    """
    num_samples = int(duration_s * rate_hz)

    if activity == "resting":
        # Resting: gravity + small breathing motion + minimal drift
        ax = np.random.normal(0.02, 0.015, num_samples)
        ay = np.random.normal(-0.98, 0.02, num_samples)  # Gravity mostly on Y
        az = np.random.normal(0.08, 0.015, num_samples)
        gx = np.random.normal(0, 0.02, num_samples)
        gy = np.random.normal(0, 0.02, num_samples)
        gz = np.random.normal(0, 0.02, num_samples)
    else:  # walking
        # Walking: rhythmic motion at ~2 Hz (120 steps/min)
        t = np.linspace(0, duration_s, num_samples)
        step_freq = 2.0  # Hz

        # Walking creates periodic acceleration in all axes
        ax = 0.15 * np.sin(2 * np.pi * step_freq * t) + np.random.normal(0, 0.05, num_samples)
        ay = -0.85 + 0.20 * np.cos(2 * np.pi * step_freq * t) + np.random.normal(0, 0.05, num_samples)
        az = 0.10 * np.sin(2 * np.pi * step_freq * t + 0.5) + np.random.normal(0, 0.05, num_samples)

        # Gyro shows rotation during walking
        gx = 0.30 * np.sin(2 * np.pi * step_freq * t + 0.3) + np.random.normal(0, 0.08, num_samples)
        gy = 0.25 * np.cos(2 * np.pi * step_freq * t) + np.random.normal(0, 0.08, num_samples)
        gz = 0.15 * np.sin(2 * np.pi * step_freq * t - 0.2) + np.random.normal(0, 0.08, num_samples)

    return ax, ay, az, gx, gy, gz


def main():
    """Generate 3-minute demo CSV with realistic variability."""

    rate_hz = 100
    sample_interval_ms = 10

    # Demo scenario timeline:
    # 0-70s: Resting (calibration) at baseline ~72 BPM
    # 70-100s: Walking, HR rises to ~105 BPM (30s minimum for recovery detection)
    # 100-180s: Return to rest, SLOW recovery (105→85 BPM exponentially, tau~90s = "slow")

    all_samples = []
    start_time_ms = 1730000000000
    current_time_ms = start_time_ms

    # Phase 1: Baseline calibration (70s at 72 BPM)
    print(f"Generating 70s of resting at ~72 BPM...")
    ppg = generate_realistic_ppg(70, 72, rate_hz)
    ax, ay, az, gx, gy, gz = generate_imu_data(70, rate_hz, "resting")
    for i in range(int(70 * rate_hz)):
        all_samples.append({
            "t": current_time_ms,
            "ppg": int(ppg[i]),
            "ax": round(ax[i], 4),
            "ay": round(ay[i], 4),
            "az": round(az[i], 4),
            "gx": round(gx[i], 4),
            "gy": round(gy[i], 4),
            "gz": round(gz[i], 4),
        })
        current_time_ms += sample_interval_ms

    # Phase 2: Walking (30s at 105 BPM)
    print(f"Generating 30s of walking at ~105 BPM...")
    ppg = generate_realistic_ppg(30, 105, rate_hz)
    ax, ay, az, gx, gy, gz = generate_imu_data(30, rate_hz, "walking")
    for i in range(int(30 * rate_hz)):
        all_samples.append({
            "t": current_time_ms,
            "ppg": int(ppg[i]),
            "ax": round(ax[i], 4),
            "ay": round(ay[i], 4),
            "az": round(az[i], 4),
            "gx": round(gx[i], 4),
            "gy": round(gy[i], 4),
            "gz": round(gz[i], 4),
        })
        current_time_ms += sample_interval_ms

    # Phase 3: Slow recovery (80s, exponential decay from 105 to 85 BPM)
    # Slow recovery: tau ~90s (vs normal ~40-60s)
    # HR(t) = baseline + (HR0 - baseline) * e^(-t/tau)
    # HR(t) = 72 + (105 - 72) * e^(-t/90) = 72 + 33 * e^(-t/90)
    print(f"Generating 80s of resting with slow recovery...")
    recovery_duration = 80
    baseline_hr = 72
    hr0 = 105
    tau = 90  # Slow recovery time constant
    t_recovery = np.linspace(0, recovery_duration, int(recovery_duration * rate_hz))
    hr_recovery = baseline_hr + (hr0 - baseline_hr) * np.exp(-t_recovery / tau)

    ppg = generate_realistic_ppg(recovery_duration, hr_recovery, rate_hz)
    ax, ay, az, gx, gy, gz = generate_imu_data(recovery_duration, rate_hz, "resting")
    for i in range(int(recovery_duration * rate_hz)):
        all_samples.append({
            "t": current_time_ms,
            "ppg": int(ppg[i]),
            "ax": round(ax[i], 4),
            "ay": round(ay[i], 4),
            "az": round(az[i], 4),
            "gx": round(gx[i], 4),
            "gy": round(gy[i], 4),
            "gz": round(gz[i], 4),
        })
        current_time_ms += sample_interval_ms

    # Create DataFrame and save
    df = pd.DataFrame(all_samples)
    output_path = ROOT / "data" / "demo_2min_elevated_rest.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ Generated {len(df)} samples ({len(df)/rate_hz:.1f}s)")
    print(f"   Saved to: {output_path}")
    print(f"\nPPG statistics:")
    print(f"   Range: {df.ppg.min()} to {df.ppg.max()}")
    print(f"   Mean: {df.ppg.mean():.0f}, Std: {df.ppg.std():.0f}")
    print(f"\nTimeline:")
    print(f"   0-70s:    Baseline calibration (~72 BPM)")
    print(f"   70-100s:  Walking (~105 BPM, 30s)")
    print(f"   100-180s: Slow recovery (105→85 BPM, tau~90s)")
    print(f"\nExpected behavior:")
    print(f"   - Baseline calibrates during first 60s of rest")
    print(f"   - Activity detector recognizes walking at 70-100s")
    print(f"   - Recovery model fits exponential decay after 100s")
    print(f"   - Recovery verdict: 'slow' (tau~90s vs normal ~40-60s)")
    print(f"   - Post-walk grace period: 100-160s")
    print(f"   - Alert triggers at ~160s+ (NOTIFY level)")
    print(f"\nRecovery dynamics (exponential decay):")
    print(f"   HR(t) = 72 + 33 * e^(-t/90)")
    print(f"   120s: ~{baseline_hr + (hr0 - baseline_hr) * np.exp(-20/tau):.0f} BPM")
    print(f"   140s: ~{baseline_hr + (hr0 - baseline_hr) * np.exp(-40/tau):.0f} BPM")
    print(f"   180s: ~{baseline_hr + (hr0 - baseline_hr) * np.exp(-80/tau):.0f} BPM")


if __name__ == "__main__":
    main()
