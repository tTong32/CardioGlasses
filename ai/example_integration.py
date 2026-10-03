"""Integration example: Activity + Baseline + Recovery pipeline.

This demonstrates the complete CardioGlasses recovery model pipeline using
all three mandatory components working together.
"""
import numpy as np
from ai.activity import ActivityDetector
from ai.baseline import BaselineTracker
from ai.recovery import RecoveryModel


def simulate_cardio_session(seed: int = 42):
    """Simulate a complete cardio session with real-time processing.

    Timeline:
    - 0-90s: Rest (baseline calibration)
    - 90-120s: Exercise (HR rises)
    - 120-300s: Recovery (HR returns to baseline)

    Simulates PPG and accelerometer sampling at 10ms (100 Hz).
    """
    rng = np.random.RandomState(seed)

    # Initialize all three modules
    activity = ActivityDetector()
    baseline = BaselineTracker()
    recovery = RecoveryModel()

    print("=" * 70)
    print("CardioGlasses Recovery Model - Integration Demo")
    print("=" * 70)
    print("\nInitializing modules...")
    print(f"  ✓ ActivityDetector (window={2000}ms, threshold={0.05} m/s²)")
    print(f"  ✓ BaselineTracker (min_rest={60}s, window={180}s)")
    print(f"  ✓ RecoveryModel (population cutoffs loaded)")
    print()

    t_ms = 0
    sample_count = 0
    last_status_time = -10

    # Main loop: process every 10ms
    for t_s in np.arange(0, 300, 0.01):  # 30,000 samples over 5 minutes
        t_ms = int(t_s * 1000)
        sample_count += 1

        # === Simulate hardware inputs ===

        # 1. Accelerometer (ax, ay, az) - 10ms sampling
        if t_s < 90:
            # Resting: minimal movement
            ax, ay, az = 0.0, 0.0, 9.81
        elif t_s < 120:
            # Exercise: significant movement
            ax = rng.normal(0, 3)
            ay = rng.normal(0, 3)
            az = rng.normal(9.81, 3)
        else:
            # Recovery: back to rest
            ax, ay, az = 0.0, 0.0, 9.81

        # 2. PPG sensor → heart rate - 10ms sampling
        if t_s < 90:
            # Resting HR
            hr_true = 72.0 + rng.normal(0, 2)
        elif t_s < 120:
            # Exercise: HR rises to ~120 bpm
            progress = (t_s - 90) / 30
            hr_true = 72 + progress * 48 + rng.normal(0, 3)
        else:
            # Recovery: exponential decay (tau ~60s)
            time_since_stop = t_s - 120
            hr0 = 120
            baseline_hr = 72
            tau = 60
            hr_true = baseline_hr + (hr0 - baseline_hr) * np.exp(-time_since_stop / tau)
            hr_true += rng.normal(0, 2)

        # === Process through pipeline (every 10ms) ===

        # Step 1: Activity detection from accelerometer
        activity_output = activity.update(ax, ay, az)

        # Step 2: Baseline calibration (update every 100ms to reduce computation)
        if sample_count % 10 == 0:
            baseline_output = baseline.update(
                t_ms,
                hr_true,
                activity_output.activity
            )

        # Step 3: Recovery model (update every 2000ms)
        if sample_count % 200 == 0:
            recovery_output = recovery.update(
                t_ms,
                hr_true,
                activity_output.activity,
                baseline_output.baseline_hr,
                baseline_output.baseline_std
            )

            # Print status updates every 10 seconds
            if t_s - last_status_time >= 10:
                last_status_time = t_s
                _print_status(t_s, activity_output, baseline_output, recovery_output, hr_true)

    print("\n" + "=" * 70)
    print("Session Complete")
    print("=" * 70)
    print("\nFinal Results:")
    print(f"  Baseline HR: {baseline_output.baseline_hr:.1f} ± {baseline_output.baseline_std:.1f} bpm")
    if recovery_output.recovery_tau_s:
        print(f"  Recovery tau: {recovery_output.recovery_tau_s:.1f}s")
        print(f"  Recovery R²: {recovery_output.recovery_r2:.3f}")
        print(f"  Population percentile: {recovery_output.recovery_percentile:.0f}%")
        print(f"  Verdict: {recovery_output.recovery_verdict}")
        if recovery_output.recovery_ratio:
            print(f"  Personal ratio: {recovery_output.recovery_ratio:.2f}x")
    print()


def _print_status(t_s, activity_out, baseline_out, recovery_out, hr):
    """Print current status."""
    time_str = f"{int(t_s):3d}s"
    hr_str = f"HR={hr:5.1f}"

    # Activity
    activity_str = f"Activity={activity_out.activity:7s}"

    # Baseline
    if baseline_out.is_calibrated:
        baseline_str = f"Baseline={baseline_out.baseline_hr:5.1f}±{baseline_out.baseline_std:3.1f}"
    else:
        rest_progress = min(baseline_out.time_resting_s / 60, 1.0) * 100
        baseline_str = f"Calibrating... {rest_progress:3.0f}%"

    # Recovery
    state_map = {
        "idle": "Idle",
        "active": "Tracking",
        "complete": "Complete ✓",
        "aborted": "Aborted"
    }
    recovery_str = f"Recovery={state_map.get(recovery_out.episode_state, recovery_out.episode_state):11s}"

    if recovery_out.recovery_tau_s:
        recovery_str += f" τ={recovery_out.recovery_tau_s:5.1f}s"

    print(f"{time_str} | {hr_str} | {activity_str} | {baseline_str:25s} | {recovery_str}")


if __name__ == "__main__":
    simulate_cardio_session(seed=42)
