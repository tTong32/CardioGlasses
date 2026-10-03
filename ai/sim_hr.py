"""Heart rate simulator for testing the recovery model.

Generates synthetic HR data at 2-second intervals with different recovery scenarios.
"""
from typing import Optional
import json
from pathlib import Path
import numpy as np


def simulate(scenario: str, seed: int = 42) -> list[tuple[int, Optional[float], str]]:
    """Simulate heart rate data for testing.

    Args:
        scenario: one of "fast", "slow", "aborted", "tiny_rise", "calibration_then_slow"
        seed: random seed for reproducibility

    Returns:
        List of (t_ms, hr, activity) tuples, sampled every 2 seconds
        - t_ms: time in milliseconds
        - hr: heart rate in bpm, or None for missing readings
        - activity: "resting" or "moving"
    """
    rng = np.random.RandomState(seed)

    # Try to load trained cutoffs for realistic tau values
    cutoffs_path = Path("ai/model/recovery_cutoffs.json")
    if cutoffs_path.exists():
        with open(cutoffs_path) as f:
            cutoffs = json.load(f)
            all_group = cutoffs["groups"]["all"]
            tau_fast = all_group["tau_p50"]
            tau_slow = 1.3 * all_group["tau_p90"]  # Phase 3 spec: slow = 1.3 × p90
    else:
        # Fallback values before training
        tau_fast = 30
        tau_slow = 150

    def add_noise(hr: float) -> Optional[float]:
        """Add Gaussian noise and occasional dropouts."""
        if rng.rand() < 0.05:  # 5% dropout rate
            return None
        return hr + rng.normal(0, 2)  # ~2 bpm noise

    def recovery_hr(t: float, baseline: float, hr0: float, tau: float) -> float:
        """Exponential recovery curve."""
        return baseline + (hr0 - baseline) * np.exp(-t / tau)

    data = []
    t_ms = 0

    if scenario == "fast":
        # 60s rest @ 70 → 60s moving ramping to 105 → 180s recovery with tau=30
        baseline = 70

        # Rest phase
        for _ in range(30):  # 60 seconds / 2s
            data.append((t_ms, add_noise(baseline), "resting"))
            t_ms += 2000

        # Moving phase - ramp up
        for i in range(30):  # 60 seconds / 2s
            progress = i / 30
            hr = baseline + progress * 35  # ramp to 105
            data.append((t_ms, add_noise(hr), "moving"))
            t_ms += 2000

        # Recovery phase
        hr0 = 105
        for i in range(90):  # 180 seconds / 2s
            t_recovery = i * 2
            hr = recovery_hr(t_recovery, baseline, hr0, tau_fast)
            data.append((t_ms, add_noise(hr), "resting"))
            t_ms += 2000

    elif scenario == "slow":
        # Same structure but tau=150 (slow recovery)
        baseline = 70

        # Rest
        for _ in range(30):
            data.append((t_ms, add_noise(baseline), "resting"))
            t_ms += 2000

        # Moving
        for i in range(30):
            progress = i / 30
            hr = baseline + progress * 35
            data.append((t_ms, add_noise(hr), "moving"))
            t_ms += 2000

        # Recovery
        hr0 = 105
        for i in range(90):
            t_recovery = i * 2
            hr = recovery_hr(t_recovery, baseline, hr0, tau_slow)
            data.append((t_ms, add_noise(hr), "resting"))
            t_ms += 2000

    elif scenario == "aborted":
        # Recovery starts but moving again at 40s
        baseline = 70

        # Rest
        for _ in range(30):
            data.append((t_ms, add_noise(baseline), "resting"))
            t_ms += 2000

        # Moving
        for i in range(30):
            progress = i / 30
            hr = baseline + progress * 35
            data.append((t_ms, add_noise(hr), "moving"))
            t_ms += 2000

        # Brief recovery (20 readings = 40 seconds)
        hr0 = 105
        for i in range(20):
            t_recovery = i * 2
            hr = recovery_hr(t_recovery, baseline, hr0, tau_fast)
            data.append((t_ms, add_noise(hr), "resting"))
            t_ms += 2000

        # Start moving again
        for i in range(30):
            progress = i / 30
            hr = 95 + progress * 10  # ramp up again
            data.append((t_ms, add_noise(hr), "moving"))
            t_ms += 2000

    elif scenario == "tiny_rise":
        # Moving but HR only reaches 75 (rise = 5, below MIN_RISE_BPM=10)
        baseline = 70

        # Rest
        for _ in range(30):
            data.append((t_ms, add_noise(baseline), "resting"))
            t_ms += 2000

        # Moving with tiny rise
        for i in range(30):
            progress = i / 30
            hr = baseline + progress * 5  # only to 75
            data.append((t_ms, add_noise(hr), "moving"))
            t_ms += 2000

        # Return to rest
        for i in range(60):
            t_recovery = i * 2
            hr = recovery_hr(t_recovery, baseline, 75, tau_fast)
            data.append((t_ms, add_noise(hr), "resting"))
            t_ms += 2000

    elif scenario == "calibration_then_slow":
        # Fast recovery, 30s rest, then a second exertion with slow recovery
        baseline = 70

        # First cycle: rest → move → fast recovery
        for _ in range(30):
            data.append((t_ms, add_noise(baseline), "resting"))
            t_ms += 2000

        for i in range(30):
            progress = i / 30
            hr = baseline + progress * 35
            data.append((t_ms, add_noise(hr), "moving"))
            t_ms += 2000

        hr0 = 105
        for i in range(90):  # 180s recovery
            t_recovery = i * 2
            hr = recovery_hr(t_recovery, baseline, hr0, tau_fast)
            data.append((t_ms, add_noise(hr), "resting"))
            t_ms += 2000

        # 30s rest
        for _ in range(15):
            data.append((t_ms, add_noise(baseline), "resting"))
            t_ms += 2000

        # Second cycle: move → slow recovery
        for i in range(30):
            progress = i / 30
            hr = baseline + progress * 35
            data.append((t_ms, add_noise(hr), "moving"))
            t_ms += 2000

        hr0 = 105
        for i in range(90):
            t_recovery = i * 2
            hr = recovery_hr(t_recovery, baseline, hr0, tau_slow)
            data.append((t_ms, add_noise(hr), "resting"))
            t_ms += 2000

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return data
