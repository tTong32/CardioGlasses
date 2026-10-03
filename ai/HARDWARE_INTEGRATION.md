# Hardware Integration Guide

## Required Hardware Inputs

The CardioGlasses recovery model requires **two sensors**, both sampling at **10 milliseconds (100 Hz)**:

### 1. PPG Sensor (Photoplethysmography)
- **Purpose**: Heart rate measurement
- **Sample Rate**: 10ms (100 Hz)
- **Output**: Raw PPG signal → processed to heart rate (bpm)
- **Contract**: `Sample.ppg` field (Contract A)

### 2. 3-Axis Accelerometer
- **Purpose**: Activity detection (resting vs moving)
- **Sample Rate**: 10ms (100 Hz)
- **Output**: `ax`, `ay`, `az` (m/s²)
- **Contract**: `Sample.ax`, `Sample.ay`, `Sample.az` fields (Contract A)

## Processing Pipeline

The complete pipeline consists of three mandatory modules:

```
Accelerometer (10ms)  →  ActivityDetector  →  "resting" / "moving"
                                                         ↓
PPG Sensor (10ms)     →  HR Processor      →  Heart Rate (bpm)
                                                         ↓
                                              BaselineTracker (calibration)
                                                         ↓
                                              RecoveryModel (episode tracking)
                                                         ↓
                                              Contract B (Reading)
```

## Module Usage

### 1. ActivityDetector

**File**: `ai/activity.py`

**Purpose**: Classifies activity as "resting" or "moving" based on accelerometer data.

**API**:
```python
from ai.activity import ActivityDetector

# Initialize once
detector = ActivityDetector(
    window_ms=2000,          # 2-second rolling window
    stillness_threshold=0.05  # m/s² std threshold
)

# Update every 10ms with accelerometer reading
output = detector.update(ax, ay, az)

# Output fields:
# - output.activity: "resting" or "moving"
# - output.accel_std: standard deviation of acceleration magnitude
```

**Tunable Constants** (in `ai/activity.py`):
- `WINDOW_SIZE_MS = 2000`: Classification window (ms)
- `STILLNESS_THRESHOLD = 0.05`: Accel std threshold for resting (m/s²)
- `SAMPLE_RATE_MS = 10`: Expected accelerometer sample rate

---

### 2. BaselineTracker

**File**: `ai/baseline.py`

**Purpose**: Calibrates resting heart rate baseline from continuous rest periods.

**API**:
```python
from ai.baseline import BaselineTracker

# Initialize once
tracker = BaselineTracker(
    min_resting_s=60,        # Need 60s continuous rest
    window_s=180,            # Use last 3 minutes of data
    update_interval_s=10     # Recompute every 10 seconds
)

# Update periodically (e.g., every 100ms) with HR and activity
output = tracker.update(t_ms, hr, activity)

# Output fields:
# - output.baseline_hr: Mean resting HR (bpm) or None
# - output.baseline_std: Std of resting HR (bpm) or None
# - output.is_calibrated: True if baseline is valid
# - output.time_resting_s: Seconds of continuous rest
```

**Tunable Constants** (in `ai/baseline.py`):
- `MIN_RESTING_DURATION_S = 60`: Minimum continuous rest for calibration
- `CALIBRATION_WINDOW_S = 180`: Rolling window size for baseline computation
- `UPDATE_INTERVAL_S = 10`: How often to recompute baseline
- `OUTLIER_THRESHOLD = 3.0`: MAD threshold for outlier rejection

---

### 3. RecoveryModel

**File**: `ai/recovery.py`

**Purpose**: Tracks heart rate recovery after exercise and classifies speed.

**API**:
```python
from ai.recovery import RecoveryModel

# Initialize once per patient
model = RecoveryModel(
    medications=["metoprolol", "aspirin"],  # HR-altering meds
    cutoffs_path="ai/model/recovery_cutoffs.json"
)

# Update every 2 seconds with HR, activity, and baseline
output = model.update(
    t_ms=timestamp_ms,
    hr=heart_rate_bpm,
    activity="resting",  # from ActivityDetector
    baseline_hr=72.0,    # from BaselineTracker
    baseline_std=3.0     # from BaselineTracker
)

# Output fields:
# - output.episode_state: "idle", "active", "complete", "aborted"
# - output.recovery_tau_s: Exponential decay time constant (seconds)
# - output.recovery_r2: Fit quality (0-1)
# - output.recovery_percentile: Population percentile (0-100)
# - output.recovery_ratio: Personal ratio vs reference (≥1.0)
# - output.recovery_verdict: "normal", "slow", "very_slow"
```

**Key Constants** (in `ai/recovery.py`):
- `MIN_MOVING_S = 30`: Minimum exercise duration
- `MIN_RISE_BPM = 10`: Minimum HR rise above baseline
- `MIN_FIT_S = 20`: Start fitting after 20s of recovery
- `TAU_BOUNDS = (5, 600)`: Tau must be in 5-600s range
- `MIN_R2 = 0.5`: Minimum R² for valid recovery
- `RATIO_SLOW = 1.5`: Personal ratio threshold for "slow"
- `RATIO_VERY_SLOW = 2.5`: Personal ratio threshold for "very_slow"

---

## Complete Integration Example

See `ai/example_integration.py` for a full working example:

```bash
python3 -m ai.example_integration
```

This simulates a complete cardio session:
1. **0-90s**: Rest → baseline calibration
2. **90-120s**: Exercise → HR rises
3. **120-300s**: Recovery → exponential decay tracking

---

## Data Flow to Backend (Contract B)

The `Reading` model (Contract B) expects these fields from the recovery pipeline:

```python
reading = Reading(
    t=timestamp_ms,
    hr=heart_rate_bpm,                    # From PPG processor
    activity=activity_output.activity,    # From ActivityDetector
    baseline_hr=baseline_output.baseline_hr,  # From BaselineTracker
    deviation=hr - baseline_hr if baseline_hr else None,
    recovery_tau_s=recovery_output.recovery_tau_s,  # From RecoveryModel
    recovery_ratio=recovery_output.recovery_ratio,
    recovery_percentile=recovery_output.recovery_percentile,
    recovery_verdict=recovery_output.recovery_verdict,
    # ... other fields
)
```

---

## Testing

All modules have comprehensive tests:

```bash
# Test activity detector + baseline tracker (17 tests)
python3 -m pytest ai/tests/test_activity_baseline.py -v

# Test recovery model (23 tests)
python3 -m pytest ai/tests/test_recovery.py -v

# Run all tests (40 tests)
python3 -m pytest ai/tests/ -v
```

---

## Hardware Specs Summary

| Component       | Sample Rate | Output                | Used By           |
|-----------------|-------------|-----------------------|-------------------|
| PPG Sensor      | 10ms (100Hz)| Heart Rate (bpm)      | All modules       |
| Accelerometer   | 10ms (100Hz)| ax, ay, az (m/s²)     | ActivityDetector  |

**Total bandwidth**: ~200 samples/second (2 sensors × 100 Hz)

**Processing updates**:
- ActivityDetector: Every 10ms (real-time)
- BaselineTracker: Every 100ms (recommended)
- RecoveryModel: Every 2000ms (required)

**Calibration requirements**:
- Baseline: 60s continuous rest minimum
- Recovery reference: 1 good recovery episode (R² ≥ 0.7)

---

## Population Cutoffs

The model uses trained percentiles from cardiac rehabilitation patients:

| Group   | N   | Tau p50 | Tau p75 | Tau p90 |
|---------|-----|---------|---------|---------|
| all     | 200 | 50.9s   | 72.9s   | 103.1s  |
| meds    | 111 | 54.2s   | 82.0s   | 109.4s  |
| no_meds | 89  | 45.6s   | 65.9s   | 88.4s   |

**Location**: `ai/model/recovery_cutoffs.json`

**Source**: PhysioNet frailty dataset (80 post-cardiac surgery patients aged 65+)

**Re-training**: See `ai/train/README.md` for instructions on re-training with real data.
