"""All tunable thresholds and the demo patient configuration."""

import os

from ai.contracts import PatientContext, Config

# ============================================================================
# Signal processing (Phase 2)
# ============================================================================

# Sampling and windowing
SAMPLE_RATE_HZ = 50
WINDOW_DURATION_S = 10.0

# Bandpass filter for PPG (Butterworth, order 2)
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 4.0
BANDPASS_ORDER = 2

# Peak detection
PEAK_MIN_DISTANCE_S = 0.33  # ~180 bpm max
PEAK_PROMINENCE_FACTOR = 0.5  # × window std

# Beat gap filtering
IBI_MIN_MS = 333  # 180 bpm max
IBI_MAX_MS = 1500  # 40 bpm min
IBI_OUTLIER_THRESHOLD = 0.30  # drop gaps > 30% from median
MIN_VALID_BEATS = 4  # minimum for HR calculation

# Quality assessment
QUALITY_FLAT_THRESHOLD = 10.0  # std in sensor counts
QUALITY_MOVING_PENALTY = 0.5  # unused: motion artifacts already lower the measured quality
QUALITY_CLIPPING_THRESHOLD = 0.05  # fraction of samples at min/max

# Signal status
SIGNAL_OFFLINE_TIMEOUT_S = 3.0

# Activity detection
ACTIVITY_ACCEL_THRESHOLD_G = 0.025  # std of |a| in g: rest ~0.004, synthetic movement 0.047+
ACTIVITY_HYSTERESIS_COUNT = 2  # consecutive checks to switch state
ACTIVITY_WINDOW_S = 2.0  # window for computing std

# ============================================================================
# Baseline and deviation (Phase 3)
# ============================================================================

BASELINE_CALIBRATION_DURATION_S = 60.0
BASELINE_MIN_STD = 4.0  # bpm, floor for deviation calculation

# ============================================================================
# Recovery model (Phase 4)
# ============================================================================

# Trigger conditions
RECOVERY_MIN_ACTIVITY_DURATION_S = 30.0
RECOVERY_MIN_HR_ELEVATION = 15.0  # bpm above baseline

# Curve fitting
RECOVERY_FIT_MIN_DURATION_S = 20.0  # before attempting fit
RECOVERY_TAU_MIN_S = 5.0
RECOVERY_TAU_MAX_S = 600.0
RECOVERY_HR0_WINDOW_S = 3.0  # ± around flip time for HR0

# Verdict thresholds
RECOVERY_NORMAL_RATIO_MAX = 1.5
RECOVERY_SLOW_RATIO_MAX = 2.5
RECOVERY_SLOW_DROP_THRESHOLD = 12.0  # bpm at 60s

# Episode termination
RECOVERY_SETTLED_WINDOW_S = 20.0  # HR within baseline + 1std for this long
RECOVERY_MAX_DURATION_S = 300.0

# ============================================================================
# Decision rules (Phase 5)
# ============================================================================

# Deviation thresholds are in PatientContext.config
# These are additional rule parameters
DEVIATION_ESCALATE_MULTIPLIER = 2.0  # deviation >= 2 × trigger
DEVIATION_NORMAL_THRESHOLD = 1.0  # return to normal
DEVIATION_NORMAL_DURATION_S = 20.0

# Confidence
CONFIDENCE_MIN_FOR_ALERT = 0.5  # cap level at monitor if below

# ============================================================================
# Explainer (Phase 5)
# ============================================================================

EXPLAINER_MODEL = "gemini-flash-lite-latest"  # ~0.7 s; override with GEMINI_MODEL in .env
EXPLAINER_TIMEOUT_S = 5.0  # wait this long, then send templated wording
EXPLAINER_MAX_VOICE_WORDS = 25
EXPLAINER_HR_BUCKET = 5  # bpm, for cache key
# A reading at or above this, with no signal note, counts as clean without asking Gemini.
SIGNAL_CHECK_STRONG = 0.85
SIGNAL_CHECK_TIMEOUT_S = 1.5  # don't hold the alert longer than this for a second opinion


def signal_check_may_delay() -> bool:
    """When set, one artifact verdict can hold a monitor or notify for a single reading.

    Escalate is never held. Off unless SIGNAL_CHECK_DELAY=1, so a wrong answer cannot
    swallow the alert the demo is there to show.
    """
    return os.environ.get("SIGNAL_CHECK_DELAY", "").strip().lower() in ("1", "true", "yes")

# ============================================================================
# Synthetic data generation (Phase 1)
# ============================================================================

# PPG synthesis
PPG_DC_LEVEL = 50000  # sensor counts
PPG_PULSE_AMPLITUDE = 1500  # sensor counts
PPG_NOISE_STD = 30  # sensor counts
PPG_DRIFT_FREQ_HZ = 0.2  # slow baseline wander
PPG_DRIFT_AMPLITUDE = 500  # sensor counts

# Motion artifacts
MOTION_ARTIFACT_AMPLITUDE = 800  # sensor counts, correlated with accel
MOTION_ARTIFACT_FREQ_HZ = 1.5  # Hz

# IMU at rest
IMU_GRAVITY_Y = -0.98  # g
IMU_REST_ACCEL_NOISE = 0.005  # g
IMU_REST_GYRO_NOISE = 0.02  # deg/s

# IMU during movement
IMU_MOVING_ACCEL_AMPLITUDE = 0.15  # g
IMU_MOVING_GYRO_AMPLITUDE = 20.0  # deg/s
IMU_MOVING_FREQ_HZ = 1.5  # Hz

# Beat-to-beat variability
HRV_JITTER_PERCENT = 3.0  # percent of IBI

# Sensor limits (for clipping)
PPG_SENSOR_MIN = 0
PPG_SENSOR_MAX = 65535

# ============================================================================
# Demo patient
# ============================================================================

DEMO_PATIENT = PatientContext(
    patient_id="demo-1",
    age=71,
    conditions=["hypertension", "type 2 diabetes"],
    medications=["metformin", "lisinopril"],
    risk_tier="high",
    config=Config(
        min_quality=0.6,
        persist_s=30,
        deviation_trigger=2.0,
        cooldown_s=120,
    ),
)
