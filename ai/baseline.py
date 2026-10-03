"""Baseline heart rate tracking for CardioGlasses.

Calibrates resting HR baseline by collecting data during extended rest periods.
"""
from dataclasses import dataclass
from collections import deque
from typing import Optional
import numpy as np


# ============================================================================
# TUNABLE CONSTANTS
# ============================================================================

MIN_RESTING_DURATION_S = 60  # Need 60s continuous rest for calibration
CALIBRATION_WINDOW_S = 180  # Use last 3 minutes of resting data
UPDATE_INTERVAL_S = 10  # Update baseline every 10 seconds
OUTLIER_THRESHOLD = 3.0  # Remove HR readings > 3 std from median


# ============================================================================
# BASELINE TRACKER
# ============================================================================


@dataclass
class BaselineOutput:
    """Output from baseline tracker."""
    baseline_hr: Optional[float]  # mean resting HR (bpm)
    baseline_std: Optional[float]  # std of resting HR (bpm)
    is_calibrated: bool  # True if we have valid baseline
    time_resting_s: float  # seconds of continuous rest


class BaselineTracker:
    """Real-time baseline HR calibration from resting periods."""

    def __init__(
        self,
        min_resting_s: float = MIN_RESTING_DURATION_S,
        window_s: float = CALIBRATION_WINDOW_S,
        update_interval_s: float = UPDATE_INTERVAL_S
    ):
        """Initialize the baseline tracker.

        Args:
            min_resting_s: minimum continuous rest time for calibration (seconds)
            window_s: rolling window size for baseline computation (seconds)
            update_interval_s: how often to recompute baseline (seconds)
        """
        self._min_resting_s = min_resting_s
        self._window_s = window_s
        self._update_interval_s = update_interval_s

        # Rolling window of (timestamp_ms, hr) during rest
        self._resting_data: deque = deque()

        # Calibrated baseline values
        self._baseline_hr: Optional[float] = None
        self._baseline_std: Optional[float] = None
        self._is_calibrated = False

        # Tracking continuous rest
        self._time_resting_s = 0.0
        self._last_rest_start_ms: Optional[int] = None
        self._last_update_ms: Optional[int] = None

    def update(
        self,
        t_ms: int,
        hr: Optional[float],
        activity: str
    ) -> BaselineOutput:
        """Process a new HR reading and activity state.

        Args:
            t_ms: timestamp in milliseconds
            hr: heart rate in bpm (or None if invalid)
            activity: "resting" or "moving"

        Returns:
            BaselineOutput with current baseline calibration
        """
        # Handle invalid HR
        if hr is None or np.isnan(hr):
            # Reset rest tracking
            self._last_rest_start_ms = None
            self._time_resting_s = 0.0
            return self._get_output()

        # Track continuous rest periods
        if activity == "resting":
            if self._last_rest_start_ms is None:
                # Just started resting
                self._last_rest_start_ms = t_ms
                self._time_resting_s = 0.0
            else:
                # Continue resting
                self._time_resting_s = (t_ms - self._last_rest_start_ms) / 1000.0

            # Add to resting data window
            self._resting_data.append((t_ms, hr))

            # Prune old data outside window
            cutoff_ms = t_ms - int(self._window_s * 1000)
            while self._resting_data and self._resting_data[0][0] < cutoff_ms:
                self._resting_data.popleft()

            # Update baseline periodically
            if self._should_update_baseline(t_ms):
                self._compute_baseline()
                self._last_update_ms = t_ms

        else:
            # Movement detected - reset rest tracking
            self._last_rest_start_ms = None
            self._time_resting_s = 0.0
            # Keep resting_data for future calibration, but stop adding to it

        return self._get_output()

    def _should_update_baseline(self, t_ms: int) -> bool:
        """Check if it's time to recompute baseline."""
        if self._last_update_ms is None:
            return True

        elapsed_s = (t_ms - self._last_update_ms) / 1000.0
        return elapsed_s >= self._update_interval_s

    def _compute_baseline(self):
        """Compute baseline HR and std from resting data."""
        if len(self._resting_data) < 10:
            # Not enough data
            self._is_calibrated = False
            return

        # Check if we have minimum continuous rest
        if self._time_resting_s < self._min_resting_s:
            self._is_calibrated = False
            return

        # Extract HR values
        hr_values = np.array([hr for _, hr in self._resting_data])

        # Remove outliers using median absolute deviation
        median_hr = np.median(hr_values)
        deviations = np.abs(hr_values - median_hr)
        mad = np.median(deviations)

        if mad > 0:
            # Use MAD for outlier detection
            threshold = OUTLIER_THRESHOLD * 1.4826 * mad  # 1.4826 makes MAD ~= std for normal
            mask = deviations <= threshold
        else:
            # No variation - keep all points
            mask = np.ones(len(hr_values), dtype=bool)

        clean_hr = hr_values[mask]

        if len(clean_hr) < 10:
            # Too much data removed
            self._is_calibrated = False
            return

        # Compute baseline
        self._baseline_hr = float(np.mean(clean_hr))
        self._baseline_std = float(np.std(clean_hr))
        self._is_calibrated = True

    def _get_output(self) -> BaselineOutput:
        """Get current baseline state."""
        return BaselineOutput(
            baseline_hr=self._baseline_hr,
            baseline_std=self._baseline_std,
            is_calibrated=self._is_calibrated,
            time_resting_s=self._time_resting_s
        )

    @property
    def baseline_hr(self) -> Optional[float]:
        """Get current baseline HR."""
        return self._baseline_hr

    @property
    def baseline_std(self) -> Optional[float]:
        """Get current baseline std."""
        return self._baseline_std

    @property
    def is_calibrated(self) -> bool:
        """Check if baseline is calibrated."""
        return self._is_calibrated

    def reset(self):
        """Reset all calibration data."""
        self._resting_data.clear()
        self._baseline_hr = None
        self._baseline_std = None
        self._is_calibrated = False
        self._time_resting_s = 0.0
        self._last_rest_start_ms = None
        self._last_update_ms = None
