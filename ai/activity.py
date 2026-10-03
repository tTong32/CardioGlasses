"""Activity detection from accelerometer data for CardioGlasses.

Classifies user activity as "resting" or "moving" based on accelerometer magnitude variance.
"""
from dataclasses import dataclass
from collections import deque
import numpy as np


# ============================================================================
# TUNABLE CONSTANTS
# ============================================================================

WINDOW_SIZE_MS = 2000  # Classify activity over 2-second windows
STILLNESS_THRESHOLD = 0.05  # m/s² std - below this = resting
SAMPLE_RATE_MS = 10  # Accelerometer samples every 10ms


# ============================================================================
# ACTIVITY DETECTOR
# ============================================================================


@dataclass
class ActivityOutput:
    """Output from activity detector."""
    activity: str  # "resting" or "moving"
    accel_std: float  # standard deviation of acceleration magnitude


class ActivityDetector:
    """Real-time activity classification from accelerometer data."""

    def __init__(
        self,
        window_ms: int = WINDOW_SIZE_MS,
        stillness_threshold: float = STILLNESS_THRESHOLD
    ):
        """Initialize the activity detector.

        Args:
            window_ms: window size in milliseconds for activity classification
            stillness_threshold: accel std below this = resting (m/s²)
        """
        self._window_ms = window_ms
        self._threshold = stillness_threshold

        # Rolling window of acceleration magnitudes
        max_samples = int(window_ms / SAMPLE_RATE_MS)
        self._accel_magnitudes: deque = deque(maxlen=max_samples)

        # Current activity state
        self._current_activity = "resting"

    def update(self, ax: float, ay: float, az: float) -> ActivityOutput:
        """Process a new accelerometer reading.

        Args:
            ax: x-axis acceleration (m/s²)
            ay: y-axis acceleration (m/s²)
            az: z-axis acceleration (m/s²)

        Returns:
            ActivityOutput with current activity classification
        """
        # Compute magnitude (handles any orientation)
        magnitude = np.sqrt(ax**2 + ay**2 + az**2)

        # Add to rolling window
        self._accel_magnitudes.append(magnitude)

        # Need enough data for classification
        if len(self._accel_magnitudes) < 10:
            return ActivityOutput(
                activity=self._current_activity,
                accel_std=0.0
            )

        # Compute standard deviation over the window
        accel_std = np.std(self._accel_magnitudes)

        # Classify activity
        if accel_std < self._threshold:
            self._current_activity = "resting"
        else:
            self._current_activity = "moving"

        return ActivityOutput(
            activity=self._current_activity,
            accel_std=accel_std
        )

    @property
    def current_activity(self) -> str:
        """Get the current activity state."""
        return self._current_activity
