"""PPG and IMU feature stubs.

TODO: real signal processing. Low quality must be reported honestly.
"""

from typing import Literal

from ai.contracts import Sample

Activity = Literal["resting", "moving"]


def compute_hr(samples: list[Sample]) -> float | None:
    """Estimate heart rate in beats per minute from a PPG window.

    TODO: detect beats or take a spectral peak on `ppg`.
    """
    return None


def classify_activity(samples: list[Sample]) -> Activity | None:
    """Label the window resting or moving from the IMU.

    TODO: use acceleration variance and gyro energy.
    """
    return None


def signal_quality(samples: list[Sample]) -> float | None:
    """Score PPG quality from 0 to 1.

    TODO: estimate SNR or perfusion. Do not invent a high score.
    """
    return None
