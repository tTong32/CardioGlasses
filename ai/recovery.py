"""Heart rate recovery model for CardioGlasses.

Fits exponential decay curves to recovery data and classifies recovery speed
against population percentiles and personal baseline.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.optimize import curve_fit


# ============================================================================
# TUNABLE CONSTANTS
# ============================================================================

# Episode detection
MIN_MOVING_S = 30  # movement needed before a recovery counts
MIN_RISE_BPM = 10  # HR must rise this much above baseline
MIN_FIT_S = 20  # seconds of data before the first fit
HR0_WINDOW_S = 3  # HR0 = median HR within ±3 s of the stop

# Fit quality
TAU_BOUNDS = (5, 600)  # tau must be in this range (seconds)
MIN_R2 = 0.5  # below this, the fit is unreliable
REF_MIN_R2 = 0.7  # quality needed to become the personal reference

# Verdict thresholds
RATIO_SLOW = 1.5  # personal tau / reference tau
RATIO_VERY_SLOW = 2.5

# Episode completion
END_NEAR_BASELINE_S = 20  # HR within baseline + 1 std for this long → complete
MAX_EPISODE_S = 300  # maximum episode duration

# Group selection
MIN_GROUP_N = 30  # fall back to "all" group below this
MEDS_KEYWORDS = [
    "metoprolol", "bisoprolol", "carvedilol", "atenolol", "propranolol",
    "nebivolol", "diltiazem", "verapamil", "digoxin", "amiodarone",
    "ivabradine", "beta blocker"
]

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class FitResult:
    """Result of fitting tau to a recovery curve."""
    tau_s: Optional[float]  # fitted time constant in seconds, or None if fit failed
    r2: Optional[float]  # coefficient of determination
    n_points: int  # number of data points used
    at_bound: bool  # True if tau hit the upper or lower bound


# ============================================================================
# CORE FIT FUNCTIONS
# ============================================================================


def fit_tau(
    t_s: np.ndarray,
    hr: np.ndarray,
    baseline: float,
    hr0: float
) -> FitResult:
    """Fit exponential decay tau to recovery data.

    Model: hr = baseline + (hr0 - baseline) * exp(-t / tau)

    Args:
        t_s: time in seconds since recovery started (0 = stop time)
        hr: heart rate at each time point
        baseline: resting heart rate
        hr0: heart rate at the moment of stopping (t=0)

    Returns:
        FitResult with tau, r2, n_points, and at_bound flag
    """
    # Validate inputs
    if len(t_s) < 5:
        return FitResult(tau_s=None, r2=None, n_points=len(t_s), at_bound=False)

    rise = hr0 - baseline
    if rise < 3:
        return FitResult(tau_s=None, r2=None, n_points=len(t_s), at_bound=False)

    # Define exponential decay model
    def model(t: np.ndarray, tau: float) -> np.ndarray:
        return baseline + rise * np.exp(-t / tau)

    # Fit tau with scipy
    try:
        (tau_fit,), _ = curve_fit(
            model,
            t_s,
            hr,
            p0=[60],  # initial guess: 60 seconds
            bounds=([TAU_BOUNDS[0]], [TAU_BOUNDS[1]]),
            maxfev=1000
        )
    except (RuntimeError, ValueError):
        return FitResult(tau_s=None, r2=None, n_points=len(t_s), at_bound=False)

    # Compute R²
    predictions = model(t_s, tau_fit)
    ss_res = np.sum((hr - predictions) ** 2)
    ss_tot = np.sum((hr - np.mean(hr)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Check if at bounds (within 1%)
    at_bound = (
        abs(tau_fit - TAU_BOUNDS[0]) < 0.01 * TAU_BOUNDS[0] or
        abs(tau_fit - TAU_BOUNDS[1]) < 0.01 * TAU_BOUNDS[1]
    )

    return FitResult(tau_s=tau_fit, r2=r2, n_points=len(t_s), at_bound=at_bound)


def hr_drop_60s(t_s: np.ndarray, hr: np.ndarray, hr0: float) -> Optional[float]:
    """Compute HR drop from hr0 to median HR at 60 seconds.

    Args:
        t_s: time in seconds since recovery started
        hr: heart rate at each time point
        hr0: heart rate at t=0

    Returns:
        hr0 - median(HR in window [57, 63] seconds), or None if no data in that window
    """
    # Find readings in the 57-63 second window
    mask = (t_s >= 57) & (t_s <= 63)
    hr_60s = hr[mask]

    if len(hr_60s) == 0:
        return None

    median_60s = np.median(hr_60s)
    return hr0 - median_60s
