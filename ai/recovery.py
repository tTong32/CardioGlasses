"""Heart rate recovery model for CardioGlasses.

Fits exponential decay curves to recovery data and classifies recovery speed
against population percentiles and personal baseline.
"""
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json
import warnings
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


# ============================================================================
# LIVE MODEL
# ============================================================================


@dataclass
class RecoveryOutput:
    """Output from the live recovery model."""
    recovery_tau_s: Optional[float] = None
    hr_drop_60s: Optional[float] = None
    recovery_ratio: Optional[float] = None  # tau / personal reference tau
    recovery_percentile: Optional[float] = None  # 0-100 vs the trained group
    recovery_verdict: Optional[str] = None  # "normal" | "slow" | "very_slow"
    recovery_r2: Optional[float] = None
    episode_state: str = "idle"  # "idle" | "active" | "complete" | "aborted"


class RecoveryModel:
    """Live heart rate recovery model for real-time monitoring."""

    def __init__(
        self,
        cutoffs_path: str = "ai/model/recovery_cutoffs.json",
        medications: Optional[list[str]] = None
    ):
        """Initialize the recovery model.

        Args:
            cutoffs_path: path to trained recovery cutoffs JSON
            medications: list of medication names to check for HR-altering drugs
        """
        self._cutoffs = self._load_cutoffs(cutoffs_path)
        self._medications = medications or []
        self._group = self._select_group()

        # Episode tracking
        self._state = "idle"
        self._episode_data: list[tuple[int, float, str]] = []  # (t_ms, hr, activity)
        self._moving_start_ms: Optional[int] = None
        self._peak_hr: float = 0.0
        self._recovery_start_ms: Optional[int] = None
        self._hr0: Optional[float] = None
        self._hr0_estimate: Optional[float] = None  # from around the stop; refined by the first recovery HR
        self._baseline: Optional[float] = None
        self._baseline_std: Optional[float] = None

        # Recovery fit tracking
        self._recovery_times: list[float] = []  # seconds since recovery start
        self._recovery_hrs: list[float] = []
        self._current_fit: Optional[FitResult] = None
        self._current_hr_drop_60s: Optional[float] = None

        # Personal reference
        self._reference_tau: Optional[float] = None
        self._last_verdict: Optional[str] = None

        # For near-baseline tracking
        self._near_baseline_start_ms: Optional[int] = None

    def _load_cutoffs(self, path: str) -> dict:
        """Load cutoffs from JSON file, with fallback to hardcoded values."""
        cutoffs_file = Path(path)

        if not cutoffs_file.exists():
            warnings.warn(f"Cutoffs file not found at {path}, using fallback values")
            return {
                "version": 0,
                "source": "fallback",
                "created": "",
                "groups": {
                    "all": {
                        "n": 100,
                        "tau_p25": 30,
                        "tau_p50": 45,
                        "tau_p75": 70,
                        "tau_p90": 100,
                        "hr_drop_60s_p10": 10,
                        "hr_drop_60s_p25": 15,
                        "hr_drop_60s_p50": 20,
                        "median_age": 70
                    }
                }
            }

        with open(cutoffs_file) as f:
            return json.load(f)

    def _select_group(self) -> str:
        """Select the appropriate cutoff group based on medications."""
        # Check if any medication keywords match
        meds_flag = False
        for med in self._medications:
            med_lower = med.lower()
            for keyword in MEDS_KEYWORDS:
                if keyword in med_lower:
                    meds_flag = True
                    break
            if meds_flag:
                break

        # Select group
        if meds_flag and "meds" in self._cutoffs["groups"]:
            group = self._cutoffs["groups"]["meds"]
            if group["n"] >= MIN_GROUP_N:
                return "meds"

        if not meds_flag and "no_meds" in self._cutoffs["groups"]:
            group = self._cutoffs["groups"]["no_meds"]
            if group["n"] >= MIN_GROUP_N:
                return "no_meds"

        # Fallback to "all"
        return "all"

    def update(
        self,
        t_ms: int,
        hr: Optional[float],
        activity: str,
        baseline_hr: Optional[float],
        baseline_std: Optional[float]
    ) -> RecoveryOutput:
        """Process a new reading and update recovery tracking.

        Args:
            t_ms: timestamp in milliseconds
            hr: heart rate in bpm, or None if missing
            activity: "resting" or "moving"
            baseline_hr: resting baseline HR, or None if not calibrated yet
            baseline_std: std of resting baseline, or None if not calibrated yet

        Returns:
            RecoveryOutput with current recovery metrics
        """
        # Handle invalid inputs gracefully
        if hr is not None and (np.isnan(hr) or hr <= 0):
            hr = None

        if baseline_hr is not None and (np.isnan(baseline_hr) or baseline_hr <= 0):
            baseline_hr = None

        if baseline_std is not None and np.isnan(baseline_std):
            baseline_std = None

        # Store reading
        self._episode_data.append((t_ms, hr, activity))

        # Trim old data (keep last 10 minutes)
        cutoff_time = t_ms - 600000  # 10 minutes
        self._episode_data = [(t, h, a) for t, h, a in self._episode_data if t >= cutoff_time]

        # Update baseline
        if baseline_hr is not None:
            self._baseline = baseline_hr
            self._baseline_std = baseline_std if baseline_std is not None else 2.0

        # State machine
        if self._state == "idle":
            return self._handle_idle(t_ms, hr, activity)
        elif self._state == "active":
            return self._handle_active(t_ms, hr, activity)
        else:  # complete or aborted
            # Reset to idle on next update
            self._state = "idle"
            return RecoveryOutput(episode_state="idle")

    def _handle_idle(self, t_ms: int, hr: Optional[float], activity: str) -> RecoveryOutput:
        """Handle updates while idle (waiting for a recovery episode)."""
        # Track moving periods
        if activity == "moving":
            if self._moving_start_ms is None:
                self._moving_start_ms = t_ms

            # Update peak HR (last 10 seconds)
            if hr is not None:
                recent_hrs = [h for t, h, a in self._episode_data
                             if t >= t_ms - 10000 and h is not None and a == "moving"]
                if recent_hrs:
                    self._peak_hr = max(recent_hrs)

        else:  # resting
            # Check if we just stopped moving → potential recovery trigger
            if self._moving_start_ms is not None:
                moving_duration_s = (t_ms - self._moving_start_ms) / 1000

                # Check trigger conditions
                if (moving_duration_s >= MIN_MOVING_S and
                    self._baseline is not None and
                    self._peak_hr >= self._baseline + MIN_RISE_BPM):

                    # Trigger recovery episode
                    self._state = "active"
                    self._recovery_start_ms = t_ms
                    self._recovery_times = []
                    self._recovery_hrs = []
                    self._current_fit = None
                    self._current_hr_drop_60s = None
                    self._near_baseline_start_ms = None

                    # Compute HR0: median HR within ±HR0_WINDOW_S of stop time
                    window_ms = HR0_WINDOW_S * 1000
                    hr0_readings = [h for t, h, a in self._episode_data
                                   if abs(t - t_ms) <= window_ms and h is not None]

                    # HR0 is settled by the first clean recovery reading (see _handle_active).
                    # Motion often corrupts HR at the stop itself, and the readings or peak
                    # from just before it can sit below the HR seen just after stopping.
                    self._hr0 = None
                    self._hr0_estimate = float(np.median(hr0_readings)) if hr0_readings else self._peak_hr

            # Reset moving tracker
            self._moving_start_ms = None
            self._peak_hr = 0.0

        return RecoveryOutput(episode_state="idle")

    def _handle_active(self, t_ms: int, hr: Optional[float], activity: str) -> RecoveryOutput:
        """Handle updates during an active recovery episode."""
        # Check for abortion (movement detected)
        if activity == "moving":
            self._state = "aborted"
            self._last_verdict = None
            return RecoveryOutput(episode_state="aborted")

        if self._hr0 is None and hr is not None:
            # A decay can't start below its own first point.
            self._hr0 = max(self._hr0_estimate or 0.0, hr)

        # Add recovery reading
        if hr is not None and self._recovery_start_ms is not None:
            t_recovery_s = (t_ms - self._recovery_start_ms) / 1000
            self._recovery_times.append(t_recovery_s)
            self._recovery_hrs.append(hr)

        # Check for completion conditions
        elapsed_s = (t_ms - self._recovery_start_ms) / 1000 if self._recovery_start_ms else 0

        # 1. Maximum duration
        if elapsed_s >= MAX_EPISODE_S:
            return self._complete_episode("complete")

        # 2. Near baseline for sufficient time
        if hr is not None and self._baseline is not None and self._baseline_std is not None:
            threshold = self._baseline + max(self._baseline_std, 2)

            if hr <= threshold:
                if self._near_baseline_start_ms is None:
                    self._near_baseline_start_ms = t_ms
                elif (t_ms - self._near_baseline_start_ms) / 1000 >= END_NEAR_BASELINE_S:
                    return self._complete_episode("complete")
            else:
                self._near_baseline_start_ms = None

        # Compute recovery metrics if we have enough data
        if len(self._recovery_times) < MIN_FIT_S / 2:  # Need at least MIN_FIT_S seconds
            return RecoveryOutput(episode_state="active")

        # Fit tau
        t_arr = np.array(self._recovery_times)
        hr_arr = np.array(self._recovery_hrs)

        if self._baseline is not None and self._hr0 is not None:
            self._current_fit = fit_tau(t_arr, hr_arr, self._baseline, self._hr0)
            self._current_hr_drop_60s = hr_drop_60s(t_arr, hr_arr, self._hr0)

        # Build output
        return self._build_output("active")

    def _complete_episode(self, final_state: str) -> RecoveryOutput:
        """Complete the current episode and return final output."""
        output = self._build_output(final_state)

        # Check if this recovery should become the personal reference
        if (self._reference_tau is None and
            self._current_fit is not None and
            self._current_fit.r2 is not None and
            self._current_fit.r2 >= REF_MIN_R2 and
            output.recovery_verdict == "normal"):

            self._reference_tau = self._current_fit.tau_s

        # Store verdict
        self._last_verdict = output.recovery_verdict

        # Transition to complete/aborted state
        self._state = final_state

        return output

    def _build_output(self, state: str) -> RecoveryOutput:
        """Build RecoveryOutput from current fit results."""
        if self._current_fit is None or self._current_fit.tau_s is None:
            return RecoveryOutput(episode_state=state)

        tau = self._current_fit.tau_s
        r2 = self._current_fit.r2

        # Only provide verdict if quality is sufficient
        if r2 is None or r2 < MIN_R2:
            return RecoveryOutput(
                recovery_tau_s=tau,
                recovery_r2=r2,
                hr_drop_60s=self._current_hr_drop_60s,
                episode_state=state
            )

        # Compute percentile
        group = self._cutoffs["groups"][self._group]
        percentile = compute_percentile(
            tau,
            group["tau_p25"],
            group["tau_p50"],
            group["tau_p75"],
            group["tau_p90"]
        )

        # Compute ratio if reference exists
        ratio = None
        if self._reference_tau is not None and self._reference_tau > 0:
            ratio = tau / self._reference_tau

        # Determine verdict
        verdict_pop = self._population_verdict(tau, group)
        verdict_personal = self._personal_verdict(ratio) if ratio is not None else None

        # Final verdict = worse of the two
        if verdict_personal is None:
            verdict = verdict_pop
        elif verdict_pop == "very_slow" or verdict_personal == "very_slow":
            verdict = "very_slow"
        elif verdict_pop == "slow" or verdict_personal == "slow":
            verdict = "slow"
        else:
            verdict = "normal"

        return RecoveryOutput(
            recovery_tau_s=tau,
            hr_drop_60s=self._current_hr_drop_60s,
            recovery_ratio=ratio,
            recovery_percentile=percentile,
            recovery_verdict=verdict,
            recovery_r2=r2,
            episode_state=state
        )

    def _population_verdict(self, tau: float, group: dict) -> str:
        """Classify tau against population percentiles."""
        if tau > group["tau_p90"]:
            return "very_slow"
        elif tau > group["tau_p75"]:
            return "slow"
        else:
            return "normal"

    def _personal_verdict(self, ratio: float) -> str:
        """Classify tau against personal reference."""
        if ratio > RATIO_VERY_SLOW:
            return "very_slow"
        elif ratio >= RATIO_SLOW:
            return "slow"
        else:
            return "normal"

    @property
    def reference_tau(self) -> Optional[float]:
        """Get the personal reference tau, or None if not set."""
        return self._reference_tau

    def set_reference_tau(self, tau: float) -> None:
        """Manually set the personal reference tau."""
        self._reference_tau = tau

    @property
    def last_verdict(self) -> Optional[str]:
        """Get the last recovery verdict, persists after episode ends."""
        return self._last_verdict


def compute_percentile(value: float, p25: float, p50: float, p75: float, p90: float) -> float:
    """Interpolate a value's percentile given key percentiles.

    Uses piecewise linear interpolation:
    - p0 = 0.5 * p25 (extrapolated)
    - p100 = 2 * p90 (extrapolated)

    Args:
        value: the value to compute percentile for
        p25, p50, p75, p90: known percentiles

    Returns:
        Percentile from 0-100
    """
    p0 = 0.5 * p25
    p100 = 2 * p90

    # Piecewise linear interpolation
    if value <= p0:
        return 0.0
    elif value <= p25:
        return 0 + 25 * (value - p0) / (p25 - p0)
    elif value <= p50:
        return 25 + 25 * (value - p25) / (p50 - p25)
    elif value <= p75:
        return 50 + 25 * (value - p50) / (p75 - p50)
    elif value <= p90:
        return 75 + 15 * (value - p75) / (p90 - p75)
    elif value <= p100:
        return 90 + 10 * (value - p90) / (p100 - p90)
    else:
        return 100.0
