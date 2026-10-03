"""Tests for the heart rate recovery model."""
import numpy as np
import pytest
from ai.recovery import fit_tau, hr_drop_60s


class TestFitTau:
    """Tests for the tau fitting function."""

    def test_noiseless_recovery(self):
        """fit_tau should recover tau within 15% on noiseless data."""
        baseline = 70
        hr0 = 110
        rise = hr0 - baseline

        for tau_true in [20, 45, 90, 180]:
            # Generate noiseless decay curve
            t_s = np.arange(0, 180, 2)
            hr = baseline + rise * np.exp(-t_s / tau_true)

            result = fit_tau(t_s, hr, baseline, hr0)

            assert result.tau_s is not None, f"Fit failed for tau={tau_true}"
            assert result.r2 is not None
            assert result.r2 > 0.98, f"Poor R² for noiseless data: {result.r2}"

            # Check tau within 15%
            error = abs(result.tau_s - tau_true) / tau_true
            assert error < 0.15, f"Tau error {error:.2%} for tau={tau_true}"

    def test_noisy_recovery(self):
        """fit_tau should recover tau within 15% on noisy data."""
        baseline = 70
        hr0 = 110
        rise = hr0 - baseline

        rng = np.random.RandomState(42)

        for tau_true in [20, 45, 90, 180]:
            # Generate noisy decay curve
            t_s = np.arange(0, 180, 2)
            hr_clean = baseline + rise * np.exp(-t_s / tau_true)
            hr = hr_clean + rng.normal(0, 2, size=len(t_s))  # ±2 bpm noise

            result = fit_tau(t_s, hr, baseline, hr0)

            assert result.tau_s is not None, f"Fit failed for noisy tau={tau_true}"
            assert result.r2 is not None
            assert result.r2 > 0.85, f"Poor R² for noisy data: {result.r2}"

            # Check tau within 15%
            error = abs(result.tau_s - tau_true) / tau_true
            assert error < 0.15, f"Tau error {error:.2%} for noisy tau={tau_true}"

    def test_too_few_points(self):
        """fit_tau should return None with < 5 points."""
        baseline = 70
        hr0 = 110
        t_s = np.array([0, 2, 4, 6])
        hr = np.array([110, 100, 92, 86])

        result = fit_tau(t_s, hr, baseline, hr0)

        assert result.tau_s is None
        assert result.r2 is None
        assert result.n_points == 4

    def test_tiny_rise(self):
        """fit_tau should return None if rise < 3 bpm."""
        baseline = 70
        hr0 = 72  # only 2 bpm rise
        t_s = np.arange(0, 60, 2)
        hr = np.full(len(t_s), 71.5)

        result = fit_tau(t_s, hr, baseline, hr0)

        assert result.tau_s is None
        assert result.r2 is None

    def test_at_bound_detection(self):
        """fit_tau should detect when tau hits bounds."""
        baseline = 70
        hr0 = 110

        # Very fast decay → should hit lower bound
        t_s = np.arange(0, 60, 2)
        hr = baseline + 40 * np.exp(-t_s / 3)  # tau=3, near lower bound of 5

        result = fit_tau(t_s, hr, baseline, hr0)
        assert result.tau_s is not None
        # Should be close to lower bound
        assert result.tau_s < 10


class TestHrDrop60s:
    """Tests for the 60-second HR drop metric."""

    def test_computes_drop(self):
        """hr_drop_60s should compute HR0 - median(HR at 60s)."""
        hr0 = 110
        t_s = np.arange(0, 120, 2)
        # Simple linear decay for testing
        hr = 110 - t_s * 0.5  # drops by 30 bpm at 60s

        drop = hr_drop_60s(t_s, hr, hr0)

        assert drop is not None
        # At 60s, HR should be around 80, so drop = 110 - 80 = 30
        assert abs(drop - 30) < 2  # allow some tolerance

    def test_returns_none_before_60s(self):
        """hr_drop_60s should return None if no data at 60s."""
        hr0 = 110
        t_s = np.arange(0, 40, 2)  # only 40 seconds
        hr = 110 - t_s * 0.5

        drop = hr_drop_60s(t_s, hr, hr0)

        assert drop is None

    def test_uses_window(self):
        """hr_drop_60s should use the 57-63s window."""
        hr0 = 110
        # Create data with a spike at exactly 60s
        t_s = np.arange(0, 120, 2)
        hr = np.full(len(t_s), 80.0)
        # Make 60s reading higher
        hr[30] = 95  # t=60s

        drop = hr_drop_60s(t_s, hr, hr0)

        # Should use median of window, not just the 60s point
        assert drop is not None
        # Most readings in window are 80, so median ~80
        assert abs(drop - 30) < 3
