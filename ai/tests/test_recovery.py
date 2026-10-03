"""Tests for the heart rate recovery model."""
import numpy as np
import pytest
from ai.recovery import fit_tau, hr_drop_60s, RecoveryModel, compute_percentile
from ai.sim_hr import simulate


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


class TestRecoveryModel:
    """Tests for the live RecoveryModel class."""

    def test_tiny_rise_never_triggers(self):
        """Model should not trigger on tiny HR rises."""
        model = RecoveryModel()
        data = simulate("tiny_rise", seed=42)

        for t_ms, hr, activity in data:
            output = model.update(t_ms, hr, activity, baseline_hr=70.0, baseline_std=3.0)

        # Should never enter active state
        assert output.episode_state == "idle"
        assert output.recovery_tau_s is None

    def test_requires_min_moving_duration(self):
        """Model should not trigger if movement < MIN_MOVING_S."""
        model = RecoveryModel()

        # Brief movement (< 30s)
        baseline = 70
        for i in range(5):  # 10 seconds of rest
            model.update(i * 2000, baseline, "resting", baseline, 3.0)

        for i in range(5, 12):  # 14 seconds of movement
            model.update(i * 2000, 95, "moving", baseline, 3.0)

        # Return to rest
        output = model.update(12 * 2000, 93, "resting", baseline, 3.0)

        # Should not trigger (only 14s of movement)
        assert output.episode_state == "idle"

    def test_fast_recovery_normal_verdict(self):
        """Fast scenario should produce normal verdict and set reference."""
        model = RecoveryModel()
        data = simulate("fast", seed=42)

        final_output = None
        for t_ms, hr, activity in data:
            output = model.update(t_ms, hr, activity, baseline_hr=70.0, baseline_std=3.0)
            if output.episode_state in ["complete", "aborted"]:
                final_output = output

        assert final_output is not None, "Episode should complete"
        assert final_output.recovery_verdict == "normal"
        assert model.reference_tau is not None, "Should set reference tau"
        assert model.last_verdict == "normal"

    def test_slow_recovery_slow_verdict(self):
        """Slow scenario should produce slow or very_slow verdict."""
        model = RecoveryModel()
        data = simulate("slow", seed=42)

        final_output = None
        for t_ms, hr, activity in data:
            output = model.update(t_ms, hr, activity, baseline_hr=70.0, baseline_std=3.0)
            if output.episode_state in ["complete", "aborted"]:
                final_output = output

        assert final_output is not None, "Episode should complete"
        assert final_output.recovery_verdict in ["slow", "very_slow"]
        assert model.last_verdict in ["slow", "very_slow"]

    def test_aborted_episode(self):
        """Aborted scenario should show aborted state and no verdict."""
        model = RecoveryModel()
        data = simulate("aborted", seed=42)

        aborted_output = None
        for t_ms, hr, activity in data:
            output = model.update(t_ms, hr, activity, baseline_hr=70.0, baseline_std=3.0)
            if output.episode_state == "aborted":
                aborted_output = output
                break

        assert aborted_output is not None, "Episode should be aborted"
        assert aborted_output.episode_state == "aborted"
        assert aborted_output.recovery_verdict is None or model.last_verdict is None

    def test_calibration_then_slow(self):
        """First recovery sets reference, second is slow with ratio ≥ 1.5."""
        model = RecoveryModel()
        data = simulate("calibration_then_slow", seed=42)

        first_complete = None
        second_complete = None

        for t_ms, hr, activity in data:
            output = model.update(t_ms, hr, activity, baseline_hr=70.0, baseline_std=3.0)

            if output.episode_state == "complete":
                if first_complete is None:
                    first_complete = output
                else:
                    second_complete = output

        assert first_complete is not None, "First episode should complete"
        assert second_complete is not None, "Second episode should complete"

        # First should set reference
        assert model.reference_tau is not None

        # Second should have ratio ≥ 1.5
        assert second_complete.recovery_ratio is not None
        assert second_complete.recovery_ratio >= 1.5, \
            f"Ratio {second_complete.recovery_ratio} should be ≥ 1.5"
        assert second_complete.recovery_verdict in ["slow", "very_slow"]

    def test_meds_keyword_matching(self):
        """Model should detect HR-altering medications."""
        # Test with beta blocker
        model_meds = RecoveryModel(medications=["metoprolol", "aspirin"])
        assert model_meds._group in ["meds", "all"]  # "meds" if n ≥ 30, else "all"

        # Test with case-insensitive match
        model_meds2 = RecoveryModel(medications=["METOPROLOL"])
        assert model_meds2._group in ["meds", "all"]

        # Test without meds
        model_no_meds = RecoveryModel(medications=["aspirin", "statin"])
        assert model_no_meds._group in ["no_meds", "all"]

    def test_missing_cutoffs_fallback(self):
        """Model should work with missing cutoffs file."""
        # Use a non-existent path
        model = RecoveryModel(cutoffs_path="nonexistent/path.json")

        # Should still work with fallback cutoffs
        data = simulate("fast", seed=42)

        for t_ms, hr, activity in data:
            output = model.update(t_ms, hr, activity, baseline_hr=70.0, baseline_std=3.0)

        # Should not crash
        assert model is not None

    def test_handles_none_hr(self):
        """Model should handle None HR gracefully."""
        model = RecoveryModel()

        # Send updates with None HR
        for i in range(100):
            output = model.update(i * 2000, None, "resting", 70.0, 3.0)

        # Should not crash
        assert output.episode_state == "idle"

    def test_handles_nan_hr(self):
        """Model should handle NaN HR gracefully."""
        model = RecoveryModel()

        # Send updates with NaN HR
        for i in range(100):
            output = model.update(i * 2000, float('nan'), "resting", 70.0, 3.0)

        # Should not crash
        assert output.episode_state == "idle"

    def test_handles_missing_baseline(self):
        """Model should handle missing baseline gracefully."""
        model = RecoveryModel()
        data = simulate("fast", seed=42)

        # Run without baseline
        for t_ms, hr, activity in data[:50]:
            output = model.update(t_ms, hr, activity, None, None)

        # Should not crash, and not trigger
        assert output.episode_state == "idle"

    def test_percentile_monotonic(self):
        """Percentile should be monotonically increasing with tau."""
        p25, p50, p75, p90 = 30, 45, 70, 100

        tau_values = [10, 25, 30, 40, 45, 55, 70, 80, 100, 120, 150]
        percentiles = [compute_percentile(tau, p25, p50, p75, p90) for tau in tau_values]

        # Check monotonicity
        for i in range(len(percentiles) - 1):
            assert percentiles[i] <= percentiles[i+1], \
                f"Percentile not monotonic: {percentiles[i]} > {percentiles[i+1]}"

    def test_set_reference_tau(self):
        """Model should allow manual reference tau setting."""
        model = RecoveryModel()

        assert model.reference_tau is None

        model.set_reference_tau(50.0)

        assert model.reference_tau == 50.0


class TestPercentileComputation:
    """Tests for percentile interpolation."""

    def test_percentile_at_known_points(self):
        """Percentile should match at known points."""
        p25, p50, p75, p90 = 30, 45, 70, 100

        assert abs(compute_percentile(30, p25, p50, p75, p90) - 25) < 1
        assert abs(compute_percentile(45, p25, p50, p75, p90) - 50) < 1
        assert abs(compute_percentile(70, p25, p50, p75, p90) - 75) < 1
        assert abs(compute_percentile(100, p25, p50, p75, p90) - 90) < 1

    def test_percentile_bounds(self):
        """Percentile should be clipped to [0, 100]."""
        p25, p50, p75, p90 = 30, 45, 70, 100

        # Very low value
        assert compute_percentile(1, p25, p50, p75, p90) == 0.0

        # Very high value
        assert compute_percentile(1000, p25, p50, p75, p90) == 100.0
