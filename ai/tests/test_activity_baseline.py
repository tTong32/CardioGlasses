"""Tests for activity detection and baseline tracking modules."""
import numpy as np
import pytest
from ai.activity import ActivityDetector
from ai.baseline import BaselineTracker


class TestActivityDetector:
    """Tests for the ActivityDetector class."""

    def test_init_default_params(self):
        """ActivityDetector should initialize with default params."""
        detector = ActivityDetector()
        assert detector.current_activity == "resting"

    def test_stillness_detection(self):
        """ActivityDetector should detect stillness."""
        detector = ActivityDetector()

        # Send 200 samples (2 seconds at 10ms) with minimal movement
        for i in range(200):
            output = detector.update(0.0, 0.0, 9.81)  # gravity only

        assert output.activity == "resting"
        assert output.accel_std < 0.05

    def test_movement_detection(self):
        """ActivityDetector should detect movement."""
        detector = ActivityDetector()

        # Send samples with significant variation
        rng = np.random.RandomState(42)
        for i in range(200):
            ax = rng.normal(0, 2)
            ay = rng.normal(0, 2)
            az = rng.normal(9.81, 2)
            output = detector.update(ax, ay, az)

        assert output.activity == "moving"
        assert output.accel_std > 0.05

    def test_transition_to_moving(self):
        """ActivityDetector should transition from rest to moving."""
        detector = ActivityDetector()

        # Start with stillness
        for i in range(200):
            detector.update(0.0, 0.0, 9.81)

        output = detector.update(0.0, 0.0, 9.81)
        assert output.activity == "resting"

        # Add movement
        for i in range(200):
            detector.update(
                0.0 + np.sin(i * 0.1) * 3,
                0.0,
                9.81 + np.cos(i * 0.1) * 3
            )

        output = detector.update(
            0.0 + np.sin(200 * 0.1) * 3,
            0.0,
            9.81 + np.cos(200 * 0.1) * 3
        )
        assert output.activity == "moving"

    def test_handles_any_orientation(self):
        """ActivityDetector should work regardless of device orientation."""
        detector = ActivityDetector()

        # Test different orientations (all still)
        orientations = [
            (0.0, 0.0, 9.81),   # upright
            (0.0, 9.81, 0.0),   # on side
            (9.81, 0.0, 0.0),   # rotated
            (0.0, 0.0, -9.81),  # upside down
        ]

        for ax, ay, az in orientations:
            detector = ActivityDetector()
            for i in range(200):
                output = detector.update(ax, ay, az)

            # All should be resting (no variation)
            assert output.activity == "resting"


class TestBaselineTracker:
    """Tests for the BaselineTracker class."""

    def test_init_not_calibrated(self):
        """BaselineTracker should start uncalibrated."""
        tracker = BaselineTracker()
        output = tracker.update(0, 70.0, "resting")

        assert not output.is_calibrated
        assert output.baseline_hr is None
        assert output.baseline_std is None

    def test_requires_continuous_rest(self):
        """BaselineTracker should require 60s continuous rest."""
        tracker = BaselineTracker()

        # Send 50 seconds of rest
        for i in range(50):
            t_ms = i * 1000
            output = tracker.update(t_ms, 70.0, "resting")

        # Not calibrated yet (need 60s)
        assert not output.is_calibrated

        # Send 20 more seconds
        for i in range(50, 70):
            t_ms = i * 1000
            output = tracker.update(t_ms, 70.0, "resting")

        # Should be calibrated now
        assert output.is_calibrated
        assert output.baseline_hr is not None

    def test_movement_resets_rest_timer(self):
        """Movement should reset the continuous rest timer."""
        tracker = BaselineTracker()

        # Rest for 50 seconds
        for i in range(50):
            t_ms = i * 1000
            tracker.update(t_ms, 70.0, "resting")

        # Move briefly
        tracker.update(50000, 80.0, "moving")

        # Rest for another 50 seconds
        for i in range(51, 101):
            t_ms = i * 1000
            output = tracker.update(t_ms, 70.0, "resting")

        # Should not be calibrated (rest timer reset by movement)
        assert not output.is_calibrated

    def test_calibration_with_noise(self):
        """BaselineTracker should calibrate with noisy HR data."""
        tracker = BaselineTracker()
        rng = np.random.RandomState(42)

        # Send 70 seconds of resting HR with ±3 bpm noise
        baseline_true = 72.0
        for i in range(70):
            t_ms = i * 1000
            hr = baseline_true + rng.normal(0, 3)
            output = tracker.update(t_ms, hr, "resting")

        # Should be calibrated
        assert output.is_calibrated
        assert output.baseline_hr is not None

        # Should be close to true baseline
        assert abs(output.baseline_hr - baseline_true) < 2.0

    def test_outlier_rejection(self):
        """BaselineTracker should reject HR outliers."""
        tracker = BaselineTracker()

        # Send 70 seconds with some outliers
        for i in range(70):
            t_ms = i * 1000

            if i in [10, 20, 30]:
                # Inject outliers
                hr = 120.0
            else:
                hr = 70.0

            output = tracker.update(t_ms, hr, "resting")

        # Should be calibrated and close to 70 (outliers removed)
        assert output.is_calibrated
        assert output.baseline_hr is not None
        assert abs(output.baseline_hr - 70.0) < 3.0

    def test_handles_none_hr(self):
        """BaselineTracker should handle None HR gracefully."""
        tracker = BaselineTracker()

        # Send None HR values
        for i in range(100):
            t_ms = i * 1000
            output = tracker.update(t_ms, None, "resting")

        # Should not crash
        assert not output.is_calibrated
        assert output.time_resting_s == 0.0

    def test_handles_nan_hr(self):
        """BaselineTracker should handle NaN HR gracefully."""
        tracker = BaselineTracker()

        # Send NaN HR values
        for i in range(100):
            t_ms = i * 1000
            output = tracker.update(t_ms, float('nan'), "resting")

        # Should not crash
        assert not output.is_calibrated
        assert output.time_resting_s == 0.0

    def test_rolling_window(self):
        """BaselineTracker should use rolling 3-minute window."""
        tracker = BaselineTracker(window_s=60)  # Use 1-minute window for faster test

        # Send 2 minutes of HR = 70
        for i in range(120):
            t_ms = i * 1000
            tracker.update(t_ms, 70.0, "resting")

        # Check calibration (should be ~70)
        output = tracker.update(120000, 70.0, "resting")
        baseline_1 = output.baseline_hr

        # Send another minute of HR = 80 (window shifts)
        for i in range(121, 181):
            t_ms = i * 1000
            output = tracker.update(t_ms, 80.0, "resting")

        # Baseline should have shifted toward 80
        baseline_2 = output.baseline_hr
        assert baseline_2 > baseline_1

    def test_reset(self):
        """BaselineTracker.reset() should clear all state."""
        tracker = BaselineTracker()

        # Calibrate
        for i in range(70):
            t_ms = i * 1000
            tracker.update(t_ms, 70.0, "resting")

        assert tracker.is_calibrated

        # Reset
        tracker.reset()

        assert not tracker.is_calibrated
        assert tracker.baseline_hr is None
        assert tracker.baseline_std is None

    def test_properties(self):
        """BaselineTracker properties should match output."""
        tracker = BaselineTracker()

        # Calibrate
        for i in range(70):
            t_ms = i * 1000
            output = tracker.update(t_ms, 70.0, "resting")

        # Properties should match output
        assert tracker.baseline_hr == output.baseline_hr
        assert tracker.baseline_std == output.baseline_std
        assert tracker.is_calibrated == output.is_calibrated


class TestIntegration:
    """Integration tests for activity + baseline together."""

    def test_full_calibration_sequence(self):
        """Test full sequence: stillness detection → baseline calibration."""
        activity = ActivityDetector()
        baseline = BaselineTracker()

        # Simulate 2 minutes of rest with 10ms sampling
        t_ms = 0
        for i in range(12000):  # 120 seconds * 100 samples/sec
            # Still accelerometer
            activity_output = activity.update(0.0, 0.0, 9.81)

            # Stable HR with noise
            hr = 72.0 + np.random.normal(0, 2)

            # Update baseline every 100ms (10 samples)
            if i % 10 == 0:
                baseline_output = baseline.update(
                    t_ms,
                    hr,
                    activity_output.activity
                )
                t_ms += 100

        # Should detect resting
        assert activity_output.activity == "resting"

        # Should be calibrated
        assert baseline_output.is_calibrated
        assert baseline_output.baseline_hr is not None
        assert abs(baseline_output.baseline_hr - 72.0) < 3.0

    def test_movement_prevents_calibration(self):
        """Movement should prevent baseline calibration."""
        activity = ActivityDetector()
        baseline = BaselineTracker()

        # Simulate 2 minutes with intermittent movement
        t_ms = 0
        rng = np.random.RandomState(42)

        for i in range(12000):
            # Alternate between still and moving
            if (i // 1000) % 2 == 0:
                # Still
                ax, ay, az = 0.0, 0.0, 9.81
            else:
                # Moving
                ax = rng.normal(0, 2)
                ay = rng.normal(0, 2)
                az = rng.normal(9.81, 2)

            activity_output = activity.update(ax, ay, az)

            hr = 75.0
            if i % 10 == 0:
                baseline_output = baseline.update(
                    t_ms,
                    hr,
                    activity_output.activity
                )
                t_ms += 100

        # Should not be calibrated (not enough continuous rest)
        assert not baseline_output.is_calibrated
