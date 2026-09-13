"""The calibrated window must keep the promise it prints.

The window this replaces was chosen, not measured: two laps either side, with a
probability summed from a softmax whose temperature was picked because it was
dimensionally sensible. exp30 measured what that was worth -- 31.9% claimed
against 24.0% delivered. These tests hold the replacement to the property that
made it worth building.
"""

from __future__ import annotations

import numpy as np
import pytest

from tyremind.models.pit_decision import PitRecommendation, PitWindowCalibrator


def recommendation(lap: int = 30) -> PitRecommendation:
    return PitRecommendation(lap, 0.1, {lap: 0.1}, {lap: 1.0})


class TestCoverage:
    @pytest.mark.parametrize("target", [0.5, 0.8, 0.9])
    def test_held_out_coverage_reaches_the_target(self, target):
        """The property the whole exercise exists for.

        Fitted on one half, measured on the other, so this is the guarantee and
        not an in-sample restatement of it.
        """
        rng = np.random.default_rng(7)
        # Skewed errors on purpose: real teams pit earlier than degradation cost
        # implies, for track position we do not model, and a method that only
        # works on symmetric noise would be no use here.
        recommended = rng.integers(20, 45, 600).astype(float)
        actual = recommended + rng.gamma(2.0, 2.0, 600) * rng.choice([-1, 1], 600, p=[0.7, 0.3])

        fitted = PitWindowCalibrator(target).fit(recommended[:300], actual[:300])
        miss = np.abs(recommended[300:] - actual[300:])
        covered = float(np.mean(miss <= fitted._half_width))
        # Conformal guarantees at least the target; it may exceed it.
        assert covered >= target - 0.06, f"claimed {target:.0%}, delivered {covered:.0%}"

    def test_a_higher_target_never_gives_a_narrower_window(self):
        rng = np.random.default_rng(11)
        rec = rng.integers(20, 45, 400).astype(float)
        act = rec + rng.normal(0, 6, 400)
        widths = [PitWindowCalibrator(t).fit(rec, act)._half_width for t in (0.5, 0.8, 0.9)]
        assert widths == sorted(widths)


class TestRefusal:
    def test_too_few_stops_to_represent_the_target_returns_no_window(self):
        """`conformal_quantile` returns infinity when the sample cannot represent
        the quantile, and an infinite window is not a recommendation. Refusing
        matches the optimiser's behaviour on a flat cost curve."""
        calibrator = PitWindowCalibrator(0.99).fit(np.array([30.0, 31.0]), np.array([33.0, 29.0]))
        assert not calibrator.fitted
        assert calibrator.window(recommendation(), final_lap=60) is None

    def test_a_declined_recommendation_gets_no_window(self):
        calibrator = PitWindowCalibrator(0.8).fit(
            np.arange(30, 130, dtype=float), np.arange(30, 130, dtype=float) + 3.0)
        declined = PitRecommendation(30, 0.0, {}, {}, reason="degradation does not decide this")
        assert calibrator.window(declined, final_lap=60) is None


class TestBounds:
    def test_the_window_is_clipped_to_the_race(self):
        calibrator = PitWindowCalibrator(0.8).fit(
            np.arange(30, 130, dtype=float), np.arange(30, 130, dtype=float) + 12.0)
        window = calibrator.window(recommendation(lap=5), final_lap=20)
        assert window is not None
        assert window.low >= 1
        assert window.high <= 20

    def test_contains_agrees_with_the_bounds(self):
        calibrator = PitWindowCalibrator(0.8).fit(
            np.arange(30, 130, dtype=float), np.arange(30, 130, dtype=float) + 4.0)
        window = calibrator.window(recommendation(lap=30), final_lap=60)
        assert window.contains(window.low) and window.contains(window.high)
        assert not window.contains(window.low - 1)

    def test_an_invalid_target_is_refused_at_construction(self):
        for bad in (0.0, 1.0, -0.5, 1.5):
            with pytest.raises(ValueError, match="target_coverage"):
                PitWindowCalibrator(bad)
