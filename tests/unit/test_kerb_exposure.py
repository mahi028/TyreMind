"""The kerb measurement has to be right before its null result means anything.

exp33's headline is a negative measurement claim -- that the public position feed
has no lateral coordinate in it, so lateral deviation from the racing line cannot
be measured at all. A claim like that is exactly what a broken distance function
produces by accident, and "our code said zero" is not evidence of anything. So the
geometry is pinned here against shapes whose answers are known from trigonometry
rather than from the feed: a car on a circle, a car on a slightly larger circle,
and a car that wanders wide at one corner and not the other.

The second half pins the pre-registered decision rule. exp33 declares in advance
what counts as support, and a rule that can be loosened after the numbers arrive
is not a rule. The verdict function is therefore tested against synthetic results
that sit on each side of each condition.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def kerb():
    return _load("build_kerb_exposure", "scripts/build_kerb_exposure.py")


@pytest.fixture(scope="module")
def exp33():
    return _load("exp33_kerb_exposure", "experiments/exp33_kerb_exposure.py")


def circle(n: int, radius: float, phase: float = 0.0) -> np.ndarray:
    """A closed anticlockwise loop, sampled evenly in angle.

    `phase` exists so a test can sample a circle at angles that do NOT land on the
    reference line's own points. Without it, a 400-point lap against a 2000-point
    reference lines up exactly every fifth point and every distance comes out at
    machine epsilon -- which looks like a perfect result and tests nothing.
    """
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False) + phase
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


@pytest.fixture(scope="module")
def reference(kerb):
    """A 300 m circle, built through the same path a real session takes."""
    return kerb.ReferenceLine(kerb.resample_closed_loop(circle(200, 300.0), 2000))


class TestReferenceLine:
    def test_resampling_preserves_the_shape(self, kerb):
        out = kerb.resample_closed_loop(circle(200, 300.0), 2000)
        assert len(out) == 2000
        assert np.linalg.norm(out, axis=1) == pytest.approx(300.0, abs=0.01)

    def test_resampling_removes_the_chording_bias(self, kerb):
        """The reason this uses a spline rather than straight-line interpolation.

        The position feed gives ~18 m between samples. Joining those with straight
        lines cuts every corner, and the cut is about a metre -- the same size as
        the effect being measured, appearing at corners specifically, which is
        where a false signal would be most convincing. A 24-point circle makes the
        same error large enough to see: the midpoint of each chord sits 2.6 m
        inside the arc.
        """
        coarse = circle(24, 300.0)
        closed = np.vstack([coarse, coarse[:1]])
        chord_midpoints = (closed[:-1] + closed[1:]) / 2
        assert (300.0 - np.linalg.norm(chord_midpoints, axis=1)).max() > 2.0

        splined = kerb.resample_closed_loop(coarse, 480)
        assert np.abs(np.linalg.norm(splined, axis=1) - 300).max() < 0.1

    def test_a_point_on_the_line_has_no_deviation(self, kerb, reference):
        deviation, _ = reference.deviation(circle(400, 300.0))
        assert np.abs(deviation).max() < 0.02

    def test_deviation_recovers_a_known_offset(self, kerb, reference):
        """A car on a circle 3 m wider is 3 m off the line, everywhere."""
        deviation, _ = reference.deviation(circle(400, 303.0))
        assert np.abs(deviation) == pytest.approx(3.0, abs=0.02)

    def test_the_sign_says_which_side_of_the_line(self, kerb, reference):
        """Positive is left of the direction of travel.

        Pinned because the sign is the only thing separating an apex kerb from an
        exit kerb, and a silent flip would merge the two without failing anything.
        On an anticlockwise loop, running wide is running right.
        """
        outside, _ = reference.deviation(circle(400, 303.0))
        inside, _ = reference.deviation(circle(400, 297.0))
        assert np.all(outside < 0)
        assert np.all(inside > 0)

    def test_distance_along_the_line_spans_the_lap(self, kerb, reference):
        _, along = reference.deviation(circle(400, 300.0))
        assert reference.length_m == pytest.approx(2 * np.pi * 300.0, rel=1e-3)
        assert along.min() < 10.0
        assert along.max() > reference.length_m - 10.0

    def test_the_nearest_point_alone_would_not_have_been_good_enough(self, kerb, reference):
        """Why `deviation` projects onto segments instead of taking the nearest
        reference point.

        A car between two reference points is measured against the line it is
        actually near, not against the nearest dot on it. The nearest dot
        over-reads by up to half the point spacing, and at real thresholds that
        error is a large fraction of the 2 m headline.
        """
        between = circle(397, 300.0, phase=np.pi / 2000)
        nearest, _ = reference.tree.query(between)
        deviation, _ = reference.deviation(between)
        assert nearest.max() > 0.2
        assert np.abs(deviation).max() < 0.02


class TestAgainstCommittedGeometry:
    """One check against a real track built by a different script.

    Every other geometry test here uses a circle, which proves the arithmetic and
    proves nothing about units. `data/geometry/*.json` was produced by
    `scripts/export_track_geometry.py` from the same feed and independently
    converted from tenths of a metre to metres, so agreeing with it is evidence
    that this file has the same idea of what a metre is.
    """

    @pytest.fixture(scope="class")
    def monza(self):
        path = ROOT / "data" / "geometry" / "monza.json"
        if not path.exists():
            pytest.skip("data/geometry/monza.json not present")
        import json
        return json.loads(path.read_text(encoding="utf-8"))

    def test_lap_length_agrees_with_the_exported_geometry(self, kerb, monza):
        line = kerb.ReferenceLine(np.array(monza["centreline"], dtype=float)[:, :2])
        # The exported figure walks the 600 points without closing the loop; this
        # one closes it, so it is longer by exactly one point spacing.
        spacing = monza["lap_length_m"] / (monza["n_points"] - 1)
        assert line.length_m == pytest.approx(monza["lap_length_m"] + spacing, rel=0.002)
        assert 4000 < line.length_m < 8000, "a lap in metres, not tenths"

    def test_the_exported_centreline_lies_on_its_own_reference(self, kerb, monza):
        line = kerb.ReferenceLine(np.array(monza["centreline"], dtype=float)[:, :2])
        deviation, _ = line.deviation(np.array(monza["centreline"], dtype=float)[:, :2])
        assert np.abs(deviation).max() < 0.01


class TestLapExposure:
    @pytest.fixture
    def corners(self, reference):
        """Two corner markers, a quarter of a lap apart."""
        return np.array([0.0, reference.length_m / 4])

    def test_a_lap_on_the_line_is_never_exposed(self, kerb, reference, corners):
        features = kerb.lap_exposure(circle(400, 300.0), reference, corners)
        for threshold in kerb.THRESHOLDS_M:
            key = kerb._threshold_key(threshold)
            assert features[f"exposure_frac_{key}"] == 0.0
            assert features[f"n_corners_wide_{key}"] == 0

    def test_a_lap_three_metres_wide_is_exposed_below_three_and_not_above(
            self, kerb, reference, corners):
        features = kerb.lap_exposure(circle(400, 303.0), reference, corners)
        assert features["exposure_frac_2m0"] == pytest.approx(1.0, abs=1e-6)
        assert features["exposure_frac_2m5"] == pytest.approx(1.0, abs=1e-6)
        assert features["exposure_frac_4m0"] == 0.0
        assert features["mean_abs_dev_m"] == pytest.approx(3.0, abs=0.02)

    def test_exposure_only_falls_as_the_threshold_rises(self, kerb, reference, corners):
        """A monotonicity that must hold for the sensitivity sweep to be readable.

        If it ever failed, a threshold could be chosen to manufacture a result,
        which is precisely what reporting the whole sweep is meant to prevent.
        """
        radius = 300.0 + 2.0 * np.sin(np.linspace(0, 6 * np.pi, 600))
        theta = np.linspace(0.0, 2 * np.pi, 600, endpoint=False)
        wandering = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        features = kerb.lap_exposure(wandering, reference, corners)
        fractions = [features[f"exposure_frac_{kerb._threshold_key(t)}"]
                     for t in kerb.THRESHOLDS_M]
        assert fractions == sorted(fractions, reverse=True)

    def test_exposure_is_weighted_by_distance_not_by_sample_count(
            self, kerb, reference, corners):
        """The lap's weights must sum to the length of the path the car drove.

        Samples arrive at a fixed time interval, so counting them would weight a
        slow corner far more than a fast one -- a statement about speed, not about
        where the car was.

        The sum is the length of the *open* path between the first and last
        sample, which is correct: a lap's samples run from just after the line to
        just before it, and there is no unmeasured segment to add. On this closed
        synthetic circle that leaves out one chord, hence the explicit comparison
        rather than the circumference.
        """
        lap = circle(400, 300.0)
        features = kerb.lap_exposure(lap, reference, corners)
        walked = float(np.linalg.norm(np.diff(lap, axis=0), axis=1).sum())
        assert features["path_length_m"] == pytest.approx(walked, rel=1e-9)
        assert features["path_length_m"] == pytest.approx(2 * np.pi * 300.0, rel=0.01)

    def test_a_corner_run_wide_is_counted_once_however_many_samples(
            self, kerb, reference, corners):
        """`n_corners_wide` counts corners, not readings. A car sitting wide
        through one corner for twenty samples must read 1, or a slow corner would
        outscore a fast one for no reason."""
        theta = np.linspace(0.0, 2 * np.pi, 1200, endpoint=False)
        radius = np.where(np.abs(theta - np.pi / 2) < 0.1, 303.0, 300.0)
        lap = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        features = kerb.lap_exposure(lap, reference, corners)
        assert features["n_corners_wide_2m0"] == 1
        assert features["exposure_frac_2m0"] < 0.1

    def test_a_pit_lane_excursion_is_reported_separately(self, kerb, reference, corners):
        """Gross excursions are flagged, not folded into the mean, because a lap
        through the pit lane is twenty metres off the line and would otherwise
        dominate any stint average it landed in."""
        theta = np.linspace(0.0, 2 * np.pi, 800, endpoint=False)
        radius = np.where(theta < 0.4, 320.0, 300.0)
        lap = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        features = kerb.lap_exposure(lap, reference, corners)
        assert features["gross_excursion_frac"] > 0.0
        assert features["max_abs_dev_m"] > 10.0


class TestTrackLimits:
    def laps(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["Driver", "LapNumber", "Deleted", "DeletedReason"])

    def session(self, frame):
        return type("FakeSession", (), {"laps": frame})()

    def test_extracts_the_corner_from_the_stewards_reason(self, kerb):
        frame = self.laps([
            {"Driver": "NOR", "LapNumber": 16, "Deleted": True,
             "DeletedReason": "TRACK LIMITS AT TURN 19 LAP 16 "},
        ])
        events, histogram = kerb.track_limits(self.session(frame))
        assert events.iloc[0]["limits_corner"] == "19"
        assert histogram == {"19": 1}

    def test_ignores_deletions_that_are_not_track_limits(self, kerb):
        frame = self.laps([
            {"Driver": "VER", "LapNumber": 5, "Deleted": True,
             "DeletedReason": "RED FLAG"},
            {"Driver": "VER", "LapNumber": 9, "Deleted": True,
             "DeletedReason": "TRACK LIMITS AT TURN 1 LAP 9 "},
        ])
        events, _ = kerb.track_limits(self.session(frame))
        assert list(events["session_lap"]) == [9]

    def test_a_lettered_corner_keeps_its_letter(self, kerb):
        frame = self.laps([
            {"Driver": "HAM", "LapNumber": 3, "Deleted": True,
             "DeletedReason": "TRACK LIMITS AT TURN 8 A LAP 3"},
        ])
        events, _ = kerb.track_limits(self.session(frame))
        assert events.iloc[0]["limits_corner"] == "8A"

    def test_an_unparseable_reason_is_kept_as_unknown_not_dropped(self, kerb):
        """A deletion we cannot locate is still a deletion. Dropping it would
        undercount excursions at exactly the circuits whose message format we
        failed to anticipate."""
        frame = self.laps([
            {"Driver": "ALO", "LapNumber": 7, "Deleted": True,
             "DeletedReason": "TRACK LIMITS"},
        ])
        events, _ = kerb.track_limits(self.session(frame))
        assert len(events) == 1
        assert events.iloc[0]["limits_corner"] == "unknown"

    def test_a_session_with_no_deletions_returns_an_empty_frame(self, kerb):
        frame = self.laps([{"Driver": "PIA", "LapNumber": 1, "Deleted": False,
                            "DeletedReason": ""}])
        events, histogram = kerb.track_limits(self.session(frame))
        assert events.empty and histogram == {}


class TestExperimentStatistics:
    def test_demeaning_removes_the_group_mean(self, exp33):
        """The whole defence against the circuit confound. Some circuits have more
        kerbed corners and more monitored ones; a pooled correlation would read
        that as an effect of kerbs on tyres."""
        frame = pd.DataFrame({
            "circuit": ["a"] * 4 + ["b"] * 4,
            "x": [1.0, 2, 3, 4, 101, 102, 103, 104],
            "y": [1.0, 2, 3, 4, 1, 2, 3, 4],
        })
        out = exp33.demean(frame, ["circuit"], ["x", "y"])
        assert out.groupby("circuit")["x"].mean().abs().max() < 1e-12
        assert out["x"].tolist() == [-1.5, -0.5, 0.5, 1.5] * 2

    def test_demeaning_drops_groups_below_the_minimum(self, exp33):
        frame = pd.DataFrame({"circuit": ["a", "a", "a", "b"],
                              "x": [1.0, 2, 3, 9], "y": [1.0, 2, 3, 9]})
        out = exp33.demean(frame, ["circuit"], ["x", "y"], min_size=3)
        assert set(out["circuit"]) == {"a"}

    def test_the_detection_floor_shrinks_with_sample_size(self, exp33):
        assert exp33.min_detectable_rho(100) > exp33.min_detectable_rho(10_000)
        assert exp33.min_detectable_rho(3) != exp33.min_detectable_rho(3)  # nan

    def test_a_correlation_on_too_little_data_is_declared_rather_than_computed(
            self, exp33):
        """Returning None rather than a rho from eight points. A number there
        would be reported as a result and it would be noise."""
        out = exp33.spearman(pd.Series(range(8), dtype=float),
                             pd.Series(range(8), dtype=float))
        assert out["rho"] is None and out["n"] == 8

    def test_a_constant_predictor_yields_no_correlation(self, exp33):
        """Most stints have zero excursions. If a circuit's stints all have zero,
        spearmanr returns nan, and a nan that reaches the result file reads as a
        missing number rather than as an impossible test."""
        out = exp33.spearman(pd.Series([0.0] * 40), pd.Series(np.arange(40.0)))
        assert out["rho"] is None


class TestPreregisteredVerdict:
    """The decision rule exp33 fixed in advance, pinned so it cannot drift."""

    GATE_PASS = {"measure_a_valid": True, "gates": {"G1": True}}
    GATE_FAIL = {"measure_a_valid": False, "gates": {"G1": True, "G3_cars_not_coincident": False}}

    def measure_b(self, rho: float, p: float, *, pairs: int = 40,
                  difference: float = 0.01, wilcoxon: float = 0.01) -> dict:
        return {
            "T3_within_driver_and_circuit": {"rho": rho, "p_value": p},
            "T4_paired_within_driver_and_circuit": {
                "n_pairs": pairs, "underpowered": pairs < 20,
                "mean_difference_s_per_lap2": difference, "p_value": wilcoxon},
        }

    def test_support_needs_all_three_tests(self, exp33):
        label, _ = exp33.verdict(self.GATE_PASS, self.measure_b(0.2, 0.001),
                                {"helps": True})
        assert label == "supported"

    def test_a_failed_leave_one_circuit_out_blocks_support(self, exp33):
        """T3 and T4 significant is not enough. exp09's lesson: a feature that
        cannot beat the compound label out of sample is not adding information,
        whatever its in-sample correlation."""
        label, _ = exp33.verdict(self.GATE_PASS, self.measure_b(0.2, 0.001),
                                {"helps": False})
        assert label == "null"

    def test_an_underpowered_paired_test_blocks_support(self, exp33):
        label, _ = exp33.verdict(self.GATE_PASS,
                                self.measure_b(0.2, 0.001, pairs=5), {"helps": True})
        assert label == "null"

    def test_a_significant_negative_correlation_is_a_wrong_signed_refutation(self, exp33):
        """The exp25 outcome, which this project reports rather than buries: more
        exposure, less degradation, which is backwards for a wear mechanism."""
        label, lines = exp33.verdict(self.GATE_PASS, self.measure_b(-0.2, 0.001),
                                     {"helps": False})
        assert label == "refuted_wrong_sign"
        assert any("WRONG SIGN" in line for line in lines)

    def test_a_failed_measurement_gate_is_announced_before_anything_else(self, exp33):
        label, lines = exp33.verdict(self.GATE_FAIL, self.measure_b(0.01, 0.8),
                                     {"helps": False})
        assert label == "null"
        assert lines[0].startswith("MEASUREMENT:")
        assert "G3_cars_not_coincident" in lines[0]

    def test_a_non_null_negative_control_is_announced(self, exp33):
        """The control failing is itself a result, and it must appear in the
        verdict rather than only in a table nobody reads to the bottom of."""
        _, lines = exp33.verdict(self.GATE_FAIL, self.measure_b(0.01, 0.8),
                                 {"helps": False}, ["2m0/cell", "1m0/cell"])
        assert any(line.startswith("CONTROL NOT NULL") for line in lines)

    def test_a_null_control_says_nothing(self, exp33):
        _, lines = exp33.verdict(self.GATE_FAIL, self.measure_b(0.01, 0.8),
                                 {"helps": False}, [])
        assert not any(line.startswith("CONTROL NOT NULL") for line in lines)

    def test_a_missing_correlation_does_not_crash_the_rule(self, exp33):
        """Every test in this experiment can legitimately return None. The verdict
        must survive that rather than raising and losing the whole run."""
        label, _ = exp33.verdict(self.GATE_FAIL,
                                 self.measure_b(None, None, wilcoxon=None),
                                 {"helps": False})
        assert label == "null"
