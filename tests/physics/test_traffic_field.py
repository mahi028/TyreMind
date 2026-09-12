from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tyremind.physics.traffic_field import (
    ProximitySnapshot,
    resample_to_common_grid,
    snapshot_at,
    traffic_field,
)


def _telemetry(session_times_s: list[float], distances_m: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SessionTime": pd.to_timedelta(session_times_s, unit="s"),
            "Distance": distances_m,
        }
    )


class TestResampleToCommonGrid:
    def test_interpolates_between_samples(self) -> None:
        per_driver = {
            "A": _telemetry([0.0, 2.0, 4.0], [0.0, 20.0, 40.0]),
            "B": _telemetry([0.0, 2.0, 4.0], [0.0, 10.0, 20.0]),
        }
        common, distances = resample_to_common_grid(per_driver, freq_s=1.0)
        assert list(common) == [0.0, 1.0, 2.0, 3.0]
        assert distances["A"] == pytest.approx([0.0, 10.0, 20.0, 30.0])
        assert distances["B"] == pytest.approx([0.0, 5.0, 10.0, 15.0])

    def test_driver_outside_its_own_span_is_nan(self) -> None:
        per_driver = {
            "A": _telemetry([0.0, 4.0], [0.0, 40.0]),
            "B": _telemetry([2.0, 4.0], [0.0, 10.0]),  # joins the session late
        }
        common, distances = resample_to_common_grid(per_driver, freq_s=1.0)
        assert np.isnan(distances["B"][0])  # t=0, before B existed
        assert not np.isnan(distances["B"][2])  # t=2, B's first sample

    def test_empty_input(self) -> None:
        common, distances = resample_to_common_grid({}, freq_s=1.0)
        assert len(common) == 0
        assert distances == {}


class TestSnapshotAt:
    def test_orders_back_to_front_and_computes_gaps(self) -> None:
        snap = snapshot_at({"LEAD": 100.0, "MID": 80.0, "LAST": 50.0})
        assert snap.order == ["LAST", "MID", "LEAD"]
        assert snap.gap_ahead_m["LAST"] == pytest.approx(30.0)
        assert snap.gap_ahead_m["MID"] == pytest.approx(20.0)
        assert snap.gap_ahead_m["LEAD"] == float("inf")

    def test_drops_non_finite_drivers(self) -> None:
        snap = snapshot_at({"A": 10.0, "B": float("nan"), "C": 20.0})
        assert snap.order == ["A", "C"]
        assert "B" not in snap.gap_ahead_m

    def test_fewer_than_two_present_returns_none(self) -> None:
        assert snapshot_at({"A": 10.0, "B": float("nan")}) is None
        assert snapshot_at({}) is None


class TestTrafficField:
    def test_isolated_cars_are_not_a_train(self) -> None:
        common = np.array([0.0])
        distances = {"A": np.array([0.0]), "B": np.array([200.0])}
        df = traffic_field(common, distances, train_gap_m=50.0)
        assert set(df["in_train"]) == {False}
        assert set(df["train_size"]) == {1}

    def test_close_cars_form_a_train(self) -> None:
        # A -- 20m -- B -- 20m -- C, all within the 50m threshold: one train of 3.
        common = np.array([0.0])
        distances = {"A": np.array([0.0]), "B": np.array([20.0]), "C": np.array([40.0])}
        df = traffic_field(common, distances, train_gap_m=50.0)
        assert (df["train_size"] == 3).all()
        assert (df["in_train"]).all()

    def test_two_separate_trains(self) -> None:
        # A--10m--B (a pair), gap of 200m, then C--10m--D (another pair).
        common = np.array([0.0])
        distances = {
            "A": np.array([0.0]),
            "B": np.array([10.0]),
            "C": np.array([210.0]),
            "D": np.array([220.0]),
        }
        df = traffic_field(common, distances, train_gap_m=50.0)
        sizes = dict(zip(df["driver"], df["train_size"]))
        assert sizes == {"A": 2, "B": 2, "C": 2, "D": 2}

    def test_gap_ahead_matches_measured_distance(self) -> None:
        common = np.array([0.0])
        distances = {"A": np.array([0.0]), "B": np.array([37.5])}
        df = traffic_field(common, distances, train_gap_m=50.0)
        row = df[df["driver"] == "A"].iloc[0]
        assert row["gap_ahead_m"] == pytest.approx(37.5)
        assert row["driver_ahead"] == "B"

    def test_front_runner_has_infinite_gap_and_no_driver_ahead(self) -> None:
        common = np.array([0.0])
        distances = {"A": np.array([0.0]), "B": np.array([100.0])}
        df = traffic_field(common, distances, train_gap_m=50.0)
        row = df[df["driver"] == "B"].iloc[0]
        assert row["gap_ahead_m"] == float("inf")
        assert row["driver_ahead"] is None

    def test_multiple_timesteps_are_independent(self) -> None:
        common = np.array([0.0, 1.0])
        # Cars start far apart, close up to form a train at t=1.
        distances = {"A": np.array([0.0, 0.0]), "B": np.array([200.0, 30.0])}
        df = traffic_field(common, distances, train_gap_m=50.0)
        t0 = df[df["time_s"] == 0.0]
        t1 = df[df["time_s"] == 1.0]
        assert (t0["train_size"] == 1).all()
        assert (t1["train_size"] == 2).all()

    def test_car_absent_from_a_snapshot_is_skipped_not_zeroed(self) -> None:
        common = np.array([0.0, 1.0])
        distances = {"A": np.array([0.0, 10.0]), "B": np.array([float("nan"), 50.0])}
        df = traffic_field(common, distances, train_gap_m=50.0)
        # At t=0 only A is present -> no row emitted at all (snapshot_at needs 2 cars).
        assert not (df["time_s"] == 0.0).any()
        assert (df["time_s"] == 1.0).sum() == 2
