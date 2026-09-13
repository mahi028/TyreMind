"""The telemetry reduction must not silently produce features that mean nothing.

exp25 rests on these numbers and its conclusion is a refutation, which is exactly
the kind of result a broken feature pipeline produces by accident. So the
properties that make the features interpretable are asserted here rather than
assumed.

One of these tests pins behaviour that looks like a bug and is not. The frictional
energy proxy is driven by *demanded acceleration* -- slip is approximated as
proportional to how hard the tyre is being asked to work -- so a car holding a
constant speed in a perfectly straight line dissipates zero energy in this model.
That is correct for a slip-based proxy and it is documented in
`frictional_power_proxy`, but it is surprising enough that it earns a test saying
so out loud.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tyremind.physics.dynamics import VehicleParameters


def lap_features(car: pd.DataFrame) -> dict:
    spec = importlib.util.spec_from_file_location(
        "build_telemetry", Path("scripts/build_telemetry.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.lap_features(car, VehicleParameters())


def synthetic_lap(*, n: int = 400, radius_m: float = 300.0,
                  speed_kmh: float = 180.0) -> pd.DataFrame:
    """A car going round a circle -- constant speed, constant lateral load.

    A circle is the simplest shape that exercises the whole reduction: there is
    real lateral acceleration, so there is real slip demand, so there is real
    energy, and by symmetry the load should sit entirely on one side.
    """
    speed_ms = speed_kmh / 3.6
    circumference = 2 * np.pi * radius_m
    duration = circumference / speed_ms
    t = np.linspace(0.0, duration, n)
    theta = 2 * np.pi * t / duration
    return pd.DataFrame({
        "Speed": np.full(n, speed_kmh),
        "SessionTime": pd.to_timedelta(t, unit="s"),
        "X": radius_m * np.cos(theta),
        "Y": radius_m * np.sin(theta),
        "Throttle": np.full(n, 85.0),
        "Brake": np.zeros(n, dtype=bool),
    })


@pytest.fixture(scope="module")
def cornering_lap() -> pd.DataFrame:
    return synthetic_lap()


class TestFeatureSanity:
    def test_energy_shares_sum_to_one(self, cornering_lap):
        """Four corners, one lap. If the shares do not sum to 1 the split is
        wrong, and every four-tyre claim built on it is wrong with it."""
        f = lap_features(cornering_lap)
        total = sum(f[f"energy_share_{c}"] for c in ("fl", "fr", "rl", "rr"))
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_energy_is_positive_when_the_car_is_cornering(self, cornering_lap):
        """Frictional dissipation cannot be negative, and a car pulling lateral
        g must dissipate something. A zero here means the slip proxy is dead."""
        f = lap_features(cornering_lap)
        assert f["energy_mj_total"] > 0.0

    def test_a_constant_radius_corner_loads_one_side(self, cornering_lap):
        """Sustained cornering transfers load to the outside. A 50/50 split
        would mean lateral load transfer is not reaching the corner loads."""
        f = lap_features(cornering_lap)
        assert abs(f["left_energy_share"] - 0.5) > 0.02

    def test_fractions_stay_in_range(self, cornering_lap):
        f = lap_features(cornering_lap)
        for key in ("loaded_fraction", "full_throttle_fraction", "braking_fraction",
                    "front_energy_share", "left_energy_share",
                    "energy_share_fl", "energy_share_rr"):
            assert 0.0 <= f[key] <= 1.0, key

    def test_lateral_g_is_physically_plausible(self, cornering_lap):
        """180 km/h round a 300 m radius is v^2/r = 8.3 m/s^2, about 0.85 g.

        This is the test that would catch the curvature estimate being wrong by
        an order of magnitude -- which matters, because the measured mean lateral
        g on real sessions comes out lower than an F1 car should pull and that
        discrepancy is still open.
        """
        f = lap_features(cornering_lap)
        expected_g = ((180 / 3.6) ** 2 / 300.0) / 9.81
        assert f["mean_abs_lateral_g"] == pytest.approx(expected_g, rel=0.25)


class TestDocumentedEdgeCases:
    def test_a_constant_speed_straight_dissipates_no_energy(self):
        """Surprising, correct, and worth pinning.

        The energy proxy is slip-based and slip is approximated from demanded
        acceleration, so a car with neither lateral nor longitudinal demand
        reports zero. Real rolling resistance is not modelled. If someone later
        adds a constant term this test will fail and they will have to decide
        deliberately rather than by accident.
        """
        n = 120
        t = np.arange(n, dtype=float) * 0.25
        straight = pd.DataFrame({
            "Speed": np.full(n, 200.0),
            "SessionTime": pd.to_timedelta(t, unit="s"),
            "X": np.linspace(0, 10_000, n), "Y": np.zeros(n),
            "Throttle": np.full(n, 100.0), "Brake": np.zeros(n, dtype=bool),
        })
        f = lap_features(straight)
        assert f["energy_mj_total"] == 0.0
        assert not [k for k in f if "share" in k], (
            "shares are undefined at zero energy and must be omitted, not zero -- "
            "a zero share would be averaged into a session aggregate as if real"
        )

    def test_too_few_samples_returns_nothing_rather_than_noise(self):
        tiny = pd.DataFrame({
            "Speed": [200.0, 201.0],
            "SessionTime": pd.to_timedelta([0.0, 0.25], unit="s"),
            "X": [0.0, 50.0], "Y": [0.0, 0.0],
            "Throttle": [100.0, 100.0], "Brake": [False, False],
        })
        assert lap_features(tiny) == {}

    def test_duplicate_timestamps_do_not_produce_infinities(self):
        """Repeated SessionTime makes every derivative infinite. Rare, but one
        such lap would poison a session's aggregates without raising."""
        n = 40
        t = np.repeat(np.arange(n // 2, dtype=float) * 0.5, 2)
        car = pd.DataFrame({
            "Speed": np.full(n, 180.0),
            "SessionTime": pd.to_timedelta(t, unit="s"),
            "X": np.linspace(0, 4000, n), "Y": np.zeros(n),
            "Throttle": np.full(n, 90.0), "Brake": np.zeros(n, dtype=bool),
        })
        f = lap_features(car)
        assert all(np.isfinite(v) for v in f.values() if isinstance(v, float))
