"""Input assembly, fitting, and the identifiability measurements.

The identifiability tests matter most. `PREREGISTRATION_exp35.md` makes five
numbered predictions before anything was run, and three of them are about what
the data cannot support. A module that measures those has to be checked against
cases where the answer is known by construction: symmetric inputs must produce
unidentifiable corners, and a straight-line loss must produce a data-driven share
of zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tyremind.models.fourcorner.dynamics import LapInput, drift
from tyremind.models.fourcorner.fit import (
    FourCornerFit,
    degradation_rate,
    fit_stint,
)
from tyremind.models.fourcorner.identifiability import (
    analyse,
    data_driven_share,
    observability_gramian,
    prior_sensitivity,
)
from tyremind.models.fourcorner.inputs import (
    ENERGY_COLUMNS,
    LOAD_COLUMNS,
    asymmetry_report,
    build_inputs,
)
from tyremind.models.fourcorner.observation import predict_lap_time
from tyremind.models.fourcorner.state import (
    FourCornerParameters,
    clamp_state,
    initial_covariance,
    initial_state,
)

TYPICAL_POWER = 1.81e5


def make_lap_table(n: int = 24, driver: str = "VER") -> pd.DataFrame:
    return pd.DataFrame({
        "driver": [driver] * n,
        "session_lap": np.arange(1, n + 1),
        "run_id": ["r1"] * n,
        "tyre_age": np.arange(1, n + 1, dtype=float),
        "lap_in_run": np.arange(0, n, dtype=float),
        "lap_time": 92.0 + 0.05 * np.arange(n),
        "compound": ["MEDIUM"] * n,
        "traffic_index": np.zeros(n),
    })


def make_telemetry(n: int = 24, driver: str = "VER", *, asymmetric: bool = True):
    shares = np.array([0.28, 0.22, 0.27, 0.23]) if asymmetric else np.full(4, 0.25)
    total_mj = 68.3
    frame = {"driver": [driver] * n, "session_lap": np.arange(1, n + 1),
             "mean_abs_lateral_g": np.full(n, 1.25), "mean_speed_kmh": np.full(n, 205.0)}
    for col, share in zip(ENERGY_COLUMNS, shares):
        frame[col] = np.full(n, total_mj * share)
    for col, load in zip(LOAD_COLUMNS, (5200.0, 3800.0, 4900.0, 3600.0)):
        frame[col] = np.full(n, load)
    return pd.DataFrame(frame)


def a_lap(i: int, *, asymmetric: bool = True) -> LapInput:
    power = (
        TYPICAL_POWER * np.array([1.10, 0.90, 1.05, 0.95])
        if asymmetric
        else np.full(4, TYPICAL_POWER)
    )
    return LapInput(
        duration_s=92.0,
        load_n=np.array([5200.0, 3800.0, 4900.0, 3600.0]),
        power_proxy_w=power,
        airspeed_ms=58.0,
        laps_completed=float(i - 1),
        traffic_index=0.0,
        session_lap=i,
    )


def simulate(n: int = 30, kappa: float = 2.4e-4, seed: int = 0, *, asymmetric: bool = True):
    """A stint generated from the model with a known wear coefficient."""
    rng = np.random.default_rng(seed)
    truth = FourCornerParameters().with_thermo(kappa=kappa)
    x = initial_state(tread_temp_c=truth.grip.temp_optimal_c)
    inputs, observations = [], []
    for i in range(1, n + 1):
        u = a_lap(i, asymmetric=asymmetric)
        dt = u.duration_s / 40
        for _ in range(40):
            x = clamp_state(x + dt * drift(x, u, truth))
        inputs.append(u)
        observations.append(predict_lap_time(x, u, truth) + rng.normal(0.0, 0.12))
    return inputs, np.asarray(observations), truth


class TestBuildInputs:
    def test_uses_telemetry_when_present(self):
        bundle = build_inputs(make_lap_table(), make_telemetry(), "VER", "r1")
        assert bundle.n_measured == 24
        assert bundle.n_synthesised == 0
        assert bundle.measured_fraction == 1.0

    def test_falls_back_and_counts_when_telemetry_is_absent(self):
        """A symmetric fallback is a one-corner model wearing nine extra states,
        so the count has to be visible to whatever scores the experiment."""
        bundle = build_inputs(make_lap_table(), None, "VER", "r1")
        assert bundle.n_measured == 0
        assert bundle.n_synthesised == 24
        assert bundle.measured_fraction == 0.0
        first = bundle.inputs[0].power_proxy_w
        assert np.allclose(first, first[0]), "fallback must be symmetric"

    def test_counts_partial_telemetry_correctly(self):
        telemetry = make_telemetry()
        telemetry.loc[5:9, list(ENERGY_COLUMNS)] = np.nan
        bundle = build_inputs(make_lap_table(), telemetry, "VER", "r1")
        assert bundle.n_measured == 19
        assert bundle.n_synthesised == 5

    def test_left_right_ratio_reflects_the_energy_split(self):
        bundle = build_inputs(make_lap_table(), make_telemetry(asymmetric=True), "VER", "r1")
        # FL 0.28 + RL 0.27 = 0.55 of the total.
        assert bundle.mean_left_right_ratio == pytest.approx(0.55, abs=1e-6)

    def test_raises_on_an_unknown_stint(self):
        """The fairness rule: every model raises on unusable input."""
        with pytest.raises(ValueError, match="no laps"):
            build_inputs(make_lap_table(), None, "HAM", "r1")

    def test_raises_when_every_lap_time_is_missing(self):
        table = make_lap_table()
        table["lap_time"] = np.nan
        with pytest.raises(ValueError, match="no usable laps"):
            build_inputs(table, None, "VER", "r1")

    def test_observations_match_lap_times(self):
        table = make_lap_table()
        bundle = build_inputs(table, make_telemetry(), "VER", "r1")
        assert np.allclose(bundle.observations, table["lap_time"].to_numpy())


class TestAsymmetryReport:
    def test_symmetric_inputs_report_zero_asymmetry(self):
        bundle = build_inputs(make_lap_table(), None, "VER", "r1")
        report = asymmetry_report(bundle)
        assert report["asymmetry"] == pytest.approx(0.0, abs=1e-9)
        assert report["left_share"] == pytest.approx(0.5, abs=1e-9)

    def test_asymmetric_inputs_are_detected(self):
        bundle = build_inputs(make_lap_table(), make_telemetry(asymmetric=True), "VER", "r1")
        report = asymmetry_report(bundle)
        assert report["asymmetry"] > 0.01
        assert report["left_share"] > 0.5

    def test_shares_sum_to_one(self):
        bundle = build_inputs(make_lap_table(), make_telemetry(), "VER", "r1")
        report = asymmetry_report(bundle)
        total = sum(report[f"power_share_{c}"] for c in ("FL", "FR", "RL", "RR"))
        assert total == pytest.approx(1.0)


class TestFit:
    def test_recovers_a_known_wear_coefficient(self):
        inputs, obs, truth = simulate(n=30, kappa=2.4e-4, seed=0)
        fit = fit_stint(inputs, obs)
        assert fit.converged
        assert fit.params.thermo.kappa == pytest.approx(truth.thermo.kappa, rel=0.25)

    def test_reports_a_positive_degradation_rate(self):
        inputs, obs, _ = simulate(n=30, seed=1)
        fit = fit_stint(inputs, obs)
        assert fit.mean_rate > 0.0
        assert np.isfinite(fit.mean_rate_sd)

    def test_fits_inside_the_preregistered_time_budget(self):
        """S4 in the pre-registration: under 60 s a session.

        A single stint is a fraction of a session, so this is a loose check that
        the per-stint cost has not blown up by an order of magnitude. The
        experiment measures the real per-session figure.
        """
        inputs, obs, _ = simulate(n=30, seed=2)
        fit = fit_stint(inputs, obs)
        assert fit.fit_seconds < 30.0

    def test_does_not_diverge_on_a_clean_stint(self):
        inputs, obs, _ = simulate(n=45, seed=3)
        fit = fit_stint(inputs, obs)
        assert fit.n_diverged == 0

    def test_raises_on_mismatched_lengths(self):
        inputs, obs, _ = simulate(n=12)
        with pytest.raises(ValueError, match="one observation per lap"):
            fit_stint(inputs, obs[:-1])

    def test_raises_on_non_finite_observations(self):
        inputs, obs, _ = simulate(n=12)
        spoiled = obs.copy()
        spoiled[3] = np.nan
        with pytest.raises(ValueError, match="finite"):
            fit_stint(inputs, spoiled)

    def test_raises_on_an_empty_stint(self):
        with pytest.raises(ValueError, match="at least one lap"):
            fit_stint([], np.array([]))

    def test_a_faster_wearing_tyre_gives_a_larger_rate(self):
        slow_in, slow_obs, _ = simulate(n=30, kappa=1.2e-4, seed=4)
        fast_in, fast_obs, _ = simulate(n=30, kappa=4.0e-4, seed=4)
        slow = fit_stint(slow_in, slow_obs)
        fast = fit_stint(fast_in, fast_obs)
        assert fast.mean_rate > slow.mean_rate


class TestDegradationRate:
    def test_first_lap_is_not_a_spurious_zero(self):
        """A zero on lap one would drag every stint mean down by a lap's worth.

        It looks conservative, which is exactly why it survives review.
        """
        inputs, obs, _ = simulate(n=20, seed=5)
        fit = fit_stint(inputs, obs)
        assert fit.rate_per_lap[0] == pytest.approx(fit.rate_per_lap[1])

    def test_rate_is_the_difference_of_the_penalty(self):
        inputs, obs, _ = simulate(n=15, seed=6)
        fit = fit_stint(inputs, obs)
        rate, sd = degradation_rate(fit.smoothed_states, fit.result.covariances, fit.params)
        assert len(rate) == len(fit.smoothed_states)
        assert (sd >= 0).all()

    def test_a_flat_state_gives_a_zero_rate(self):
        p = FourCornerParameters()
        states = np.tile(initial_state(tread_temp_c=p.grip.temp_optimal_c), (10, 1))
        covs = np.tile(initial_covariance(), (10, 1, 1))
        rate, _ = degradation_rate(states, covs, p)
        assert np.allclose(rate, 0.0)


class TestIdentifiability:
    def test_symmetric_inputs_make_corners_inseparable(self):
        """The cleanest case: identical corners cannot be told apart.

        With the same power at every corner and equal front and rear
        sensitivities the FL-FR direction leaves the lap time exactly unchanged,
        so the Gramian must be blind to it. This is prediction P3 in a setting
        where the answer is known by construction.
        """
        p = FourCornerParameters().with_grip(sensitivity_s=(2.4, 2.4, 2.4, 2.4))
        inputs = [a_lap(i, asymmetric=False) for i in range(1, 26)]
        obs = np.array([92.0 + 0.04 * i for i in range(25)])
        report = analyse(inputs, obs, p)
        assert report.corner_separability < 1e-8
        assert report.numerical_rank < report.n_states

    def test_asymmetric_inputs_give_the_corners_some_separation(self):
        p = FourCornerParameters()
        inputs = [a_lap(i, asymmetric=True) for i in range(1, 26)]
        obs = np.array([92.0 + 0.04 * i for i in range(25)])
        report = analyse(inputs, obs, p)
        assert report.corner_separability > 0.0

    def test_gramian_is_symmetric_positive_semidefinite(self):
        p = FourCornerParameters()
        inputs = [a_lap(i) for i in range(1, 21)]
        obs = np.array([92.0 + 0.04 * i for i in range(20)])
        from tyremind.models.fourcorner.ekf import run_filter
        r = run_filter(inputs, obs, initial_state(tread_temp_c=105.0),
                       initial_covariance(), p, obs_scale_s=0.2)
        g = observability_gramian(inputs, r.states, p)
        assert np.allclose(g, g.T)
        assert np.linalg.eigvalsh(g).min() >= -1e-8

    def test_a_straight_line_loss_is_entirely_unidentified(self):
        """exp18's definition, on a case where the answer is zero by construction.

        If the cumulative loss is exactly linear in age, none of it is separable
        from fuel, so the data-driven share must be zero.
        """
        p = FourCornerParameters()
        n = 30
        states = np.tile(initial_state(tread_temp_c=p.grip.temp_optimal_c), (n, 1))
        # A perfectly linear wear ramp.
        states[:, 0:4] = np.linspace(0.0, 0.4, n)[:, None]
        share = data_driven_share(states, [a_lap(i) for i in range(1, n + 1)], p)
        assert share["data_driven_share"] < 1e-3

    def test_curvature_registers_as_identified_signal(self):
        p = FourCornerParameters()
        n = 30
        states = np.tile(initial_state(tread_temp_c=p.grip.temp_optimal_c), (n, 1))
        # A cliff: flat then steep, which is strongly non-linear.
        ramp = np.concatenate([np.linspace(0.0, 0.2, n // 2),
                               np.linspace(0.2, 0.95, n - n // 2)])
        states[:, 0:4] = ramp[:, None]
        share = data_driven_share(states, [a_lap(i) for i in range(1, n + 1)], p)
        assert share["data_driven_share"] > 1e-3

    def test_share_is_reported_against_exp18(self):
        """The comparison number must travel with the measurement.

        An earlier version of this module compared a different quantity against
        exp18's 5.96% and reported a sixteenfold improvement that was pure metric
        mismatch. Carrying the reference in the payload makes that harder to
        repeat.
        """
        p = FourCornerParameters()
        n = 20
        states = np.tile(initial_state(tread_temp_c=p.grip.temp_optimal_c), (n, 1))
        states[:, 0:4] = np.linspace(0.0, 0.3, n)[:, None]
        share = data_driven_share(states, [a_lap(i) for i in range(1, n + 1)], p)
        assert share["exp18_two_state_data_driven_share"] == pytest.approx(0.0596)

    def test_prior_sensitivity_is_a_separate_named_quantity(self):
        """It must not be confusable with the exp18-comparable share."""
        inputs, obs, _ = simulate(n=25, seed=8)
        ps = prior_sensitivity(inputs, obs, FourCornerParameters())
        assert "data_driven_share" not in ps
        assert "relative_shift" in ps
        assert ps["relative_shift"] >= 0.0
