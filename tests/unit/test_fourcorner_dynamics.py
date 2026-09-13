"""The four-corner physics, checked against what physics requires.

These tests do not compare the model against itself. Each one asserts a property
that has to hold for the equations to be the equations they claim to be: grip
falls with wear, the thermal window peaks where it is told to, wear cannot be
negative, and the hand-written Jacobian matches a numerical one.

The Jacobian test is the load-bearing one. Every interval this model would ever
publish comes out of the covariance, the covariance is propagated through `F`,
and an `F` that is merely plausible produces intervals that are merely plausible.
Finite differences are slow and exact enough to catch a sign error or a missing
chain-rule term, which are the two ways a hand-derived Jacobian actually fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from tyremind.models.fourcorner.dynamics import (
    LapInput,
    diffusion,
    drift,
    frictional_power,
    grip_coefficient,
    jacobian,
    thermal_rate,
    wear_rate,
)
from tyremind.models.fourcorner.state import (
    I_FUEL,
    I_TRACK,
    N_STATE,
    T,
    W,
    FourCornerParameters,
    clamp_state,
    describe,
    initial_covariance,
    initial_state,
)


#: Per-corner power proxy on a representative lap, in the units the telemetry
#: corpus actually produces: median total 68.3 MJ over a ~92 s lap is 7.25e5, so
#: about 1.81e5 a corner. Fixtures use those magnitudes rather than round numbers
#: because the grip model is a Gaussian in temperature -- testing at a power that
#: puts the tyre 50 C outside its window exercises a regime the model never runs
#: in and hides bugs in the one it does.
TYPICAL_POWER = 1.81e5


def a_lap_input(*, asymmetric: bool = True) -> LapInput:
    """A lap with a left-loaded corner split, as a clockwise circuit produces."""
    load = np.array([5200.0, 3800.0, 4900.0, 3600.0]) if asymmetric else np.full(4, 4400.0)
    power = (
        TYPICAL_POWER * np.array([1.10, 0.90, 1.05, 0.95])
        if asymmetric
        else np.full(4, TYPICAL_POWER)
    )
    return LapInput(
        duration_s=92.0,
        load_n=load,
        power_proxy_w=power,
        airspeed_ms=58.0,
        laps_completed=6.0,
        traffic_index=0.1,
        session_lap=14,
    )


class TestGrip:
    def test_falls_monotonically_with_wear(self):
        p = FourCornerParameters()
        temps = np.full(4, p.grip.temp_optimal_c)
        previous = np.inf
        for w in np.linspace(0.0, 1.0, 25):
            mu = grip_coefficient(np.full(4, w), temps, p)[0]
            assert mu <= previous + 1e-12, "grip must not increase with wear"
            previous = mu

    def test_peaks_at_the_optimal_temperature(self):
        p = FourCornerParameters()
        fresh = np.zeros(4)
        at_optimum = grip_coefficient(fresh, np.full(4, p.grip.temp_optimal_c), p)[0]
        for offset in (-40.0, -15.0, 15.0, 40.0):
            off = grip_coefficient(fresh, np.full(4, p.grip.temp_optimal_c + offset), p)[0]
            assert off < at_optimum

    def test_cliff_accelerates_the_loss(self):
        """Past the critical wear the second derivative must turn negative.

        This is the only thing separating the cliff term from a steeper line, and
        exp17 measured 337 real stints where the loss accelerates.
        """
        p = FourCornerParameters()
        temps = np.full(4, p.grip.temp_optimal_c)
        before = [
            grip_coefficient(np.full(4, w), temps, p)[0]
            for w in (p.grip.wear_critical - 0.2, p.grip.wear_critical - 0.1, p.grip.wear_critical)
        ]
        after = [
            grip_coefficient(np.full(4, w), temps, p)[0]
            for w in (p.grip.wear_critical, p.grip.wear_critical + 0.1, p.grip.wear_critical + 0.2)
        ]
        drop_before = before[0] - before[-1]
        drop_after = after[0] - after[-1]
        assert drop_after > drop_before

    def test_never_negative(self):
        p = FourCornerParameters()
        mu = grip_coefficient(np.full(4, 1.5), np.full(4, 20.0), p)
        assert (mu >= 0.0).all()

    def test_corners_are_independent(self):
        """One corner's wear must not change another's grip.

        If this ever fails the four states have been coupled by accident and the
        identifiability analysis downstream is measuring the wrong thing.
        """
        p = FourCornerParameters()
        base = grip_coefficient(np.zeros(4), np.full(4, 100.0), p)
        one_worn = grip_coefficient(np.array([0.9, 0.0, 0.0, 0.0]), np.full(4, 100.0), p)
        assert one_worn[0] < base[0]
        assert np.allclose(one_worn[1:], base[1:])


class TestWear:
    def test_zero_power_gives_zero_wear(self):
        p = FourCornerParameters()
        assert np.allclose(wear_rate(np.zeros(4), p), 0.0)

    def test_monotone_in_power(self):
        p = FourCornerParameters()
        rates = [wear_rate(np.full(4, q), p)[0] for q in (0.0, 5e4, 1.8e5, 4e5)]
        assert rates == sorted(rates)

    def test_never_negative_even_on_negative_power(self):
        """A negative power is unphysical and must not produce a NaN.

        A fractional exponent of a negative number is NaN, and a single NaN in
        the wear state poisons the whole filter silently.
        """
        p = FourCornerParameters()
        rate = wear_rate(np.array([-1e4, 0.0, 1.8e5, -5.0]), p)
        assert np.isfinite(rate).all()
        assert (rate >= 0.0).all()

    def test_exponent_above_one_penalises_hard_laps(self):
        linear = FourCornerParameters().with_thermo(alpha=1.0)
        superlinear = FourCornerParameters().with_thermo(alpha=1.4)
        soft, hard = 9e4, 3.6e5
        # Same total energy, split evenly versus concentrated.
        even = 2 * wear_rate(np.full(4, (soft + hard) / 2), linear)[0]
        assert np.isfinite(even)
        spiky_linear = wear_rate(np.full(4, soft), linear)[0] + wear_rate(np.full(4, hard), linear)[0]
        spiky_super = (
            wear_rate(np.full(4, soft), superlinear)[0]
            + wear_rate(np.full(4, hard), superlinear)[0]
        )
        even_super = 2 * wear_rate(np.full(4, (soft + hard) / 2), superlinear)[0]
        # Under a convex law the spiky pair costs more than the even pair.
        assert spiky_super > even_super
        assert spiky_linear == pytest.approx(even, rel=1e-9)


class TestThermal:
    def test_heats_when_working_and_cools_when_idle(self):
        p = FourCornerParameters()
        cold = np.full(4, 60.0)
        working = thermal_rate(cold, np.full(4, 3.0e5), 55.0, p)
        idle = thermal_rate(np.full(4, 140.0), np.zeros(4), 55.0, p)
        assert (working > 0).all()
        assert (idle < 0).all()

    def test_settles_on_the_analytic_equilibrium(self):
        """With constant power the temperature must reach the balance point.

        Asserting against the closed-form equilibrium rather than against "the
        rate got small" -- the first version of this test integrated for 200 s,
        which is under four time constants, and failed a correct implementation.
        The time constant here is C / (k_cond + h) = 9000 / 165, about 55 s, so
        anything short of ~500 s is still visibly in transit.
        """
        p = FourCornerParameters()
        th = p.thermo
        power, airspeed = 1.81e5, 55.0

        generation = th.heat_fraction * power
        h = th.convection_base_w_per_k + th.convection_speed_w_per_k_per_ms * airspeed
        expected = (
            generation + th.conduction_w_per_k * th.carcass_temp_c + h * th.ambient_temp_c
        ) / (th.conduction_w_per_k + h)

        t = np.full(4, 80.0)
        for _ in range(20_000):  # 1000 s, ~18 time constants
            t = t + 0.05 * thermal_rate(t, np.full(4, power), airspeed, p)

        assert np.isfinite(t).all()
        assert t == pytest.approx(np.full(4, expected), abs=1e-6)

    def test_relaxes_at_the_expected_time_constant(self):
        """The approach to equilibrium must be exponential with tau = C / k.

        This is what makes the RK4 sub-stepping in the filter necessary: a time
        constant of 55 s against a 90 s lap means a single Euler step across the
        lap overshoots, and an overshoot in temperature reads downstream as
        thermal degradation that never happened.
        """
        p = FourCornerParameters()
        th = p.thermo
        airspeed = 55.0
        h = th.convection_base_w_per_k + th.convection_speed_w_per_k_per_ms * airspeed
        tau = th.thermal_capacity_j_per_k / (th.conduction_w_per_k + h)

        generation = th.heat_fraction * 1.81e5
        equilibrium = (
            generation + th.conduction_w_per_k * th.carcass_temp_c + h * th.ambient_temp_c
        ) / (th.conduction_w_per_k + h)

        start = 60.0
        t = np.full(4, start)
        step = 0.01
        for _ in range(int(tau / step)):  # exactly one time constant
            t = t + step * thermal_rate(t, np.full(4, 1.81e5), airspeed, p)

        remaining = (t[0] - equilibrium) / (start - equilibrium)
        assert remaining == pytest.approx(np.exp(-1.0), rel=0.01)

    def test_more_airspeed_cools_harder(self):
        p = FourCornerParameters()
        hot = np.full(4, 130.0)
        slow = thermal_rate(hot, np.full(4, 1.2e5), 20.0, p)
        fast = thermal_rate(hot, np.full(4, 1.2e5), 80.0, p)
        assert (fast < slow).all()

    def test_radiation_is_off_by_default_and_works_when_on(self):
        off = FourCornerParameters()
        on = FourCornerParameters().with_thermo(emissivity=0.8)
        hot = np.full(4, 160.0)
        cooler_with_radiation = thermal_rate(hot, np.full(4, 1.2e5), 55.0, on)
        without = thermal_rate(hot, np.full(4, 1.2e5), 55.0, off)
        assert (cooler_with_radiation < without).all()


class TestJacobian:
    """The analytic Jacobian against finite differences, on random states."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_matches_finite_differences(self, seed: int):
        rng = np.random.default_rng(seed)
        p = FourCornerParameters()
        u = a_lap_input()

        x = np.zeros(N_STATE)
        x[W] = rng.uniform(0.05, 0.9, 4)
        x[T] = rng.uniform(70.0, 130.0, 4)
        x[I_FUEL] = rng.uniform(20.0, 100.0)
        x[I_TRACK] = rng.uniform(0.1, 0.9)

        analytic = jacobian(x, u, p)
        numeric = np.zeros_like(analytic)
        for j in range(N_STATE):
            step = 1e-6 * max(abs(x[j]), 1.0)
            up, down = x.copy(), x.copy()
            up[j] += step
            down[j] -= step
            numeric[:, j] = (drift(up, u, p) - drift(down, u, p)) / (2.0 * step)

        assert np.allclose(analytic, numeric, atol=1e-7, rtol=1e-4)

    def test_matches_with_the_cliff_active(self):
        """Past the critical wear the quadratic term switches on.

        The `max(0, w - w_crit)` makes the derivative piecewise, which is exactly
        where a hand-derived Jacobian goes wrong.
        """
        p = FourCornerParameters()
        u = a_lap_input()
        x = initial_state()
        x[W] = np.array([0.85, 0.80, 0.78, 0.90])
        x[T] = np.full(4, 105.0)

        analytic = jacobian(x, u, p)
        numeric = np.zeros_like(analytic)
        for j in range(N_STATE):
            step = 1e-7 * max(abs(x[j]), 1.0)
            up, down = x.copy(), x.copy()
            up[j] += step
            down[j] -= step
            numeric[:, j] = (drift(up, u, p) - drift(down, u, p)) / (2.0 * step)
        assert np.allclose(analytic, numeric, atol=1e-6, rtol=1e-3)

    def test_fuel_row_is_zero(self):
        """Fuel drains at a rate independent of every state, by construction."""
        p = FourCornerParameters()
        j = jacobian(initial_state(), a_lap_input(), p)
        assert np.allclose(j[I_FUEL, :], 0.0)


class TestDrift:
    def test_fuel_falls_and_track_improves(self):
        p = FourCornerParameters()
        d = drift(initial_state(), a_lap_input(), p)
        assert d[I_FUEL] < 0.0
        assert d[I_TRACK] > 0.0

    def test_a_full_lap_burns_the_expected_fuel(self):
        p = FourCornerParameters()
        u = a_lap_input()
        d = drift(initial_state(), u, p)
        assert d[I_FUEL] * u.duration_s == pytest.approx(-p.session.burn_kg_per_lap)

    def test_asymmetric_loads_produce_asymmetric_wear(self):
        """The whole reason for four states.

        With a left-loaded split the left corners must wear faster. If this
        fails, the model is a scalar with decoration.
        """
        p = FourCornerParameters()
        d = drift(initial_state(), a_lap_input(asymmetric=True), p)
        fl, fr, rl, rr = d[W]
        assert fl > fr
        assert rl > rr

    def test_symmetric_loads_produce_identical_wear(self):
        p = FourCornerParameters()
        d = drift(initial_state(), a_lap_input(asymmetric=False), p)
        assert np.allclose(d[W], d[W][0])


class TestDiffusion:
    def test_is_positive_semidefinite(self):
        p = FourCornerParameters()
        q = diffusion(initial_state(), p)
        assert np.linalg.eigvalsh(q).min() >= 0.0

    def test_wear_noise_grows_with_wear(self):
        """A worn tyre is less predictable than a fresh one."""
        p = FourCornerParameters()
        fresh = diffusion(initial_state(), p)
        worn_state = initial_state()
        worn_state[W] = 0.9
        worn = diffusion(worn_state, p)
        assert worn[0, 0] > fresh[0, 0]


class TestStateHousekeeping:
    def test_clamp_holds_bounds(self):
        x = initial_state()
        x[W] = np.array([-0.5, 0.3, 2.0, 0.1])
        x[T] = np.array([-100.0, 400.0, 90.0, 100.0])
        x[I_FUEL] = -3.0
        x[I_TRACK] = 1.7
        c = clamp_state(x)
        assert (c[W] >= 0).all() and (c[W] <= 1.5).all()
        assert (c[T] >= -20).all() and (c[T] <= 250).all()
        assert c[I_FUEL] >= 0
        assert 0.0 <= c[I_TRACK] <= 1.0

    def test_clamp_does_not_mutate_its_argument(self):
        x = initial_state()
        x[W] = -1.0
        before = x.copy()
        clamp_state(x)
        assert np.array_equal(x, before)

    def test_describe_reports_the_aggregates_the_preregistration_scores(self):
        x = initial_state()
        x[W] = np.array([0.6, 0.2, 0.5, 0.1])
        d = describe(x)
        assert d["wear_left_minus_right"] == pytest.approx(0.4)
        assert d["wear_front_minus_rear"] == pytest.approx(0.1)

    def test_initial_covariance_is_positive_definite(self):
        assert np.linalg.eigvalsh(initial_covariance()).min() > 0.0

    def test_initial_state_does_not_infer_wear_from_age(self):
        """A used set has unknown wear; putting the answer in the prior is the
        failure mode this guards against."""
        assert np.allclose(initial_state(tyre_age_laps=25.0)[W], 0.0)


class TestLapInputValidation:
    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="finite"):
            LapInput(
                duration_s=90.0,
                load_n=np.array([np.nan, 1.0, 1.0, 1.0]),
                power_proxy_w=np.full(4, TYPICAL_POWER),
                airspeed_ms=50.0,
                laps_completed=1.0,
                traffic_index=0.0,
                session_lap=1,
            )

    def test_rejects_wrong_corner_count(self):
        with pytest.raises(ValueError, match="per corner"):
            LapInput(
                duration_s=90.0,
                load_n=np.ones(3),
                power_proxy_w=np.ones(3),
                airspeed_ms=50.0,
                laps_completed=1.0,
                traffic_index=0.0,
                session_lap=1,
            )

    def test_rejects_negative_power(self):
        """A negative proxy means the upstream reduction went wrong, and it must
        surface here rather than as a silent zero inside the wear law."""
        with pytest.raises(ValueError, match="cannot be negative"):
            LapInput(
                duration_s=90.0,
                load_n=np.ones(4),
                power_proxy_w=np.array([-1.0, 1.0, 1.0, 1.0]),
                airspeed_ms=50.0,
                laps_completed=1.0,
                traffic_index=0.0,
                session_lap=1,
            )

    def test_rejects_non_positive_duration(self):
        with pytest.raises(ValueError, match="duration"):
            LapInput(
                duration_s=0.0,
                load_n=np.ones(4),
                power_proxy_w=np.ones(4),
                airspeed_ms=50.0,
                laps_completed=1.0,
                traffic_index=0.0,
                session_lap=1,
            )


def test_parameter_validation():
    with pytest.raises(ValueError, match="thermal capacity"):
        FourCornerParameters().with_thermo(thermal_capacity_j_per_k=0.0)
    with pytest.raises(ValueError, match="negative"):
        FourCornerParameters().with_thermo(kappa=-1.0)
    with pytest.raises(ValueError, match="window width"):
        FourCornerParameters().with_grip(temp_width_c=0.0)
    with pytest.raises(ValueError, match="per corner"):
        FourCornerParameters().with_grip(sensitivity_s=(1.0, 2.0))
