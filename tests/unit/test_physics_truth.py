"""The physics truth engine only earns its name if two things are true.

**It must be usable.** Its lap table has to be the same schema
`tyremind.data.synthetic.generate_session` emits, or no model in the ladder can
be scored on it and the whole experiment is a category error.

**It must be structurally different.** The point of the module is that the tyre's
time loss is *derived* rather than *declared*, so the tests that matter are the
ones that would fail if someone quietly reintroduced a degradation parameter: the
rate has to move with tyre age, move with compound in the order the wear
literature says, respond to a change in the *car* rather than to a change in a
rate constant, and be reproducible from the physics rather than from a lookup.

These are cheap sessions -- a handful of cars and short runs -- because the
question is whether the chain is wired up correctly, not whether a full session
reproduces a particular number.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tyremind.data.physics_truth import (
    COMPOUND_PEAK_MU,
    CircuitLayout,
    PhysicsSessionConfig,
    build_track,
    generate_session,
    grip_multiplier,
    lap_physics,
    lap_time_from_profile,
    speed_profile,
    telemetry_frame,
)
from tyremind.data.synthetic import SessionConfig
from tyremind.data.synthetic import generate_session as generate_synthetic_session

#: Small enough to run in seconds, large enough that a run reaches an age where
#: the emergent degradation has visibly bent.
SMALL = PhysicsSessionConfig(
    n_drivers=4,
    runs_per_driver=2,
    session_slots=40,
    min_run_laps=8,
    max_run_laps=14,
    seed=20260913,
)


@pytest.fixture(scope="module")
def session():
    return generate_session(SMALL)


# --------------------------------------------------------------------------
# Track and speed profile
# --------------------------------------------------------------------------


def test_the_track_closes() -> None:
    """Signed turn angles sum to a full turn, so the path is a loop."""
    total = sum(angle for angle, _, _ in CircuitLayout().corners)
    assert total == pytest.approx(360.0)


def test_curvature_is_recoverable_from_the_path() -> None:
    """The X/Y the generator emits must differentiate back to the curvature it used.

    This is the test that keeps the emitted telemetry honest. The physics chain
    consumes the analytic `Curvature` column, so a broken path integration would
    never show up in a lap time -- but it would show up the moment anybody ran
    `dynamics.path_curvature` on the output, which is what every real-telemetry
    code path in this repository does.
    """
    from tyremind.physics.dynamics import path_curvature

    cfg = replace(SMALL, n_track_points=600)
    track = build_track(cfg.layout, cfg.n_track_points)
    speeds = speed_profile(
        track,
        mu=1.55,
        mass_kg=850.0,
        lift_area_m2=cfg.lift_area_m2,
        drag_area_m2=cfg.drag_area_m2,
        engine_power_w=cfg.engine_power_w,
        max_speed_ms=cfg.max_speed_ms,
    )
    telemetry = telemetry_frame(track, speeds, hz=cfg.telemetry_hz)

    recovered = path_curvature(
        telemetry["X"].to_numpy(dtype=float) / 10.0,
        telemetry["Y"].to_numpy(dtype=float) / 10.0,
    )
    analytic = telemetry["Curvature"].to_numpy(dtype=float)

    # Smoothing in both directions flattens the peaks, so the agreement worth
    # asserting is in shape, not amplitude.
    interior = slice(20, -20)
    assert np.corrcoef(recovered[interior], analytic[interior])[0, 1] > 0.9


def test_less_grip_costs_lap_time_and_more_downforce_gains_it() -> None:
    """The speed profile has to respond to the two things wear and traffic move."""
    track = build_track(SMALL.layout, 200)

    def lap(mu: float, cla: float, mass: float = 850.0) -> float:
        return lap_time_from_profile(
            track,
            speed_profile(
                track,
                mu=mu,
                mass_kg=mass,
                lift_area_m2=cla,
                drag_area_m2=SMALL.drag_area_m2,
                engine_power_w=SMALL.engine_power_w,
                max_speed_ms=SMALL.max_speed_ms,
            ),
        )

    base = lap(1.55, SMALL.lift_area_m2)
    assert lap(1.45, SMALL.lift_area_m2) > base  # worn tyre is slower
    assert lap(1.55, SMALL.lift_area_m2 * 0.9) > base  # dirty air is slower
    assert lap(1.55, SMALL.lift_area_m2, mass=820.0) < base  # less fuel is faster


def test_speed_profile_rejects_impossible_inputs() -> None:
    track = build_track(SMALL.layout, 120)
    with pytest.raises(ValueError, match="must be positive"):
        speed_profile(
            track,
            mu=0.0,
            mass_kg=850.0,
            lift_area_m2=5.4,
            drag_area_m2=1.2,
            engine_power_w=7e5,
            max_speed_ms=90.0,
        )


def test_build_track_rejects_a_degenerate_layout() -> None:
    with pytest.raises(ValueError, match="radius must be positive"):
        build_track(CircuitLayout(corners=((90.0, 0.0, 100.0),)))


# --------------------------------------------------------------------------
# The wear chain
# --------------------------------------------------------------------------


def test_grip_falls_nonlinearly_with_accumulated_wear() -> None:
    """The second half of a tyre's life must cost more grip than the first.

    If this ever becomes linear, the generator has stopped being structurally
    different from `synthetic` in the way that matters and exp20's whole claim
    goes with it.
    """
    first_half = grip_multiplier(0.0, SMALL) - grip_multiplier(0.5, SMALL)
    second_half = grip_multiplier(0.5, SMALL) - grip_multiplier(1.0, SMALL)
    assert second_half > first_half
    assert grip_multiplier(0.0, SMALL) == pytest.approx(1.0)
    assert grip_multiplier(1e6, SMALL) == pytest.approx(SMALL.min_grip_fraction)


def test_a_softer_compound_wears_faster_at_identical_load() -> None:
    """Compound ordering comes from `physics.wear`, not from anything set here."""
    track = build_track(SMALL.layout, 200)
    speeds = speed_profile(
        track,
        mu=1.55,
        mass_kg=850.0,
        lift_area_m2=SMALL.lift_area_m2,
        drag_area_m2=SMALL.drag_area_m2,
        engine_power_w=SMALL.engine_power_w,
        max_speed_ms=SMALL.max_speed_ms,
    )
    wear = {
        compound: lap_physics(
            track,
            speeds,
            compound=compound,
            mass_kg=850.0,
            lift_area_m2=SMALL.lift_area_m2,
            cfg=SMALL,
        ).wear_increment
        for compound in ("SOFT", "MEDIUM", "HARD")
    }
    assert wear["SOFT"] > wear["MEDIUM"] > wear["HARD"] > 0


def test_the_tread_runs_near_its_working_window() -> None:
    """`thermal_power_scale` is a calibration, and this is what it is calibrated to.

    If the scale drifts, `temperature_wear_multiplier` leaves its quadratic
    neighbourhood and the wear chain stops being a tyre model and starts being an
    exponential. Worth a guard rather than a comment.
    """
    track = build_track(SMALL.layout, 200)
    speeds = speed_profile(
        track,
        mu=1.55,
        mass_kg=850.0,
        lift_area_m2=SMALL.lift_area_m2,
        drag_area_m2=SMALL.drag_area_m2,
        engine_power_w=SMALL.engine_power_w,
        max_speed_ms=SMALL.max_speed_ms,
    )
    physics = lap_physics(
        track,
        speeds,
        compound="MEDIUM",
        mass_kg=850.0,
        lift_area_m2=SMALL.lift_area_m2,
        cfg=SMALL,
    )
    assert 70.0 < physics.mean_surface_c < 130.0
    assert physics.fraction_in_window > 0.4


# --------------------------------------------------------------------------
# The session, and the schema the ladder needs
# --------------------------------------------------------------------------


def test_the_lap_table_matches_the_synthetic_schema(session) -> None:
    """Any column `synthetic` emits and the ladder reads must be present here.

    Sector times are the one documented exception: nothing in
    `extended_ladder()` consumes them, and the sector model has its own
    experiment on its own generator.
    """
    reference = generate_synthetic_session(replace(SessionConfig(), seed=7, n_drivers=3))
    expected = set(reference.lap_table.columns) - {"sector_1", "sector_2", "sector_3"}
    assert expected <= set(session.lap_table.columns)

    table = session.lap_table
    assert not table["lap_time"].isna().any()
    assert table["tyre_age"].min() >= 0
    assert table["traffic_index"].between(0.0, 1.0).all()
    assert table["lap_in_run"].min() == 0


def test_tyre_age_is_non_decreasing_within_a_run(session) -> None:
    """`fit_tyre_ssm` validates this and raises otherwise, which would fail the arm."""
    monotone = session.lap_table.groupby("run_id")["tyre_age"].apply(
        lambda s: s.is_monotonic_increasing
    )
    assert monotone.all()


def test_the_ssm_accepts_the_table(session) -> None:
    """The end-to-end contract: our own estimator must fit this without special-casing."""
    from tyremind.models.baselines import TyreStateModel

    model = TyreStateModel().fit(session.lap_table)
    rates = model.compound_rates()
    assert set(rates) <= set(session.lap_table["compound"].unique())
    assert all(np.isfinite(mean) for mean, _ in rates.values())


def test_the_same_seed_gives_the_same_session() -> None:
    a = generate_session(replace(SMALL, seed=99))
    b = generate_session(replace(SMALL, seed=99))
    pd.testing.assert_frame_equal(a.lap_table, b.lap_table)
    assert a.truth.compound_rates == b.truth.compound_rates


def test_an_impossible_configuration_raises() -> None:
    with pytest.raises(ValueError, match="empty session"):
        generate_session(replace(SMALL, session_slots=3, min_run_laps=8, max_run_laps=8))


# --------------------------------------------------------------------------
# The truth is derived, not declared
# --------------------------------------------------------------------------


def test_no_degradation_rate_is_an_input(session) -> None:
    """The config must not contain the answer.

    A regression guard with teeth: if someone adds a `compound_rates` field to
    `PhysicsSessionConfig`, the generator has silently become `synthetic` with
    extra steps and exp20 stops measuring anything.
    """
    fields = set(vars(session.config))
    assert "compound_rates" not in fields
    assert "cliff_onset" not in fields
    assert "fuel_slope" not in fields
    assert "traffic_coefficient" not in fields
    assert "track_evolution_total" not in fields


def test_the_emergent_rate_orders_the_compounds_correctly(session) -> None:
    rates = session.truth.compound_rates
    assert rates["SOFT"] > rates["MEDIUM"] > rates["HARD"] > 0


def test_the_emergent_rate_rises_with_tyre_age(session) -> None:
    """The degradation curve bends, and it bends because the physics bends it.

    `synthetic` has to be told to bend, with a cliff onset and a severity. Here
    the bend is what falls out of a nonlinear grip law and a thermal state that
    carries across laps.
    """
    curve = session.truth.compound_rate_curve
    block = curve[curve["compound"] == "SOFT"].sort_values("tyre_age")
    assert len(block) >= 6
    third = max(len(block) // 3, 1)
    early = float(block["true_rate"].iloc[:third].mean())
    late = float(block["true_rate"].iloc[-third:].mean())
    assert late > early * 1.2


def test_the_three_rate_targets_disagree(session) -> None:
    """A fresh set, the stint average and the best linear fit are different numbers.

    This is the reason exp20 scores all three. If they ever collapse onto each
    other the generator has gone linear, and the experiment's target-sensitivity
    finding would be vacuous.
    """
    truth = session.truth
    fresh = truth.compound_rate_fresh["SOFT"]
    mean = truth.compound_rates["SOFT"]
    slope = truth.compound_rate_slopes["SOFT"]
    assert fresh < mean < slope


def test_true_tyre_is_zero_on_a_brand_new_set(session) -> None:
    """The counterfactual is defined against an unworn tyre, so age zero costs nothing."""
    lap_truth = session.truth.lap_truth
    fresh = lap_truth[lap_truth["tyre_age"] == 0.0]
    assert len(fresh) > 0
    assert fresh["true_tyre"].abs().max() < 1e-9


def test_true_tyre_accumulates_within_a_run(session) -> None:
    """Wear only goes one way, so the cost of the tyre only goes one way.

    Asserted on a noise-free session. With per-lap commitment jitter the observed
    and unworn counterfactual laps are perturbed by slightly different amounts,
    so `true_tyre` can dip by a millisecond between consecutive laps -- which is
    measurement noise in the counterfactual, not the tyre recovering grip.
    Traffic is switched off for the same reason: dirty air changes the marginal
    cost of grip, so the counterfactual gap can genuinely narrow on a lap spent
    behind another car. Accumulated wear is monotone unconditionally, and that is
    checked too.
    """
    quiet = generate_session(
        replace(
            SMALL,
            lap_commitment_sd=0.0,
            timing_noise_sd=0.0,
            traffic_probability=0.0,
            seed=4242,
        )
    )
    for _, block in quiet.truth.lap_truth.sort_values(["run_id", "tyre_age"]).groupby("run_id"):
        assert block["true_tyre"].is_monotonic_increasing

    for _, block in session.truth.lap_truth.sort_values(["run_id", "tyre_age"]).groupby("run_id"):
        assert block["wear_units"].is_monotonic_increasing


def test_the_confounders_are_outputs_with_plausible_magnitudes(session) -> None:
    """Fuel, traffic and track evolution are measured after the fact, not set.

    The bounds are wide on purpose. A tight assertion here would be asserting
    that the calibration has not changed, which is not what this test is for --
    it is for catching a chain that has come unwired and is emitting zero.
    """
    truth = session.truth
    assert 0.03 < truth.fuel_slope < 0.20
    assert 0.3 < truth.traffic_coefficient < 3.0
    total_track = -float(truth.track_evolution["true_track_effect"].min())
    assert 0.2 < total_track < 2.5


def test_the_additive_decomposition_leaves_a_residual(session) -> None:
    """Lap time is a line integral, so its causes do not add up. That is the point.

    A zero residual would mean the generator had become additive after all, and
    with it the main structural difference from `synthetic`.
    """
    residual = session.truth.lap_truth["decomposition_residual"]
    assert np.isfinite(residual).all()
    assert residual.std() > 1e-3


def test_a_faster_wearing_car_degrades_faster_without_touching_any_rate() -> None:
    """Change the *car*, not a rate constant, and the emergent degradation moves.

    This is the sharpest available demonstration that the rate is derived. A
    softer peak friction across the board means more grip, higher cornering
    speeds, more frictional energy through the contact patch and therefore more
    wear per lap -- and the degradation rate follows, although nothing called a
    rate was edited.
    """
    grippier = {c: mu * 1.04 for c, mu in COMPOUND_PEAK_MU.items()}
    base = generate_session(SMALL)
    moved = generate_session(replace(SMALL, compound_mu=grippier))
    assert set(base.truth.compound_rates) == set(moved.truth.compound_rates)
    for compound, rate in base.truth.compound_rates.items():
        assert moved.truth.compound_rates[compound] != pytest.approx(rate, rel=1e-3)


def test_a_shorter_tyre_life_scales_the_rate_but_not_the_shape() -> None:
    """`reference_life_laps` is a scale, and the test says so out loud.

    Shortening the reference life must raise every compound's degradation rate and
    must not reorder the compounds: what the constant sets is how long a tyre
    lasts, and the rest -- which compound degrades faster, and how the rate grows
    along a stint -- still comes from the wear law and the grip law.

    It deliberately does **not** assert a uniform scaling. The grip law is cubic
    in accumulated wear, so a shorter life pushes a soft tyre further up the
    nonlinear part of the curve and its rate rises by more than a hard tyre's.
    That asymmetry is the shape the constant does not control, and asserting it
    away would be asserting the generator is linear.
    """
    base = generate_session(SMALL)
    short = generate_session(replace(SMALL, reference_life_laps=25.0))
    for compound, rate in base.truth.compound_rates.items():
        assert short.truth.compound_rates[compound] > rate
    assert (
        short.truth.compound_rates["SOFT"]
        > short.truth.compound_rates["MEDIUM"]
        > short.truth.compound_rates["HARD"]
    )
