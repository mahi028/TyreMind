"""A truth engine whose structural form is not our estimator's.

Read this before reading any number that comes out of it.

**The honest framing, stated first because it is the point.** We wrote this
generator too. It is *not* the same thing as validating against a third party's
tyre model, and nothing here should be quoted as if it were. What it removes is
the **functional-form advantage** -- the fact that
`tyremind.data.synthetic.generate_session` builds a lap time out of a latent
linear-Gaussian tyre term plus additive fuel, track and traffic terms, which is
the same algebraic shape `tyremind.models.ssm.tyre_ssm` assumes. A generator that
shares the estimator's assumptions cannot falsify it, so exp19 Leg C's margin
could in principle be an artefact of that agreement rather than evidence about
the estimator. This module makes the data-generating form different so the test
can fail. It does not remove self-reference, and the remaining self-reference is
listed in `experiments/PREREGISTRATION_exp20.md` §6.


What is different, concretely
-----------------------------
In `synthetic`, the tyre's time loss is a **parameter**: `compound_rates["SOFT"]
= 0.115` is written down, multiplied by tyre age, and added to a base lap time.
The truth is the number we typed.

Here the tyre's time loss is **not a parameter and is not typed anywhere**. The
chain is:

    track curvature profile + grip level
      -> quasi-steady-state speed profile (cornering, traction, braking limits)
      -> lap time = the line integral of ds/v around the lap
      -> per-corner vertical loads          (physics.dynamics.corner_loads)
      -> frictional power                   (physics.dynamics.frictional_power_proxy)
      -> two-state tread/carcass temperature (physics.thermal.simulate_thermal)
      -> wear rate                          (physics.wear.energy_wear_rate,
                                             physics.thermal.temperature_wear_multiplier)
      -> accumulated wear W
      -> grip mu(W), a NONLINEAR function of accumulated wear
      -> back to the speed profile on the next lap

The "true tyre seconds" on a lap is then a **derived counterfactual**: the lap
time actually achieved minus the lap time the same car would have achieved on the
same lap with an unworn tyre and everything else held fixed. The "true
degradation rate" is the derivative of that quantity with respect to tyre age,
obtained by differencing. Nobody sets it. It comes out at whatever the physics
says, and it varies within a stint because the physics says it should.

Four structural consequences that `synthetic` does not have, and that our
estimator does not assume:

1. **Lap time is not additive in its causes.** It is `∮ ds / v(s)` where every
   cause moves `v`. Fuel, track evolution, traffic and wear therefore *interact*:
   the fuel effect is larger on a worn tyre than a fresh one, because both act
   through the same cornering-speed bottleneck. `lap_truth` carries a
   `decomposition_residual` column measuring exactly how much of the lap time the
   additive decomposition fails to explain. An additive estimator -- ours
   included -- is mis-specified here by construction.
2. **Fuel acts through mass, not through a slope.** Burning 2.7 kg changes the
   load on every tyre, which changes the cornering limit, which changes lap time
   *and* changes the wear rate. There is no `fuel_slope` constant; the emergent
   one is measured after the fact and reported.
3. **Traffic acts through aerodynamics, not through a coefficient.** A lap in
   dirty air loses downforce, which costs cornering speed and also changes how
   hard the tyre is worked. Again: the penalty is emergent, and its value is
   measured, not set.
4. **Degradation is not linear and its curvature is not a cliff term we added.**
   It bends because grip falls nonlinearly in accumulated wear, because the
   thermal state carries across laps within a run (a cold first lap out of the
   blankets grains, per `temperature_wear_multiplier`'s cold branch), and because
   a slower car generates less frictional energy, which feeds back on the wear
   rate.


What is still calibrated, and why that is not the same as setting the answer
---------------------------------------------------------------------------
Two numbers are calibrated rather than derived, and both are declared:

* **`thermal_power_scale`.** `frictional_power_proxy` says in its own docstring
  that its absolute value in watts is not meaningful -- slip velocity is not
  observable, so the proxy carries an unknown constant. Feeding it to a thermal
  model with real heat capacities therefore needs that constant supplied. It is
  chosen so the tread sits around its working window on a representative lap,
  which is the only defensible anchor available.
* **`reference_life_laps`.** Wear is in arbitrary units (see `physics.wear`), so
  the accumulated-wear axis needs a scale before a grip law can be written on it.
  It is set by running one reference lap (MEDIUM compound, mid fuel, unworn) and
  declaring that `reference_life_laps` such laps consume one unit of wear.

These fix the **scale** of the degradation, not its **shape**, not its ordering
across compounds (that comes from `wear.COMPOUND_WEAR_FACTOR`), not its
interaction with fuel, traffic or temperature, and not the value our estimator is
scored against -- which is a derivative of a simulated lap-time curve, not either
of these constants. Setting a lifetime is a weaker assumption than setting a
rate, and the difference is the reason this module exists. It is nonetheless an
assumption, and it is listed in the pre-registration as one.

A third constant, `fuel_burn_kg_per_lap`, is chosen so the *emergent* per-lap
fuel gain lands near the 0.081 s/lap the model ladder hard-codes. That one is a
concession **against** our interest: three of the rungs assume that number and
would otherwise be penalised for a mis-specified constant rather than for their
structure, and structure is the only thing exp20 is trying to measure.


One number, three answers
-------------------------
Because the emergent degradation is nonlinear, "the degradation rate" is not a
single quantity. `PhysicsTruth` therefore publishes three -- the rate on a fresh
set, the stint average, and the best straight-line fit -- and on the soft
compound they can differ by a factor of two. `synthetic` hides that choice by
making degradation piecewise linear with a declared baseline. exp20 scores every
model against all three and reports that the ranking depends on which is meant,
because picking one after seeing the answer is exactly the thing this module
exists to stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tyremind.physics.dynamics import (
    VehicleParameters,
    corner_loads,
    frictional_power_proxy,
    lateral_acceleration,
    longitudinal_acceleration,
)
from tyremind.physics.thermal import ThermalParameters, simulate_thermal
from tyremind.physics.wear import COMPOUND_WEAR_FACTOR, WearParameters, energy_wear_rate

G = 9.81

#: Tyre ages below this count as a "fresh set" when summarising the emergent
#: degradation rate. Five laps is short enough that the nonlinearity has not yet
#: bitten and long enough to average out per-lap thermal and traffic variation.
FRESH_AGE_LAPS = 5.0

#: Peak friction coefficient per compound, before any wear. Softer rubber grips
#: harder and -- through `COMPOUND_WEAR_FACTOR` in `physics.wear`, which this
#: module does not override -- wears faster. The *degradation rates* that result
#: are not set here and are not set anywhere; they emerge, and differ by compound
#: because these two numbers differ by compound.
COMPOUND_PEAK_MU: dict[str, float] = {
    "SOFT": 1.600,
    "MEDIUM": 1.550,
    "HARD": 1.508,
}


@dataclass(frozen=True)
class CircuitLayout:
    """A circuit as a closed sequence of constant-radius corners and straights.

    Deliberately crude geometry: constant-radius arcs joined by straights, with
    the curvature smoothed at the joins so the car is not asked for a step change
    in lateral acceleration. It is enough to produce a speed profile with the
    right structure -- slow corners, fast sweepers, long straights -- which is all
    the wear chain needs. It is not a claim about any real venue.

    Signed turn angles sum to 360 degrees, so the path closes.

    Attributes:
        name: Label.
        corners: `(signed turn angle in degrees, radius in metres, length of the
            straight that follows in metres)` per corner.
        straight_scale: Multiplier on every straight, used to tune total lap
            length without changing the corner mix.
    """

    name: str = "Synthetic Park"
    corners: tuple[tuple[float, float, float], ...] = (
        (+95.0, 45.0, 420.0),
        (-70.0, 85.0, 260.0),
        (+90.0, 30.0, 180.0),
        (+35.0, 150.0, 650.0),
        (-40.0, 140.0, 240.0),
        (+110.0, 38.0, 300.0),
        (-25.0, 150.0, 780.0),
        (+85.0, 60.0, 210.0),
        (+40.0, 120.0, 340.0),
        (-50.0, 95.0, 190.0),
        (+100.0, 25.0, 260.0),
        (-35.0, 145.0, 900.0),
        (+55.0, 70.0, 230.0),
        (-30.0, 130.0, 520.0),
    )
    straight_scale: float = 0.98


@dataclass(frozen=True)
class TrackProfile:
    """Curvature sampled uniformly in arc length.

    Attributes:
        s_m: Arc-length grid, metres, one lap.
        curvature: Signed curvature at each grid point, 1/m.
        ds_m: Grid spacing, metres.
        length_m: Lap length, metres.
        heading_rad: Heading at each grid point, for reconstructing X/Y.
    """

    s_m: np.ndarray
    curvature: np.ndarray
    ds_m: float
    length_m: float
    heading_rad: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.s_m))


def build_track(layout: CircuitLayout | None = None, n_points: int = 200) -> TrackProfile:
    """Sample a layout onto a uniform arc-length grid.

    Args:
        layout: Circuit description.
        n_points: Grid points around one lap. 200 over roughly 4.3 km is a ~21 m
            spacing, fine enough to resolve a 25 m-radius hairpin and coarse
            enough that five speed-profile solutions per lap stay affordable.

    Returns:
        A TrackProfile.

    Raises:
        ValueError: If the layout has no corners or a non-positive radius.
    """
    layout = layout or CircuitLayout()
    if not layout.corners:
        raise ValueError("layout has no corners")

    segments: list[tuple[float, float]] = []  # (length, curvature)
    for angle_deg, radius_m, straight_m in layout.corners:
        if radius_m <= 0:
            raise ValueError(f"corner radius must be positive, got {radius_m}")
        arc = abs(np.deg2rad(angle_deg)) * radius_m
        segments.append((arc, float(np.sign(angle_deg)) / radius_m))
        segments.append((max(straight_m * layout.straight_scale, 0.0), 0.0))

    length = float(sum(seg for seg, _ in segments))
    s = np.linspace(0.0, length, n_points, endpoint=False)

    edges = np.cumsum([seg for seg, _ in segments])
    values = np.array([k for _, k in segments], dtype=float)
    kappa = values[np.clip(np.searchsorted(edges, s, side="right"), 0, len(values) - 1)]

    # Soften corner entry and exit. A step change in curvature demands a step
    # change in lateral acceleration, which no car can do and which would put a
    # spike into every downstream energy figure. Circular smoothing, because the
    # lap is a loop.
    window = 3
    kernel = np.ones(window) / window
    kappa = np.convolve(np.r_[kappa[-window:], kappa, kappa[:window]], kernel, mode="same")[
        window:-window
    ]

    ds = length / n_points
    heading = np.cumsum(kappa) * ds
    return TrackProfile(
        s_m=s, curvature=kappa, ds_m=float(ds), length_m=length, heading_rad=heading
    )


@dataclass(frozen=True)
class PhysicsSessionConfig:
    """Controls for a physics-derived session.

    Every field here is a property of the *car, track or weekend*. None of them is
    a degradation rate, a fuel slope, a traffic coefficient or a track-evolution
    amount in seconds -- those four quantities are outputs of this module, not
    inputs to it, which is the entire difference from `SessionConfig`.

    Attributes:
        n_drivers: Cars on track.
        session_slots: Session length in lap slots.
        runs_per_driver: Runs (tyre sets) per car.
        min_run_laps: Shortest run.
        max_run_laps: Longest run.
        layout: Circuit geometry.
        n_track_points: Arc-length grid resolution.
        telemetry_hz: Rate the lap is resampled to before the physics chain runs.
            10 Hz matches the public telemetry the rest of the project consumes.
        car_mass_kg: Car plus driver, dry.
        fuel_start_kg: Fuel aboard at the start of a run.
        fuel_burn_kg_per_lap: Burn rate, kg/lap. Set so the *emergent* per-lap
            fuel gain lands on the 0.081 s/lap the model ladder assumes. That is
            a deliberate concession against our own interest: three of the rungs
            hard-code 0.081 and would otherwise be penalised for a mis-specified
            constant rather than for their structure, and structure is what this
            experiment is about. 3.0 kg/lap is also the right order for a lap of
            this length -- the repository's 2.7 is a mid-length-circuit figure.
        drag_area_m2: CdA.
        lift_area_m2: ClA.
        engine_power_w: Power at the wheels, which caps straight-line speed.
        max_speed_ms: Hard ceiling on speed.
        compound_mu: Peak friction per compound.
        grip_loss_linear: Fractional grip lost at one unit of accumulated wear,
            linear part.
        grip_loss_cliff: Fractional grip lost at one unit of accumulated wear,
            nonlinear part.
        grip_loss_exponent: Power on the nonlinear part. Greater than one is what
            makes late-stint degradation accelerate without a cliff term being
            written into a lap-time equation.
        min_grip_fraction: Floor on the grip multiplier, so a pathological stint
            cannot drive the cornering limit to zero.
        reference_life_laps: Reference laps of MEDIUM-compound wear that define
            one unit of accumulated wear. A scale, not a rate -- see the module
            docstring.
        thermal_power_scale: Converts the arbitrary-unit frictional power proxy
            into watts for the thermal model. See the module docstring.
        track_grip_gain: Fractional peak-friction gain from a fully rubbered-in
            track. Its value *in seconds* is an output.
        track_grip_shape: Exponential rate at which the track matures.
        traffic_probability: Probability a lap is run in another car's wake.
        traffic_downforce_loss: Fraction of ClA lost at a traffic index of 1.0.
            Its cost *in seconds* is an output.
        driver_commitment_sd: Spread of per-driver grip utilisation across the
            field. Produces the field's pace spread.
        lap_commitment_sd: Per-lap scatter in grip utilisation.
        lap_commitment_df: Student-t degrees of freedom for that scatter.
        timing_noise_sd: Additive timing noise, seconds.
        scrubbed_set_probability: Probability a run starts on a used set.
        full_decomposition: Whether to compute the fuel, track and traffic
            counterfactuals as well as the tyre one. Four extra speed-profile
            solutions per lap; turn off when only the rate truth is needed.
        seed: Random seed.
    """

    n_drivers: int = 20
    session_slots: int = 60
    runs_per_driver: int = 3
    min_run_laps: int = 6
    max_run_laps: int = 16

    layout: CircuitLayout = field(default_factory=CircuitLayout)
    n_track_points: int = 300
    telemetry_hz: float = 10.0

    car_mass_kg: float = 798.0
    fuel_start_kg: float = 60.0
    fuel_burn_kg_per_lap: float = 3.0
    drag_area_m2: float = 1.20
    lift_area_m2: float = 5.40
    engine_power_w: float = 700_000.0
    max_speed_ms: float = 94.0

    compound_mu: dict[str, float] = field(default_factory=lambda: dict(COMPOUND_PEAK_MU))
    grip_loss_linear: float = 0.0697
    grip_loss_cliff: float = 0.012
    grip_loss_exponent: float = 3.0
    min_grip_fraction: float = 0.50
    reference_life_laps: float = 34.0
    thermal_power_scale: float = 0.005

    track_grip_gain: float = 0.027
    track_grip_shape: float = 0.055

    traffic_probability: float = 0.18
    traffic_downforce_loss: float = 0.095

    driver_commitment_sd: float = 0.018
    lap_commitment_sd: float = 0.003
    lap_commitment_df: float = 5.0
    timing_noise_sd: float = 0.05

    scrubbed_set_probability: float = 0.25
    full_decomposition: bool = True

    seed: int = 20260913


@dataclass(frozen=True)
class PhysicsTruth:
    """What actually happened, derived rather than declared.

    Attributes:
        lap_truth: Per-lap decomposition. `true_tyre` is the counterfactual cost
            of the tyre's wear state in seconds; `true_rate` is its derivative
            with respect to tyre age, computed by differencing within a run.
        compound_rates: **Primary truth target.** Lap-weighted mean of the
            per-lap instantaneous rate, per compound, s/lap.
        compound_rate_fresh: The emergent rate on a *fresh* set -- the mean
            instantaneous rate over the first `FRESH_AGE_LAPS` laps of tyre age.
            This is the closest analogue of `synthetic.GroundTruth.compound_rates`,
            which is a baseline rate before any cliff, and therefore the target
            that a model reporting a "baseline" parameter should be scored on.
        compound_rate_slopes: OLS slope of `true_tyre` on `tyre_age`, per
            compound -- the best straight-line approximation to the emergent
            degradation, and therefore the target a model reporting a *linear*
            slope should be scored on.
        compound_rate_curve: Mean instantaneous rate against tyre age, per
            compound. The shape of the emergent degradation, for inspection.

    Three per-compound rates rather than one, because on a nonlinear generator
    "the degradation rate" is not a single well-defined number. A fresh set, the
    stint average and the best linear fit are three different quantities, and on
    this generator they can differ by a factor of two on the soft compound. The
    synthetic generator hides that choice by making degradation piecewise linear
    with a declared baseline; here the choice has to be made explicitly, and
    exp20 reports the ranking under all three rather than picking the flattering
    one.
        fuel_slope: Emergent fuel burn-off gain, s/lap, from regressing
            `true_fuel` on `lap_in_run`.
        traffic_coefficient: Emergent traffic penalty at index 1.0, seconds.
        track_evolution: Emergent track effect per session lap, seconds.
        driver_pace: Emergent per-driver pace offset, seconds.
        wear_reference: Accumulated wear, in `physics.wear` units, that counts as
            one unit on the grip law's axis.
        reference_lap_time: Lap time of the reference lap used for calibration.
    """

    lap_truth: pd.DataFrame
    compound_rates: dict[str, float]
    compound_rate_fresh: dict[str, float]
    compound_rate_slopes: dict[str, float]
    compound_rate_curve: pd.DataFrame
    fuel_slope: float
    traffic_coefficient: float
    track_evolution: pd.DataFrame
    driver_pace: dict[str, float]
    wear_reference: float
    reference_lap_time: float


@dataclass(frozen=True)
class PhysicsSession:
    """A generated session: what an estimator sees, and what really happened.

    Attributes:
        lap_table: Observable data, in the same schema
            `tyremind.data.synthetic.generate_session` emits.
        truth: The derived truth.
        config: Configuration used.
        track: The sampled circuit.
    """

    lap_table: pd.DataFrame
    truth: PhysicsTruth
    config: PhysicsSessionConfig
    track: TrackProfile


# --------------------------------------------------------------------------
# The quasi-steady-state lap simulator
# --------------------------------------------------------------------------


def grip_multiplier(wear_units: float, cfg: PhysicsSessionConfig) -> float:
    """Fraction of peak friction surviving at a given accumulated wear.

        mu(W)/mu_0 = 1 - a * x - b * x^p,        x = W / W_ref

    Nonlinear in accumulated wear, and that nonlinearity is where the emergent
    degradation curve gets its bend. With `p = 3` most of a tyre's life is nearly
    linear and the end of it is not, which is the shape exp17 measured on 2,827
    real stints -- but here it is a property of the *rubber*, not a term added to
    a lap-time equation, so the resulting seconds-per-lap curve is something we
    read off rather than something we wrote. The instantaneous rate at the end of
    a long soft stint comes out around 1.6x its value on a fresh set, and that
    ratio was not chosen either.

    A more defensible functional form probably exists. This one is declared
    rather than fitted, and the pre-registration lists it as an assumption.

    Args:
        wear_units: Accumulated wear in units of `wear_reference`.
        cfg: Session configuration.

    Returns:
        Grip multiplier in `[min_grip_fraction, 1.0]`.
    """
    x = max(float(wear_units), 0.0)
    loss = cfg.grip_loss_linear * x + cfg.grip_loss_cliff * x**cfg.grip_loss_exponent
    return float(np.clip(1.0 - loss, cfg.min_grip_fraction, 1.0))


def speed_profile(
    track: TrackProfile,
    *,
    mu: float,
    mass_kg: float,
    lift_area_m2: float,
    drag_area_m2: float,
    engine_power_w: float,
    max_speed_ms: float,
    air_density: float = 1.225,
) -> np.ndarray:
    """Grip-limited speed around one lap, m/s.

    The standard quasi-steady-state construction, in three stages:

    1. **Cornering limit.** Solve `v^2 kappa = mu (g + rho ClA v^2 / 2m)` for `v`
       at every point. Downforce appears on both sides, which is why fast corners
       have a much higher limit than a constant-mu circle would predict, and why
       a corner above a critical radius is flat out regardless of grip.
    2. **Forward pass.** Walk the lap accelerating out of every corner, limited by
       whichever binds first: the grip left over after cornering (friction
       ellipse) or engine power against drag.
    3. **Backward pass.** Walk it again braking into every corner, limited by grip
       left over after cornering plus the deceleration drag supplies for free.

    Both passes run twice around the loop so a limit set in the final corner
    propagates into the first.

    This is not a lap-time optimiser and makes no claim to be one -- it assumes
    the car is on the friction limit everywhere it can be, which is the classic
    QSS idealisation. What matters for this module is that lap time responds to
    grip, mass and downforce the way a car does: nonlinearly, and through the
    same bottleneck for all three.

    Args:
        track: Sampled circuit.
        mu: Effective peak friction coefficient, already including wear, track
            evolution and driver commitment.
        mass_kg: Car plus fuel.
        lift_area_m2: ClA, already including any dirty-air loss.
        drag_area_m2: CdA.
        engine_power_w: Power at the wheels.
        max_speed_ms: Hard speed ceiling.
        air_density: Ambient air density.

    Returns:
        Speed at each arc-length grid point, m/s.

    Raises:
        ValueError: If `mu` or `mass_kg` is not positive.
    """
    if mu <= 0 or mass_kg <= 0:
        raise ValueError(f"mu and mass must be positive; got mu={mu}, mass={mass_kg}")

    k = np.abs(track.curvature)
    aero_term = mu * air_density * lift_area_m2 / (2.0 * mass_kg)
    denominator = k - aero_term
    with np.errstate(divide="ignore", invalid="ignore"):
        cornering = np.where(denominator > 1e-9, np.sqrt(mu * G / denominator), max_speed_ms)
    v = np.minimum(np.nan_to_num(cornering, nan=max_speed_ms, posinf=max_speed_ms), max_speed_ms)

    # Python floats in the hot loop: numpy scalars are an order of magnitude
    # slower here, and this runs five times per lap for every lap of a session.
    speeds = [float(x) for x in v]
    curve = [float(x) for x in k]
    n = len(speeds)
    ds = track.ds_m
    half_rho = 0.5 * air_density

    # Forward: traction and power out of the corners.
    for _ in range(2):
        for i in range(n):
            j = (i - 1) % n
            u = speeds[j]
            downforce = half_rho * lift_area_m2 * u * u
            grip = mu * (G + downforce / mass_kg)
            lateral = u * u * curve[j]
            spare = grip * grip - lateral * lateral
            traction = np.sqrt(spare) if spare > 0.0 else 0.0
            drag = half_rho * drag_area_m2 * u * u / mass_kg
            power_limited = engine_power_w / (mass_kg * max(u, 5.0)) - drag
            a_x = traction if traction < power_limited else power_limited
            candidate = u * u + 2.0 * a_x * ds
            candidate = np.sqrt(candidate) if candidate > 1.0 else 1.0
            if candidate < speeds[i]:
                speeds[i] = candidate

    # Backward: braking into the corners. Drag helps, so it is added rather than
    # subtracted -- a decelerating car is slowed by the air as well as the tyres.
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            u = speeds[j]
            downforce = half_rho * lift_area_m2 * u * u
            grip = mu * (G + downforce / mass_kg)
            lateral = u * u * curve[j]
            spare = grip * grip - lateral * lateral
            braking = (np.sqrt(spare) if spare > 0.0 else 0.0) + (
                half_rho * drag_area_m2 * u * u / mass_kg
            )
            candidate = u * u + 2.0 * braking * ds
            candidate = np.sqrt(candidate) if candidate > 1.0 else 1.0
            if candidate < speeds[i]:
                speeds[i] = candidate

    return np.asarray(speeds, dtype=float)


def lap_time_from_profile(track: TrackProfile, speed_ms: np.ndarray) -> float:
    """Lap time as the line integral of `ds / v`, seconds.

    Trapezoidal in `1/v` around the closed loop, which is the correct quadrature
    for a quantity sampled uniformly in distance rather than time.
    """
    inverse = 1.0 / np.maximum(np.asarray(speed_ms, dtype=float), 1e-3)
    wrapped = np.r_[inverse, inverse[0]]
    return float(np.trapezoid(wrapped) * track.ds_m)


def telemetry_frame(
    track: TrackProfile, speed_ms: np.ndarray, hz: float = 10.0
) -> pd.DataFrame:
    """Resample a distance-domain solution into a time-domain telemetry table.

    Produces exactly the columns `physics.dynamics.lap_energy` and
    `physics.wear.lap_wear` consume -- `Time`, `Speed` in km/h, `X`/`Y` in tenths
    of a metre -- so the same functions that run on real FastF1 telemetry run on
    this without a special case. That is deliberate: a truth engine that needed
    its own private copy of the physics would be testing the copy.

    Args:
        track: Sampled circuit.
        speed_ms: Speed at each arc-length point, m/s.
        hz: Output sampling rate.

    Returns:
        A telemetry frame, one row per sample, plus a `Curvature` column carrying
        the analytic curvature so callers need not re-differentiate the path.
    """
    v = np.maximum(np.asarray(speed_ms, dtype=float), 1e-3)
    inverse = 1.0 / v
    # Time at each grid point: cumulative trapezoid of 1/v.
    increments = 0.5 * (inverse + np.r_[inverse[1:], inverse[0]]) * track.ds_m
    t_grid = np.r_[0.0, np.cumsum(increments)[:-1]]
    lap_time = float(np.sum(increments))

    n_samples = max(int(np.ceil(lap_time * hz)), 12)
    t = np.linspace(0.0, lap_time, n_samples, endpoint=False)

    speed = np.interp(t, t_grid, v, period=lap_time)
    kappa = np.interp(t, t_grid, track.curvature, period=lap_time)
    heading = np.interp(t, t_grid, track.heading_rad, period=lap_time)

    # Position by integrating heading along the path. Only used for display and
    # for the curvature-recovery test; the physics chain uses `Curvature`.
    step = np.r_[0.0, np.diff(np.interp(t, t_grid, track.s_m, period=lap_time))]
    step = np.where(step < 0, 0.0, step)
    x = np.cumsum(np.cos(heading) * step)
    y = np.cumsum(np.sin(heading) * step)

    return pd.DataFrame(
        {
            "Time": pd.to_timedelta(t, unit="s"),
            "Speed": speed * 3.6,
            "X": x * 10.0,
            "Y": y * 10.0,
            "Curvature": kappa,
        }
    )


@dataclass(frozen=True)
class LapPhysics:
    """What the physics chain produced for one lap.

    Attributes:
        wear_increment: Wear accumulated, in `physics.wear` units.
        energy_mj: Frictional energy per corner, MJ-equivalent.
        end_bulk_c: Carcass temperature at the flag, carried into the next lap.
        mean_surface_c: Mean tread temperature.
        fraction_in_window: Share of the lap inside the working range.
        thermal_stress: Mean squared excursion outside the window, K^2.
    """

    wear_increment: float
    energy_mj: dict[str, float]
    end_bulk_c: float
    mean_surface_c: float
    fraction_in_window: float
    thermal_stress: float


def lap_physics(
    track: TrackProfile,
    speed_ms: np.ndarray,
    *,
    compound: str,
    mass_kg: float,
    lift_area_m2: float,
    cfg: PhysicsSessionConfig,
    initial_bulk_c: float | None = None,
    thermal_params: ThermalParameters | None = None,
    wear_params: WearParameters | None = None,
) -> LapPhysics:
    """Run one lap through loads, frictional power, temperature and wear.

    Every step is the repository's own implementation, called unmodified:
    `corner_loads`, `frictional_power_proxy`, `simulate_thermal` and
    `energy_wear_rate`. The only thing this function adds is
    `cfg.thermal_power_scale`, which supplies the unknown constant the power
    proxy explicitly declares it is missing.

    Args:
        track: Sampled circuit.
        speed_ms: Speed at each arc-length point, m/s.
        compound: Compound in use.
        mass_kg: Car plus fuel.
        lift_area_m2: Effective ClA for this lap.
        cfg: Session configuration.
        initial_bulk_c: Carcass temperature carried in from the previous lap.
            Passing it through a run is what gives the tyre thermal memory --
            without it every lap starts from the blankets and the cold-graining
            branch of `temperature_wear_multiplier` fires on every lap instead of
            only the out-lap.
        thermal_params: Thermal coefficients.
        wear_params: Wear coefficients.

    Returns:
        A LapPhysics summary.
    """
    telemetry = telemetry_frame(track, speed_ms, hz=cfg.telemetry_hz)
    t = telemetry["Time"].dt.total_seconds().to_numpy()
    v = telemetry["Speed"].to_numpy(dtype=float) / 3.6
    kappa = telemetry["Curvature"].to_numpy(dtype=float)

    a_lat = lateral_acceleration(v, kappa)
    a_long = longitudinal_acceleration(v, t)

    vehicle = VehicleParameters(
        mass_kg=cfg.car_mass_kg,
        fuel_mass_kg=max(mass_kg - cfg.car_mass_kg, 0.0),
        drag_area_m2=cfg.drag_area_m2,
        lift_area_m2=lift_area_m2,
    )
    loads = corner_loads(v, a_long, a_lat, vehicle)
    power = frictional_power_proxy(loads, v, a_long, a_lat)
    total_power = sum(power.values()) * cfg.thermal_power_scale

    thermal = simulate_thermal(
        total_power, v, t, thermal_params, initial_bulk_c=initial_bulk_c
    )
    rate = energy_wear_rate(total_power, thermal.surface_c, compound, wear_params)

    return LapPhysics(
        wear_increment=float(np.trapezoid(rate, t)),
        energy_mj={
            corner: float(np.trapezoid(p, t) * cfg.thermal_power_scale / 1.0e6)
            for corner, p in power.items()
        },
        end_bulk_c=float(thermal.bulk_c[-1]),
        mean_surface_c=float(thermal.surface_c.mean()),
        fraction_in_window=thermal.fraction_in_window(),
        thermal_stress=thermal.thermal_stress(),
    )


# --------------------------------------------------------------------------
# Session generation
# --------------------------------------------------------------------------


def _track_grip_factor(session_lap: int, cfg: PhysicsSessionConfig) -> float:
    """Peak-friction multiplier from rubber going down on the track.

    Saturating exponential in *grip*, not in seconds. What it is worth in seconds
    is whatever the speed profile says it is worth, which is neither constant
    across the lap nor independent of the tyre's state -- and that dependence is
    one of the things an additive estimator cannot represent.
    """
    return 1.0 + cfg.track_grip_gain * (1.0 - float(np.exp(-cfg.track_grip_shape * session_lap)))


def _calibrate_wear_reference(
    track: TrackProfile, cfg: PhysicsSessionConfig
) -> tuple[float, float]:
    """One reference lap, to put a scale on the arbitrary wear units.

    MEDIUM compound, unworn, half a tank, a green track and full commitment. The
    wear that lap accumulates, multiplied by `reference_life_laps`, defines one
    unit on the grip law's axis.

    Returns:
        `(wear_reference, reference_lap_time)`.
    """
    mass = cfg.car_mass_kg + cfg.fuel_start_kg - 0.5 * cfg.fuel_burn_kg_per_lap * 8.0
    mu = cfg.compound_mu["MEDIUM"]
    speeds = speed_profile(
        track,
        mu=mu,
        mass_kg=mass,
        lift_area_m2=cfg.lift_area_m2,
        drag_area_m2=cfg.drag_area_m2,
        engine_power_w=cfg.engine_power_w,
        max_speed_ms=cfg.max_speed_ms,
    )
    physics = lap_physics(
        track,
        speeds,
        compound="MEDIUM",
        mass_kg=mass,
        lift_area_m2=cfg.lift_area_m2,
        cfg=cfg,
    )
    reference = float(physics.wear_increment) * float(cfg.reference_life_laps)
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError(
            "reference lap produced no wear; the thermal or dynamics chain is "
            f"degenerate (wear_increment={physics.wear_increment})"
        )
    return reference, lap_time_from_profile(track, speeds)


def _instantaneous_rates(lap_truth: pd.DataFrame) -> np.ndarray:
    """Derivative of `true_tyre` with respect to `tyre_age`, within each run.

    `np.gradient` rather than a fitted slope: the whole point is that the rate
    changes along the stint, so anything that summarises the run to one number
    would throw away the quantity being measured. Runs with a single lap get NaN,
    which is honest -- one point has no derivative.
    """
    rates = np.full(len(lap_truth), np.nan)
    for _, block in lap_truth.groupby("run_id", sort=False):
        order = block.sort_values("tyre_age")
        age = order["tyre_age"].to_numpy(dtype=float)
        loss = order["true_tyre"].to_numpy(dtype=float)
        if len(order) < 2 or np.ptp(age) <= 0:
            continue
        rates[order.index.to_numpy()] = np.gradient(loss, age)
    return rates


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope, or NaN if the fit is not possible.

    Guarded: `np.polyfit` raises `LinAlgError` on degenerate input, and an
    unguarded call has killed two runs in this repository.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3 or np.ptp(x[good]) <= 0:
        return float("nan")
    try:
        return float(np.polyfit(x[good], y[good], 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")


def generate_session(config: PhysicsSessionConfig | None = None) -> PhysicsSession:
    """Generate one confounded session whose truth is derived, not declared.

    For each lap the simulator solves the speed profile up to five times:

    * with everything as it actually was -> the observed lap time;
    * with the tyre unworn -> `true_tyre`, the counterfactual cost of wear;
    * with a full tank -> `true_fuel`;
    * on a green track -> `true_track`;
    * in clean air -> `true_traffic`.

    Those four counterfactuals do **not** sum to the total, because the lap time
    is not additive in them. The shortfall is recorded per lap as
    `decomposition_residual`, and it is the cleanest single measure of how
    mis-specified an additive estimator is on this data.

    Args:
        config: Session parameters.

    Returns:
        A PhysicsSession.

    Raises:
        ValueError: If the configuration produces no laps at all.
    """
    cfg = config or PhysicsSessionConfig()
    rng = np.random.default_rng(cfg.seed)
    track = build_track(cfg.layout, n_points=cfg.n_track_points)
    wear_reference, reference_lap_time = _calibrate_wear_reference(track, cfg)

    drivers = [f"CAR{i + 1:02d}" for i in range(cfg.n_drivers)]
    commitment = {
        d: float(np.clip(rng.normal(1.0, cfg.driver_commitment_sd), 0.90, 1.05)) for d in drivers
    }
    compounds = list(cfg.compound_mu)

    def solve(mu: float, mass: float, cla: float) -> np.ndarray:
        return speed_profile(
            track,
            mu=mu,
            mass_kg=mass,
            lift_area_m2=cla,
            drag_area_m2=cfg.drag_area_m2,
            engine_power_w=cfg.engine_power_w,
            max_speed_ms=cfg.max_speed_ms,
        )

    rows: list[dict] = []
    truth_rows: list[dict] = []
    run_counter = 0

    for driver in drivers:
        cursor = int(rng.integers(0, max(1, cfg.session_slots // 6)))

        for _ in range(cfg.runs_per_driver):
            run_length = int(rng.integers(cfg.min_run_laps, cfg.max_run_laps + 1))
            if cursor + run_length >= cfg.session_slots:
                break

            run_counter += 1
            compound = str(rng.choice(compounds))
            peak_mu = float(cfg.compound_mu[compound])

            starting_age = (
                float(rng.integers(3, 10)) if rng.random() < cfg.scrubbed_set_probability else 0.0
            )
            # A scrubbed set arrives with wear already on it. Charge it the wear a
            # reference lap costs, scaled by the compound, so tyre age and
            # laps-in-run are decoupled in wear as well as in the label.
            wear = (
                starting_age
                * COMPOUND_WEAR_FACTOR.get(compound, 1.0)
                / max(cfg.reference_life_laps, 1e-9)
            )
            bulk_c: float | None = None

            for lap_in_run in range(run_length):
                session_lap = cursor + lap_in_run
                tyre_age = starting_age + lap_in_run

                mass = cfg.car_mass_kg + max(
                    cfg.fuel_start_kg - cfg.fuel_burn_kg_per_lap * lap_in_run, 0.0
                )
                traffic_index = (
                    float(rng.beta(2.0, 3.0)) if rng.random() < cfg.traffic_probability else 0.0
                )
                cla = cfg.lift_area_m2 * (1.0 - cfg.traffic_downforce_loss * traffic_index)

                lap_commitment = commitment[driver] * (
                    1.0
                    + float(rng.standard_t(cfg.lap_commitment_df)) * cfg.lap_commitment_sd
                )
                track_factor = _track_grip_factor(session_lap, cfg)
                base_mu = peak_mu * track_factor * lap_commitment

                mu_worn = base_mu * grip_multiplier(wear / wear_reference, cfg)
                speeds = solve(mu_worn, mass, cla)
                clean_time = lap_time_from_profile(track, speeds)
                noise = float(rng.normal(0.0, cfg.timing_noise_sd))
                lap_time = clean_time + noise

                # The tyre counterfactual: same car, same fuel, same air, same
                # track, unworn rubber. This is the quantity every model in the
                # ladder is trying to recover, and it is a simulation output.
                fresh_time = lap_time_from_profile(track, solve(base_mu, mass, cla))
                true_tyre = clean_time - fresh_time

                if cfg.full_decomposition:
                    full_tank = cfg.car_mass_kg + cfg.fuel_start_kg
                    true_fuel = clean_time - lap_time_from_profile(
                        track, solve(mu_worn, full_tank, cla)
                    )
                    green_mu = mu_worn / track_factor
                    true_track = clean_time - lap_time_from_profile(
                        track, solve(green_mu, mass, cla)
                    )
                    true_traffic = (
                        clean_time
                        - lap_time_from_profile(track, solve(mu_worn, mass, cfg.lift_area_m2))
                        if traffic_index > 0
                        else 0.0
                    )
                else:
                    true_fuel = true_track = true_traffic = float("nan")

                physics = lap_physics(
                    track,
                    speeds,
                    compound=compound,
                    mass_kg=mass,
                    lift_area_m2=cla,
                    cfg=cfg,
                    initial_bulk_c=bulk_c,
                )
                bulk_c = physics.end_bulk_c
                wear += physics.wear_increment

                rows.append(
                    {
                        "driver": driver,
                        "session_lap": session_lap,
                        "run_id": run_counter,
                        "tyre_age": float(tyre_age),
                        "lap_in_run": lap_in_run,
                        "lap_time": float(lap_time),
                        "compound": compound,
                        "traffic_index": traffic_index,
                    }
                )
                truth_rows.append(
                    {
                        "driver": driver,
                        "session_lap": session_lap,
                        "run_id": run_counter,
                        "compound": compound,
                        "tyre_age": float(tyre_age),
                        "lap_in_run": lap_in_run,
                        "lap_time": float(lap_time),
                        "clean_lap_time": float(clean_time),
                        "fresh_tyre_lap_time": float(fresh_time),
                        "true_tyre": float(true_tyre),
                        "true_fuel": float(true_fuel),
                        "true_track": float(true_track),
                        "true_traffic": float(true_traffic),
                        "true_noise": noise,
                        "wear_units": float(wear / wear_reference),
                        "grip_fraction": grip_multiplier(wear / wear_reference, cfg),
                        "mean_surface_c": physics.mean_surface_c,
                        "fraction_in_window": physics.fraction_in_window,
                        "thermal_stress": physics.thermal_stress,
                        "energy_mj": float(sum(physics.energy_mj.values())),
                    }
                )

            cursor += run_length + int(rng.integers(2, 6))

    if not rows:
        raise ValueError(
            "generated an empty session; session_slots is too small for the "
            "requested runs_per_driver and run length range"
        )

    lap_table = pd.DataFrame(rows).sort_values(["session_lap", "driver"]).reset_index(drop=True)
    lap_truth = pd.DataFrame(truth_rows).sort_values(["session_lap", "driver"]).reset_index(
        drop=True
    )
    lap_truth["true_rate"] = _instantaneous_rates(lap_truth)

    # How much of the lap time the additive decomposition fails to account for.
    # Strip each cause from the clean lap time in turn; whatever is left should be
    # that car's base pace on that compound, and any variation in it is
    # interaction the additive form cannot carry. Centred per (driver, compound),
    # because both of those are constants an estimator is entitled to absorb into
    # an intercept -- a softer compound is simply faster, and counting that as
    # mis-specification would overstate the residual tenfold.
    if cfg.full_decomposition:
        reference = (
            lap_truth["clean_lap_time"]
            - lap_truth["true_tyre"]
            - lap_truth["true_fuel"]
            - lap_truth["true_track"]
            - lap_truth["true_traffic"]
        )
        lap_truth["decomposition_residual"] = reference - reference.groupby(
            [lap_truth["driver"], lap_truth["compound"]]
        ).transform("mean")
    else:
        lap_truth["decomposition_residual"] = float("nan")

    scored = lap_truth[np.isfinite(lap_truth["true_rate"].to_numpy(dtype=float))]
    compound_rates = {
        str(c): float(block["true_rate"].mean()) for c, block in scored.groupby("compound")
    }
    fresh = scored[scored["tyre_age"] < FRESH_AGE_LAPS]
    compound_rate_fresh = {
        str(c): float(block["true_rate"].mean()) for c, block in fresh.groupby("compound")
    }
    compound_rate_slopes = {
        str(c): _ols_slope(block["tyre_age"].to_numpy(), block["true_tyre"].to_numpy())
        for c, block in lap_truth.groupby("compound")
    }
    curve = (
        scored.groupby(["compound", "tyre_age"])["true_rate"]
        .mean()
        .reset_index()
        .sort_values(["compound", "tyre_age"])
        .reset_index(drop=True)
    )

    # Emergent confounder magnitudes, measured the way an analyst would measure
    # them if they could see the truth. None of these was an input.
    fuel_slope = -_ols_slope(
        lap_truth["lap_in_run"].to_numpy(), lap_truth["true_fuel"].to_numpy()
    )
    with_traffic = lap_table["traffic_index"].to_numpy(dtype=float)
    traffic_coefficient = _ols_slope(with_traffic, lap_truth["true_traffic"].to_numpy())

    session_laps = sorted(lap_table["session_lap"].unique().tolist())
    track_effect = (
        lap_truth.groupby("session_lap")["true_track"].mean().reindex(session_laps).to_numpy()
    )
    track_evolution = pd.DataFrame(
        {"session_lap": session_laps, "true_track_effect": track_effect}
    )

    # Driver pace, as the counterfactual it actually is: the same reference lap
    # driven at this car's grip utilisation instead of the field's nominal one.
    # Taking the mean of each driver's observed laps instead would fold in
    # whatever tyre state and fuel load they happened to run, which is how a
    # 0.4-second field spread reads as 2.3 seconds.
    reference_mass = cfg.car_mass_kg + cfg.fuel_start_kg - 0.5 * cfg.fuel_burn_kg_per_lap * 8.0
    nominal_mu = float(cfg.compound_mu["MEDIUM"])
    nominal_time = lap_time_from_profile(
        track, solve(nominal_mu, reference_mass, cfg.lift_area_m2)
    )
    driver_pace = {
        str(d): float(
            lap_time_from_profile(track, solve(nominal_mu * c, reference_mass, cfg.lift_area_m2))
            - nominal_time
        )
        for d, c in commitment.items()
    }

    truth = PhysicsTruth(
        lap_truth=lap_truth,
        compound_rates=compound_rates,
        compound_rate_fresh=compound_rate_fresh,
        compound_rate_slopes=compound_rate_slopes,
        compound_rate_curve=curve,
        fuel_slope=fuel_slope,
        traffic_coefficient=traffic_coefficient,
        track_evolution=track_evolution,
        driver_pace=driver_pace,
        wear_reference=wear_reference,
        reference_lap_time=reference_lap_time,
    )
    return PhysicsSession(lap_table=lap_table, truth=truth, config=cfg, track=track)
