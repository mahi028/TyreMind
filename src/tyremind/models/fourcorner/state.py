"""The 10-dimensional state of a four-corner tyre model, and its parameters.

The shipped estimator tracks two numbers per car: how much performance the tyre
has lost, and how fast it is losing it. That is deliberately the smallest state
that can express a cliff, and `src/tyremind/models/ssm/tyre_ssm.py` explains why
it was kept that small.

This module is the other end of the trade. A real car has four tyres, each
carrying a different vertical load through every corner, each running its own
temperature, and each losing grip on its own schedule. A clockwise circuit works
the left-hand pair harder all afternoon. None of that is expressible in a scalar.

    x = [w_FL w_FR w_RL w_RR  T_FL T_FR T_RL T_RR  m_fuel  D_track]

**What is estimated and what is measured.** The corner loads `F_z,i` are *not*
states. They are computed from speed and acceleration by
`physics.dynamics.corner_loads`, which is a rigid-body calculation with no free
parameters, and they enter as inputs. That distinction is what makes a four-corner
model worth attempting at all: a scalar lap time cannot separate four wear states
on its own, so the separation has to come from measured load asymmetry rather
than from the likelihood.

`PREREGISTRATION_exp35.md` records, before any of this was run, the prediction
that the corner states will turn out **not** to be separately identifiable and
that only a left/right aggregate will survive. This module is written so that
prediction can be tested rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

#: Corner order, fixed once. Every array in this package is in this order and
#: several of the Jacobians index positionally, so changing it silently breaks
#: the filter rather than raising.
CORNERS: tuple[str, str, str, str] = ("FL", "FR", "RL", "RR")

#: State vector layout. Indices are module constants rather than magic numbers
#: because the Jacobian blocks are written by hand and an off-by-one there
#: produces a filter that runs and is wrong, which is the worst failure mode.
W = slice(0, 4)       # normalised wear, one per corner
T = slice(4, 8)       # tread temperature, degrees C
I_FUEL = 8            # fuel mass, kg
I_TRACK = 9           # track evolution progress, dimensionless 0..1
N_STATE = 10

#: Index of each corner inside the W and T blocks.
CORNER_INDEX = {name: i for i, name in enumerate(CORNERS)}

#: Reference per-corner frictional power, used to non-dimensionalise the wear
#: law so `kappa` is a small ordinary number rather than something near 1e-9.
#:
#: Measured, not guessed: the median of `energy_mj_* / lap_time` across the
#: telemetry corpus is 1.81e5 in the proxy's units. It is a *scale*, not a
#: physical wattage -- `dynamics.frictional_power` explains why the proxy has no
#: absolute meaning -- and any change to it is absorbed by the fitted `kappa`.
REFERENCE_CORNER_POWER_W = 1.81e5

#: Wear is a fraction of usable life. Clamped because a Kalman update is a linear
#: correction and can push a bounded quantity outside its bounds; letting that
#: happen produces negative grip and a lap time of minus infinity.
WEAR_MIN, WEAR_MAX = 0.0, 1.5

#: Temperatures outside this are not physical for a slick in dry running and
#: indicate the filter has diverged.
TEMP_MIN_C, TEMP_MAX_C = -20.0, 250.0


@dataclass(frozen=True)
class ThermoMechanicalParameters:
    """Physical constants of the rubber and the thermal path.

    Split from `GripParameters` because these are the ones a hierarchical prior
    pools across sessions by compound: a C3 does not become a different rubber
    when the transporter reaches Monza. `GripParameters` holds the ones that are
    genuinely per-circuit or per-car.

    Attributes:
        kappa: Wear coefficient. Multiplies normalised frictional power. The
            default puts a typical stint near half its usable life after thirty
            laps, which is where the reference power above was calibrated to put
            it; it is fitted in practice.
        alpha: Wear exponent. 1.0 is Archard-linear; above 1 makes hard laps
            disproportionately expensive, which is the usual finding in the
            friction literature and the reason it is fitted rather than fixed.
        thermal_capacity_j_per_k: Lumped tread heat capacity, J/K.
        heat_fraction: Share of frictional power that reaches the tread rather
            than the track or the air. The default is set so a corner running at
            the reference power settles near the thermal optimum of 105 C rather
            than somewhere the tyre would never work, which matters because grip
            falls off a Gaussian either side of it.
        conduction_w_per_k: Conduction to the carcass.
        convection_base_w_per_k: Convective cooling at rest.
        convection_speed_w_per_k_per_ms: Extra convection per m/s of airspeed.
        emissivity: Radiative emissivity of the tread.
        carcass_temp_c: Carcass temperature, treated as a slow reservoir.
        ambient_temp_c: Air temperature.
        track_temp_c: Track surface temperature, the radiative sink.
    """

    kappa: float = 1.8e-4
    alpha: float = 1.0
    thermal_capacity_j_per_k: float = 9_000.0
    heat_fraction: float = 0.055
    conduction_w_per_k: float = 55.0
    convection_base_w_per_k: float = 22.0
    convection_speed_w_per_k_per_ms: float = 1.6
    emissivity: float = 0.0
    carcass_temp_c: float = 85.0
    ambient_temp_c: float = 25.0
    track_temp_c: float = 35.0

    def __post_init__(self) -> None:
        if self.thermal_capacity_j_per_k <= 0:
            raise ValueError("thermal capacity must be positive")
        if self.kappa < 0:
            raise ValueError("wear coefficient cannot be negative")


@dataclass(frozen=True)
class GripParameters:
    """How wear and temperature turn into lost grip, and grip into lap time.

    Attributes:
        mu_nominal: Peak friction coefficient of a fresh tyre in its window.
        xi_wear: Linear grip loss per unit wear.
        xi_cliff: Quadratic grip loss once wear passes `wear_critical`. This is
            the cliff, and it is a separate term rather than a steeper line
            because exp17 measured 337 real stints where the loss accelerates
            rather than merely continuing.
        wear_critical: Where the quadratic term switches on.
        temp_optimal_c: Centre of the thermal working window.
        temp_width_c: Gaussian width of the window.
        sensitivity_s: Lap-time cost of losing all grip at each corner, seconds,
            in CORNERS order. Front entries are steering and understeer;
            rear entries are traction and oversteer.
    """

    mu_nominal: float = 1.55
    xi_wear: float = 0.30
    xi_cliff: float = 1.20
    wear_critical: float = 0.72
    temp_optimal_c: float = 105.0
    temp_width_c: float = 28.0
    sensitivity_s: tuple[float, float, float, float] = (2.6, 2.6, 2.2, 2.2)

    def __post_init__(self) -> None:
        if self.temp_width_c <= 0:
            raise ValueError("thermal window width must be positive")
        if len(self.sensitivity_s) != 4:
            raise ValueError("sensitivity_s needs one entry per corner")


@dataclass(frozen=True)
class SessionParameters:
    """Terms that belong to the session rather than to the tyre.

    Attributes:
        base_lap_time_s: Lap time of a fresh tyre, full tank, green track.
        fuel_effect_s_per_kg: Seconds per kilogram. **Pinned, not fitted.** This
            is the same 0.030 the rest of the repository uses, and the whole
            identifiability argument in exp18 rests on it being a prior rather
            than a free parameter.
        burn_kg_per_lap: Fuel burned per lap.
        track_amplitude_s: Total seconds the track gives back as it rubbers in.
        track_rate: Saturation rate of the rubbering-in curve.
        traffic_s: Seconds lost per unit of traffic index.
    """

    base_lap_time_s: float = 90.0
    fuel_effect_s_per_kg: float = 0.030
    burn_kg_per_lap: float = 2.7
    track_amplitude_s: float = 0.9
    track_rate: float = 0.055
    traffic_s: float = 0.40


@dataclass(frozen=True)
class ProcessNoise:
    """Diffusion of the SDE, as standard deviations per second.

    Kept separate from the physics because these are nuisance parameters fitted
    by maximum likelihood, whereas the physics constants are either fitted under
    a prior or fixed outright. Confusing the two is how a physical model quietly
    becomes a curve fit with extra steps.

    Attributes:
        wear_sd: Micro-roughness variation in the wear rate.
        temp_sd: Unmodelled thermal disturbance, K per sqrt(s).
        fuel_sd: Fuel mass is nearly deterministic; this is small on purpose.
        track_sd: Track state wander.
    """

    wear_sd: float = 2.0e-4
    temp_sd: float = 0.35
    fuel_sd: float = 1.0e-3
    track_sd: float = 1.5e-3

    def as_diagonal(self) -> np.ndarray:
        """Diffusion variances on the state diagonal, per second."""
        q = np.zeros(N_STATE, dtype=float)
        q[W] = self.wear_sd ** 2
        q[T] = self.temp_sd ** 2
        q[I_FUEL] = self.fuel_sd ** 2
        q[I_TRACK] = self.track_sd ** 2
        return q


@dataclass(frozen=True)
class FourCornerParameters:
    """Everything the forward model needs, in one object."""

    thermo: ThermoMechanicalParameters = field(default_factory=ThermoMechanicalParameters)
    grip: GripParameters = field(default_factory=GripParameters)
    session: SessionParameters = field(default_factory=SessionParameters)
    noise: ProcessNoise = field(default_factory=ProcessNoise)

    def with_thermo(self, **kw) -> FourCornerParameters:
        return replace(self, thermo=replace(self.thermo, **kw))

    def with_grip(self, **kw) -> FourCornerParameters:
        return replace(self, grip=replace(self.grip, **kw))

    def with_session(self, **kw) -> FourCornerParameters:
        return replace(self, session=replace(self.session, **kw))

    def with_noise(self, **kw) -> FourCornerParameters:
        return replace(self, noise=replace(self.noise, **kw))


def initial_state(
    *,
    tyre_age_laps: float = 0.0,
    fuel_kg: float = 100.0,
    tread_temp_c: float = 90.0,
    wear_at_fit_start: float = 0.0,
) -> np.ndarray:
    """A starting state.

    `tyre_age_laps` is accepted but deliberately does **not** set wear. A used
    set fitted at lap 20 has unknown wear, and pretending otherwise would put
    the answer into the prior. Pass `wear_at_fit_start` explicitly when a scrubbed
    set is known; otherwise the filter is told it does not know, which is true.
    """
    x = np.zeros(N_STATE, dtype=float)
    x[W] = float(wear_at_fit_start)
    x[T] = float(tread_temp_c)
    x[I_FUEL] = float(fuel_kg)
    x[I_TRACK] = 0.0
    return x


def initial_covariance(
    *,
    wear_sd: float = 0.12,
    temp_sd: float = 12.0,
    fuel_sd: float = 4.0,
    track_sd: float = 0.15,
) -> np.ndarray:
    """Starting uncertainty.

    `wear_sd` is wide because a used set is genuinely unknown, and this is the
    one place where being honest about ignorance costs nothing: the filter will
    narrow it from data if the data can narrow it, and exp35 exists to find out
    whether it can.
    """
    p = np.zeros(N_STATE, dtype=float)
    p[W] = wear_sd ** 2
    p[T] = temp_sd ** 2
    p[I_FUEL] = fuel_sd ** 2
    p[I_TRACK] = track_sd ** 2
    return np.diag(p)


def clamp_state(x: np.ndarray) -> np.ndarray:
    """Hold the state inside physical bounds.

    A Kalman update is a linear correction to a nonlinear system, so it can and
    does push wear below zero on a lap where the car was simply quick. Left
    alone that produces negative grip loss, then a negative lap time, then a
    likelihood of minus infinity, and the optimiser walks away from a region it
    should have explored. Clamping is the honest fix and it is applied after
    every update, not only when something looks wrong.
    """
    out = np.array(x, dtype=float, copy=True)
    out[W] = np.clip(out[W], WEAR_MIN, WEAR_MAX)
    out[T] = np.clip(out[T], TEMP_MIN_C, TEMP_MAX_C)
    out[I_FUEL] = max(out[I_FUEL], 0.0)
    out[I_TRACK] = np.clip(out[I_TRACK], 0.0, 1.0)
    return out


def describe(x: np.ndarray) -> dict[str, float]:
    """The state as named quantities, for logging and for tests to assert on."""
    x = np.asarray(x, dtype=float)
    out: dict[str, float] = {}
    for name, i in CORNER_INDEX.items():
        out[f"wear_{name}"] = float(x[W][i])
        out[f"temp_{name}"] = float(x[T][i])
    out["fuel_kg"] = float(x[I_FUEL])
    out["track_progress"] = float(x[I_TRACK])
    out["wear_mean"] = float(np.mean(x[W]))
    # Two aggregates the pre-registration predicts will be identifiable where the
    # individual corners are not, so they are computed here rather than in the
    # experiment that scores them.
    out["wear_left_minus_right"] = float(x[W][0] + x[W][2] - x[W][1] - x[W][3]) / 2.0
    out["wear_front_minus_rear"] = float(x[W][0] + x[W][1] - x[W][2] - x[W][3]) / 2.0
    return out
