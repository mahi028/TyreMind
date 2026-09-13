"""The continuous-time drift and diffusion of the four-corner model.

This is the `f(x, u; theta)` and `G(x)` of

    dx(t) = f(x(t), u(t); theta) dt + G(x(t)) dW(t)

and the grip coupling that ties the two physical blocks together. Everything
here is a pure function of state, input and parameters, with no filtering and no
fitting, so each equation can be tested against the behaviour physics requires
rather than against whatever the estimator happens to produce.

The four equations, in the order they depend on each other:

1. **Grip** `mu_i = mu_0 (1 - xi_w w_i - xi_cliff max(0, w_i - w_crit)^2)
   exp(-(T_i - T_opt)^2 / 2 sigma_T^2)`
2. **Wear** `dw_i/dt = kappa (P_i / P_ref)^alpha`, with
   `P_i = (mu_i / mu_0) * proxy_i` the measured per-corner power proxy
3. **Thermal** `C dT_i/dt = beta P_i + k(T_carcass - T_i)
   - h(v)(T_i - T_amb) - sigma eps (T_i^4 - T_track^4)`
4. **Fuel and track**, both nearly deterministic, carried as states so their
   uncertainty propagates into the tyre posterior instead of being assumed away.

**Radiation is off by default** (`emissivity = 0`). The Stefan-Boltzmann term is
implemented and tested because the specification calls for it, but at tread
temperatures it contributes about two orders of magnitude less than convection
while making the Jacobian stiff, and a term that changes nothing except the
integrator's step size is a liability. It is a parameter rather than a deletion
so the claim can be checked.
"""

from __future__ import annotations

import numpy as np

from .state import (
    I_FUEL,
    I_TRACK,
    N_STATE,
    REFERENCE_CORNER_POWER_W,
    T,
    W,
    FourCornerParameters,
)

#: Stefan-Boltzmann constant, W/m^2/K^4, folded with an effective area into
#: `emissivity` so the parameter is a single lumped coefficient.
STEFAN_BOLTZMANN = 5.670374419e-8

#: Absolute zero offset, for the radiation term which needs kelvin.
KELVIN = 273.15


def grip_coefficient(
    wear: np.ndarray,
    temp_c: np.ndarray,
    params: FourCornerParameters,
) -> np.ndarray:
    """Available friction at each corner.

    Two multiplicative factors, because they fail independently: a cold tyre and
    a worn tyre are different problems and a driver reports them differently. The
    mechanical factor is floored at zero -- a tyre can stop working, but negative
    friction would mean it pushes the car forwards.

    Args:
        wear: Normalised wear per corner.
        temp_c: Tread temperature per corner, degrees C.
        params: Model parameters.

    Returns:
        Friction coefficient per corner.
    """
    g = params.grip
    w = np.asarray(wear, dtype=float)
    t = np.asarray(temp_c, dtype=float)

    over_cliff = np.maximum(0.0, w - g.wear_critical)
    mechanical = 1.0 - g.xi_wear * w - g.xi_cliff * over_cliff ** 2
    mechanical = np.maximum(mechanical, 0.0)

    thermal = np.exp(-((t - g.temp_optimal_c) ** 2) / (2.0 * g.temp_width_c ** 2))
    return g.mu_nominal * mechanical * thermal


def frictional_power(
    mu: np.ndarray,
    power_proxy_w: np.ndarray,
    params: FourCornerParameters,
) -> np.ndarray:
    """Power dissipated in each contact patch, in the proxy's own units.

    **This function used to invent a slip velocity, and that was wrong.** The
    telemetry column `energy_mj_*` is not physical joules. `physics.dynamics`
    computes it as `F_z * (a_demand / g) * v` and says so in its own docstring:
    *"Absolute values in watts are not meaningful and are never reported as
    such."* The slip factor is a demand proxy in g, not a velocity in m/s.

    So the honest thing is to consume the proxy directly rather than to invert it
    into a fabricated slip speed and multiply it back. What the proxy is missing
    is the `mu` dependence -- it carries `F_z` and a slip surrogate but not the
    friction coefficient -- so grip enters as a ratio against nominal:

        power_i = (mu_i / mu_nominal) * proxy_i

    A fresh tyre in its window therefore dissipates the proxy as measured, and a
    tyre that has lost grip dissipates proportionally less, which is the feedback
    that makes the wear equation self-limiting.

    Everything downstream is relative: the fitted wear coefficient absorbs the
    proxy's unknown scale, and what survives is the corner-to-corner ratio, which
    is the only thing four separate wear states can be identified from.
    """
    ratio = np.asarray(mu, dtype=float) / params.grip.mu_nominal
    return np.maximum(ratio, 0.0) * np.maximum(np.asarray(power_proxy_w, dtype=float), 0.0)


def wear_rate(power_w: np.ndarray, params: FourCornerParameters) -> np.ndarray:
    """Wear accumulated per second at each corner.

    Generalised Archard: wear goes as a power of the frictional power, normalised
    by a reference so `kappa` is a number near 0.004 rather than near 1e-9. The
    exponent is fitted because the friction literature does not agree on it and
    fixing it at 1.0 would be asserting the answer to a question the data can be
    asked.

    Power is floored at zero before exponentiation. A negative power is not
    physical, and a fractional exponent of a negative number is a NaN that
    propagates silently through the whole filter.
    """
    p = np.maximum(np.asarray(power_w, dtype=float), 0.0) / REFERENCE_CORNER_POWER_W
    return params.thermo.kappa * np.power(p, params.thermo.alpha)


def thermal_rate(
    temp_c: np.ndarray,
    power_w: np.ndarray,
    airspeed_ms: float,
    params: FourCornerParameters,
) -> np.ndarray:
    """Temperature change per second at each corner.

    Four terms, and their relative sizes are the interesting part: at racing
    speeds convection dominates cooling, conduction to the carcass sets the floor
    a tyre returns to between corners, and radiation is negligible unless the
    tread is genuinely overheating.
    """
    th = params.thermo
    t = np.asarray(temp_c, dtype=float)

    generation = th.heat_fraction * np.asarray(power_w, dtype=float)
    conduction = th.conduction_w_per_k * (th.carcass_temp_c - t)
    convection_coefficient = (
        th.convection_base_w_per_k + th.convection_speed_w_per_k_per_ms * max(airspeed_ms, 0.0)
    )
    convection = convection_coefficient * (t - th.ambient_temp_c)

    if th.emissivity > 0.0:
        radiation = (
            STEFAN_BOLTZMANN
            * th.emissivity
            * ((t + KELVIN) ** 4 - (th.track_temp_c + KELVIN) ** 4)
        )
    else:
        radiation = 0.0

    return (generation + conduction - convection - radiation) / th.thermal_capacity_j_per_k


class LapInput:
    """Measured kinematics for one lap, held in the form the SDE consumes.

    Built by `inputs.py` from the telemetry corpus. The important property is
    that **nothing in here is estimated**: corner loads come from a rigid-body
    calculation on measured speed and acceleration, and slip speed from a
    measured proxy. The filter is told what the car did and asked only what the
    tyres did about it.

    Attributes:
        duration_s: Lap time actually taken, seconds.
        load_n: Vertical load per corner, newtons, in CORNERS order. Genuinely
            newtons -- this one is a rigid-body calculation, not a proxy.
        power_proxy_w: Per-corner frictional power proxy, in the units
            `physics.dynamics.frictional_power_proxy` produces. Relative, not
            absolute; see `frictional_power` for why that is fine and what it
            costs.
        airspeed_ms: Mean airspeed over the lap, for convective cooling.
        laps_completed: Laps of this run completed before this one, the fuel
            clock.
        traffic_index: 0 to 1.
        session_lap: Lap number in the session, the track-evolution clock.
    """

    __slots__ = (
        "duration_s",
        "load_n",
        "power_proxy_w",
        "airspeed_ms",
        "laps_completed",
        "traffic_index",
        "session_lap",
    )

    def __init__(
        self,
        *,
        duration_s: float,
        load_n: np.ndarray,
        power_proxy_w: np.ndarray,
        airspeed_ms: float,
        laps_completed: float,
        traffic_index: float,
        session_lap: int,
    ) -> None:
        load = np.asarray(load_n, dtype=float)
        power = np.asarray(power_proxy_w, dtype=float)
        if load.shape != (4,) or power.shape != (4,):
            raise ValueError("load and power must have one entry per corner")
        if not np.isfinite(load).all() or not np.isfinite(power).all():
            raise ValueError("load and power must be finite; a NaN here silently kills the filter")
        if (power < 0).any():
            raise ValueError("frictional power proxy cannot be negative")
        if duration_s <= 0:
            raise ValueError("lap duration must be positive")
        self.duration_s = float(duration_s)
        self.load_n = load
        self.power_proxy_w = power
        self.airspeed_ms = float(airspeed_ms)
        self.laps_completed = float(laps_completed)
        self.traffic_index = float(traffic_index)
        self.session_lap = int(session_lap)


def drift(x: np.ndarray, u: LapInput, params: FourCornerParameters) -> np.ndarray:
    """`f(x, u; theta)`: the deterministic part of the state's motion.

    Fuel drains at a constant rate per second over the lap, and track progress
    relaxes towards one at the saturating rate, which is the differential form of
    the `A(1 - e^{-k L})` curve the scalar model uses. Writing it as a relaxation
    rather than as a closed form is what lets the filter carry uncertainty on it.
    """
    x = np.asarray(x, dtype=float)
    dx = np.zeros(N_STATE, dtype=float)

    mu = grip_coefficient(x[W], x[T], params)
    power = frictional_power(mu, u.power_proxy_w, params)

    dx[W] = wear_rate(power, params)
    dx[T] = thermal_rate(x[T], power, u.airspeed_ms, params)
    dx[I_FUEL] = -params.session.burn_kg_per_lap / u.duration_s
    dx[I_TRACK] = params.session.track_rate * (1.0 - x[I_TRACK]) / u.duration_s
    return dx


def diffusion(x: np.ndarray, params: FourCornerParameters) -> np.ndarray:
    """`G(x) G(x)^T`: process noise covariance per second.

    State-dependent in one respect that matters. Wear noise scales with how much
    wear is already present, because a fresh tyre's rate is far more predictable
    than a worn one's -- which is the same asymmetry exp17 measured as the cliff,
    expressed as uncertainty instead of as a mean.
    """
    x = np.asarray(x, dtype=float)
    q = params.noise.as_diagonal()
    scale = np.ones(N_STATE, dtype=float)
    scale[W] = 1.0 + 2.0 * np.clip(x[W], 0.0, None)
    return np.diag(q * scale)


def jacobian(x: np.ndarray, u: LapInput, params: FourCornerParameters) -> np.ndarray:
    """`df/dx`, analytic.

    Hand-written rather than finite-differenced because the filter evaluates it
    once per integrator stage per lap, and because a closed form can be checked
    against a numerical one -- which `tests/unit/test_fourcorner_dynamics.py`
    does, on random states, to a tight tolerance. A Jacobian that is merely
    plausible produces a covariance that is merely plausible, and every interval
    this project publishes rests on it.
    """
    x = np.asarray(x, dtype=float)
    g, th = params.grip, params.thermo
    w, t = x[W], x[T]

    mu = grip_coefficient(w, t, params)
    power = frictional_power(mu, u.power_proxy_w, params)

    over = np.maximum(0.0, w - g.wear_critical)
    mechanical = np.maximum(1.0 - g.xi_wear * w - g.xi_cliff * over ** 2, 0.0)
    thermal_factor = np.exp(-((t - g.temp_optimal_c) ** 2) / (2.0 * g.temp_width_c ** 2))

    # d(mechanical)/dw, zero wherever the mechanical factor has bottomed out.
    d_mech_dw = np.where(mechanical > 0.0, -g.xi_wear - 2.0 * g.xi_cliff * over, 0.0)
    d_mu_dw = g.mu_nominal * d_mech_dw * thermal_factor
    d_mu_dt = mu * (-(t - g.temp_optimal_c) / g.temp_width_c ** 2)

    # Power is linear in mu with slope proxy / mu_nominal, so its derivatives
    # follow directly from the grip derivatives above.
    slope = np.maximum(u.power_proxy_w, 0.0) / g.mu_nominal
    d_power_dw = d_mu_dw * slope
    d_power_dt = d_mu_dt * slope

    # d(wear rate)/d(power), from the power law. Guarded at zero power where the
    # derivative of p^alpha is singular for alpha < 1.
    p_norm = np.maximum(power, 0.0) / REFERENCE_CORNER_POWER_W
    with np.errstate(divide="ignore", invalid="ignore"):
        d_rate_dp = np.where(
            p_norm > 1e-12,
            th.kappa * th.alpha * np.power(p_norm, th.alpha - 1.0) / REFERENCE_CORNER_POWER_W,
            0.0,
        )

    convection_coefficient = (
        th.convection_base_w_per_k + th.convection_speed_w_per_k_per_ms * max(u.airspeed_ms, 0.0)
    )
    d_rad_dt = (
        4.0 * STEFAN_BOLTZMANN * th.emissivity * (t + KELVIN) ** 3
        if th.emissivity > 0.0
        else np.zeros(4)
    )

    j = np.zeros((N_STATE, N_STATE), dtype=float)
    for i in range(4):
        # Wear block: each corner depends only on its own wear and temperature,
        # which is what makes the corner states separable in principle and
        # unidentifiable in practice when the loads are symmetric.
        j[i, i] = d_rate_dp[i] * d_power_dw[i]
        j[i, 4 + i] = d_rate_dp[i] * d_power_dt[i]
        # Thermal block.
        j[4 + i, i] = th.heat_fraction * d_power_dw[i] / th.thermal_capacity_j_per_k
        j[4 + i, 4 + i] = (
            th.heat_fraction * d_power_dt[i]
            - th.conduction_w_per_k
            - convection_coefficient
            - (d_rad_dt[i] if th.emissivity > 0.0 else 0.0)
        ) / th.thermal_capacity_j_per_k

    j[I_TRACK, I_TRACK] = -params.session.track_rate / u.duration_s
    # Fuel drains at a rate that does not depend on the state, so its row is zero.
    return j
