"""Mapping the ten-dimensional state onto the one number a timing loop gives us.

    y_k = t_base + dt_grip(mu_FL..mu_RR) - phi * m_burned - A * D_track
          + gamma * TI_k + eps_k

Everything the filter knows arrives through this scalar. Ten states, one
observation per lap: that ratio is the whole difficulty of the four-corner model
and `identifiability.py` exists to measure what survives it.

**The observation error is skewed, and that is a physical statement.** A driver
can lose two tenths to a lock-up, a missed apex or a lift for traffic. A driver
cannot gain two tenths by making a mistake. Modelling `eps` as Gaussian forces
the same width on both sides and pulls the fitted baseline towards the slow tail,
which biases every degradation estimate that sits on top of it. Fernandez and
Steel's skewed Student-t gives a heavy right tail and a light left one with two
parameters, and it subsumes the 3xMAD outlier filter the loader applies -- a
heavy tail *is* an outlier policy, expressed as a likelihood instead of as a
threshold, so the laps are down-weighted rather than deleted.
"""

from __future__ import annotations

import numpy as np
from scipy import special

from .dynamics import LapInput, grip_coefficient
from .state import I_FUEL, I_TRACK, N_STATE, T, W, FourCornerParameters


def grip_penalty_seconds(mu: np.ndarray, params: FourCornerParameters) -> float:
    """Seconds lost to the grip currently available at the four corners.

    A first-order sensitivity: each corner contributes its own coefficient times
    its fractional shortfall from nominal grip. The coefficients differ front to
    rear because losing the front is understeer and losing the rear is traction,
    and those cost different amounts of lap time on the same circuit.

    The shortfall is measured against `mu_nominal` rather than against the tyre's
    own best, so a tyre that never reaches its window is correctly reported as
    slow rather than as fine.
    """
    shortfall = 1.0 - np.asarray(mu, dtype=float) / params.grip.mu_nominal
    return float(np.dot(np.asarray(params.grip.sensitivity_s, dtype=float), shortfall))


def predict_lap_time(x: np.ndarray, u: LapInput, params: FourCornerParameters) -> float:
    """`h(x, u)`: the lap time this state implies."""
    s = params.session
    mu = grip_coefficient(np.asarray(x)[W], np.asarray(x)[T], params)
    burned = s.burn_kg_per_lap * u.laps_completed
    return (
        s.base_lap_time_s
        + grip_penalty_seconds(mu, params)
        - s.fuel_effect_s_per_kg * burned
        - s.track_amplitude_s * float(np.asarray(x)[I_TRACK])
        + s.traffic_s * u.traffic_index
    )


def observation_jacobian(
    x: np.ndarray, u: LapInput, params: FourCornerParameters
) -> np.ndarray:
    """`dh/dx`, a row vector of length ten.

    Only six entries are non-zero: four wear, four temperature -- through grip --
    and the track state. Fuel enters through `laps_completed`, which is an input
    rather than the fuel state, so the fuel row is zero.

    **That zero is worth pausing on.** Carrying fuel as a state while observing
    it through a counter means the filter never learns fuel mass from lap times.
    That is deliberate and it is the four-corner restatement of exp18's finding:
    the fuel coefficient is a prior, not a measurement, and a model that appeared
    to estimate it would be fooling itself.
    """
    x = np.asarray(x, dtype=float)
    g = params.grip
    w, t = x[W], x[T]

    over = np.maximum(0.0, w - g.wear_critical)
    mechanical = np.maximum(1.0 - g.xi_wear * w - g.xi_cliff * over ** 2, 0.0)
    thermal_factor = np.exp(-((t - g.temp_optimal_c) ** 2) / (2.0 * g.temp_width_c ** 2))
    mu = g.mu_nominal * mechanical * thermal_factor

    d_mech_dw = np.where(mechanical > 0.0, -g.xi_wear - 2.0 * g.xi_cliff * over, 0.0)
    d_mu_dw = g.mu_nominal * d_mech_dw * thermal_factor
    d_mu_dt = mu * (-(t - g.temp_optimal_c) / g.temp_width_c ** 2)

    sensitivity = np.asarray(g.sensitivity_s, dtype=float)
    h = np.zeros(N_STATE, dtype=float)
    # d(penalty)/d(mu) = -c_i / mu_nominal, then chain through mu.
    h[W] = -sensitivity * d_mu_dw / g.mu_nominal
    h[T] = -sensitivity * d_mu_dt / g.mu_nominal
    h[I_TRACK] = -params.session.track_amplitude_s
    return h


# ---------------------------------------------------------------------------
# Fernandez & Steel skewed Student-t
# ---------------------------------------------------------------------------


def skewed_t_logpdf(
    z: np.ndarray | float,
    nu: float,
    zeta: float,
) -> np.ndarray:
    """Log density of the standardised Fernandez-Steel skewed t.

    The construction: take a symmetric Student-t, compress one half by `zeta` and
    stretch the other by `1/zeta`, then renormalise. `zeta = 1` recovers the
    symmetric t exactly, which is the identity the tests check first.

        f(z) = 2 / (zeta + 1/zeta) * t_nu(z * zeta)      for z < 0
               2 / (zeta + 1/zeta) * t_nu(z / zeta)      for z >= 0

    With `zeta > 1` the right tail is the stretched one, so large positive
    residuals -- laps slower than predicted -- are cheap and large negative ones
    are expensive. That is the asymmetry a lock-up creates.

    Args:
        z: Standardised residual.
        nu: Degrees of freedom, > 2 so the variance exists.
        zeta: Skew. 1 is symmetric, > 1 leans right.

    Returns:
        Log density, elementwise.
    """
    if nu <= 2.0:
        raise ValueError("nu must exceed 2 for a finite variance")
    if zeta <= 0.0:
        raise ValueError("zeta must be positive")

    z = np.asarray(z, dtype=float)
    scaled = np.where(z < 0.0, z * zeta, z / zeta)

    log_norm = (
        special.gammaln((nu + 1.0) / 2.0)
        - special.gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
    )
    log_kernel = -((nu + 1.0) / 2.0) * np.log1p(scaled ** 2 / nu)
    log_skew = np.log(2.0) - np.log(zeta + 1.0 / zeta)
    return log_norm + log_kernel + log_skew


def skewed_t_mean(nu: float, zeta: float) -> float:
    """Mean of the standardised skewed t.

    Non-zero whenever `zeta != 1`, which matters: if the filter subtracts a
    predicted lap time from an observation whose noise has a non-zero mean, the
    bias lands in the tyre state. Subtracting this is what keeps the skew a
    statement about the tails rather than a silent offset on the answer.

    From Fernandez & Steel: `E[z] = M_1 (zeta - 1/zeta)` with
    `M_1 = 2 sqrt(nu) Gamma((nu+1)/2) / ((nu-1) sqrt(pi) Gamma(nu/2))`.
    """
    if nu <= 1.0:
        raise ValueError("mean does not exist for nu <= 1")
    m1 = (
        2.0
        * np.sqrt(nu)
        * np.exp(special.gammaln((nu + 1.0) / 2.0) - special.gammaln(nu / 2.0))
        / ((nu - 1.0) * np.sqrt(np.pi))
    )
    return float(m1 * (zeta - 1.0 / zeta))


def skewed_t_variance(nu: float, zeta: float) -> float:
    """Variance of the standardised skewed t.

    Needed to turn the fitted scale into a comparable standard deviation, and to
    give the Kalman update an `R` that means the same thing whatever the skew.
    """
    if nu <= 2.0:
        raise ValueError("variance does not exist for nu <= 2")
    m2 = nu / (nu - 2.0)
    mean = skewed_t_mean(nu, zeta)
    second = m2 * (zeta ** 3 + 1.0 / zeta ** 3) / (zeta + 1.0 / zeta)
    return float(second - mean ** 2)


def robust_weight(residual: float, scale: float, nu: float, zeta: float) -> float:
    """How much a lap should count, given how surprising it is.

    The Kalman update is derived under a Gaussian, so a heavy-tailed likelihood
    cannot be used directly without leaving the closed form. The standard
    resolution is an iteratively-reweighted update: run the Gaussian algebra but
    inflate `R` for laps the heavy tail says are unlikely, which is exactly the
    weight a Student-t's own score function assigns.

        weight = (nu + 1) / (nu + z^2)

    A lap at one sigma keeps almost all its weight; a lock-up three sigma slow
    keeps roughly a third of it. The skew enters through `z`, so a slow outlier is
    down-weighted less than an equally large fast one -- the fast one is the more
    surprising event and should move the state less.
    """
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    z = residual / scale
    z = z * zeta if z < 0.0 else z / zeta
    return float((nu + 1.0) / (nu + z * z))
