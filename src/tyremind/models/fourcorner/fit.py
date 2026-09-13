"""Fitting the four-corner model, and reporting a rate the ladder can score.

Two jobs.

**Fit.** Maximum likelihood over a deliberately small parameter set, by L-BFGS-B,
exactly as `ssm/tyre_ssm.py` does for the two-state model. The restraint is the
point: the forward model has more than twenty constants, and fitting all of them
against one observation per lap would be curve-fitting with a physics-shaped
excuse. Three are free here. Everything else is either pinned by physics, set by
the calibration recorded in `state.py`, or profiled out analytically.

**Report.** The ladder scores a degradation rate in seconds per lap, so the
ten-dimensional state has to be reduced to that one number. It is the lap-on-lap
increase in the grip penalty -- what the tyre is costing now, per extra lap of
age -- which is the same quantity the two-state model's `rate` holds, computed a
different way. Without this the model could not be compared to anything.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from .dynamics import LapInput, grip_coefficient
from .ekf import DEFAULT_SUBSTEPS, FilterResult, run_filter, rts_smooth
from .observation import grip_penalty_seconds
from .state import (
    T,
    W,
    FourCornerParameters,
    initial_covariance,
    initial_state,
)

#: Free parameters, in optimiser order. Three, against the two-state model's six.
#:
#: - `log_kappa`   wear coefficient. The one parameter that genuinely sets how
#:                 fast the tyre goes away.
#: - `log_obs_sd`  observation noise scale.
#: - `base_offset` a constant on the lap time. Not profiled out analytically
#:                 because the skewed-t likelihood is not quadratic in it, so the
#:                 closed form that works for a Gaussian does not apply here.
PARAMETER_NAMES = ("log_kappa", "log_obs_sd", "base_offset")

#: Bounds. Wide enough not to bind at a sensible optimum, tight enough that a
#: diverged evaluation cannot wander somewhere the integrator falls over.
BOUNDS = (
    (np.log(1e-6), np.log(5e-2)),   # kappa
    (np.log(0.03), np.log(3.0)),    # observation sd, seconds
    (-8.0, 8.0),                    # base offset, seconds
)

#: Returned when an evaluation fails. Large and finite, because an inf makes
#: L-BFGS-B's line search give up rather than step away.
PENALTY = 1e9


@dataclass
class FourCornerFit:
    """A fitted stint.

    Attributes:
        params: The fitted parameters.
        result: The filtered pass at the optimum.
        smoothed_states: RTS-smoothed states.
        rate_per_lap: Degradation rate per lap, seconds, one entry per lap.
        rate_sd_per_lap: Its standard deviation, propagated from the covariance.
        converged: Whether the optimiser reported success.
        n_iterations: Optimiser iterations.
        fit_seconds: Wall clock, which the pre-registration gates on.
        log_likelihood: At the optimum.
        n_diverged: Covariance repairs during the final pass.
    """

    params: FourCornerParameters
    result: FilterResult
    smoothed_states: np.ndarray
    rate_per_lap: np.ndarray
    rate_sd_per_lap: np.ndarray
    converged: bool
    n_iterations: int
    fit_seconds: float
    log_likelihood: float
    n_diverged: int

    @property
    def mean_rate(self) -> float:
        """One number for the ladder: the stint-average degradation rate.

        exp20 found that "the degradation rate" is three different quantities on
        a nonlinear generator and that a different model wins each. This is the
        stint average, which is the target exp20 pre-registered as primary.
        """
        finite = self.rate_per_lap[np.isfinite(self.rate_per_lap)]
        return float(finite.mean()) if len(finite) else float("nan")

    @property
    def mean_rate_sd(self) -> float:
        finite = self.rate_sd_per_lap[np.isfinite(self.rate_sd_per_lap)]
        return float(np.sqrt(np.mean(finite ** 2))) if len(finite) else float("nan")


def _penalty_seconds(states: np.ndarray, params: FourCornerParameters) -> np.ndarray:
    """Grip penalty implied by each state in a trajectory."""
    out = np.empty(len(states))
    for k, x in enumerate(states):
        mu = grip_coefficient(x[W], x[T], params)
        out[k] = grip_penalty_seconds(mu, params)
    return out


def degradation_rate(
    states: np.ndarray,
    covariances: np.ndarray,
    params: FourCornerParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Seconds per lap the tyre is currently costing, and how sure we are.

    The rate is the lap-on-lap difference of the grip penalty. The first lap has
    no predecessor, so it takes the second lap's value rather than a zero -- a
    zero there would drag the stint mean down by a lap's worth every time and is
    the kind of off-by-one that survives review because it looks conservative.

    Uncertainty is propagated linearly: `sd = |d(penalty)/dw| * sd(w)`, summed
    over corners in quadrature. That ignores the wear-temperature covariance, so
    it is an under-estimate, and it is labelled as one here rather than being
    quietly published as if it were exact.
    """
    penalty = _penalty_seconds(states, params)
    rate = np.empty_like(penalty)
    rate[1:] = np.diff(penalty)
    rate[0] = rate[1] if len(rate) > 1 else 0.0

    g = params.grip
    sensitivity = np.asarray(g.sensitivity_s, dtype=float)
    sd = np.empty_like(penalty)
    for k, x in enumerate(states):
        w = x[W]
        over = np.maximum(0.0, w - g.wear_critical)
        mechanical = np.maximum(1.0 - g.xi_wear * w - g.xi_cliff * over ** 2, 0.0)
        d_mech = np.where(mechanical > 0.0, -g.xi_wear - 2.0 * g.xi_cliff * over, 0.0)
        thermal = np.exp(-((x[T] - g.temp_optimal_c) ** 2) / (2.0 * g.temp_width_c ** 2))
        d_penalty_dw = -sensitivity * d_mech * thermal
        wear_var = np.clip(np.diag(covariances[k])[W], 0.0, None)
        sd[k] = float(np.sqrt(np.sum((d_penalty_dw ** 2) * wear_var)))
    return rate, sd


def _apply(theta: np.ndarray, base: FourCornerParameters) -> tuple[FourCornerParameters, float, float]:
    """Unpack the optimiser vector into parameters, noise scale and offset."""
    log_kappa, log_obs_sd, offset = theta
    params = base.with_thermo(kappa=float(np.exp(log_kappa)))
    params = params.with_session(base_lap_time_s=base.session.base_lap_time_s + float(offset))
    return params, float(np.exp(log_obs_sd)), float(offset)


def fit_stint(
    inputs: list[LapInput],
    observations: np.ndarray,
    *,
    base: FourCornerParameters | None = None,
    nu: float = 6.0,
    zeta: float = 1.15,
    substeps: int = DEFAULT_SUBSTEPS,
    max_iterations: int = 60,
) -> FourCornerFit:
    """Fit one stint by maximum likelihood.

    Raises:
        ValueError: On an empty or mismatched stint. Raising rather than
            returning NaN is the fairness rule the whole ladder is held to --
            see `tests/unit/test_model_fairness.py` and the note in
            `inputs.build_inputs`.
    """
    observations = np.asarray(observations, dtype=float)
    if not inputs or len(inputs) != len(observations):
        raise ValueError("fit_stint needs one observation per lap and at least one lap")
    if not np.isfinite(observations).all():
        raise ValueError("observations must be finite")

    base = base or FourCornerParameters()
    # Start the baseline at the stint's own median so the offset has little work
    # to do and the optimiser does not spend its budget on a constant.
    base = base.with_session(base_lap_time_s=float(np.median(observations)))

    x0 = initial_state(tread_temp_c=base.grip.temp_optimal_c)
    p0 = initial_covariance()

    def negative_log_likelihood(theta: np.ndarray) -> float:
        try:
            params, obs_sd, _ = _apply(theta, base)
            r = run_filter(inputs, observations, x0, p0, params,
                           obs_scale_s=obs_sd, nu=nu, zeta=zeta, substeps=substeps)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return PENALTY
        if not np.isfinite(r.log_likelihood):
            return PENALTY
        # A pass that needed repairing on most laps has diverged; scoring it as
        # a good fit would let the optimiser chase numerical damage.
        if r.n_diverged > len(inputs):
            return PENALTY
        return -r.log_likelihood

    start = np.array([np.log(base.thermo.kappa), np.log(0.25), 0.0])
    began = time.perf_counter()
    opt = optimize.minimize(
        negative_log_likelihood,
        start,
        method="L-BFGS-B",
        bounds=BOUNDS,
        options={"maxiter": max_iterations},
    )
    elapsed = time.perf_counter() - began

    params, obs_sd, _ = _apply(opt.x, base)
    result = run_filter(inputs, observations, x0, p0, params,
                        obs_scale_s=obs_sd, nu=nu, zeta=zeta, substeps=substeps)
    smoothed, smoothed_cov = rts_smooth(result, inputs, params, substeps=substeps)
    rate, rate_sd = degradation_rate(smoothed, smoothed_cov, params)

    return FourCornerFit(
        params=params,
        result=result,
        smoothed_states=smoothed,
        rate_per_lap=rate,
        rate_sd_per_lap=rate_sd,
        converged=bool(opt.success),
        n_iterations=int(opt.nit),
        fit_seconds=float(elapsed),
        log_likelihood=float(result.log_likelihood),
        n_diverged=int(result.n_diverged),
    )
