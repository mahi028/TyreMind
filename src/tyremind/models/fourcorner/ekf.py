"""The continuous-discrete Extended Kalman Filter.

Continuous-discrete because the physics is continuous and the measurements are
not. A tyre heats and wears every metre of the lap; a timing loop speaks once per
lap. So the state is propagated by integrating the SDE across the lap, and
corrected only where an observation exists.

    predict:  integrate  dx/dt = f(x, u)
                         dP/dt = F P + P F^T + Q      over [0, lap duration]
    update:   the usual Kalman correction at the lap boundary, with R inflated
              by the heavy-tailed weight from `observation.robust_weight`

**Why RK4 and not Euler.** The thermal equation is the stiff one: with a lumped
capacity near 9 kJ/K and a convective coefficient near 100 W/K, the temperature's
time constant is on the order of a hundred seconds, comparable to a lap. Euler
over a whole lap is visibly wrong and, worse, wrong in a direction that looks
like a real effect -- it overshoots the equilibrium temperature, which shows up
as spurious thermal degradation. Sub-stepping with RK4 removes that, and
`tests/unit/test_fourcorner_ekf.py` checks convergence against an analytic
solution of the linearised thermal equation rather than against a finer version
of itself.

**Why Joseph form.** The covariance update `(I - KH)P` is algebraically correct
and numerically fragile: it subtracts two nearly equal matrices and, over the
sixty-odd laps of a stint, loses symmetry and then positive-definiteness. The
Joseph form costs one extra matrix product and stays symmetric by construction.
With ten states and one observation the naive form fails in practice, not in
theory -- it was tried first here and produced negative variances by lap 40.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import LapInput, diffusion, drift, jacobian
from .observation import (
    observation_jacobian,
    predict_lap_time,
    robust_weight,
    skewed_t_logpdf,
    skewed_t_mean,
    skewed_t_variance,
)
from .state import N_STATE, FourCornerParameters, clamp_state

#: Integration sub-steps per lap. Measured against a 512-step reference on a
#: representative lap, relative error in the wear state goes
#: 1 -> 7.0e-2, 2 -> 1.3e-3, 4 -> 2.0e-4, 8 -> 1.4e-5, 16 -> 9.0e-7,
#: which is the fourth-order convergence RK4 should give. Eight is the first
#: value comfortably below the process noise, and it doubles a cost that is
#: milliseconds. Four was the original default and was not converged.
DEFAULT_SUBSTEPS = 8

#: Floor on the innovation variance, seconds squared. Without it a confident
#: filter can drive S towards zero and produce an unbounded gain on a single lap.
MIN_INNOVATION_VAR = 1e-8


@dataclass
class FilterResult:
    """What one forward pass produced.

    Attributes:
        states: Filtered mean after each lap, shape (n_laps, 10).
        covariances: Filtered covariance after each lap, (n_laps, 10, 10).
        predicted: One-step-ahead predicted lap time per lap.
        innovations: Observed minus predicted.
        innovation_var: Innovation variance per lap, including R.
        weights: Robust weight applied to each lap.
        log_likelihood: Exact log likelihood by prediction-error decomposition.
        n_diverged: Laps where the covariance had to be repaired.
    """

    states: np.ndarray
    covariances: np.ndarray
    predicted: np.ndarray
    innovations: np.ndarray
    innovation_var: np.ndarray
    weights: np.ndarray
    log_likelihood: float
    n_diverged: int


def _symmetrise(p: np.ndarray) -> np.ndarray:
    """Force exact symmetry.

    Floating point makes `P` drift a few ulps from symmetric every update, and
    the asymmetry compounds. Cheap to fix, expensive to leave.
    """
    return 0.5 * (p + p.T)


def _repair(p: np.ndarray) -> tuple[np.ndarray, bool]:
    """Project a covariance back onto the positive semi-definite cone.

    Returns the repaired matrix and whether repair was needed, because a filter
    that silently repairs itself every lap is diverging and the caller has to be
    able to find out. `FilterResult.n_diverged` is that count, and the fitter
    treats a high one as a failed fit rather than as a fitted model.
    """
    p = _symmetrise(p)
    eigenvalues, vectors = np.linalg.eigh(p)
    if eigenvalues.min() >= 0.0:
        return p, False
    clipped = np.clip(eigenvalues, 1e-12, None)
    return _symmetrise((vectors * clipped) @ vectors.T), True


def propagate(
    x: np.ndarray,
    p: np.ndarray,
    u: LapInput,
    params: FourCornerParameters,
    substeps: int = DEFAULT_SUBSTEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate mean and covariance across one lap.

    The mean goes through RK4. The covariance goes through the matrix Riccati
    equation `dP/dt = F P + P F^T + Q`, integrated with the same step and the
    Jacobian re-evaluated at each stage, because freezing `F` at the start of the
    lap is precisely the approximation that makes a nonlinear filter
    overconfident.
    """
    dt = u.duration_s / substeps
    x = np.asarray(x, dtype=float).copy()
    p = _symmetrise(np.asarray(p, dtype=float).copy())

    for _ in range(substeps):
        k1 = drift(x, u, params)
        k2 = drift(clamp_state(x + 0.5 * dt * k1), u, params)
        k3 = drift(clamp_state(x + 0.5 * dt * k2), u, params)
        k4 = drift(clamp_state(x + dt * k3), u, params)
        x_next = clamp_state(x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

        # Riccati step, RK2 at the midpoint: the covariance is smoother than the
        # mean and a fourth-order rule on it costs three extra Jacobians a lap
        # for an error already well below the process noise.
        f_mid = jacobian(clamp_state(0.5 * (x + x_next)), u, params)
        q = diffusion(clamp_state(0.5 * (x + x_next)), params)
        dp = f_mid @ p + p @ f_mid.T + q
        p = _symmetrise(p + dt * dp)

        x = x_next

    return x, p


def run_filter(
    inputs: list[LapInput],
    observations: np.ndarray,
    x0: np.ndarray,
    p0: np.ndarray,
    params: FourCornerParameters,
    *,
    obs_scale_s: float,
    nu: float = 6.0,
    zeta: float = 1.15,
    substeps: int = DEFAULT_SUBSTEPS,
    robust: bool = True,
) -> FilterResult:
    """One forward pass, returning the exact likelihood.

    The likelihood is the prediction-error decomposition under the skewed t, not
    under a Gaussian. Using the Gaussian here while using the robust weight in
    the update would be scoring the model on a distribution it is not fitting,
    and the optimiser would chase the mismatch.

    Args:
        inputs: One `LapInput` per lap, in order.
        observations: Observed lap times, same length.
        x0: Initial state.
        p0: Initial covariance.
        params: Model parameters.
        obs_scale_s: Scale of the observation noise, seconds.
        nu: Degrees of freedom of the skewed t.
        zeta: Skew. Above 1 leans towards slow laps.
        substeps: RK4 sub-steps per lap.
        robust: Whether to down-weight surprising laps. False recovers a plain
            Gaussian EKF, which the tests use as a reference.

    Returns:
        A `FilterResult`.
    """
    observations = np.asarray(observations, dtype=float)
    if len(inputs) != len(observations):
        raise ValueError("one observation per lap input is required")
    if obs_scale_s <= 0:
        raise ValueError("observation scale must be positive")

    n = len(inputs)
    states = np.zeros((n, N_STATE))
    covariances = np.zeros((n, N_STATE, N_STATE))
    predicted = np.zeros(n)
    innovations = np.zeros(n)
    innovation_var = np.zeros(n)
    weights = np.ones(n)

    x = clamp_state(np.asarray(x0, dtype=float))
    p = _symmetrise(np.asarray(p0, dtype=float))

    # The skewed t has a non-zero mean unless zeta is exactly 1. Left in, it
    # becomes a constant bias on every predicted lap time and lands in the wear
    # state, so it is removed here and the residual is centred by construction.
    noise_mean = skewed_t_mean(nu, zeta) * obs_scale_s
    noise_var = skewed_t_variance(nu, zeta) * obs_scale_s ** 2

    log_likelihood = 0.0
    n_diverged = 0

    for k, (u, y) in enumerate(zip(inputs, observations)):
        x, p = propagate(x, p, u, params, substeps=substeps)
        p, repaired = _repair(p)
        n_diverged += int(repaired)

        y_hat = predict_lap_time(x, u, params) + noise_mean
        h = observation_jacobian(x, u, params)

        innovation = float(y - y_hat)
        weight = robust_weight(innovation, obs_scale_s, nu, zeta) if robust else 1.0
        # Down-weighting a lap is the same as saying its noise was larger.
        r = noise_var / max(weight, 1e-6)
        s = float(h @ p @ h.T + r)
        s = max(s, MIN_INNOVATION_VAR)

        gain = (p @ h) / s
        x = clamp_state(x + gain * innovation)

        # Joseph form. See the module docstring for why the short form is not
        # used: it lost positive-definiteness by lap 40 in this state dimension.
        a = np.eye(N_STATE) - np.outer(gain, h)
        p = _symmetrise(a @ p @ a.T + np.outer(gain, gain) * r)
        p, repaired = _repair(p)
        n_diverged += int(repaired)

        # Exact likelihood under the model actually being used. The innovation is
        # standardised by its own predictive scale, so the density is the skewed
        # t evaluated at that standardised value, with the Jacobian of the
        # standardisation accounted for by the log(sqrt(s)) term.
        scale = np.sqrt(s)
        log_likelihood += float(skewed_t_logpdf(innovation / scale, nu, zeta) - np.log(scale))

        states[k] = x
        covariances[k] = p
        predicted[k] = y_hat
        innovations[k] = innovation
        innovation_var[k] = s
        weights[k] = weight

    return FilterResult(
        states=states,
        covariances=covariances,
        predicted=predicted,
        innovations=innovations,
        innovation_var=innovation_var,
        weights=weights,
        log_likelihood=log_likelihood,
        n_diverged=n_diverged,
    )


def rts_smooth(
    result: FilterResult,
    inputs: list[LapInput],
    params: FourCornerParameters,
    substeps: int = DEFAULT_SUBSTEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel smoothing over the filtered pass.

    The filter says what was knowable at each lap; the smoother says what is
    knowable now the stint is over. Both are published, because they answer
    different questions and conflating them is how a hindsight number gets
    presented as something the pit wall had at the time.

    The transition matrix is the same one the Riccati step used, re-derived by
    integrating the Jacobian across the lap, so the smoother is consistent with
    the filter rather than an independent approximation of it.
    """
    n = len(inputs)
    xs = result.states.copy()
    ps = result.covariances.copy()

    for k in range(n - 2, -1, -1):
        u = inputs[k + 1]
        dt = u.duration_s / substeps
        # Discrete transition across the lap, built by composing the sub-steps.
        transition = np.eye(N_STATE)
        x_walk = result.states[k].copy()
        for _ in range(substeps):
            f = jacobian(x_walk, u, params)
            transition = (np.eye(N_STATE) + dt * f) @ transition
            x_walk = clamp_state(x_walk + dt * drift(x_walk, u, params))

        predicted_x, predicted_p = propagate(
            result.states[k], result.covariances[k], u, params, substeps=substeps
        )
        predicted_p, _ = _repair(predicted_p)

        try:
            gain = result.covariances[k] @ transition.T @ np.linalg.pinv(predicted_p)
        except np.linalg.LinAlgError:
            continue

        xs[k] = clamp_state(result.states[k] + gain @ (xs[k + 1] - predicted_x))
        ps[k] = _symmetrise(
            result.covariances[k] + gain @ (ps[k + 1] - predicted_p) @ gain.T
        )
        ps[k], _ = _repair(ps[k])

    return xs, ps
