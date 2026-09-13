"""How much of the four-corner answer is data, and how much is prior.

exp18 asked this of the shipped two-state model and got an uncomfortable number:
within a run, tyre age and fuel are collinear at rho = 1.000, the information
matrix is singular in 11 of 11 sessions, and **6.0%** of the recovered rate is
data-driven. The rest is the fuel prior propagating through.

Adding eight latent states to the same one-observation-per-lap budget cannot
create information. `PREREGISTRATION_exp35.md` therefore predicts, before any of
this ran, that the four-corner share falls **below** 6.0% and that the individual
corner states are not separately identifiable at all. This module is the
measurement that scores those predictions.

Three quantities, each answering a different question:

- **`observability_gramian`** -- can the states be distinguished from the lap
  times at all? This is the discrete-time Gramian `sum_k Phi_k^T H_k^T H_k Phi_k`,
  and its eigenvalue spectrum says which directions in state space the
  observations touch. A direction with a zero eigenvalue is invisible: the filter
  can put anything there and no lap time changes.
- **`corner_separability`** -- specifically, can FL be told from FR? This is the
  Gramian restricted to differences between corners, which is the degeneracy the
  pre-registration is about.
- **`data_driven_share`** -- how much of the tyre signal the data can separate
  from fuel at all. exp18's definition, reused verbatim: fuel is linear in laps
  and tyre age advances one lap at a time, so only the *curvature* of the tyre
  loss is identified; the straight-line part is indistinguishable from fuel. The
  residual from a straight line, as a share of the total, is the number, and
  exp18 measured 5.96% for the shipped model.
- **`prior_sensitivity`** -- separately, whether the filter has forgotten its
  starting prior by the end of the stint. A convergence property, **not**
  comparable with the 5.96% above, and named differently because an earlier
  version of this module conflated the two and reported a sixteenfold
  improvement that was purely a mismatched metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import LapInput, drift, grip_coefficient, jacobian
from .ekf import DEFAULT_SUBSTEPS, run_filter
from .observation import grip_penalty_seconds, observation_jacobian
from .state import (
    N_STATE,
    T,
    W,
    FourCornerParameters,
    clamp_state,
    initial_covariance,
    initial_state,
)

#: Eigenvalues below this fraction of the largest are treated as numerically
#: zero. Relative rather than absolute because the Gramian's scale depends on the
#: stint length and on the units of the sensitivity coefficients.
RANK_TOLERANCE = 1e-10


@dataclass
class IdentifiabilityReport:
    """What a stint can and cannot tell us.

    Attributes:
        n_laps: Laps in the stint.
        eigenvalues: Gramian spectrum, descending, normalised by the largest.
        numerical_rank: Directions the observations actually touch.
        n_states: Total state dimension, for comparison with the rank.
        corner_separability: Smallest normalised eigenvalue among the three
            corner-difference directions. Near zero means the corners are
            exchangeable and three of the four wear states are decoration.
        condition_number: Largest over smallest non-negligible eigenvalue.
        wear_direction_strength: Normalised eigenvalue of the mean-wear
            direction, the one the product would actually report.
    """

    n_laps: int
    eigenvalues: np.ndarray
    numerical_rank: int
    n_states: int
    corner_separability: float
    condition_number: float
    wear_direction_strength: float

    def to_dict(self) -> dict:
        return {
            "n_laps": int(self.n_laps),
            "eigenvalues": [float(v) for v in self.eigenvalues],
            "numerical_rank": int(self.numerical_rank),
            "n_states": int(self.n_states),
            "corner_separability": float(self.corner_separability),
            "condition_number": float(self.condition_number),
            "wear_direction_strength": float(self.wear_direction_strength),
            "corners_are_separable": bool(self.corner_separability > 1e-6),
        }


def transition_matrix(
    x: np.ndarray,
    u: LapInput,
    params: FourCornerParameters,
    substeps: int = DEFAULT_SUBSTEPS,
) -> np.ndarray:
    """Discrete state transition across one lap, by composing linearised steps.

    The same construction the smoother uses, kept here rather than imported so
    the identifiability analysis does not silently change when the smoother is
    tuned.
    """
    dt = u.duration_s / substeps
    phi = np.eye(N_STATE)
    walk = np.asarray(x, dtype=float).copy()
    for _ in range(substeps):
        phi = (np.eye(N_STATE) + dt * jacobian(walk, u, params)) @ phi
        walk = clamp_state(walk + dt * drift(walk, u, params))
    return phi


def observability_gramian(
    inputs: list[LapInput],
    states: np.ndarray,
    params: FourCornerParameters,
) -> np.ndarray:
    """`sum_k Phi_k^T H_k^T H_k Phi_k`, the information the lap times carry.

    Evaluated along the filtered trajectory rather than at a single point,
    because the model is nonlinear and a Gramian at the initial state would
    describe a car that never ran.

    The result is a 10x10 symmetric positive semi-definite matrix whose null
    space is exactly the set of state perturbations that leave every lap time
    unchanged. Those directions are unidentifiable in the strict sense: no
    estimator, however good, can recover them from this data.
    """
    gramian = np.zeros((N_STATE, N_STATE))
    phi = np.eye(N_STATE)
    for k, u in enumerate(inputs):
        x = states[k]
        h = observation_jacobian(x, u, params).reshape(1, -1)
        stacked = h @ phi
        gramian += stacked.T @ stacked
        phi = transition_matrix(x, u, params) @ phi
    return 0.5 * (gramian + gramian.T)


def analyse(
    inputs: list[LapInput],
    observations: np.ndarray,
    params: FourCornerParameters,
    *,
    obs_scale_s: float = 0.25,
) -> IdentifiabilityReport:
    """Run the filter, then ask what its observations could possibly have told it."""
    result = run_filter(
        inputs,
        observations,
        initial_state(tread_temp_c=params.grip.temp_optimal_c),
        initial_covariance(),
        params,
        obs_scale_s=obs_scale_s,
    )
    gramian = observability_gramian(inputs, result.states, params)

    eigenvalues = np.linalg.eigvalsh(gramian)[::-1]
    largest = eigenvalues[0] if eigenvalues[0] > 0 else 1.0
    normalised = eigenvalues / largest
    rank = int((normalised > RANK_TOLERANCE).sum())

    # The three directions that distinguish corners from one another: FL-FR,
    # RL-RR, and front-rear. If the Gramian is blind to these, four wear states
    # collapse to one.
    contrasts = np.zeros((3, N_STATE))
    contrasts[0, 0], contrasts[0, 1] = 1.0, -1.0     # FL - FR
    contrasts[1, 2], contrasts[1, 3] = 1.0, -1.0     # RL - RR
    contrasts[2, 0:2], contrasts[2, 2:4] = 0.5, -0.5  # front - rear
    contrast_strength = np.array(
        [float(c @ gramian @ c.T) / largest for c in contrasts]
    )

    mean_wear = np.zeros(N_STATE)
    mean_wear[W] = 0.5
    wear_strength = float(mean_wear @ gramian @ mean_wear.T) / largest

    positive = normalised[normalised > RANK_TOLERANCE]
    condition = float(positive[0] / positive[-1]) if len(positive) else float("inf")

    return IdentifiabilityReport(
        n_laps=len(inputs),
        eigenvalues=normalised,
        numerical_rank=rank,
        n_states=N_STATE,
        corner_separability=float(contrast_strength.min()),
        condition_number=condition,
        wear_direction_strength=wear_strength,
    )


def data_driven_share(
    states: np.ndarray,
    inputs: list[LapInput],
    params: FourCornerParameters,
) -> dict[str, float]:
    """How much of the tyre signal the data can separate from fuel.

    **exp18's definition, deliberately reused verbatim so the numbers compare.**
    Fuel burn is linear in laps completed and tyre age advances one lap at a
    time, so within a run the two are collinear and the straight-line part of the
    tyre effect is indistinguishable from fuel. Only the *curvature* is
    identified by data. So: regress the model's own cumulative tyre loss on tyre
    age within the run, and report the residual variance as a share of the total.

    exp18 measured **5.96%** for the shipped two-state model.

    An earlier version of this function measured something else entirely -- how
    much the answer moved when the initial wear covariance was widened -- and
    reported 97.5% against exp18's 6.0% as though the two were comparable. They
    were not. That version was testing sensitivity to the *starting* prior while
    the fuel coefficient stayed pinned, which is not the collinearity exp18 is
    about, and it made a four-corner model look sixteen times better identified
    than a two-state one for no reason other than a mismatched metric.

    Adding states cannot add information, so a share above exp18's would be a
    result needing a very good explanation rather than a success.
    """
    penalty = np.array([
        grip_penalty_seconds(grip_coefficient(x[W], x[T], params), params) for x in states
    ])
    # Tyre age within the stint. The filter is run per stint, so lap index is age
    # up to the offset a scrubbed set introduces, and an offset does not change
    # a residual-from-a-straight-line.
    age = np.arange(len(penalty), dtype=float)
    if len(penalty) < 4 or np.ptp(age) < 2:
        return {"n_laps": int(len(penalty)), "nonlinear_share": float("nan")}

    design = np.column_stack([np.ones_like(age), age])
    coef, *_ = np.linalg.lstsq(design, penalty, rcond=None)
    residual = penalty - design @ coef

    total_var = float(np.var(penalty - penalty.mean()))
    residual_var = float(np.var(residual))
    if total_var <= 0:
        return {"n_laps": int(len(penalty)), "nonlinear_share": float("nan")}

    share = residual_var / total_var
    return {
        "n_laps": int(len(penalty)),
        "nonlinear_share": float(share),
        "linear_share": float(1.0 - share),
        "data_driven_share": float(share),
        "exp18_two_state_data_driven_share": 0.0596,
        "total_loss_s": float(penalty[-1] - penalty[0]),
    }


def prior_sensitivity(
    inputs: list[LapInput],
    observations: np.ndarray,
    params: FourCornerParameters,
    *,
    obs_scale_s: float = 0.25,
    widen: float = 10.0,
) -> dict[str, float]:
    """How much the answer depends on the starting wear prior.

    A different question from `data_driven_share` and reported under a different
    name so the two cannot be confused again. This one asks whether the filter
    has forgotten where it started by the end of the stint, which is a
    convergence property. It says nothing about the fuel collinearity and must
    not be compared with exp18's 6.0%.
    """
    x0 = initial_state(tread_temp_c=params.grip.temp_optimal_c)
    tight = run_filter(inputs, observations, x0, initial_covariance(), params,
                       obs_scale_s=obs_scale_s)
    loose = run_filter(inputs, observations, x0, initial_covariance(wear_sd=0.12 * widen),
                       params, obs_scale_s=obs_scale_s)

    tight_wear = tight.states[:, W].mean(axis=1)
    loose_wear = loose.states[:, W].mean(axis=1)
    scale = float(np.abs(tight_wear).mean()) or 1.0
    return {
        "prior_widened_by": float(widen),
        "mean_absolute_shift": float(np.abs(loose_wear - tight_wear).mean()),
        "relative_shift": float(np.abs(loose_wear - tight_wear).mean() / scale),
        "tight_final_wear": float(tight_wear[-1]),
        "loose_final_wear": float(loose_wear[-1]),
    }
