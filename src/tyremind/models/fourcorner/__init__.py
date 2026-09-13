"""A four-corner physics-informed continuous-discrete EKF, kept separate.

This package is a **candidate**, not the shipped estimator. Nothing in
`api/`, `apps/web/` or the router imports it, and that isolation is deliberate:
`experiments/PREREGISTRATION_exp35.md` fixes, in advance, the four thresholds it
has to clear before it is allowed near the product, and predicts on the record
that it will fail at least three of them.

The reason to build it anyway is that the prediction is worth testing. The
scalar model resolves the fuel/tyre collinearity with a prior and exp18 measured
that only 6% of its answer is data-driven. A four-corner model driven by measured
per-corner loads has a different information structure, and whether that helps is
an empirical question this package exists to answer.

Read `state.py` first; it defines the ten-dimensional state and says which parts
are estimated and which are measured.
"""

from .dynamics import LapInput, drift, grip_coefficient, jacobian, wear_rate
from .ekf import FilterResult, propagate, rts_smooth, run_filter
from .inputs import LapInputBundle, asymmetry_report, build_inputs
from .observation import predict_lap_time, skewed_t_logpdf
from .state import (
    CORNERS,
    FourCornerParameters,
    GripParameters,
    ProcessNoise,
    SessionParameters,
    ThermoMechanicalParameters,
    describe,
    initial_covariance,
    initial_state,
)

__all__ = [
    "CORNERS",
    "FilterResult",
    "FourCornerParameters",
    "GripParameters",
    "LapInput",
    "LapInputBundle",
    "ProcessNoise",
    "SessionParameters",
    "ThermoMechanicalParameters",
    "asymmetry_report",
    "build_inputs",
    "describe",
    "drift",
    "grip_coefficient",
    "initial_covariance",
    "initial_state",
    "jacobian",
    "predict_lap_time",
    "propagate",
    "rts_smooth",
    "run_filter",
    "skewed_t_logpdf",
    "wear_rate",
]
