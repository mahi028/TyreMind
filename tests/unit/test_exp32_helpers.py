"""The numerical helpers exp32 rests on, pinned.

exp32 concludes that traffic *masks* degradation rather than reducing it, and it
concludes that from the sign of a mediated proportion. Four things underneath
that conclusion can bend it without crashing:

  * a `polyfit` that raises on a degenerate stint and takes the run with it, or
    returns a non-finite coefficient and takes the mean with it;
  * a group demeaning that quietly loses rows or demeans by the wrong key, which
    would turn a within-driver contrast back into a between-car one;
  * an OLS that raises on a singular bootstrap replicate;
  * a mediation decomposition whose indirect effect does not actually equal
    `c - c'`, which is the identity the whole argument reads.

Each gets a test here, including the suppression case -- a mediator that makes
the treatment effect *larger* -- because that is the case exp32 actually found
and the one a decomposition is most likely to get wrong.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_experiment():
    """Import exp32 by path; `experiments/` is a script directory, not a package."""
    path = ROOT / "experiments" / "exp32_traffic_mechanism.py"
    spec = importlib.util.spec_from_file_location("exp32_traffic_mechanism", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp32 = _load_experiment()


class TestFitSlope:
    def test_recovers_a_known_slope(self):
        x = np.arange(12.0)
        assert exp32.fit_slope(x, 3.0 + 0.07 * x) == pytest.approx(0.07)

    def test_returns_none_rather_than_raising_on_a_constant_x(self):
        """The LinAlgError case that has ended two runs in this project."""
        assert exp32.fit_slope(np.full(10, 5.0), np.arange(10.0)) is None

    def test_returns_none_on_non_finite_input(self):
        x = np.arange(10.0)
        y = x.copy()
        y[3] = np.nan
        assert exp32.fit_slope(x, y) is None
        y[3] = np.inf
        assert exp32.fit_slope(x, y) is None

    def test_returns_none_when_there_is_nothing_to_fit(self):
        assert exp32.fit_slope(np.array([1.0]), np.array([2.0])) is None
        assert exp32.fit_slope(np.arange(5.0), np.arange(4.0)) is None


class TestGroupDemean:
    def test_each_group_ends_with_zero_mean(self):
        values = np.array([[1.0], [3.0], [10.0], [20.0], [30.0]])
        codes = np.array([0, 0, 1, 1, 1])
        out = exp32.group_demean(values, codes)
        assert out[:2, 0].sum() == pytest.approx(0.0)
        assert out[2:, 0].sum() == pytest.approx(0.0)
        assert out.shape == values.shape

    def test_a_between_group_effect_cannot_survive_it(self):
        """The reason S2 exists: a constant per group is removed entirely."""
        rng = np.random.default_rng(7)
        codes = np.repeat(np.arange(20), 5)
        car_pace = np.repeat(rng.normal(size=20) * 10.0, 5)
        within = rng.normal(size=codes.size)
        out = exp32.group_demean((car_pace + within).reshape(-1, 1), codes)[:, 0]
        assert np.corrcoef(out, car_pace)[0, 1] == pytest.approx(0.0, abs=1e-10)

    def test_handles_several_columns_at_once(self):
        values = np.column_stack([np.arange(6.0), np.arange(6.0) ** 2])
        codes = np.array([0, 0, 0, 1, 1, 1])
        out = exp32.group_demean(values, codes)
        for column in range(2):
            assert out[:3, column].sum() == pytest.approx(0.0)
            assert out[3:, column].sum() == pytest.approx(0.0)


class TestOls:
    def test_recovers_known_coefficients(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(400, 2))
        y = 5.0 + 1.5 * x[:, 0] - 0.4 * x[:, 1]
        beta = exp32.ols(y, x)
        assert beta == pytest.approx([1.5, -0.4])

    def test_returns_none_rather_than_raising_on_a_degenerate_design(self):
        x = np.zeros((5, 8))
        assert exp32.ols(np.arange(5.0), x) is None

    def test_returns_none_on_non_finite_input(self):
        x = np.arange(20.0).reshape(-1, 1)
        y = np.arange(20.0)
        y[2] = np.nan
        assert exp32.ols(y, x) is None


class TestMediationPaths:
    def test_full_mediation_is_recovered(self):
        """Treatment acts only through the mediator: proportion mediated is 1."""
        rng = np.random.default_rng(3)
        treatment = rng.normal(size=4000)
        mediator = -0.8 * treatment + rng.normal(size=4000) * 0.3
        outcome = 0.5 * mediator + rng.normal(size=4000) * 0.3
        out = exp32.mediation_paths(treatment, mediator.reshape(-1, 1), outcome)
        assert out["a"][0] == pytest.approx(-0.8, abs=0.02)
        assert out["b"][0] == pytest.approx(0.5, abs=0.02)
        assert out["c"] == pytest.approx(-0.4, abs=0.03)
        assert out["c_prime"] == pytest.approx(0.0, abs=0.03)
        assert out["proportion_mediated"] == pytest.approx(1.0, abs=0.1)

    def test_no_mediation_is_recovered(self):
        """The mediator moves with the treatment but does not touch the outcome."""
        rng = np.random.default_rng(5)
        treatment = rng.normal(size=4000)
        mediator = -0.8 * treatment + rng.normal(size=4000) * 0.3
        outcome = -0.4 * treatment + rng.normal(size=4000) * 0.3
        out = exp32.mediation_paths(treatment, mediator.reshape(-1, 1), outcome)
        assert out["b"][0] == pytest.approx(0.0, abs=0.05)
        assert out["c_prime"] == pytest.approx(out["c"], abs=0.05)
        assert abs(out["proportion_mediated"]) < 0.1

    def test_suppression_gives_a_negative_proportion(self):
        """The case exp32 found: controlling for the mediator *enlarges* c.

        Here the mediator rises with the treatment and with the outcome, so
        partialling it out makes the direct effect more negative than the total.
        A decomposition that clipped or absolute-valued the proportion would
        report partial mediation for what is really suppression.
        """
        rng = np.random.default_rng(13)
        treatment = rng.normal(size=6000)
        mediator = 0.6 * treatment + rng.normal(size=6000) * 0.5
        outcome = -0.5 * treatment + 0.4 * mediator + rng.normal(size=6000) * 0.3
        out = exp32.mediation_paths(treatment, mediator.reshape(-1, 1), outcome)
        assert out["c"] < 0
        assert out["c_prime"] < out["c"]
        assert out["proportion_mediated"] < 0

    def test_indirect_effect_equals_c_minus_c_prime(self):
        """The identity the argument reads. It must hold for a mediator block too."""
        rng = np.random.default_rng(17)
        treatment = rng.normal(size=2000)
        mediators = np.column_stack(
            [0.5 * treatment + rng.normal(size=2000) for _ in range(4)]
        )
        outcome = -0.3 * treatment + mediators @ np.array([0.2, -0.1, 0.4, 0.05])
        outcome = outcome + rng.normal(size=2000) * 0.2
        out = exp32.mediation_paths(treatment, mediators, outcome)
        assert out["indirect"] == pytest.approx(out["indirect_as_difference"], abs=1e-9)

    def test_covariates_are_partialled_out_of_every_path(self):
        rng = np.random.default_rng(19)
        covariate = rng.normal(size=3000)
        treatment = 0.7 * covariate + rng.normal(size=3000) * 0.5
        mediator = rng.normal(size=3000)
        # The outcome depends on the covariate only; a specification that
        # ignored it would attribute that to the treatment.
        outcome = 1.2 * covariate + rng.normal(size=3000) * 0.2
        out = exp32.mediation_paths(
            treatment, mediator.reshape(-1, 1), outcome, covariate.reshape(-1, 1)
        )
        assert out["c"] == pytest.approx(0.0, abs=0.03)
        assert out["c_prime"] == pytest.approx(0.0, abs=0.03)


class TestPercentileInterval:
    def test_covers_the_truth_of_a_known_distribution(self):
        rng = np.random.default_rng(23)
        draws = rng.normal(loc=2.0, scale=1.0, size=5000).tolist()
        out = exp32.percentile_interval(draws)
        assert out["lo"] == pytest.approx(2.0 - 1.96, abs=0.1)
        assert out["hi"] == pytest.approx(2.0 + 1.96, abs=0.1)

    def test_refuses_to_invent_an_interval_from_too_few_draws(self):
        out = exp32.percentile_interval([1.0, 2.0, 3.0])
        assert out["lo"] is None and out["hi"] is None

    def test_non_finite_draws_are_dropped_not_propagated(self):
        draws = [1.0] * 50 + [np.nan, np.inf, -np.inf]
        out = exp32.percentile_interval(draws)
        assert out["n"] == 50
        assert out["lo"] == pytest.approx(1.0)


class TestVerdict:
    """The pre-registered decision rule, exercised on constructed results."""

    @staticmethod
    def _result(*, c, c_prime, a, b, prop_lo, prop_hi, direct_lo, direct_hi, ind_lo, ind_hi):
        return {
            "total_effect_c": c,
            "direct_effect_c_prime": c_prime,
            "indirect_effect": c - c_prime,
            "proportion_mediated": (c - c_prime) / c,
            "direct_share_of_total": abs(c_prime) / abs(c),
            "a": a,
            "b": b,
            "ci": {
                "proportion_mediated": {"n": 2000, "lo": prop_lo, "hi": prop_hi},
                "c_prime": {"n": 2000, "lo": direct_lo, "hi": direct_hi},
                "indirect": {"n": 2000, "lo": ind_lo, "hi": ind_hi},
                "c": {"n": 2000, "lo": c - 0.01, "hi": c + 0.01},
            },
        }

    def test_calls_the_mechanism_when_energy_carries_the_effect(self):
        out = exp32.verdict(
            self._result(
                c=-0.10, c_prime=-0.01, a=-0.5, b=0.4,
                prop_lo=0.6, prop_hi=0.99, direct_lo=-0.04, direct_hi=0.02,
                ind_lo=-0.12, ind_hi=-0.05,
            )
        )
        assert out["call"] == "A_real_mechanism"

    def test_calls_the_artefact_when_the_direct_effect_survives(self):
        out = exp32.verdict(
            self._result(
                c=-0.13, c_prime=-0.21, a=0.16, b=0.52,
                prop_lo=-1.6, prop_hi=-0.18, direct_lo=-0.33, direct_hi=-0.11,
                ind_lo=0.02, ind_hi=0.14,
            )
        )
        assert out["call"] == "B_measurement_artefact"
        assert out["diagnostics"]["a_has_the_sign_mechanism_A_requires"] is False

    def test_a_mediator_that_does_not_predict_the_outcome_is_inconclusive(self):
        """Pre-registered as a live risk: exp25 found b negative at stint level."""
        out = exp32.verdict(
            self._result(
                c=-0.10, c_prime=-0.09, a=-0.5, b=-0.4,
                prop_lo=-0.5, prop_hi=0.5, direct_lo=-0.15, direct_hi=-0.03,
                ind_lo=-0.05, ind_hi=0.03,
            )
        )
        assert out["call"] == "inconclusive"
        assert "b <= 0" in out["reason"]

    def test_a_total_effect_of_the_wrong_sign_is_reported_as_a_failed_replication(self):
        out = exp32.verdict(
            self._result(
                c=0.05, c_prime=0.04, a=-0.5, b=0.4,
                prop_lo=0.0, prop_hi=0.5, direct_lo=0.01, direct_hi=0.07,
                ind_lo=0.0, ind_hi=0.02,
            )
        )
        assert out["call"] == "inconclusive"
        assert "did not replicate" in out["reason"]

    def test_a_missing_specification_is_inconclusive_not_a_crash(self):
        assert exp32.verdict(None)["call"] == "inconclusive"
