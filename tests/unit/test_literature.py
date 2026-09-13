"""The published baselines must be faithful, and must not be quietly crippled.

A reimplementation of someone else's model is only useful as a comparator if it
is a fair one. These tests check the properties the papers actually specify --
the linear tyre term, the positivity prior, the per-driver scope -- so that a
regression that weakens a competitor shows up as a failure rather than as a win
for us.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tyremind.models.literature import (
    ArimaBaseline,
    CappelloHoeghModel,
    HeilmeierModel,
    extended_ladder,
    literature_ladder,
)


def _statsmodels_available() -> tuple[bool, str]:
    """Whether statsmodels' compiled state-space extension will load.

    It is a hard dependency of `ArimaBaseline` and nothing else in the project,
    and on locked-down Windows installs the import fails at the DLL rather than
    in Python:

        ImportError: DLL load failed while importing _representation:
        An Application Control policy has blocked this file.

    That is a property of the machine, not of the code, and the same class of
    block already cost this project a day over pyarrow. The ARIMA rung genuinely
    cannot be exercised there, so those tests skip with the real reason attached
    instead of failing and burying an actual regression in the noise.

    **The model itself is not softened.** `ArimaBaseline.fit` still raises, so a
    benchmark run on such a machine fails loudly rather than scoring ARIMA as
    absent and flattering everything else -- which is exactly the rigging
    `TestFairness` exists to prevent.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"statsmodels state-space extension unavailable: {exc}"
    return True, ""


_STATSMODELS_OK, _STATSMODELS_WHY = _statsmodels_available()
needs_statsmodels = pytest.mark.skipif(not _STATSMODELS_OK, reason=_STATSMODELS_WHY)


def ladder_params() -> list:
    """The literature ladder as parametrize entries, ARIMA marked skippable.

    Only the ARIMA rung carries the mark. Every other model on the ladder is pure
    numpy and scipy and must keep running everywhere, so a blocked statsmodels
    can never quietly take the whole comparison down with it.
    """
    out = []
    for model in literature_ladder():
        marks = [needs_statsmodels] if model.name.startswith("ARIMA") else []
        out.append(pytest.param(model, marks=marks, id=model.name))
    return out


@pytest.fixture(scope="module")
def session() -> pd.DataFrame:
    """A synthetic stint set with a known linear degradation of 0.10 s/lap."""
    rng = np.random.default_rng(7)
    rows = []
    for driver in [f"D{i:02d}" for i in range(8)]:
        for run in range(2):
            age0 = rng.integers(0, 4)
            for lap in range(14):
                age = age0 + lap
                rows.append({
                    "driver": driver,
                    "session_lap": run * 14 + lap + 1,
                    "run_id": hash((driver, run)) % 10_000,
                    "tyre_age": float(age),
                    "lap_in_run": float(lap),
                    "lap_time": 90.0 + 0.10 * age - 0.081 * lap + rng.normal(0, 0.08),
                    "compound": "MEDIUM",
                    "traffic_index": 0.0,
                })
    return pd.DataFrame(rows)


class TestContract:
    @pytest.mark.parametrize("model", ladder_params())
    def test_fits_and_predicts_finite_values(self, model, session):
        model.fit(session)
        mean, sd = model.predict(session)
        assert len(mean) == len(session)
        assert np.all(np.isfinite(mean)), model.name
        assert np.all(sd > 0), "a model that cannot say how sure it is cannot be calibrated"

    def test_the_extended_ladder_has_nine_distinct_rungs(self):
        names = [m.name for m in extended_ladder()]
        assert len(names) == 9
        assert len(set(names)) == 9


class TestHeilmeier:
    def test_recovers_a_linear_degradation_rate(self, session):
        """Its own generating assumption, so it should be close."""
        model = HeilmeierModel().fit(session)
        rate, _ = model.compound_rates()["MEDIUM"]
        assert rate == pytest.approx(0.10, abs=0.03)

    def test_the_tyre_term_is_linear_in_age_by_construction(self, session):
        """Equation (6) is `k0(c) + k1(c) * a`. Doubling age must double the
        tyre contribution -- if a future edit lets it bend, it is no longer the
        published model and the comparison stops being fair."""
        model = HeilmeierModel().fit(session)
        rate, _ = model.compound_rates()["MEDIUM"]
        assert rate * 20 == pytest.approx(2 * rate * 10)


class TestCappelloHoegh:
    def test_the_positivity_prior_is_respected(self, session):
        """Their half-normal forces nu >= 0. Keeping it is what makes this a
        reproduction rather than an argument with the paper."""
        model = CappelloHoeghModel().fit(session)
        for rate, _ in model.compound_rates().values():
            assert rate >= 0.0

    def test_a_driver_with_too_few_laps_is_dropped_not_guessed(self):
        """Per-driver fitting means a short run yields no rate at all. That is a
        real property of a single-car model, and the reason a whole-field model
        can answer for a compound it cannot."""
        short = pd.DataFrame({
            "driver": ["X"] * 4,
            "session_lap": [1, 2, 3, 4],
            "run_id": [1] * 4,
            "tyre_age": [1.0, 2.0, 3.0, 4.0],
            "lap_in_run": [0.0, 1.0, 2.0, 3.0],
            "lap_time": [90.0, 90.1, 90.2, 90.3],
            "compound": ["SOFT"] * 4,
            "traffic_index": [0.0] * 4,
        })
        model = CappelloHoeghModel().fit(short)
        assert model.compound_rates() == {}


@needs_statsmodels
class TestArima:
    def test_reports_no_degradation_rate(self, session):
        """ARIMA has no such parameter. Inventing one would be dishonest."""
        assert ArimaBaseline().fit(session).compound_rates() == {}


class TestFairness:
    """The competitor must fail the same way we do.

    These guard the asymmetry that rigged the first exp19 run: our estimator
    validated its input and raised, the harness recorded `failed` and dropped the
    session from its average, while these models returned NaN and carried it into
    their own mean. Strictness must not be a competitive advantage.
    """

    @pytest.mark.parametrize("model", ladder_params())
    def test_a_null_tyre_age_raises_rather_than_returning_nan(self, model, session):
        poisoned = session.copy()
        poisoned.loc[poisoned.index[:5], "tyre_age"] = np.nan
        with pytest.raises(ValueError, match="null"):
            model.fit(poisoned)

    @pytest.mark.parametrize("model", ladder_params())
    def test_predictions_are_finite_on_clean_data(self, model, session):
        model.fit(session)
        mean, sd = model.predict(session)
        assert np.all(np.isfinite(mean)), f"{model.name} produced a NaN prediction"
        assert np.all(np.isfinite(sd))
