"""The state is the contract and the verifier is what enforces it.

Both are tested for the failure they exist to prevent rather than for coverage.
The state exists so four surfaces cannot drift apart; the verifier exists because
a fluent generator invents numbers exactly when the grounding state is thin,
which is early in a stint, which is when a strategist is reading.
"""

from __future__ import annotations

import json
import math

import pytest

from tyremind.models.state import (
    Contribution,
    Estimate,
    PitAdvice,
    TrustReport,
    TyreState,
)
from tyremind.models.verifier import VerificationLog, publish, verify


@pytest.fixture
def state() -> TyreState:
    return TyreState(
        session_id="2024-monza-R",
        driver="VER",
        lap=20,
        compound="MEDIUM",
        tyre_age=14.0,
        laps_in_stint=12.0,
        regime="linear",
        degradation_rate=Estimate(0.198, 0.022, "s/lap", "MEDIUM, 12 laps", calibrated=True),
        contributions=[
            Contribution("tyre", "Tyre degradation", Estimate(1.693, 0.400, "s"), is_tyre=True),
            Contribution("fuel", "Fuel burn-off", Estimate(-0.549, 0.111, "s")),
            Contribution("residual", "Unexplained", Estimate(2.768, None, "s"), is_residual=True),
        ],
        projected_loss={3: Estimate(0.62, 0.19, "s"), 15: Estimate(4.31, 1.08, "s")},
        life_remaining_pct=Estimate(62.0, 6.5, "%", "cliff onset, exp17, MEDIUM"),
        pit=PitAdvice(
            recommended_lap=39, confidence_in_lap=0.093,
            window=(37, 42), confidence_in_window=0.51,
            cost_of_being_late=Estimate(1.4, 0.5, "s"),
            alternative_compound="SOFT",
        ),
        trust=TrustReport(0.030, 4, False, 0.952, 69206, "Four methods agree within 0.030 s/lap."),
    )


class TestEstimate:
    def test_an_interval_is_always_available_when_a_number_is(self):
        assert Estimate(0.2, 0.02).ci95 == pytest.approx((0.16080, 0.23920), abs=1e-4)

    def test_unknown_is_a_value_not_a_gap(self):
        """"We cannot answer this" has to survive to the screen as words. If it
        became a null, a template would render it as zero and the Monaco case --
        degradation is not what decides this stop -- would silently become a
        confident recommendation."""
        unknown = Estimate.unknown("degradation is not what decides this stop", "s/lap")
        assert unknown.known is False
        assert unknown.ci95 is None
        assert unknown.to_dict()["reason"]

    def test_serialised_output_is_strict_json(self, state):
        """No bare NaN or Infinity. One such field poisons a whole frame, and it
        has already happened once on the live WebSocket."""
        def boom(token):
            raise AssertionError(f"non-finite token in payload: {token}")

        json.loads(json.dumps(state.to_dict()), parse_constant=boom)

    def test_a_non_finite_value_serialises_as_null_not_nan(self):
        assert Estimate(float("nan"), float("inf")).to_dict()["value"] is None


class TestState:
    def test_the_residual_is_reachable_and_labelled(self, state):
        """It must be renderable as its own band. Folding it into the other causes
        to make the bars add up would claim we explain more than we do."""
        assert state.unexplained_seconds.value == pytest.approx(2.768)
        assert any(c.is_residual for c in state.contributions)

    def test_tyre_contribution_is_flagged_not_string_matched(self, state):
        assert state.tyre_seconds.value == pytest.approx(1.693)

    def test_numeric_claims_include_interval_bounds(self, state):
        claims = state.numeric_claims()
        assert claims["degradation_rate"] == pytest.approx(0.198)
        assert "degradation_rate_low" in claims and "degradation_rate_high" in claims
        assert claims["recommended_lap"] == 39.0
        assert claims["confidence_in_window"] == pytest.approx(0.51)


class TestVerifier:
    def test_a_true_sentence_passes(self, state):
        verdict = verify("Degradation is 0.198 s/lap on the mediums.", state)
        assert verdict.supported

    def test_an_invented_number_is_caught(self, state):
        """The failure this module exists for."""
        verdict = verify("Degradation is 0.450 s/lap on the mediums.", state)
        assert not verdict.supported
        assert verdict.unsupported[0].number == pytest.approx(0.450)

    def test_a_percentage_matches_a_stored_fraction(self, state):
        assert verify("We are 51% confident in the window.", state).supported

    def test_a_lap_number_cannot_be_satisfied_by_a_non_lap_quantity(self, state):
        """Without the unit check, "lap 62" would match life_remaining_pct of
        62.0 and a fabricated pit lap would verify clean."""
        assert not verify("Box on lap 62.", state).supported

    def test_a_qualitative_sentence_needs_no_grounding(self, state):
        assert verify("The tyre is wearing steadily.", state).supported

    def test_rounding_for_display_is_not_a_violation(self, state):
        assert verify("Degradation is 0.20 s/lap.", state).supported


class TestPublish:
    def test_the_first_verifiable_candidate_wins(self, state):
        text, verdict = publish(
            ["Degradation is 0.999 s/lap.", "Degradation is 0.198 s/lap."],
            "fallback", state)
        assert text == "Degradation is 0.198 s/lap."
        assert verdict is not None

    def test_everything_unverifiable_falls_back(self, state):
        text, verdict = publish(["0.999 s/lap.", "0.888 s/lap."], "fallback", state)
        assert text == "fallback"
        assert verdict is None

    def test_the_verification_rate_is_recorded(self, state):
        log = VerificationLog()
        publish(["Degradation is 0.198 s/lap."], "fallback", state, log)
        publish(["Degradation is 9.999 s/lap."], "fallback", state, log)
        assert log.checked == 2
        assert log.published == 1
        assert log.fell_back == 1
        assert log.verification_rate == pytest.approx(0.5)
        assert not math.isnan(log.verification_rate)
