"""Routes must agree with the experiments they cite.

A routing table is a set of claims about which model is best at what. Claims
written by hand drift: an experiment is re-run, a number moves, and the table
keeps asserting the old winner while pointing at a file that no longer says so.
These tests re-read the result files and fail when that happens, so the table
cannot quietly become fiction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tyremind.models.literature import extended_ladder
from tyremind.models.router import ROUTES, Task, models_used, route, routing_table, summary

RESULTS = Path("experiments/results")


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


class TestTableIntegrity:
    def test_every_task_has_a_route(self):
        for task in Task:
            assert route(task).model

    def test_every_route_names_a_model_that_exists(self):
        """A route pointing at a model we do not have is a route that cannot run."""
        available = {m.name for m in extended_ladder()}
        for r in ROUTES.values():
            assert r.model in available, f"{r.task} routes to unknown model {r.model!r}"

    def test_no_route_can_be_built_without_evidence(self):
        from dataclasses import replace
        with pytest.raises(ValueError, match="preference"):
            replace(ROUTES[Task.PIT_TIMING], evidence="")

    def test_the_router_actually_routes_away_from_us(self):
        """The point of the exercise.

        A table that always picks our own model is not routing, it is a
        preference with extra steps. Two of five tasks must go elsewhere:
        lap-time forecasting to the pooled regression, practice-to-race to
        Cappello & Hoegh.
        """
        assert summary()["tasks_routed_away"] >= 2
        assert len(models_used()) >= 3


class TestRoutesMatchTheirEvidence:
    def test_lap_time_forecast_route_matches_exp19(self):
        rows = load("exp19_field_comparison")["lap_time_prediction"]
        best = min(rows, key=lambda r: r["crps"])
        r = route(Task.FORECAST_LAP_TIME)
        assert best["model"] == r.model
        assert best["crps"] == pytest.approx(r.winner_score, abs=5e-4)

    def test_degradation_route_matches_exp19(self):
        rows = load("exp19_field_comparison")["degradation_recovery"]
        best = min(rows, key=lambda r: r["rate_mae"])
        r = route(Task.RECOVER_DEGRADATION_RATE)
        assert best["model"] == r.model
        assert best["rate_mae"] == pytest.approx(r.winner_score, abs=5e-5)

    def test_pit_timing_route_matches_exp22_and_is_marked_a_tie(self):
        rows = load("exp22_pit_stop_validation")["like_for_like"]
        best = min(rows, key=lambda r: r["mae_laps"])
        r = route(Task.PIT_TIMING)
        assert best["model"] == r.model
        assert best["mae_laps"] == pytest.approx(r.winner_score, abs=0.01)
        # Three models sit within one standard error. Marking this decisive would
        # be claiming a 0.06-lap win that the data does not support.
        within = [x for x in rows if x["mae_laps"] <= best["mae_laps"] + best["mae_se"]]
        assert len(within) > 1
        assert r.margin_is_decisive is False

    def test_practice_to_race_route_goes_to_a_competitor(self):
        rows = load("exp27_decircularised_practice_to_race")["summary"]
        best = max(rows, key=lambda r: r["skill_vs_own_climatology"])
        r = route(Task.PRACTICE_TO_RACE)
        assert best["model"] == r.model
        assert r.model != "TyreMind state-space"

    def test_every_model_has_negative_practice_to_race_skill(self):
        """The finding behind that route, pinned so it cannot be softened.

        No model beats its own average race rate from a Friday fit. If this ever
        turns positive the route's rationale needs rewriting, not quietly keeping.
        """
        rows = load("exp27_decircularised_practice_to_race")["summary"]
        assert all(r["skill_vs_own_climatology"] < 0 for r in rows)

    def test_confidence_route_matches_exp30(self):
        rows = load("exp30_pit_confidence_calibration")["summary"]
        best = min(rows, key=lambda r: abs(r["calibration_gap"]))
        r = route(Task.CONFIDENCE)
        assert best["model"] == r.model
        assert best["calibration_gap"] == pytest.approx(r.winner_score, abs=5e-3)


class TestDisplay:
    def test_the_table_is_serialisable_and_complete(self):
        table = routing_table()
        assert len(table) == len(Task)
        for row in table:
            assert row["evidence"] and row["rationale"]
        json.dumps(table)
