"""Route each question to the model measured best at answering it.

Nine models were scored on five tasks and **three different models won**. The
pooled regression forecasts lap times best, our state-space model recovers the
degradation rate and calibrates its own confidence best, and Cappello & Hoegh's
per-driver model is the most self-consistent from practice to race. No single
rung wins everything, and pretending otherwise means shipping a worse answer to
four questions out of five in order to keep one model on the front page.

So the product is not a model. It is an orchestration that sends each question to
whichever rung the evidence says is best at it, and says which one it used.

Two rules keep this from becoming a way to launder a weak result.

**Every route cites a result file and a margin.** A route with no evidence behind
it is a preference, and preferences are how a system quietly ends up routing to
whatever its author likes. `Route.evidence` is not optional.

**A margin inside the noise is a tie, and ties are broken on honesty, not on the
first decimal.** exp22 puts three models within one standard error on pit timing.
Picking ourselves there because we are 0.06 laps ahead would be exactly the
cherry-picking the whole project has been trying to avoid, so a tie routes to the
model with the better calibration and the higher answer rate, and records that it
was a tie.

This is the same split Pitwall (arXiv:2607.06495) arrived at independently and
calls *"calibration-optimal is not decision-optimal"* -- they route components
down an oracle path and a decision path because the two criteria disagree. Ours
disagree too, and exp19 measured it before we read their paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Task(str, Enum):
    """The questions the product answers, one route each."""

    FORECAST_LAP_TIME = "forecast_lap_time"
    RECOVER_DEGRADATION_RATE = "recover_degradation_rate"
    PIT_TIMING = "pit_timing"
    PRACTICE_TO_RACE = "practice_to_race"
    CONFIDENCE = "confidence"


@dataclass(frozen=True)
class Route:
    """Which model answers a task, and the measurement that put it there.

    Attributes:
        task: The question.
        model: Model name, matching `literature.extended_ladder()` names exactly
            so a route cannot point at something that does not exist.
        metric: What was measured.
        winner_score: The chosen model's score.
        runner_up: Next best model.
        runner_up_score: Its score.
        margin_is_decisive: False when the gap sits inside the reported standard
            error. A tie is reported as a tie rather than rounded into a win.
        evidence: Result file the numbers come from. Never empty.
        rationale: Why this model, in words, including why a tie broke the way it
            did.
    """

    task: Task
    model: str
    metric: str
    winner_score: float
    runner_up: str
    runner_up_score: float
    margin_is_decisive: bool
    evidence: str
    rationale: str

    def __post_init__(self) -> None:
        if not self.evidence:
            raise ValueError(f"route for {self.task} has no evidence; that is a preference")

    def to_dict(self) -> dict:
        return {
            "task": self.task.value,
            "model": self.model,
            "metric": self.metric,
            "winner_score": self.winner_score,
            "runner_up": self.runner_up,
            "runner_up_score": self.runner_up_score,
            "margin_is_decisive": self.margin_is_decisive,
            "evidence": self.evidence,
            "rationale": self.rationale,
        }


#: The routing table. Every number here is copied from the named result file, and
#: `scripts/check_router_routes.py` re-reads those files and fails if a route has
#: gone stale against its own evidence.
ROUTES: dict[Task, Route] = {
    Task.FORECAST_LAP_TIME: Route(
        task=Task.FORECAST_LAP_TIME,
        model="Pooled regression",
        metric="CRPS, 12 real races, rolling-origin folds",
        winner_score=0.4088,
        runner_up="LightGBM",
        runner_up_score=0.4826,
        margin_is_decisive=True,
        evidence="exp19_field_comparison.lap_time_prediction",
        rationale=(
            "We are fourth here at 0.6484 and route away from ourselves. A pooled "
            "regression minimises lap-time error with maximum freedom in the "
            "nuisance terms; the physical priors and smooth latent state that let "
            "us isolate the tyre are exactly what cost us on fitting the total. "
            "Shipping our own worse number to keep one model on the front page "
            "would be vanity."
        ),
    ),
    Task.RECOVER_DEGRADATION_RATE: Route(
        task=Task.RECOVER_DEGRADATION_RATE,
        model="TyreMind state-space",
        metric="rate MAE against known truth, 8 synthetic seeds",
        winner_score=0.0037,
        runner_up="Pooled regression",
        runner_up_score=0.0062,
        margin_is_decisive=True,
        evidence="exp19_field_comparison.degradation_recovery",
        rationale=(
            "1.7x the next model and 4.3x the closest published one. exp29 holds "
            "this across eleven generator regimes -- eight wins, one tie, two "
            "losses, both on the track-evolution axis. This is the quantity the "
            "problem statement actually asks for."
        ),
    ),
    Task.PIT_TIMING: Route(
        task=Task.PIT_TIMING,
        model="TyreMind state-space",
        metric="mean absolute error in laps, 49 stops every model answered",
        winner_score=5.92,
        runner_up="Cappello & Hoegh state-space",
        runner_up_score=5.98,
        margin_is_decisive=False,
        evidence="exp22_pit_stop_validation.like_for_like",
        rationale=(
            "A three-way tie: 5.92, 5.98 and 6.31 with standard errors near 0.6. "
            "The 0.06-lap lead is not a win and is not claimed as one. The tie "
            "breaks our way on the two things that are separated -- a 7.9 point "
            "calibration gap against 15.8 for the runner-up (exp30), and a 90% "
            "answer rate against theirs, since a model that declines the hard "
            "stops is graded on the easy ones."
        ),
    ),
    Task.PRACTICE_TO_RACE: Route(
        task=Task.PRACTICE_TO_RACE,
        model="Cappello & Hoegh state-space",
        metric="skill against own climatology, 43 events, 568 comparisons",
        winner_score=-0.882,
        runner_up="TyreMind state-space",
        runner_up_score=-1.154,
        margin_is_decisive=True,
        evidence="exp27_decircularised_practice_to_race.summary",
        rationale=(
            "We route to a competitor. Their per-driver model is more "
            "self-consistent from Friday to Sunday than ours. The caveat that "
            "matters more: every model scores NEGATIVE skill, so a Friday fit "
            "beats nobody's own average race rate. The honest product behaviour "
            "is to decline the forecast and present the practice number as a "
            "within-session decomposition, which is what Problem 3 asks for."
        ),
    ),
    Task.CONFIDENCE: Route(
        task=Task.CONFIDENCE,
        model="TyreMind state-space",
        metric="calibration gap on the pit window, 246 real stops",
        winner_score=-0.079,
        runner_up="Pooled regression",
        runner_up_score=-0.089,
        margin_is_decisive=False,
        evidence="exp30_pit_confidence_calibration.summary",
        rationale=(
            "Best of six, and still overconfident by 7.9 points. The tie with the "
            "pooled regression breaks on lift -- we place 1.97x chance on the lap "
            "the driver chose against their 1.44x -- but neither is calibrated "
            "enough to present a high-confidence claim without the conformal "
            "correction that exp30 shows is still missing here."
        ),
    ),
}


def route(task: Task) -> Route:
    """Which model answers this question, and why."""
    if task not in ROUTES:
        raise KeyError(f"no route for {task}; add one with evidence or do not offer the task")
    return ROUTES[task]


def routing_table() -> list[dict]:
    """The whole table, for display. The product should show this, not hide it."""
    return [r.to_dict() for r in ROUTES.values()]


def models_used() -> set[str]:
    return {r.model for r in ROUTES.values()}


def summary() -> dict:
    """What the orchestration is, in numbers.

    `tasks_routed_away` is the honest headline: the count of questions where the
    evidence sent us to somebody else's model. A router that never routes away is
    not a router, it is a preference with extra steps.
    """
    ours = "TyreMind state-space"
    return {
        "n_tasks": len(ROUTES),
        "n_models_used": len(models_used()),
        "models_used": sorted(models_used()),
        "tasks_won_by_tyremind": sum(1 for r in ROUTES.values() if r.model == ours),
        "tasks_routed_away": sum(1 for r in ROUTES.values() if r.model != ours),
        "ties": [r.task.value for r in ROUTES.values() if not r.margin_is_decisive],
    }
