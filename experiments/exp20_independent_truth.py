"""exp20 -- does the exp19 margin survive a truth engine shaped unlike our estimator?

Pre-registered in `experiments/PREREGISTRATION_exp20.md`. Read that first; it
fixes the arms, the truth targets, the metrics and the losing conditions, and it
records the one amendment made after a two-seed pilot.

**The honest framing, and it belongs at the top rather than in a footnote.**

Our headline result -- exp19 Leg C, rate MAE 0.0037 s/lap against 0.0158 for the
closest published model -- is measured on `tyremind.data.synthetic`, a generator
we wrote whose lap time is a latent linear-Gaussian tyre term plus additive fuel,
track and traffic terms. That is the same algebraic shape our estimator assumes.
A generator that shares the estimator's assumptions cannot falsify it, and "you
graded your own homework" is a fair objection to it.

The scoped answer was to capture true wear from a commercial F1 game's UDP
telemetry. That needs the game running and is not available here.

So this runs the ladder against `tyremind.data.physics_truth` instead: a second
truth engine in which the tyre's time loss is **not a parameter**. Lap time is the
line integral of a quasi-steady-state speed profile; grip enters through the
cornering limit; grip falls nonlinearly with accumulated wear; wear is the
integral of the repository's own energy-dissipation wear law over the lap's
frictional power and estimated tread temperature. The degradation rate is
obtained by differencing the simulated lap times. Nobody sets it.

**We wrote that generator too.** This is not the same as validating against a
third party's tyre model, and no number here may be quoted as though it were.
What it removes is the *functional-form* advantage, not all self-reference.

    If our advantage shrinks or vanishes here, THAT IS THE RESULT and it is
    reported plainly: it would mean the exp19 margin was partly an artefact of a
    generator shaped like our estimator, which is exactly what we need to know
    before a judge asks. If the advantage holds, the claim becomes considerably
    stronger and the exp19 section 6.1 caveat can be narrowed.

Both arms run at matched seed counts so the two generators are read side by side,
and every model is scored against **three** definitions of "the true degradation
rate" -- a fresh set, the stint average, and the best linear fit. On a generator
whose degradation is nonlinear those are three different numbers, and which model
wins can depend on which one is meant. Reporting all three is the only honest way
to present that.

    python experiments/exp20_independent_truth.py
    python experiments/exp20_independent_truth.py --seeds 4 --arms physics

Writes experiments/results/exp20_independent_truth.json.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tyremind.data.physics_truth import FRESH_AGE_LAPS, PhysicsSessionConfig
from tyremind.data.physics_truth import generate_session as generate_physics_session
from tyremind.data.synthetic import SessionConfig
from tyremind.data.synthetic import generate_session as generate_synthetic_session
from tyremind.models.baselines import DegradationModel
from tyremind.models.evaluation import score_rate_recovery
from tyremind.models.literature import extended_ladder

RESULTS = Path(__file__).parent / "results" / "exp20_independent_truth.json"

#: Identical across both arms and every model, so every comparison is paired on
#: the generated session rather than on an arm average.
BASE_SEED = 20260901

OUR_MODEL = "TyreMind state-space"

#: The three definitions of "the true degradation rate", and what each one means.
#: On `synthetic` they nearly coincide, because degradation there is a declared
#: constant with a mild cliff bolted on. On `physics_truth` they do not, and the
#: gap between them is a property of the tyre rather than of the experiment.
TARGETS = {
    "fresh": (
        f"mean instantaneous rate over the first {FRESH_AGE_LAPS:.0f} laps of tyre "
        "age -- the analogue of a declared baseline rate, and what a model whose "
        "parameter means 'baseline' should be scored on"
    ),
    "mean_instantaneous": (
        "lap-weighted mean instantaneous rate over the laps actually run -- what "
        "the tyre cost per lap on average"
    ),
    "ols_slope": (
        "least-squares slope of the true cumulative tyre loss on tyre age -- the "
        "best straight line through the truth, and what a model reporting a "
        "linear slope should be scored on"
    ),
}

#: Pre-registered as primary in section 5 before any result existed.
PRIMARY_TARGET = "mean_instantaneous"

#: Pre-registered in section 5. Declared here so a shrink cannot be presented as
#: a win after the fact.
EXPECTATIONS = {
    "margin_shrinks_on_physics": True,
    "absolute_mae_worse_on_physics_for_every_model": True,
    "coverage_degrades_on_physics": True,
}


def rate_bearing_models() -> list[DegradationModel]:
    """The rungs that have a degradation parameter at all.

    Detected by introspection rather than by name, so a rung that later gains a
    rate is picked up instead of being silently left out. ARIMA, LightGBM and the
    MLP never override `compound_rates`, which is the exp19 finding and is
    reported rather than hidden.
    """
    return [
        m
        for m in extended_ladder()
        if type(m).compound_rates is not DegradationModel.compound_rates
    ]


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope, or NaN if the fit is not possible.

    Guarded: `np.polyfit` raises `LinAlgError` on degenerate input, and an
    unguarded call has killed two runs in this repository.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3 or np.ptp(x[good]) <= 0:
        return float("nan")
    try:
        return float(np.polyfit(x[good], y[good], 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")


def targets_from_lap_truth(lap_truth: pd.DataFrame) -> dict[str, dict[str, float]]:
    """The three truth targets, computed identically for both arms.

    Both generators publish a per-lap `true_tyre` (cumulative tyre cost in
    seconds) and `true_rate` (its derivative in tyre age), so all three targets
    come out of the same four lines of arithmetic on either arm. That matters:
    a target computed one way on one generator and another way on the other would
    make the side-by-side comparison meaningless.
    """
    frame = lap_truth.copy()
    frame["compound"] = frame["compound"].astype(str)
    rate = frame["true_rate"].to_numpy(dtype=float)
    scored = frame[np.isfinite(rate)]
    fresh = scored[scored["tyre_age"].astype(float) < FRESH_AGE_LAPS]

    return {
        "fresh": {
            str(c): float(b["true_rate"].mean()) for c, b in fresh.groupby("compound")
        },
        "mean_instantaneous": {
            str(c): float(b["true_rate"].mean()) for c, b in scored.groupby("compound")
        },
        "ols_slope": {
            str(c): _ols_slope(b["tyre_age"].to_numpy(), b["true_tyre"].to_numpy())
            for c, b in frame.groupby("compound")
        },
    }


def _rate_curvature(curve: pd.DataFrame, compound: str) -> float:
    """Ratio of the late-stint emergent rate to the early-stint one.

    Greater than one is the whole reason a linear-in-age model is mis-specified
    on the physics arm, and the number says by how much.
    """
    block = curve[curve["compound"].astype(str) == compound].sort_values("tyre_age")
    if len(block) < 6:
        return float("nan")
    third = max(len(block) // 3, 1)
    early = float(block["true_rate"].iloc[:third].mean())
    late = float(block["true_rate"].iloc[-third:].mean())
    return float(late / early) if np.isfinite(early) and early > 0 else float("nan")


def physics_arm(seed: int) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict]:
    """One physics-derived session: lap table, the three truth targets, notes.

    Every one of the three targets is a derivative or a fit of *simulated lap
    times*. None of them is a number the generator was handed.
    """
    session = generate_physics_session(replace(PhysicsSessionConfig(), seed=seed))
    truth = session.truth
    targets = targets_from_lap_truth(truth.lap_truth)
    notes = {
        "emergent_fuel_slope": float(truth.fuel_slope),
        "emergent_traffic_coefficient": float(truth.traffic_coefficient),
        "emergent_track_evolution_total": float(
            -truth.track_evolution["true_track_effect"].min()
        ),
        "driver_pace_sd": float(np.std(list(truth.driver_pace.values()))),
        "additive_decomposition_residual_sd": float(
            truth.lap_truth["decomposition_residual"].std()
        ),
        "max_true_tyre_seconds": float(truth.lap_truth["true_tyre"].max()),
        "mean_lap_time": float(session.lap_table["lap_time"].mean()),
        "n_laps": int(len(session.lap_table)),
        "rate_curvature": {
            str(c): _rate_curvature(truth.compound_rate_curve, str(c))
            for c in truth.compound_rates
        },
    }
    return session.lap_table, targets, notes


def synthetic_arm(seed: int) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict]:
    """One `tyremind.data.synthetic` session -- the exp19 Leg C setting.

    The three targets are computed from this generator's own `lap_truth` by the
    same function the physics arm uses. The generator's *declared* baseline rate
    -- the number exp19 Leg C scored against -- is recorded in the notes so the
    link to the published result stays auditable, and because the `fresh` target
    should reproduce it to within rounding.
    """
    session = generate_synthetic_session(replace(SessionConfig(), seed=seed))
    truth = session.truth
    targets = targets_from_lap_truth(truth.lap_truth)
    notes = {
        "declared_compound_rates": {str(k): float(v) for k, v in truth.compound_rates.items()},
        "declared_fuel_slope": float(truth.fuel_slope),
        "declared_traffic_coefficient": float(truth.traffic_coefficient),
        "mean_lap_time": float(session.lap_table["lap_time"].mean()),
        "n_laps": int(len(session.lap_table)),
    }
    return session.lap_table, targets, notes


ARMS = {"physics": physics_arm, "synthetic": synthetic_arm}


def score_one_session(
    lap_table: pd.DataFrame, targets: dict[str, dict[str, float]]
) -> list[dict]:
    """Fit every rate-bearing rung once and score it against all three targets.

    `evaluation.score_rate_recovery` is called for the **primary** target, so the
    estimates, their standard deviations, the error and the coverage flag are
    produced by exactly the function exp19 used. The other two targets reuse the
    estimate that call returned rather than refitting: an error is
    `estimate - truth` and coverage is `|estimate - truth| <= 1.96 sd` under any
    target, so refitting would repeat the arithmetic and, because models carry
    state, risk making the second score depend on the first.
    """
    frame = score_rate_recovery(rate_bearing_models(), lap_table, targets[PRIMARY_TARGET])

    rows: list[dict] = []
    for _, row in frame.iterrows():
        compound = row.get("compound")
        if not isinstance(compound, str):
            rows.append(
                {
                    "model": str(row["model"]),
                    "compound": None,
                    "failed": row.get("failed"),
                    "note": row.get("note"),
                }
            )
            continue

        estimate = row.get("estimate")
        estimate = float(estimate) if estimate is not None else float("nan")
        sd = row.get("estimate_sd")
        sd = float(sd) if sd is not None else float("nan")

        out = {
            "model": str(row["model"]),
            "compound": compound,
            "estimate": estimate if np.isfinite(estimate) else None,
            "estimate_sd": sd if np.isfinite(sd) else None,
            "failed": None,
        }
        for target, truths in targets.items():
            truth = float(truths.get(compound, np.nan))
            error = estimate - truth
            out[f"true_{target}"] = truth if np.isfinite(truth) else None
            out[f"error_{target}"] = float(error) if np.isfinite(error) else float("nan")
            out[f"abs_error_{target}"] = float(abs(error)) if np.isfinite(error) else float("nan")
            out[f"covered_{target}"] = bool(
                np.isfinite(error) and np.isfinite(sd) and sd > 0 and abs(error) <= 1.96 * sd
            )
        rows.append(out)
    return rows


def summarise(block: pd.DataFrame, target: str) -> pd.DataFrame:
    """Per-model aggregate within one arm and one target, best rate MAE first."""
    error_col, abs_col, cover_col = f"error_{target}", f"abs_error_{target}", f"covered_{target}"
    out = []
    for model, rows in block.groupby("model"):
        if abs_col not in rows.columns:
            continue
        scored = rows[np.isfinite(rows[abs_col].to_numpy(dtype=float))]
        if scored.empty:
            out.append({"model": model, "n": 0, "rate_mae": float("nan")})
            continue
        errors = scored[error_col].to_numpy(dtype=float)
        absolute = np.abs(errors)
        n = int(absolute.size)
        out.append(
            {
                "model": model,
                "n": n,
                "rate_mae": float(absolute.mean()),
                "rate_mae_se": (
                    float(absolute.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
                ),
                "rate_bias": float(errors.mean()),
                "coverage_95": float(scored[cover_col].mean()),
                "n_failed": int(rows["failed"].notna().sum()),
            }
        )
    return pd.DataFrame(out).sort_values("rate_mae", na_position="last").reset_index(drop=True)


def paired_margin(block: pd.DataFrame, better: str, worse: str, target: str) -> dict:
    """Paired difference in absolute rate error, `MAE(worse) - MAE(better)`.

    Paired on `(seed, compound)`: the same generated session, the same compound.
    Between-seed variance is far larger than the between-model difference, so an
    unpaired comparison would drown a real effect -- exp29 recorded a four-seed
    pilot flipping a ranking that eight seeds reversed.

    Positive means `better` really is better. `decisive` is the pre-registered
    tie rule: a win needs a margin larger than one standard error.
    """
    column = f"abs_error_{target}"
    left = block[block["model"] == better].set_index(["seed", "compound"])[column]
    right = block[block["model"] == worse].set_index(["seed", "compound"])[column]
    shared = left.index.intersection(right.index)
    if len(shared) < 2:
        return {"better": better, "worse": worse, "n_paired": int(len(shared))}
    diff = (right.loc[shared] - left.loc[shared]).to_numpy(dtype=float)
    diff = diff[np.isfinite(diff)]
    n = int(diff.size)
    if n < 2:
        return {"better": better, "worse": worse, "n_paired": n}
    se = float(diff.std(ddof=1) / np.sqrt(n))
    return {
        "better": better,
        "worse": worse,
        "n_paired": n,
        "margin": float(diff.mean()),
        "margin_se": se,
        "decisive": bool(np.isfinite(se) and diff.mean() > se),
    }


def read_arm(block: pd.DataFrame, target: str) -> dict:
    """Rank one arm under one target and apply the pre-registered tie rule."""
    table = summarise(block[block["model"].notna()], target)
    if table.empty or not np.isfinite(table.loc[0, "rate_mae"]):
        return {"target": target, "failed": "nothing scored"}

    best = str(table.loc[0, "model"])
    runner_up = str(table.loc[1, "model"]) if len(table) > 1 else best
    margin = paired_margin(block, best, runner_up, target) if best != runner_up else {}
    winner = best if bool(margin.get("decisive", False)) else "TIE"

    ours = table.index[table["model"] == OUR_MODEL]
    our_rank = int(ours[0]) + 1 if len(ours) else None
    our_mae = float(table.loc[ours[0], "rate_mae"]) if len(ours) else float("nan")

    # The margin that matters for the headline: ours over the best rung that is
    # not ours. Reported whichever way it points.
    others = [m for m in table["model"].tolist() if m != OUR_MODEL]
    best_other = others[0] if others else None
    our_margin = paired_margin(block, OUR_MODEL, best_other, target) if best_other else {}

    return {
        "target": target,
        "winner": winner,
        "best_model": best,
        "best_mae": float(table.loc[0, "rate_mae"]),
        "runner_up": runner_up,
        "paired_margin_best_over_runner_up": margin,
        "tyremind_mae": our_mae,
        "tyremind_rank": our_rank,
        "best_non_tyremind": best_other,
        "paired_margin_tyremind_over_best_other": our_margin,
        "table": table.to_dict(orient="records"),
    }


def run_arm(name: str, seeds: list[int]) -> tuple[list[dict], list[dict]]:
    """Every seed of one arm. The same session is handed to every model."""
    build = ARMS[name]
    rows: list[dict] = []
    notes: list[dict] = []
    for seed in seeds:
        started = time.perf_counter()
        try:
            lap_table, targets, note = build(seed)
        except Exception as exc:  # noqa: BLE001 - a failed seed is reported, not fatal
            rows.append({"arm": name, "seed": seed, "model": None, "failed": f"generator: {exc}"})
            print(f"  {name:<10} seed {seed}  GENERATOR FAILED: {exc}", flush=True)
            continue
        note.update({"arm": name, "seed": seed, "targets": targets})
        notes.append(note)
        for row in score_one_session(lap_table, targets):
            rows.append({"arm": name, "seed": seed, **row})
        primary = targets[PRIMARY_TARGET]
        print(
            f"  {name:<10} seed {seed}  {len(lap_table):>4} laps  truth "
            + ", ".join(f"{k}={v:.4f}" for k, v in sorted(primary.items()))
            + f"  {time.perf_counter() - started:>6.1f}s",
            flush=True,
        )
    return rows, notes


def print_arm(arm: str, reading: dict, n_seeds: int) -> None:
    """One arm, one target, as a table."""
    print("\n" + "=" * 100)
    print(
        f"{arm.upper()} ARM -- target '{reading['target']}' -- "
        f"{n_seeds} seeds x 3 compounds, paired on the session"
    )
    print("=" * 100)
    print(f"{'model':<34}{'rate MAE':>11}{'+-SE':>9}{'bias':>10}{'cover95':>9}{'n':>5}")
    for row in reading["table"]:
        print(
            f"{str(row['model'])[:32]:<34}{row.get('rate_mae', float('nan')):>11.4f}"
            f"{row.get('rate_mae_se', float('nan')):>9.4f}"
            f"{row.get('rate_bias', float('nan')):>10.4f}"
            f"{row.get('coverage_95', float('nan')):>9.2f}{row.get('n', 0):>5}"
        )
    m = reading["paired_margin_best_over_runner_up"]
    print(
        f"  winner: {reading['winner']}  ({reading['best_model']} over "
        f"{reading['runner_up']} by {m.get('margin', float('nan')):+.4f} "
        f"+- {m.get('margin_se', float('nan')):.4f} s/lap, paired n={m.get('n_paired', 0)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--arms", nargs="*", default=["physics", "synthetic"])
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    unknown = [a for a in args.arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s): {unknown}. Known: {sorted(ARMS)}")

    seeds = [BASE_SEED + i for i in range(args.seeds)]
    no_rate_models = sorted(
        {m.name for m in extended_ladder()} - {m.name for m in rate_bearing_models()}
    )

    all_rows: list[dict] = []
    all_notes: list[dict] = []
    readings: dict[str, dict[str, dict]] = {}

    for arm in args.arms:
        print(f"\n{arm} arm -- {args.seeds} seeds")
        rows, notes = run_arm(arm, seeds)
        all_rows.extend(rows)
        all_notes.extend(notes)
        block = pd.DataFrame(rows)
        readings[arm] = {t: read_arm(block, t) for t in TARGETS}

    for arm in args.arms:
        for target in TARGETS:
            reading = readings[arm][target]
            if reading.get("failed"):
                print(f"\n{arm} / {target}: FAILED -- {reading['failed']}")
            else:
                print_arm(arm, reading, args.seeds)

    # --- the pre-registered reading -----------------------------------------
    verdict: dict = {"primary_target": PRIMARY_TARGET}
    physics = readings.get("physics", {})
    synthetic = readings.get("synthetic", {})

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)

    primary = physics.get(PRIMARY_TARGET)
    if primary and not primary.get("failed"):
        decisive = primary["winner"] == OUR_MODEL
        our_margin = primary["paired_margin_tyremind_over_best_other"]
        verdict.update(
            {
                "tyremind_first_on_physics": primary["tyremind_rank"] == 1,
                "tyremind_decisive_on_physics": decisive,
                "physics_margin": our_margin.get("margin"),
                "physics_margin_se": our_margin.get("margin_se"),
            }
        )
        if decisive:
            print(
                f"On the pre-registered primary target ('{PRIMARY_TARGET}') the exp19 Leg C "
                "ranking SURVIVES a change of functional form. On a generator whose lap time "
                "is a line integral rather than a sum of terms, and whose degradation rate is "
                f"derived rather than declared, TyreMind is first at {primary['tyremind_mae']:.4f} "
                f"s/lap, ahead of {primary['best_non_tyremind']} by "
                f"{our_margin.get('margin', float('nan')):+.4f} "
                f"+- {our_margin.get('margin_se', float('nan')):.4f} s/lap paired."
            )
        elif primary["tyremind_rank"] == 1:
            print(
                f"On the primary target TyreMind is first at {primary['tyremind_mae']:.4f} s/lap "
                "but by less than one standard error of the paired difference "
                f"({our_margin.get('margin', float('nan')):+.4f} "
                f"+- {our_margin.get('margin_se', float('nan')):.4f}). Under the pre-registered "
                "tie rule this is COMPARABLE, not better."
            )
        else:
            print(
                f"THE exp19 LEG C RANKING DOES NOT SURVIVE. On the physics arm's primary target "
                f"{primary['best_model']} leads at {primary['best_mae']:.4f} s/lap against our "
                f"{primary['tyremind_mae']:.4f}; we rank {primary['tyremind_rank']}. The exp19 "
                "margin was therefore substantially an artefact of a generator shaped like our "
                "estimator, and exp19 Leg C must be presented with that attached."
            )

    # Does the ranking depend on what "the degradation rate" means?
    for arm, arm_readings in readings.items():
        winners = {
            t: (r.get("best_model"), r.get("winner"))
            for t, r in arm_readings.items()
            if not r.get("failed")
        }
        distinct = {best for best, _ in winners.values()}
        verdict[f"{arm}_best_by_target"] = {t: v[0] for t, v in winners.items()}
        verdict[f"{arm}_ranking_depends_on_target"] = len(distinct) > 1
        print()
        print(f"Target sensitivity on the {arm} arm -- who leads under each definition:")
        for target, (best, winner) in winners.items():
            ours = arm_readings[target]
            print(
                f"  {target:<19} leads: {str(best)[:32]:<34} "
                f"(ours {ours['tyremind_mae']:.4f}, rank {ours['tyremind_rank']}, "
                f"call: {winner})"
            )
        if len(distinct) > 1:
            print(
                "  The ranking DEPENDS ON THE TARGET. There is no single 'true degradation "
                "rate' when degradation is nonlinear, so no one of these rankings may be "
                "quoted on its own. This is a finding, not a caveat."
            )

    # How much of the published margin survives?
    p = (physics.get(PRIMARY_TARGET) or {}).get("paired_margin_tyremind_over_best_other", {})
    s = (synthetic.get(PRIMARY_TARGET) or {}).get("paired_margin_tyremind_over_best_other", {})
    if p.get("margin") is not None and s.get("margin"):
        ratio = float(p["margin"] / s["margin"])
        verdict["synthetic_margin"] = s["margin"]
        verdict["survival_ratio"] = ratio
        print()
        print(
            f"Margin over the best non-TyreMind rung on the primary target: synthetic "
            f"{s['margin']:+.4f} s/lap, physics {p['margin']:+.4f} s/lap. "
            f"Survival ratio {ratio:.2f}."
        )
        if ratio < 1.0:
            print(
                f"  {(1 - ratio) * 100:.0f}% of the published advantage does not survive the "
                "change of functional form. That was the pre-registered expectation and it is "
                "reported as a cost, not explained away."
            )
        else:
            print(
                "  The margin did not shrink, against the pre-registered expectation that it "
                "would. That is a surprise and is flagged as one rather than claimed. The most "
                "likely reason is not that our estimator got better: it is that the physics "
                "arm's confounders are harder for every rung, so the spread between rungs "
                "widens. Read the absolute MAEs, not only the margin."
            )

    # Pre-registered expectation: every rung should be worse on the physics arm.
    if physics.get(PRIMARY_TARGET) and synthetic.get(PRIMARY_TARGET):
        synth_mae = {
            r["model"]: r.get("rate_mae") for r in synthetic[PRIMARY_TARGET]["table"]
        }
        improved = []
        for row in physics[PRIMARY_TARGET]["table"]:
            here, there = row.get("rate_mae"), synth_mae.get(row["model"])
            if (
                there is not None
                and np.isfinite(here or np.nan)
                and np.isfinite(there)
                and here < there
            ):
                improved.append(f"{row['model']} ({here:.4f} vs {there:.4f})")
        verdict["models_better_on_physics_than_synthetic"] = improved
        print()
        print(
            "Pre-registered expectation -- every rung should do worse on the physics arm, "
            "because every rung is additive and the physics arm is not."
        )
        if improved:
            print("  NOT MET for: " + "; ".join(improved) + ".")
            print(
                "  A rung doing better on the harder generator means its synthetic score was "
                "being held back by something specific to that generator rather than by its "
                "own structure. Recorded rather than smoothed over."
            )
        else:
            print("  Met: every rung's rate MAE is worse on the physics arm.")

    print()
    print(
        "STANDING CAVEAT, which travels with every number above: we wrote the physics "
        "generator too. It removes the functional-form advantage -- the estimator's algebra "
        "is no longer the generator's algebra -- and it removes nothing else. This is not "
        "validation against a third party's tyre model, and exp20 as originally scoped (a "
        "commercial simulator's telemetry) remains unbuilt."
    )
    print(f"Rungs with no degradation parameter at all: {', '.join(no_rate_models)}.")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "experiment": "exp20_independent_truth",
                "preregistration": "experiments/PREREGISTRATION_exp20.md",
                "generated_at": datetime.now(UTC).isoformat(),
                "n_seeds": args.seeds,
                "base_seed": BASE_SEED,
                "arms": args.arms,
                "targets": TARGETS,
                "primary_target": PRIMARY_TARGET,
                "pre_registered_expectations": EXPECTATIONS,
                "models_without_degradation_parameter": no_rate_models,
                "honest_framing": (
                    "The physics arm's generator was written by us. It removes the "
                    "functional-form advantage the synthetic arm gives our estimator; it "
                    "does not remove self-reference and is not third-party validation."
                ),
                "verdict": verdict,
                "readings": readings,
                "generator_notes": all_notes,
                "rows": all_rows,
            },
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
