"""exp27 -- score every model's Friday prediction against its OWN Sunday fit.

Pre-registered in `experiments/PREREGISTRATION_exp27.md`. Read that first; it
fixes the metrics, the directions and the losing conditions.

exp03 asks whether a practice degradation curve predicts the race, and answers it
by scoring every method -- ours and the naive comparator alike -- against *our*
race fit. That makes our own model the yardstick, so a competitor is charged both
for its practice estimate being wrong and for its race estimate differing from
ours, while we are charged once. It is the weakest evidence in the project.

Here each model is scored against its own race-derived estimate:

    error(model, event, compound) = rate_practice(model) - rate_race(model)

which asks every method the same question on equal terms. The old circular error
is computed alongside it, so the size of the distortion is measured rather than
asserted.

Self-consistency is gameable -- a model returning a constant scores a perfect
zero -- so the headline is a **skill score against each model's own climatology**:

    skill = 1 - MAE_self / MAE_constant

where MAE_constant is what that model would have scored by ignoring practice and
predicting the mean of its own race rates. A constant model scores exactly 0 and
cannot win.

    python experiments/exp27_decircularised_practice_to_race.py
    python experiments/exp27_decircularised_practice_to_race.py --limit 6

Writes experiments/results/exp27_decircularised_practice_to_race.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tyremind.data.corpus import CORPUS_DIR, sessions
from tyremind.models.literature import extended_ladder

RESULTS = Path(__file__).parent / "results" / "exp27_decircularised_practice_to_race.json"
CONDITIONS = Path("data/reference/session_conditions.json")

#: Matches `validate_practice_to_race`'s `min_laps_per_compound`. A compound with
#: fewer laps than this on either side is recorded as skipped, never as an error.
MIN_LAPS_PER_COMPOUND = 8

#: The reference model whose race fit exp03 used as everyone's yardstick. Kept
#: only so the circular metric can be reproduced and compared against.
REFERENCE_MODEL = "TyreMind state-space"


def wet_sessions() -> dict[str, float]:
    """Session id to wet-lap fraction, for every session that is not dry.

    A session that ran most of its laps on wet rubber is not a dry-degradation
    session, and a rate fitted from the dry remainder describes a drying track.
    The rule is applied to both sides of every comparison.
    """
    if not CONDITIONS.exists():
        return {}
    return {
        entry["session_id"]: float(entry["wet_lap_fraction"])
        for entry in json.loads(CONDITIONS.read_text())
        if not entry["dry_session"]
    }


def paired_events(practice: str, limit: int) -> list[tuple]:
    """Events holding both a practice session and a race, most recent first.

    `min_laps=0` on both sides: the corpus-wide 200-lap floor exists to support
    chronological folds, which this experiment does not use, and it would remove
    most practice sessions for no methodological reason. The per-compound
    threshold is what protects against thin evidence here.
    """
    races = {(s.year, s.event): s for s in sessions(CORPUS_DIR, session_type="R", min_laps=0)}
    practices = {
        (s.year, s.event): s
        for s in sessions(CORPUS_DIR, session_type=practice, min_laps=0)
    }
    keys = [k for k in races if k in practices]
    # sessions() already returns season-descending, round-ascending; preserve it.
    order = {(s.year, s.event): i for i, s in enumerate(races.values())}
    keys.sort(key=lambda k: order[k])
    if limit:
        keys = keys[:limit]
    return [(year, event, practices[(year, event)], races[(year, event)]) for year, event in keys]


def fit_rates(lap_table: pd.DataFrame) -> tuple[dict[str, dict], dict[str, str]]:
    """Fit all nine rungs and collect each one's per-compound degradation rate.

    Returns:
        (rates, failures). `rates[model]` maps compound to (mean, sd) and is
        empty for the three models that have no degradation parameter at all --
        which is a reported finding, not a failure.
    """
    rates: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for model in extended_ladder():
        try:
            model.fit(lap_table)
            rates[model.name] = {
                str(c): (float(m), float(s)) for c, (m, s) in model.compound_rates().items()
            }
        except Exception as exc:  # noqa: BLE001 - one rung failing must not stop the sweep
            failures[model.name] = f"{type(exc).__name__}: {exc}"
            rates[model.name] = {}
    return rates, failures


def compare_event(
    year: int,
    event: str,
    practice_table: pd.DataFrame,
    race_table: pd.DataFrame,
) -> dict:
    """One event: every model's practice rate against its own race rate."""
    practice_rates, practice_failures = fit_rates(practice_table)
    race_rates, race_failures = fit_rates(race_table)

    practice_counts = practice_table["compound"].astype(str).value_counts().to_dict()
    race_counts = race_table["compound"].astype(str).value_counts().to_dict()

    reference = race_rates.get(REFERENCE_MODEL, {})

    rows: list[dict] = []
    skipped: dict[str, str] = {}

    for model_name in practice_rates:
        p_rates, r_rates = practice_rates[model_name], race_rates[model_name]
        if not p_rates or not r_rates:
            continue
        for compound in sorted(set(p_rates) & set(r_rates)):
            p_laps = int(practice_counts.get(compound, 0))
            r_laps = int(race_counts.get(compound, 0))
            if p_laps < MIN_LAPS_PER_COMPOUND or r_laps < MIN_LAPS_PER_COMPOUND:
                skipped[f"{model_name}|{compound}"] = (
                    f"too few laps ({p_laps} practice, {r_laps} race; "
                    f"need {MIN_LAPS_PER_COMPOUND})"
                )
                continue
            p_mean, p_sd = p_rates[compound]
            r_mean, r_sd = r_rates[compound]
            if not (np.isfinite(p_mean) and np.isfinite(r_mean)):
                skipped[f"{model_name}|{compound}"] = "non-finite rate"
                continue

            # The old exp03 yardstick: everyone scored against TyreMind's race
            # fit. Kept only to measure how much it distorted the comparison.
            circular = reference.get(compound)
            circular_error = (
                float(p_mean - circular[0]) if circular and np.isfinite(circular[0]) else float("nan")
            )

            combined_sd = float(np.hypot(p_sd if np.isfinite(p_sd) else 0.0,
                                         r_sd if np.isfinite(r_sd) else 0.0))
            rows.append({
                "year": year,
                "event": event,
                "model": model_name,
                "compound": compound,
                "practice_rate": float(p_mean),
                "practice_sd": float(p_sd),
                "race_rate": float(r_mean),
                "race_sd": float(r_sd),
                "self_error": float(p_mean - r_mean),
                "circular_error": circular_error,
                "combined_sd": combined_sd,
                "covered_95": bool(abs(p_mean - r_mean) <= 1.96 * combined_sd) if combined_sd > 0 else False,
                "practice_laps": p_laps,
                "race_laps": r_laps,
            })

    return {
        "rows": rows,
        "skipped": skipped,
        "practice_failures": practice_failures,
        "race_failures": race_failures,
        "no_degradation_parameter": sorted(
            name for name, r in race_rates.items() if not r and name not in race_failures
        ),
    }


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-model aggregate, with the skill score as the primary column.

    The skill denominator is each model's error had it ignored practice entirely
    and predicted the mean of *its own* race rates. That is the comparison that
    makes self-consistency non-gameable: a model returning a constant has a zero
    numerator and a zero denominator's worth of information, and scores 0.
    """
    out = []
    for model, block in frame.groupby("model"):
        errors = block["self_error"].to_numpy(dtype=float)
        race = block["race_rate"].to_numpy(dtype=float)
        errors = errors[np.isfinite(errors)]
        if errors.size == 0:
            continue

        mae = float(np.abs(errors).mean())
        # Climatology: this model's own mean race rate, used as a rate forecast.
        constant_errors = np.abs(race - race.mean())
        mae_constant = float(constant_errors.mean())
        skill = float(1.0 - mae / mae_constant) if mae_constant > 0 else float("nan")

        # Standard error of the skill score, by the delta method on the ratio of
        # two means. Reported so "better" can be told from "0.02 apart on 60
        # comparisons".
        n = errors.size
        se_mae = float(np.abs(errors).std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        se_const = float(constant_errors.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        skill_se = (
            float(abs(mae / mae_constant) * np.hypot(se_mae / mae, se_const / mae_constant))
            if mae > 0 and mae_constant > 0 and np.isfinite(se_mae) and np.isfinite(se_const)
            else float("nan")
        )

        circular = block["circular_error"].to_numpy(dtype=float)
        circular = circular[np.isfinite(circular)]

        practice = block["practice_rate"].to_numpy(dtype=float)
        if n > 2 and np.ptp(practice) > 0 and np.ptp(race) > 0:
            spearman = float(pd.Series(practice).corr(pd.Series(race), method="spearman"))
        else:
            spearman = float("nan")

        out.append({
            "model": model,
            "n": int(n),
            "skill_vs_own_climatology": skill,
            "skill_se": skill_se,
            "self_mae": mae,
            "self_mae_se": se_mae,
            "self_bias": float(errors.mean()),
            "coverage_95": float(block["covered_95"].mean()),
            "race_rate_sd": float(race.std(ddof=1)) if n > 1 else float("nan"),
            "climatology_mae": mae_constant,
            "spearman_practice_vs_race": spearman,
            "circular_mae": float(np.abs(circular).mean()) if circular.size else float("nan"),
            "circular_n": int(circular.size),
        })
    return pd.DataFrame(out).sort_values("skill_vs_own_climatology", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practice", default="FP2")
    parser.add_argument("--limit", type=int, default=0, help="0 means every paired event")
    parser.add_argument("--allow-wet", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    wet = {} if args.allow_wet else wet_sessions()
    targets = paired_events(args.practice, args.limit)
    if not targets:
        raise SystemExit("no events hold both a practice session and a race")

    excluded: dict[str, str] = {}
    kept = []
    for year, event, practice, race in targets:
        bad = [s for s in (practice, race) if s.session_id in wet]
        if bad:
            excluded[f"{year} {event}"] = "; ".join(
                f"{s.session} ran {wet[s.session_id]:.0%} of laps on wet rubber" for s in bad
            )
            print(f"  {year} {event[:34]:<34} EXCLUDED -- {excluded[f'{year} {event}']}")
            continue
        kept.append((year, event, practice, race))

    all_rows: list[dict] = []
    skipped: dict[str, str] = {}
    failures: dict[str, str] = {}
    no_rate_models: set[str] = set()

    for year, event, practice, race in kept:
        started = time.perf_counter()
        try:
            result = compare_event(year, event, practice.load(), race.load())
        except Exception as exc:  # noqa: BLE001
            failures[f"{year} {event}"] = f"{type(exc).__name__}: {exc}"
            print(f"  {year} {event[:34]:<34} SKIPPED -- {type(exc).__name__}: {exc}", flush=True)
            continue
        all_rows.extend(result["rows"])
        for key, reason in result["skipped"].items():
            skipped[f"{year} {event}|{key}"] = reason
        for name, reason in result["practice_failures"].items():
            failures[f"{year} {event}|{args.practice}|{name}"] = reason
        for name, reason in result["race_failures"].items():
            failures[f"{year} {event}|R|{name}"] = reason
        no_rate_models.update(result["no_degradation_parameter"])
        print(
            f"  {year} {event[:34]:<34} {len(result['rows']):>3} comparisons  "
            f"{time.perf_counter() - started:>6.1f}s",
            flush=True,
        )

    if not all_rows:
        raise SystemExit(f"no comparison could be scored. Failures: {failures}")

    frame = pd.DataFrame(all_rows)
    table = summarise(frame)

    # An event is a (season, name) pair. Counting names alone folds 2023 Bahrain
    # into 2024 Bahrain and understates the corpus by more than half.
    n_events = int(frame.drop_duplicates(["year", "event"]).shape[0])

    print("\n" + "=" * 108)
    print(f"DE-CIRCULARISED PRACTICE ({args.practice}) -> RACE, "
          f"{n_events} events, {len(frame)} comparisons")
    print("Each model scored against ITS OWN race fit. Skill is measured against "
          "that model's own climatology,")
    print("so a model returning a constant scores 0 and cannot win.")
    print("=" * 108)
    print(f"{'model':<34}{'skill':>8}{'+-SE':>7}{'selfMAE':>9}{'bias':>9}"
          f"{'cover':>7}{'raceSD':>8}{'rho':>7}{'circMAE':>9}{'n':>5}")
    for _, row in table.iterrows():
        print(
            f"{row['model']:<34}{row['skill_vs_own_climatology']:>8.3f}{row['skill_se']:>7.3f}"
            f"{row['self_mae']:>9.4f}{row['self_bias']:>+9.4f}{row['coverage_95']:>7.0%}"
            f"{row['race_rate_sd']:>8.4f}{row['spearman_practice_vs_race']:>7.2f}"
            f"{row['circular_mae']:>9.4f}{int(row['n']):>5}"
        )
    print("=" * 108)

    if no_rate_models:
        print("No degradation parameter, so not scored (the exp19 finding, unchanged): "
              + ", ".join(sorted(no_rate_models)))

    # The pre-registered readings, printed so the result cannot be reported
    # selectively later.
    best = table.iloc[0]
    ours = table[table["model"] == REFERENCE_MODEL]
    verdict: dict = {}
    if not ours.empty:
        ours_row = ours.iloc[0]
        rank_new = int(table.index.get_indexer([ours_row.name])[0]) + 1
        by_circular = table.sort_values("circular_mae")
        rank_old = int(by_circular.index.get_indexer([ours_row.name])[0]) + 1
        verdict = {
            "tyremind_skill": float(ours_row["skill_vs_own_climatology"]),
            "tyremind_rank_decircularised": rank_new,
            "tyremind_rank_circular_metric": rank_old,
            "best_model": str(best["model"]),
            "win_claimed": bool(
                str(best["model"]) == REFERENCE_MODEL
                and abs(float(ours_row["coverage_95"]) - 0.95) <= 0.10
            ),
            "practice_uninformative": bool(float(ours_row["skill_vs_own_climatology"]) <= 0.0),
        }
        comparable = table[
            table["skill_vs_own_climatology"]
            >= float(best["skill_vs_own_climatology"]) - float(best["skill_se"] or 0.0)
        ]
        verdict["comparable_to_best"] = [str(m) for m in comparable["model"]]

        print()
        print(f"Best on skill        : {best['model']} "
              f"({best['skill_vs_own_climatology']:.3f})")
        print(f"TyreMind skill       : {ours_row['skill_vs_own_climatology']:+.3f} "
              f"+- {ours_row['skill_se']:.3f}")
        print(f"TyreMind rank        : {rank_new} of {len(table)} de-circularised, "
              f"{rank_old} of {len(table)} under the old circular metric")
        if verdict["practice_uninformative"]:
            print("FINDING: our practice fit carries no event-specific information beyond "
                  "our own average race rate.")
        if len(verdict["comparable_to_best"]) > 1:
            print("Within one standard error of the best, so comparable rather than better: "
                  + ", ".join(verdict["comparable_to_best"]))
        if not verdict["win_claimed"]:
            print("No win claimed on the pre-registered condition "
                  "(highest skill AND coverage within 10 points of 0.95).")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "experiment": "exp27_decircularised_practice_to_race",
                "preregistration": "experiments/PREREGISTRATION_exp27.md",
                "generated_at": datetime.now(UTC).isoformat(),
                "practice_session": args.practice,
                "n_events": n_events,
                "n_comparisons": int(len(frame)),
                "min_laps_per_compound": MIN_LAPS_PER_COMPOUND,
                "reference_model_for_circular_metric": REFERENCE_MODEL,
                "excluded_not_dry": excluded,
                "models_without_degradation_parameter": sorted(no_rate_models),
                "summary": table.to_dict(orient="records"),
                "verdict": verdict,
                "skipped_compounds": skipped,
                "failures": failures,
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
