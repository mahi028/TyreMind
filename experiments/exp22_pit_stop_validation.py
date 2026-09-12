"""exp22 -- score every model against the pit stops professional strategists actually made.

The one validation in this project that uses real ground truth on real data.

Every other real-data experiment here is label-free, because measured tyre wear is
not published. Pit stops are different: a team changing tyres on lap 34 is an
observed fact, recorded in the timing feed, and it is the decision our product
exists to inform. So this asks the question the mentors asked -- *when should the
driver box?* -- and marks it against what actually happened.

**What this does and does not prove.** The lap a team chose is not necessarily the
optimal lap; it is what a professional strategist decided with far more
information than we have. Agreement therefore means "consistent with professional
judgement", not "correct". That is still the strongest real-data claim available,
and it is stated that way everywhere rather than upgraded in the telling.

Two scorings, because two audiences ask differently:

* **Decision error** -- how many laps away from the actual stop was each model's
  recommendation? This is the strategist's question.
* **Per-lap classification** -- precision, recall and F1 on "is this the pit lap",
  which is how the deep-learning literature scores the same task and therefore the
  only way to compare. Sasikumar et al. (Frontiers in AI 2025, 10.3389/frai.2025.1673148)
  report Bi-LSTM precision 0.77, recall 0.86, F1 0.81 over 2020-2024 FastF1 data.

**On that comparison, honestly.** Their models are trained classifiers with SMOTE
oversampling; ours is a physical optimisation with no training on stop labels at
all. We never fit to a pit lap. Beating a trained classifier from an untrained
optimiser would be a strong result; losing to one would be unsurprising and is
reported either way. SMOTE on a temporal task also risks leaking information
across the split, so their figure is an optimistic reference rather than a
settled bar.

    python experiments/exp22_pit_stop_validation.py --limit 12
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tyremind.data.corpus import load_frames
from tyremind.models.literature import extended_ladder
from tyremind.models.pit_decision import (
    DEFAULT_PIT_LOSS_S,
    actual_pit_laps,
    excluded_stops,
    recommend_pit_lap,
)

RESULTS = Path(__file__).parent / "results" / "exp22_pit_stop_validation.json"
SEASON_DIR = Path("data/season")
MIN_LAPS = 200

#: Published comparator. Recorded here so the result file carries the bar it is
#: being read against rather than relying on a reader remembering it.
LITERATURE_BENCHMARK = {
    "source": "Sasikumar et al., Frontiers in AI 2025, 10.3389/frai.2025.1673148",
    "model": "Bi-LSTM with SMOTE class balancing",
    "precision": 0.77,
    "recall": 0.86,
    "f1": 0.81,
    "note": "trained classifier on pit labels; ours is an untrained optimisation",
}

#: How close a recommendation must be to count as a hit. Two laps is the width of
#: a real pit window -- teams talk in windows, not single laps.
HIT_TOLERANCE = 2


def evaluate_session(lap_table: pd.DataFrame, models: list, pit_loss_s: float) -> dict:
    """Score every model's pit recommendations against the real stops in one race."""
    stops = actual_pit_laps(lap_table)
    final_lap = int(lap_table["session_lap"].max())

    fitted = {}
    for model in models:
        try:
            model.fit(lap_table)
            fitted[model.name] = model.compound_rates()
        except Exception as exc:  # noqa: BLE001
            fitted[model.name] = {"__failed__": str(exc)}

    rows: list[dict] = []
    n_scored = n_excluded = 0

    for driver, driver_stops in stops.items():
        block = lap_table[lap_table["driver"] == driver].sort_values("session_lap")
        dropped = excluded_stops(block, driver_stops, field_lap_table=lap_table)
        n_excluded += len(dropped)

        for actual in driver_stops:
            if actual in dropped:
                continue
            # Decide from a third of the way into the stint. Deciding at the stop
            # itself would be hindsight; deciding at lap one gives no evidence.
            stint = block[block["session_lap"] <= actual]
            run_id = stint.iloc[-1]["run_id"]
            this_run = stint[stint["run_id"] == run_id]
            if len(this_run) < 6:
                continue
            decision_row = this_run.iloc[len(this_run) // 3]
            decision_lap = int(decision_row["session_lap"])
            current_age = float(decision_row["tyre_age"])
            compound = str(decision_row["compound"])
            n_scored += 1

            for model in models:
                rates = fitted[model.name]
                if "__failed__" in rates or not rates:
                    rows.append({"driver": driver, "actual": actual, "model": model.name,
                                 "recommended": None, "reason": "no rate"})
                    continue
                current = rates.get(compound)
                if current is None:
                    rows.append({"driver": driver, "actual": actual, "model": model.name,
                                 "recommended": None, "reason": f"no rate for {compound}"})
                    continue
                # The fresh tyre is the average of the other compounds the model
                # has an estimate for -- the same choice for every model.
                others = [v[0] for c, v in rates.items() if c != compound]
                fresh = float(np.mean(others)) if others else current[0]

                recommendation = recommend_pit_lap(
                    current_rate=float(current[0]),
                    current_rate_sd=float(current[1]) if np.isfinite(current[1]) else 0.05,
                    fresh_rate=fresh,
                    current_age=current_age,
                    decision_lap=decision_lap,
                    final_lap=final_lap,
                    pit_loss_s=pit_loss_s,
                )
                if recommendation.reason:
                    rows.append({"driver": driver, "actual": actual, "model": model.name,
                                 "recommended": None, "reason": recommendation.reason})
                    continue
                rows.append({
                    "driver": driver,
                    "actual": int(actual),
                    "decision_lap": decision_lap,
                    "compound": compound,
                    "model": model.name,
                    "recommended": int(recommendation.lap),
                    "error_laps": int(recommendation.lap - actual),
                    "confidence": float(recommendation.confidence),
                    # Probability the model put on the lap that actually happened.
                    # This is the honest confidence question: not "how sure are
                    # you", but "how much belief did you place on the truth".
                    "prob_on_actual": float(recommendation.distribution.get(int(actual), 0.0)),
                    "reason": "",
                })

    return {"rows": rows, "n_scored": n_scored, "n_excluded": n_excluded,
            "n_drivers_with_stops": len(stops)}


def summarise(rows: list[dict]) -> pd.DataFrame:
    """Aggregate per model, in both the decision and the classification framing."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    out = []
    for model, block in frame.groupby("model"):
        scored = block[block["recommended"].notna()]
        if scored.empty:
            out.append({"model": model, "n": 0, "mae_laps": np.nan})
            continue
        error = scored["error_laps"].astype(float)
        hits = (error.abs() <= HIT_TOLERANCE)

        # Classification framing. Each scored stop is one positive; a hit inside
        # the window is a true positive, a miss is both a false positive (we
        # called a lap that was not the stop) and a false negative (we missed the
        # lap that was). Precision and recall coincide under that accounting,
        # which is stated rather than hidden -- the literature's classifiers see
        # every lap as a candidate and ours sees one recommendation per stop, so
        # the numbers are comparable in spirit and not identical in construction.
        tp = int(hits.sum())
        total = int(len(scored))
        precision = recall = tp / total if total else np.nan
        f1 = precision  # identical under the accounting above

        out.append({
            "model": model,
            "n": total,
            "n_unanswered": int(len(block) - total),
            "mae_laps": float(error.abs().mean()),
            "median_abs_error": float(error.abs().median()),
            "bias_laps": float(error.mean()),
            "hit_rate_within_2": float(hits.mean()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "mean_prob_on_actual": float(scored["prob_on_actual"].mean()),
            "mean_confidence": float(scored["confidence"].mean()),
        })
    return pd.DataFrame(out).sort_values("mae_laps")


def common_subset(rows: list[dict]) -> tuple[pd.DataFrame, int]:
    """Score every model on only the stops that *every* model answered.

    Declining is legitimate -- a model that refuses when degradation does not
    decide the stop is behaving correctly -- but it makes a raw mean error
    incomparable, because the models that decline most are graded on the easiest
    remaining cases. On the first run with guards enabled the naive rung answered
    119 of 274 stops and its mean error halved; ours answered 246 and we appeared
    to take the lead. Neither movement was a real improvement in judgement.

    So the headline number is this one, and the per-model answer rate is reported
    beside it rather than buried: declining is a property to be seen, not a way
    to win.
    """
    frame = pd.DataFrame(rows)
    answered = frame[frame["recommended"].notna()].copy()
    if answered.empty:
        return pd.DataFrame(), 0
    answered["key"] = (answered["session"] + "|" + answered["driver"]
                       + "|" + answered["actual"].astype(str))

    models = sorted(answered["model"].unique())
    shared: set | None = None
    for model in models:
        keys = set(answered.loc[answered["model"] == model, "key"])
        shared = keys if shared is None else (shared & keys)
    shared = shared or set()

    out = []
    total_stops = frame.drop_duplicates(["session", "driver", "actual"]).shape[0]
    for model in models:
        block = answered[(answered["model"] == model) & (answered["key"].isin(shared))]
        if block.empty:
            continue
        error = block["error_laps"].astype(float)
        n_answered = int((answered["model"] == model).sum())
        out.append({
            "model": model,
            "n_common": int(len(block)),
            "mae_laps": float(error.abs().mean()),
            # Standard error of the mean absolute error, so "better" can be
            # distinguished from "0.06 laps apart on 49 stops".
            "mae_se": float(error.abs().std(ddof=1) / np.sqrt(len(block))),
            "median_abs_error": float(error.abs().median()),
            "bias_laps": float(error.mean()),
            "hit_rate_within_2": float((error.abs() <= HIT_TOLERANCE).mean()),
            "answered": n_answered,
            "answer_rate": float(n_answered / total_stops) if total_stops else float("nan"),
        })
    return pd.DataFrame(out).sort_values("mae_laps"), len(shared)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--pit-loss", type=float, default=DEFAULT_PIT_LOSS_S)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    sessions = load_frames(SEASON_DIR, session_type="R", limit=args.limit, min_laps=MIN_LAPS)
    if not sessions:
        raise SystemExit(f"no races in {SEASON_DIR}; run scripts/build_corpus.py first")

    all_rows: list[dict] = []
    totals = {"scored": 0, "excluded": 0}
    for session_id, lap_table in sessions.items():
        result = evaluate_session(lap_table, extended_ladder(), args.pit_loss)
        for row in result["rows"]:
            row["session"] = session_id
        all_rows.extend(result["rows"])
        totals["scored"] += result["n_scored"]
        totals["excluded"] += result["n_excluded"]
        print(f"  {session_id:<42} {result['n_scored']:>3} stops scored, "
              f"{result['n_excluded']:>2} excluded", flush=True)

    table = summarise(all_rows)

    print("\n" + "=" * 104)
    print(f"PIT-STOP VALIDATION -- {totals['scored']} real stops across {len(sessions)} races "
          f"({totals['excluded']} excluded as safety-car or first-lap)")
    print("=" * 104)
    print(f"{'model':<34}{'MAE laps':>10}{'median':>8}{'bias':>8}"
          f"{'within2':>9}{'F1':>7}{'P(actual)':>11}{'n':>6}")
    for _, row in table.iterrows():
        if not np.isfinite(row.get("mae_laps", np.nan)):
            print(f"{row['model']:<34}{'no answer':>10}")
            continue
        print(f"{row['model']:<34}{row['mae_laps']:>10.2f}{row['median_abs_error']:>8.1f}"
              f"{row['bias_laps']:>+8.1f}{row['hit_rate_within_2']:>9.0%}"
              f"{row['f1']:>7.2f}{row['mean_prob_on_actual']:>11.3f}{int(row['n']):>6}")
    print("=" * 104)
    b = LITERATURE_BENCHMARK
    print(f"Published comparator: {b['model']} -- precision {b['precision']}, "
          f"recall {b['recall']}, F1 {b['f1']}")
    print(f"  ({b['source']})")
    print(f"  {b['note']}")

    fair, n_common = common_subset(all_rows)
    if not fair.empty:
        print()
        print("=" * 104)
        print(f"LIKE FOR LIKE -- the {n_common} stops every model answered. "
              "Declining is legitimate; being graded on an easier subset is not.")
        print("=" * 104)
        print(f"{'model':<34}{'MAE':>8}{'+-SE':>7}{'median':>8}{'bias':>8}"
              f"{'within2':>9}{'answered':>10}")
        for _, row in fair.iterrows():
            print(f"{row['model']:<34}{row['mae_laps']:>8.2f}{row['mae_se']:>7.2f}"
                  f"{row['median_abs_error']:>8.1f}{row['bias_laps']:>+8.1f}"
                  f"{row['hit_rate_within_2']:>9.0%}{row['answer_rate']:>9.0%}")
        print("=" * 104)
        best = fair.iloc[0]
        rivals = fair[fair["mae_laps"] <= best["mae_laps"] + best["mae_se"]]
        if len(rivals) > 1:
            print("Within one standard error of the best, so comparable rather than better: "
                  + ", ".join(rivals["model"]))

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp22_pit_stop_validation",
        "generated_at": datetime.now(UTC).isoformat(),
        "n_sessions": len(sessions),
        "n_stops_scored": totals["scored"],
        "n_stops_excluded": totals["excluded"],
        "hit_tolerance_laps": HIT_TOLERANCE,
        "pit_loss_s": args.pit_loss,
        "literature_benchmark": LITERATURE_BENCHMARK,
        "summary": table.to_dict(orient="records"),
        "like_for_like": fair.to_dict(orient="records"),
        "n_common_stops": n_common,
        "rows": all_rows,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
