"""exp28 -- what each model's pit call would have been worth, in seconds and positions.

Pre-registered in `experiments/PREREGISTRATION_exp28.md`. Read that first; it
fixes the referees, the metrics and the losing conditions.

Everything else in this project reports seconds per lap of degradation error. No
sporting director buys seconds per lap. This replays real races, and at every
real pit stop asks what each model would have recommended and what that
recommendation would have been worth against the stop the team actually made.

Three things make this honest rather than a demo:

* **No hindsight.** Models are refitted on an expanding prefix of the race and a
  decision only ever uses a fit that ended at or before it. exp22 fits on the
  whole race; this does not.
* **A referee that is not us.** The counterfactual needs a degradation truth, and
  whichever estimate supplies it flatters the model closest to it. The headline
  referee is a median per-run Theil-Sen slope on fuel-corrected lap time -- robust,
  model-free, and structurally closest to a *competitor*. Our own race fit is run
  as a sensitivity check, and if the answer only holds under that one, this file
  says so.
* **Signed value.** `strategy_regret` clamps at zero, which is right for a product
  surface and wrong for a benchmark. A model that recommends worse laps has to be
  able to score negative here.

    python experiments/exp28_value_experiment.py --limit 12

Writes experiments/results/exp28_value_experiment.json.
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

from tyremind.data.corpus import load_frames
from tyremind.models.baselines import FUEL_SLOPE_S_PER_LAP
from tyremind.models.literature import extended_ladder
from tyremind.models.pit_decision import (
    DEFAULT_PIT_LOSS_S,
    actual_pit_laps,
    excluded_stops,
    recommend_pit_lap,
)
from tyremind.simulate.race import RaceState, Strategy, TyreModel, simulate_strategy

RESULTS = Path(__file__).parent / "results" / "exp28_value_experiment.json"
SEASON_DIR = Path("data/season")
MIN_LAPS = 200

#: Expanding-window refit points, as fractions of race distance. A decision uses
#: the latest checkpoint at or before it, so no lap after the decision is ever in
#: the fit that informed it.
CHECKPOINTS = (0.15, 0.30, 0.50, 0.70)

#: Shortest run that can carry a Theil-Sen slope worth trusting.
MIN_RUN_LAPS_FOR_REFEREE = 6

#: The assumption `explain/business.py` ships with, reported beside the measured
#: figure so the shipped claim is checkable rather than taken on faith.
ASSUMED_SECONDS_PER_POSITION = 8.0

N_SIMS = 5000
SIM_SEED = 7

OUR_MODEL = "TyreMind state-space"
NAIVE_MODEL = "Naive (lap time vs tyre age)"


def theil_sen_referee(lap_table: pd.DataFrame) -> dict[str, float]:
    """Referee A: median per-run Theil-Sen degradation slope, per compound.

    Fuel is removed physically rather than fitted, and the slope is taken *within*
    a run so the run intercept and the driver's pace cancel. Pooling runs before
    fitting would let differing intercepts masquerade as degradation.

    Theil-Sen rather than least squares because a race stint contains traffic
    laps, a scruffy lap and an out-of-position lap, and a single 3-second lap
    moves an OLS slope a long way. Theil-Sen is the median of pairwise slopes and
    barely notices.

    This is deliberately not a rung on the ladder. Its nearest relative is the
    fuel-corrected baseline, which is a competitor.
    """
    from scipy.stats import theilslopes

    per_compound: dict[str, list[float]] = {}
    for (_, compound), run in lap_table.groupby(["run_id", "compound"]):
        if len(run) < MIN_RUN_LAPS_FOR_REFEREE:
            continue
        age = run["tyre_age"].to_numpy(dtype=float)
        if np.ptp(age) == 0:
            continue
        corrected = (
            run["lap_time"].to_numpy(dtype=float)
            + FUEL_SLOPE_S_PER_LAP * run["lap_in_run"].to_numpy(dtype=float)
        )
        keep = np.isfinite(age) & np.isfinite(corrected)
        if keep.sum() < MIN_RUN_LAPS_FOR_REFEREE:
            continue
        try:
            slope = float(theilslopes(corrected[keep], age[keep])[0])
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(slope):
            per_compound.setdefault(str(compound), []).append(slope)

    return {c: float(np.median(v)) for c, v in per_compound.items() if v}


def tyremind_referee(lap_table: pd.DataFrame) -> dict[str, float]:
    """Referee B: our own race fit. The sensitivity check, never the headline."""
    from tyremind.models.baselines import TyreStateModel

    try:
        model = TyreStateModel().fit(lap_table)
    except Exception:  # noqa: BLE001 - a referee that will not fit is simply absent
        return {}
    return {
        str(c): float(m)
        for c, (m, _) in model.compound_rates().items()
        if np.isfinite(m)
    }


def seconds_per_position(lap_table: pd.DataFrame, final_lap: int) -> float:
    """Median race-time gap between adjacent cars, estimated from pace.

    Each driver's median lap time projected over the race distance, sorted, and
    the median difference between neighbours taken. It is an estimate of the
    *local* cost of one position and cannot represent the real spread -- the gap
    to the leader and the gap in the midfield differ by an order of magnitude.
    That is why positions are the softer of the two numbers this experiment
    reports.
    """
    pace = lap_table.groupby("driver")["lap_time"].median().to_numpy(dtype=float)
    pace = np.sort(pace[np.isfinite(pace)]) * float(final_lap)
    if pace.size < 3:
        return float("nan")
    gaps = np.diff(pace)
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    return float(np.median(gaps)) if gaps.size else float("nan")


def fit_checkpoints(lap_table: pd.DataFrame, final_lap: int) -> dict[int, dict[str, dict]]:
    """Fit the whole ladder at each expanding-window checkpoint.

    Returns:
        checkpoint lap -> model name -> compound -> (rate, sd). A model that
        raises, or that has no degradation parameter, gets an empty dict at that
        checkpoint; both cases are reported rather than hidden.
    """
    out: dict[int, dict[str, dict]] = {}
    for fraction in CHECKPOINTS:
        cutoff = int(round(fraction * final_lap))
        prefix = lap_table[lap_table["session_lap"] <= cutoff]
        if prefix.empty or prefix["driver"].nunique() < 2:
            continue
        rates: dict[str, dict] = {}
        for model in extended_ladder():
            try:
                model.fit(prefix)
                rates[model.name] = {
                    str(c): (float(m), float(s))
                    for c, (m, s) in model.compound_rates().items()
                    if np.isfinite(m)
                }
            except Exception:  # noqa: BLE001 - a rung failing must not stop the replay
                rates[model.name] = {}
        out[cutoff] = rates
    return out


def referee_tyres(rates: dict[str, float]) -> dict[str, TyreModel]:
    """Build the simulator's tyre models from a referee's rates.

    `base_pace_s` is zero for every compound on purpose: a fresh-compound pace
    offset would let the choice of tyre, rather than the choice of lap, drive the
    counterfactual. The cliff and its severity are the simulator's shipped
    defaults, held identical across models so the only thing that differs between
    two rows of this experiment is the lap the model picked.
    """
    return {
        compound: TyreModel(
            compound=compound,
            base_pace_s=0.0,
            degradation_rate=max(float(rate), 0.0),
            degradation_rate_sd=0.0,
        )
        for compound, rate in rates.items()
        if np.isfinite(rate)
    }


def price(state: RaceState, tyres: dict[str, TyreModel], lap: int | None,
          new_compound: str | None) -> float:
    """Expected remaining race time under one pit lap, seconds."""
    outcome = simulate_strategy(
        state, Strategy("candidate", lap, new_compound), tyres,
        n_sims=N_SIMS, seed=SIM_SEED,
    )
    return outcome.expected_time


def replay_session(session_id: str, lap_table: pd.DataFrame, pit_loss_s: float) -> dict:
    """Replay one race: every scored stop, every model, both referees."""
    final_lap = int(lap_table["session_lap"].max())
    stops = actual_pit_laps(lap_table)
    checkpoints = fit_checkpoints(lap_table, final_lap)
    checkpoint_laps = sorted(checkpoints)

    referees = {
        "A_theil_sen": theil_sen_referee(lap_table),
        "B_tyremind_race_fit": tyremind_referee(lap_table),
    }
    tyre_models = {name: referee_tyres(rates) for name, rates in referees.items()}
    spp = seconds_per_position(lap_table, final_lap)
    base_lap_time = float(lap_table["lap_time"].median())

    model_names = sorted({name for rates in checkpoints.values() for name in rates})
    no_rate_models = sorted(
        name for name in model_names
        if all(not checkpoints[c].get(name) for c in checkpoint_laps)
    )

    rows: list[dict] = []
    counters = {"scored": 0, "excluded": 0, "no_checkpoint": 0, "no_referee": 0}

    for driver, driver_stops in stops.items():
        block = lap_table[lap_table["driver"] == driver].sort_values("session_lap")
        dropped = excluded_stops(block, driver_stops, field_lap_table=lap_table)
        counters["excluded"] += len(dropped)

        final_stop = max(driver_stops) if driver_stops else None

        for actual in driver_stops:
            if actual in dropped:
                continue
            stint = block[block["session_lap"] <= actual]
            if stint.empty:
                continue
            run_id = stint.iloc[-1]["run_id"]
            this_run = stint[stint["run_id"] == run_id]
            if len(this_run) < 6:
                continue

            decision_row = this_run.iloc[len(this_run) // 3]
            decision_lap = int(decision_row["session_lap"])
            current_age = float(decision_row["tyre_age"])
            compound = str(decision_row["compound"])

            usable = [c for c in checkpoint_laps if c <= decision_lap]
            if not usable:
                counters["no_checkpoint"] += 1
                continue
            cutoff = usable[-1]

            # The compound the driver actually fitted at this stop. Using the real
            # one for both candidate laps keeps the comparison about the lap.
            after = block[block["session_lap"] > actual]
            new_compound = str(after.iloc[0]["compound"]) if not after.empty else None

            priced: dict[str, dict[str, float]] = {}
            for referee, tyres in tyre_models.items():
                if compound not in tyres or (new_compound and new_compound not in tyres):
                    continue
                state = RaceState(
                    current_lap=decision_lap,
                    total_laps=final_lap,
                    position=0,
                    current_compound=compound,
                    current_tyre_age=current_age,
                    gap_ahead_s=2.0,
                    gap_behind_s=2.0,
                    base_lap_time_s=base_lap_time,
                    pit_loss_s=pit_loss_s,
                )
                priced[referee] = {"state": state, "tyres": tyres,
                                   "actual_time": price(state, tyres, int(actual), new_compound)}
            if not priced:
                counters["no_referee"] += 1
                continue

            counters["scored"] += 1

            for name in model_names:
                rates = checkpoints[cutoff].get(name, {})
                base = {
                    "session": session_id,
                    "driver": driver,
                    "actual_lap": int(actual),
                    "decision_lap": decision_lap,
                    "checkpoint_lap": int(cutoff),
                    "compound": compound,
                    "new_compound": new_compound,
                    "model": name,
                    "seconds_per_position": spp,
                    # The simulator prices one stop and then runs to the flag. For
                    # a driver's LAST stop that is exactly what happened, so both
                    # candidate laps are priced correctly. For an earlier stop it
                    # is not: a long final stint is charged to whichever candidate
                    # boxed sooner, which systematically punishes the earlier lap.
                    # Only final stops carry the headline; the rest are recorded
                    # and reported separately, carrying that bias on the label.
                    "is_final_stop": bool(final_stop is not None and actual == final_stop),
                }
                if not rates or compound not in rates:
                    rows.append({**base, "recommended_lap": None,
                                 "reason": "no rate for this compound at this checkpoint"})
                    continue

                current = rates[compound]
                others = [v[0] for c, v in rates.items() if c != compound]
                fresh = float(np.mean(others)) if others else float(current[0])

                recommendation = recommend_pit_lap(
                    current_rate=float(current[0]),
                    current_rate_sd=float(current[1]) if np.isfinite(current[1]) else 0.05,
                    fresh_rate=fresh,
                    current_age=current_age,
                    decision_lap=decision_lap,
                    final_lap=final_lap,
                    pit_loss_s=pit_loss_s,
                    # One remaining stint, because that is exactly what the
                    # counterfactual prices. Letting the optimiser plan a
                    # two-stopper while the simulator charges it for a
                    # one-stopper would score the model for a stop the
                    # counterfactual never lets it take. Identical for every rung.
                    max_remaining_stops=1,
                )
                if recommendation.reason:
                    rows.append({**base, "recommended_lap": None,
                                 "reason": recommendation.reason})
                    continue

                row = {**base, "recommended_lap": int(recommendation.lap),
                       "error_laps": int(recommendation.lap - actual),
                       "confidence": float(recommendation.confidence), "reason": ""}
                for referee, cached in priced.items():
                    recommended_time = price(
                        cached["state"], cached["tyres"],
                        int(recommendation.lap), new_compound,
                    )
                    value = float(cached["actual_time"] - recommended_time)
                    row[f"value_s_{referee}"] = value
                    row[f"positions_{referee}"] = (
                        value / spp if np.isfinite(spp) and spp > 0 else float("nan")
                    )
                    row[f"positions_assumed_{referee}"] = value / ASSUMED_SECONDS_PER_POSITION
                    # The clamped product-surface figure, for continuity with
                    # /api/session/{id}/regret. Not the benchmark quantity.
                    row[f"regret_s_clamped_{referee}"] = max(0.0, value)
                rows.append(row)

    return {"rows": rows, "counters": counters, "referees": referees,
            "seconds_per_position": spp, "no_degradation_parameter": no_rate_models,
            "checkpoints": checkpoint_laps, "final_lap": final_lap}


def final_stops_only(frame: pd.DataFrame) -> pd.DataFrame:
    """The subset the counterfactual can price honestly. See `is_final_stop`."""
    return frame[frame["is_final_stop"]]


def like_for_like(frame: pd.DataFrame, referee: str) -> tuple[pd.DataFrame, int]:
    """Score every model on only the stops every model answered.

    Declining is legitimate -- `recommend_pit_lap` refuses when the cost curve is
    flat or the rate is indistinguishable from zero -- but it makes a raw mean
    incomparable, because the models that decline most are graded on the easiest
    remaining cases. exp22 found exactly that: the naive rung answered 119 of 274
    stops and its mean error halved.
    """
    column = f"value_s_{referee}"
    if column not in frame.columns:
        return pd.DataFrame(), 0
    answered = frame[frame["recommended_lap"].notna() & frame[column].notna()].copy()
    if answered.empty:
        return pd.DataFrame(), 0
    answered["key"] = (answered["session"] + "|" + answered["driver"]
                       + "|" + answered["actual_lap"].astype(int).astype(str))

    models = sorted(answered["model"].unique())
    shared: set | None = None
    for model in models:
        keys = set(answered.loc[answered["model"] == model, "key"])
        shared = keys if shared is None else (shared & keys)
    shared = shared or set()

    total_stops = frame.drop_duplicates(["session", "driver", "actual_lap"]).shape[0]
    out = []
    for model in models:
        block = answered[(answered["model"] == model) & (answered["key"].isin(shared))]
        if block.empty:
            continue
        value = block[column].to_numpy(dtype=float)
        n = value.size
        # Per car-race: the sum over that driver's scored stops in that race.
        per_car = block.groupby(["session", "driver"])[column].sum()
        answered_n = int((answered["model"] == model).sum())
        out.append({
            "model": model,
            "n_common_stops": int(n),
            "value_s_per_stop": float(value.mean()),
            "value_se_per_stop": float(value.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            "value_s_per_car_race": float(per_car.mean()),
            "value_se_per_car_race": (
                float(per_car.std(ddof=1) / np.sqrt(len(per_car))) if len(per_car) > 1
                else float("nan")
            ),
            "n_car_races": int(len(per_car)),
            "positions_per_car_race": float(
                per_car.mean() / frame["seconds_per_position"].median()
            ),
            "positions_per_car_race_assumed_8s": float(
                per_car.mean() / ASSUMED_SECONDS_PER_POSITION
            ),
            "mean_abs_error_laps": float(block["error_laps"].abs().mean()),
            "answered": answered_n,
            "answer_rate": float(answered_n / total_stops) if total_stops else float("nan"),
        })
    if not out:
        return pd.DataFrame(), len(shared)
    return pd.DataFrame(out).sort_values("value_s_per_car_race", ascending=False), len(shared)


def summarise_all_answered(frame: pd.DataFrame, referee: str) -> pd.DataFrame:
    """Every stop each model answered, without the common-subset restriction.

    Reported beside the like-for-like table, never instead of it. A model that
    declines often is graded here on the subset it found easy, which is exactly
    the distortion `like_for_like` exists to remove -- but when the common subset
    collapses to nothing, this is the only view left, and an empty table would
    hide that rather than explain it.
    """
    column = f"value_s_{referee}"
    if column not in frame.columns:
        return pd.DataFrame()
    answered = frame[frame["recommended_lap"].notna() & frame[column].notna()]
    total_stops = frame.drop_duplicates(["session", "driver", "actual_lap"]).shape[0]
    out = []
    for model, block in answered.groupby("model"):
        value = block[column].to_numpy(dtype=float)
        n = value.size
        per_car = block.groupby(["session", "driver"])[column].sum()
        out.append({
            "model": model,
            "n_stops": int(n),
            "value_s_per_stop": float(value.mean()),
            "value_se_per_stop": float(value.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            "value_s_per_car_race": float(per_car.mean()),
            "mean_abs_error_laps": float(block["error_laps"].abs().mean()),
            "answer_rate": float(n / total_stops) if total_stops else float("nan"),
        })
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_values("value_s_per_car_race", ascending=False)


def paired_difference(frame: pd.DataFrame, referee: str, a: str, b: str) -> dict:
    """Paired difference in value between two models, on the stops both answered.

    Paired, because the two models are scored on the same stops with the same
    referee and the same random numbers. An unpaired comparison would drown a real
    difference in the variance between stops, which is enormous.
    """
    column = f"value_s_{referee}"
    if column not in frame.columns:
        return {}
    answered = frame[frame["recommended_lap"].notna() & frame[column].notna()].copy()
    answered["key"] = (answered["session"] + "|" + answered["driver"]
                       + "|" + answered["actual_lap"].astype(int).astype(str))
    left = answered[answered["model"] == a].set_index("key")[column]
    right = answered[answered["model"] == b].set_index("key")[column]
    shared = left.index.intersection(right.index)
    if len(shared) < 2:
        return {"n": int(len(shared))}
    paired = (left.loc[shared] - right.loc[shared]).dropna()
    diff = paired.to_numpy(dtype=float)
    diff = diff[np.isfinite(diff)]
    n = diff.size
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

    # Per car-race: sum the paired difference over one driver's scored stops in
    # one race, which is the unit the headline sentence is stated in.
    keys = answered.drop_duplicates("key").set_index("key")
    carrace = paired.to_frame("diff").join(keys[["session", "driver"]])
    per_car = carrace.groupby(["session", "driver"])["diff"].sum()
    car_se = (
        float(per_car.std(ddof=1) / np.sqrt(len(per_car))) if len(per_car) > 1 else float("nan")
    )

    return {
        "model_a": a, "model_b": b, "n_paired_stops": int(n),
        "mean_difference_s_per_stop": float(diff.mean()),
        "se_s_per_stop": se,
        "n_car_races": int(len(per_car)),
        "mean_difference_s_per_car_race": float(per_car.mean()),
        "se_s_per_car_race": car_se,
        "significant_at_1_se": bool(np.isfinite(se) and abs(diff.mean()) > se),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--pit-loss", type=float, default=DEFAULT_PIT_LOSS_S)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    races = load_frames(SEASON_DIR, session_type="R", limit=args.limit, min_laps=MIN_LAPS)
    if not races:
        raise SystemExit(f"no races in {SEASON_DIR}; run scripts/build_corpus.py first")

    all_rows: list[dict] = []
    totals = {"scored": 0, "excluded": 0, "no_checkpoint": 0, "no_referee": 0}
    per_session: list[dict] = []
    no_rate_models: set[str] = set()

    for session_id, lap_table in races.items():
        started = time.perf_counter()
        result = replay_session(session_id, lap_table, args.pit_loss)
        all_rows.extend(result["rows"])
        for key in totals:
            totals[key] += result["counters"][key]
        no_rate_models.update(result["no_degradation_parameter"])
        per_session.append({
            "session": session_id,
            "final_lap": result["final_lap"],
            "checkpoints": result["checkpoints"],
            "seconds_per_position": result["seconds_per_position"],
            "referee_rates": result["referees"],
            **result["counters"],
        })
        print(f"  {session_id:<42} {result['counters']['scored']:>3} stops, "
              f"{result['counters']['excluded']:>2} excluded, "
              f"s/position {result['seconds_per_position']:.1f}  "
              f"{time.perf_counter() - started:>6.1f}s", flush=True)

    frame = pd.DataFrame(all_rows)
    if frame.empty:
        raise SystemExit("no stop could be replayed")

    # The counterfactual prices one stop and runs to the flag, which is the truth
    # only for a driver's last stop. Everything headline is computed on those.
    scored = final_stops_only(frame)
    n_final = int(scored.drop_duplicates(["session", "driver", "actual_lap"]).shape[0])
    n_earlier = int(
        frame.drop_duplicates(["session", "driver", "actual_lap"]).shape[0] - n_final
    )
    print(f"\n{n_final} final stops carry the headline; {n_earlier} earlier stops are "
          "reported separately -- the single-stop counterfactual mis-prices them.")

    report: dict = {}
    for referee in ("A_theil_sen", "B_tyremind_race_fit"):
        table, n_common = like_for_like(scored, referee)
        every = summarise_all_answered(scored, referee)
        every_earlier = summarise_all_answered(frame[~frame["is_final_stop"]], referee)
        report[referee] = {"table": table, "n_common": n_common, "all_answered": every,
                           "earlier_stops": every_earlier}

        label = ("A -- Theil-Sen, model-free (HEADLINE)" if referee == "A_theil_sen"
                 else "B -- our own race fit (SENSITIVITY CHECK ONLY)")
        print("\n" + "=" * 104)
        print(f"VALUE OF THE PIT CALL, referee {label}")
        print(f"{n_common} final stops answered by every model, out of {n_final} final stops "
              f"across {len(races)} races")
        print("Positive = following the model would have been faster than the stop the team made.")
        print("=" * 104)
        if table.empty:
            print("  NO STOP was answered by every model, so there is no like-for-like "
                  "subset. Every model's own answered stops:")
            if not every.empty:
                print(f"{'model':<34}{'s/car-race':>12}{'s/stop':>9}{'+-SE':>8}"
                      f"{'MAElaps':>9}{'answer':>8}{'n':>6}")
                for _, row in every.iterrows():
                    print(f"{row['model']:<34}{row['value_s_per_car_race']:>+12.2f}"
                          f"{row['value_s_per_stop']:>+9.2f}{row['value_se_per_stop']:>8.2f}"
                          f"{row['mean_abs_error_laps']:>9.1f}{row['answer_rate']:>8.0%}"
                          f"{int(row['n_stops']):>6}")
            continue
        print(f"{'model':<34}{'s/car-race':>12}{'+-SE':>8}{'s/stop':>9}{'+-SE':>8}"
              f"{'pos/race':>10}{'MAElaps':>9}{'answer':>8}")
        for _, row in table.iterrows():
            print(f"{row['model']:<34}{row['value_s_per_car_race']:>+12.2f}"
                  f"{row['value_se_per_car_race']:>8.2f}{row['value_s_per_stop']:>+9.2f}"
                  f"{row['value_se_per_stop']:>8.2f}{row['positions_per_car_race']:>+10.2f}"
                  f"{row['mean_abs_error_laps']:>9.1f}{row['answer_rate']:>8.0%}")
        print("=" * 104)

    headline = report["A_theil_sen"]["all_answered"]
    sentence = "no comparison was possible"
    verdict: dict = {}
    if not headline.empty and {OUR_MODEL, NAIVE_MODEL} <= set(headline["model"]):
        # The headline sentence is a claim about two models, so it is computed on
        # the stops those two both answered -- a far larger and better-matched
        # sample than the six-way intersection, which the ranking table uses.
        paired = paired_difference(scored, "A_theil_sen", OUR_MODEL, NAIVE_MODEL)
        paired_b = paired_difference(scored, "B_tyremind_race_fit", OUR_MODEL, NAIVE_MODEL)

        ours = headline[headline["model"] == OUR_MODEL].iloc[0]
        naive = headline[headline["model"] == NAIVE_MODEL].iloc[0]

        gap_s = float(paired.get("mean_difference_s_per_car_race", float("nan")))
        spp_median = float(scored["seconds_per_position"].median())
        gap_pos = gap_s / spp_median if np.isfinite(spp_median) and spp_median > 0 else float("nan")

        underpowered = bool(paired.get("n_paired_stops", 0) < 20)
        flipped = bool(
            paired.get("mean_difference_s_per_stop", 0.0)
            * paired_b.get("mean_difference_s_per_stop", 0.0) < 0
        )

        if gap_s >= 0:
            sentence = (
                f"Using the naive estimate instead of ours would have cost "
                f"{gap_s:.1f} seconds and {gap_pos:.2f} positions per car per race."
            )
        else:
            sentence = (
                f"Using the naive estimate instead of ours would have GAINED "
                f"{abs(gap_s):.1f} seconds and {abs(gap_pos):.2f} positions per car per "
                f"race. Our recommendation was worth less than the naive one."
            )

        verdict = {
            "gap_seconds_per_car_race": gap_s,
            "gap_positions_per_car_race": gap_pos,
            "seconds_per_position_measured_median": spp_median,
            "gap_positions_at_assumed_8s": gap_s / ASSUMED_SECONDS_PER_POSITION,
            "tyremind_value_s_per_car_race": float(ours["value_s_per_car_race"]),
            "naive_value_s_per_car_race": float(naive["value_s_per_car_race"]),
            "tyremind_value_negative": bool(ours["value_s_per_car_race"] < 0),
            "paired_referee_a": paired,
            "paired_referee_b": paired_b,
            "claim_stands": bool(
                paired.get("significant_at_1_se", False)
                and paired.get("mean_difference_s_per_stop", 0.0) > 0
                and not underpowered
                and not flipped
            ),
            "underpowered": underpowered,
            "referees_disagree": flipped,
            "sentence": sentence,
        }

        print("\n" + "-" * 104)
        print("THE ONE SENTENCE")
        print("-" * 104)
        print(sentence)
        print()
        print(f"Paired over the {paired.get('n_paired_stops', 0)} stops both models answered, "
              f"referee A: {paired.get('mean_difference_s_per_stop', float('nan')):+.2f} "
              f"+- {paired.get('se_s_per_stop', float('nan')):.2f} s/stop")
        print(f"Same comparison under referee B (our own race fit): "
              f"{paired_b.get('mean_difference_s_per_stop', float('nan')):+.2f} "
              f"+- {paired_b.get('se_s_per_stop', float('nan')):.2f} s/stop")
        if flipped:
            print("THE REFEREES DISAGREE ON THE SIGN. No claim is made from either; "
                  "the disagreement is the finding.")
        if underpowered:
            print(f"UNDERPOWERED: only {paired.get('n_paired_stops', 0)} final stops were "
                  "answered by both models, below the pre-registered floor of 20. "
                  "No claim is made.")
        if verdict["tyremind_value_negative"]:
            print("FINDING, REPORTED FIRST: following our recommendation would have been "
                  "SLOWER than the stops the teams actually made.")
        if not verdict["claim_stands"]:
            print("The pre-registered claim does NOT stand.")
        print("-" * 104)

    if no_rate_models:
        print("\nNo degradation parameter, so cannot recommend a pit lap at all "
              "(the exp19 finding, unchanged): " + ", ".join(sorted(no_rate_models)))

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp28_value_experiment",
        "preregistration": "experiments/PREREGISTRATION_exp28.md",
        "generated_at": datetime.now(UTC).isoformat(),
        "n_sessions": len(races),
        "checkpoint_fractions": list(CHECKPOINTS),
        "n_sims": N_SIMS,
        "sim_seed": SIM_SEED,
        "pit_loss_s": args.pit_loss,
        "assumed_seconds_per_position": ASSUMED_SECONDS_PER_POSITION,
        "totals": totals,
        "n_final_stops": n_final,
        "n_earlier_stops": n_earlier,
        "models_without_degradation_parameter": sorted(no_rate_models),
        "headline_sentence": sentence,
        "verdict": verdict,
        "referee_a_theil_sen": {
            "n_common_stops": report["A_theil_sen"]["n_common"],
            "summary": report["A_theil_sen"]["table"].to_dict(orient="records"),
            "all_answered_final_stops": report["A_theil_sen"]["all_answered"].to_dict(
                orient="records"),
            "earlier_stops_mispriced": report["A_theil_sen"]["earlier_stops"].to_dict(
                orient="records"),
        },
        "referee_b_tyremind_race_fit": {
            "n_common_stops": report["B_tyremind_race_fit"]["n_common"],
            "summary": report["B_tyremind_race_fit"]["table"].to_dict(orient="records"),
            "all_answered_final_stops": report["B_tyremind_race_fit"]["all_answered"].to_dict(
                orient="records"),
            "earlier_stops_mispriced": report["B_tyremind_race_fit"]["earlier_stops"].to_dict(
                orient="records"),
        },
        "deviations": [
            "Value is signed. strategy_regret clamps at zero, which is correct for the "
            "product surface and wrong for a benchmark; the clamped figure is recorded "
            "per row as regret_s_clamped_* for continuity with /api/session/{id}/regret.",
            "Models are refitted on expanding prefixes rather than on the whole race, "
            "which is stricter than exp22 and in our disfavour.",
            "Only a driver's FINAL stop carries the headline. The simulator prices one "
            "stop and then runs to the flag, which is the truth for a final stop and "
            "systematically punishes the earlier of two candidate laps otherwise. "
            "Earlier stops are still scored and reported under earlier_stops_mispriced, "
            "labelled with that bias.",
            "recommend_pit_lap is called with max_remaining_stops=1 so the optimiser and "
            "the counterfactual price the same number of stints. Identical for every rung.",
            "The headline sentence is a paired two-model comparison on the final stops "
            "both models answered, not the six-way common subset, which is far smaller.",
        ],
        "per_session": per_session,
        "rows": all_rows,
    }, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
