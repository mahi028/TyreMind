"""Experiment 19 -- would the model's real-time signal have warned before the real stop?

`exp17` asked whether the shape of a stint can be forecast in advance, by fitting
a curve on the first 70% and predicting the rest. The answer was no, outside a
warm-up pattern: a broken stick fitted early does not beat a straight line on
what a cliff stint does later (-26.5% MAE; PMC12915245-style rubber physics does
not make an under-determined extrapolation determined). That result rules out
one specific product: a system that forecasts the cliff's shape before it
arrives.

This asks a narrower, weaker, more defensible question instead. Not "can the
rest of the stint be predicted", but:

    At the actual lap a driver was still on track, was the state-space model's
    FILTERED degradation rate -- the one computable in real time, from only
    laps already run -- already statistically distinguishable from the
    compound's own pooled baseline? And when it was, did that happen at or
    before the last lap of the real stint, i.e. before the team actually
    pitted?

This is a detection claim about the current state, not a forecasting claim
about the future, and it is exactly the input a live pit-wall decision needs:
not "here is the shape of the rest of the stint", but "is what's happening
right now already outside the range this compound normally shows".

No ground truth on true wear is used or needed. The label is real and public:
`run_id` incrementing IS the team's own real pit decision, already in the lap
table `f1_loader.build_lap_table` produces (pit in/out laps themselves are
excluded, so the last lap of a run is the last measurement before the real
stop). The corpus used here is the four committed demo races (real 2024
sessions), not the full season corpus in data/season/, which this checkout
does not currently have rebuilt (see scripts/build_corpus.py).

CORRECTION, found while building this experiment: the "distinguishable from
baseline" test below reads `fit.degradation(smoothed=False)`'s per-driver
`rate`, but `exp20_rate_pooling_degeneracy.py` shows that state carries
essentially no individual information on stints this short -- it collapses to
the compound's own pooled rate for every driver on that compound (confirmed
on all four races here). So what fires below is not "this specific tyre is
anomalous"; it is "the compound's pooled rate, across the whole field, has
risen above ITS OWN session baseline by the time this driver's run ends" --
still a real, usable signal (a compound-wide degradation-acceleration alarm),
just not the individual one the framing above implies. Read `fired` and
`lead_time_laps` accordingly.

    python experiments/exp19_pit_warning_backtest.py

Writes experiments/results/exp19_pit_warning_backtest.json.
"""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from tyremind.models.ssm.tyre_ssm import fit_tyre_ssm

RESULTS = Path(__file__).parent / "results" / "exp19_pit_warning_backtest.json"
DEMO = Path("data/demo")
RACES = ["monza", "silverstone", "zandvoort", "barcelona"]

#: A run needs laps on both sides of a plausible warning to say anything. Below
#: this, "the model never had a chance to see a trend" and "the tyre was fine"
#: are indistinguishable.
MIN_RUN_LAPS = 5

#: How many combined standard errors above the compound's pooled baseline rate
#: counts as "distinguishable", not "within the noise this compound always
#: shows". 2 is the usual one-sided ~97.5% threshold; not tuned per stint.
Z_THRESHOLD = 2.0


def analyse_race(session_id: str, lap_table: pd.DataFrame) -> list[dict]:
    fit = fit_tyre_ssm(lap_table)
    degradation = fit.degradation(smoothed=False)  # filtered = knowable in real time
    baselines = fit.compound_rates()

    records: list[dict] = []
    for (driver, run_id), run in degradation.groupby(["driver", "run_id"]):
        run = run.sort_values("session_lap")
        if len(run) < MIN_RUN_LAPS:
            continue
        # The driver's last run of the race ends the race, not a real pit
        # decision -- excluding it is what makes "before the real stop" mean
        # something.
        if run_id == degradation.loc[degradation["driver"] == driver, "run_id"].max():
            continue

        compound = str(run["compound"].iloc[0])
        if compound not in baselines:
            continue
        base_mean, base_sd = baselines[compound]

        last_lap = int(run["session_lap"].iloc[-1])
        warning_lap = None
        for row in run.itertuples(index=False):
            combined_sd = (row.rate_sd**2 + base_sd**2) ** 0.5
            if combined_sd > 0 and (row.rate - base_mean) / combined_sd > Z_THRESHOLD:
                warning_lap = int(row.session_lap)
                break

        records.append(
            {
                "session_id": session_id,
                "driver": driver,
                "run_id": int(run_id),
                "compound": compound,
                "n_laps": int(len(run)),
                "last_lap": last_lap,
                "first_lap": int(run["session_lap"].iloc[0]),
                "warning_lap": warning_lap,
                "fired": warning_lap is not None,
                "lead_time_laps": (last_lap - warning_lap) if warning_lap is not None else None,
            }
        )
    return records


def main() -> None:
    warnings.filterwarnings("ignore")
    print(f"\n  fitting the SSM per race, filtered (causal) rate only\n")

    all_records: list[dict] = []
    for name in RACES:
        path = DEMO / f"2024-{name}-R.parquet"
        if not path.exists():
            print(f"  {name:<12} skip (not on disk)")
            continue
        lap_table = pd.read_parquet(path)
        records = analyse_race(f"2024-{name}-R", lap_table)
        all_records.extend(records)
        n_fired = sum(r["fired"] for r in records)
        print(f"  {name:<12} {len(records):>3} stints analysed, {n_fired:>3} fired a warning")

    if len(all_records) < 10:
        raise SystemExit(f"only {len(all_records)} usable stints -- not enough to say anything")

    df = pd.DataFrame(all_records)
    fired = df[df["fired"]]
    never_fired = df[~df["fired"]]

    print("\n" + "=" * 80)
    print(f"WOULD THE REAL-TIME SIGNAL HAVE WARNED BEFORE THE REAL PIT STOP?")
    print(f"({len(df)} real stints, {df['session_id'].nunique()} races, "
          f"excluding each driver's final stint)")
    print("=" * 80)

    print(f"\n  fired at all:        {len(fired)}/{len(df)}  ({len(fired) / len(df):.0%})")
    if len(fired):
        print(f"  median lead time:    {fired['lead_time_laps'].median():.0f} laps before the real stop")
        print(f"  lead time IQR:       {fired['lead_time_laps'].quantile(.25):.0f}"
              f" - {fired['lead_time_laps'].quantile(.75):.0f} laps")
        zero_lead = (fired["lead_time_laps"] == 0).mean()
        print(f"  fired ONLY on the last lap (no real lead time): {zero_lead:.0%} of those that fired")

    print(f"\n  never distinguishable from baseline: {len(never_fired)}/{len(df)}  "
          f"({len(never_fired) / len(df):.0%})")
    print("  (expected for most stints -- most real pit stops are strategic timing,")
    print("   not a tyre that became statistically anomalous; exp17 found only ~12%")
    print("   of stints show a real cliff at all)")

    by_compound = (
        df.groupby("compound")
        .agg(n=("fired", "size"), fire_rate=("fired", "mean"))
        .to_dict("index")
    )
    print("\n  by compound:")
    for compound, stats in by_compound.items():
        print(f"    {compound:<10} n={stats['n']:>3}  fire rate={stats['fire_rate']:.0%}")

    print("\n" + "=" * 80)
    if len(fired) and fired["lead_time_laps"].median() >= 2:
        print("  VERDICT: when the signal fires, it typically fires with real lead time,")
        print("           not just on the last lap in hindsight -- usable as one input to a")
        print("           pit-wall decision, not as a lap-perfect predictor of when to stop.")
    elif len(fired):
        print("  VERDICT: the signal mostly fires very late -- close to or on the last lap")
        print("           of the real stint -- which is consistent with exp17's finding that")
        print("           cliffs are often only visible once they are nearly over, not before.")
    else:
        print("  VERDICT: the signal essentially never distinguishes itself from baseline in")
        print("           this sample -- no evidence it would help, at this threshold.")
    print("=" * 80)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp19_pit_warning_backtest",
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus": "data/demo (4 real 2024 races -- data/season/ full corpus not rebuilt in this checkout)",
        "z_threshold": Z_THRESHOLD,
        "min_run_laps": MIN_RUN_LAPS,
        "n_stints": int(len(df)),
        "n_races": int(df["session_id"].nunique()),
        "fire_rate": float(len(fired) / len(df)),
        "median_lead_time_laps": float(fired["lead_time_laps"].median()) if len(fired) else None,
        "lead_time_iqr": [
            float(fired["lead_time_laps"].quantile(.25)),
            float(fired["lead_time_laps"].quantile(.75)),
        ] if len(fired) else None,
        "by_compound": by_compound,
        "stints": all_records,
    }, indent=2, default=float))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
