"""exp30 -- is the confidence on a pit call worth anything?

exp22 asked whether we recommend the right lap. This asks the harder and more
commercially important question: **when we say we are 51% confident, does the
driver box inside that window 51% of the time?**

Those are different properties and only the second one makes the product usable.
A recommendation that is wrong but honestly uncertain is actionable -- a
strategist widens the window and keeps options open. A recommendation that is
wrong while claiming certainty is worse than no recommendation at all, because it
removes the strategist's own judgement from the loop.

This matters because exp22's headline is unflattering: across 246 real stops we
land within two laps 24% of the time. That figure is only defensible if the
confidence said so. A system claiming 95% and hitting 24% is worthless; one
claiming roughly 50% for a window and hitting roughly 50% is trustworthy, and the
difference is entirely in this experiment.

Three tests, in increasing strictness:

1. **Lift.** Is the probability we place on the lap the driver actually chose
   higher than uniform chance over the candidate laps? If not, the distribution
   carries no information and everything downstream is decoration.

2. **Top-k.** Does the actual stop fall in our k most likely laps more often than
   chance? A strategist does not act on one lap; they act on a shortlist.

3. **Window calibration.** The real test. Bin stops by the probability mass we
   assigned to our window, and check the observed frequency of the actual stop
   landing inside. Perfect calibration is the diagonal. This is the same
   reliability-diagram machinery exp16 applies to lap times, pointed at a
   decision instead of a measurement.

Every model on the ladder is scored, so "our confidence is meaningful" is a
comparative claim rather than an assertion.

    python experiments/exp30_pit_confidence_calibration.py --limit 14
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

RESULTS = Path(__file__).parent / "results" / "exp30_pit_confidence_calibration.json"
SEASON_DIR = Path("data/season")
MIN_LAPS = 200

#: Window half-width in laps. Teams talk in windows, not single laps, and two
#: either side is the width a race engineer actually treats as "now".
WINDOW = 2

#: Shortlist sizes for the top-k test.
TOP_K = (1, 3, 5)


def collect(lap_table: pd.DataFrame, models: list, pit_loss_s: float) -> list[dict]:
    """Every model's full belief over pit laps, for every scoreable real stop."""
    stops = actual_pit_laps(lap_table)
    final_lap = int(lap_table["session_lap"].max())

    fitted = {}
    for model in models:
        try:
            model.fit(lap_table)
            fitted[model.name] = model.compound_rates()
        except Exception:  # noqa: BLE001
            fitted[model.name] = {}

    rows = []
    for driver, driver_stops in stops.items():
        block = lap_table[lap_table["driver"] == driver].sort_values("session_lap")
        dropped = excluded_stops(block, driver_stops, field_lap_table=lap_table)

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
            decision = this_run.iloc[len(this_run) // 3]
            decision_lap = int(decision["session_lap"])
            compound = str(decision["compound"])

            for model in models:
                rates = fitted[model.name]
                current = rates.get(compound)
                if current is None:
                    continue
                others = [v[0] for c, v in rates.items() if c != compound]
                fresh = float(np.mean(others)) if others else current[0]

                rec = recommend_pit_lap(
                    current_rate=float(current[0]),
                    current_rate_sd=float(current[1]) if np.isfinite(current[1]) else 0.05,
                    fresh_rate=fresh,
                    current_age=float(decision["tyre_age"]),
                    decision_lap=decision_lap,
                    final_lap=final_lap,
                    pit_loss_s=pit_loss_s,
                )
                if rec.reason or not rec.distribution:
                    continue

                laps = sorted(rec.distribution)
                probabilities = np.array([rec.distribution[l] for l in laps], dtype=float)
                n_candidates = len(laps)

                # The window we would actually have shown, and the mass on it.
                window = [l for l in laps if abs(l - rec.lap) <= WINDOW]
                window_mass = float(sum(rec.distribution[l] for l in window))
                actual_in_window = bool(abs(int(actual) - rec.lap) <= WINDOW)

                order = [laps[i] for i in np.argsort(-probabilities)]
                rank = order.index(int(actual)) + 1 if int(actual) in order else None

                rows.append({
                    "model": model.name,
                    "driver": driver,
                    "actual": int(actual),
                    "recommended": int(rec.lap),
                    "decision_lap": decision_lap,
                    "n_candidates": int(n_candidates),
                    "uniform_chance": 1.0 / n_candidates,
                    "prob_on_actual": float(rec.distribution.get(int(actual), 0.0)),
                    "window_mass": window_mass,
                    "actual_in_window": actual_in_window,
                    "rank_of_actual": rank,
                })
    return rows


def summarise(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    out = []
    for model, block in frame.groupby("model"):
        chance = block["uniform_chance"].mean()
        observed = block["prob_on_actual"].mean()
        record = {
            "model": model,
            "n": int(len(block)),
            "mean_prob_on_actual": float(observed),
            "uniform_chance": float(chance),
            # Lift below 1.0 means the distribution is worse than guessing.
            "lift": float(observed / chance) if chance > 0 else float("nan"),
            "mean_window_mass": float(block["window_mass"].mean()),
            "observed_window_hit": float(block["actual_in_window"].mean()),
        }
        # Calibration gap: what we claimed for the window against what happened.
        record["calibration_gap"] = record["observed_window_hit"] - record["mean_window_mass"]
        ranked = block["rank_of_actual"].dropna()
        for k in TOP_K:
            record[f"top_{k}_hit"] = float((ranked <= k).mean()) if len(ranked) else float("nan")
            record[f"top_{k}_chance"] = float((k / block["n_candidates"]).mean())
        out.append(record)
    return pd.DataFrame(out).sort_values("lift", ascending=False)


def reliability(rows: list[dict], model: str, bins: int = 5) -> list[dict]:
    """Claimed window probability against observed frequency, binned."""
    block = pd.DataFrame([r for r in rows if r["model"] == model])
    if block.empty:
        return []
    edges = np.linspace(block["window_mass"].min(), block["window_mass"].max(), bins + 1)
    out = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = block[(block["window_mass"] >= low) & (block["window_mass"] <= high)]
        if len(selected) < 5:
            continue
        out.append({
            "claimed_low": float(low),
            "claimed_high": float(high),
            "mean_claimed": float(selected["window_mass"].mean()),
            "observed": float(selected["actual_in_window"].mean()),
            "n": int(len(selected)),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=14)
    parser.add_argument("--pit-loss", type=float, default=DEFAULT_PIT_LOSS_S)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    sessions = load_frames(SEASON_DIR, session_type="R", limit=args.limit, min_laps=MIN_LAPS)
    if not sessions:
        raise SystemExit(f"no races in {SEASON_DIR}")

    rows: list[dict] = []
    for session_id, lap_table in sessions.items():
        got = collect(lap_table, extended_ladder(), args.pit_loss)
        for row in got:
            row["session"] = session_id
        rows.extend(got)
        print(f"  {session_id:<42} {len(got):>4} model-stop pairs", flush=True)

    table = summarise(rows)

    print("\n" + "=" * 100)
    print("DOES THE CONFIDENCE MEAN ANYTHING?")
    print("=" * 100)
    print(f"{'model':<34}{'P(actual)':>11}{'chance':>9}{'lift':>7}"
          f"{'claimed':>9}{'observed':>10}{'gap':>8}{'n':>6}")
    for _, r in table.iterrows():
        print(f"{r['model']:<34}{r['mean_prob_on_actual']:>11.4f}{r['uniform_chance']:>9.4f}"
              f"{r['lift']:>7.2f}{r['mean_window_mass']:>9.1%}{r['observed_window_hit']:>10.1%}"
              f"{r['calibration_gap']:>+8.1%}{int(r['n']):>6}")
    print("=" * 100)
    print("lift > 1 means the distribution beats guessing.")
    print("claimed vs observed is the calibration test: we say the window holds")
    print("this much probability; the driver boxed inside it this often.")

    print()
    print(f"{'model':<34}" + "".join(f"{'top' + str(k):>9}" for k in TOP_K)
          + "   (chance in brackets)")
    for _, r in table.iterrows():
        cells = "".join(f"{r[f'top_{k}_hit']:>9.1%}" for k in TOP_K)
        chance = ", ".join(f"{r[f'top_{k}_chance']:.0%}" for k in TOP_K)
        print(f"{r['model']:<34}{cells}   ({chance})")

    ours = "TyreMind state-space"
    curve = reliability(rows, ours)
    if curve:
        print()
        print(f"RELIABILITY OF THE WINDOW -- {ours}")
        print(f"  {'claimed':>10}{'observed':>11}{'n':>6}")
        for point in curve:
            print(f"  {point['mean_claimed']:>10.1%}{point['observed']:>11.1%}{point['n']:>6}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp30_pit_confidence_calibration",
        "generated_at": datetime.now(UTC).isoformat(),
        "n_sessions": len(sessions),
        "window_half_width_laps": WINDOW,
        "summary": table.to_dict(orient="records"),
        "reliability_tyremind": curve,
        "rows": rows,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
