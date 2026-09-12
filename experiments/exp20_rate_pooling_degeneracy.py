"""Experiment 20 -- does the SSM's per-driver degradation rate carry any driver
information at all, or does it collapse to the compound average?

`tyre_ssm.py`'s own docstring calls `rate` "the deliverable: instantaneous
degradation in seconds per lap", per driver, per lap. This experiment checks
whether that claim survives contact with real data -- not by reading the code,
by fitting it and inspecting what the optimiser actually chose.

THE MECHANISM. Two hyperparameters govern how much an individual tyre's rate
is allowed to differ from its compound's pooled baseline:

  stint_rate_sd   spread of a freshly-fitted tyre's rate around the compound
                  baseline, at the moment it goes on (the "hierarchical
                  shrinkage parameter", per the module's own docstring)
  q_rate          how much rate is allowed to drift within a stint afterwards

Both are fitted by maximum marginal likelihood alongside everything else, each
bounded away from exactly zero in `HYPER_BOUNDS` (stint_rate_sd floor is
1e-3, q_rate floor is 1e-10) so the optimiser cannot collapse them to a
literal Dirac delta. If the MLE nonetheless drives a bounded parameter to sit
numerically on its floor, on every session tried, that is the optimiser
saying the true unconstrained optimum is even more degenerate -- individual
variation earns the model nothing, given the data.

THIS WAS FOUND BY ACCIDENT, not by design: while backtesting a pit-warning
rule (`exp19`) against two Silverstone drivers on the same compound who
started their stint on the same lap, their fitted `rate` and `rate_sd`
matched to six decimal places despite genuinely different lap times (checked
against the raw lap table, not assumed). That degree of agreement is not
physically plausible for two independently-updated states; it is the
signature of both states being, in effect, the same state.

WHAT THIS IS NOT. It is not a claim that individual tyres degrade identically
in reality -- obviously false. It is a claim about what THIS model, fitted to
stints of this length, can currently DISTINGUISH: given the confounders it
must also estimate (fuel, track evolution, traffic) and the length of a real
stint (15-25 laps), the likelihood gains nothing from letting one tyre's rate
depart from its compound's. Forcing that departure anyway -- by loosening the
bound without a principled prior to replace it -- would manufacture
individual-looking numbers with no support in the data, which is precisely
the failure mode `configs/physics.yaml`'s epistemic tagging exists to catch.

CORPUS. The four committed demo races (real 2024 sessions), not the full
`data/season/` corpus (203 sessions), which this checkout has not rebuilt.
The finding should be re-checked at that scale before being treated as
settled -- four races established the pattern, they do not exhaust it.

    python experiments/exp20_rate_pooling_degeneracy.py

Writes experiments/results/exp20_rate_pooling_degeneracy.json.
"""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tyremind.models.ssm.tyre_ssm import HYPER_BOUNDS, fit_tyre_ssm

RESULTS = Path(__file__).parent / "results" / "exp20_rate_pooling_degeneracy.json"
DEMO = Path("data/demo")
RACES = ["monza", "silverstone", "zandvoort", "barcelona"]

#: Hyperparameter vector order matches `TyreSSMHyper.to_vector`.
HYPER_NAMES = ("log_q_track", "log_q_level", "log_q_rate", "log_obs_sd", "log_stint_rate_sd", "log_track_shape")

#: How close to a bound counts as "sitting on it" rather than "landed nearby".
#: The observed matches are exact to float precision; this is generous on
#: purpose so a near-miss still counts as evidence of the same pull.
BOUND_TOLERANCE = 1e-6


def bound_status(name: str, value: float) -> str | None:
    lo, hi = HYPER_BOUNDS[HYPER_NAMES.index(name)]
    if abs(value - lo) < BOUND_TOLERANCE:
        return "floor"
    if abs(value - hi) < BOUND_TOLERANCE:
        return "ceiling"
    return None


def analyse_race(session_id: str, lap_table: pd.DataFrame) -> dict:
    fit = fit_tyre_ssm(lap_table)
    h = fit.hyper

    deg = fit.degradation(smoothed=False)
    # Real per-driver differentiation would show up as spread in `rate` across
    # drivers who share a compound at the same session lap -- same tyre type,
    # same point in the race, so any spread left is individual, not shared cause.
    cross_driver_spread = (
        deg.groupby(["compound", "session_lap"])["rate"]
        .std()
        .mean()
    )
    baseline_scale = np.mean([m for m, _ in fit.compound_rates().values()])

    return {
        "session_id": session_id,
        "converged": bool(fit.converged),
        "hyper": {name: float(getattr(h, name)) for name in HYPER_NAMES},
        "bound_hits": {
            name: bound_status(name, getattr(h, name))
            for name in HYPER_NAMES
            if bound_status(name, getattr(h, name)) is not None
        },
        "q_rate": h.q_rate,
        "stint_rate_sd": float(np.exp(h.log_stint_rate_sd)),
        "cross_driver_rate_spread": float(cross_driver_spread),
        "mean_compound_baseline_rate": float(baseline_scale),
        "spread_as_fraction_of_baseline": float(cross_driver_spread / baseline_scale) if baseline_scale else None,
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    print(f"\n  fitting each demo race, inspecting where the optimiser actually landed\n")

    results = []
    for name in RACES:
        path = DEMO / f"2024-{name}-R.parquet"
        if not path.exists():
            print(f"  {name:<12} skip (not on disk)")
            continue
        lap_table = pd.read_parquet(path)
        r = analyse_race(f"2024-{name}-R", lap_table)
        results.append(r)
        hits = ", ".join(f"{k}={v}" for k, v in r["bound_hits"].items()) or "none"
        print(f"  {name:<12} bound hits: {hits}")
        print(f"  {'':<12} cross-driver rate spread: {r['cross_driver_rate_spread']:.6f} s/lap "
              f"({r['spread_as_fraction_of_baseline']:.1%} of baseline rate)")

    if len(results) < 3:
        raise SystemExit(f"only {len(results)} usable races -- not enough to say anything")

    n_rate_sd_floor = sum(1 for r in results if r["bound_hits"].get("log_stint_rate_sd") == "floor")
    n_q_rate_near_floor = sum(1 for r in results if r["q_rate"] < 1e-6)
    median_spread_frac = float(np.median([r["spread_as_fraction_of_baseline"] for r in results]))

    print("\n" + "=" * 84)
    print("DOES THE PER-DRIVER RATE STATE CARRY DRIVER INFORMATION, OR IS IT POOLED?")
    print(f"({len(results)} real races)")
    print("=" * 84)
    print(f"\n  stint_rate_sd pinned at its floor:  {n_rate_sd_floor}/{len(results)} races")
    print(f"  q_rate effectively zero (<1e-6):    {n_q_rate_near_floor}/{len(results)} races")
    print(f"  median cross-driver rate spread:    {median_spread_frac:.2%} of the compound's own baseline rate")

    print("\n" + "=" * 84)
    if n_rate_sd_floor == len(results) and median_spread_frac < 0.01:
        print("  VERDICT: in every race tested, the model's per-driver rate state carries")
        print("           essentially no individual information -- it reports the compound's")
        print("           pooled rate for every driver on that compound. This is a real")
        print("           identifiability limit of fitting one race-length stint (15-25 laps),")
        print("           not a bug in the arithmetic. Any product surface that presents `rate`")
        print("           as this specific car's own number is currently overstating what the")
        print("           model can actually tell apart -- it is, in effect, the compound's rate.")
    else:
        print("  VERDICT: mixed -- at least one race shows real individual differentiation.")
        print("           The degeneracy is not universal; worth checking which conditions avoid it.")
    print("=" * 84)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp20_rate_pooling_degeneracy",
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus": "data/demo (4 real 2024 races -- data/season/ full corpus not rebuilt in this checkout)",
        "bound_tolerance": BOUND_TOLERANCE,
        "n_races": len(results),
        "n_stint_rate_sd_at_floor": n_rate_sd_floor,
        "n_q_rate_near_zero": n_q_rate_near_floor,
        "median_spread_as_fraction_of_baseline": median_spread_frac,
        "races": results,
    }, indent=2, default=float))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
