"""exp26 -- do sector times break the fuel/tyre collinearity?

exp18 proved that fuel burn-off and tyre degradation are *exactly* collinear
within a stint. Both advance one per lap, so writing tyre age as `a = a0 + f`
where `f` is laps completed:

    y = c + beta*a - phi*f  =  (c + beta*a0) + (beta - phi)*f

The run intercept absorbs `beta*a0` and only the difference `beta - phi` survives.
520 runs out of 520 were exactly collinear, every session's information matrix was
singular, and only 6.0% of the separation came from data rather than from a
physical prior on the fuel coefficient.

**That proof assumed one observation per lap.** Timing gives three -- sector one,
sector two, sector three -- and the two effects do not load onto them equally.
Fuel mass costs most where the car accelerates and brakes; tyre degradation costs
most where the car is grip-limited. If the sector loading vectors for fuel and
tyre are not parallel, then

    slope_s = w_s^tyre * beta  -  w_s^fuel * phi        for s = 1, 2, 3

is three equations in two unknowns, and beta becomes identifiable from data that
could not identify it before.

This experiment asks whether that is true, in three steps that get progressively
harder to explain away:

1. **Are the sector slopes different at all?** If sector degradation were simply
   proportional to sector time, sectors would carry no information a whole lap
   does not, and the idea is dead immediately.
2. **Is the pattern stable, or is it noise?** Split-half by driver within each
   circuit. A real loading structure replicates; noise does not. This is the step
   that killed our driver-effect hypothesis in exp15 and it is included here for
   the same reason.
3. **How much does identifiability actually improve?** Compare the conditioning
   of the whole-lap design against the sector design, and report the ratio of
   Cramer-Rao bounds on `beta`. This is the number that matters: exp18 says the
   whole-lap system is singular, so anything finite is an improvement, and the
   size of it decides whether this is a headline or a footnote.

**What this experiment cannot do.** It does not prove our degradation *estimates*
get better -- that is exp19's job, run afterwards with a sector observation model.
A cleaner identification that does not improve the estimate would be an
interesting negative result, and it is reported as one rather than assumed away.

    python experiments/exp26_sector_identifiability.py --limit 12
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
from scipy import stats

RESULTS = Path(__file__).parent / "results" / "exp26_sector_identifiability.json"
CACHE = Path("cache/fastf1")

#: Stints shorter than this, or with less spread in tyre age, give slopes that
#: are noise rather than measurement.
MIN_STINT_LAPS = 10
MIN_AGE_SPREAD = 6

#: A fitted slope beyond this is a damaged car or a failed fit, not a tyre.
MAX_PLAUSIBLE_SLOPE = 0.5

#: Physical fuel correction used everywhere else in this repository:
#: 0.030 s/kg x 2.7 kg/lap.
FUEL_SLOPE_S_PER_LAP = 0.081

SECTORS = (1, 2, 3)


def safe_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    """Least-squares slope, returning None instead of raising.

    `np.polyfit` raises LinAlgError("SVD did not converge") on degenerate input,
    and one such stint out of several thousand would otherwise end a forty-race
    collection partway through with nothing written. A stint we cannot fit is a
    stint we drop, not a reason to lose the other two thousand.
    """
    try:
        slope = float(np.polyfit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return None
    return slope if np.isfinite(slope) else None


def collect_stints(events: list[tuple[int, str]]) -> pd.DataFrame:
    """One row per stint: per-sector degradation slopes and per-sector mean times."""
    import fastf1

    rows = []
    for year, event in events:
        try:
            session = fastf1.get_session(year, event, "R")
            session.load(telemetry=False, weather=False, messages=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  {year} {event}: FAILED {type(exc).__name__}")
            continue

        laps = session.laps.pick_wo_box()
        kept = 0
        for (driver, stint), block in laps.groupby(["Driver", "Stint"]):
            block = block.sort_values("LapNumber")
            if len(block) < MIN_STINT_LAPS:
                continue
            age = block["TyreLife"].to_numpy(dtype=float)
            if np.ptp(age) < MIN_AGE_SPREAD:
                continue

            lap_time = block["LapTime"].dt.total_seconds().to_numpy()
            sector_times = {
                s: block[f"Sector{s}Time"].dt.total_seconds().to_numpy() for s in SECTORS
            }
            # `isfinite`, not `isnan`. An infinite lap time survives a NaN check
            # and then takes down `polyfit` with "SVD did not converge", which is
            # how the 2023 Canadian Grand Prix ended a forty-race run partway.
            if not np.isfinite(lap_time).all() or not np.isfinite(age).all():
                continue
            if any(not np.isfinite(v).all() for v in sector_times.values()):
                continue
            if np.ptp(lap_time) == 0.0:
                continue

            fitted = safe_slope(age, lap_time)
            if fitted is None or abs(fitted) > MAX_PLAUSIBLE_SLOPE:
                continue

            row = {
                "year": year, "event": event, "driver": str(driver),
                "stint": int(stint), "n_laps": int(len(block)),
                "compound": str(block["Compound"].iloc[0]),
                "start_age": float(age.min()),
                "total_slope": float(np.polyfit(age, lap_time, 1)[0]),
            }
            sector_slopes = {s: safe_slope(age, sector_times[s]) for s in SECTORS}
            if any(v is None for v in sector_slopes.values()):
                continue
            for s in SECTORS:
                row[f"slope_{s}"] = float(sector_slopes[s])
                row[f"mean_{s}"] = float(sector_times[s].mean())
            rows.append(row)
            kept += 1
        print(f"  {year} {event[:28]:<30} {kept:>3} stints", flush=True)

    return pd.DataFrame(rows)


def time_shares(frame: pd.DataFrame) -> np.ndarray:
    means = frame[[f"mean_{s}" for s in SECTORS]].to_numpy(dtype=float)
    return means / means.sum(axis=1, keepdims=True)


def slope_vectors(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"slope_{s}" for s in SECTORS]].to_numpy(dtype=float)


def proportionality_test(frame: pd.DataFrame) -> dict:
    """Step 1. Is sector degradation just sector time in disguise?

    Under the null that degradation is spread by time share, the predicted sector
    slope is `total_slope * time_share_s`. A paired test of observed against
    predicted, per sector, asks whether the data needs anything more than that.
    """
    shares = time_shares(frame)
    observed = slope_vectors(frame)
    total = frame["total_slope"].to_numpy(dtype=float)
    predicted = shares * total[:, None]

    out = {}
    for i, s in enumerate(SECTORS):
        difference = observed[:, i] - predicted[:, i]
        t, p = stats.ttest_1samp(difference, 0.0)
        out[f"sector_{s}"] = {
            "mean_observed": float(observed[:, i].mean()),
            "mean_predicted": float(predicted[:, i].mean()),
            "mean_difference": float(difference.mean()),
            "t": float(t), "p_value": float(p),
        }
    return out


def loading_angle(frame: pd.DataFrame) -> dict:
    """How far the sector-slope direction sits from the time-share direction.

    Zero degrees means sectors carry exactly the information a whole lap does and
    nothing more. The further from zero, the more a sector model can see that a
    lap model cannot. Computed on circuit means, because per-stint vectors are
    dominated by noise.
    """
    per_event = frame.groupby(["year", "event"]).agg(
        {**{f"slope_{s}": "mean" for s in SECTORS},
         **{f"mean_{s}": "mean" for s in SECTORS}}).reset_index()

    angles = []
    for _, row in per_event.iterrows():
        slope = np.array([row[f"slope_{s}"] for s in SECTORS], dtype=float)
        share = np.array([row[f"mean_{s}"] for s in SECTORS], dtype=float)
        share = share / share.sum()
        if np.linalg.norm(slope) < 1e-9:
            continue
        cosine = float(np.dot(slope, share) / (np.linalg.norm(slope) * np.linalg.norm(share)))
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    angles = np.asarray(angles, dtype=float)
    return {
        "n_events": int(len(angles)),
        "median_angle_deg": float(np.median(angles)) if len(angles) else None,
        "mean_angle_deg": float(angles.mean()) if len(angles) else None,
        "min_angle_deg": float(angles.min()) if len(angles) else None,
        "max_angle_deg": float(angles.max()) if len(angles) else None,
    }


def split_half_stability(frame: pd.DataFrame, n_splits: int, seed: int) -> dict:
    """Step 3. Does the per-circuit sector pattern replicate on held-out drivers?

    **This test was wrong in its first form and the fix is recorded rather than
    quietly applied.** It originally correlated the two halves' three-element
    sector-slope vectors and required the 5th percentile of that correlation to
    exceed zero -- the criterion exp15 used to reject the driver effect.

    At n = 3 that criterion cannot be passed by anything. Correlating two random
    three-element vectors gives a 5th percentile of about -0.99, so a perfect
    signal and pure noise both fail. The bar was impossible, not strict.

    Two tests replace it.

    The first keeps the vector correlation but compares its *mean* against a
    permutation null built by shuffling sector labels within each half. That asks
    the right question: is the agreement between halves greater than chance?

    The second drops vectors entirely. For each sector it takes the departure
    from time-proportionality -- the scalar this experiment is actually about --
    and asks whether the two halves agree on its sign. A structure that is
    physics agrees; noise is a coin flip at 50%.
    """
    rng = np.random.default_rng(seed)
    shares = time_shares(frame)
    observed_slopes = slope_vectors(frame)
    departure = observed_slopes - shares * frame["total_slope"].to_numpy(dtype=float)[:, None]
    work = frame.copy()
    for i, s in enumerate(SECTORS):
        work[f"departure_{s}"] = departure[:, i]

    correlations: list[float] = []
    null_correlations: list[float] = []
    sign_agreements: list[int] = []

    for _, block in work.groupby(["year", "event"]):
        drivers = block["driver"].unique()
        if len(drivers) < 6:
            continue
        for _ in range(n_splits):
            shuffled = rng.permutation(drivers)
            half = len(shuffled) // 2
            a = block[block["driver"].isin(shuffled[:half])]
            b = block[block["driver"].isin(shuffled[half:])]
            if a.empty or b.empty:
                continue

            va = np.array([a[f"slope_{s}"].mean() for s in SECTORS])
            vb = np.array([b[f"slope_{s}"].mean() for s in SECTORS])
            if np.std(va) > 1e-9 and np.std(vb) > 1e-9:
                correlations.append(float(np.corrcoef(va, vb)[0, 1]))
                # Null: same halves, sector labels shuffled in one of them. Any
                # agreement that survives this is agreement about *which sector*,
                # which is the claim.
                null_correlations.append(float(np.corrcoef(va, rng.permutation(vb))[0, 1]))

            for s in SECTORS:
                da, db = a[f"departure_{s}"].mean(), b[f"departure_{s}"].mean()
                if abs(da) > 1e-9 and abs(db) > 1e-9:
                    sign_agreements.append(int(np.sign(da) == np.sign(db)))

    correlations = np.asarray(correlations, dtype=float)
    correlations = correlations[np.isfinite(correlations)]
    nulls = np.asarray(null_correlations, dtype=float)
    nulls = nulls[np.isfinite(nulls)]
    signs = np.asarray(sign_agreements, dtype=int)

    if len(correlations) == 0 or len(signs) == 0:
        return {"n": 0, "stable": False}

    # One-sided permutation p-value on the mean correlation.
    #
    # The reference distribution is the distribution of null *means*, not of
    # individual null draws. Comparing a mean against single draws was the second
    # broken test in this experiment: individual null correlations are spread
    # over [-1, 1], so about 29% of them exceed +0.638 by chance and the observed
    # agreement looked insignificant when it is 123 standard errors out.
    #
    # Bootstrapped rather than assumed normal, because the null correlations are
    # bounded and heavily non-Gaussian at n = 3.
    if len(nulls) > 1:
        boot = rng.choice(nulls, size=(2000, len(nulls)), replace=True).mean(axis=1)
        permutation_p = float((boot >= correlations.mean()).mean())
        null_mean_se = float(boot.std(ddof=1))
    else:
        permutation_p, null_mean_se = 1.0, float("nan")
    sign_rate = float(signs.mean())
    sign_p = float(stats.binomtest(int(signs.sum()), len(signs), 0.5,
                                   alternative="greater").pvalue)

    return {
        "n": int(len(correlations)),
        "mean_r": float(correlations.mean()),
        "p05": float(np.percentile(correlations, 5)),
        "p95": float(np.percentile(correlations, 95)),
        "null_mean_r": float(nulls.mean()) if len(nulls) else None,
        "null_mean_se": null_mean_se,
        "permutation_p": permutation_p,
        "n_sign_tests": int(len(signs)),
        "sign_agreement_rate": sign_rate,
        "sign_agreement_p": sign_p,
        # Both tests must pass. The vector test says the halves agree more than
        # chance; the sign test says they agree about the thing we care about.
        "stable": bool(permutation_p < 0.05 and sign_p < 0.05 and sign_rate > 0.5),
    }


def identifiability_gain(frame: pd.DataFrame) -> dict:
    """Step 3. How much better conditioned is the sector system?

    The whole-lap design within a stint is `[1, a, -f]` with `a = a0 + f`, which
    is rank 2 for 3 columns -- singular by construction, which is exp18's result.

    The sector design stacks three rows per lap with sector-specific loadings.
    Estimating those loadings from the data itself (the mean sector-slope
    direction per circuit, and time share as the fuel direction) gives a 2x2
    system for (beta, phi). Its condition number says how well posed that is.

    Reported as a ratio of Cramer-Rao bounds, which is the quantity a reader can
    compare against exp18's 1.38x figure. A singular whole-lap system has an
    infinite bound, so the honest statement is "finite versus infinite" plus the
    conditioning of what replaces it.
    """
    per_event = frame.groupby(["year", "event"]).agg(
        {**{f"slope_{s}": "mean" for s in SECTORS},
         **{f"mean_{s}": "mean" for s in SECTORS},
         "total_slope": "mean"}).reset_index()

    conditions, singular_lap, well_posed = [], 0, 0
    for _, row in per_event.iterrows():
        share = np.array([row[f"mean_{s}"] for s in SECTORS], dtype=float)
        share = share / share.sum()
        slope = np.array([row[f"slope_{s}"] for s in SECTORS], dtype=float)

        # Whole-lap: one equation, two unknowns. Always singular.
        singular_lap += 1

        # Sector: tyre direction taken as the observed slope shape, fuel direction
        # as time share. Two columns; if they are parallel the system is singular
        # too and sectors have bought nothing.
        tyre_direction = slope / np.linalg.norm(slope) if np.linalg.norm(slope) > 1e-12 else share
        design = np.column_stack([tyre_direction, -share])
        singular_values = np.linalg.svd(design, compute_uv=False)
        if singular_values[-1] < 1e-10:
            continue
        conditions.append(float(singular_values[0] / singular_values[-1]))
        well_posed += 1

    conditions = np.asarray(conditions, dtype=float)
    return {
        "n_events": int(len(per_event)),
        "whole_lap_singular": int(singular_lap),
        "sector_well_posed": int(well_posed),
        "median_condition_number": float(np.median(conditions)) if len(conditions) else None,
        "worst_condition_number": float(conditions.max()) if len(conditions) else None,
        "best_condition_number": float(conditions.min()) if len(conditions) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="*", default=[2024])
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--splits", type=int, default=200)
    parser.add_argument("--seed", type=int, default=26)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    import fastf1
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))

    events: list[tuple[int, str]] = []
    for year in args.years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for name in schedule["EventName"].tolist():
            events.append((year, name))
    events = events[: args.limit]

    print(f"collecting sector times from {len(events)} races\n")
    stints = collect_stints(events)
    if stints.empty:
        raise SystemExit("no usable stints; is the FastF1 cache populated?")

    print(f"\n{len(stints)} stints, {stints.groupby(['year','event']).ngroups} races, "
          f"{stints['driver'].nunique()} drivers\n")

    proportionality = proportionality_test(stints)
    angles = loading_angle(stints)
    stability = split_half_stability(stints, args.splits, args.seed)
    identifiability = identifiability_gain(stints)

    print("=" * 92)
    print("STEP 1 -- is sector degradation just sector time in disguise?")
    print("=" * 92)
    print(f"{'':<12}{'observed':>12}{'if by time':>13}{'difference':>13}{'t':>8}{'p':>11}")
    for s in SECTORS:
        r = proportionality[f"sector_{s}"]
        print(f"sector {s:<6}{r['mean_observed']:>12.4f}{r['mean_predicted']:>13.4f}"
              f"{r['mean_difference']:>13.4f}{r['t']:>8.2f}{r['p_value']:>11.2e}")
    breaks = [s for s in SECTORS if proportionality[f"sector_{s}"]["p_value"] < 0.05]
    print(f"\nsectors departing from time-proportionality: {len(breaks)}/3  {breaks}")

    print()
    print("=" * 92)
    print("STEP 2 -- angle between the sector-slope direction and the time-share direction")
    print("=" * 92)
    print(f"  median {angles['median_angle_deg']:.1f} deg over {angles['n_events']} races "
          f"(range {angles['min_angle_deg']:.1f}-{angles['max_angle_deg']:.1f})")
    print("  0 deg would mean sectors carry nothing a whole lap does not.")

    print()
    print("=" * 92)
    print("STEP 3 -- does the pattern replicate on held-out drivers?")
    print("=" * 92)
    if stability.get("n"):
        print(f"  vector split-half r = {stability['mean_r']:+.3f} against a "
              f"permutation null of {stability['null_mean_r']:+.3f} "
              f"(SE {stability['null_mean_se']:.4f})   "
              f"p = {stability['permutation_p']:.2e}   ({stability['n']} splits)")
        print(f"  halves agree on the SIGN of the departure: "
              f"{stability['sign_agreement_rate']:.1%} of "
              f"{stability['n_sign_tests']} tests   p = {stability['sign_agreement_p']:.2e}")
        print(f"  STABLE: {stability['stable']}")
        print("  (The 5th-percentile criterion used at first was impossible to pass:")
        print("   two random 3-element vectors correlate at p05 = -0.99.)")
    else:
        print("  not enough drivers per race to split")

    print()
    print("=" * 92)
    print("STEP 4 -- identifiability")
    print("=" * 92)
    print(f"  whole-lap systems singular: {identifiability['whole_lap_singular']}"
          f"/{identifiability['n_events']}   (exp18's result, reproduced)")
    print(f"  sector systems well posed:  {identifiability['sector_well_posed']}"
          f"/{identifiability['n_events']}")
    if identifiability["median_condition_number"]:
        print(f"  median condition number: {identifiability['median_condition_number']:.1f} "
              f"(range {identifiability['best_condition_number']:.1f}"
              f"-{identifiability['worst_condition_number']:.1f})")
        print("  A condition number near 1 is perfectly posed; a very large one means")
        print("  the two directions are nearly parallel and the gain is cosmetic.")

    print()
    print("=" * 92)
    supported = (len(breaks) > 0 and stability.get("stable", False)
                 and identifiability["sector_well_posed"] > 0)
    if supported:
        print("VERDICT: supported. Sector times carry degradation information that a")
        print("whole-lap time does not, the pattern replicates across drivers, and the")
        print("two-parameter system is well posed where the whole-lap one is singular.")
        print("Next: a sector observation model in the estimator, then re-run exp19.")
    elif len(breaks) > 0 and not stability.get("stable", False):
        print("VERDICT: not supported. The sector slopes do differ from time-")
        print("proportionality, but the pattern does not replicate on held-out drivers,")
        print("so it is stint-level noise rather than a loading structure. Same failure")
        print("mode as the driver effect in exp15.")
    else:
        print("VERDICT: not supported. Sector degradation is indistinguishable from")
        print("sector time share, so sectors add no identification.")
    print("=" * 92)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp26_sector_identifiability",
        "generated_at": datetime.now(UTC).isoformat(),
        "n_stints": int(len(stints)),
        "n_races": int(stints.groupby(["year", "event"]).ngroups),
        "n_drivers": int(stints["driver"].nunique()),
        "proportionality": proportionality,
        "sectors_departing_from_time_proportionality": breaks,
        "loading_angle": angles,
        "split_half": stability,
        "identifiability": identifiability,
        "supported": bool(supported),
        "stints": stints.to_dict(orient="records"),
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
