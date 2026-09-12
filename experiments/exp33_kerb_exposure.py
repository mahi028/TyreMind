"""exp33 -- does riding the kerbs cost tyre life, and can public data even see it?

A judge asked what happens to a tyre when a car hits the kerbs. TyreMind had no
answer: nothing in the project models a kerb strike, so a big one becomes an
outlier lap the MAD filter removes, or it ends the stint.

`PREREGISTRATION_exp33.md` fixes everything below before any of it was run. Two
proxies, because no public channel sees a kerb:

  A  lateral deviation from the session's fastest lap's racing line, per lap,
     summarised as the fraction of lap distance beyond a threshold
  B  laps race control deleted for exceeding track limits, with the corner

and, first, a **measurement-validity gate** on A. The gate exists because a
five-session pilot found something worse than noise: at the 2024 Austrian Grand
Prix every one of the 190 pairs of cars reaches a minimum separation of 0.00 m.
Two Formula 1 cars cannot occupy the same point. If that holds corpus-wide, the
public position feed reports distance along a single path and has thrown the
lateral coordinate away, and Measure A is not a weak measurement of kerb use --
it is not a measurement of anything.

A failed gate does not end the experiment. It promotes Measure A to a **negative
control**: a quantity that provably carries no information about where the car
was, run through the same contrasts as Measure B. If the control comes out
significant, the contrasts are not removing what they are supposed to remove and
every other number here is worth less.

    python experiments/exp33_kerb_exposure.py

Writes experiments/results/exp33_kerb_exposure.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import statistics as st
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tyremind.data.corpus import read_lap_table, read_telemetry_table

RESULTS = Path(__file__).parent / "results" / "exp33_kerb_exposure.json"
KERB_DIR = Path("data/kerb")
DIAGNOSTIC_DIR = KERB_DIR / "diagnostics"
SEASON_DIR = Path("data/season")

#: Physical fuel correction, matching the rest of the ladder. Without it the
#: measured slope is (degradation - fuel) and frequently negative, which is the
#: failure exp14 is about.
FUEL_SLOPE_S_PER_LAP = 0.081

#: Slopes beyond this are a damaged car, a wet patch or a failed fit, not a tyre.
MAX_PLAUSIBLE_SLOPE = 0.5

#: Pre-registered stint filters. Stricter than exp25's 8/4 because a slope fitted
#: from ten lap times over six laps of age is the weakest thing this experiment is
#: willing to call a measurement.
MIN_STINT_LAPS = 10
MIN_AGE_SPREAD = 6

#: The lateral thresholds `scripts/build_kerb_exposure.py` measured. 2.0 m is the
#: pre-registered headline; the rest are the declared sensitivity sweep.
THRESHOLDS = ("1m0", "1m5", "2m0", "2m5", "3m0", "4m0")
HEADLINE_THRESHOLD = "2m0"

#: Pre-registered gate on Measure A. See PREREGISTRATION_exp33.md section 4.
GATE_MIN_P99_M = 2.0
GATE_STABILITY_MULTIPLE = 5.0
GATE_MAX_COINCIDENT_FRACTION = 0.25

#: A (driver, circuit) cell needs this many stints before a within-cell contrast
#: means anything, and T4 needs this many cells before it is powered at all.
MIN_CELL_STINTS = 3
MIN_T4_PAIRS = 20

#: Bonferroni denominator for Measure A: six thresholds by two contrasts.
MEASURE_A_TESTS = 12


def min_detectable_rho(n: int) -> float:
    """Smallest Spearman correlation detectable at 80% power, alpha 0.05.

    Reported next to every null so that "no effect" reads as "no effect larger
    than this", which is the only honest form of a negative correlation result.
    """
    return float(2.8 / np.sqrt(n - 3)) if n > 4 else float("nan")


def spearman(x: pd.Series, y: pd.Series) -> dict:
    """Spearman rho with its n and its detection floor, or a declared nothing."""
    block = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(block) < 10 or block["x"].nunique() < 3:
        return {"n": int(len(block)), "rho": None, "p_value": None,
                "min_detectable_rho": min_detectable_rho(len(block))}
    rho, p = stats.spearmanr(block["x"], block["y"])
    if not np.isfinite(rho):
        return {"n": int(len(block)), "rho": None, "p_value": None,
                "min_detectable_rho": min_detectable_rho(len(block))}
    return {"n": int(len(block)), "rho": float(rho), "p_value": float(p),
            "min_detectable_rho": min_detectable_rho(len(block))}


def demean(frame: pd.DataFrame, by: list[str], columns: list[str],
           min_size: int = 1) -> pd.DataFrame:
    """Subtract each group's mean, dropping groups too small to have one.

    This is how the circuit and driving-style confounds are handled. Kerb
    exposure is not randomly assigned: some circuits have more kerbed corners and
    more monitored ones, and some drivers use them more than others. A pooled
    correlation would read either of those as an effect of kerbs on tyres.
    """
    block = frame.groupby(by).filter(lambda g: len(g) >= min_size).copy()
    if block.empty:
        return block
    for column in columns:
        block[column] = block[column] - block.groupby(by)[column].transform("mean")
    return block


def measurement_gate() -> dict:
    """Can the position feed see a car off the racing line at all?

    Three conditions, fixed in the pre-registration before this ran. The third is
    the one that does not depend on any modelling choice of ours: two cars at the
    same coordinate is not a hard measurement, it is an impossible one.
    """
    sessions = []
    for path in sorted(DIAGNOSTIC_DIR.glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry["session_id"].endswith("-R"):
            sessions.append(entry)
    if not sessions:
        raise SystemExit("no diagnostics; run scripts/build_kerb_exposure.py first")

    frame = pd.DataFrame(sessions)
    p99 = float(frame["dev_p99_m"].median())
    stability = float(frame["reference_stability_m"].median())
    coincident = frame.get("n_pairs_exactly_coincident", pd.Series(dtype=float)).fillna(0)
    coincident_fraction = float((coincident > 0).mean())

    gates = {
        "G1_p99_at_least_2m": bool(p99 >= GATE_MIN_P99_M),
        "G2_p99_exceeds_5x_reference_stability": bool(p99 >= GATE_STABILITY_MULTIPLE * stability),
        "G3_cars_not_coincident": bool(coincident_fraction < GATE_MAX_COINCIDENT_FRACTION),
    }
    return {
        "n_race_sessions": int(len(frame)),
        "median_dev_median_m": float(frame["dev_median_m"].median()),
        "median_dev_p90_m": float(frame["dev_p90_m"].median()),
        "median_dev_p99_m": p99,
        "median_dev_p999_m": float(frame["dev_p999_m"].median()),
        "median_dev_max_m": float(frame["dev_max_m"].median()),
        "median_reference_stability_m": stability,
        "median_pair_min_separation_m": float(frame["pair_min_separation_m"].median()),
        "fraction_of_sessions_with_coincident_cars": coincident_fraction,
        "mean_pairs_within_2m_per_session": float(frame["n_pairs_within_2m"].mean()),
        "mean_pairs_per_session": float(frame["n_car_pairs"].mean()),
        # The single most decisive number in this experiment: how many of all the
        # car pairs across every race in the corpus are reported, at some moment,
        # at the same coordinate.
        "n_coincident_pairs": int(coincident.sum()),
        "n_pairs_total": int(frame["n_car_pairs"].sum()),
        "gates": gates,
        "measure_a_valid": bool(all(gates.values())),
    }


def build_stints() -> pd.DataFrame:
    """One row per race stint: its fuel-corrected slope and its kerb exposure."""
    rows = []
    for path in sorted(glob.glob(str(KERB_DIR / "*-R.parquet"))):
        session_id = Path(path).stem
        lap_path = SEASON_DIR / f"{session_id}.parquet"
        if not lap_path.exists():
            continue

        # `read_telemetry_table` rather than a bare read: it is this project's
        # reader for a per-lap derived table keyed on (driver, session_lap), which
        # is exactly the shape of the kerb table, and tests/unit/
        # test_loader_discipline.py forbids direct parquet reads for a reason worth
        # keeping absolute. The fuel repair applies to the lap table, which does
        # go through read_lap_table below.
        kerb = read_telemetry_table(Path(path))
        merged = read_lap_table(lap_path).merge(
            kerb, on=["driver", "session_lap"], how="inner")
        if merged.empty:
            continue

        # Pre-registered lap filters. All three are off the racing line for
        # reasons that have nothing to do with kerbs: the pit lane is a different
        # road, lap 1 is a standing start into a first corner nobody takes on the
        # line, and a car behind a safety car is not racing.
        merged = merged[~merged["is_pit_lap"].astype(bool)]
        merged = merged[merged["track_status"].astype(str) == "1"]
        merged = merged[merged["session_lap"] > 1]
        if merged.empty:
            continue

        circuit = session_id.split("-", 1)[1].rsplit("-", 1)[0]
        year = int(session_id.split("-", 1)[0])

        for (driver, run_id), stint in merged.groupby(["driver", "run_id"]):
            if len(stint) < MIN_STINT_LAPS:
                continue
            age = stint["tyre_age"].to_numpy(dtype=float)
            corrected = (stint["lap_time"].to_numpy(dtype=float)
                         + FUEL_SLOPE_S_PER_LAP * stint["lap_in_run"].to_numpy(dtype=float))
            usable = np.isfinite(age) & np.isfinite(corrected)
            if usable.sum() < MIN_STINT_LAPS or np.ptp(age[usable]) < MIN_AGE_SPREAD:
                continue
            try:
                slope = float(np.polyfit(age[usable], corrected[usable], 1)[0])
            except (np.linalg.LinAlgError, ValueError):
                # A singular design has ended two runs in this project. It is a
                # skipped stint, not a crashed experiment.
                continue
            if not np.isfinite(slope) or abs(slope) > MAX_PLAUSIBLE_SLOPE:
                continue

            flagged = stint["limits_event"].astype(bool).to_numpy()
            row = {
                "session": session_id, "circuit": circuit, "year": year,
                "driver": str(driver), "run_id": int(run_id), "n_laps": int(len(stint)),
                "compound": str(stint["compound"].iloc[0]), "slope": slope,
                "limits_rate": float(flagged.mean()),
                "n_limits": int(flagged.sum()),
                "any_limits": bool(flagged.any()),
                "mean_abs_dev_m": float(stint["mean_abs_dev_m"].mean()),
                "max_abs_dev_m": float(stint["max_abs_dev_m"].max()),
                "reference_stability_m": float(stint["reference_stability_m"].iloc[0]),
            }
            for threshold in THRESHOLDS:
                column = f"exposure_frac_{threshold}"
                row[f"exposure_{threshold}"] = (
                    float(stint[column].mean()) if column in stint else np.nan)
            # Where in the stint the excursions happen. If they cluster early, a
            # deleted (and therefore quick) lap early in a stint flattens the
            # fitted slope; if late, it steepens it. Limitation 4 of the
            # pre-registration needs this number to be checkable.
            if flagged.any():
                position = stint["lap_in_run"].to_numpy(dtype=float)
                row["mean_excursion_lap_in_run"] = float(np.nanmean(position[flagged]))
                row["mean_stint_lap_in_run"] = float(np.nanmean(position))
            rows.append(row)
    return pd.DataFrame(rows)


def measure_b(stints: pd.DataFrame) -> dict:
    """T1 to T4: track-limits excursions against the degradation slope."""
    out: dict = {}
    out["T1_pooled"] = spearman(stints["limits_rate"], stints["slope"])

    by_circuit = demean(stints, ["circuit"], ["limits_rate", "slope"])
    out["T2_within_circuit"] = spearman(by_circuit["limits_rate"], by_circuit["slope"])

    by_cell = demean(stints, ["circuit", "driver"], ["limits_rate", "slope"],
                     min_size=MIN_CELL_STINTS)
    out["T3_within_driver_and_circuit"] = spearman(by_cell["limits_rate"], by_cell["slope"])
    out["T3_n_cells"] = int(by_cell.groupby(["circuit", "driver"]).ngroups) if len(by_cell) else 0

    # T4: the paired contrast. Within one driver at one circuit, compare the
    # stints where they went off against the stints where they did not. This is
    # the cleanest form of the question available -- same driver, same track, same
    # car -- and it is also the one with the least data.
    differences, pairs = [], []
    for (circuit, driver), cell in stints.groupby(["circuit", "driver"]):
        wide = cell[cell["any_limits"]]
        clean = cell[~cell["any_limits"]]
        if wide.empty or clean.empty:
            continue
        difference = float(wide["slope"].mean() - clean["slope"].mean())
        if not np.isfinite(difference):
            continue
        differences.append(difference)
        pairs.append({"circuit": circuit, "driver": driver,
                      "n_wide": int(len(wide)), "n_clean": int(len(clean)),
                      "difference_s_per_lap2": difference})

    t4: dict = {"n_pairs": len(differences), "underpowered": len(differences) < MIN_T4_PAIRS}
    if differences:
        t4["mean_difference_s_per_lap2"] = float(np.mean(differences))
        t4["median_difference_s_per_lap2"] = float(np.median(differences))
        t4["fraction_positive"] = float(np.mean(np.array(differences) > 0))
    if len(differences) > 2:
        # A null needs a size attached or it says nothing. This interval is what
        # the experiment can rule out: an effect outside it would have shown.
        error = float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
        t4["standard_error"] = error
        t4["ci95_s_per_lap2"] = [float(np.mean(differences) - 1.96 * error),
                                 float(np.mean(differences) + 1.96 * error)]
    if len(differences) >= 6 and np.ptp(differences) > 0:
        try:
            statistic, p_value = stats.wilcoxon(differences)
        except ValueError:
            # All-zero differences, which scipy refuses rather than calling a tie.
            t4["wilcoxon_error"] = "degenerate"
        else:
            t4 |= {"wilcoxon_statistic": float(statistic), "p_value": float(p_value)}
    t4["pairs"] = sorted(pairs, key=lambda r: -abs(r["difference_s_per_lap2"]))[:25]
    out["T4_paired_within_driver_and_circuit"] = t4
    return out


def leave_one_circuit_out(stints: pd.DataFrame, features: list[str]) -> dict:
    """T5 -- does exposure predict a circuit the model has never seen?

    The baseline is the compound label's mean rate, which exp08 found surprisingly
    hard to beat because Pirelli nominates compounds to equalise degradation and
    the label therefore already carries severity. A feature that cannot beat it
    out of sample is not adding information, whatever its in-sample correlation.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    cases = []
    for held in sorted(stints["circuit"].unique()):
        train = stints[stints["circuit"] != held]
        test = stints[stints["circuit"] == held]
        if len(train) < 50 or test.empty:
            continue

        global_mean = float(train["slope"].mean())
        compound_means = train.groupby("compound")["slope"].mean().to_dict()

        design = pd.get_dummies(train["compound"], prefix="compound", dtype=float)
        columns = list(design.columns)
        x_train = np.column_stack([design.to_numpy(float),
                                   train[features].to_numpy(float)])
        keep = np.isfinite(x_train).all(axis=1)
        if keep.sum() < 50:
            continue
        scaler = StandardScaler().fit(x_train[keep])
        model = Ridge(alpha=5.0).fit(scaler.transform(x_train[keep]),
                                     train["slope"].to_numpy(float)[keep])

        test_design = pd.get_dummies(test["compound"], prefix="compound", dtype=float)
        test_design = test_design.reindex(columns=columns, fill_value=0.0)
        x_test = np.column_stack([test_design.to_numpy(float),
                                  test[features].to_numpy(float)])
        finite = np.isfinite(x_test).all(axis=1)
        if not finite.any():
            continue
        predicted = model.predict(scaler.transform(x_test[finite]))

        for i, (_, row) in enumerate(test[finite].iterrows()):
            cases.append({
                "circuit": held, "actual": float(row["slope"]),
                "compound_mean": float(compound_means.get(row["compound"], global_mean)),
                "compound_plus_exposure": float(predicted[i]),
            })

    if len(cases) < 50:
        return {"n": len(cases), "conclusive": False}

    baseline = [abs(c["compound_mean"] - c["actual"]) for c in cases]
    augmented = [abs(c["compound_plus_exposure"] - c["actual"]) for c in cases]
    try:
        _, p_value = stats.wilcoxon(augmented, baseline)
    except ValueError:
        return {"n": len(cases), "conclusive": False, "reason": "identical predictors"}
    return {
        "n": len(cases),
        "n_circuits": len({c["circuit"] for c in cases}),
        "features": features,
        "baseline_mae": st.fmean(baseline),
        "augmented_mae": st.fmean(augmented),
        "improvement_pct": 100.0 * (st.fmean(baseline) - st.fmean(augmented)) / st.fmean(baseline),
        "wilcoxon_p": float(p_value),
        "helps": bool(st.fmean(augmented) < st.fmean(baseline) and p_value < 0.05),
        "hurts": bool(st.fmean(augmented) > st.fmean(baseline) and p_value < 0.05),
        "conclusive": True,
    }


def measure_a_control(stints: pd.DataFrame) -> dict:
    """The lateral-deviation sweep, in whichever role the gate assigns it.

    If the gate passed this is the primary analysis. If the gate failed it is a
    negative control on a quantity known to carry no information, and every entry
    here is expected to be null. A significant result would indict the design, not
    vindicate the measure.
    """
    out: dict = {}
    for threshold in THRESHOLDS:
        column = f"exposure_{threshold}"
        by_circuit = demean(stints, ["circuit"], [column, "slope"])
        by_cell = demean(stints, ["circuit", "driver"], [column, "slope"],
                         min_size=MIN_CELL_STINTS)
        out[threshold] = {
            "pooled": spearman(stints[column], stints["slope"]),
            "within_circuit": spearman(by_circuit[column], by_circuit["slope"]),
            "within_driver_and_circuit": spearman(by_cell[column], by_cell["slope"]),
            "mean_stint_exposure_frac": float(stints[column].mean()),
        }
    return out


def where_do_they_go_off() -> dict:
    """The descriptive answer the judge's question deserves regardless of H1.

    Which corners cars are actually caught leaving, and how often. This is a
    stewards' record, not an inference, and it stands whether or not the
    degradation link survives.
    """
    totals: dict[str, int] = {}
    corners: dict[str, dict[str, int]] = {}
    per_race = []
    for path in sorted(DIAGNOSTIC_DIR.glob("*-R.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        circuit = entry["circuit"]
        events = int(entry.get("n_limits_events", 0))
        totals[circuit] = totals.get(circuit, 0) + events
        histogram = corners.setdefault(circuit, {})
        for corner, count in entry.get("limits_by_corner", {}).items():
            histogram[corner] = histogram.get(corner, 0) + int(count)
        per_race.append({"session": entry["session_id"], "circuit": circuit,
                         "n_events": events, "n_laps": int(entry.get("n_laps", 0)),
                         "by_corner": entry.get("limits_by_corner", {})})

    ordered = sorted(totals.items(), key=lambda kv: -kv[1])
    return {
        "events_by_circuit": dict(ordered),
        # The concentration is the substance of the descriptive answer: if a
        # circuit's excursions are nearly all at one corner, "riding the kerbs" at
        # that venue is one specific piece of geometry rather than a driving style.
        "corners_by_circuit": {
            circuit: dict(sorted(corners.get(circuit, {}).items(), key=lambda kv: -kv[1]))
            for circuit, _ in ordered},
        "total_events": int(sum(totals.values())),
        "races_with_no_events": int(sum(1 for r in per_race if r["n_events"] == 0)),
        "n_races": len(per_race),
        "busiest_races": sorted(per_race, key=lambda r: -r["n_events"])[:10],
    }


def verdict(gate: dict, b: dict, t5: dict) -> tuple[str, list[str]]:
    """The pre-registered decision rule, applied without discretion."""
    alpha = 0.05
    t3 = b["T3_within_driver_and_circuit"]
    t4 = b["T4_paired_within_driver_and_circuit"]

    t3_ok = (t3["rho"] is not None and t3["rho"] > 0 and t3["p_value"] < alpha)
    t4_ok = (not t4["underpowered"] and t4.get("p_value") is not None
             and t4["p_value"] < alpha and t4.get("mean_difference_s_per_lap2", 0) > 0)
    t5_ok = bool(t5.get("helps"))

    lines = []
    if not gate["measure_a_valid"]:
        failed = [k for k, v in gate["gates"].items() if not v]
        lines.append(
            "MEASUREMENT: Measure A fails its pre-registered validity gate "
            f"({', '.join(failed)}). The public position feed reports on-track "
            "position along a single path; the lateral coordinate is not in it. "
            "Lateral deviation from the racing line cannot be measured from public "
            "F1 data, and the sweep below is a negative control, not a result.")

    if t3_ok and t4_ok and t5_ok:
        lines.append("H1 SUPPORTED: kerb exposure predicts degradation on all three "
                     "pre-registered tests. This would be the first public-data "
                     "measurement of the effect and needs replication before it is "
                     "believed.")
        return "supported", lines

    if t3["rho"] is not None and t3["rho"] < 0 and t3["p_value"] < alpha:
        lines.append("H1 REFUTED WITH THE WRONG SIGN: within driver and circuit, more "
                     "track-limits excursions go with LESS degradation. That is "
                     "backwards for a wear mechanism and is the exp25 pattern -- a "
                     "selection effect, not physics. Drivers run wide when they are "
                     "pushing, and they push when the tyre is good.")
        return "refuted_wrong_sign", lines

    lines.append("H1 NULL: kerb exposure, as public data can measure it, does not "
                 "predict the degradation slope. Reported as a null and not as a "
                 "trend, per the pre-registration.")
    return "null", lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    gate = measurement_gate()
    print("MEASUREMENT-VALIDITY GATE (Measure A -- lateral deviation)")
    print(f"  {gate['n_race_sessions']} race sessions")
    print(f"  lateral deviation of a moving on-track car from the racing line:")
    print(f"    median {gate['median_dev_median_m']:.3f} m   p90 {gate['median_dev_p90_m']:.3f} m"
          f"   p99 {gate['median_dev_p99_m']:.3f} m   p99.9 {gate['median_dev_p999_m']:.3f} m")
    print(f"  reference stability (two fast drivers' lines): "
          f"{gate['median_reference_stability_m']:.3f} m")
    print(f"  closest any two cars come, per session:        "
          f"{gate['median_pair_min_separation_m']:.3f} m")
    print(f"  car pairs reported at the same coordinate:     "
          f"{gate['n_coincident_pairs']} of {gate['n_pairs_total']} "
          f"({100 * gate['n_coincident_pairs'] / max(gate['n_pairs_total'], 1):.1f}%) "
          f"in {100 * gate['fraction_of_sessions_with_coincident_cars']:.0f}% of races")
    for name, passed in gate["gates"].items():
        print(f"    {'PASS' if passed else 'FAIL'}  {name}")
    print(f"  -> Measure A is {'VALID' if gate['measure_a_valid'] else 'NOT MEASURABLE'}\n")

    stints = build_stints()
    if stints.empty:
        raise SystemExit("no joinable stints; run scripts/build_kerb_exposure.py first")

    print(f"{len(stints)} race stints, {stints['circuit'].nunique()} circuits, "
          f"{stints['driver'].nunique()} drivers, {stints['session'].nunique()} races")
    print(f"  stints containing a track-limits excursion: "
          f"{int(stints['any_limits'].sum())} ({100 * stints['any_limits'].mean():.1f}%)")
    print(f"  total excursions: {int(stints['n_limits'].sum())}")
    print(f"  mean fuel-corrected slope: {stints['slope'].mean():+.4f} s/lap per lap\n")

    b = measure_b(stints)
    print("MEASURE B -- track-limits excursions (adjudicated)")
    print(f"{'test':<34}{'n':>7}{'rho':>9}{'p':>11}{'detectable rho':>16}")
    for key in ("T1_pooled", "T2_within_circuit", "T3_within_driver_and_circuit"):
        r = b[key]
        rho = f"{r['rho']:+.4f}" if r["rho"] is not None else "--"
        p = f"{r['p_value']:.4f}" if r["p_value"] is not None else "--"
        print(f"{key:<34}{r['n']:>7}{rho:>9}{p:>11}{r['min_detectable_rho']:>16.3f}")

    t4 = b["T4_paired_within_driver_and_circuit"]
    print(f"\nT4 paired within (driver, circuit): {t4['n_pairs']} pairs"
          f"{'  -- UNDERPOWERED' if t4['underpowered'] else ''}")
    if "mean_difference_s_per_lap2" in t4:
        print(f"  mean slope difference (wide stints - clean stints): "
              f"{t4['mean_difference_s_per_lap2']:+.4f} s/lap per lap")
        print(f"  positive in {100 * t4['fraction_positive']:.0f}% of pairs"
              + (f", Wilcoxon p = {t4['p_value']:.4f}" if "p_value" in t4 else ""))
    if "ci95_s_per_lap2" in t4:
        low, high = t4["ci95_s_per_lap2"]
        print(f"  95% interval [{low:+.4f}, {high:+.4f}] -- an effect outside this "
              "would have been visible")

    t5 = leave_one_circuit_out(stints, ["limits_rate"])
    print("\nT5 leave-one-circuit-out (compound label vs compound + excursion rate)")
    if t5.get("conclusive"):
        print(f"  {t5['n']} held-out stints over {t5['n_circuits']} circuits")
        print(f"  compound-label MAE {t5['baseline_mae']:.4f}  ->  "
              f"with exposure {t5['augmented_mae']:.4f}   "
              f"({t5['improvement_pct']:+.1f}%, Wilcoxon p = {t5['wilcoxon_p']:.4f})")
    else:
        print(f"  only {t5['n']} scored cases -- not conclusive")

    control = measure_a_control(stints)
    role = "PRIMARY" if gate["measure_a_valid"] else "NEGATIVE CONTROL"
    bonferroni = args.alpha / MEASURE_A_TESTS
    print(f"\nMEASURE A -- lateral deviation, threshold sweep [{role}]")
    print(f"  significance judged at Bonferroni alpha = {bonferroni:.4f}")
    print(f"{'threshold':<12}{'mean frac':>12}{'circuit rho':>14}{'p':>10}"
          f"{'cell rho':>11}{'p':>10}")
    # Bonferroni counts the two within-group contrasts at each threshold, which is
    # the twelve the pre-registration declared. The pooled correlation is computed
    # and stored but not counted, because it was never going to carry a verdict --
    # it is the one exp25 showed can be significant and backwards.
    survivors = []
    for threshold in THRESHOLDS:
        entry = control[threshold]
        c, d = entry["within_circuit"], entry["within_driver_and_circuit"]
        c_rho = f"{c['rho']:+.4f}" if c["rho"] is not None else "--"
        c_p = f"{c['p_value']:.4f}" if c["p_value"] is not None else "--"
        d_rho = f"{d['rho']:+.4f}" if d["rho"] is not None else "--"
        d_p = f"{d['p_value']:.4f}" if d["p_value"] is not None else "--"
        headline = "  <- headline" if threshold == HEADLINE_THRESHOLD else ""
        print(f"{threshold:<12}{entry['mean_stint_exposure_frac']:>12.6f}"
              f"{c_rho:>14}{c_p:>10}{d_rho:>11}{d_p:>10}{headline}")
        survivors += [f"{threshold}/{name}" for name, r in
                      (("circuit", c), ("cell", d))
                      if r["p_value"] is not None and r["p_value"] < bonferroni]
    if survivors:
        print(f"  survives Bonferroni: {', '.join(survivors)}")
        if not gate["measure_a_valid"]:
            print("  WARNING: the negative control is not null. The within-group "
                  "contrasts are not removing what they are supposed to remove, and "
                  "every Measure B result above must be discounted accordingly.")
    else:
        print("  nothing survives Bonferroni"
              + ("" if gate["measure_a_valid"] else " -- the control behaves as a control should"))

    geography = where_do_they_go_off()
    retained = int(stints["n_limits"].sum())
    geography["n_events_on_analysed_laps"] = retained
    geography["retention"] = retained / geography["total_events"] if geography["total_events"] else None
    print(f"\nWHERE CARS ACTUALLY LEAVE THE TRACK ({geography['total_events']} adjudicated "
          f"excursions over {geography['n_races']} races, "
          f"{geography['races_with_no_events']} with none)")
    # How much of the evidence the analysis actually sees. The losses are pit and
    # safety-car laps, lap 1, and stints too short to fit a slope to -- not the
    # MAD filter, which targets slow outliers while a lap deleted for track limits
    # is usually a quick one.
    print(f"  {retained} of those ({100 * retained / max(geography['total_events'], 1):.0f}%) "
          f"fall on laps inside a stint the slope fit could use")
    for circuit, count in list(geography["events_by_circuit"].items())[:10]:
        top = list(geography["corners_by_circuit"].get(circuit, {}).items())[:3]
        where = ", ".join(f"T{corner} x{n}" for corner, n in top) or "--"
        print(f"  {circuit:<22}{count:>5}   {where}")

    positions = stints.dropna(subset=["mean_excursion_lap_in_run"]) \
        if "mean_excursion_lap_in_run" in stints else stints.iloc[0:0]
    if len(positions):
        # Pre-registration limitation 4: a deleted lap is usually a quick lap, so
        # where in the stint the excursions fall decides which way that biases a
        # fitted slope. Early excursions flatten it, late ones steepen it.
        at = float(positions["mean_excursion_lap_in_run"].mean())
        mid = float(positions["mean_stint_lap_in_run"].mean())
        direction = "later, which steepens" if at > mid else "earlier, which flattens"
        print(f"\n  excursions fall at lap {at:.1f} of a stint whose mean lap is "
              f"{mid:.1f} -- {direction} the fitted slope")

    label, lines = verdict(gate, b, t5)
    print()
    for line in lines:
        print(line)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp33_kerb_exposure",
        "generated_at": datetime.now(UTC).isoformat(),
        "preregistration": "experiments/PREREGISTRATION_exp33.md",
        "n_stints": int(len(stints)),
        "n_circuits": int(stints["circuit"].nunique()),
        "n_races": int(stints["session"].nunique()),
        "n_drivers": int(stints["driver"].nunique()),
        "n_stints_with_excursion": int(stints["any_limits"].sum()),
        "n_excursions": int(stints["n_limits"].sum()),
        "mean_slope_s_per_lap2": float(stints["slope"].mean()),
        "excursion_lap_position": {
            "mean_excursion_lap_in_run": float(stints["mean_excursion_lap_in_run"].mean())
            if "mean_excursion_lap_in_run" in stints else None,
            "mean_stint_lap_in_run": float(stints["mean_stint_lap_in_run"].mean())
            if "mean_stint_lap_in_run" in stints else None,
        },
        "measurement_gate": gate,
        "measure_b_track_limits": b,
        "T5_leave_one_circuit_out": t5,
        "measure_a_lateral_deviation": control,
        "measure_a_role": role,
        "bonferroni_alpha": bonferroni,
        "measure_a_survivors": survivors,
        "where_they_go_off": geography,
        "verdict": label,
        "verdict_lines": lines,
        "supported": label == "supported",
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
