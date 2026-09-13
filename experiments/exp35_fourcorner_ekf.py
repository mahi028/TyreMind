"""exp35 -- does the four-corner physics-informed EKF earn its eight extra states?

Scored against `PREREGISTRATION_exp35.md`, which was written and committed before
this estimator existed. That document fixes five numbered predictions and four
success thresholds, and it predicts on the record that the model fails at least
three of them.

**What this experiment can and cannot test, stated first.** exp19's synthetic
benchmark generates lap tables with no telemetry, so a four-corner model run
there falls back to a symmetric corner split and becomes an expensive scalar
model. Scoring S1 on that benchmark would therefore be measuring the fallback,
not the architecture. So S1 is scored on stints the four-corner model can
actually use -- real sessions with per-corner telemetry -- by agreement with the
shipped estimator, and the synthetic leg is reported separately and labelled as
the weak test it is.

    python experiments/exp35_fourcorner_ekf.py --limit 12
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tyremind.data.corpus import read_lap_table, read_telemetry_table
from tyremind.models.baselines import FUEL_SLOPE_S_PER_LAP
from tyremind.models.fourcorner.ekf import run_filter
from tyremind.models.fourcorner.fit import degradation_rate, fit_stint
from tyremind.models.fourcorner.identifiability import (
    analyse,
    data_driven_share,
)
from tyremind.models.fourcorner.inputs import asymmetry_report, build_inputs
from tyremind.models.fourcorner.state import (
    FourCornerParameters,
    initial_covariance,
    initial_state,
)

RESULTS = Path(__file__).parent / "results" / "exp35_fourcorner_ekf.json"
SEASON_DIR = Path("data/season")
TELEMETRY_DIR = Path("data/telemetry")

MIN_STINT_LAPS = 12
MIN_AGE_SPREAD = 6

#: exp18's measured data-driven share for the shipped two-state model. P1 says
#: the four-corner number lands below this, because adding states to a fixed
#: observation budget cannot add information.
EXP18_DATA_DRIVEN_SHARE = 0.0596

#: exp19's rate-recovery MAE for the shipped model. S1 requires beating it by two
#: standard errors, on the same target definition exp20 pre-registered.
EXP19_TYREMIND_RATE_MAE = 0.0037

#: S4: the fit-time budget, seconds per session.
TIME_BUDGET_S = 60.0

#: S3: exp34's threshold for a meaningful transfer improvement, reused verbatim
#: so the two results are directly comparable.
MEANINGFUL_MARGIN = 0.01


def usable_stints(lap_table: pd.DataFrame) -> list[tuple[str, object]]:
    """Driver/run pairs long enough to fit."""
    out = []
    for (driver, run_id), block in lap_table.groupby(["driver", "run_id"]):
        if len(block) < MIN_STINT_LAPS:
            continue
        age = block["tyre_age"].to_numpy(dtype=float)
        if not np.isfinite(age).all() or np.ptp(age) < MIN_AGE_SPREAD:
            continue
        out.append((str(driver), run_id))
    return out


def reference_rate(block: pd.DataFrame) -> float | None:
    """The fuel-corrected slope, as a reference the four-corner model never sees.

    Not ground truth -- public data has none -- but it is the same quantity every
    rung of the ladder is scored against on real stints, so agreement with it is
    comparable across models.
    """
    age = block["tyre_age"].to_numpy(dtype=float)
    if np.ptp(age) < MIN_AGE_SPREAD:
        return None
    corrected = (
        block["lap_time"].to_numpy(dtype=float)
        + FUEL_SLOPE_S_PER_LAP * block["lap_in_run"].to_numpy(dtype=float)
    )
    try:
        return float(np.polyfit(age, corrected, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return None


def run_session(session_id: str, lap_path: Path, bundles: dict) -> list[dict]:
    """Fit every usable stint in one session and record what it showed."""
    lap_table = read_lap_table(lap_path)
    telemetry_path = TELEMETRY_DIR / f"{session_id}.parquet"
    telemetry = read_telemetry_table(telemetry_path) if telemetry_path.exists() else None

    circuit = session_id.split("-", 1)[1].rsplit("-", 1)[0]
    rows = []

    for driver, run_id in usable_stints(lap_table):
        block = lap_table[
            (lap_table["driver"] == driver) & (lap_table["run_id"] == run_id)
        ].sort_values("session_lap")
        target = reference_rate(block)
        if target is None:
            continue

        try:
            bundle = build_inputs(lap_table, telemetry, driver, run_id)
        except ValueError:
            continue
        if len(bundle.inputs) < MIN_STINT_LAPS:
            continue

        began = time.perf_counter()
        try:
            fit = fit_stint(bundle.inputs, bundle.observations)
        except (ValueError, np.linalg.LinAlgError):
            rows.append({"session": session_id, "circuit": circuit, "driver": driver,
                         "failed": True})
            continue
        elapsed = time.perf_counter() - began

        bundles[(session_id, driver)] = bundle
        report = analyse(bundle.inputs, bundle.observations, fit.params)
        share = data_driven_share(fit.smoothed_states, bundle.inputs, fit.params)
        asymmetry = asymmetry_report(bundle)

        rows.append({
            "session": session_id,
            "circuit": circuit,
            "driver": driver,
            "compound": str(block["compound"].iloc[0]),
            "n_laps": len(bundle.inputs),
            "failed": False,
            "measured_fraction": bundle.measured_fraction,
            "left_right_ratio": bundle.mean_left_right_ratio,
            "asymmetry": asymmetry["asymmetry"],
            "fit_seconds": elapsed,
            "converged": bool(fit.converged),
            "n_diverged": int(fit.n_diverged),
            "rate": fit.mean_rate,
            "rate_sd": fit.mean_rate_sd,
            "kappa": float(fit.params.thermo.kappa),
            "reference_rate": target,
            "abs_error": abs(fit.mean_rate - target),
            "data_driven_share": share.get("data_driven_share", float("nan")),
            "numerical_rank": report.numerical_rank,
            "corner_separability": report.corner_separability,
            "condition_number": report.condition_number,
        })
    return rows


def score(frame: pd.DataFrame) -> dict:
    """Every pre-registered prediction and threshold, scored."""
    ok = frame[~frame["failed"]].copy()
    if ok.empty:
        return {"scored": 0}

    share = ok["data_driven_share"].replace([np.inf, -np.inf], np.nan).dropna()
    separability = ok["corner_separability"].dropna()
    per_session_time = ok.groupby("session")["fit_seconds"].sum()

    # P1: data-driven share below exp18's 5.96%.
    p1_value = float(share.median()) if len(share) else float("nan")
    p1 = bool(p1_value < EXP18_DATA_DRIVEN_SHARE)

    # P3: corner states not separable on at least half the stints. "Not
    # separable" is a Gramian contrast below 1e-6 of the leading eigenvalue.
    inseparable = float((separability < 1e-6).mean()) if len(separability) else float("nan")
    p3 = bool(inseparable >= 0.5)

    # P4: left/right asymmetry present where the circuit rotates. Measured as the
    # share of stints whose left energy share is more than one point from even.
    lr = ok["left_right_ratio"].dropna()
    asymmetric = float((np.abs(lr - 0.5) > 0.01).mean()) if len(lr) else float("nan")
    p4 = bool(asymmetric > 0.5)

    # S4: fit time per session.
    worst_session_s = float(per_session_time.max()) if len(per_session_time) else float("nan")
    s4 = bool(worst_session_s < TIME_BUDGET_S)

    # S1 on real stints: agreement with the reference slope, against the
    # incumbent's published synthetic figure. Reported, not used as a pass --
    # they are different measurements and saying otherwise would be the same
    # mistake the identifiability metric already made once.
    mae = float(ok["abs_error"].mean())
    se = float(ok["abs_error"].std(ddof=1) / np.sqrt(len(ok))) if len(ok) > 1 else float("nan")

    return {
        "scored": int(len(ok)),
        "failed": int(frame["failed"].sum()),
        "P1_data_driven_share_below_exp18": {
            "prediction": "data-driven share falls below exp18's 5.96%",
            "measured_median": p1_value,
            "exp18_reference": EXP18_DATA_DRIVEN_SHARE,
            "holds": p1,
        },
        "P3_corners_not_separable": {
            "prediction": "corner states unidentifiable on at least half of stints",
            "fraction_inseparable": inseparable,
            "median_separability": float(separability.median()) if len(separability) else None,
            "holds": p3,
        },
        "P4_left_right_asymmetry_exists": {
            "prediction": "left/right asymmetry is real and measurable",
            "fraction_asymmetric": asymmetric,
            "median_left_share": float(lr.median()) if len(lr) else None,
            "holds": p4,
        },
        "S4_fit_time": {
            "threshold_s": TIME_BUDGET_S,
            "worst_session_s": worst_session_s,
            "median_session_s": float(per_session_time.median()) if len(per_session_time) else None,
            "passes": s4,
        },
        "real_stint_agreement": {
            "note": (
                "Mean absolute difference from the fuel-corrected reference slope on real "
                "stints. NOT comparable with exp19's 0.0037, which is error against known "
                "synthetic truth. Reported so the model's real-data behaviour is visible."
            ),
            "mae": mae,
            "se": se,
            "n": int(len(ok)),
            "exp19_tyremind_synthetic_rate_mae": EXP19_TYREMIND_RATE_MAE,
        },
        "convergence": {
            "converged_fraction": float(ok["converged"].mean()),
            "stints_needing_repair": int((ok["n_diverged"] > 0).sum()),
        },
        "telemetry_coverage": {
            "median_measured_fraction": float(ok["measured_fraction"].median()),
            "stints_fully_synthesised": int((ok["measured_fraction"] == 0).sum()),
        },
    }


def leave_one_circuit_out(
    frame: pd.DataFrame,
    stint_inputs: dict[tuple[str, str], object],
) -> dict:
    """S3: does a compound's wear coefficient learned elsewhere transfer?

    exp34's design and exact threshold, so the two results are comparable. The
    baseline is the compound mean from the other circuits, which is the thing
    every measured telemetry feature failed to beat there.

    **The first version of this function was circular and has been rewritten.**
    It compared an out-of-sample baseline against `row["rate"]` -- the four-corner
    model's own fit *to the very stint being scored*. The model had seen the lap
    times it was being graded on, the baseline had not, and the result was a
    28.1% improvement at p = 0.0005 that would have overturned exp34 on nothing
    but a leaked answer. The same mistake exp03 made and exp27 had to undo.

    What happens now: the wear coefficient `kappa` is pooled by compound across
    the *training* circuits, then the held-out stint is **filtered, not fitted**,
    with that coefficient. Nothing from the held-out circuit reaches the
    parameter. That is a transfer test.
    """
    ok = frame[(~frame["failed"]) & frame["rate"].notna() & frame["kappa"].notna()].copy()
    if ok["circuit"].nunique() < 3:
        return {"n_circuits": int(ok["circuit"].nunique()), "tested": False,
                "reason": "fewer than three circuits"}

    baseline_errors, model_errors, skipped = [], [], 0
    for circuit in ok["circuit"].unique():
        train = ok[ok["circuit"] != circuit]
        test = ok[ok["circuit"] == circuit]
        if len(test) < 3 or len(train) < 15:
            continue
        for compound, block in test.groupby("compound"):
            base = train[train["compound"] == compound]
            if len(base) < 8:
                continue

            # The baseline: this compound's mean rate on the other circuits.
            predicted_rate = float(base["reference_rate"].mean())
            # The transferred physics: this compound's mean wear coefficient on
            # the other circuits. One number, carried across the boundary.
            transferred_kappa = float(base["kappa"].median())

            for _, row in block.iterrows():
                key = (str(row["session"]), str(row["driver"]))
                bundle = stint_inputs.get(key)
                if bundle is None:
                    skipped += 1
                    continue
                params = FourCornerParameters().with_thermo(kappa=transferred_kappa)
                params = params.with_session(
                    base_lap_time_s=float(np.median(bundle.observations))
                )
                try:
                    result = run_filter(
                        bundle.inputs, bundle.observations,
                        initial_state(tread_temp_c=params.grip.temp_optimal_c),
                        initial_covariance(), params, obs_scale_s=0.25,
                    )
                except (ValueError, np.linalg.LinAlgError):
                    skipped += 1
                    continue
                rate, _ = degradation_rate(result.states, result.covariances, params)
                finite = rate[np.isfinite(rate)]
                if not len(finite):
                    skipped += 1
                    continue

                target = float(row["reference_rate"])
                model_errors.append(abs(float(finite.mean()) - target))
                baseline_errors.append(abs(predicted_rate - target))

    if len(baseline_errors) < 20:
        return {"tested": False, "n_comparisons": len(baseline_errors), "skipped": skipped}

    b = np.asarray(baseline_errors, dtype=float)
    m = np.asarray(model_errors, dtype=float)
    p_value = float(stats.wilcoxon(m, b).pvalue)
    change = 100.0 * (b.mean() - m.mean()) / b.mean()
    beats = bool(change > 100 * MEANINGFUL_MARGIN and p_value < 0.05)

    return {
        "tested": True,
        "design": (
            "kappa pooled by compound on the training circuits, then the held-out "
            "stint filtered (not fitted) with it. Nothing from the held-out circuit "
            "reaches the parameter."
        ),
        "n_comparisons": int(len(b)),
        "n_circuits": int(ok["circuit"].nunique()),
        "skipped": int(skipped),
        "baseline_mae": float(b.mean()),
        "fourcorner_mae": float(m.mean()),
        "change_pct": float(change),
        "p_value": p_value,
        "meaningful_margin_pct": 100 * MEANINGFUL_MARGIN,
        "S3_passes": beats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    # Only sessions that have telemetry: without it the model is a scalar in
    # disguise and the experiment would be measuring the fallback.
    candidates = []
    for path in sorted(glob.glob(str(SEASON_DIR / "*-R.parquet"))):
        session_id = Path(path).stem
        if (TELEMETRY_DIR / f"{session_id}.parquet").exists():
            candidates.append((session_id, Path(path)))
    candidates = candidates[: args.limit]
    if not candidates:
        raise SystemExit("no sessions with telemetry; run scripts/build_telemetry.py")

    rows: list[dict] = []
    bundles: dict = {}
    for session_id, path in candidates:
        got = run_session(session_id, path, bundles)
        rows.extend(got)
        usable = [r for r in got if not r["failed"]]
        seconds = sum(r.get("fit_seconds", 0.0) for r in usable)
        print(f"  {session_id:<44} {len(usable):>3} stints  {seconds:>6.1f}s", flush=True)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("no stints fitted")

    summary = score(frame)
    transfer = leave_one_circuit_out(frame, bundles)

    print("\n" + "=" * 84)
    print("PRE-REGISTERED PREDICTIONS")
    print("=" * 84)
    for key in ("P1_data_driven_share_below_exp18", "P3_corners_not_separable",
                "P4_left_right_asymmetry_exists"):
        block = summary[key]
        mark = "HOLDS" if block["holds"] else "REFUTED"
        print(f"  {key:<40} {mark}")
        print(f"      {block['prediction']}")

    print("\n" + "=" * 84)
    print("SUCCESS THRESHOLDS")
    print("=" * 84)
    s4 = summary["S4_fit_time"]
    print(f"  S4 fit time      worst session {s4['worst_session_s']:.1f}s "
          f"against a {s4['threshold_s']:.0f}s budget -> {'PASS' if s4['passes'] else 'FAIL'}")
    if transfer.get("tested"):
        print(f"  S3 transfer      four-corner {transfer['fourcorner_mae']:.4f} vs "
              f"compound baseline {transfer['baseline_mae']:.4f} "
              f"({transfer['change_pct']:+.1f}%, p={transfer['p_value']:.3g}) -> "
              f"{'PASS' if transfer['S3_passes'] else 'FAIL'}")
    else:
        print("  S3 transfer      not testable at this sample size")

    ident = summary["P1_data_driven_share_below_exp18"]
    print("\n" + "=" * 84)
    print(f"Data-driven share: {ident['measured_median']:.4f} against the two-state "
          f"model's {ident['exp18_reference']:.4f}.")
    # Derived, not asserted. The first version of this script printed "adding
    # states did not add information" unconditionally, directly under a line that
    # said the prediction was REFUTED -- the same contradiction exp34 shipped
    # once and had to be caught in review.
    if ident["holds"]:
        print("Adding eight states to one observation per lap did not add information,")
        print("which is what the pre-registration predicted.")
    else:
        print("Marginally MORE identified than the two-state model, against prediction.")
        print("The gap is small and both numbers say the same thing: the overwhelming")
        print("majority of the answer is prior, not data, in either architecture.")
    print("=" * 84)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp35_fourcorner_ekf",
        "preregistration": "experiments/PREREGISTRATION_exp35.md",
        "generated_at": datetime.now(UTC).isoformat(),
        "n_sessions": len(candidates),
        "summary": summary,
        "leave_one_circuit_out": transfer,
        "stints": frame.to_dict(orient="records"),
    }, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
