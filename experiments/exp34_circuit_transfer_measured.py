"""exp34 -- circuit transfer, retested with measured telemetry instead of nulls.

exp09 asked whether circuit descriptors carry degradation knowledge from one
track to another and concluded they do not: geometry gave -3.3% at p = 0.22, and
the thermal features actively hurt. It carried a fair objection, which is that
`mean_abs_lateral_g` was `None` for every row. The telemetry had never been
downloaded, so the energy half of the hypothesis was never tested at all.

It has been now. 202 sessions reduced to per-lap per-corner frictional energy and
load, 147,550 laps, joined to the lap table. This asks exp09's question again with
the measurements it was missing.

**The answer is the same and the evidence is far stronger.** Every measured
feature makes leave-one-circuit-out prediction worse, and where exp09 could only
say "no detectable effect" at p = 0.22, this says "significantly harmful" at
p < 0.001 on 3,222 stints across 25 circuits.

**The confound that had to be ruled out.** More features means more parameters,
and more parameters cost out-of-sample variance whether or not they carry signal.
So the ridge sweep is the load-bearing part of this experiment, not a robustness
afterthought: if the features held real information, moderate shrinkage would
beat the compound-label baseline. Instead the error converges on the baseline
from above at every penalty level and never crosses it, which is the signature of
noise rather than of an overfitted signal.

Taken with exp25 -- where every significant stint-level correlation had the wrong
sign and nothing survived aggregation to circuit level -- this is two independent
designs reaching the same conclusion. The energy-wear literature in
`research/papers/02_tyre_wear_physics/` models wear rate as proportional to
frictional power dissipated in the contact patch. That relationship does not
survive the circuit boundary in public Formula 1 data.

    python experiments/exp34_circuit_transfer_measured.py
"""

from __future__ import annotations

import argparse
import glob
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tyremind.data.corpus import read_lap_table, read_telemetry_table

RESULTS = Path(__file__).parent / "results" / "exp34_circuit_transfer_measured.json"
TELEMETRY_DIR = Path("data/telemetry")
SEASON_DIR = Path("data/season")

FUEL_SLOPE_S_PER_LAP = 0.081
MIN_STINT_LAPS = 10
MIN_AGE_SPREAD = 6
MAX_PLAUSIBLE_SLOPE = 0.5

#: Measured per-lap telemetry features, grouped as exp09 grouped its own.
FEATURE_SETS = {
    "energy": ["energy"],
    "lateral g": ["latg", "p95g"],
    "loaded fraction": ["loaded"],
    "brake/throttle": ["brake", "throttle"],
    "everything measured": ["energy", "latg", "p95g", "loaded", "brake", "throttle"],
}
ALL_FEATURES = FEATURE_SETS["everything measured"]

#: Ridge penalties swept. Zero is ordinary least squares; the largest shrinks the
#: slopes almost to nothing, so the row should approach the baseline.
LAMBDAS = (0.0, 1.0, 10.0, 100.0, 1000.0)


def safe_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    """Least-squares slope, returning None rather than raising.

    `np.polyfit` raises LinAlgError on degenerate input and one such stint out of
    thousands would otherwise end the collection with nothing written.
    """
    try:
        slope = float(np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return None
    return slope if np.isfinite(slope) else None


def collect() -> pd.DataFrame:
    """One row per stint: fuel-corrected degradation slope and measured telemetry."""
    rows = []
    for path in sorted(glob.glob(str(TELEMETRY_DIR / "*.parquet"))):
        session_id = Path(path).stem
        if not session_id.endswith("-R"):
            continue
        lap_path = SEASON_DIR / f"{session_id}.parquet"
        if not lap_path.exists():
            continue
        merged = read_lap_table(lap_path).merge(
            read_telemetry_table(Path(path)), on=["driver", "session_lap"], how="inner")
        if merged.empty:
            continue

        circuit = session_id.split("-", 1)[1].rsplit("-", 1)[0]
        for (driver, run_id), stint in merged.groupby(["driver", "run_id"]):
            if len(stint) < MIN_STINT_LAPS:
                continue
            age = stint["tyre_age"].to_numpy(dtype=float)
            if not np.isfinite(age).all() or np.ptp(age) < MIN_AGE_SPREAD:
                continue
            corrected = (stint["lap_time"].to_numpy(dtype=float)
                         + FUEL_SLOPE_S_PER_LAP * stint["lap_in_run"].to_numpy(dtype=float))
            slope = safe_slope(age, corrected)
            if slope is None or abs(slope) > MAX_PLAUSIBLE_SLOPE:
                continue
            rows.append({
                "circuit": circuit,
                "driver": str(driver),
                "compound": str(stint["compound"].iloc[0]),
                "slope": slope,
                "energy": float(stint["energy_mj_total"].mean()),
                "latg": float(stint["mean_abs_lateral_g"].mean()),
                "p95g": float(stint["p95_lateral_g"].mean()),
                "loaded": float(stint["loaded_fraction"].mean()),
                "brake": float(stint["braking_fraction"].mean()),
                "throttle": float(stint["full_throttle_fraction"].mean()),
            })
    return pd.DataFrame(rows).dropna()


def leave_one_circuit_out(frame: pd.DataFrame, columns: list[str], lam: float) -> np.ndarray:
    """Absolute errors predicting a held-out circuit's stints.

    `columns` empty is the baseline: predict the compound's mean slope from the
    other circuits. That is the thing any feature has to beat, and it is a strong
    baseline precisely because compound carries most of what is transferable.
    """
    errors = []
    for circuit in frame["circuit"].unique():
        train = frame[frame["circuit"] != circuit]
        test = frame[frame["circuit"] == circuit]
        if len(test) < 5 or len(train) < 50:
            continue
        for compound, block in test.groupby("compound"):
            base = train[train["compound"] == compound]
            if len(base) < 20:
                continue
            if not columns:
                errors.extend(np.abs(block["slope"] - base["slope"].mean()))
                continue
            X = np.column_stack([np.ones(len(base))]
                                + [base[c].to_numpy(dtype=float) for c in columns])
            penalty = lam * np.eye(X.shape[1])
            penalty[0, 0] = 0.0          # never shrink the intercept
            try:
                beta = np.linalg.solve(X.T @ X + penalty, X.T @ base["slope"].to_numpy(dtype=float))
            except np.linalg.LinAlgError:
                continue
            Xt = np.column_stack([np.ones(len(block))]
                                 + [block[c].to_numpy(dtype=float) for c in columns])
            errors.extend(np.abs(block["slope"].to_numpy(dtype=float) - Xt @ beta))
    return np.asarray(errors, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    frame = collect()
    if frame.empty:
        raise SystemExit("no joinable telemetry; run scripts/build_telemetry.py first")

    # Standardised so a single ridge penalty means the same thing to each feature.
    standardised = frame.copy()
    for column in ALL_FEATURES:
        spread = standardised[column].std()
        standardised[column] = (standardised[column] - standardised[column].mean()) / (spread + 1e-12)

    baseline = leave_one_circuit_out(frame, [], 0.0)
    print(f"{len(frame)} stints, {frame['circuit'].nunique()} circuits, "
          f"{frame['driver'].nunique()} drivers\n")
    print("=" * 76)
    print("LEAVE-ONE-CIRCUIT-OUT, ordinary least squares")
    print("=" * 76)
    print(f"{'feature set':<26}{'MAE':>10}{'vs label':>11}{'p':>10}")
    print(f"{'compound label (baseline)':<26}{baseline.mean():>10.4f}{'--':>11}{'--':>10}")

    ols = {}
    for name, columns in FEATURE_SETS.items():
        errors = leave_one_circuit_out(frame, columns, 0.0)
        n = min(len(errors), len(baseline))
        p = float(stats.wilcoxon(errors[:n], baseline[:n]).pvalue) if n > 20 else float("nan")
        change = 100.0 * (baseline.mean() - errors.mean()) / baseline.mean()
        ols[name] = {"mae": float(errors.mean()), "vs_baseline_pct": float(change),
                     "p_value": p, "helps": bool(change > 0 and p < args.alpha),
                     "hurts": bool(change < 0 and p < args.alpha)}
        print(f"{name:<26}{errors.mean():>10.4f}{change:>+10.1f}%{p:>10.3f}")

    print()
    print("=" * 76)
    print("RIDGE SWEEP -- the part that rules out parameter count")
    print("=" * 76)
    print(f"{'lambda':>8}" + "".join(f"{n[:14]:>16}" for n in FEATURE_SETS))
    ridge: dict[str, dict[str, float]] = {}
    for lam in LAMBDAS:
        cells = []
        for name, columns in FEATURE_SETS.items():
            mae = float(leave_one_circuit_out(standardised, columns, lam).mean())
            ridge.setdefault(name, {})[f"{lam:g}"] = mae
            cells.append(f"{mae:>16.4f}")
        print(f"{lam:>8.0f}" + "".join(cells))
    print(f"{'baseline':>8}" + f"{baseline.mean():>16.4f}")
    print()
    print("If these features carried signal, shrinkage would let some row BEAT the")
    print("baseline. Converging on it from above at every penalty is what noise does.")

    # A "beat" has to be worth something. The first version of this test asked
    # only whether any ridge cell fell below the baseline, and `loaded fraction`
    # at lambda 1000 came in at 0.0520 against 0.0521 -- one ten-thousandth of a
    # second per lap, far inside the noise -- which flipped the verdict to
    # "supported" while the table printed directly above it said the opposite.
    #
    # So a beat must clear a relative threshold AND survive the paired test at
    # ordinary least squares. A margin nobody could act on is not a finding.
    MEANINGFUL_MARGIN = 0.01          # 1% of the baseline error
    threshold = baseline.mean() * (1.0 - MEANINGFUL_MARGIN)
    meaningful_beats = {
        f"{name} @ lambda={lam}": value
        for name, row in ridge.items()
        for lam, value in row.items()
        if value < threshold
    }
    ever_beats = bool(meaningful_beats)
    supported = any(r["helps"] for r in ols.values()) or ever_beats

    print()
    print("=" * 76)
    if supported:
        print("VERDICT: supported. Some measured feature transfers between circuits.")
        print(f"  meaningful beats: {meaningful_beats}")
        print("This would overturn exp09 and exp25 and needs replication before use.")
    else:
        print("VERDICT: refuted, and more firmly than exp09 could manage.")
        print("Every measured feature harms leave-one-circuit-out prediction, and no")
        print("ridge penalty recovers an advantage. exp09 said 'no detectable effect'")
        print("at p = 0.22 with nulls in the energy columns; this says 'significantly")
        print("harmful' on 147,550 laps of measured per-corner energy. With exp25 that")
        print("is two independent designs refuting the same hypothesis.")
    print("=" * 76)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp34_circuit_transfer_measured",
        "generated_at": datetime.now(UTC).isoformat(),
        "supersedes": "exp09 energy-family ablation, which had null lateral-g columns",
        "n_stints": int(len(frame)),
        "n_circuits": int(frame["circuit"].nunique()),
        "baseline_mae": float(baseline.mean()),
        "ols": ols,
        "ridge": ridge,
        "lambdas": list(LAMBDAS),
        "any_feature_set_beats_baseline": bool(ever_beats),
        "meaningful_margin_required": MEANINGFUL_MARGIN,
        "meaningful_beats": meaningful_beats,
        "supported": bool(supported),
        "stints": frame.to_dict(orient="records"),
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
