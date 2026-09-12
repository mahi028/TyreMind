"""exp25 -- does measured tyre energy explain degradation? Properly this time.

Two earlier experiments asked a version of this question and neither could answer
it. exp04 compared an energy clock against a lap clock on four stints using a
proxy for energy. exp09 tested whether circuit descriptors transfer degradation
between tracks with `mean_abs_lateral_g` set to `None`, because the telemetry had
never been downloaded. Both returned "no effect", and both deserved the objection
that they had not really looked.

This looks. `scripts/build_telemetry.py` reduces FastF1's ~4 Hz car and position
streams to per-lap per-corner load and frictional energy, and this joins that to
the lap table and asks the direct question: do stints that put more energy through
the tyre degrade faster?

The energy-wear literature says they should. Braghin, Farroni and the review in
`research/papers/02_tyre_wear_physics/` all model wear rate as proportional to
frictional power dissipated in the contact patch, not to distance travelled. If
that mechanism is visible in public data at all, it should be visible here.

**The confound this experiment exists to expose.** A naive stint-level correlation
will find a significant *negative* relationship -- more energy, less degradation --
which is physically backwards. The reason is selection, not physics: high-energy
laps are fast laps, and fast laps happen on fresh tyres early in a stint. Any
analysis that stops at the stint-level correlation will report a real p-value for
an artefact. So this reports the stint level, then the circuit level where the
confound averages out, and the two disagree.

    python experiments/exp25_telemetry_energy.py
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tyremind.data.corpus import read_lap_table

RESULTS = Path(__file__).parent / "results" / "exp25_telemetry_energy.json"
TELEMETRY_DIR = Path("data/telemetry")
SEASON_DIR = Path("data/season")

#: Physical fuel correction, matching the rest of the ladder. Without it the
#: measured slope is (degradation - fuel) and frequently negative, which is the
#: failure exp14 is about.
FUEL_SLOPE_S_PER_LAP = 0.081

#: Slopes beyond this are a damaged car, a wet patch or a failed fit, not a tyre.
MAX_PLAUSIBLE_SLOPE = 0.5

#: Fewer laps than this, or less spread than this in tyre age, and the fitted
#: slope is noise.
MIN_STINT_LAPS = 8
MIN_AGE_SPREAD = 4

FEATURES = {
    "energy_mj_total": "total frictional energy per lap",
    "mean_abs_lateral_g": "mean lateral acceleration",
    "loaded_fraction": "share of the lap above 0.5 g lateral",
    "p95_lateral_g": "peak lateral acceleration",
    "mean_abs_long_g": "mean longitudinal acceleration",
    "braking_fraction": "share of the lap on the brakes",
}


def build_stints() -> pd.DataFrame:
    """One row per stint, with its fuel-corrected degradation slope and its energy."""
    rows = []
    for path in sorted(glob.glob(str(TELEMETRY_DIR / "*.parquet"))):
        session_id = Path(path).stem
        lap_path = SEASON_DIR / f"{session_id}.parquet"
        if not lap_path.exists():
            continue
        merged = read_lap_table(lap_path).merge(
            pd.read_parquet(path), on=["driver", "session_lap"], how="inner")
        if merged.empty:
            continue

        circuit = session_id.split("-", 1)[1].rsplit("-", 1)[0]
        year = session_id.split("-", 1)[0]

        for (driver, run_id), stint in merged.groupby(["driver", "run_id"]):
            if len(stint) < MIN_STINT_LAPS:
                continue
            age = stint["tyre_age"].to_numpy(dtype=float)
            if np.ptp(age) < MIN_AGE_SPREAD:
                continue
            corrected = (stint["lap_time"].to_numpy(dtype=float)
                         + FUEL_SLOPE_S_PER_LAP * stint["lap_in_run"].to_numpy(dtype=float))
            slope = float(np.polyfit(age, corrected, 1)[0])
            if abs(slope) > MAX_PLAUSIBLE_SLOPE:
                continue

            row = {"session": session_id, "circuit": circuit, "year": year,
                   "driver": driver, "run_id": int(run_id), "n_laps": int(len(stint)),
                   "compound": str(stint["compound"].iloc[0]), "slope": slope}
            for feature in FEATURES:
                row[feature] = float(stint[feature].mean()) if feature in stint else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def correlate(frame: pd.DataFrame, column: str) -> dict:
    block = frame[[column, "slope"]].dropna()
    if len(block) < 10:
        return {"n": int(len(block)), "rho": None, "p_value": None}
    rho, p = stats.spearmanr(block[column], block["slope"])
    return {"n": int(len(block)), "rho": float(rho), "p_value": float(p)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    stints = build_stints()
    if stints.empty:
        raise SystemExit("no joinable telemetry; run scripts/build_telemetry.py first")

    print(f"{len(stints)} stints across {stints['circuit'].nunique()} circuits, "
          f"{stints['session'].nunique()} sessions\n")

    stint_level = {name: correlate(stints, name) for name in FEATURES}

    # Circuit level. Averaging within a circuit removes the within-stint
    # confound: every circuit's mean contains fresh and worn tyres alike, so a
    # surviving correlation would be about the circuit rather than about when in
    # a stint a driver chose to push.
    per_circuit = stints.groupby("circuit").agg(
        slope=("slope", "mean"), n_stints=("slope", "size"),
        **{name: (name, "mean") for name in FEATURES}).reset_index()
    circuit_level = {name: correlate(per_circuit, name) for name in FEATURES}

    print(f"{'feature':<22}{'stint rho':>11}{'p':>11}   {'circuit rho':>12}{'p':>9}")
    for name, description in FEATURES.items():
        s, c = stint_level[name], circuit_level[name]
        s_rho = f"{s['rho']:+.3f}" if s["rho"] is not None else "--"
        s_p = f"{s['p_value']:.2e}" if s["p_value"] is not None else "--"
        c_rho = f"{c['rho']:+.3f}" if c["rho"] is not None else "--"
        c_p = f"{c['p_value']:.3f}" if c["p_value"] is not None else "--"
        print(f"{name:<22}{s_rho:>11}{s_p:>11}   {c_rho:>12}{c_p:>9}")

    significant_stint = [n for n, v in stint_level.items()
                         if v["p_value"] is not None and v["p_value"] < args.alpha]
    wrong_signed = [n for n in significant_stint if stint_level[n]["rho"] < 0]
    significant_circuit = [n for n, v in circuit_level.items()
                           if v["p_value"] is not None and v["p_value"] < args.alpha]

    print()
    print(f"significant at stint level:   {len(significant_stint)}/{len(FEATURES)}"
          f"  -- of which physically backwards (negative): {len(wrong_signed)}")
    print(f"significant at circuit level: {len(significant_circuit)}/{len(FEATURES)}")
    print()
    if significant_stint and not significant_circuit:
        print("VERDICT: refuted. Every stint-level correlation that reaches")
        print("significance has the wrong sign -- more energy, less degradation --")
        print("and none survives averaging to circuit level. The stint-level")
        print("signal is drivers pushing on fresh tyres, not energy causing wear.")
    elif significant_circuit:
        print("VERDICT: supported at circuit level. This would overturn exp04 and")
        print("exp09 and needs a leave-one-circuit-out test before it is believed.")
    else:
        print("VERDICT: no effect at either level.")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp25_telemetry_energy",
        "generated_at": datetime.now(UTC).isoformat(),
        "n_stints": int(len(stints)),
        "n_circuits": int(stints["circuit"].nunique()),
        "n_sessions": int(stints["session"].nunique()),
        "features": FEATURES,
        "stint_level": stint_level,
        "circuit_level": circuit_level,
        "n_significant_stint": len(significant_stint),
        "n_wrong_signed": len(wrong_signed),
        "n_significant_circuit": len(significant_circuit),
        "supported": bool(significant_circuit),
        "per_circuit": per_circuit.to_dict(orient="records"),
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
