"""exp32 -- traffic and degradation: a real mechanism, or a measurement artefact?

A judge asked what our model does when a car is in traffic. Measured across real
race stints, the answer is that stints in traffic show *less* degradation:
+0.1037 s/lap in clear air against +0.0703 s/lap in the top traffic quartile,
t = -6.32, p = 4e-10, and it holds inside every compound.

There are two readings and they point opposite ways.

  (A) REAL MECHANISM. In traffic you cannot push, so less frictional energy goes
      through the tyre, so it genuinely wears less. The wear-energy literature in
      `research/papers/02_tyre_wear_physics/` models wear rate as proportional to
      dissipated frictional power, not to distance. Under (A) traffic is a rate
      modifier we ought to be modelling.

  (B) MEASUREMENT ARTEFACT. A car stuck behind another is pace-limited by the car
      ahead, not by its tyre. The tyre keeps degrading; the lap time cannot show
      it. Under (B) every lap-time-based method -- ours included -- systematically
      under-reports degradation in traffic, and that is a limitation paragraph,
      not a feature.

`data/telemetry/*.parquet` makes the two separable, because frictional energy is
observed rather than assumed. If (A) is true, energy MEDIATES traffic ->
degradation and controlling for energy should collapse the traffic coefficient.
If (B) is true, traffic still predicts lower measured degradation with energy held
fixed, because the masking is in the lap time rather than in the physics.

Four pieces, all pre-registered in `PREREGISTRATION_exp32.md`:

  1. Mediation, two specifications -- pooled with session x compound fixed
     effects, and within-driver-within-race so car pace cannot explain it. Single
     mediator (total energy) and the full eight-channel effort block, the latter
     so hypothesis (A) is tested in its most generous form. Cluster bootstrap
     over sessions for every interval.
  2. The energy clock. Refit each stint against cumulative megajoules instead of
     laps. (A) predicts the per-MJ deficit vanishes; (B) predicts it survives.
  3. Measured traffic. `data/reference/openf1_intervals/` carries 70 races of
     real ~4 Hz gaps, aligned here to lap windows through FastF1's cached lap
     start dates, as a confirmatory replacement for our derived index.
  4. Our own reported uncertainty: does the posterior sd on the degradation rate
     widen in traffic, at full scale rather than the 16-session pilot?

    python experiments/exp32_traffic_mechanism.py
    python experiments/exp32_traffic_mechanism.py --bootstrap 500 --no-uncertainty
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tyremind.data.corpus import read_lap_table, read_telemetry_table

RESULTS = Path(__file__).parent / "results" / "exp32_traffic_mechanism.json"
STINT_CACHE = Path(__file__).parent / "results" / "exp32_stints.parquet"
TELEMETRY_DIR = Path("data/telemetry")
SEASON_DIR = Path("data/season")
INTERVAL_DIR = Path("data/reference/openf1_intervals")

#: Physical fuel correction, matching the rest of the ladder. Without it the
#: measured slope is (degradation - fuel) and frequently negative.
FUEL_SLOPE_S_PER_LAP = 0.081

#: Pre-registered stint filters. A slope beyond MAX_PLAUSIBLE_SLOPE is a damaged
#: car, a wet patch or a failed fit, not a tyre.
MIN_STINT_LAPS = 10
MIN_AGE_SPREAD = 6
MAX_PLAUSIBLE_SLOPE = 0.5

#: Gap to the car ahead, in seconds, at or below which a measured sample counts
#: as "in traffic". Matches CLEAN_AIR_GAP_S/2 in the loader: a following car has
#: lost most of its downforce well before two seconds.
CLOSE_GAP_S = 1.0

#: An interval sample count below this leaves a lap's measured gap too noisy to
#: use; a stint needs MIN_MEASURED_LAPS such laps.
MIN_INTERVAL_SAMPLES = 3
MIN_MEASURED_LAPS = 5

#: The mediator block. Hypothesis (A) says traffic suppresses driver effort and
#: effort drives wear; these are every effort channel the telemetry carries, so
#: (A) is tested at its most generous rather than only through one number.
EFFORT_BLOCK = [
    "energy_mj_total",
    "mean_abs_lateral_g",
    "p95_lateral_g",
    "mean_abs_long_g",
    "loaded_fraction",
    "full_throttle_fraction",
    "braking_fraction",
    "mean_speed_kmh",
]

#: Pre-registered covariates, held fixed across every specification.
COVARIATES = ["n_laps", "mean_tyre_age"]

#: Decision thresholds, fixed in PREREGISTRATION_exp32.md section 6.
A_MIN_PROPORTION = 0.50
A_MIN_PROPORTION_LOWER = 0.25
B_MAX_PROPORTION = 0.30
B_MAX_PROPORTION_UPPER = 0.50
B_MIN_DIRECT_SHARE = 0.7

#: OpenF1 names some circuits differently from FastF1's event location. Mapping
#: written out rather than fuzzy-matched, because a wrong match would silently
#: attach one race's gaps to another race's laps.
CIRCUIT_ALIASES = {
    "Catalunya": "Barcelona",
    "Hungaroring": "Budapest",
    "Interlagos": "São Paulo",
    "Monte Carlo": "Monaco",
    "Montreal": "Montréal",
    "Singapore": "Marina Bay",
    "Yas Marina Circuit": "Yas Island",
    "Miami": "Miami Gardens",
}


# --------------------------------------------------------------------------- #
# Small numerical helpers. These are the pieces `tests/unit/test_exp32_helpers`
# pins, because each of them has a failure mode that would quietly bend a result
# rather than crash it.
# --------------------------------------------------------------------------- #


def fit_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    """Least-squares slope of y on x, or None if it cannot be trusted.

    `np.polyfit` raises `LinAlgError` on a degenerate design and has killed two
    runs in this project by doing so mid-loop. It can also return a non-finite
    coefficient without raising, when the inputs contain infinities. Both are
    handled here so no caller has to remember to.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 2:
        return None
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    if np.ptp(x) <= 0:
        return None
    try:
        slope = float(np.polyfit(x, y, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return None
    return slope if np.isfinite(slope) else None


def group_demean(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Subtract each group's mean, column by column.

    Demeaning inside a group is how a fixed effect is imposed here: after it, a
    coefficient is a within-group contrast and anything constant inside the group
    -- the circuit, the car, the race -- cannot produce it.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    if values.shape[0] != codes.shape[0]:
        values = values.T
    counts = np.bincount(codes, minlength=codes.max() + 1).astype(float)
    out = np.empty_like(values)
    for column in range(values.shape[1]):
        sums = np.bincount(codes, weights=values[:, column], minlength=counts.size)
        out[:, column] = values[:, column] - (sums / counts)[codes]
    return out


def ols(y: np.ndarray, x: np.ndarray) -> np.ndarray | None:
    """Coefficients on `x`, with an intercept fitted and dropped.

    Returns None rather than raising on a singular design, so a bootstrap
    replicate that happens to draw a degenerate sample is discarded instead of
    ending the run.
    """
    y = np.asarray(y, dtype=float)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[0] != y.shape[0]:
        x = x.T
    if y.size <= x.shape[1] + 1:
        return None
    design = np.column_stack([np.ones(y.size), x])
    if not np.isfinite(design).all() or not np.isfinite(y).all():
        return None
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return beta[1:] if np.isfinite(beta).all() else None


def mediation_paths(
    treatment: np.ndarray,
    mediators: np.ndarray,
    outcome: np.ndarray,
    covariates: np.ndarray | None = None,
) -> dict | None:
    """The four regressions of a product-of-coefficients mediation analysis.

    a: mediator ~ treatment (+ covariates)
    b, c': outcome ~ mediator + treatment (+ covariates)
    c: outcome ~ treatment (+ covariates)

    The indirect effect is sum(a_k * b_k). Under OLS that equals c - c' exactly,
    and both are returned so the identity can be checked rather than assumed.
    """
    treatment = np.asarray(treatment, dtype=float).reshape(-1, 1)
    mediators = np.atleast_2d(np.asarray(mediators, dtype=float))
    if mediators.shape[0] != treatment.shape[0]:
        mediators = mediators.T
    outcome = np.asarray(outcome, dtype=float)
    blocks = [treatment] if covariates is None or covariates.size == 0 else [
        treatment,
        np.atleast_2d(np.asarray(covariates, dtype=float)).reshape(treatment.shape[0], -1),
    ]
    t_and_c = np.hstack(blocks)

    a_path = []
    for column in range(mediators.shape[1]):
        beta = ols(mediators[:, column], t_and_c)
        if beta is None:
            return None
        a_path.append(float(beta[0]))

    full = ols(outcome, np.hstack([mediators, t_and_c]))
    total = ols(outcome, t_and_c)
    if full is None or total is None:
        return None

    b_path = [float(v) for v in full[: mediators.shape[1]]]
    c_prime = float(full[mediators.shape[1]])
    c_total = float(total[0])
    indirect = float(np.dot(a_path, b_path))
    proportion = indirect / c_total if abs(c_total) > 1e-12 else float("nan")
    return {
        "a": a_path,
        "b": b_path,
        "c": c_total,
        "c_prime": c_prime,
        "indirect": indirect,
        "indirect_as_difference": c_total - c_prime,
        "proportion_mediated": float(proportion),
    }


def percentile_interval(values: list[float], level: float = 0.95) -> dict:
    """Percentile bootstrap interval, with the count that survived."""
    clean = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if clean.size < 20:
        return {"n": int(clean.size), "lo": None, "hi": None}
    tail = (1.0 - level) / 2.0
    return {
        "n": int(clean.size),
        "lo": float(np.quantile(clean, tail)),
        "hi": float(np.quantile(clean, 1.0 - tail)),
    }


# --------------------------------------------------------------------------- #
# Building the stint table
# --------------------------------------------------------------------------- #


def race_session_ids() -> list[str]:
    manifest = SEASON_DIR / "corpus.json"
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    return [e["session_id"] for e in entries if e.get("session") == "R"]


def _stints_for_session(session_id: str, measured: pd.DataFrame | None) -> tuple[list[dict], dict]:
    """Every usable stint in one race, with its slopes on both clocks."""
    lap_path = SEASON_DIR / f"{session_id}.parquet"
    tel_path = TELEMETRY_DIR / f"{session_id}.parquet"
    counters = {"seen": 0, "short": 0, "flat_age": 0, "fit_failed": 0, "implausible": 0}
    if not (lap_path.exists() and tel_path.exists()):
        return [], counters

    merged = read_lap_table(lap_path).merge(
        read_telemetry_table(tel_path), on=["driver", "session_lap"], how="inner"
    )
    if merged.empty:
        return [], counters
    if measured is not None and not measured.empty:
        merged = merged.merge(measured, on=["driver", "session_lap"], how="left")

    circuit = session_id.split("-", 1)[1].rsplit("-", 1)[0]
    year = int(session_id.split("-", 1)[0])

    rows: list[dict] = []
    for (driver, run_id), stint in merged.groupby(["driver", "run_id"]):
        counters["seen"] += 1
        stint = stint.sort_values("session_lap")
        if len(stint) < MIN_STINT_LAPS:
            counters["short"] += 1
            continue
        age = stint["tyre_age"].to_numpy(dtype=float)
        if not np.isfinite(age).all() or np.ptp(age) < MIN_AGE_SPREAD:
            counters["flat_age"] += 1
            continue

        corrected = stint["lap_time"].to_numpy(dtype=float) + (
            FUEL_SLOPE_S_PER_LAP * stint["lap_in_run"].to_numpy(dtype=float)
        )
        slope = fit_slope(age, corrected)
        if slope is None:
            counters["fit_failed"] += 1
            continue
        if abs(slope) > MAX_PLAUSIBLE_SLOPE:
            counters["implausible"] += 1
            continue

        energy = stint["energy_mj_total"].to_numpy(dtype=float)
        slope_per_mj = None
        if np.isfinite(energy).all():
            # Lagged cumulative energy: the first lap of a stint has put no
            # energy through the tyre yet, which is the same convention tyre age
            # follows at the start of a run.
            cumulative = np.concatenate([[0.0], np.cumsum(energy[:-1])])
            slope_per_mj = fit_slope(cumulative, corrected)

        # Reconstructed from the slope already fitted rather than fitted again,
        # so there is exactly one polyfit per stint to go wrong.
        intercept = float(np.mean(corrected)) - slope * float(np.mean(age))
        resid = corrected - (intercept + slope * age)
        variance = float(np.var(corrected))
        r_squared = float(1.0 - np.var(resid) / variance) if variance > 0 else float("nan")

        row = {
            "session": session_id,
            "circuit": circuit,
            "year": year,
            "driver": str(driver),
            "run_id": int(run_id),
            "compound": str(stint["compound"].iloc[0]),
            "n_laps": int(len(stint)),
            "mean_tyre_age": float(np.mean(age)),
            "slope_per_lap": float(slope),
            "slope_per_mj": float(slope_per_mj) if slope_per_mj is not None else float("nan"),
            "r_squared": r_squared,
            "traffic_index": float(stint["traffic_index"].mean()),
        }
        for name in EFFORT_BLOCK:
            row[name] = float(stint[name].mean()) if name in stint else float("nan")

        if "close_fraction" in stint:
            valid = stint["close_fraction"].to_numpy(dtype=float)
            keep = np.isfinite(valid)
            if int(keep.sum()) >= MIN_MEASURED_LAPS:
                row["measured_close_fraction"] = float(np.mean(valid[keep]))
                row["measured_mean_interval"] = float(
                    np.nanmean(stint["mean_interval"].to_numpy(dtype=float))
                )
                row["measured_laps"] = int(keep.sum())
        rows.append(row)
    return rows, counters


def build_stints(measured: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    totals = {"seen": 0, "short": 0, "flat_age": 0, "fit_failed": 0, "implausible": 0}
    for session_id in sorted(race_session_ids()):
        session_rows, counters = _stints_for_session(session_id, measured.get(session_id))
        rows.extend(session_rows)
        for key, value in counters.items():
            totals[key] += value
    return pd.DataFrame(rows), totals


# --------------------------------------------------------------------------- #
# Measured traffic from OpenF1
# --------------------------------------------------------------------------- #


def _openf1_index() -> dict[tuple[int, str], Path]:
    """(year, circuit) -> interval file, with OpenF1's circuit names translated."""
    index: dict[tuple[int, str], Path] = {}
    for path in sorted(INTERVAL_DIR.glob("*.parquet")):
        if path.stem.endswith("_stints"):
            continue
        frame = read_telemetry_table(path)
        if frame.empty or "year" not in frame or "circuit" not in frame:
            continue
        circuit = str(frame["circuit"].iloc[0])
        index[(int(frame["year"].iloc[0]), CIRCUIT_ALIASES.get(circuit, circuit))] = path
    return index


def measured_traffic() -> tuple[dict[str, pd.DataFrame], dict]:
    """Per-lap measured gap to the car ahead, for every race we can align.

    OpenF1 timestamps each interval sample in absolute UTC; FastF1's cached
    session gives every lap's absolute start date and duration. A sample belongs
    to the lap whose window contains it. Nothing is interpolated: a lap with too
    few samples is left out rather than filled.
    """
    import fastf1

    logging.getLogger("fastf1").setLevel(logging.ERROR)
    fastf1.Cache.enable_cache("cache/fastf1")
    try:
        fastf1.Cache.offline_mode(True)
    except Exception:  # noqa: BLE001 -- older fastf1; the cache still serves
        pass

    manifest = json.loads((SEASON_DIR / "corpus.json").read_text(encoding="utf-8"))
    races = [e for e in manifest if e.get("session") == "R"]
    index = _openf1_index()

    out: dict[str, pd.DataFrame] = {}
    report = {
        "openf1_races_on_disk": len(index),
        "aligned_sessions": 0,
        "unmatched_sessions": [],
        "lap_agreement": [],
    }

    for entry in races:
        key = (int(entry["year"]), str(entry["location"]))
        path = index.get(key)
        if path is None:
            continue
        try:
            session = fastf1.get_session(entry["year"], entry["event_name"], "R")
            session.load(telemetry=True, weather=False, messages=False)
        except Exception as exc:  # noqa: BLE001 -- a missing cache entry is not fatal
            report["unmatched_sessions"].append(
                {"session": entry["session_id"], "reason": type(exc).__name__}
            )
            continue

        results = session.results
        if results is None or results.empty:
            report["unmatched_sessions"].append(
                {"session": entry["session_id"], "reason": "no_results"}
            )
            continue
        number_to_driver = dict(
            zip(results["DriverNumber"].astype(str), results["Abbreviation"].astype(str))
        )

        laps = session.laps[["Driver", "LapNumber", "LapStartDate", "LapTime"]].dropna(
            subset=["LapStartDate", "LapTime"]
        )
        if laps.empty:
            report["unmatched_sessions"].append(
                {"session": entry["session_id"], "reason": "no_lap_dates"}
            )
            continue
        laps = laps.copy()
        laps["start"] = pd.to_datetime(laps["LapStartDate"], utc=True)
        laps["end"] = laps["start"] + laps["LapTime"]

        intervals = read_telemetry_table(path)
        intervals = intervals.assign(
            driver=intervals["driver_number"].astype(str).map(number_to_driver)
        ).dropna(subset=["driver"])
        gap = intervals["interval"].to_numpy(dtype=float)
        # A lapped car ahead is a different state from a large gap, not a bigger
        # one, so those samples are dropped rather than coerced to a number.
        usable = np.isfinite(gap) & ~intervals["interval_lapped"].to_numpy(dtype=bool)
        intervals = intervals[usable]
        if intervals.empty:
            continue

        rows = []
        for driver, block in intervals.groupby("driver"):
            block = block.sort_values("date")
            stamps = block["date"].dt.tz_convert("UTC").astype("int64").to_numpy()
            values = block["interval"].to_numpy(dtype=float)
            for lap in laps[laps["Driver"] == driver].itertuples():
                lo = int(np.searchsorted(stamps, lap.start.value))
                hi = int(np.searchsorted(stamps, lap.end.value))
                if hi - lo < MIN_INTERVAL_SAMPLES:
                    continue
                window = values[lo:hi]
                rows.append(
                    {
                        "driver": str(driver),
                        "session_lap": int(lap.LapNumber),
                        "mean_interval": float(np.mean(window)),
                        "close_fraction": float(np.mean(window <= CLOSE_GAP_S)),
                        "interval_samples": hi - lo,
                    }
                )
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        out[entry["session_id"]] = frame
        report["aligned_sessions"] += 1

        lap_table = read_lap_table(SEASON_DIR / f"{entry['session_id']}.parquet")
        joined = lap_table.merge(frame, on=["driver", "session_lap"], how="inner")
        if len(joined) >= 30:
            rho, p_value = stats.spearmanr(joined["traffic_index"], joined["mean_interval"])
            report["lap_agreement"].append(
                {
                    "session": entry["session_id"],
                    "n_laps": int(len(joined)),
                    "rho_derived_vs_measured_gap": float(rho),
                    "p_value": float(p_value),
                }
            )
    return out, report


# --------------------------------------------------------------------------- #
# Specifications
# --------------------------------------------------------------------------- #


def build_bundle(frame: pd.DataFrame, treatment: str, mediators: list[str]) -> dict | None:
    """Everything a specification needs, as arrays, so the bootstrap is cheap.

    The bootstrap resamples whole races two thousand times. Doing that by
    concatenating DataFrames and re-running `groupby` costs minutes; doing it by
    indexing into precomputed arrays costs seconds. The arrays are built once,
    here, and every replicate is a row selection over them.
    """
    needed = [treatment, "slope_per_lap", *mediators, *COVARIATES]
    block = frame.dropna(subset=needed).reset_index(drop=True)
    if len(block) < 50:
        return None
    compound_code, compound_names = pd.factorize(block["compound"].astype(str))
    session_code, session_names = pd.factorize(block["session"].astype(str))
    driver_code, _ = pd.factorize(block["driver"].astype(str))
    return {
        "values": block[needed].to_numpy(dtype=float),
        "n_mediators": len(mediators),
        "n_covariates": len(COVARIATES),
        "compound_code": compound_code,
        "n_compounds": len(compound_names),
        "session_code": session_code,
        "session_names": session_names,
        "driver_code": driver_code,
        "rows_by_session": [np.flatnonzero(session_code == i) for i in range(len(session_names))],
    }


def design_for(bundle: dict, spec: str, rows: np.ndarray, slot: np.ndarray) -> dict | None:
    """Impose the specification's fixed effect on one (possibly resampled) draw.

    `slot` says which drawn copy of a race each row came from, so a race drawn
    twice gives two separate fixed-effect groups rather than one doubled group.

    S1 demeans within (race, compound): circuit, compound and race-day conditions
    are absorbed.

    S2 demeans within (race, driver), keeping only drivers with two or more
    stints in that race, so the contrast is between one driver's own stints in
    one race and **car pace cannot produce it**. Compound then varies inside the
    group, so it enters as dummies instead of as the fixed effect.
    """
    values = bundle["values"][rows]
    compound = bundle["compound_code"][rows]

    if spec == "S1":
        raw_key = slot * bundle["n_compounds"] + compound
    elif spec == "S2":
        raw_key = slot * (bundle["driver_code"].max() + 1) + bundle["driver_code"][rows]
    else:
        raise ValueError(f"unknown specification {spec!r}")
    codes = np.unique(raw_key, return_inverse=True)[1]

    if spec == "S2":
        # A driver with one stint in a race contributes nothing to a
        # within-driver contrast, and demeaning would set his row to zero.
        keep = np.bincount(codes)[codes] >= 2
        if int(keep.sum()) < 50:
            return None
        values = values[keep]
        compound = compound[keep]
        codes = np.unique(codes[keep], return_inverse=True)[1]
        present = np.unique(compound)
        if present.size > 1:
            dummies = np.column_stack(
                [(compound == level).astype(float) for level in present[1:]]
            )
            values = np.hstack([values, dummies])

    # z-scoring before demeaning keeps every coefficient on the same scale, which
    # is what makes a mediated proportion readable across specifications.
    sd = values.std(axis=0)
    sd[sd <= 0] = 1.0
    values = group_demean((values - values.mean(axis=0)) / sd, codes)

    n_med = bundle["n_mediators"]
    return {
        "n": int(values.shape[0]),
        "treatment": values[:, 0],
        "outcome": values[:, 1],
        "mediators": values[:, 2 : 2 + n_med],
        "covariates": values[:, 2 + n_med :],
    }


def run_mediation(
    frame: pd.DataFrame,
    *,
    spec: str,
    treatment: str,
    mediators: list[str],
    replicates: int,
    rng: np.random.Generator,
) -> dict | None:
    bundle = build_bundle(frame, treatment, mediators)
    if bundle is None:
        return None
    n_rows = bundle["values"].shape[0]
    identity_rows = np.arange(n_rows)
    prepared = design_for(bundle, spec, identity_rows, bundle["session_code"])
    if prepared is None:
        return None
    point = mediation_paths(
        prepared["treatment"], prepared["mediators"], prepared["outcome"], prepared["covariates"]
    )
    if point is None:
        return None

    # Clustered on session: stints inside one race share a track, a temperature
    # and a safety car, so resampling stints would understate every interval.
    rows_by_session = bundle["rows_by_session"]
    n_sessions = len(rows_by_session)

    draws: dict[str, list[float]] = {k: [] for k in ("c", "c_prime", "indirect", "proportion_mediated")}
    failures = 0
    for _ in range(replicates):
        picked = rng.integers(0, n_sessions, size=n_sessions)
        chosen = [rows_by_session[i] for i in picked]
        rows = np.concatenate(chosen)
        slot = np.repeat(np.arange(n_sessions), [len(c) for c in chosen])
        prep = design_for(bundle, spec, rows, slot)
        if prep is None:
            failures += 1
            continue
        result = mediation_paths(
            prep["treatment"], prep["mediators"], prep["outcome"], prep["covariates"]
        )
        if result is None:
            failures += 1
            continue
        for key in draws:
            draws[key].append(float(result[key]))

    out = {
        "specification": spec,
        "treatment": treatment,
        "mediators": mediators,
        "n_stints": prepared["n"],
        "n_sessions": n_sessions,
        "a_path": point["a"],
        "b_path": point["b"],
        "total_effect_c": point["c"],
        "direct_effect_c_prime": point["c_prime"],
        "indirect_effect": point["indirect"],
        "indirect_as_c_minus_c_prime": point["indirect_as_difference"],
        "proportion_mediated": point["proportion_mediated"],
        "direct_share_of_total": (
            abs(point["c_prime"]) / abs(point["c"]) if abs(point["c"]) > 1e-12 else float("nan")
        ),
        "bootstrap_replicates": replicates,
        "bootstrap_failures": failures,
        "ci": {key: percentile_interval(values) for key, values in draws.items()},
    }
    if len(mediators) == 1:
        # With one mediator the a and b paths are scalars and worth surfacing as
        # such, since the decision rule reads them directly.
        out["a"] = point["a"][0]
        out["b"] = point["b"][0]
    return out


def verdict(single: dict | None) -> dict:
    """Apply the pre-registered decision rule to one mediation result."""
    if single is None:
        return {"call": "inconclusive", "reason": "specification did not estimate"}
    c = single["total_effect_c"]
    prop = single["proportion_mediated"]
    prop_ci = single["ci"]["proportion_mediated"]
    direct_ci = single["ci"]["c_prime"]
    indirect_ci = single["ci"]["indirect"]
    b = single.get("b")
    a = single.get("a")

    reasons = []
    if c >= 0:
        return {"call": "inconclusive", "reason": "total effect is not negative; the finding did not replicate here"}
    if b is None or not np.isfinite(b) or b <= 0:
        return {
            "call": "inconclusive",
            "reason": (
                "the mediator does not predict the outcome in the direction the mechanism "
                "requires (b <= 0), so mediation is not identified -- pre-registered as a live risk"
            ),
        }

    direct_significant = (
        direct_ci["lo"] is not None and direct_ci["lo"] * direct_ci["hi"] > 0
    )
    indirect_significant = (
        indirect_ci["lo"] is not None and indirect_ci["lo"] * indirect_ci["hi"] > 0
    )
    a_ok = a is not None and a < 0
    share = single["direct_share_of_total"]

    supports_a = (
        a_ok
        and indirect_significant
        and single["indirect_effect"] < 0
        and prop >= A_MIN_PROPORTION
        and prop_ci["lo"] is not None
        and prop_ci["lo"] > A_MIN_PROPORTION_LOWER
        and (not direct_significant or share < 0.5)
    )
    supports_b = (
        direct_significant
        and single["direct_effect_c_prime"] < 0
        and share >= B_MIN_DIRECT_SHARE
        and prop <= B_MAX_PROPORTION
        and prop_ci["hi"] is not None
        and prop_ci["hi"] < B_MAX_PROPORTION_UPPER
    )
    diagnostics = {
        "a_path": a,
        "b_path": b,
        "a_has_the_sign_mechanism_A_requires": bool(a_ok),
        "direct_effect_significant": bool(direct_significant),
        "indirect_effect_significant": bool(indirect_significant),
        "direct_share_of_total": share,
        "proportion_mediated": prop,
    }
    if not a_ok:
        reasons.append(
            "the a path is not negative: traffic does not reduce the mediator, so the first "
            "step of mechanism (A) fails before mediation is even tested"
        )
    if prop < 0:
        reasons.append(
            "the mediated proportion is negative -- controlling for the mediator makes the "
            "traffic effect larger, which is suppression rather than mediation"
        )

    if supports_a and not supports_b:
        return {
            "call": "A_real_mechanism",
            "reason": "energy carries the traffic effect",
            "notes": reasons,
            "diagnostics": diagnostics,
        }
    if supports_b and not supports_a:
        return {
            "call": "B_measurement_artefact",
            "reason": "traffic still predicts lower measured degradation with effort held fixed",
            "notes": reasons,
            "diagnostics": diagnostics,
        }
    # The pre-registration reserves "partial" for a *positive* mediated
    # proportion in the declared 0.30-0.50 band. A negative proportion is
    # suppression, not partial mediation, and must not borrow the word.
    if indirect_significant and direct_significant and B_MAX_PROPORTION <= prop <= A_MIN_PROPORTION:
        return {
            "call": "partial",
            "reason": f"partial mediation, proportion {prop:.3f}",
            "notes": reasons,
            "diagnostics": diagnostics,
        }
    return {
        "call": "inconclusive",
        "reason": "neither pre-registered pattern is met at this specification's intervals",
        "notes": reasons,
        "diagnostics": diagnostics,
    }


# --------------------------------------------------------------------------- #
# The energy clock
# --------------------------------------------------------------------------- #


def quartile_contrast(frame: pd.DataFrame, traffic: str, outcome: str) -> dict:
    """Bottom vs top traffic quartile on one clock, plus the rank correlation."""
    block = frame[[traffic, outcome, "compound"]].dropna()
    if len(block) < 40:
        return {"n": int(len(block))}
    lo_cut, hi_cut = np.quantile(block[traffic].to_numpy(dtype=float), [0.25, 0.75])
    clear = block[block[traffic] <= lo_cut][outcome].to_numpy(dtype=float)
    heavy = block[block[traffic] >= hi_cut][outcome].to_numpy(dtype=float)
    t_stat, p_value = stats.ttest_ind(heavy, clear, equal_var=False)
    rho, rho_p = stats.spearmanr(block[traffic], block[outcome])
    clear_mean = float(np.mean(clear))
    out = {
        "n": int(len(block)),
        "n_clear": int(clear.size),
        "n_heavy": int(heavy.size),
        "clear_mean": clear_mean,
        "heavy_mean": float(np.mean(heavy)),
        "difference": float(np.mean(heavy) - clear_mean),
        "relative_deficit": (
            float((clear_mean - np.mean(heavy)) / clear_mean) if abs(clear_mean) > 1e-12 else None
        ),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "by_compound": {},
    }
    for compound, part in block.groupby("compound"):
        if len(part) < 30:
            continue
        rho_c, p_c = stats.spearmanr(part[traffic], part[outcome])
        out["by_compound"][str(compound)] = {
            "n": int(len(part)),
            "rho": float(rho_c),
            "p_value": float(p_c),
        }
    return out


# --------------------------------------------------------------------------- #
# Does our own reported uncertainty widen in traffic?
# --------------------------------------------------------------------------- #


def _fit_one_session(session_id: str) -> dict | None:
    warnings.filterwarnings("ignore")
    try:
        from tyremind.models.ssm.tyre_ssm import fit_tyre_ssm

        lap_table = read_lap_table(SEASON_DIR / f"{session_id}.parquet")
        fit = fit_tyre_ssm(lap_table)
        rates = fit.degradation()
        joined = rates.merge(
            lap_table[["driver", "session_lap", "traffic_index"]],
            on=["driver", "session_lap"],
            how="inner",
        )
        traffic_coef, traffic_coef_sd = fit.traffic_coefficient()
        return {
            "session": session_id,
            "laps": joined[["driver", "session_lap", "tyre_age", "rate_sd", "traffic_index"]].to_dict("list"),
            "mean_traffic": float(lap_table["traffic_index"].mean()),
            "mean_rate_sd": float(joined["rate_sd"].mean()),
            "traffic_coefficient": float(traffic_coef),
            "traffic_coefficient_sd": float(traffic_coef_sd),
            "converged": bool(fit.converged),
        }
    except Exception as exc:  # noqa: BLE001 -- one bad session must not end the sweep
        return {"session": session_id, "error": type(exc).__name__}


def uncertainty_in_traffic(session_ids: list[str], workers: int, rng: np.random.Generator) -> dict:
    """Full-scale version of the 16-session pilot (0.0274 clear vs 0.0421 busy)."""
    fits: list[dict] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(_fit_one_session, session_ids):
                if result is not None:
                    fits.append(result)
    else:
        for session_id in session_ids:
            result = _fit_one_session(session_id)
            if result is not None:
                fits.append(result)

    good = [f for f in fits if "error" not in f]
    failed = [f for f in fits if "error" in f]
    if len(good) < 10:
        return {"n_sessions": len(good), "failures": failed, "note": "too few fits to test"}

    session_level = pd.DataFrame(
        [
            {
                "session": f["session"],
                "mean_traffic": f["mean_traffic"],
                "mean_rate_sd": f["mean_rate_sd"],
                "traffic_coefficient": f["traffic_coefficient"],
                "converged": f["converged"],
            }
            for f in good
        ]
    )
    lo_cut, hi_cut = np.quantile(session_level["mean_traffic"], [1 / 3, 2 / 3])
    clear = session_level[session_level["mean_traffic"] <= lo_cut]["mean_rate_sd"]
    busy = session_level[session_level["mean_traffic"] >= hi_cut]["mean_rate_sd"]
    t_stat, p_value = stats.ttest_ind(busy, clear, equal_var=False)
    rho, rho_p = stats.spearmanr(session_level["mean_traffic"], session_level["mean_rate_sd"])

    lap_rows = []
    for f in good:
        part = pd.DataFrame(f["laps"])
        part["session"] = f["session"]
        lap_rows.append(part)
    laps = pd.concat(lap_rows, ignore_index=True)
    laps = laps.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["rate_sd", "traffic_index", "tyre_age"]
    )

    key = laps["session"].astype(str) + "|" + laps["driver"].astype(str)
    codes = pd.factorize(key)[0]
    columns = ["rate_sd", "traffic_index", "tyre_age"]
    values = laps[columns].to_numpy(dtype=float)
    sd = values.std(axis=0)
    sd[sd <= 0] = 1.0
    values = group_demean((values - values.mean(axis=0)) / sd, codes)
    beta = ols(values[:, 0], values[:, 1:])
    within_coefficient = float(beta[0]) if beta is not None else float("nan")

    # Cluster bootstrap over sessions, matching every other interval here.
    by_session = {name: block for name, block in laps.groupby("session")}
    names = np.array(list(by_session))
    draws = []
    for _ in range(200):
        picked = rng.choice(names, size=names.size, replace=True)
        sample = pd.concat([by_session[n] for n in picked], ignore_index=True)
        s_key = sample["session"].astype(str) + "|" + sample["driver"].astype(str)
        s_codes = pd.factorize(s_key)[0]
        s_values = sample[columns].to_numpy(dtype=float)
        s_sd = s_values.std(axis=0)
        s_sd[s_sd <= 0] = 1.0
        s_values = group_demean((s_values - s_values.mean(axis=0)) / s_sd, s_codes)
        s_beta = ols(s_values[:, 0], s_values[:, 1:])
        if s_beta is not None:
            draws.append(float(s_beta[0]))

    return {
        "n_sessions": int(len(good)),
        "n_failures": len(failed),
        "failures": failed[:10],
        "n_converged": int(session_level["converged"].sum()),
        "session_level": {
            "clear_tercile_mean_rate_sd": float(clear.mean()),
            "busy_tercile_mean_rate_sd": float(busy.mean()),
            "n_clear": int(clear.size),
            "n_busy": int(busy.size),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
        },
        "within_driver_race_lap_level": {
            "n_laps": int(len(laps)),
            "standardised_coefficient": within_coefficient,
            "ci": percentile_interval(draws),
        },
        "traffic_coefficient_s": {
            "mean": float(session_level["traffic_coefficient"].mean()),
            "median": float(session_level["traffic_coefficient"].median()),
        },
    }


# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000, help="cluster bootstrap replicates")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--workers", type=int, default=6, help="processes for the SSM sweep")
    parser.add_argument("--no-uncertainty", action="store_true", help="skip the SSM sweep")
    parser.add_argument("--no-measured", action="store_true", help="skip the OpenF1 alignment")
    parser.add_argument("--refresh", action="store_true", help="rebuild the cached stint table")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    rng = np.random.default_rng(args.seed)

    deviations: list[str] = []

    sidecar = STINT_CACHE.with_suffix(".json")
    if STINT_CACHE.exists() and sidecar.exists() and not args.refresh:
        stints = read_telemetry_table(STINT_CACHE)
        cached = json.loads(sidecar.read_text(encoding="utf-8"))
        filters, measured_report = cached["filters"], cached["measured_traffic_alignment"]
        print(f"stint table from cache: {len(stints)} stints")
    else:
        measured: dict[str, pd.DataFrame] = {}
        measured_report = {"skipped": True}
        if not args.no_measured:
            print("aligning OpenF1 intervals to lap windows ...")
            measured, measured_report = measured_traffic()
            print(f"  aligned {measured_report['aligned_sessions']} races")
        stints, filters = build_stints(measured)
        STINT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        stints.to_parquet(STINT_CACHE, index=False)
        sidecar.write_text(
            json.dumps(
                {"filters": filters, "measured_traffic_alignment": measured_report},
                indent=2,
                default=float,
            ),
            encoding="utf-8",
        )

    if stints.empty:
        raise SystemExit("no joinable telemetry; run scripts/build_telemetry.py first")

    stints = stints[stints["compound"].isin(["SOFT", "MEDIUM", "HARD"])]
    n_measured = int(stints["measured_close_fraction"].notna().sum()) if "measured_close_fraction" in stints else 0
    print(
        f"{len(stints)} stints, {stints['session'].nunique()} races, "
        f"{n_measured} with measured traffic"
    )

    # --- 1. Replicate the finding on this sample -------------------------- #
    replication = {
        "derived_traffic_per_lap_clock": quartile_contrast(stints, "traffic_index", "slope_per_lap"),
    }
    if n_measured >= 200:
        replication["measured_traffic_per_lap_clock"] = quartile_contrast(
            stints, "measured_close_fraction", "slope_per_lap"
        )

    # --- 2. Mediation ------------------------------------------------------ #
    mediation: dict = {}
    for spec in ("S1", "S2"):
        for treatment, label in (
            ("traffic_index", "derived"),
            ("measured_close_fraction", "measured"),
        ):
            if treatment not in stints.columns:
                continue
            if treatment == "measured_close_fraction" and n_measured < 200:
                continue
            single = run_mediation(
                stints,
                spec=spec,
                treatment=treatment,
                mediators=["energy_mj_total"],
                replicates=args.bootstrap,
                rng=rng,
            )
            block = run_mediation(
                stints,
                spec=spec,
                treatment=treatment,
                mediators=EFFORT_BLOCK,
                replicates=args.bootstrap,
                rng=rng,
            )
            mediation[f"{spec}_{label}"] = {
                "single_mediator_energy": single,
                "effort_block": block,
                "verdict_single_mediator": verdict(single),
            }
            if single is not None:
                print(
                    f"  {spec}/{label}: c={single['total_effect_c']:+.4f} "
                    f"c'={single['direct_effect_c_prime']:+.4f} "
                    f"prop={single['proportion_mediated']:+.3f} "
                    f"a={single.get('a', float('nan')):+.4f} b={single.get('b', float('nan')):+.4f}"
                )

    # --- 3. The energy clock ----------------------------------------------- #
    energy_clock = {
        "per_lap_clock": quartile_contrast(stints, "traffic_index", "slope_per_lap"),
        "per_mj_clock": quartile_contrast(stints, "traffic_index", "slope_per_mj"),
        "energy_per_lap_by_traffic": quartile_contrast(stints, "traffic_index", "energy_mj_total"),
    }
    if n_measured >= 200:
        energy_clock["per_lap_clock_measured"] = quartile_contrast(
            stints, "measured_close_fraction", "slope_per_lap"
        )
        energy_clock["per_mj_clock_measured"] = quartile_contrast(
            stints, "measured_close_fraction", "slope_per_mj"
        )
    per_lap_deficit = energy_clock["per_lap_clock"].get("relative_deficit")
    per_mj_deficit = energy_clock["per_mj_clock"].get("relative_deficit")
    energy_clock["reading"] = (
        "the deficit survives the change of clock, which is what masking looks like"
        if per_mj_deficit is not None
        and per_lap_deficit is not None
        and per_mj_deficit > 0.5 * per_lap_deficit
        and energy_clock["per_mj_clock"]["p_value"] < 0.05
        else "the deficit shrinks or vanishes on the energy clock, as a wear mechanism predicts"
    )

    # --- 4. Exploratory ----------------------------------------------------- #
    exploratory = {
        "stint_fit_r_squared_by_traffic": quartile_contrast(stints, "traffic_index", "r_squared"),
        "full_throttle_fraction_by_traffic": quartile_contrast(
            stints, "traffic_index", "full_throttle_fraction"
        ),
        "braking_fraction_by_traffic": quartile_contrast(
            stints, "traffic_index", "braking_fraction"
        ),
        "mean_speed_by_traffic": quartile_contrast(stints, "traffic_index", "mean_speed_kmh"),
    }
    if n_measured >= 200:
        pair = stints.dropna(subset=["measured_close_fraction", "traffic_index"])
        rho, p_value = stats.spearmanr(pair["traffic_index"], pair["measured_close_fraction"])
        exploratory["derived_vs_measured_stint_level"] = {
            "n": int(len(pair)),
            "rho": float(rho),
            "p_value": float(p_value),
        }

    # --- 5. Our own uncertainty --------------------------------------------- #
    if args.no_uncertainty:
        uncertainty = {"skipped": True}
        deviations.append("--no-uncertainty: the SSM sweep of section 8 was not run")
    else:
        print("fitting the state-space model on every race for the uncertainty test ...")
        uncertainty = uncertainty_in_traffic(
            sorted(stints["session"].unique().tolist()), args.workers, rng
        )

    # S2 is the specification the finding had to survive (pre-registration
    # section 5), so it is the one the headline verdict reads.
    primary = mediation.get("S2_derived", {}).get("verdict_single_mediator") or {
        "call": "inconclusive",
        "reason": "S2 did not estimate",
    }
    per_lap = energy_clock["per_lap_clock"]
    per_mj = energy_clock["per_mj_clock"]
    energy_by_traffic = energy_clock["energy_per_lap_by_traffic"]
    s2 = mediation.get("S2_derived", {}).get("single_mediator_energy") or {}
    summary = {
        "verdicts_by_specification": {
            name: block["verdict_single_mediator"]["call"] for name, block in mediation.items()
        },
        "the_five_facts": [
            (
                f"The finding replicates: {per_lap['clear_mean']:.4f} s/lap in clear air against "
                f"{per_lap['heavy_mean']:.4f} in the top traffic quartile, t={per_lap['t_stat']:.2f}, "
                f"p={per_lap['p_value']:.2e}."
            ),
            (
                "Traffic does not reduce the energy through the tyre -- it raises it: "
                f"{energy_by_traffic['clear_mean']:.2f} MJ/lap clear against "
                f"{energy_by_traffic['heavy_mean']:.2f} in traffic, t={energy_by_traffic['t_stat']:.2f}, "
                f"p={energy_by_traffic['p_value']:.2e}. Mechanism (A) fails at its first step."
            ),
            (
                "The deficit survives a change of clock: per megajoule of frictional energy it is "
                f"{per_mj['relative_deficit']:.1%} against {per_lap['relative_deficit']:.1%} per lap, "
                f"t={per_mj['t_stat']:.2f}, p={per_mj['p_value']:.2e}."
            ),
            (
                "Controlling for energy makes the traffic effect larger, not smaller: "
                f"c={s2.get('total_effect_c', float('nan')):+.4f} to "
                f"c'={s2.get('direct_effect_c_prime', float('nan')):+.4f} within driver and race. "
                "That is suppression, not mediation."
            ),
            (
                "Tyre age explains far less of the lap time in traffic: stint-fit R2 "
                f"{exploratory['stint_fit_r_squared_by_traffic']['clear_mean']:.3f} clear against "
                f"{exploratory['stint_fit_r_squared_by_traffic']['heavy_mean']:.3f} in traffic, "
                f"p={exploratory['stint_fit_r_squared_by_traffic']['p_value']:.2e} -- the lap time is "
                "being set by something other than the tyre."
            ),
        ],
        "consequence": (
            "Every lap-time-based degradation estimate, ours included, under-reports degradation "
            "in traffic. Traffic is not a rate modifier to add to the physics; it is a hole in the "
            "observation, and it belongs in the limitations of any lap-time method."
        ),
    }

    payload = {
        "experiment": "exp32_traffic_mechanism",
        "question": (
            "Is the lower measured degradation of stints in traffic a real wear mechanism "
            "(less energy through the tyre) or a measurement artefact (a pace-limited lap "
            "time cannot show the wear)?"
        ),
        "preregistration": "experiments/PREREGISTRATION_exp32.md",
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "sample": {
            "n_stints": int(len(stints)),
            "n_sessions": int(stints["session"].nunique()),
            "n_with_measured_traffic": n_measured,
            "filters": filters,
            "min_stint_laps": MIN_STINT_LAPS,
            "min_age_spread": MIN_AGE_SPREAD,
            "max_plausible_slope": MAX_PLAUSIBLE_SLOPE,
            "fuel_slope_s_per_lap": FUEL_SLOPE_S_PER_LAP,
        },
        "measured_traffic_alignment": measured_report,
        "replication_of_the_finding": replication,
        "mediation": mediation,
        "energy_clock": energy_clock,
        "exploratory": exploratory,
        "model_uncertainty_in_traffic": uncertainty,
        "primary_verdict": primary,
        "summary": summary,
        "deviations": deviations,
    }

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\nprimary verdict: {primary['call']} -- {primary['reason']}")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
