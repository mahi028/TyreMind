"""Phase 6 — race validation. The brief asks for this explicitly: fit on
practice only, predict degradation, and check it against what actually
happened in the race, where fuel load is known accurately (unlike practice).

Two numbers per event (Section 8), and the slope error is the one that
matters -- a constant pace offset is just engine mode/fuel-load difference and
doesn't change strategy; the slope does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.evaluate import mae
from src.models.dataset import Vocabs, make_batch
from src.models.nam import NAM


@dataclass
class RaceValidationResult:
    event_id: object
    compound: str
    predicted_slope_s_per_lap: float
    observed_slope_s_per_lap: float
    mae_fuel_corrected_s: float

    @property
    def slope_error_s_per_lap(self) -> float:
        return abs(self.predicted_slope_s_per_lap - self.observed_slope_s_per_lap)


def fuel_correct_race_pace(race_df: pd.DataFrame, *, race_start_fuel_kg: float,
                             fuel_time_per_kg_s: float, session_lap_col: str = "session_lap",
                             race_distance_col: str = "race_distance") -> pd.Series:
    """Subtracts the KNOWN (not estimated) fuel effect from race lap times.

    Unlike practice, a race's start fuel is well approximated by
    `race_start_fuel_kg` (~110kg) burned evenly across the full race distance
    -- there is no per-stint ambiguity here, because the race is one
    continuous fuel burn from a known start regardless of pit stops.

    Args:
        race_df: Must carry `session_lap_col` (1-indexed lap number in the
            race) and `race_distance_col` (total race laps, constant per event).
        race_start_fuel_kg: Assumed starting fuel load, kg.
        fuel_time_per_kg_s: Same fixed coefficient the model's `FuelHead` uses.

    Returns:
        `lap_time - true_fuel_effect`, i.e. lap time with the known fuel
        contribution removed, leaving tyre + evolution + conditions + traffic
        + driver + circuit + noise -- comparable to the model's non-fuel heads
        summed.
    """
    burn_per_lap = race_start_fuel_kg / race_df[race_distance_col]
    fuel_kg = race_start_fuel_kg - burn_per_lap * race_df[session_lap_col]
    fuel_effect = fuel_time_per_kg_s * fuel_kg
    return race_df["lap_time"] - fuel_effect


def observed_slope_per_stint(race_df: pd.DataFrame, *, tyre_age_col: str = "tyre_age",
                               fuel_corrected_col: str = "lap_time_fuel_corrected",
                               min_stint_laps: int = 4) -> pd.DataFrame:
    """Per-(driver, stint) linear slope of fuel-corrected pace vs. tyre age --
    the empirical counterpart to the model's recovered `g_tyre` slope.

    Returns:
        One row per (driver_id, stint_id, compound) with `n_laps` and `slope`.
    """
    rows = []
    group_cols = [c for c in ["driver_id", "stint_id"] if c in race_df.columns]
    for keys, group in race_df.groupby(group_cols):
        if len(group) < min_stint_laps:
            continue
        x = group[tyre_age_col].to_numpy(dtype=np.float64)
        y = group[fuel_corrected_col].to_numpy(dtype=np.float64)
        slope, intercept = np.polyfit(x, y, 1)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({"compound": group["compound"].iloc[0], "n_laps": len(group), "slope": slope})
        rows.append(row)
    return pd.DataFrame(rows)


def validate_event(
    model: NAM, vocabs: Vocabs, race_df: pd.DataFrame, *, event_id, race_start_fuel_kg: float,
    fuel_time_per_kg_s: float, race_distance_col: str = "race_distance",
    reference_circuit_id: int = 0, reference_track_temp_c: float = 35.0,
    device: torch.device | None = None,
) -> list[RaceValidationResult]:
    """Compares a model fitted on PRACTICE ONLY against one event's actual race.

    Args:
        model: A `NAM` already fit on practice sessions (never shown this race).
        vocabs: The vocab the model was fit with (for encoding `compound`).
        race_df: This event's race lap table (real schema, from `features.py`).
        event_id: Just carried through into the result rows for bookkeeping.
        race_start_fuel_kg: See `fuel_correct_race_pace`.
        fuel_time_per_kg_s: See `fuel_correct_race_pace`.

    Returns:
        One `RaceValidationResult` per compound present in the race.
    """
    device = device or next(model.parameters()).device
    race_df = race_df.copy()
    race_df["lap_time_fuel_corrected"] = fuel_correct_race_pace(
        race_df, race_start_fuel_kg=race_start_fuel_kg, fuel_time_per_kg_s=fuel_time_per_kg_s,
        race_distance_col=race_distance_col,
    )
    empirical = observed_slope_per_stint(race_df)

    results = []
    model.eval()
    for compound, group in empirical.groupby("compound"):
        if compound not in vocabs.compound.mapping:
            continue  # never seen in practice -- nothing to compare against
        compound_id = vocabs.compound.mapping[compound]
        circuit_id = vocabs.circuit.mapping.get(reference_circuit_id, vocabs.circuit.unk_index)
        curve = model.degradation_curve(
            compound_id=compound_id, circuit_id=circuit_id,
            track_temp_c=reference_track_temp_c, device=device,
        ).cpu().numpy()
        predicted_slope = float(np.mean(np.diff(curve)[0:8]))
        observed_slope = float(group["slope"].mean())

        compound_race = race_df[race_df["compound"] == compound]
        with torch.no_grad():
            batch = make_batch(compound_race, vocabs, device)
            pred_lap_time = model(batch)["lap_time_hat"].cpu().numpy()
        error = mae(pred_lap_time, compound_race["lap_time_fuel_corrected"].to_numpy())

        results.append(
            RaceValidationResult(
                event_id=event_id, compound=compound,
                predicted_slope_s_per_lap=predicted_slope, observed_slope_s_per_lap=observed_slope,
                mae_fuel_corrected_s=error,
            )
        )
    return results


def summarise(results: list[RaceValidationResult]) -> pd.DataFrame:
    """Flattens results for the predicted-vs-observed scatter (Section 8's
    "single figure that is the credibility of the whole project")."""
    return pd.DataFrame(
        [
            {
                "event_id": r.event_id, "compound": r.compound,
                "predicted_slope": r.predicted_slope_s_per_lap,
                "observed_slope": r.observed_slope_s_per_lap,
                "slope_error": r.slope_error_s_per_lap,
                "mae_fuel_corrected_s": r.mae_fuel_corrected_s,
            }
            for r in results
        ]
    )
