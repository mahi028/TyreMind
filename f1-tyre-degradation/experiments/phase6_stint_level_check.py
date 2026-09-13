"""Stint-level correlation check for Phase 6.

The event-level validation (`phase6_race_validation.py`) averages every
stint's empirical slope down to one number per (event, compound) before
comparing to the model. That throws away most of the real sample: 79
event-compound points come from several hundred individual real stints, and
event-level averaging both shrinks n and adds noise (an event's average slope
is itself a noisy estimate when only 1-3 stints of a compound ran that race).

This re-fits the same practice-only model (same seed, same setup) and instead
of aggregating, pairs the model's predicted slope for (compound, circuit,
event's mean track_temp) against EVERY individual real stint's own empirical
slope -- one point per real stint, not per event-compound. If stint-level
correlation is meaningfully higher than the event-level r=0.11, event
aggregation was adding the noise. If it's still weak, the ceiling is
elsewhere (most likely n and the comparable scale of event-to-event true
variation vs. measurement noise, per FINDINGS.md #3's detection floor).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
import yaml

from src.data.clean import clean_corpus
from src.data.features import split_by_event
from src.models.dataset import Vocabs
from src.models.nam import NAM, NAMConfig
from src.train import TrainConfig, fit, pick_device
from src.validate_race import observed_slope_per_stint, fuel_correct_race_pace

PRACTICE_CODES = ["FP1", "FP2", "FP3"]


def _adapt_schema(df: pd.DataFrame, *, fuel_burn_per_lap_kg: float) -> pd.DataFrame:
    df = df.rename(columns={"driver": "driver_id", "real_lap_time": "lap_time"})
    df["fuel_kg_rel"] = -fuel_burn_per_lap_kg * df["lap_in_stint"]
    return df


def main():
    raw_config = yaml.safe_load(open("config/default.yaml"))
    fuel_cfg = raw_config["fuel"]
    device = pick_device()

    print("cleaning real PRACTICE sessions...")
    practice_frames = []
    for code in PRACTICE_CODES:
        table, _ = clean_corpus(Path("data/raw"), session_code=code)
        if len(table):
            practice_frames.append(table)
    practice_df = _adapt_schema(pd.concat(practice_frames, ignore_index=True),
                                  fuel_burn_per_lap_kg=fuel_cfg["burn_per_lap_kg"])

    print("cleaning real RACE sessions...")
    race_df_all = _adapt_schema(clean_corpus(Path("data/raw"), session_code="R")[0],
                                  fuel_burn_per_lap_kg=fuel_cfg["burn_per_lap_kg"])

    torch.manual_seed(42)
    vocabs = Vocabs.fit(practice_df)
    nam_cfg = NAMConfig(
        n_circuits=vocabs.circuit.size, n_entries=vocabs.entry.size,
        n_compounds=vocabs.compound.size, max_age=raw_config["tyre"]["max_age"],
        fuel_time_per_kg_s=fuel_cfg["time_per_kg_s"],
    )
    model = NAM(nam_cfg)
    train_cfg = TrainConfig.from_yaml_dict(raw_config)
    practice_train, practice_val = split_by_event(practice_df, val_fraction=0.15, seed=42)
    print("fitting NAM on real practice laps only (same setup as phase6_race_validation.py)...")
    fit(model, practice_train, practice_val, vocabs, train_cfg, device=device, verbose=False)
    model.eval()

    rows = []
    for event_id, race_event in race_df_all.groupby("event_id"):
        circuit_string = race_event["circuit_id"].iloc[0]
        if circuit_string not in vocabs.circuit.mapping:
            continue
        circuit_id = vocabs.circuit.mapping[circuit_string]
        mean_track_temp = float(race_event["track_temp"].mean())
        race_distance = race_event["lap_number"].max()

        race_event = race_event.copy()
        race_event["race_distance"] = race_distance
        race_event["session_lap"] = race_event["lap_number"]
        race_event["lap_time_fuel_corrected"] = fuel_correct_race_pace(
            race_event, race_start_fuel_kg=fuel_cfg["race_start_fuel_kg"],
            fuel_time_per_kg_s=fuel_cfg["time_per_kg_s"],
        )
        stint_slopes = observed_slope_per_stint(race_event)  # one row per (driver_id, stint_id)

        for compound in stint_slopes["compound"].unique():
            if compound not in vocabs.compound.mapping:
                continue
            compound_id = vocabs.compound.mapping[compound]
            curve = model.degradation_curve(
                compound_id=compound_id, circuit_id=circuit_id,
                track_temp_c=mean_track_temp, device=device,
            ).cpu().numpy()
            predicted_slope = float(np.mean(np.diff(curve)[0:8]))

            for _, stint_row in stint_slopes[stint_slopes["compound"] == compound].iterrows():
                rows.append({
                    "event_id": event_id, "compound": compound,
                    "n_laps": stint_row["n_laps"], "observed_slope": stint_row["slope"],
                    "predicted_slope": predicted_slope,
                })

    stint_df = pd.DataFrame(rows)
    Path("outputs").mkdir(exist_ok=True)
    stint_df.to_csv("outputs/phase6_stint_level.csv", index=False)

    print(f"\n{len(stint_df)} individual real stints (vs. 79 event-compound points aggregated)")
    r, p = stats.pearsonr(stint_df["observed_slope"], stint_df["predicted_slope"])
    print(f"stint-level pooled Pearson r = {r:.3f} (p={p:.4f})")
    for compound, g in stint_df.groupby("compound"):
        rc, pc = stats.pearsonr(g["observed_slope"], g["predicted_slope"])
        print(f"  {compound:8s} n={len(g):4d}  r={rc:.3f} (p={pc:.4f})")

    # Also: event-level r recomputed on this same fit, for a same-run comparison
    event_level = stint_df.groupby(["event_id", "compound"]).agg(
        observed_slope=("observed_slope", "mean"), predicted_slope=("predicted_slope", "mean"),
    ).reset_index()
    r_evt, p_evt = stats.pearsonr(event_level["observed_slope"], event_level["predicted_slope"])
    print(f"\n(for comparison, event-level r on this same fit: {r_evt:.3f}, p={p_evt:.4f}, n={len(event_level)})")


if __name__ == "__main__":
    main()
