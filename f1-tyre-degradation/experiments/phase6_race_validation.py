"""Phase 6 -- race validation, the first use of real data with NO synthetic
injection anywhere in the pipeline.

Fits the NAM on real PRACTICE sessions only (FP1/FP2/FP3), across every
2022-2023 event currently on disk, then for each event compares the fitted
tyre curve's slope against the empirical slope of that SAME event's real RACE
laps (fuel-corrected with the known ~110kg race start load, which is a much
better fuel assumption in a race than in practice -- see
`src/validate_race.py`'s docstring).

This is a practice-to-race GENERALIZATION check, not a cross-event one: the
model never sees race laps during fitting, but does see practice laps from
the same events (so circuit identity transfers; per-stint entry bias mostly
does not, by design -- entry_id is keyed per (event, driver, stint), and race
stint numbers don't correspond to practice run numbers, so nearly every race
entry_id is unseen and falls back to the zero-initialized <UNK> embedding).
That means the MAE number will be inflated by a missing car/driver pace
offset that was never expected to transfer -- the slope comparison is the one
that matters (Section 8), and is reported as the headline number.

Also carries a specific, stated expectation from the residual-bias finding
(FINDINGS.md #5): if the fitted curve compresses the stint's true dynamics,
predicted slopes should be smaller than observed ones on average -- i.e. the
predicted-vs-observed scatter's best-fit slope should come in below 1, not as
a surprise but as the same bias showing up a third way.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.clean import clean_corpus
from src.models.dataset import Vocabs
from src.models.nam import NAM, NAMConfig
from src.train import TrainConfig, fit, pick_device
from src.validate_race import summarise, validate_event

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
COLORS = {"SOFT": "#2a78d6", "MEDIUM": "#eb6834", "HARD": "#1baf7a"}

PRACTICE_CODES = ["FP1", "FP2", "FP3"]


def _adapt_schema(df: pd.DataFrame, *, fuel_burn_per_lap_kg: float) -> pd.DataFrame:
    """`clean.py`'s column names -> what `dataset.py`/`validate_race.py` expect.

    Real, additive renames only -- no synthetic values introduced anywhere.
    """
    df = df.rename(columns={"driver": "driver_id", "real_lap_time": "lap_time"})
    df["fuel_kg_rel"] = -fuel_burn_per_lap_kg * df["lap_in_stint"]
    return df


def main():
    raw_config = yaml.safe_load(open("config/default.yaml"))
    fuel_cfg = raw_config["fuel"]
    device = pick_device()

    print("cleaning real PRACTICE sessions (FP1/FP2/FP3, every event on disk)...")
    practice_frames = []
    for code in PRACTICE_CODES:
        table, _ = clean_corpus(Path("data/raw"), session_code=code)
        if len(table):
            practice_frames.append(table)
    practice_df = pd.concat(practice_frames, ignore_index=True)
    practice_df = _adapt_schema(practice_df, fuel_burn_per_lap_kg=fuel_cfg["burn_per_lap_kg"])
    print(f"practice laps: {len(practice_df)}, events: {practice_df['event_id'].nunique()}")

    print("cleaning real RACE sessions (same events)...")
    race_df_all, _ = clean_corpus(Path("data/raw"), session_code="R")
    race_df_all = _adapt_schema(race_df_all, fuel_burn_per_lap_kg=fuel_cfg["burn_per_lap_kg"])
    print(f"race laps: {len(race_df_all)}, events: {race_df_all['event_id'].nunique()}")

    # Fit on practice only -- race laps never enter training.
    torch.manual_seed(42)
    vocabs = Vocabs.fit(practice_df)
    nam_cfg = NAMConfig(
        n_circuits=vocabs.circuit.size, n_entries=vocabs.entry.size,
        n_compounds=vocabs.compound.size, max_age=raw_config["tyre"]["max_age"],
        fuel_time_per_kg_s=fuel_cfg["time_per_kg_s"],
    )
    model = NAM(nam_cfg)
    train_cfg = TrainConfig.from_yaml_dict(raw_config)

    # A real train/val split of the practice data itself (by event), so the
    # existing early-stopping / best-checkpoint machinery in `fit` still has
    # a held-out signal to select on -- distinct from the race-holdout
    # question this script is actually answering.
    from src.data.features import split_by_event
    practice_train, practice_val = split_by_event(practice_df, val_fraction=0.15, seed=42)

    print("fitting NAM on real practice laps only...")
    fit(model, practice_train, practice_val, vocabs, train_cfg, device=device, verbose=False)
    model.eval()

    all_results = []
    events_with_race = sorted(race_df_all["event_id"].unique())
    print(f"validating against {len(events_with_race)} real race events...")
    for event_id in events_with_race:
        race_event = race_df_all[race_df_all["event_id"] == event_id].copy()
        circuit_string = race_event["circuit_id"].iloc[0]
        if circuit_string not in vocabs.circuit.mapping:
            continue  # circuit never seen in practice (shouldn't happen for same-event data, but be safe)

        race_event["race_distance"] = race_event["lap_number"].max()
        race_event["session_lap"] = race_event["lap_number"]
        mean_track_temp = float(race_event["track_temp"].mean())

        results = validate_event(
            model, vocabs, race_event, event_id=event_id,
            race_start_fuel_kg=fuel_cfg["race_start_fuel_kg"],
            fuel_time_per_kg_s=fuel_cfg["time_per_kg_s"],
            reference_circuit_id=circuit_string,
            reference_track_temp_c=mean_track_temp,
            device=device,
        )
        all_results.extend(results)

    summary = summarise(all_results)
    print(f"\n{len(summary)} (event, compound) validation points")
    print(summary.groupby("compound")[["predicted_slope", "observed_slope", "slope_error"]].describe())

    Path("outputs").mkdir(exist_ok=True)
    summary.to_csv("outputs/phase6_race_validation.csv", index=False)

    # The scatter: predicted vs observed slope, one point per (event, compound).
    fig, ax = plt.subplots(figsize=(7, 6.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    lims_lo = min(summary["predicted_slope"].min(), summary["observed_slope"].min(), -0.02)
    lims_hi = max(summary["predicted_slope"].max(), summary["observed_slope"].max()) * 1.05
    ax.plot([lims_lo, lims_hi], [lims_lo, lims_hi], color=INK_MUTED, linewidth=1.2, linestyle="--",
            label="Perfect agreement (slope = 1)")

    for compound, group in summary.groupby("compound"):
        color = COLORS.get(compound, INK_SECONDARY)
        ax.scatter(group["observed_slope"], group["predicted_slope"], color=color, s=38,
                   alpha=0.85, edgecolor="white", linewidth=0.5, label=compound.title())

    if len(summary) >= 2:
        fit_slope, fit_intercept = np.polyfit(summary["observed_slope"], summary["predicted_slope"], 1)
        xs = np.array([lims_lo, lims_hi])
        ax.plot(xs, fit_slope * xs + fit_intercept, color=INK_PRIMARY, linewidth=1.6,
                label=f"Best fit, all {len(summary)} points (slope={fit_slope:.2f})")
    else:
        fit_slope = float("nan")

    ax.set_xlim(lims_lo, lims_hi)
    ax.set_ylim(lims_lo, lims_hi)
    ax.set_xlabel("Observed race slope, fuel-corrected (s/lap)", color=INK_SECONDARY)
    ax.set_ylabel("Predicted slope, fit on practice only (s/lap)", color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRIDLINE)
    ax.grid(color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    fig.text(0.08, 0.97, "Practice-fit tyre slope vs. what actually happened in the race",
              color=INK_PRIMARY, fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(0.08, 0.925,
              f"One point per (event, compound), {len(summary)} points across {len(events_with_race)} real events.\n"
              "Below the dashed line = the model under-predicts race degradation.",
              color=INK_SECONDARY, fontsize=9.5, ha="left", va="top")

    fig.tight_layout(rect=[0, 0, 1, 0.82])
    out_path = "outputs/figure6_race_validation_scatter.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved {out_path}")
    print(f"best-fit slope (predicted vs observed): {fit_slope:.3f}")
    print(f"mean |slope error|: {summary['slope_error'].mean():.4f} s/lap")
    print(f"median MAE (fuel-corrected, includes untransferred entry bias): "
          f"{summary['mae_fuel_corrected_s'].median():.2f} s")


if __name__ == "__main__":
    main()
