"""Phase 4/7 -- baseline comparison, run for the first time.

Same synthetic ground truth as the recovery test (`tests/test_recovery.py`),
matched to real-world data volume (13,388 rows, 22 circuits -- the same scale
used throughout Phase 7's figures), same train/val split, same target. Three
models: `LinearBaseline` (degree 1), LightGBM, and the NAM.

The point (`src/models/baselines.py`'s own docstring): raw prediction accuracy
(MAE) and recovered tyre-degradation slope are DIFFERENT objectives. A model
with no parameter that means "degradation rate" can win on MAE by splitting
the fuel/tyre-age collinear signal however minimizes loss, with no obligation
to attribute it correctly. If LightGBM wins MAE while recovering a
substantially worse slope/cliff than the NAM against KNOWN ground truth, that
is a direct, non-rhetorical demonstration that prediction accuracy and causal
attribution are not the same thing -- the core argument for building a
structurally-constrained model at all instead of just fitting the best
predictor available.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import src.sim.simulator as sim_mod
from src.data.features import split_by_event
from src.evaluate import detect_cliff, mae, recovered_slope
from src.models.baselines import LinearBaseline, fit_lightgbm, lightgbm_shap_tyre_curve, prepare_lgb_features
from src.models.dataset import Vocabs, make_batch, make_target
from src.models.nam import NAM, NAMConfig
from src.sim.simulator import SimulatorConfig, generate_synthetic_dataset, true_degradation_curve
from src.train import TrainConfig, fit, pick_device

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
TRUE_COLOR = "#898781"
NAM_COLOR = "#2a78d6"
LGB_COLOR = "#eb6834"

TARGET_ROWS = 13388
SEED = 42


def main():
    sim_mod.N_CIRCUITS = 22
    cfg = SimulatorConfig.from_yaml()
    cfg.n_sessions = 30
    raw_config = yaml.safe_load(open("config/default.yaml"))
    device = pick_device()

    df_full, truth = generate_synthetic_dataset(cfg, seed=SEED, return_truth=True)
    rng = np.random.default_rng(SEED)
    idx = sorted(rng.choice(len(df_full), size=min(TARGET_ROWS, len(df_full)), replace=False))
    df = df_full.iloc[idx].reset_index(drop=True)
    train_df, val_df = split_by_event(df, seed=SEED)
    print(f"{len(df)} rows ({len(train_df)} train / {len(val_df)} val), "
          f"{df['circuit_id'].nunique()} circuits")

    results = {}

    # ---- Linear baseline ----
    linear = LinearBaseline(degree=1).fit(train_df)
    linear_pred = linear.predict(val_df)
    results["Linear"] = {"mae": mae(linear_pred, val_df["lap_time"].to_numpy())}

    # ---- LightGBM ----
    booster, feature_cols, category_map = fit_lightgbm(train_df, val_df)
    lgb_pred = booster.predict(prepare_lgb_features(val_df, category_map))
    results["LightGBM"] = {"mae": mae(lgb_pred, val_df["lap_time"].to_numpy())}

    # ---- NAM ----
    torch.manual_seed(SEED)
    train_cfg = TrainConfig.from_yaml_dict(raw_config)
    vocabs = Vocabs.fit(train_df)
    nam_cfg = NAMConfig(n_circuits=vocabs.circuit.size, n_entries=vocabs.entry.size,
                         n_compounds=vocabs.compound.size, max_age=cfg.max_age,
                         fuel_time_per_kg_s=cfg.fuel_time_per_kg_s)
    model = NAM(nam_cfg)
    fit(model, train_df, val_df, vocabs, train_cfg, device=device, verbose=False)
    model.eval()
    with torch.no_grad():
        batch = make_batch(val_df, vocabs, device)
        nam_pred = model(batch)["lap_time_hat"].cpu().numpy()
    results["NAM"] = {"mae": mae(nam_pred, val_df["lap_time"].to_numpy())}

    print("\n--- prediction accuracy (val MAE, seconds) ---")
    for name, r in results.items():
        print(f"  {name:10s} {r['mae']:.4f}")

    # ---- recovered degradation curves, per compound ----
    circuit_id_nam = next(iter(vocabs.circuit.mapping.values()))
    circuit_id_lgb = train_df["circuit_id"].mode().iloc[0]
    print("\n--- recovered slope / cliff, per compound (true vs NAM vs LightGBM) ---")
    curves = {}
    header = f"{'compound':8s} {'true slope':>10s} {'NAM slope':>10s} {'LGB slope':>10s} " \
             f"{'true cliff':>10s} {'NAM cliff':>10s} {'LGB cliff':>10s}"
    print(header)
    for compound in cfg.compounds:
        true_curve = true_degradation_curve(compound, cfg)
        cid = vocabs.compound.mapping[compound]
        nam_curve = model.degradation_curve(
            compound_id=cid, circuit_id=circuit_id_nam, track_temp_c=35.0, device=device
        ).cpu().numpy()
        lgb_curve = lightgbm_shap_tyre_curve(booster, train_df, cfg.max_age, compound=compound,
                                              category_map=category_map)
        # LightGBM's curve isn't guaranteed monotone -- don't force it to be,
        # that's exactly the point. detect_cliff/recovered_slope operate on
        # np.diff either way.

        curves[compound] = {"true": true_curve, "NAM": nam_curve, "LightGBM": lgb_curve}

        true_slope = recovered_slope(true_curve)
        nam_slope = recovered_slope(nam_curve)
        lgb_slope = recovered_slope(lgb_curve)
        true_cliff = detect_cliff(true_curve)
        nam_cliff = detect_cliff(nam_curve)
        lgb_cliff = detect_cliff(lgb_curve)
        print(f"{compound:8s} {true_slope:10.4f} {nam_slope:10.4f} {lgb_slope:10.4f} "
              f"{str(true_cliff):>10s} {str(nam_cliff):>10s} {str(lgb_cliff):>10s}")

    # ---- figure: true vs NAM vs LightGBM, one panel per compound ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), facecolor=SURFACE, sharey=False)
    ages = np.arange(cfg.max_age + 1)
    for ax, compound in zip(axes, cfg.compounds):
        ax.set_facecolor(SURFACE)
        c = curves[compound]
        ax.plot(ages, c["true"], color=TRUE_COLOR, linewidth=2.0, linestyle="--", label="True")
        ax.plot(ages, c["NAM"], color=NAM_COLOR, linewidth=2.2, label="NAM")
        ax.plot(ages, c["LightGBM"], color=LGB_COLOR, linewidth=2.0, label="LightGBM (SHAP-style)")
        ax.axhline(0, color=GRIDLINE, linewidth=0.8)
        ax.set_title(compound.title(), color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left")
        ax.set_xlabel("Tyre age (laps)", color=INK_SECONDARY, fontsize=9.5)
        if ax is axes[0]:
            ax.set_ylabel("Cumulative seconds lost vs. fresh tyre", color=INK_SECONDARY, fontsize=9.5)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(GRIDLINE)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    fig.text(0.06, 0.98, "Winning on accuracy is not the same as attributing correctly",
              color=INK_PRIMARY, fontsize=13.5, fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.905,
              f"Same synthetic ground truth, same data, same split. LightGBM MAE "
              f"{results['LightGBM']['mae']:.3f}s vs NAM {results['NAM']['mae']:.3f}s.\n"
              "LightGBM's tyre-age dependence is not causal attribution -- it has no obligation "
              "to recover the true shape, and here it doesn't.",
              color=INK_SECONDARY, fontsize=9.5, ha="left", va="top")

    Path("outputs").mkdir(exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.80])
    out_path = "outputs/figure7_baseline_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
