"""Pooled residual-vs-tyre-age check, one panel per compound.

Figure 5 (one MEDIUM stint) showed a directional pattern, not scatter: the
model's prediction sits above actual early in the stint and below actual late
in the stint -- a compression of the stint's real dynamics into a narrower
band than the data shows. Same directional signature as the measured Medium
slope undershoot (0.023 s/lap under true). This script checks whether that's
one stint's noise or a systematic, pooled-across-stints bias: residual
(actual - predicted) binned by tyre age, across every real stint in the
semi-synthetic fit, faceted by compound.

A flat-at-zero band across age = no systematic bias, Figure 5's pattern was
this stint's noise. A consistent positive-then-negative tilt (or U-shape),
pooled across hundreds of stints, is a quantified systematic bias worth
reporting alongside the detection floor -- not a new bug, the visual/pooled
confirmation of a number already measured.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data.clean import clean_corpus
from src.data.features import split_by_event
from src.models.dataset import Vocabs, make_batch, make_target
from src.models.nam import NAM, NAMConfig
from src.sim.semi_synthetic import build_semi_synthetic_dataset
from src.sim.simulator import SimulatorConfig
from src.train import TrainConfig, fit, pick_device

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
COLORS = {"SOFT": "#2a78d6", "MEDIUM": "#eb6834", "HARD": "#1baf7a"}

MAX_AGE_SHOWN = 40


def main():
    cfg = SimulatorConfig.from_yaml()
    raw_config = yaml.safe_load(open("config/default.yaml"))
    device = pick_device()

    print("cleaning real corpus...")
    real_covariates, _ = clean_corpus(Path("data/raw"))
    df, truth = build_semi_synthetic_dataset(real_covariates, cfg, seed=42)

    torch.manual_seed(42)
    train_df, val_df = split_by_event(df, val_fraction=0.15, seed=42)
    vocabs = Vocabs.fit(train_df)
    nam_cfg = NAMConfig(
        n_circuits=vocabs.circuit.size, n_entries=vocabs.entry.size,
        n_compounds=vocabs.compound.size, max_age=cfg.max_age,
        fuel_time_per_kg_s=cfg.fuel_time_per_kg_s,
    )
    model = NAM(nam_cfg)
    train_cfg = TrainConfig.from_yaml_dict(raw_config)
    print("fitting model on real covariates (same setup as figure_decomposition.py)...")
    fit(model, train_df, val_df, vocabs, train_cfg, device=device, verbose=False)
    model.eval()

    # Pooled residuals across BOTH splits -- the question is whether the bias
    # is a property of the fitted function's shape, which shows up regardless
    # of train/val membership, not a train-only overfitting artifact.
    full = df.reset_index(drop=True)
    with torch.no_grad():
        batch = make_batch(full, vocabs, device)
        out = model(batch)
        predicted = out["lap_time_hat"].cpu().numpy()
    actual = full["lap_time"].to_numpy()
    residual = actual - predicted
    full = full.assign(residual=residual)
    full = full[full["tyre_age"] <= MAX_AGE_SHOWN]

    compounds = ["SOFT", "MEDIUM", "HARD"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), facecolor=SURFACE, sharey=True)

    bin_edges = np.arange(0, MAX_AGE_SHOWN + 2)
    for ax, compound in zip(axes, compounds):
        ax.set_facecolor(SURFACE)
        sub = full[full["compound"] == compound]
        color = COLORS[compound]

        ages = sub["tyre_age"].to_numpy()
        res = sub["residual"].to_numpy()
        bin_idx = np.digitize(ages, bin_edges) - 1
        n_bins = len(bin_edges) - 1
        bin_mean = np.full(n_bins, np.nan)
        bin_se = np.full(n_bins, np.nan)
        bin_n = np.zeros(n_bins, dtype=int)
        for b in range(n_bins):
            vals = res[bin_idx == b]
            bin_n[b] = len(vals)
            if len(vals) >= 5:
                bin_mean[b] = vals.mean()
                bin_se[b] = vals.std(ddof=1) / np.sqrt(len(vals))

        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        valid = ~np.isnan(bin_mean)
        ax.axhline(0.0, color=INK_MUTED, linewidth=1.0, linestyle="-")
        ax.fill_between(
            centers[valid], (bin_mean - 1.96 * bin_se)[valid], (bin_mean + 1.96 * bin_se)[valid],
            color=color, alpha=0.20, linewidth=0,
        )
        ax.plot(centers[valid], bin_mean[valid], color=color, linewidth=2.0)

        ax.set_title(f"{compound.title()}  (n={len(sub)})", color=INK_PRIMARY, fontsize=11,
                      fontweight="bold", loc="left")
        ax.set_xlabel("Tyre age (laps)", color=INK_SECONDARY, fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Residual: actual − predicted (s)", color=INK_SECONDARY, fontsize=9)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(GRIDLINE)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, MAX_AGE_SHOWN)

    fig.text(0.06, 0.97, "Is the stint-compression pattern systematic, or one stint's noise?",
              color=INK_PRIMARY, fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.905,
              "Mean residual (actual − predicted) by tyre age, pooled across every real stint, "
              "95% CI band.\nA positive-then-negative tilt across all three compounds means the "
              "model systematically compresses the stint.",
              color=INK_SECONDARY, fontsize=9, ha="left", va="top")

    Path("outputs").mkdir(exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.80])
    out_path = "outputs/figure_residual_bias.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")

    # Numeric summary: early-stint vs late-stint mean residual, per compound.
    print("\nearly (age<=8) vs late (age>=25) mean residual, per compound:")
    for compound in compounds:
        sub = full[full["compound"] == compound]
        early = sub.loc[sub["tyre_age"] <= 8, "residual"]
        late = sub.loc[sub["tyre_age"] >= 25, "residual"]
        e_mean = early.mean() if len(early) else float("nan")
        l_mean = late.mean() if len(late) else float("nan")
        print(f"  {compound:8s} early(n={len(early):5d}): {e_mean:+.3f}s   "
              f"late(n={len(late):5d}): {l_mean:+.3f}s")


if __name__ == "__main__":
    main()
