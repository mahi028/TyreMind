"""Phase 7, figure 1 — per-compound degradation curves with uncertainty bands.

Ensembles 8 random seeds (Section 5's prescribed approach: spread across seeds
is the uncertainty band) on matched-size synthetic data (13,388 rows, 22
circuits — the same scale as the real 2022-2023 corpus, so the bands shown are
representative of what real data currently supports, not the much larger
40k-row synthetic default). True curves overlaid as a dashed reference, since
this is a recovery check as much as a results figure.

Categorical hues assigned by fixed slot, one per compound, never cycled:
Soft=blue (slot 1), Medium=orange (slot 2), Hard=aqua (slot 3) -- these three
slots validate all-pairs in both light and dark, per the dataviz skill.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

import src.sim.simulator as sim_mod
from src.sim.simulator import SimulatorConfig, generate_synthetic_dataset, true_degradation_curve
from src.data.features import split_by_event
from src.models.dataset import Vocabs
from src.models.nam import NAM, NAMConfig
from src.train import TrainConfig, fit, pick_device

import matplotlib.pyplot as plt

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
COLORS = {"SOFT": "#2a78d6", "MEDIUM": "#eb6834", "HARD": "#1baf7a"}

TARGET_ROWS = 13388
N_SEEDS = 8


def main():
    sim_mod.N_CIRCUITS = 22
    cfg = SimulatorConfig.from_yaml()
    cfg.n_sessions = 30
    raw_config = yaml.safe_load(open("config/default.yaml"))
    device = pick_device()

    curves_by_compound = {c: [] for c in cfg.compounds}
    true_curves = {c: true_degradation_curve(c, cfg) for c in cfg.compounds}

    for i in range(1, N_SEEDS + 1):
        seed = 2000 + i
        df_full = generate_synthetic_dataset(cfg, seed=seed)
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(df_full), size=min(TARGET_ROWS, len(df_full)), replace=False))
        df = df_full.iloc[idx].reset_index(drop=True)
        train_df, val_df = split_by_event(df, seed=seed)

        torch.manual_seed(seed)
        train_cfg = TrainConfig.from_yaml_dict(raw_config)
        vocabs = Vocabs.fit(train_df)
        nam_cfg = NAMConfig(n_circuits=vocabs.circuit.size, n_entries=vocabs.entry.size,
                             n_compounds=vocabs.compound.size, max_age=cfg.max_age,
                             fuel_time_per_kg_s=cfg.fuel_time_per_kg_s)
        model = NAM(nam_cfg)
        fit(model, train_df, val_df, vocabs, train_cfg, device=device, verbose=False)
        model.eval()

        circuit_id = next(iter(vocabs.circuit.mapping.values()))
        for compound in cfg.compounds:
            cid = vocabs.compound.mapping[compound]
            curve = model.degradation_curve(
                compound_id=cid, circuit_id=circuit_id, track_temp_c=35.0, device=device
            ).cpu().numpy()
            curves_by_compound[compound].append(curve)
        print(f"seed {i}/{N_SEEDS} done")

    fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ages = np.arange(cfg.max_age + 1)

    for compound in cfg.compounds:
        stack = np.stack(curves_by_compound[compound])  # (n_seeds, max_age+1)
        median = np.median(stack, axis=0)
        lo = np.percentile(stack, 12.5, axis=0)
        hi = np.percentile(stack, 87.5, axis=0)
        color = COLORS[compound]

        ax.fill_between(ages, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(ages, median, color=color, linewidth=2.2, label=f"{compound.title()} — recovered")
        ax.plot(ages, true_curves[compound], color=color, linewidth=1.3, linestyle="--", alpha=0.75,
                label=f"{compound.title()} — true")

    ax.set_xlabel("Tyre age (laps)", color=INK_SECONDARY)
    ax.set_ylabel("Cumulative seconds lost vs. a fresh tyre", color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRIDLINE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, ncol=1)

    fig.text(0.06, 0.97, "Recovered tyre degradation curves, 8-seed ensemble",
              color=INK_PRIMARY, fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.925,
              "Shaded band = spread across 8 random seeds at real-world data volume (~13,300 laps).\n"
              "Dashed lines are the known true curves this synthetic test was built to recover.",
              color=INK_SECONDARY, fontsize=9.5, ha="left", va="top")

    Path("outputs").mkdir(exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    out_path = "outputs/figure1_degradation_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
