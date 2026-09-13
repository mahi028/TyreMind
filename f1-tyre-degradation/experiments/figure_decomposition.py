"""Phase 7, figure 5 — additive decomposition for one real stint.

Fits the model on real 2022-2023 race covariates with known synthetic wear
injected (`src/sim/semi_synthetic.py`), picks one real stint, and plots how the
model splits its lap times into fuel / tyre / evolution / conditions / traffic.
This is the figure that makes "stripping out the noise" visible in one look.

Palette: validated categorical order (dataviz skill), assigned by fixed slot,
never cycled: blue=fuel, orange=tyre, aqua=evolution, yellow=conditions,
magenta=traffic. Text stays in ink colors; only the stacked segments and the
actual/predicted lines carry series color.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data.clean import clean_corpus
from src.data.features import split_by_event
from src.models.dataset import Vocabs, make_batch
from src.models.nam import NAM, NAMConfig
from src.sim.semi_synthetic import build_semi_synthetic_dataset
from src.sim.simulator import SimulatorConfig
from src.train import TrainConfig, fit, pick_device

# Fixed categorical slots (never reassigned/cycled) -- dataviz skill palette.
COLOR_FUEL = "#2a78d6"      # slot 1 blue
COLOR_TYRE = "#eb6834"      # slot 2 orange
COLOR_EVO = "#1baf7a"       # slot 3 aqua
COLOR_COND = "#eda100"      # slot 4 yellow
COLOR_TRAFFIC = "#e87ba4"   # slot 5 magenta
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


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
    print("fitting model on real covariates...")
    fit(model, train_df, val_df, vocabs, train_cfg, device=device, verbose=False)
    model.eval()

    # Pick the longest surviving MEDIUM-compound stint. Not Hard: Hard's true
    # wear rate (0.035 s/lap) sits inside the measured detection floor (see
    # FINDINGS.md #3 and figure_detection_floor.py) -- the weakest signal of
    # the three, a poor choice for the flagship figure. Not Soft either: cliff
    # detection is currently unexplained-broken for Soft (see
    # figure_tyre_age_coverage.py) even though a coverage check ruled out the
    # obvious "sparse low-age data" explanation -- putting a known-wrong curve
    # in the flagship figure would undercut the rest of the deck. Medium's
    # cliff is detected correctly (23 vs. true 22) and its wear rate clears
    # the floor by ~2.5x, and its magnitude sits between Fuel and Hard's, so
    # the fuel/tyre cancellation the subtitle describes reads more clearly.
    FLAGSHIP_COMPOUND = "MEDIUM"
    df_compound = df[df["compound"] == FLAGSHIP_COMPOUND]
    stint_sizes = df_compound.groupby(["event_id", "driver_id", "stint_id"]).size()
    best_key = stint_sizes.idxmax()
    stint = df[
        (df.event_id == best_key[0]) & (df.driver_id == best_key[1]) & (df.stint_id == best_key[2])
    ].sort_values("tyre_age").reset_index(drop=True)
    print(f"selected stint: {len(stint)} laps, compound={stint['compound'].iloc[0]}, "
          f"tyre_age {stint['tyre_age'].min():.0f}-{stint['tyre_age'].max():.0f}")

    with torch.no_grad():
        batch = make_batch(stint, vocabs, device)
        out = model(batch)

    laps = stint["tyre_age"].to_numpy()
    fuel = out["fuel"].cpu().numpy()
    tyre = out["tyre"].cpu().numpy()
    evo = out["evo"].cpu().numpy()
    cond = out["cond"].cpu().numpy()
    traffic = out["traffic"].cpu().numpy()
    baseline = (out["circuit"] + out["entry"]).cpu().numpy()
    predicted = out["lap_time_hat"].cpu().numpy()
    actual = stint["lap_time"].to_numpy()

    # Stack the five varying components around the (roughly constant) baseline,
    # in fixed slot order: fuel, tyre, evo, cond, traffic.
    fig, ax = plt.subplots(figsize=(9.5, 6.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    components = [
        ("Fuel burn-off", fuel, COLOR_FUEL),
        ("Tyre wear", tyre, COLOR_TYRE),
        ("Track evolution", evo, COLOR_EVO),
        ("Conditions", cond, COLOR_COND),
        ("Traffic", traffic, COLOR_TRAFFIC),
    ]

    cum_pos = np.zeros(len(stint))
    cum_neg = np.zeros(len(stint))
    for name, values, color in components:
        top = cum_pos + np.clip(values, 0, None)
        bottom = cum_neg + np.clip(values, None, 0)
        # Stack positive contributions upward, negative ones downward, each
        # relative to the baseline -- a literal additive decomposition, not a
        # cosmetic stack.
        y0 = baseline + np.where(values >= 0, cum_pos, cum_neg + values)
        y1 = baseline + np.where(values >= 0, cum_pos + values, cum_neg)
        ax.fill_between(laps, y0, y1, color=color, alpha=0.85, linewidth=0, label=name)
        cum_pos = np.where(values >= 0, cum_pos + values, cum_pos)
        cum_neg = np.where(values < 0, cum_neg + values, cum_neg)

    ax.plot(laps, baseline, color=INK_MUTED, linewidth=1.5, linestyle=":", label="Baseline (circuit + car/driver)")
    ax.plot(laps, actual, color=INK_PRIMARY, linewidth=2.2, label="Actual lap time")
    ax.plot(laps, predicted, color=INK_PRIMARY, linewidth=1.4, linestyle="--", label="Model prediction (sum of all terms)")

    # Explicit margins, rather than relying on default autoscale: the stacked
    # bands' cumulative extent is largest at the oldest tyre age (the right
    # edge), and the default 5% autoscale margin wasn't enough headroom there
    # -- the top band was getting visually clipped against the axes frame.
    ax.margins(x=0.02, y=0.12)

    ax.set_xlabel("Tyre age (laps)", color=INK_SECONDARY)
    ax.set_ylabel("Lap time (s)", color=INK_SECONDARY)
    fig.text(0.06, 0.97, f"Where the lap time goes — one real stint, {stint['compound'].iloc[0]} tyre",
             color=INK_PRIMARY, fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.925, "Fuel burn-off and track evolution pull the lap faster; tyre wear pulls it slower.\n"
                          "The actual time is the net of both — which is why a stopwatch alone can hide real degradation.",
             color=INK_SECONDARY, fontsize=9.5, ha="left", va="top")
    ax.tick_params(colors=INK_SECONDARY)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRIDLINE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    # 8 series is too many for a single-column in-axes legend without covering
    # data -- 4 columns in a strip below the plot instead.
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4,
        frameon=False, fontsize=7.5, labelcolor=INK_SECONDARY, columnspacing=1.2, handlelength=1.5,
    )

    Path("outputs").mkdir(exist_ok=True)
    out_path = "outputs/figure5_additive_decomposition.png"
    fig.tight_layout(rect=[0, 0.09, 1, 0.80])
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
