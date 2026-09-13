"""Calibration attempt for the Phase 6 practice-to-race compression, with the
generalization discipline this project has applied to every other tuned
value: fit the correction on one subset of real events, check it holds on a
DIFFERENT subset before trusting it.

Two things forced this to be more careful than "just fit a scale factor":

1. Pooled correlation between predicted and observed event-level slopes is
   weak (Pearson r=0.11, p=0.33 -- not distinguishable from zero) and the
   original regression slope (0.23) is fully explained by that weak
   correlation times a 2.1x variance mismatch (predicted is noisier than
   observed) -- `slope = r * sd(pred)/sd(obs)`, exactly. A least-squares
   scale factor computed the same way inherits the same distortion.
2. A ratio-of-means (robust to that distortion) computed on the FULL 79
   points is only 1.22 -- barely any correction needed -- but jumps to 2.04
   once the two track-temp-extrapolation outlier events are excluded. Which
   number is "the" calibration factor depends heavily on which events are in
   the sample.

This script checks (2) directly: fit a ratio-of-means calibration factor on a
random 60% of events, and see whether it predicts the correction actually
needed on the other 40% -- repeated across many splits, not just one.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
BAND_FILL = "#cde2fb"
LINE_COLOR = "#2a78d6"

N_SPLITS = 200
FIT_FRACTION = 0.6


def main():
    df = pd.read_csv("outputs/phase6_race_validation.csv")
    events = np.array(sorted(df["event_id"].unique()))

    fit_factors = []
    needed_ratios = []
    for seed in range(N_SPLITS):
        rng = np.random.default_rng(seed)
        shuffled = events.copy()
        rng.shuffle(shuffled)
        n_fit = int(len(shuffled) * FIT_FRACTION)
        fit_events, held_events = set(shuffled[:n_fit]), set(shuffled[n_fit:])
        fit_df = df[df.event_id.isin(fit_events)]
        held_df = df[df.event_id.isin(held_events)]

        c = fit_df["observed_slope"].mean() / fit_df["predicted_slope"].mean()
        needed = held_df["observed_slope"].mean() / held_df["predicted_slope"].mean()
        fit_factors.append(c)
        needed_ratios.append(needed)

    fit_factors = np.array(fit_factors)
    needed_ratios = np.array(needed_ratios)
    # How far off would applying the fit-set's factor be, on the held-out set
    # it wasn't fit on -- 1.0 would mean a perfect transfer.
    transfer_ratio = fit_factors / needed_ratios

    print(f"across {N_SPLITS} random 60/40 event splits:")
    print(f"  fit-set calibration factor:  mean={fit_factors.mean():.3f}  "
          f"std={fit_factors.std():.3f}  range=[{fit_factors.min():.2f}, {fit_factors.max():.2f}]")
    print(f"  held-out factor ACTUALLY needed: mean={needed_ratios.mean():.3f}  "
          f"std={needed_ratios.std():.3f}  range=[{needed_ratios.min():.2f}, {needed_ratios.max():.2f}]")
    print(f"  direction check: held-out needed > 1 (under-prediction) in "
          f"{(needed_ratios > 1).mean():.0%} of splits")
    print(f"  transfer quality: fit-set factor / held-out-needed factor, "
          f"mean={transfer_ratio.mean():.2f}, std={transfer_ratio.std():.2f} "
          f"(1.0 = perfect transfer; this wide a spread means a single global "
          f"factor fit on 60% of today's 30 events does not reliably predict "
          f"the correction the other 40% actually needs)")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), facecolor=SURFACE)

    ax = axes[0]
    ax.set_facecolor(SURFACE)
    ax.hist(needed_ratios, bins=20, color=LINE_COLOR, alpha=0.85, linewidth=0)
    ax.axvline(1.0, color=INK_PRIMARY, linewidth=1.4, linestyle="--")
    ax.text(1.02, ax.get_ylim()[1] * 0.95, "no correction\nneeded", color=INK_SECONDARY, fontsize=8, va="top")
    ax.set_xlabel("Calibration factor actually needed on the held-out 40%", color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylabel("Splits (of 200)", color=INK_SECONDARY, fontsize=9.5)
    ax.set_title(
        f"The correction needed varies {needed_ratios.min():.1f}x–{needed_ratios.max():.1f}x "
        "depending on which\nevents are held out",
        color=INK_PRIMARY, fontsize=10.5, fontweight="bold", loc="left",
    )

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    ax.scatter(fit_factors, needed_ratios, color=LINE_COLOR, s=18, alpha=0.5, edgecolor="none")
    lo = min(fit_factors.min(), needed_ratios.min()) * 0.9
    hi = max(fit_factors.max(), needed_ratios.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, linewidth=1.2, linestyle="--", label="Perfect transfer")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Factor fit on 60% of events", color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylabel("Factor actually needed on the other 40%", color=INK_SECONDARY, fontsize=9.5)
    ax.set_title("A factor fit on one subset of events does not\npredict what the rest of the season needs",
                  color=INK_PRIMARY, fontsize=10.5, fontweight="bold", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)

    for ax in axes:
        ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(GRIDLINE)
        ax.grid(color=GRIDLINE, linewidth=0.7)
        ax.set_axisbelow(True)

    fig.text(0.06, 0.98, "Calibration was attempted, and it does not hold up out of sample",
              color=INK_PRIMARY, fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.905,
              f"A single global scale factor fit on 60% of the 30 available real events, "
              f"checked against the other 40%, across {N_SPLITS} random splits.\n"
              "The direction of under-prediction is consistent; the size of the correction is not "
              "yet stable at this data volume (30 real races).",
              color=INK_SECONDARY, fontsize=9.5, ha="left", va="top")

    Path("outputs").mkdir(exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.80])
    out_path = "outputs/figure_calibration_instability.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
