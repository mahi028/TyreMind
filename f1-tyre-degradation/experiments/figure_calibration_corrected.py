"""Final calibration figure: raw vs. corrected practice-to-race scatter.

Calibration factor c=2.04 fit as a ratio-of-means on the 28 events NOT
already flagged as track-temp-extrapolation artifacts (`FINDINGS.md` #6),
with a bootstrap 95% CI of [1.25, 3.87] (resampling events, not rows, since
points within an event share a practice fit and a weather day). The interval
excludes 1 -- the under-prediction is real -- but is wide: with 28 real
events, the exact magnitude is not pinned down tighter than roughly 1.25x-3.9x.

Both panels share a FIXED, clipped axis range (not autoscaled to fit the two
known outlier events) -- multiplying by 2.04 pushes those two events' already-
extreme values to ~1.1, and autoscaling to that crushes all 73 normal points
into a sliver of the frame, hiding the exact improvement the figure exists to
show. The two outlier events are instead drawn at the clipped edge with an
arrow and their true value labeled, same convention as marking a value
"off-scale" on any bounded chart.

The two best-fit lines are labeled with exactly which points they're fit on
(28 non-outlier events, not all 79) -- Figure 6's headline 0.23 and this
figure's 0.20 are different subsets, not a typo.
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
COLORS = {"SOFT": "#2a78d6", "MEDIUM": "#eb6834", "HARD": "#1baf7a"}
OUTLIER_EVENTS = {"2022_4_R", "2023_16_R"}
CALIBRATION_FACTOR = 2.037
AXIS_LO, AXIS_HI = -0.02, 0.35


def _scatter_panel(ax, df, y_col, title, n_fit_label: str):
    ax.plot([AXIS_LO, AXIS_HI], [AXIS_LO, AXIS_HI], color=INK_MUTED, linewidth=1.2, linestyle="--",
            label="Perfect agreement")

    normal = df[~df["event_id"].isin(OUTLIER_EVENTS)]
    outliers = df[df["event_id"].isin(OUTLIER_EVENTS)]
    for compound, group in normal.groupby("compound"):
        ax.scatter(group["observed_slope"], group[y_col], color=COLORS.get(compound, INK_SECONDARY),
                   s=36, alpha=0.85, edgecolor="white", linewidth=0.5, label=compound.title())

    # Outlier events: draw as a triangle marker pinned to the top edge (not
    # autoscaled out to their true, much larger value), with a compact text
    # block off to the side giving the exact numbers -- six individual
    # arrow/text callouts here overlapped and were unreadable.
    clip_y = AXIS_HI - 0.015
    if len(outliers):
        for compound, group in outliers.groupby("compound"):
            xs_o = group["observed_slope"].clip(AXIS_LO, AXIS_HI)
            ax.scatter(xs_o, [clip_y] * len(group), marker="^", s=70,
                       color=COLORS.get(compound, INK_SECONDARY), edgecolor=INK_PRIMARY, linewidth=0.8, zorder=6)
        ax.scatter([], [], marker="^", color="none", edgecolor=INK_PRIMARY, s=70, linewidth=0.8,
                   label="Known track-temp outlier events (pinned to top edge)")

        lines = []
        for event_id, group in outliers.groupby("event_id"):
            vals = "  ".join(f"{c[0]}{v:.2f}" for c, v in zip(group["compound"], group[y_col]))
            label = "2022 R4" if event_id == "2022_4_R" else "2023 R16"
            lines.append(f"{label}: {vals}")
        ax.text(0.98, 0.99, "off-scale values\n" + "\n".join(lines), transform=ax.transAxes,
                ha="right", va="top", fontsize=7.3, color=INK_SECONDARY, linespacing=1.5)

    if len(normal) >= 2:
        s, i = np.polyfit(normal["observed_slope"], normal[y_col], 1)
        xs = np.array([AXIS_LO, AXIS_HI])
        ax.plot(xs, s * xs + i, color=INK_PRIMARY, linewidth=1.6,
                label=f"Best fit, {n_fit_label} (slope={s:.2f})")

    ax.set_xlim(AXIS_LO, AXIS_HI)
    ax.set_ylim(AXIS_LO, AXIS_HI)
    ax.set_xlabel("Observed race slope, fuel-corrected (s/lap)", color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylabel(("Corrected" if "Corrected" in title else "Raw") + " predicted slope (s/lap)",
                   color=INK_SECONDARY, fontsize=9.5)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRIDLINE)
    ax.grid(color=GRIDLINE, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=7.3, labelcolor=INK_SECONDARY)


def main():
    df = pd.read_csv("outputs/phase6_race_validation.csv")
    df["corrected_slope"] = df["predicted_slope"] * CALIBRATION_FACTOR

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.5), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    n_events = df.loc[~df["event_id"].isin(OUTLIER_EVENTS), "event_id"].nunique()
    n_points = (~df["event_id"].isin(OUTLIER_EVENTS)).sum()
    fit_label = f"{n_events} events, {n_points} pts"
    _scatter_panel(axes[0], df, "predicted_slope", "Raw (as fit)", n_fit_label=fit_label)
    _scatter_panel(axes[1], df, "corrected_slope", f"Corrected (x{CALIBRATION_FACTOR:.2f})",
                    n_fit_label=fit_label)

    fig.text(0.06, 0.98, "Calibrating the practice-to-race compression",
              color=INK_PRIMARY, fontsize=13.5, fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.915,
              "A single scale factor (2.04x, bootstrap 95% CI [1.25, 3.87] across 28 non-outlier real events) "
              "moves the typical event onto the line.\nAxes clipped to 0.35 s/lap -- the two known track-temp-"
              "extrapolation events are labeled off-scale, not autoscaled into crushing everything else.\n"
              "Figure 6's headline slope (0.23) is fit on all 79 points; the 0.20/0.41 slopes here are fit on "
              "the 28 non-outlier events only -- different subsets, not a discrepancy.",
              color=INK_SECONDARY, fontsize=9, ha="left", va="top")

    Path("outputs").mkdir(exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.76])
    out_path = "outputs/figure6b_calibration_corrected.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
