"""Detection-floor chart: each compound's true wear rate against the measured
noise floor (`FINDINGS.md` #3 — 0.022 +/- 0.013 s/lap, max 0.043, n=8 seeds at
~13,300 laps). Answers "which of these curves can we actually trust" in one
look, rather than presenting all three with equal, unearned confidence.

Sequential encoding (magnitude: floor band vs. true rate), not categorical --
one hue (blue), light shaded band for the floor, a single dark step for the
"max observed" edge. Bars use the muted/ink palette, not a competing hue, since
the floor is the only thing being compared against.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
BAND_FILL = "#cde2fb"    # sequential step 100 (blue ramp) -- the floor band
BAND_EDGE = "#3987e5"    # sequential step 400 -- max-observed edge
BAR_COLOR = "#52514e"    # ink, not a competing categorical hue

# From FINDINGS.md #3 -- measured across 8 seeds at ~13,300 laps, post-fix.
FLOOR_MEAN = 0.022
FLOOR_STD = 0.013
FLOOR_MAX = 0.043

TRUE_RATES = {"Soft": 0.090, "Medium": 0.055, "Hard": 0.035}


def main():
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    compounds = list(TRUE_RATES.keys())
    values = list(TRUE_RATES.values())
    x = range(len(compounds))

    ax.axhspan(0, FLOOR_MAX, color=BAND_FILL, alpha=0.6, zorder=0)
    ax.axhspan(0, FLOOR_MEAN + FLOOR_STD, color=BAND_FILL, alpha=0.9, zorder=0)
    ax.axhline(FLOOR_MEAN, color=BAND_EDGE, linewidth=1.6, linestyle="-", zorder=1)
    ax.axhline(FLOOR_MAX, color=BAND_EDGE, linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)

    bars = ax.bar(x, values, width=0.5, color=BAR_COLOR, zorder=2)
    for rect, val in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.003, f"{val:.3f}",
                ha="center", va="bottom", color=INK_PRIMARY, fontsize=10, fontweight="bold")

    ax.text(2.55, FLOOR_MEAN, "measured floor\n(mean 0.022)", color=BAND_EDGE, fontsize=8.5,
            ha="left", va="center")
    ax.text(2.55, FLOOR_MAX, "max observed (0.043)", color=BAND_EDGE, fontsize=8, alpha=0.8,
            ha="left", va="center")

    ax.set_xticks(list(x))
    ax.set_xticklabels(compounds, color=INK_SECONDARY, fontsize=11)
    ax.set_ylabel("True wear rate (s/lap)", color=INK_SECONDARY)
    ax.set_ylim(0, 0.11)
    ax.set_xlim(-0.6, 3.3)
    ax.tick_params(colors=INK_SECONDARY)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRIDLINE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.text(0.06, 0.97, "Which wear rates can we actually trust?", color=INK_PRIMARY,
              fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.925,
              "The estimator's own noise floor, measured on synthetic data with zero true wear.\n"
              "Hard's true rate sits inside it — not reliably distinguishable from noise at this data volume.",
              color=INK_SECONDARY, fontsize=9.5, ha="left", va="top")

    Path("outputs").mkdir(exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.80])
    out_path = "outputs/figure_detection_floor.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
