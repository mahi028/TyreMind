"""Tyre-age coverage histogram, per compound -- the cheap confirmation for the
Soft cliff-detection failure.

Hypothesis under test: Soft's cliff (true age 14) is undetected not because of
a model bug, but because real Soft stints are short and often run on scrubbed
tyres, leaving few real laps at low tyre age -- precisely where the Soft cliff
sits. Medium (cliff 22) and Hard (cliff 30) are detected correctly; if their
low-age coverage is denser than Soft's at the equivalent point, that supports
a data-coverage explanation rather than a model failure. If Soft's coverage
near age 14 turns out to be fine, this hypothesis is wrong and should not be
asserted in the deck.

Uses the REAL 2022-2023 corpus (not the semi-synthetic injected dataset) --
the question is about real-world data availability, not the synthetic test.

Sequential-by-compound small multiples, one hue per panel (not overlaid
categorical bars) -- three separate distributions are easier to read stacked
than superimposed, and the cliff-age reference line matters more than
cross-compound comparison at a glance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.data.clean import clean_corpus

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
COLORS = {"SOFT": "#2a78d6", "MEDIUM": "#eb6834", "HARD": "#1baf7a"}
CLIFF_AGE = {"SOFT": 14, "MEDIUM": 22, "HARD": 30}


def main():
    raw_cfg = yaml.safe_load(open("config/default.yaml"))
    print("cleaning real corpus...")
    real_covariates, _ = clean_corpus(Path("data/raw"))
    print(f"{len(real_covariates)} real laps, {real_covariates['compound'].value_counts().to_dict()}")

    compounds = ["SOFT", "MEDIUM", "HARD"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), facecolor=SURFACE, sharey=False)

    max_age_shown = 45
    bins = np.arange(0, max_age_shown + 2) - 0.5

    coverage_below_cliff = {}
    for ax, compound in zip(axes, compounds):
        ax.set_facecolor(SURFACE)
        ages = real_covariates.loc[real_covariates["compound"] == compound, "tyre_age"].to_numpy()
        ages = ages[np.isfinite(ages)]
        color = COLORS[compound]
        cliff = CLIFF_AGE[compound]

        ax.hist(ages, bins=bins, color=color, alpha=0.85, linewidth=0)
        ax.axvline(cliff, color=INK_PRIMARY, linewidth=1.4, linestyle="--")
        ax.text(cliff + 0.8, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                f"true cliff\n(age {cliff})", color=INK_SECONDARY, fontsize=8, va="top")

        n_below = int((ages < cliff).sum())
        n_total = len(ages)
        pct_below = n_below / n_total if n_total else float("nan")
        coverage_below_cliff[compound] = (n_below, n_total, pct_below)

        ax.set_title(f"{compound.title()}", color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left")
        ax.set_xlabel("Tyre age (laps)", color=INK_SECONDARY, fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Real laps observed", color=INK_SECONDARY, fontsize=9)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(GRIDLINE)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max_age_shown)

    fig.text(0.06, 0.97, "Real tyre-age coverage by compound",
              color=INK_PRIMARY, fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.905,
              "Laps observed at each tyre age, real 2022-2023 race corpus. Dashed line = each "
              "compound's true synthetic cliff age.\nSparse bars left of the line mean few "
              "observations exactly where a cliff would need to be detected.",
              color=INK_SECONDARY, fontsize=9, ha="left", va="top")

    Path("outputs").mkdir(exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.80])
    out_path = "outputs/figure_tyre_age_coverage.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")

    print("\ncoverage below each compound's true cliff age:")
    for compound in compounds:
        n_below, n_total, pct = coverage_below_cliff[compound]
        print(f"  {compound:8s} cliff={CLIFF_AGE[compound]:2d}  laps below cliff={n_below:5d} / {n_total:5d} total  ({pct:.1%})")


if __name__ == "__main__":
    main()
