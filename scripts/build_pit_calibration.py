"""Fit the pit-window conformal threshold and ship it as an artefact.

The window served until now was two laps either side of the recommendation with
its stated probability taken from a softmax mass. Both were choices: the width
because teams talk in windows of about that size, the softmax temperature because
it was dimensionally sensible. exp30 measured what they were worth -- 31.9%
claimed against 24.0% delivered.

Split conformal replaces both with a measured guarantee. The score is how many
laps a recommendation missed by; the threshold is the finite-sample-corrected
quantile of those misses. It assumes nothing about the error's shape, which
matters because the misses are skewed: teams pit earlier than pure degradation
cost implies, for track position we do not model.

Fitted on exp30's stops and written to `data/reference/pit_calibration.json` so
the API serves a number that was measured rather than chosen.

    python scripts/build_pit_calibration.py --target 0.8
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tyremind.models.pit_decision import PitWindowCalibrator

SOURCE = Path("experiments/results/exp30_pit_confidence_calibration.json")
OUT = Path("data/reference/pit_calibration.json")
MODEL = "TyreMind state-space"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=float, nargs="*", default=[0.5, 0.8, 0.9])
    parser.add_argument("--splits", type=int, default=200)
    parser.add_argument("--seed", type=int, default=30)
    args = parser.parse_args()

    if not SOURCE.exists():
        raise SystemExit(f"run exp30 first; {SOURCE} missing")

    rows = pd.DataFrame(json.loads(SOURCE.read_text(encoding="utf-8"))["rows"])
    block = rows[rows["model"] == MODEL]
    if len(block) < 40:
        raise SystemExit(f"only {len(block)} stops for {MODEL}; too few to calibrate")

    rng = np.random.default_rng(args.seed)
    entries = {}
    for target in args.target:
        # Held-out coverage, averaged over repeated splits, so the number shipped
        # is what the window achieves on stops it was not fitted on.
        covers, widths = [], []
        for _ in range(args.splits):
            order = rng.permutation(len(block))
            half = len(order) // 2
            calibration = block.iloc[order[:half]]
            held_out = block.iloc[order[half:]]
            fitted = PitWindowCalibrator(target).fit(
                calibration["recommended"].to_numpy(), calibration["actual"].to_numpy())
            if not fitted.fitted:
                continue
            miss = np.abs(held_out["recommended"].to_numpy() - held_out["actual"].to_numpy())
            covers.append(float(np.mean(miss <= fitted._half_width)))
            widths.append(float(fitted._half_width))

        # The shipped threshold is fitted on everything; the coverage beside it is
        # the held-out figure, so the number we claim is the one we measured.
        full = PitWindowCalibrator(target).fit(
            block["recommended"].to_numpy(), block["actual"].to_numpy())
        entries[f"{target:.2f}"] = {
            "target_coverage": target,
            "half_width_laps": float(full._half_width) if full.fitted else None,
            "measured_coverage_held_out": float(np.mean(covers)) if covers else None,
            "mean_half_width_held_out": float(np.mean(widths)) if widths else None,
            "n_calibration": int(len(block)),
        }
        e = entries[f"{target:.2f}"]
        print(f"  target {target:.0%}  ->  +-{e['half_width_laps']:.1f} laps, "
              f"held-out coverage {e['measured_coverage_held_out']:.1%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "_note": (
            "Conformal pit-window thresholds. `half_width_laps` is fitted on all "
            "stops; `measured_coverage_held_out` is what that width achieved on "
            "stops it was not fitted on, averaged over repeated splits. The "
            "coverage we claim is the one we measured."
        ),
        "model": MODEL,
        "source": str(SOURCE),
        "generated_at": datetime.now(UTC).isoformat(),
        "n_splits": args.splits,
        "entries": entries,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
