"""Turn a recorded F1-game telemetry capture into a second fuel-truth benchmark.

`tyremind.data.synthetic` generates the platform's only current ground-truth
benchmark, under this project's own assumptions. This script builds a second
one from a genuinely different source: a real F1 24/23 session, recorded with
`pahansen/f1-telemetry` (github.com/pahansen/f1-telemetry) or a byte-compatible
tool, where `CarStatusData.m_fuelInTank` gives exact fuel mass per frame from
the game's own physics engine -- not estimated, not this project's prior.

    python scripts/build_f1_game_validation_corpus.py capture.bin --car-index 0

This only reduces a capture to the lap-table schema
`tyremind.models.ssm.tyre_ssm.fit_tyre_ssm` already consumes, with one extra
column (`true_fuel_kg`) no synthetic session can supply honestly. It does not
fit the estimator or score it -- that comparison (does the fitted fuel/
degradation split match `true_fuel_kg`'s actual trend?) belongs next to
`experiments/exp01_ground_truth_recovery.py`, which already knows how to score
a recovered rate against a known one, once a real capture exists to point it
at. Nobody on this project has produced one yet.

Writes <output>.json: a list of per-lap rows, one file per capture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tyremind.data.f1_game_telemetry import build_fuel_ground_truth_table, read_capture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="Path to a .bin telemetry capture")
    parser.add_argument(
        "--car-index", type=int, default=0, help="Which of the up-to-22 grid slots to extract (0-21)"
    )
    parser.add_argument("--out", type=Path, default=None, help="Output path; defaults next to the capture")
    args = parser.parse_args()

    if not args.capture.exists():
        print(f"  {args.capture} not found")
        return 1

    print(f"  reading {args.capture} ...")
    status_samples, lap_samples = read_capture(args.capture)
    print(f"  {len(status_samples)} car-status samples, {len(lap_samples)} lap-data samples")

    rows = build_fuel_ground_truth_table(status_samples, lap_samples, args.car_index)
    if not rows:
        print(
            f"  no completed laps found for car index {args.car_index}; check that this "
            "index was actually on track in the capture (0 is the player car by default, "
            "but only if the header's player_car_index agrees)"
        )
        return 1

    out = args.out or args.capture.with_suffix(".laptable.json")
    out.write_text(json.dumps(rows, indent=2))

    fuel_start, fuel_end = rows[0]["true_fuel_kg"], rows[-1]["true_fuel_kg"]
    print(f"  {len(rows)} completed laps, fuel {fuel_start:.2f} kg -> {fuel_end:.2f} kg")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
