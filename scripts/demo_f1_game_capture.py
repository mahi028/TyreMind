"""Generate and print an illustrative F1-game capture -- not real data.

For seeing the *shape* of what `scripts/build_f1_game_validation_corpus.py`
will eventually produce from a real recording, before one exists. Every
coefficient is pulled from configs/physics.yaml and tyremind.data.synthetic
where a prior already exists (see tyremind.data.f1_game_telemetry's module
comment above `generate_demo_capture` for exactly which do and don't) -- this
is not random test data, but it is not a real session either.

    python scripts/demo_f1_game_capture.py
    python scripts/demo_f1_game_capture.py --laps 30 --pit-lap 15 --out demo.bin
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tyremind.data.f1_game_telemetry import (
    DemoCaptureConfig,
    build_lap_ground_truth_table,
    generate_demo_capture,
    parse_capture_bytes,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--laps", type=int, default=50)
    parser.add_argument("--pit-lap", type=int, default=24)
    parser.add_argument("--out", type=Path, default=None, help="Also write the raw capture bytes here")
    args = parser.parse_args()

    config = DemoCaptureConfig(n_laps=args.laps, pit_lap=args.pit_lap)
    data = generate_demo_capture(config)

    if args.out:
        args.out.write_bytes(data)
        print(f"  wrote {len(data):,} bytes to {args.out} (SYNTHETIC -- not a real capture)\n")

    samples = parse_capture_bytes(data)
    rows = build_lap_ground_truth_table(
        samples.car_status, samples.lap_data, config.car_index,
        telemetry_samples=samples.car_telemetry, damage_samples=samples.car_damage,
    )

    print(
        f"  {len(rows)} laps  |  {len(samples.car_status)} status, {len(samples.lap_data)} lap, "
        f"{len(samples.car_telemetry)} telemetry, {len(samples.car_damage)} damage samples"
    )
    header = (
        f"  {'lap':>4} {'compound':>9} {'tyre_age':>9} {'lap_time':>9} {'fuel_kg':>8} "
        f"{'psi':>6} {'surf_C':>7} {'inner_C':>8} {'brake_C':>8} {'wear_%':>7}"
    )
    print(header)
    for r in rows:
        print(
            f"  {r['session_lap']:>4} {r['compound']:>9} {r['tyre_age']:>9.0f} "
            f"{r['lap_time']:>9.3f} {r['true_fuel_kg']:>8.2f} "
            f"{r['tyre_pressure_psi']:>6.2f} {r['tyre_surface_temp_c']:>7.1f} "
            f"{r['tyre_inner_temp_c']:>8.1f} {r['brake_temp_c']:>8.1f} {r['tyre_wear_pct']:>7.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
