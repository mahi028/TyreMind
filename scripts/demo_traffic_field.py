"""Reconstruct the whole field's real-time proximity for one real session.

Every car's X/Y position is public telemetry; this just aligns all of them on
one clock and reads off who was running in whose wake, and when. Unlike
`f1_loader._traffic_index` (inferred from lap start times), the gaps here are
measured directly from position data at a known instant.

    python scripts/demo_traffic_field.py --year 2024 --event "Bahrain Grand Prix"
"""

from __future__ import annotations

import argparse
import logging
import warnings

import numpy as np
import pandas as pd

from tyremind.physics.traffic_field import TRAIN_GAP_THRESHOLD_M, resample_to_common_grid, traffic_field


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--event", default="Bahrain Grand Prix")
    parser.add_argument("--session", default="R")
    parser.add_argument("--freq", type=float, default=0.5, help="common time grid spacing, seconds")
    parser.add_argument("--cache", default="cache/fastf1")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    import fastf1

    fastf1.Cache.enable_cache(args.cache)
    session = fastf1.get_session(args.year, args.event, args.session)
    session.load(telemetry=True, weather=False, messages=False)

    print(f"\n  {args.year} {args.event} ({args.session}) -- {len(session.drivers)} drivers\n")

    per_driver: dict[str, pd.DataFrame] = {}
    for drv in session.drivers:
        try:
            merged = session.car_data[drv].merge_channels(session.pos_data[drv]).add_distance()
        except Exception as exc:  # noqa: BLE001 - a driver with no usable telemetry is data
            print(f"  driver {drv}: skip ({type(exc).__name__})")
            continue
        per_driver[drv] = merged

    common_time, distances = resample_to_common_grid(per_driver, freq_s=args.freq)
    print(f"  {len(common_time)} common timesteps at {args.freq}s spacing "
          f"({common_time[-1] / 60:.1f} minutes of session)\n")

    field = traffic_field(common_time, distances, train_gap_m=TRAIN_GAP_THRESHOLD_M)
    print(f"  {len(field)} (driver, instant) rows  |  train threshold {TRAIN_GAP_THRESHOLD_M:.0f} m\n")

    in_traffic_frac = field["in_train"].mean()
    print(f"  fraction of (driver, instant) rows running in a train: {in_traffic_frac:.1%}")

    by_size = field.groupby("train_size")["driver"].count()
    print("\n  train-size distribution (row count, not unique events):")
    for size, count in by_size.items():
        print(f"    size {size:>2}: {count:>7,}  ({count / len(field):.1%})")

    # A real, mid-race multi-car train (small, not the formation-lap/grid
    # artifact where the whole field sits at distance zero together).
    racing = field[field["time_s"] > 600]  # past lights-out, clear of the grid/formation lap
    candidates = racing[racing["train_size"].between(2, 5)]
    example_time = float(candidates["time_s"].iloc[len(candidates) // 2]) if len(candidates) else float(field["time_s"].iloc[0])
    # Preserve the frame's own row order (back-to-front track order) rather
    # than re-sorting by gap, which would scramble who is actually next to whom.
    example = field[np.isclose(field["time_s"], example_time)]
    print(f"\n  example mid-race snapshot at t={example_time / 60:.1f} min (session time), "
          f"back-to-front track order:")
    print(f"  {'driver':>6} {'gap_ahead_m':>12} {'train_size':>10}")
    for _, row in example.iterrows():
        gap = "inf" if not np.isfinite(row["gap_ahead_m"]) else f"{row['gap_ahead_m']:.1f}"
        print(f"  {row['driver']:>6} {gap:>12} {row['train_size']:>10}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
