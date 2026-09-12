"""Export a short window of real position telemetry for a browser replay.

Pulls every car's actual X/Y from FastF1 (the same public position feed
`traffic_field.py` reconstructs traffic from), resamples the whole field onto
one shared clock, and writes compact JSON a canvas can animate directly --
real recorded positions, not a simulation.

    python scripts/export_race_replay.py --year 2024 --event "Bahrain Grand Prix" \
        --start-offset -8 --duration 130 --out replay.json
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from tyremind.physics.traffic_field import TRAIN_GAP_THRESHOLD_M, snapshot_at

#: FastF1 position telemetry is in tenths of a metre.
POS_UNITS_PER_M = 10.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--event", default="Bahrain Grand Prix")
    parser.add_argument("--session", default="R")
    parser.add_argument("--start-offset", type=float, default=-8.0,
                         help="seconds relative to the lights-out (lap-1 start) to begin the clip")
    parser.add_argument("--duration", type=float, default=130.0, help="clip length, seconds")
    parser.add_argument("--freq", type=float, default=0.2, help="output frame spacing, seconds")
    parser.add_argument("--cache", default="cache/fastf1")
    parser.add_argument("--out", type=Path, default=Path("replay.json"))
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    import fastf1

    fastf1.Cache.enable_cache(args.cache)
    session = fastf1.get_session(args.year, args.event, args.session)
    session.load(telemetry=True, weather=False, messages=False)

    lights_out = session.laps.loc[session.laps["LapNumber"] == 1, "LapStartTime"].dt.total_seconds().min()
    t0, t1 = lights_out + args.start_offset, lights_out + args.start_offset + args.duration
    print(f"  lights-out at session time {lights_out:.1f}s; clip [{t0:.1f}, {t1:.1f}]s")

    drivers_meta = []
    per_driver_xy: dict[str, pd.DataFrame] = {}
    for drv in session.drivers:
        try:
            info = session.get_driver(drv)
            merged = session.car_data[drv].merge_channels(session.pos_data[drv]).add_distance()
        except Exception as exc:  # noqa: BLE001 - a driver with no usable telemetry is data
            print(f"  driver {drv}: skip ({type(exc).__name__})")
            continue
        t = merged["SessionTime"].dt.total_seconds().to_numpy(dtype=float)
        mask = (t >= t0 - 2) & (t <= t1 + 2)  # small pad so interpolation has edge context
        if mask.sum() < 2:
            continue
        per_driver_xy[drv] = pd.DataFrame(
            {
                "t": t[mask],
                "x": merged["X"].to_numpy(dtype=float)[mask] / POS_UNITS_PER_M,
                "y": merged["Y"].to_numpy(dtype=float)[mask] / POS_UNITS_PER_M,
                "dist": merged["Distance"].to_numpy(dtype=float)[mask],
            }
        )
        drivers_meta.append(
            {
                "num": drv,
                "abbr": str(info["Abbreviation"]),
                "team": str(info["TeamName"]),
                "color": "#" + str(info["TeamColor"]),
            }
        )

    common = np.arange(t0, t1, args.freq)
    pos: dict[str, list[list[float]]] = {}
    dist_by_driver: dict[str, np.ndarray] = {}
    for drv, df in per_driver_xy.items():
        x = np.interp(common, df["t"], df["x"], left=np.nan, right=np.nan)
        y = np.interp(common, df["t"], df["y"], left=np.nan, right=np.nan)
        d = np.interp(common, df["t"], df["dist"], left=np.nan, right=np.nan)
        pos[drv] = [
            [round(float(px), 1), round(float(py), 1)] if np.isfinite(px) else None
            for px, py in zip(x, y)
        ]
        dist_by_driver[drv] = d

    # Real measured along-track gap and train grouping, from `traffic_field`'s
    # own logic -- the same function this session already validated against
    # the full field, applied here to the frames the replay actually shows.
    gap_ahead: dict[str, list[float | None]] = {drv: [] for drv in dist_by_driver}
    train_size: dict[str, list[int]] = {drv: [] for drv in dist_by_driver}
    order_front_to_back: list[list[str]] = []
    for i in range(len(common)):
        snap = snapshot_at({drv: dist_by_driver[drv][i] for drv in dist_by_driver})
        if snap is None:
            for drv in dist_by_driver:
                gap_ahead[drv].append(None)
                train_size[drv].append(1)
            order_front_to_back.append([])
            continue
        order_front_to_back.append(list(reversed(snap.order)))
        sizes: dict[str, int] = {}
        groups = [[snap.order[0]]]
        for j in range(1, len(snap.order)):
            if snap.gap_ahead_m[snap.order[j - 1]] <= TRAIN_GAP_THRESHOLD_M:
                groups[-1].append(snap.order[j])
            else:
                groups.append([snap.order[j]])
        for group in groups:
            for drv in group:
                sizes[drv] = len(group)
        for drv in dist_by_driver:
            if drv in snap.gap_ahead_m:
                g = snap.gap_ahead_m[drv]
                gap_ahead[drv].append(round(float(g), 1) if np.isfinite(g) else None)
                train_size[drv].append(sizes[drv])
            else:
                gap_ahead[drv].append(None)
                train_size[drv].append(1)

    # Track outline: one lap's worth of position from whoever has the most
    # samples, well clear of this clip so it is a full closed loop, not the
    # partial arc the clip itself covers.
    ref_drv = max(per_driver_xy, key=lambda d: len(per_driver_xy[d]))
    full = session.car_data[ref_drv].merge_channels(session.pos_data[ref_drv])
    lap2 = session.laps.pick_drivers(ref_drv).pick_lap(2)
    lap_tel = lap2.get_telemetry() if len(lap2) else full
    track = [
        [round(float(x) / POS_UNITS_PER_M, 1), round(float(y) / POS_UNITS_PER_M, 1)]
        for x, y in zip(lap_tel["X"], lap_tel["Y"])
        if pd.notna(x) and pd.notna(y)
    ]

    out = {
        "meta": {
            "year": args.year,
            "event": args.event,
            "session": args.session,
            "t0": float(t0),
            "t1": float(t1),
            "freq_s": args.freq,
            "lights_out": float(lights_out),
        },
        "track": track,
        "drivers": drivers_meta,
        "times": [round(float(t), 2) for t in common],
        "pos": pos,
        "gapAhead": gap_ahead,
        "trainSize": train_size,
        "order": order_front_to_back,
    }
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    size_kb = args.out.stat().st_size / 1024
    print(f"  {len(common)} frames x {len(pos)} drivers, track outline {len(track)} points")
    print(f"  wrote {args.out} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
