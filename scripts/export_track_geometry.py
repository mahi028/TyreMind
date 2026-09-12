"""Export real 3D track centrelines for the visualisation work.

The dashboard's 3D view needs a track, and the timing feed already contains one:
FastF1's position stream carries X, Y **and Z** per sample, so the centreline that
comes out of this has real elevation rather than a flat plan traced from a logo.
Spa's Eau Rouge and Austin's Turn 1 render as the hills they are.

Output is deliberately renderer-agnostic -- plain arrays of coordinates in metres,
plus corner markers and the rotation needed to orient the map the way broadcast
does. Nothing here assumes three.js, Unity or anything else, because the
visualisation is someone else's job and a format that presumes their tool is a
format they will have to undo.

    python scripts/export_track_geometry.py --sessions 2024-monza-R 2024-silverstone-R
    python scripts/export_track_geometry.py --all-demo

Writes `data/geometry/<circuit>.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

OUT_DIR = Path("data/geometry")
CACHE = Path("cache/fastf1")

#: FastF1 position coordinates are in tenths of a metre. Monza reads about
#: 12,500 units across for a 5.79 km lap, which is the factor below. Exporting
#: metres rather than raw units means the renderer never has to guess.
UNITS_PER_METRE = 10.0

logging.getLogger("fastf1").setLevel(logging.ERROR)


def resample_closed_loop(xyz: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a lap to evenly spaced points by arc length.

    Raw telemetry samples at a fixed *time* interval, so points bunch up in slow
    corners and stretch out on straights -- which is exactly backwards for a mesh,
    where corners need the detail. Resampling by distance travelled fixes it.
    """
    deltas = np.diff(xyz, axis=0)
    seg = np.sqrt((deltas ** 2).sum(axis=1))
    distance = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(distance[-1])
    if total <= 0:
        return xyz

    targets = np.linspace(0.0, total, n_points, endpoint=False)
    out = np.empty((n_points, xyz.shape[1]), dtype=float)
    for axis in range(xyz.shape[1]):
        out[:, axis] = np.interp(targets, distance, xyz[:, axis])
    return out


def export_session(session_id: str, n_points: int) -> dict:
    """Pull one circuit's geometry from the fastest lap of a session.

    The fastest lap is used because it is the cleanest single trace of the racing
    line available -- no traffic, no lift-and-coast, no in-lap. It is a racing
    line rather than a track centreline, which is stated here rather than
    implied: the rendered ribbon is where a car actually went, not where the
    white lines are.
    """
    import fastf1

    year, rest = session_id.split("-", 1)
    event_slug, session_code = rest.rsplit("-", 1)
    session = fastf1.get_session(int(year), event_slug.replace("-", " ").title(), session_code)
    session.load(telemetry=True, laps=True, weather=False, messages=False)

    lap = session.laps.pick_fastest()
    pos = lap.get_pos_data()
    xyz = np.column_stack([
        pos["X"].to_numpy(dtype=float),
        pos["Y"].to_numpy(dtype=float),
        pos["Z"].to_numpy(dtype=float),
    ]) / UNITS_PER_METRE

    # Close the loop before resampling, or the last segment of the lap is missing
    # and the mesh has a visible seam at the start/finish line.
    if np.linalg.norm(xyz[-1] - xyz[0]) > 1.0:
        xyz = np.vstack([xyz, xyz[0]])
    centreline = resample_closed_loop(xyz, n_points)

    info = session.get_circuit_info()
    corners = [
        {
            "number": int(row["Number"]),
            "letter": str(row["Letter"]) if str(row["Letter"]).strip() else None,
            "x": float(row["X"]) / UNITS_PER_METRE,
            "y": float(row["Y"]) / UNITS_PER_METRE,
            "angle_deg": float(row["Angle"]),
            "distance_m": float(row["Distance"]),
        }
        for _, row in info.corners.iterrows()
    ]

    elevation = centreline[:, 2]
    return {
        "session_id": session_id,
        "circuit": session.event["Location"],
        "event": session.event["EventName"],
        "year": int(year),
        "_note": (
            "Centreline is the racing line of the session's fastest lap, resampled "
            "evenly by distance. Coordinates are metres. Z is real elevation from "
            "the positioning feed, not synthesised."
        ),
        "units": "metres",
        "rotation_deg": float(info.rotation),
        "lap_length_m": float(np.sqrt((np.diff(centreline[:, :2], axis=0) ** 2).sum(axis=1)).sum()),
        "elevation_range_m": [float(elevation.min()), float(elevation.max())],
        "elevation_gain_m": float(elevation.max() - elevation.min()),
        "n_points": int(len(centreline)),
        "centreline": [[round(float(v), 2) for v in point] for point in centreline],
        "corners": corners,
        "start_finish": [round(float(v), 2) for v in centreline[0]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="*", default=[])
    parser.add_argument("--all-demo", action="store_true",
                        help="every race in data/demo, which is what ships offline")
    parser.add_argument("--points", type=int, default=600,
                        help="centreline resolution; 600 is smooth at any sane zoom")
    args = parser.parse_args()

    import fastf1
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))

    targets = list(args.sessions)
    if args.all_demo:
        targets += [p.stem for p in sorted(Path("data/demo").glob("*-R.parquet"))]
    if not targets:
        raise SystemExit("pass --sessions or --all-demo")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for session_id in dict.fromkeys(targets):
        try:
            geometry = export_session(session_id, args.points)
        except Exception as exc:  # noqa: BLE001
            print(f"  {session_id}: FAILED {type(exc).__name__}: {exc}")
            continue
        name = geometry["circuit"].lower().replace(" ", "-")
        destination = OUT_DIR / f"{name}.json"
        destination.write_text(json.dumps(geometry, indent=1), encoding="utf-8")
        print(f"  {geometry['circuit']:<22} {geometry['lap_length_m']:>7.0f} m  "
              f"elevation {geometry['elevation_gain_m']:>5.1f} m  "
              f"{len(geometry['corners'])} corners  -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
