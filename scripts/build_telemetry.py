"""Reduce FastF1 car telemetry to per-lap energy and load features.

The lap table carries eight columns because that is all the *timing* feed gives.
The telemetry feed carries speed, throttle, brake, gear and RPM at roughly 4 Hz,
which is where per-corner load, frictional energy and a grip index have to come
from. This script turns the second into columns that join onto the first.

It deliberately does NOT store raw samples. A race session is ~1.5 million
telemetry rows; 203 of them is gigabytes of data whose only use is to be reduced
to the twenty numbers below. Reduction happens here, once.

    python scripts/build_telemetry.py --limit 2      # try it on two sessions
    python scripts/build_telemetry.py --delay 4.0    # the overnight run

Output: `data/telemetry/<session_id>.parquet`, one row per driver per lap, joining
onto `data/season/<session_id>.parquet` on (driver, session_lap).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CORPUS_DIR = Path("data/season")
OUT_DIR = Path("data/telemetry")
CACHE = Path("cache/fastf1")

logging.getLogger("fastf1").setLevel(logging.ERROR)


def lap_features(car: pd.DataFrame, params) -> dict:
    """Reduce one lap of ~4 Hz telemetry to the features the estimator can use.

    Every quantity here is derived from speed and time. Lateral acceleration comes
    from speed and path curvature, longitudinal from the speed derivative, and
    load from static weight plus aerodynamic downforce. None of it needs the
    load-cell or slip channels that are team-private -- which is the whole point,
    and also the reason these are proxies rather than measurements.
    """
    from tyremind.physics import dynamics

    if len(car) < 8:
        return {}

    speed_ms = car["Speed"].to_numpy(dtype=float) / 3.6
    time_s = car["SessionTime"].dt.total_seconds().to_numpy(dtype=float)
    time_s = time_s - time_s[0]
    if not np.all(np.diff(time_s) > 0):
        # Duplicate timestamps make every derivative infinite. Rare, but one such
        # lap would otherwise poison the session's aggregates silently.
        keep = np.concatenate([[True], np.diff(time_s) > 0])
        speed_ms, time_s = speed_ms[keep], time_s[keep]
        car = car.iloc[keep]
        if len(car) < 8:
            return {}

    has_position = {"X", "Y"} <= set(car.columns) and car["X"].notna().any()
    if has_position:
        curvature = dynamics.path_curvature(
            car["X"].to_numpy(dtype=float), car["Y"].to_numpy(dtype=float))
        a_lat = dynamics.lateral_acceleration(speed_ms, curvature)
    else:
        a_lat = np.zeros_like(speed_ms)

    a_long = dynamics.longitudinal_acceleration(speed_ms, time_s)
    loads = dynamics.corner_loads(speed_ms, a_long, a_lat, params)
    power = dynamics.frictional_power_proxy(loads, speed_ms, a_long, a_lat)

    dt = np.diff(time_s, prepend=time_s[0])
    g = 9.81

    features = {
        "lap_time_telemetry_s": float(time_s[-1]),
        "mean_abs_lateral_g": float(np.mean(np.abs(a_lat)) / g),
        "p95_lateral_g": float(np.percentile(np.abs(a_lat), 95) / g),
        "mean_abs_long_g": float(np.mean(np.abs(a_long)) / g),
        "p95_long_g": float(np.percentile(np.abs(a_long), 95) / g),
        "top_speed_kmh": float(np.max(car["Speed"])),
        "mean_speed_kmh": float(np.mean(car["Speed"])),
        # Fraction of the lap spent above 0.5 g laterally: "how much of this lap
        # was actually working the tyres" rather than "how long was the lap".
        "loaded_fraction": float(np.mean(np.abs(a_lat) / g > 0.5)),
        "full_throttle_fraction": float(np.mean(car["Throttle"] > 98))
        if "Throttle" in car else np.nan,
        "braking_fraction": float(np.mean(car["Brake"].astype(bool)))
        if "Brake" in car else np.nan,
    }

    total_energy = 0.0
    for corner, series in power.items():
        energy_mj = float(np.sum(np.asarray(series, dtype=float) * dt) / 1e6)
        features[f"energy_mj_{corner.lower()}"] = energy_mj
        total_energy += energy_mj
        features[f"mean_load_n_{corner.lower()}"] = float(np.mean(loads[corner]))
    features["energy_mj_total"] = total_energy

    if total_energy > 0:
        # The asymmetry is the four-tyre signal: which corner is doing the work.
        for corner in power:
            key = f"energy_mj_{corner.lower()}"
            features[f"energy_share_{corner.lower()}"] = features[key] / total_energy
        left = features.get("energy_share_fl", 0) + features.get("energy_share_rl", 0)
        features["left_energy_share"] = left
        front = features.get("energy_share_fl", 0) + features.get("energy_share_fr", 0)
        features["front_energy_share"] = front

    return features


def process_session(session_id: str) -> tuple[int, str]:
    """Fetch and reduce one session. Returns (rows written, message)."""
    import fastf1
    from tyremind.physics.dynamics import VehicleParameters

    destination = OUT_DIR / f"{session_id}.parquet"
    if destination.exists():
        return 0, "skip (exists)"

    year, rest = session_id.split("-", 1)
    event_slug, session_code = rest.rsplit("-", 1)
    event_name = event_slug.replace("-", " ").title()

    session = fastf1.get_session(int(year), event_name, session_code)
    session.load(telemetry=True, laps=True, weather=False, messages=False)

    params = VehicleParameters()
    rows = []
    for _, lap in session.laps.iterlaps():
        try:
            car = lap.get_telemetry()
        except Exception:
            continue
        features = lap_features(car, params)
        if not features:
            continue
        rows.append({"driver": lap["Driver"],
                     "session_lap": int(lap["LapNumber"]), **features})

    if not rows:
        return 0, "FAILED no usable telemetry"

    frame = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return len(frame), f"ok {len(frame)} laps"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=4.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sessions", nargs="*", help="specific session ids")
    args = parser.parse_args()

    import fastf1
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))

    ids = args.sessions or sorted(p.stem for p in CORPUS_DIR.glob("*.parquet"))
    if args.limit:
        ids = ids[: args.limit]

    done, failed = [], []
    for n, session_id in enumerate(ids, 1):
        try:
            count, message = process_session(session_id)
        except Exception as exc:  # one bad session must not end an overnight run
            count, message = 0, f"FAILED {type(exc).__name__}: {exc}"
        print(f"[{n}/{len(ids)}] {session_id}: {message}", flush=True)
        (done if message.startswith("ok") else
         failed if message.startswith("FAILED") else done).append(session_id)
        if not message.startswith("skip"):
            time.sleep(args.delay)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.joinpath("_run.json").write_text(
        json.dumps({"done": len(done), "failed": failed}, indent=2), encoding="utf-8")
    print(f"\n{len(done)} done, {len(failed)} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
