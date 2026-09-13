"""Pull measured gaps from OpenF1, to replace a traffic index we invented.

`traffic_index` in the lap table is derived, not measured. We sort every lap in a
session by start time, walk back to the most recent lap begun by a different car,
and map that gap onto 0-1. It is a decent idea and it has never been checked
against a real measurement.

OpenF1 publishes `interval` and `gap_to_leader` roughly every four seconds,
free for 2023 onwards. That is a measurement, and having both lets us ask the
question exp26 should have asked of the sector idea before we built on it: does
the proxy agree with the thing it proxies?

**This script only collects.** Whether the measured index changes any result is
exp31's job, and the honest outcomes are three: the proxy agrees and we have
validated a clever shortcut; it disagrees and every result resting on traffic
needs revisiting; or traffic turns out not to matter either way, which exp10
already hints at (traffic gap versus practice-to-race bias came out at rho
-0.167, p = 0.109, not significant).

    python scripts/build_openf1_traffic.py --years 2024 2023 --delay 0.4

Writes `data/reference/openf1_intervals/<session_key>.parquet`.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://api.openf1.org/v1"
OUT_DIR = Path("data/reference/openf1_intervals")
UA = "TyreMind-research/0.1 (academic use)"

#: OpenF1's free tier serves history from 2023. Asking for 2022 returns an empty
#: list rather than an error, which would look like a session with no traffic.
FIRST_YEAR = 2023


def get(path: str, params: dict, *, retries: int = 4, delay: float = 0.4) -> list[dict]:
    """GET a list endpoint with backoff.

    OpenF1 is free and rate-limited, and a 429 answered by retrying immediately
    earns a longer ban. Backoff is linear and the caller sets the base delay.
    """
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 502, 503) and attempt < retries - 1:
                time.sleep(delay * 4 * (attempt + 1))
                continue
            print(f"    HTTP {exc.code} on {path} {params}")
            return []
        except Exception as exc:  # noqa: BLE001
            if attempt < retries - 1:
                time.sleep(delay * 2)
                continue
            print(f"    {type(exc).__name__} on {path}")
            return []
    return []


def race_sessions(year: int, delay: float) -> list[dict]:
    sessions = get("sessions", {"year": year, "session_name": "Race"}, delay=delay)
    return [s for s in sessions if s.get("session_key")]


def collect_session(session: dict, delay: float) -> pd.DataFrame | None:
    """Intervals, stints and pit stops for one race, joined on driver number.

    `interval` is the gap to the car ahead and `gap_to_leader` the gap to the
    leader; both are what our derived index is trying to approximate. Stints give
    an independent reading of compound and tyre age at stint start, which is a
    free cross-check on the run reconstruction we do ourselves.
    """
    key = session["session_key"]
    intervals = get("intervals", {"session_key": key}, delay=delay)
    if not intervals:
        return None
    time.sleep(delay)
    stints = get("stints", {"session_key": key}, delay=delay)

    frame = pd.DataFrame(intervals)
    if frame.empty or "date" not in frame:
        return None
    frame["date"] = pd.to_datetime(frame["date"], format="ISO8601", utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"])
    # `gap_to_leader` and `interval` mix floats with strings like "+1 LAP".
    # That is not dirt -- a lapped car is a different state from a large gap, and
    # coercing it to a number would invent a 60-second gap where the truth is
    # "a lap down". Split into a numeric column and a flag, and keep the raw text.
    for column in ("gap_to_leader", "interval"):
        if column not in frame.columns:
            continue
        raw = frame[column]
        frame[f"{column}_raw"] = raw.astype(str)
        frame[f"{column}_lapped"] = raw.astype(str).str.contains("LAP", case=False, na=False)
        frame[column] = pd.to_numeric(raw, errors="coerce")

    frame["session_key"] = key
    frame["year"] = session.get("year")
    frame["circuit"] = session.get("circuit_short_name") or session.get("location")

    if stints:
        stint_frame = pd.DataFrame(stints)
        # Kept alongside rather than merged: stints are per lap-range and
        # intervals are per timestamp, so joining them needs a lap alignment this
        # script deliberately does not attempt. Storing both lets exp31 decide.
        stint_path = OUT_DIR / f"{key}_stints.parquet"
        stint_frame.to_parquet(stint_path, index=False)

    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="*", default=[2024, 2023])
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--limit", type=int, default=0, help="cap sessions per year")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written, skipped, empty = 0, 0, 0

    for year in args.years:
        if year < FIRST_YEAR:
            print(f"  {year}: skipped, OpenF1's free history starts at {FIRST_YEAR}")
            continue
        sessions = race_sessions(year, args.delay)
        if args.limit:
            sessions = sessions[: args.limit]
        print(f"  {year}: {len(sessions)} races")
        time.sleep(args.delay)

        for session in sessions:
            key = session["session_key"]
            destination = OUT_DIR / f"{key}.parquet"
            name = f"{session.get('circuit_short_name', '?')}"
            if destination.exists():
                print(f"    {name:<22} skip (have it)")
                skipped += 1
                continue
            frame = collect_session(session, args.delay)
            if frame is None or frame.empty:
                print(f"    {name:<22} no interval data")
                empty += 1
                time.sleep(args.delay)
                continue
            frame.to_parquet(destination, index=False)
            print(f"    {name:<22} {len(frame):>7} interval samples")
            written += 1
            time.sleep(args.delay)

    print(f"\n{written} written, {skipped} already present, {empty} empty -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
