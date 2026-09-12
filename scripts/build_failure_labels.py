"""Collect real tyre and structural failure events — the labels we thought we lacked.

`docs/data_doc.md` §3 says the answer key for tyre *wear* is not public, and that
remains true. But the answer key for tyre *failure* is: every race classification
records why each retirement happened, and the status vocabulary distinguishes
Puncture, Tyre, Wheel, Wheel nut and Wheel bearing from Suspension, Brakes and
Collision damage.

That distinction is what makes a hazard model possible rather than decorative. A
puncture caused by debris is a different physical event from a structural failure
caused by wear, and pooling them would inflate the apparent wear signal. They are
competing risks and are kept apart here, at the point of collection.

Suspension and Brakes are collected too. They are not tyre failures, and they are
the labels the "full vehicle health monitoring" extension needs.

    python scripts/build_failure_labels.py --years 2022 2023 2024 2025

Output: `data/reference/failure_events.json`
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

API = "https://api.jolpi.ca/ergast/f1"
UA = "TyreMind-research/0.1"
OUT = Path("data/reference/failure_events.json")

#: Status strings grouped by physical cause. The grouping is the modelling
#: decision: `tyre` is what the hazard model predicts, `competing` is what must be
#: modelled separately so it does not contaminate the wear signal, and `vehicle`
#: is the extension to non-tyre components.
CAUSE_GROUPS = {
    "tyre": {"Puncture", "Tyre"},
    "wheel_assembly": {"Wheel", "Wheel nut", "Wheel bearing"},
    "competing": {"Collision damage", "Collision", "Accident", "Debris", "Spun off"},
    "vehicle": {"Suspension", "Brakes", "Driveshaft", "Transmission", "Gearbox"},
}
INTERESTING = {s for group in CAUSE_GROUPS.values() for s in group}


def cause_group(status: str) -> str | None:
    for group, members in CAUSE_GROUPS.items():
        if status in members:
            return group
    return None


def get(url: str, *, retries: int = 4) -> dict | None:
    """GET with backoff. Jolpica is a small volunteer-run service; be gentle."""
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt < retries - 1:
                time.sleep(4 * (attempt + 1))
                continue
            print(f"    HTTP {exc.code} on {url}")
            return None
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(3)
                continue
            print(f"    {type(exc).__name__} on {url}")
            return None
    return None


def season_results(year: int, delay: float) -> list[dict]:
    """Every result row for a season, paged.

    The API caps a page at 100 rows and a season is ~450, so this pages rather
    than silently taking the first 100 — a truncation that would look like a
    season with very few retirements.
    """
    rows, offset = [], 0
    while True:
        payload = get(f"{API}/{year}/results/?limit=100&offset={offset}&format=json")
        if payload is None:
            break
        table = payload["MRData"]["RaceTable"]["Races"]
        if not table:
            break
        for race in table:
            for result in race.get("Results", []):
                rows.append({
                    "year": int(race["season"]),
                    "round": int(race["round"]),
                    "event": race["raceName"],
                    "circuit": race["Circuit"]["circuitId"],
                    "driver": result["Driver"].get("code")
                              or result["Driver"]["driverId"][:3].upper(),
                    "constructor": result["Constructor"]["constructorId"],
                    "grid": int(result.get("grid", 0)),
                    "position": result.get("positionText"),
                    "laps_completed": int(result.get("laps", 0)),
                    "status": result.get("status", ""),
                })
        total = int(payload["MRData"]["total"])
        offset += 100
        if offset >= total:
            break
        time.sleep(delay)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="*",
                        default=[2022, 2023, 2024, 2025])
    parser.add_argument("--delay", type=float, default=1.5)
    args = parser.parse_args()

    everything: list[dict] = []
    for year in args.years:
        rows = season_results(year, args.delay)
        print(f"  {year}: {len(rows)} result rows")
        everything.extend(rows)
        time.sleep(args.delay)

    if not everything:
        print("no results retrieved")
        return 1

    events = []
    for row in everything:
        group = cause_group(row["status"])
        if group is None:
            continue
        events.append(row | {"cause_group": group})

    by_status = Counter(e["status"] for e in events)
    by_group = Counter(e["cause_group"] for e in events)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "_note": (
            "Retirement causes from the public results table. `cause_group` "
            "separates competing risks: a debris puncture is not a wear-driven "
            "structural failure, and pooling them would inflate the wear signal. "
            "`laps_completed` is the censoring/event time for a hazard model; "
            "every non-retirement in the same race is a censored observation."
        ),
        "years": args.years,
        "n_result_rows_scanned": len(everything),
        "n_failure_events": len(events),
        "by_status": dict(by_status.most_common()),
        "by_cause_group": dict(by_group.most_common()),
        "events": events,
    }, indent=1), encoding="utf-8")

    print(f"\n{len(events)} failure events from {len(everything)} result rows")
    for status, count in by_status.most_common():
        print(f"  {status:<20} {count:>4}   ({cause_group(status)})")
    print(f"\n-> {OUT}")

    tyre_side = by_group.get("tyre", 0) + by_group.get("wheel_assembly", 0)
    print(f"\nTyre-side positives: {tyre_side}. "
          f"Rare-event regime -- use a hazard model with partial pooling and "
          f"report the calibration curve, not a point probability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
