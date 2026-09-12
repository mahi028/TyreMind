"""Fetch FIA's official pit-stop timing and reduce it to a per-event summary.

`configs/physics.yaml`'s `strategy.pit_lane_loss_s` (21.0 s, sd 3.0) is sourced
from FastF1's pit in/out lap deltas, falling back to a prior when that is
missing. FIA publishes something more direct after every race: a "Race Pit
Stop Summary" document timing each stop's time in the pit lane -- entry to
exit, not just the stationary tyre change -- which is exactly what that prior
stands in for.

Only the aggregate (median, sample size) is ever written here, never the
per-driver table: FIA's documents carry an explicit notice restricting
redistribution of "these results/data" beyond press use, and an aggregate
statistic makes no claim on any individual driver's data.

    python scripts/build_pitstop_reference.py --years 2024
    python scripts/build_pitstop_reference.py --years 2024 2023 --delay 3

Requires the `pypdf` optional dependency (requirements-optional.txt) and a
network connection; nothing else in the platform depends on either. Writes
data/reference/pit_stop_official.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.request
from pathlib import Path

from tyremind.data.pit_stops import discover_pitstop_pdf_url, parse_pitstop_text, summarize

logger = logging.getLogger(__name__)

CORPUS = Path("data/season")
OUT = Path("data/reference/pit_stop_official.json")
CACHE_DIR = Path("cache/fia_pitstops")


def _fetch_pdf_text(url: str, *, cache_dir: Path, timeout: float) -> str:
    """Download a pit-stop-summary PDF (cached) and extract its text.

    The PDF itself is cached under `cache/` -- gitignored, never committed --
    purely so re-running the script after an interruption does not re-fetch
    documents it already has.
    """
    from pypdf import PdfReader

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / url.rsplit("/", 1)[-1]

    if not cache_path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            cache_path.write_bytes(response.read())

    reader = PdfReader(str(cache_path))
    return "\n".join(page.extract_text() for page in reader.pages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="*", default=[2024])
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    manifest_path = CORPUS / "corpus.json"
    if not manifest_path.exists():
        logger.error(
            "%s not found; run scripts/build_corpus.py first (or point --years at "
            "seasons already fetched)",
            manifest_path,
        )
        return 1

    manifest = json.loads(manifest_path.read_text())
    events = {
        (e["year"], e["event_name"]): e
        for e in manifest
        if e["year"] in args.years and e["session"] == "R"
    }
    wanted = sorted(events.values(), key=lambda e: (e["year"], e["round_number"]))

    existing: dict[str, dict] = {}
    if args.out.exists():
        existing = {r["session_id"]: r for r in json.loads(args.out.read_text())}

    print(f"\n  {len(wanted)} events to fetch, {len(existing)} already on file\n")
    added = 0

    for entry in wanted:
        session_id = f"{entry['year']}-{entry['event_name']}-R"
        if session_id in existing:
            continue
        try:
            url = discover_pitstop_pdf_url(entry["year"], entry["event_name"])
            text = _fetch_pdf_text(url, cache_dir=CACHE_DIR, timeout=20.0)
            rows = parse_pitstop_text(text)
        except Exception as exc:  # noqa: BLE001 - a missing document is data, not a crash
            print(f"  {session_id:<45} skip  {type(exc).__name__}: {exc}")
            continue

        summary = summarize(rows)
        row = {
            "session_id": session_id,
            "year": entry["year"],
            "round": entry["round_number"],
            "event": entry["event_name"],
            "source_url": url,
            **summary,
        }
        existing[session_id] = row
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(list(existing.values()), indent=2))
        added += 1

        print(
            f"  {session_id:<45} n={summary['n_stops']:<3} "
            f"median={summary['median_s']:.3f}s  clean_median={summary['clean_median_s']:.3f}s"
        )
        time.sleep(args.delay)

    print(f"\n  fetched {added} events, {len(existing)} on file")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
