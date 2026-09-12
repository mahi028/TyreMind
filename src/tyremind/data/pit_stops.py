"""Official FIA pit-stop timing: a real number for `strategy.pit_lane_loss_s`.

`configs/physics.yaml` currently sources `pit_lane_loss_s` (21.0 s, sd 3.0) from
FastF1's pit in/out lap deltas, falling back to the prior when that is missing.
FIA publishes something more direct after every session: a "Race Pit Stop
Summary" timing document giving each stop's `STOP DURATION` -- time in the pit
lane, not just the stationary tyre change -- to the millisecond, alongside the
lap it happened on. That is exactly the quantity the prior is standing in for.

The document is a PDF at a URL FIA does not advertise as a stable API, so this
module does not guess it. Guessing the URL from year, round and a country code
looked tempting -- FIA's own paths use short codes like "ita", "hun", "ned" --
until Miami and Austin, both nominally "usa", showed this cannot be inverted
from country alone. Instead, `discover_pitstop_pdf_url` fetches the event's own
"Event & Timing Information" page (a stable, human-navigable URL built from the
event name) and reads the real link off it. That is slower than string
formatting but it is not a guess.

Only aggregate statistics are ever written to committed reference data --
never the per-driver table. FIA's documents carry an explicit notice
restricting redistribution of "these results/data" beyond press use; a median
and a sample size make no claim on any individual driver's data and are safe
to keep, the raw table is not.
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from statistics import median
from urllib.parse import urljoin

FIA_BASE = "https://www.fia.com"

#: `pypdf`'s text extraction -- confirmed against a real document, the 2025
#: Italian GP race pit-stop summary -- puts one table CELL per line, not one
#: row per line: "NO\nDRIVER\nENTRANT\nLAP\nTIME OF DAY\nSTOP\nDURATION\n
#: TOTAL TIME\n30\nLiam LAWSON\nVisa Cash App Racing Bulls F1 Team\n9\n...".
#: The header is these 8 field names once; every row after it is exactly 8
#: more lines in the same order. This is what `_ROW_FIELDS` walks.
_HEADER_FIELDS = ("NO", "DRIVER", "ENTRANT", "LAP", "TIME OF DAY", "STOP", "DURATION", "TOTAL TIME")
_ROW_LENGTH = len(_HEADER_FIELDS)

_TIME_OF_DAY_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")

#: FIA's copyright/reproduction notice, appended after the table, tokenises
#: unpredictably under PDF extraction -- kerning around "F1" and "90 days"
#: splits digits onto their own lines, so a stray "1" or "90" line can look
#: like the start of a new row. Validating every field's *type* (not just
#: consuming 8 lines) is what tells that boilerplate apart from a real row:
#: the giveaway is always the TIME OF DAY field, which boilerplate text never
#: happens to match.


@dataclass(frozen=True)
class PitStopRow:
    """One stop, as FIA timed it.

    Attributes:
        driver_number: Car number.
        driver_name: "Firstname SURNAME", as printed -- a short name, not
            necessarily the full legal name (e.g. "Kimi ANTONELLI").
        entrant: Constructor entry name, e.g. "Scuderia Ferrari HP".
        lap: Lap the stop happened on.
        time_of_day: Local time of day the stop occurred, "HH:MM:SS".
        stop_number: 1 for a driver's first stop, 2 for their second, etc.
        duration_s: Time in the pit lane for this stop, seconds. This is pit
            *lane* time (entry to exit), not just the stationary tyre change --
            it is the quantity `strategy.pit_lane_loss_s` estimates.
        total_time_s: FIA's running total across this driver's stops so far.
    """

    driver_number: int
    driver_name: str
    entrant: str
    lap: int
    time_of_day: str
    stop_number: int
    duration_s: float
    total_time_s: float


def _parse_row(fields: list[str]) -> PitStopRow | None:
    """Validate and build one row from 8 consecutive extracted lines.

    Every field's type is checked, not just its presence -- a window of 8
    lines pulled from the copyright notice will almost always fail one of
    these (usually TIME OF DAY, which free text never matches), and returning
    `None` rather than raising is what lets `parse_pitstop_text` treat "the
    next 8 lines are not a row" as the ordinary signal that the table ended.
    """
    no, driver, entrant, lap, time_of_day, stop, duration, total = fields
    if not (no.isdigit() and lap.isdigit() and stop.isdigit()):
        return None
    if not _TIME_OF_DAY_RE.match(time_of_day):
        return None
    if not driver or not entrant:
        return None
    try:
        duration_s, total_s = float(duration), float(total)
    except ValueError:
        return None
    # A real stop is seconds, not milliseconds or minutes; this range is
    # generous (F1's slowest recorded pit-lane transits are well under two
    # minutes) and exists only to reject boilerplate that happens to reach
    # this far, not to filter real outliers -- see `summarize` for that.
    if not (1.0 <= duration_s <= 120.0):
        return None
    return PitStopRow(
        driver_number=int(no),
        driver_name=driver,
        entrant=entrant,
        lap=int(lap),
        time_of_day=time_of_day,
        stop_number=int(stop),
        duration_s=duration_s,
        total_time_s=total_s,
    )


def parse_pitstop_text(text: str) -> list[PitStopRow]:
    """Parse the extracted text of an FIA "Pit Stop Summary" PDF.

    Expects `pypdf`-style extraction, one table cell per line (see
    `_HEADER_FIELDS`). Skips the header, then reads consecutive 8-line groups
    as rows until a group fails validation -- which is what happens at the
    boundary with FIA's trailing copyright notice, ending the table without
    needing to recognise that notice's text specifically.

    Args:
        text: Plain text extracted from the PDF (e.g. via `pypdf`).

    Returns:
        One `PitStopRow` per stop, in document order.

    Raises:
        ValueError: If zero rows parsed -- most likely a format change in the
            document, which should fail loudly rather than silently produce
            an empty reference entry.
    """
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    start = 0
    while start < len(lines) and lines[start] in _HEADER_FIELDS:
        start += 1

    rows: list[PitStopRow] = []
    pos = start
    while pos + _ROW_LENGTH <= len(lines):
        row = _parse_row(lines[pos : pos + _ROW_LENGTH])
        if row is None:
            break
        rows.append(row)
        pos += _ROW_LENGTH

    if not rows:
        raise ValueError(
            "no pit-stop rows matched; the document format may have changed "
            "from the one this parser was written against"
        )
    return rows


def summarize(rows: list[PitStopRow], *, outlier_multiple: float = 1.4) -> dict:
    """Aggregate stop durations into the statistic worth keeping.

    A pit-lane loss prior should describe a *clean* stop, not the field's full
    spread -- a stop-and-go penalty or a wheel-nut problem inflates duration for
    reasons that have nothing to do with the pit lane's geometry or speed limit.
    `clean_median_s` excludes stops beyond `outlier_multiple` times the raw
    median before re-computing, which is a light touch: against the 2025 Monza
    data (raw median 24.916 s) it catches only Lance Stroll's 38.418 s stop
    (1.54x) while keeping Esteban Ocon's slower-but-clean 30.754 s (1.23x) --
    without needing a hand-picked threshold per circuit.

    Args:
        rows: Stops from one session, as `parse_pitstop_text` returns.
        outlier_multiple: A stop above this multiple of the raw median is
            excluded from `clean_median_s`.

    Returns:
        A dict with `n_stops`, `median_s`, `min_s`, `clean_median_s` and
        `clean_n_stops` -- everything downstream should read from this, never
        from the raw rows, so that no per-driver data leaves this module.
    """
    durations = [r.duration_s for r in rows]
    raw_median = median(durations)
    clean = [d for d in durations if d <= raw_median * outlier_multiple]
    return {
        "n_stops": len(durations),
        "median_s": raw_median,
        "min_s": min(durations),
        "clean_median_s": median(clean) if clean else raw_median,
        "clean_n_stops": len(clean),
    }


def slugify_event_name(event_name: str) -> str:
    """"Italian Grand Prix" -> "italian-grand-prix", matching FIA's event URLs."""
    slug = event_name.strip().lower().replace("'", "").replace("’", "")
    return re.sub(r"[^a-z0-9]+", "-", slug).strip("-")


def discover_pitstop_pdf_url(
    year: int, event_name: str, *, session: str = "race", timeout: float = 15.0
) -> str:
    """Find the real pit-stop-summary PDF link from the event's timing page.

    Reads `https://www.fia.com/events/.../season-{year}/{slug}/
    eventtiming-information` and returns whichever document link on it matches
    the requested session's pit-stop-summary pattern. This is a live network
    call and is not covered by the offline unit tests.

    Args:
        year: Season.
        event_name: FastF1's `EventName`, e.g. "Italian Grand Prix".
        session: "race" or "sprint".
        timeout: Request timeout, seconds.

    Returns:
        Absolute URL to the PDF.

    Raises:
        ValueError: If the event page has no matching link. FIA's document
            naming or page structure may have changed since this was written
            against the 2025 season.
    """
    if session not in ("race", "sprint"):
        raise ValueError(f"session must be 'race' or 'sprint', got {session!r}")

    slug = slugify_event_name(event_name)
    page_url = (
        f"{FIA_BASE}/events/fia-formula-one-world-championship/"
        f"season-{year}/{slug}/eventtiming-information"
    )
    request = urllib.request.Request(page_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        html = response.read().decode("utf-8", errors="replace")

    pattern = re.compile(
        rf'href="([^"]*{session}pitstopsummary[^"]*\.pdf)"', re.IGNORECASE
    )
    match = pattern.search(html)
    if not match:
        raise ValueError(
            f"no {session} pit-stop-summary link found on {page_url}; FIA's page "
            "structure or naming may have changed"
        )
    return urljoin(FIA_BASE, match.group(1))
