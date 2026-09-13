"""Measure kerb exposure from public F1 data -- and measure whether it is measurable.

A judge asked what happens to a tyre when a car hits the kerbs. We had no answer,
because nothing in this project models a kerb strike: a big one becomes an outlier
lap the MAD filter removes, or it ends the stint. The physics is not in doubt --
`research/papers/02_tyre_wear_physics/` models wear rate as proportional to
frictional power dissipated in the contact patch, and a kerb is a harder, ridged,
higher-slip surface than asphalt -- but nobody in our literature review has
measured the effect from public data.

This script builds two candidate measures per driver per lap, and the evidence
needed to decide whether either of them means anything.

**A. Lateral deviation from the racing line.** The obvious approach: a car running
wide over a kerb sits further from the line at a specific corner than one that
stayed on the asphalt. `data/telemetry/` cannot answer this because X/Y were
reduced to curvature and discarded, so position is re-derived from the FastF1
cache here.

**B. Track-limits excursions.** Every lap deleted by race control for exceeding
track limits, with the corner it happened at, taken from FastF1's `Deleted` and
`DeletedReason` lap columns. These are adjudicated, per-lap and per-corner.

**C. The feed-projection diagnostic, which decides whether A is usable at all.**
Two statistics per session: the distribution of lateral deviation over moving
on-track samples, and the minimum distance any pair of cars ever reached. The
second is the decisive one. Real position data cannot put two cars at the same
coordinate; a feed that projects on-track position onto a single path puts every
pair there eventually. A pilot on five sessions found exactly that -- all 190
pairs at the 2024 Austrian Grand Prix reach 0.00 m -- so this runs corpus-wide and
the result is reported as a finding rather than assumed away.

    python scripts/build_kerb_exposure.py --limit 2       # try it on two sessions
    python scripts/build_kerb_exposure.py --races-only    # what exp33 needs, ~1 h
    python scripts/build_kerb_exposure.py                 # practice as well, ~3 h

Output: `data/kerb/<session_id>.parquet`, one row per driver per lap, joining onto
`data/season/<session_id>.parquet` on (driver, session_lap); and
`data/kerb/diagnostics/<session_id>.json`, the measurement-validity evidence and
the per-corner histogram of where cars actually exceed track limits.

WHAT THIS IS NOT
----------------
1. **It is exposure, not impact.** Detecting an actual kerb strike needs the
   vertical accelerometer channel, which is team-private, exactly like tyre
   temperature and pressure. Neither measure here sees a kerb; A sees position and
   B sees a stewards' decision.
2. **The reference is a racing line, not a surveyed centreline.** It is where the
   fastest lap of *this session* went, built the same way
   `scripts/export_track_geometry.py` builds `data/geometry/`. Deviation is
   therefore relative to "what a fast lap did", and a driver running a different
   line for fuel, traffic or balance registers deviation without touching a kerb.
   `reference_stability_m` puts a number on that: the median lateral distance
   between the reference and one built from the next-fastest lap by a different
   driver.
3. **The position feed samples at ~4.2 Hz**, so consecutive samples are 7 m apart
   in a slow corner and 22 m apart on a straight. A kerb strike lasting 0.3 s can
   fall between two samples. What could survive that is *sustained* wide running.
4. **Every threshold is a choice**, so all of `THRESHOLDS_M` is measured and the
   experiment reports the sensitivity rather than picking one and hiding it.
5. **A track-limits excursion is not a kerb strike either.** It is the subset of
   wide running that a monitored corner caught, at the two or three corners per
   circuit that are monitored, and it is entangled with how hard the driver was
   pushing. It is the most direct public evidence of a car leaving the asphalt;
   it is not evidence of how many kerbs were ridden inside the white lines.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CORPUS_DIR = Path("data/season")
OUT_DIR = Path("data/kerb")
DIAGNOSTIC_DIR = OUT_DIR / "diagnostics"
CACHE = Path("cache/fastf1")

#: FastF1 position coordinates are in tenths of a metre, the same factor
#: `scripts/export_track_geometry.py` and `scripts/build_telemetry.py` apply.
UNITS_PER_METRE = 10.0

#: Reference-line resolution. 2000 points over a 5-7 km lap is ~3 m spacing, fine
#: enough that the nearest-segment projection below is exact to well under the
#: feed's own noise.
REFERENCE_POINTS = 2000

#: Lateral thresholds, in metres, at which exposure is counted. Deliberately a
#: sweep and not a choice: the low end is expected to be mostly disagreement about
#: the racing line and the high end mostly genuine wide running, and which is
#: which is a question for the experiment, not for this file.
THRESHOLDS_M = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)

#: Half-width of the window around a corner marker that counts as "at the corner".
#: FastF1's marker sits near the apex; kerbs run from entry to exit, so 75 m either
#: side covers the kerbed part of a typical corner without swallowing the next one.
CORNER_WINDOW_M = 75.0

#: Beyond this the car is not off-line, it is in the pit lane, in the gravel or
#: facing the wrong way. Recorded rather than dropped, so the experiment can see
#: how often it happens.
GROSS_EXCURSION_M = 10.0

#: A lap with fewer position samples than this cannot be summarised.
MIN_SAMPLES = 40

#: Grid for the pairwise-proximity diagnostic. 0.5 s is fine enough that two cars
#: genuinely side by side are seen at their closest, and coarse enough that twenty
#: cars over two hours stays a few thousand points.
PROXIMITY_STEP = "500ms"

#: A sample this far from the previous one is the car moving rather than sitting in
#: the garage or on the grid, where a parked car would otherwise contribute tens of
#: thousands of identical readings to the deviation distribution.
MOVING_STEP_M = 0.5

logging.getLogger("fastf1").setLevel(logging.ERROR)


def _threshold_key(threshold: float) -> str:
    """`2.5` -> `2m5`, so column names survive a round trip through parquet."""
    return f"{threshold:.1f}".replace(".", "m")


def resample_closed_loop(xy: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a closed lap to evenly spaced points, through a periodic spline.

    Straight-line resampling -- what `export_track_geometry.py` does, correctly, for
    a renderer -- chords across corners. The position feed gives ~18 m between raw
    samples, and a chord across 18 m of a 50 m-radius corner sits nearly a metre
    inside the true arc. That is the same size as the effect being measured, and it
    would appear at corners specifically, which is precisely where a false signal
    would be most convincing. A periodic cubic spline removes it.
    """
    from scipy.interpolate import CubicSpline

    loop = xy if np.linalg.norm(xy[-1] - xy[0]) < 1e-9 else np.vstack([xy, xy[0]])
    chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))])

    # CubicSpline needs a strictly increasing parameter; a stationary car on the
    # grid or under a red flag produces repeated positions.
    keep = np.concatenate([[True], np.diff(chord) > 1e-9])
    loop, chord = loop[keep], chord[keep]
    if len(loop) < 8 or chord[-1] <= 0:
        return xy

    loop[-1] = loop[0]  # periodic boundary conditions demand exact closure
    spline = CubicSpline(chord, loop, bc_type="periodic")
    return spline(np.linspace(0.0, chord[-1], n_points, endpoint=False))


def arc_length(points: np.ndarray) -> np.ndarray:
    """Cumulative distance along an open polyline, starting at zero."""
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


class ReferenceLine:
    """A racing line, and the machinery to ask how far a point sits from it."""

    def __init__(self, points: np.ndarray) -> None:
        from scipy.spatial import cKDTree

        closed = arc_length(np.vstack([points, points[0]]))
        self.points = points
        self.distance_m = closed[:-1]
        self.length_m = float(closed[-1])
        self.tree = cKDTree(points)

    def deviation(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Signed perpendicular distance to the line, and distance along it.

        Sign is positive to the left of the direction of travel. It is carried
        because inside-kerb use at an apex and outside-kerb use at an exit are
        different behaviours that a magnitude alone would merge.
        """
        _, index = self.tree.query(xy)
        n = len(self.points)
        best_d = np.full(len(xy), np.inf)
        best_s = np.zeros(len(xy))
        best_sign = np.ones(len(xy))

        # The nearest reference *point* over-reads the distance to the reference
        # *line* by up to half the point spacing. Projecting onto the two segments
        # that meet at that point recovers the true perpendicular distance.
        for offset in (-1, 0):
            start = (index + offset) % n
            end = (index + offset + 1) % n
            a, b = self.points[start], self.points[end]
            ab = b - a
            length_sq = np.einsum("ij,ij->i", ab, ab)
            length_sq = np.where(length_sq > 0, length_sq, 1.0)
            t = np.clip(np.einsum("ij,ij->i", xy - a, ab) / length_sq, 0.0, 1.0)
            foot = a + t[:, None] * ab
            d = np.linalg.norm(xy - foot, axis=1)
            sign = np.sign(ab[:, 0] * (xy[:, 1] - a[:, 1]) - ab[:, 1] * (xy[:, 0] - a[:, 0]))
            closer = d < best_d
            best_d = np.where(closer, d, best_d)
            best_sign = np.where(closer, np.where(sign == 0, 1.0, sign), best_sign)
            best_s = np.where(closer, self.distance_m[start] + t * np.sqrt(length_sq), best_s)

        return best_d * best_sign, best_s


def scaled_xy(frame: pd.DataFrame) -> np.ndarray:
    """On-track X/Y in metres, with unusable rows removed."""
    if frame is None or frame.empty:
        return np.empty((0, 2))
    if "Status" in frame.columns:
        frame = frame[frame["Status"] == "OnTrack"]
    xy = np.column_stack([frame["X"].to_numpy(dtype=float),
                          frame["Y"].to_numpy(dtype=float)]) / UNITS_PER_METRE
    return xy[np.isfinite(xy).all(axis=1)]


def lap_positions(lap) -> np.ndarray | None:
    """Metres-scaled X/Y for one lap, or None if the feed has nothing usable."""
    try:
        xy = scaled_xy(lap.get_pos_data())
    except Exception:  # noqa: BLE001 -- one bad lap must not end a session
        return None
    return xy if len(xy) >= MIN_SAMPLES else None


def lap_exposure(xy: np.ndarray, reference: ReferenceLine,
                 corner_s: np.ndarray) -> dict:
    """Reduce one lap's path to exposure summaries, whole-lap and corner-restricted.

    Exposure is weighted by *distance travelled*, not by sample count. A sample is
    one fixed slice of time, so counting samples would weight a slow corner far
    more heavily than a fast one -- which is a statement about speed, not about
    where the car was. Distance-weighting asks the question actually intended: how
    many metres of this lap were driven off the line.
    """
    deviation, along = reference.deviation(xy)
    absolute = np.abs(deviation)

    # Each sample owns the half-segment either side of it, so that the weights sum
    # to the path length rather than to the path length minus its last segment.
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    weight = np.concatenate([[step[0] / 2], (step[:-1] + step[1:]) / 2, [step[-1] / 2]])
    path_m = float(weight.sum())
    if not np.isfinite(path_m) or path_m <= 0:
        return {}

    clean = absolute < GROSS_EXCURSION_M
    features: dict[str, float] = {
        "n_samples": int(len(xy)),
        "path_length_m": path_m,
        "mean_abs_dev_m": float(np.average(absolute, weights=weight)),
        "p95_abs_dev_m": float(np.percentile(absolute, 95)),
        "max_abs_dev_m": float(absolute.max()),
        "mean_signed_dev_m": float(np.average(deviation, weights=weight)),
        "gross_excursion_frac": float(weight[~clean].sum() / path_m),
    }

    # Distance from the nearest corner marker, the short way round the lap.
    gap = np.abs(along[:, None] - corner_s[None, :])
    gap = np.minimum(gap, reference.length_m - gap)
    nearest_corner = gap.argmin(axis=1)
    at_corner = gap.min(axis=1) <= CORNER_WINDOW_M
    corner_m = float(weight[at_corner].sum())
    features["corner_path_length_m"] = corner_m

    for threshold in THRESHOLDS_M:
        key = _threshold_key(threshold)
        over = absolute > threshold
        features[f"exposure_m_{key}"] = float(weight[over].sum())
        features[f"exposure_frac_{key}"] = float(weight[over].sum() / path_m)
        features[f"corner_exposure_frac_{key}"] = (
            float(weight[over & at_corner].sum() / corner_m) if corner_m > 0 else np.nan)
        # Count a corner as "run wide" once, however many samples were over.
        features[f"n_corners_wide_{key}"] = int(
            len(np.unique(nearest_corner[over & at_corner])))

    if at_corner.any():
        worst = int(nearest_corner[at_corner][absolute[at_corner].argmax()])
        features["worst_corner_index"] = worst
        features["worst_corner_dev_m"] = float(absolute[at_corner].max())
    return features


def reference_stability(session, reference: ReferenceLine, fastest) -> float:
    """How far apart two fast drivers' idea of the racing line is, in metres.

    This is the honest floor on every threshold in `THRESHOLDS_M`. The reference is
    one lap by one driver; if the next-fastest driver's line sits a metre away from
    it everywhere, then "a metre off the line" is a sentence about which driver was
    quickest that day, not about kerbs.
    """
    others = session.laps[session.laps["Driver"] != fastest["Driver"]]
    others = others[others["LapTime"].notna()].sort_values("LapTime")
    for _, lap in others.head(3).iterrows():
        xy = lap_positions(lap)
        if xy is None:
            continue
        deviation, _ = reference.deviation(resample_closed_loop(xy, REFERENCE_POINTS))
        return float(np.median(np.abs(deviation)))
    return float("nan")


def feed_projection_diagnostic(session, reference: ReferenceLine) -> dict:
    """Is the position feed carrying lateral information at all?

    Two independent checks, because the conclusion they support kills the primary
    measure and one check would not be enough to justify that.

    The first is the distribution of lateral deviation over every moving on-track
    sample in the session. Twenty cars racing for two hours occupy a track twelve
    metres wide; if the ninety-ninth percentile of that distribution is a few
    centimetres, the feed is not reporting where on the track they were.

    The second is decisive and needs no reference line: the closest any pair of
    cars ever came to each other. Two Formula 1 cars cannot occupy the same point.
    A feed that reports position as distance along a single path puts every pair
    that ever laps another at zero separation, because it has thrown away the only
    coordinate that keeps them apart.
    """
    deviations, samples = [], 0
    for frame in session.pos_data.values():
        xy = scaled_xy(frame)
        if len(xy) < MIN_SAMPLES:
            continue
        step = np.concatenate([[0.0], np.linalg.norm(np.diff(xy, axis=0), axis=1)])
        moving = step > MOVING_STEP_M
        if moving.sum() < MIN_SAMPLES:
            continue
        deviation, _ = reference.deviation(xy[moving])
        absolute = np.abs(deviation)
        # Pit-lane running is a genuinely different path and would otherwise be
        # read as evidence that the feed does carry lateral information.
        deviations.append(absolute[absolute < GROSS_EXCURSION_M])
        samples += int(moving.sum())

    out: dict = {"n_moving_samples": samples}
    if deviations:
        pooled = np.concatenate(deviations)
        out |= {
            "n_on_track_samples": int(len(pooled)),
            "dev_median_m": float(np.median(pooled)),
            "dev_p90_m": float(np.percentile(pooled, 90)),
            "dev_p99_m": float(np.percentile(pooled, 99)),
            "dev_p999_m": float(np.percentile(pooled, 99.9)),
            "dev_max_m": float(pooled.max()),
        }

    tracks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    grid: np.ndarray | None = None
    for number, frame in session.pos_data.items():
        frame = frame[frame["Status"] == "OnTrack"] if "Status" in frame else frame
        if len(frame) < 200:
            continue
        if grid is None:
            span = pd.date_range(frame["Date"].min(), frame["Date"].max(), freq=PROXIMITY_STEP)
            grid = span.astype("int64").to_numpy()
        stamps = frame["Date"].astype("int64").to_numpy()
        tracks[str(number)] = (
            np.interp(grid, stamps, frame["X"].to_numpy(float)) / UNITS_PER_METRE,
            np.interp(grid, stamps, frame["Y"].to_numpy(float)) / UNITS_PER_METRE)

    if len(tracks) >= 2:
        closest = [float(np.hypot(ax - bx, ay - by).min())
                   for (ax, ay), (bx, by) in itertools.combinations(tracks.values(), 2)]
        out |= {
            "n_car_pairs": len(closest),
            "pair_min_separation_m": float(np.min(closest)),
            "pair_median_of_minima_m": float(np.median(closest)),
            "n_pairs_within_2m": int(sum(1 for c in closest if c < 2.0)),
            "n_pairs_exactly_coincident": int(sum(1 for c in closest if c < 0.05)),
        }
    return out


def track_limits(session) -> tuple[pd.DataFrame, dict]:
    """Every lap race control deleted for track limits, and where it happened.

    FastF1 already attributes each deletion to a driver and a lap number, which is
    a stewards' decision rather than an inference, so this reads those columns
    rather than re-parsing the race control message stream.
    """
    laps = session.laps
    if "Deleted" not in laps.columns:
        return pd.DataFrame(columns=["driver", "session_lap", "limits_corner"]), {}

    deleted = laps[laps["Deleted"].fillna(False).astype(bool)].copy()
    reason = deleted["DeletedReason"].astype(str).str.upper()
    deleted = deleted[reason.str.contains("TRACK LIMITS", na=False)]
    if deleted.empty:
        return pd.DataFrame(columns=["driver", "session_lap", "limits_corner"]), {}

    # "TURN 8 A LAP 3" is corner 8A; "TURN 19 LAP 16" is corner 19, and a pattern
    # that lets the letter float would read the L of LAP as a corner suffix and
    # invent corner "19L" at every circuit. Hence the word boundary.
    parts = deleted["DeletedReason"].astype(str).str.extract(
        r"TURN\s+(\d+)(?:\s*([A-Z])\b)?")
    corner = (parts[0] + parts[1].fillna("")).where(parts[0].notna())
    events = pd.DataFrame({
        "driver": deleted["Driver"].astype(str),
        "session_lap": deleted["LapNumber"].astype("Int64"),
        "limits_corner": corner.fillna("unknown"),
    }).dropna(subset=["session_lap"])
    events["session_lap"] = events["session_lap"].astype(int)
    histogram = events["limits_corner"].value_counts().to_dict()
    return events.drop_duplicates(subset=["driver", "session_lap"]), histogram


def process_session(session_id: str) -> tuple[int, str]:
    """Build one session's exposure table and diagnostics. Returns (rows, message)."""
    import fastf1

    destination = OUT_DIR / f"{session_id}.parquet"
    if destination.exists():
        return 0, "skip (exists)"

    year, rest = session_id.split("-", 1)
    event_slug, session_code = rest.rsplit("-", 1)
    session = fastf1.get_session(int(year), event_slug.replace("-", " ").title(), session_code)
    session.load(telemetry=True, laps=True, weather=False, messages=True)

    fastest = session.laps.pick_fastest()
    fastest_xy = lap_positions(fastest)
    if fastest_xy is None:
        return 0, "FAILED no position data on the fastest lap"
    reference = ReferenceLine(resample_closed_loop(fastest_xy, REFERENCE_POINTS))

    corners = session.get_circuit_info().corners
    if corners is None or corners.empty:
        return 0, "FAILED no corner markers"
    corner_xy = np.column_stack([corners["X"].to_numpy(dtype=float),
                                 corners["Y"].to_numpy(dtype=float)]) / UNITS_PER_METRE
    # Locate each corner on OUR reference rather than trusting the feed's own
    # Distance column, which is measured along a line we did not build.
    _, corner_index = reference.tree.query(corner_xy)
    corner_s = reference.distance_m[corner_index]
    corner_names = [f"{int(r['Number'])}{str(r['Letter']).strip()}"
                    if str(r["Letter"]).strip() else str(int(r["Number"]))
                    for _, r in corners.iterrows()]

    events, histogram = track_limits(session)
    flagged = set(zip(events["driver"], events["session_lap"], strict=True))
    corner_of = dict(zip(zip(events["driver"], events["session_lap"], strict=True),
                         events["limits_corner"], strict=True))

    stability = reference_stability(session, reference, fastest)
    diagnostic = feed_projection_diagnostic(session, reference)
    constants = {
        "reference_stability_m": stability,
        "reference_lap_length_m": reference.length_m,
        "n_corners": int(len(corner_s)),
        "feed_pair_min_separation_m": diagnostic.get("pair_min_separation_m", np.nan),
        "feed_dev_p99_m": diagnostic.get("dev_p99_m", np.nan),
    }

    rows = []
    for _, lap in session.laps.iterlaps():
        xy = lap_positions(lap)
        if xy is None:
            continue
        features = lap_exposure(xy, reference, corner_s)
        if not features:
            continue
        driver, number = str(lap["Driver"]), int(lap["LapNumber"])
        rows.append({
            "driver": driver, "session_lap": number,
            # Flags rather than filters. Pit laps and safety-car laps are wildly
            # off the racing line for reasons that have nothing to do with kerbs,
            # but dropping them here would force a 50-minute rebuild to revisit
            # the decision, so the experiment gets to make it.
            "is_pit_lap": bool(pd.notna(lap["PitInTime"]) or pd.notna(lap["PitOutTime"])),
            "track_status": str(lap["TrackStatus"]),
            "is_accurate": bool(lap["IsAccurate"]) if pd.notna(lap["IsAccurate"]) else False,
            "limits_event": (driver, number) in flagged,
            "limits_corner": corner_of.get((driver, number)),
            **features, **constants,
        })

    if not rows:
        return 0, "FAILED no usable laps"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC_DIR.joinpath(f"{session_id}.json").write_text(json.dumps({
        "session_id": session_id,
        "circuit": str(session.event["Location"]),
        "year": int(year),
        "reference_driver": str(fastest["Driver"]),
        "reference_stability_m": stability,
        "reference_lap_length_m": reference.length_m,
        "n_laps": len(rows),
        "n_limits_events": int(len(events)),
        "limits_by_corner": histogram,
        "corner_names": corner_names,
        **diagnostic,
    }, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_parquet(destination, index=False)
    return len(rows), (f"ok {len(rows)} laps, ref +/-{stability:.2f} m, "
                       f"{len(events)} limits events, "
                       f"closest pair {diagnostic.get('pair_min_separation_m', float('nan')):.2f} m")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sessions", nargs="*", help="specific session ids")
    parser.add_argument("--races-only", action="store_true",
                        help="skip practice: exp33 uses races, and a race session "
                             "carries a thousand laps to a practice session's two "
                             "hundred, so this is most of the runtime")
    args = parser.parse_args()

    import fastf1
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))

    ids = args.sessions or sorted(p.stem for p in CORPUS_DIR.glob("*.parquet"))
    if args.races_only:
        ids = [i for i in ids if i.endswith("-R")]
    if args.limit:
        ids = ids[: args.limit]

    started = time.time()
    done, failed = [], []
    for n, session_id in enumerate(ids, 1):
        try:
            _, message = process_session(session_id)
        except Exception as exc:  # noqa: BLE001 -- one bad session must not end the run
            message = f"FAILED {type(exc).__name__}: {exc}"
        print(f"[{n}/{len(ids)}] {session_id}: {message}", flush=True)
        (failed if message.startswith("FAILED") else done).append(session_id)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.joinpath("_run.json").write_text(json.dumps({
        "done": len(done), "failed": failed,
        "thresholds_m": list(THRESHOLDS_M),
        "corner_window_m": CORNER_WINDOW_M,
        "reference_points": REFERENCE_POINTS,
        "elapsed_s": round(time.time() - started, 1),
    }, indent=2), encoding="utf-8")
    print(f"\n{len(done)} done, {len(failed)} failed in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
