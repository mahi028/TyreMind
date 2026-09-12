"""Multi-car proximity, reconstructed from real position telemetry.

Every function upstream of this module (`dynamics.py`) treats one car at a
time. But wake and traffic are relational: what matters for a tyre is not
this car's speed and line alone, but how far the nearest car is *ahead of it,
along the track it is actually driving* -- not raw X/Y distance, which would
call a car on the inbound leg of a hairpin "close" to one on the outbound leg
a few metres away but a full corner apart.

Public telemetry has exactly what this needs: every car's position,
timestamped, at several Hz. This module aligns every car in a session onto
one shared time grid, turns each car's position into cumulative distance
travelled since the session started (FastF1's own `add_distance`, integrated
from `Speed`), and reads off the one quantity `f1_loader`'s `_traffic_index`
could only approximate from lap start times: the actual gap to the car ahead,
at any instant, for every car simultaneously -- and, from the gap matrix, who
is running in a multi-car train rather than isolated traffic.

What this recovers is real, not a proxy: an along-track gap, from measured
position, at a known instant. What is a design choice, not a measurement:
the distance that calls two cars "a train" -- there is no bright physical
line, so `TRAIN_GAP_THRESHOLD_M` is a stated, documented cutoff, not a fitted
one, exactly like `f1_loader.CLEAN_AIR_GAP_S` before it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Along-track gap at or below which a car is considered to be running in the
#: wake of the one ahead -- a train, for counting how many cars are
#: compromising each other at once. Roughly two car lengths plus a reaction
#: gap; not derived from an aero model, since none is public. Chosen wider
#: than a single car length so a train is not fragmented by GPS jitter.
TRAIN_GAP_THRESHOLD_M = 50.0


def resample_to_common_grid(
    per_driver: dict[str, pd.DataFrame], *, freq_s: float = 0.5
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Interpolate every driver's cumulative distance onto one shared time axis.

    Args:
        per_driver: driver -> telemetry with `SessionTime` and `Distance`
            columns, as produced by calling `.add_distance()` on car data
            merged with position data (`car.merge_channels(pos).add_distance()`).
        freq_s: Sample spacing of the common grid, seconds.

    Returns:
        `(common_time_s, {driver: distance_m at each common_time_s})`. A
        driver's distance is NaN outside the span its own telemetry actually
        covers, so a car that has not yet started or has retired does not get
        a manufactured position by extrapolation.
    """
    spans = [
        (
            tel["SessionTime"].dt.total_seconds().min(),
            tel["SessionTime"].dt.total_seconds().max(),
        )
        for tel in per_driver.values()
        if len(tel)
    ]
    if not spans:
        return np.array([]), {}

    t0, t1 = min(s[0] for s in spans), max(s[1] for s in spans)
    common = np.arange(t0, t1, freq_s)

    out: dict[str, np.ndarray] = {}
    for driver, tel in per_driver.items():
        t = tel["SessionTime"].dt.total_seconds().to_numpy(dtype=float)
        d = tel["Distance"].to_numpy(dtype=float)
        valid = np.isfinite(t) & np.isfinite(d)
        if valid.sum() < 2:
            out[driver] = np.full(common.shape, np.nan)
            continue
        t_valid, d_valid = t[valid], d[valid]
        interp = np.interp(common, t_valid, d_valid, left=np.nan, right=np.nan)
        # np.interp clamps to the edge value outside the source range rather
        # than returning `left`/`right` for points beyond a *ragged* series
        # that starts after / ends before the common grid's own bounds.
        interp[(common < t_valid.min()) | (common > t_valid.max())] = np.nan
        out[driver] = interp
    return common, out


@dataclass(frozen=True)
class ProximitySnapshot:
    """Every present car's gap to the car ahead, at one instant."""

    #: Drivers present at this instant, ordered back to front.
    order: list[str]
    #: driver -> along-track distance to the next car ahead, metres. `inf`
    #: for whoever is at the front of the field present in this snapshot.
    gap_ahead_m: dict[str, float]


def snapshot_at(distances: dict[str, float]) -> ProximitySnapshot | None:
    """Reduce one instant's per-driver distances to an ordering and gaps.

    Args:
        distances: driver -> cumulative distance at this instant. Drivers
            with a non-finite distance (not yet running, retired, outside
            this telemetry's span) are dropped rather than guessed at.

    Returns:
        None if fewer than two cars are present -- a gap needs two cars.
    """
    present = {d: v for d, v in distances.items() if np.isfinite(v)}
    if len(present) < 2:
        return None
    order = sorted(present, key=lambda d: present[d])
    gaps = {
        order[i]: (present[order[i + 1]] - present[order[i]] if i + 1 < len(order) else float("inf"))
        for i in range(len(order))
    }
    return ProximitySnapshot(order=order, gap_ahead_m=gaps)


def _train_sizes(order: list[str], gap_ahead_m: dict[str, float], threshold_m: float) -> dict[str, int]:
    """Size of the contiguous train each driver belongs to.

    A train is a maximal run of consecutive cars (in track order) where every
    gap between neighbours is at or below `threshold_m`. A car with nobody
    close on either side is its own train of size one.
    """
    groups: list[list[str]] = [[order[0]]] if order else []
    for i in range(1, len(order)):
        gap_from_previous = gap_ahead_m[order[i - 1]]
        if gap_from_previous <= threshold_m:
            groups[-1].append(order[i])
        else:
            groups.append([order[i]])
    return {driver: len(group) for group in groups for driver in group}


def traffic_field(
    common_time: np.ndarray,
    distances: dict[str, np.ndarray],
    *,
    train_gap_m: float = TRAIN_GAP_THRESHOLD_M,
) -> pd.DataFrame:
    """Reduce the whole field's synchronized positions to one row per driver per instant.

    This is the traffic signal `f1_loader._traffic_index` could only
    approximate: `gap_ahead_m` here is a measured along-track distance at a
    known instant, not inferred from when two laps happened to start, and it
    is available for every car simultaneously rather than one nearest-ahead
    scalar.

    Args:
        common_time: Shared time axis, seconds, from `resample_to_common_grid`.
        distances: driver -> distance_m at each `common_time`, same source.
        train_gap_m: Gap at or below which consecutive cars count as one train.

    Returns:
        Long-form frame with columns `time_s`, `driver`, `gap_ahead_m`,
        `driver_ahead`, `in_train` (this car's train has 2+ cars), and
        `train_size`.
    """
    rows: list[dict] = []
    for t, per_driver_distance in zip(common_time, _transpose(distances, len(common_time))):
        snap = snapshot_at(per_driver_distance)
        if snap is None:
            continue
        sizes = _train_sizes(snap.order, snap.gap_ahead_m, train_gap_m)
        for pos, driver in enumerate(snap.order):
            rows.append(
                {
                    "time_s": float(t),
                    "driver": driver,
                    "gap_ahead_m": snap.gap_ahead_m[driver],
                    "driver_ahead": snap.order[pos + 1] if pos + 1 < len(snap.order) else None,
                    "in_train": sizes[driver] > 1,
                    "train_size": sizes[driver],
                }
            )
    return pd.DataFrame(
        rows,
        columns=["time_s", "driver", "gap_ahead_m", "driver_ahead", "in_train", "train_size"],
    )


def _transpose(distances: dict[str, np.ndarray], n: int) -> list[dict[str, float]]:
    """`{driver: array}` -> one `{driver: value}` dict per time step, in order."""
    drivers = list(distances)
    return [{d: distances[d][i] for d in drivers} for i in range(n)]
