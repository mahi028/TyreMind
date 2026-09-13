"""Phase 2 (subset) — real-session cleaning and real-covariate extraction.

Reduces one downloaded session (`laps.parquet` + `weather.parquet` + `meta.json`,
as `download.py` produces them) to the real feature columns the model consumes,
applying the filter cascade from `PROMPT.md` Section 4. Every filter is a named
function returning a count removed, logged into the session's `FilterCascade`,
so retention is auditable rather than a black box.

This is deliberately a SUBSET of the full Phase 2 spec (train/val/test season
splits, scaler fitting, `is_quali_sim` flagging are NOT here) -- built specifically
to supply real covariates for the semi-synthetic validation
(`src/sim/semi_synthetic.py`), though it is directly reusable when full Phase 2 is
built later.

Units: `track_temp`/`air_temp` in Celsius, `wind_speed` in m/s, `gap_ahead_s` and
`lap_time` in seconds, `session_progress` in [0, 1].
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

#: Same era boundaries as `download.py`'s config-driven `era_for` -- duplicated
#: here as a simple standalone function since this module doesn't need the rest
#: of `DownloadConfig`. Keep in sync with `config/default.yaml`'s `era_boundaries`.
_ERA_BOUNDARIES = {"GE1": (2022, 2022), "GE2": (2023, 2025), "R26": (2026, 2026)}

_WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
_UNUSABLE_COMPOUNDS = {"UNKNOWN", "TEST_UNKNOWN", "NAN", "NONE", ""}

#: Gap to the car ahead at which a lap is dropped as traffic-compromised
#: (Section 4's filter), vs. the wider 10s clip applied to the surviving laps'
#: `gap_ahead_s` feature.
TRAFFIC_DROP_GAP_S = 2.0
GAP_CLIP_S = 10.0


def era_for(year: int) -> str:
    for name, (start, end) in _ERA_BOUNDARIES.items():
        if start <= year <= end:
            return name
    raise ValueError(f"no era configured for year {year}")


@dataclass
class FilterCascade:
    """Named filter -> laps removed, in application order, plus final counts."""

    session_id: str
    total_laps: int = 0
    removed: dict[str, int] = field(default_factory=dict)
    retained_laps: int = 0
    whole_session_dropped_reason: str | None = None

    def to_dict(self) -> dict:
        out = {"session_id": self.session_id, "total_laps": self.total_laps,
               "retained_laps": self.retained_laps, "removed": self.removed}
        if self.whole_session_dropped_reason:
            out["whole_session_dropped_reason"] = self.whole_session_dropped_reason
        return out


def _real_gap_ahead_s(laps: pd.DataFrame) -> pd.Series:
    """Gap to the nearest preceding car (different driver), in seconds, clipped
    at `GAP_CLIP_S`. Recoverable from lap start times alone: two cars that cross
    the line n seconds apart are n seconds apart on track for the lap that
    follows. Comparison is across session time, not lap number, since two
    drivers' "lap 5" can be minutes apart in practice.
    """
    if laps["LapStartTime"].isna().all():
        return pd.Series(GAP_CLIP_S, index=laps.index)

    start = pd.to_timedelta(laps["LapStartTime"]).dt.total_seconds()
    drivers = laps["Driver"].to_numpy()

    order = np.argsort(start.to_numpy(dtype=float), kind="stable")
    ordered_times = start.to_numpy(dtype=float)[order]
    ordered_drivers = drivers[order]

    gaps = np.full(len(order), np.inf)
    for position in range(1, len(order)):
        look_back = position - 1
        while look_back >= 0 and ordered_drivers[look_back] == ordered_drivers[position]:
            look_back -= 1
        if look_back >= 0:
            gaps[position] = ordered_times[position] - ordered_times[look_back]

    clipped = np.clip(gaps, 0.0, GAP_CLIP_S)
    clipped[~np.isfinite(gaps)] = GAP_CLIP_S

    result = pd.Series(GAP_CLIP_S, index=laps.index, dtype=float)
    result.iloc[order] = clipped
    return result


def _lap_in_stint(laps: pd.DataFrame) -> pd.Series:
    """Laps completed on the current stint: `TyreLife - 1` for a fresh tyre
    (verified directly against raw, unfiltered data: every fresh-tyre stint's
    first lap has `TyreLife == 1`, with zero exceptions checked). For a
    non-fresh (scrubbed/scored) set, the true zero-point isn't recoverable from
    public data at all -- same accepted limitation as upstream projects that
    do this same correction.

    Earlier this used `TyreLife - TyreLife.min()` per (driver, stint) --
    correct on paper (ported from the sibling TyreMind project's
    `laps_completed_in_run`, which needs it for GAPS in the middle of a run,
    e.g. a safety car), but wrong here: this project's own traffic filter
    (Section 4 -- drop laps with a <2s gap ahead) disproportionately removes
    the FIRST laps of a stint, since cars often bunch up right after a pit
    stop. Measured directly on the real 2022-2023 corpus: 99.8% of stints do
    not survive with their true first lap, 55% are missing their first 3+ laps
    (max observed: 39). `TyreLife.min()` over what's LEFT after that head
    truncation is not the stint's true start -- it silently invents a later
    "start" for the fuel feature (`fuel_kg_rel`, built from this), while
    `tyre_age` (raw `TyreLife`) stays correct throughout. `TyreLife - 1` is
    immune to this by construction: it never depends on which rows survived.
    """
    return (laps["TyreLife"].astype(float) - 1.0).clip(lower=0.0)


def clean_session(session_dir: Path, *, include_wet: bool = False,
                    min_stint_laps: int = 4, outlier_mad_multiplier: float = 3.0
                    ) -> tuple[pd.DataFrame, FilterCascade]:
    """Reduces one session directory to a real-covariate lap table.

    Args:
        session_dir: A directory `download.py` wrote (`laps.parquet`,
            `weather.parquet`, `meta.json`).
        include_wet: Keep intermediate/wet compounds (off by default -- the
            model's priors don't describe wet-tyre physics).
        min_stint_laps: Shortest stint retained after all other filters.
        outlier_mad_multiplier: Robust-deviation threshold for the per-stint
            outlier filter.

    Returns:
        `(lap_table, cascade)`. `lap_table` is empty if the whole session was
        dropped (e.g. a wet session) or nothing survived -- check
        `cascade.whole_session_dropped_reason` / `len(lap_table) == 0`.
    """
    meta = json.loads((session_dir / "meta.json").read_text())
    session_id = f"{meta['year']}_{meta.get('round', 0)}_{meta.get('session', '?')}"
    laps = pd.read_parquet(session_dir / "laps.parquet")
    weather = pd.read_parquet(session_dir / "weather.parquet")

    cascade = FilterCascade(session_id=session_id, total_laps=len(laps))

    def drop(mask: pd.Series, reason: str) -> None:
        nonlocal laps
        removed = int(mask.sum())
        if removed:
            cascade.removed[reason] = cascade.removed.get(reason, 0) + removed
        laps = laps.loc[~mask].copy()

    # Rain: drop the WHOLE session, not just wet laps -- wet-tyre degradation is a
    # different physical process the model's priors don't describe.
    if bool(weather["Rainfall"].any()):
        cascade.whole_session_dropped_reason = "rainfall_in_session"
        cascade.retained_laps = 0
        return laps.iloc[0:0], cascade

    drop(laps["LapTime"].isna(), "no_lap_time")
    drop(laps["TrackStatus"].astype(str) != "1", "not_green_flag")
    drop(laps["PitInTime"].notna() | laps["PitOutTime"].notna(), "pit_in_out_lap")
    drop(laps["Deleted"].astype(str) == "True", "deleted")

    compound = laps["Compound"].astype(str).str.upper()
    drop(compound.isin(_UNUSABLE_COMPOUNDS), "unknown_compound")
    if not include_wet:
        drop(laps["Compound"].astype(str).str.upper().isin(_WET_COMPOUNDS), "wet_compound")

    if laps.empty:
        cascade.retained_laps = 0
        return laps, cascade

    drop(laps["TyreLife"].isna() | (laps["TyreLife"] > 45), "tyre_life_null_or_over_45")

    if laps.empty:
        cascade.retained_laps = 0
        return laps, cascade

    gap_ahead_s = _real_gap_ahead_s(laps)
    drop(gap_ahead_s < TRAFFIC_DROP_GAP_S, "traffic_under_2s")
    gap_ahead_s = gap_ahead_s.loc[laps.index]  # re-align after the drop

    if laps.empty:
        cascade.retained_laps = 0
        return laps, cascade

    # Per-stint outlier filter: lap time > median(stint) + k * MAD(stint).
    seconds = laps["LapTime"].dt.total_seconds()
    stint_key = [laps["Driver"], laps["Stint"]]
    stint_median = seconds.groupby(stint_key).transform("median")
    stint_mad = (seconds - stint_median).abs().groupby(stint_key).transform("median") * 1.4826
    threshold = stint_median + outlier_mad_multiplier * stint_mad.clip(lower=0.05)
    drop(seconds > threshold, "stint_outlier")
    gap_ahead_s = gap_ahead_s.loc[laps.index]

    if laps.empty:
        cascade.retained_laps = 0
        return laps, cascade

    stint_sizes = laps.groupby(["Driver", "Stint"])["LapNumber"].transform("size")
    drop(stint_sizes < min_stint_laps, "stint_too_short")
    gap_ahead_s = gap_ahead_s.loc[laps.index]

    if laps.empty:
        cascade.retained_laps = 0
        return laps, cascade

    # ---- feature construction on the surviving laps ----
    lap_start_s = pd.to_timedelta(laps["LapStartTime"]).dt.total_seconds()
    span = lap_start_s.max() - lap_start_s.min()
    session_progress = ((lap_start_s - lap_start_s.min()) / span) if span > 0 else lap_start_s * 0.0

    weather_sorted = weather.sort_values("Time")
    joined = pd.merge_asof(
        laps[["LapStartTime"]].rename(columns={"LapStartTime": "Time"}).sort_values("Time"),
        weather_sorted, on="Time", direction="nearest",
    )

    out = pd.DataFrame(
        {
            "driver": laps["Driver"].astype(str).to_numpy(),
            "team_id": laps["Team"].astype(str).to_numpy(),
            "circuit_id": meta["event"],
            "event_id": session_id,
            "season": meta["year"],
            "era": era_for(meta["year"]),
            "session_type": meta.get("session_type", meta.get("session", "?")),
            "stint_id": laps["Stint"].astype(int).to_numpy(),
            "tyre_age": laps["TyreLife"].astype(float).to_numpy(),
            "lap_in_stint": _lap_in_stint(laps).to_numpy(),
            "compound": laps["Compound"].astype(str).str.upper().to_numpy(),
            "is_fresh_tyre": laps["FreshTyre"].astype(bool).to_numpy(),
            "session_progress": session_progress.to_numpy(),
            "gap_ahead_s": gap_ahead_s.to_numpy(),
            "lap_number": laps["LapNumber"].astype(float).to_numpy(),
            "real_lap_time": laps["LapTime"].dt.total_seconds().to_numpy(),
        }
    )
    out["track_temp"] = joined["TrackTemp"].to_numpy()
    out["air_temp"] = joined["AirTemp"].to_numpy()
    out["wind_speed"] = joined["WindSpeed"].to_numpy()
    out["humidity"] = joined["Humidity"].to_numpy()
    out["pressure"] = joined["Pressure"].to_numpy()

    cascade.retained_laps = len(out)
    return out, cascade


def clean_corpus(raw_dir: Path, *, session_code: str = "R") -> tuple[pd.DataFrame, list[FilterCascade]]:
    """Cleans every session matching `session_code` (default: races, for the
    longest/most varied real stints) under `raw_dir`. Concatenates into one
    real-covariate table, with a globally unique `driver_id`/`circuit_id`
    encoding deferred to the caller (`Vocabs.fit`, which handles arbitrary
    string ids already).

    Returns:
        `(lap_table, cascades)` -- one cascade per session attempted, including
        sessions dropped entirely (wet, or nothing survived).
    """
    frames = []
    cascades = []
    for session_dir in sorted(raw_dir.glob(f"*/r*_{session_code}")):
        if not (session_dir / "laps.parquet").exists():
            continue
        table, cascade = clean_session(session_dir)
        cascades.append(cascade)
        if len(table):
            frames.append(table)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, cascades


def print_cascade_table(cascades: list[FilterCascade]) -> None:
    """Section 4's required output: a filter cascade table, laps in -> laps out."""
    total_in = sum(c.total_laps for c in cascades)
    total_out = sum(c.retained_laps for c in cascades)
    dropped_sessions = sum(1 for c in cascades if c.whole_session_dropped_reason)

    removed_totals: dict[str, int] = {}
    for c in cascades:
        for reason, count in c.removed.items():
            removed_totals[reason] = removed_totals.get(reason, 0) + count

    print("=" * 60)
    print("FILTER CASCADE")
    print("=" * 60)
    print(f"sessions attempted     : {len(cascades)}")
    print(f"sessions dropped (wet) : {dropped_sessions}")
    print(f"laps in                : {total_in}")
    for reason, count in removed_totals.items():
        print(f"  - {reason:<28}: -{count}")
    print(f"laps out               : {total_out}")
    print(f"retention rate         : {total_out / total_in:.1%}" if total_in else "n/a")
    print("=" * 60)


if __name__ == "__main__":
    table, cascades = clean_corpus(Path("data/raw"))
    print_cascade_table(cascades)
    print(f"\nreal-covariate table: {len(table)} rows, {table['event_id'].nunique()} sessions")
