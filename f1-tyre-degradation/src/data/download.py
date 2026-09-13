"""Phase 1 — FastF1 acquisition.

Downloads every (season, event, session) 2022-2026 (to date) plus pre-season
testing, and persists each session as parquet, tagged with `era` and
`session_type`.

Sessions are enumerated PER EVENT from the schedule's own `Session1..Session5`
columns rather than a fixed list -- a sprint weekend has a different session
lineup (`Practice 1 / Sprint Qualifying / Sprint / Qualifying / Race`, no
FP2/FP3) than a conventional one, and a fixed list would silently skip or
mis-probe sessions on those weekends.

Fully resumable: a (year, round, session) already written to `data/raw/` is
skipped. Every failure is logged to `data/download_errors.csv` and the run
continues -- a missing or corrupt upstream session must never kill the whole
download.

Units: `weather_data.TrackTemp`/`AirTemp` in Celsius, `WindSpeed` in m/s, lap
times as pandas Timedelta (seconds via `.dt.total_seconds()`).
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

#: FastF1's underlying data providers cap requests at 500/hour. That budget is
#: ~7-8 calls per session (session info, driver list, status, lap count, track
#: status, timing data, timing app data, weather), so a few dozen sessions can
#: exhaust it well inside an hour. Once exhausted, EVERY subsequent call fails
#: instantly -- without backoff, a long run burns through its whole remaining
#: schedule in seconds, logging hundreds of false "failures" that are really
#: rate-limit hits, not missing or corrupt data.
RATE_LIMIT_BACKOFF_S = 300.0
RATE_LIMIT_MAX_RETRIES = 20  # 20 * 300s = 100 min, comfortably longer than the 1h window

#: Generic transient-failure retry (network hiccup, upstream 500, etc.) --
#: separate from and much shorter than the rate-limit backoff above, which
#: needs to wait out a whole rolling hour, not a few seconds.
GENERIC_RETRY_ATTEMPTS = 3
GENERIC_RETRY_BASE_S = 1.0

#: Full FastF1 session names (as they appear in `Session1..Session5`) to the
#: short codes `features.py` and the model expect as `session_type`.
SESSION_NAME_TO_CODE = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Qualifying": "Q",
    "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SQ",  # 2023 naming for the same session
    "Sprint": "SPRINT",
    "Race": "R",
}


def _is_rate_limit_error(exc: Exception) -> bool:
    return type(exc).__name__ == "RateLimitExceededError" or "RateLimitExceeded" in str(type(exc))


def _call_with_retries(fn, *, what: str):
    """Runs `fn()` with two layers of retry:

    - A `ValueError` (FastF1's signal for "this session/testing-slot does not
      exist") is never retried -- it is deterministic, not transient.
    - Any other exception gets `GENERIC_RETRY_ATTEMPTS` short exponential
      backoffs (transient network/upstream issues).
    - A rate-limit error escalates to the long `RATE_LIMIT_BACKOFF_S` wait,
      since a whole-hour quota cannot be waited out in a few seconds.
    """
    for rl_attempt in range(1, RATE_LIMIT_MAX_RETRIES + 1):
        rate_limited = False
        for attempt in range(1, GENERIC_RETRY_ATTEMPTS + 1):
            try:
                return fn()
            except ValueError:
                raise  # "does not exist" -- deterministic, no point retrying
            except Exception as exc:  # noqa: BLE001 - re-raised or retried below
                if _is_rate_limit_error(exc):
                    rate_limited = True
                    break
                if attempt == GENERIC_RETRY_ATTEMPTS:
                    raise
                backoff = GENERIC_RETRY_BASE_S * (2 ** (attempt - 1))
                logger.warning(
                    "transient error while %s (attempt %d/%d): %s -- retrying in %.0fs",
                    what, attempt, GENERIC_RETRY_ATTEMPTS, exc, backoff,
                )
                time.sleep(backoff)
        if rate_limited:
            logger.warning(
                "rate limit hit while %s (rate-limit attempt %d/%d) -- sleeping %.0fs",
                what, rl_attempt, RATE_LIMIT_MAX_RETRIES, RATE_LIMIT_BACKOFF_S,
            )
            time.sleep(RATE_LIMIT_BACKOFF_S)
            continue
        raise RuntimeError(f"unreachable retry state while {what}")  # pragma: no cover
    raise RuntimeError(f"rate limit still exceeded after {RATE_LIMIT_MAX_RETRIES} retries: {what}")


@dataclass
class EraBoundary:
    start_year: int
    end_year: int


@dataclass
class DownloadConfig:
    seasons: list[int]
    cache_dir: str
    raw_dir: str
    errors_csv: str
    rate_limit_sleep_s: float
    era_boundaries: dict[str, EraBoundary]
    testing_probe: dict[str, list[int]]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DownloadConfig":
        cfg = dict(yaml.safe_load(Path(path).read_text())["data"])
        cfg["era_boundaries"] = {
            name: EraBoundary(**bounds) for name, bounds in cfg["era_boundaries"].items()
        }
        return cls(**cfg)

    def era_for(self, year: int) -> str:
        for name, bounds in self.era_boundaries.items():
            if bounds.start_year <= year <= bounds.end_year:
                return name
        raise ValueError(f"no era configured for year {year} -- add it to era_boundaries")


def _session_dir(raw_dir: Path, year: int, round_number: int, session_code: str) -> Path:
    return raw_dir / f"{year}" / f"r{round_number:02d}_{session_code}"


def _already_downloaded(session_dir: Path) -> bool:
    return (session_dir / "laps.parquet").exists() and (session_dir / "meta.json").exists()


def _log_error(errors_csv: Path, year: int, event: str, session: str, error: Exception) -> None:
    errors_csv.parent.mkdir(parents=True, exist_ok=True)
    is_new = not errors_csv.exists()
    with open(errors_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["year", "event", "session", "error_type", "error_message"])
        writer.writerow([year, event, session, type(error).__name__, str(error)])


def _persist_session(session, out_dir: Path, *, year: int, event_name: str, round_number: int,
                      session_code: str, era: str, session_type: str) -> int:
    """Save laps, weather and metadata for one loaded session. Returns lap count."""
    out_dir.mkdir(parents=True, exist_ok=True)

    laps = session.laps.copy()
    # Serialise object columns FastF1 sometimes returns as mixed types so parquet doesn't choke.
    for col in laps.columns:
        if laps[col].dtype == "object":
            laps[col] = laps[col].astype(str)
    laps.to_parquet(out_dir / "laps.parquet")

    weather = session.weather_data.copy()
    weather.to_parquet(out_dir / "weather.parquet")

    meta = {
        "year": year,
        "event": event_name,
        "round": round_number,
        "session": session_code,
        "era": era,
        "session_type": session_type,
        "total_laps": int(len(laps)),
        "n_drivers": int(laps["Driver"].nunique()) if len(laps) else 0,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return len(laps)


def _enumerate_event_sessions(event: pd.Series) -> list[str]:
    """The event's own session lineup, in order -- handles conventional and
    sprint-format weekends correctly without guessing from a fixed list."""
    names = []
    for i in range(1, 6):
        name = event.get(f"Session{i}")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _download_regular_sessions(config: DownloadConfig, fastf1_module) -> list[dict]:
    fastf1 = fastf1_module
    raw_dir = Path(config.raw_dir)
    errors_csv = Path(config.errors_csv)
    rows: list[dict] = []

    for year in config.seasons:
        era = config.era_for(year)
        try:
            schedule = _call_with_retries(
                lambda: fastf1.get_event_schedule(year, include_testing=False),
                what=f"fetching {year} schedule",
            )
        except Exception as exc:  # noqa: BLE001 - a whole season failing must not kill the run
            logger.error("could not fetch schedule for %s: %s", year, exc)
            _log_error(errors_csv, year, "SCHEDULE", "-", exc)
            continue

        for _, event in schedule.iterrows():
            round_number = int(event["RoundNumber"])
            event_name = str(event["EventName"])

            for session_name in _enumerate_event_sessions(event):
                session_code = SESSION_NAME_TO_CODE.get(
                    session_name, session_name.upper().replace(" ", "_")
                )
                if session_code not in SESSION_NAME_TO_CODE.values():
                    logger.warning(
                        "unrecognised session name %r for %s %s -- using code %r",
                        session_name, year, event_name, session_code,
                    )
                session_dir = _session_dir(raw_dir, year, round_number, session_code)

                if _already_downloaded(session_dir):
                    rows.append(
                        {"year": year, "round": round_number, "event": event_name,
                         "session": session_code, "era": era, "outcome": "skipped_cached",
                         "laps": None}
                    )
                    continue

                try:
                    def _load(_year=year, _round=round_number, _name=session_name):
                        s = fastf1.get_session(_year, _round, _name)
                        s.load(laps=True, telemetry=False, weather=True, messages=True)
                        return s

                    loaded = _call_with_retries(
                        _load, what=f"{year} {event_name} R{round_number} {session_name}"
                    )
                    n_laps = _persist_session(
                        loaded, session_dir, year=year, event_name=event_name,
                        round_number=round_number, session_code=session_code, era=era,
                        session_type=session_code,
                    )
                    rows.append(
                        {"year": year, "round": round_number, "event": event_name,
                         "session": session_code, "era": era, "outcome": "downloaded",
                         "laps": n_laps}
                    )
                    logger.info("downloaded %s %s R%d %s: %d laps",
                                year, event_name, round_number, session_code, n_laps)
                except Exception as exc:  # noqa: BLE001 - log and continue, never abort the run
                    _log_error(errors_csv, year, event_name, session_code, exc)
                    rows.append(
                        {"year": year, "round": round_number, "event": event_name,
                         "session": session_code, "era": era, "outcome": "failed", "laps": None}
                    )
                    logger.warning("failed %s %s R%d %s: %s",
                                    year, event_name, round_number, session_code, exc)

                time.sleep(config.rate_limit_sleep_s)

    return rows


def _download_testing_sessions(config: DownloadConfig, fastf1_module) -> list[dict]:
    """Probes `testing_probe`'s bounded grid per year. A `ValueError` means
    that (test_number, session_number) slot doesn't exist for that year --
    expected for most of the grid, not a failure."""
    fastf1 = fastf1_module
    raw_dir = Path(config.raw_dir)
    errors_csv = Path(config.errors_csv)
    rows: list[dict] = []

    test_numbers = config.testing_probe["test_numbers"]
    session_numbers = config.testing_probe["session_numbers"]

    for year in config.seasons:
        era = config.era_for(year)
        for test_number in test_numbers:
            for session_number in session_numbers:
                session_code = f"TEST{test_number}S{session_number}"
                session_dir = _session_dir(raw_dir, year, 0, session_code)

                if _already_downloaded(session_dir):
                    rows.append(
                        {"year": year, "round": 0, "event": "Pre-Season Testing",
                         "session": session_code, "era": era, "outcome": "skipped_cached",
                         "laps": None}
                    )
                    continue

                try:
                    def _load(_year=year, _t=test_number, _s=session_number):
                        s = fastf1.get_testing_session(_year, _t, _s)
                        s.load(laps=True, telemetry=False, weather=True, messages=True)
                        return s

                    loaded = _call_with_retries(
                        _load, what=f"{year} testing test={test_number} session={session_number}"
                    )
                    n_laps = _persist_session(
                        loaded, session_dir, year=year, event_name="Pre-Season Testing",
                        round_number=0, session_code=session_code, era=era,
                        session_type="TESTING",
                    )
                    rows.append(
                        {"year": year, "round": 0, "event": "Pre-Season Testing",
                         "session": session_code, "era": era, "outcome": "downloaded",
                         "laps": n_laps}
                    )
                    logger.info("downloaded %s testing t%ds%d: %d laps",
                                year, test_number, session_number, n_laps)
                except ValueError:
                    # Doesn't exist -- expected for most of the probe grid, not an error.
                    rows.append(
                        {"year": year, "round": 0, "event": "Pre-Season Testing",
                         "session": session_code, "era": era, "outcome": "not_applicable",
                         "laps": None}
                    )
                except Exception as exc:  # noqa: BLE001 - log and continue
                    _log_error(errors_csv, year, "Pre-Season Testing", session_code, exc)
                    rows.append(
                        {"year": year, "round": 0, "event": "Pre-Season Testing",
                         "session": session_code, "era": era, "outcome": "failed", "laps": None}
                    )
                    logger.warning("failed %s testing t%ds%d: %s",
                                    year, test_number, session_number, exc)

                time.sleep(config.rate_limit_sleep_s)

    return rows


def download_all(config: DownloadConfig) -> pd.DataFrame:
    """Runs the full resumable download: every regular session plus the
    pre-season testing probe grid. Returns a summary dataframe.

    Args:
        config: Parsed `data:` block of `config/default.yaml`.

    Returns:
        One row per (year, round, session) attempted, with an `outcome` column
        in {"skipped_cached", "downloaded", "failed", "not_applicable"} (the
        last only for testing probe slots that don't exist) and a `laps` count.
    """
    import fastf1

    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    rows = _download_regular_sessions(config, fastf1)
    rows += _download_testing_sessions(config, fastf1)
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame) -> None:
    total = len(summary)
    succeeded = int((summary["outcome"] == "downloaded").sum())
    cached = int((summary["outcome"] == "skipped_cached").sum())
    failed = int((summary["outcome"] == "failed").sum())
    not_applicable = int((summary["outcome"] == "not_applicable").sum())
    total_laps = int(summary["laps"].fillna(0).sum())

    present = summary[summary["outcome"].isin(["downloaded", "skipped_cached"])]

    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"sessions attempted     : {total}")
    print(f"  downloaded           : {succeeded}")
    print(f"  already cached       : {cached}")
    print(f"  failed               : {failed}")
    print(f"  testing slot n/a     : {not_applicable}")
    print(f"total laps             : {total_laps}")
    print()
    print("present sessions per era:")
    print(present.groupby("era").size().to_string())
    print()
    print("laps per era:")
    print(present.groupby("era")["laps"].sum().to_string())
    print()
    testing_present = present[present["session"].str.startswith("TEST")]
    print("testing sessions present per year:")
    print(testing_present.groupby("year").size().to_string() if len(testing_present) else "  none")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = DownloadConfig.from_yaml("config/default.yaml")
    result = download_all(cfg)
    result.to_csv("data/download_summary.csv", index=False)
    print_summary(result)
