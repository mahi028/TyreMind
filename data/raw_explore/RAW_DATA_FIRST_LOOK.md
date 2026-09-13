# Raw FastF1 data — first look

Downloaded directly from FastF1 (which reads F1's official live-timing API), for the
same 4 events used in `data/demo/` — Monza, Silverstone, Zandvoort, Barcelona, 2024,
FP2 + Race — so it can be compared 1:1 against the already-processed lap tables.

This folder is the *raw* material. `data/demo/` is what's left after
[f1_loader.py](../../src/tyremind/data/f1_loader.py) filters and reduces it.

```
data/raw_explore/
  cache/                       FastF1's own on-disk API cache (raw JSON/pickle, not for direct reading)
  sessions/<session-id>/
    laps.parquet                one row per driver per lap — 31 raw columns
    weather.parquet             one row per ~60s — track/air conditions
    results.parquet              one row per driver — session classification
    track_status.parquet        event log of flag/SC/VSC status changes
    race_control_messages.parquet   the official FIA race control log
    meta.json                   session identifiers + row counts
  raw_session_summary.csv       cross-session stats (this report's numbers)
  column_glossary.json          every column below, machine-readable
```

## What `session.laps` actually contains (31 columns)

One row per driver per lap. Nothing has been filtered yet — pit laps, safety-car
laps, deleted laps and laps with no time at all are all still present.

| column | dtype | meaning |
|---|---|---|
| `Time` | timedelta | session time the lap time was registered (line crossing) |
| `Driver` | str | 3-letter code |
| `DriverNumber` | str | car number |
| `LapTime` | timedelta | lap duration — **null** for laps that never closed a clean sector chain |
| `LapNumber` | float | lap count, shared across the whole field |
| `Stint` | float | increments on every tyre change, starts at 1 |
| `PitOutTime` | timedelta | non-null only on an out-lap |
| `PitInTime` | timedelta | non-null only on an in-lap |
| `Sector1/2/3Time` | timedelta | duration of each sector |
| `Sector1/2/3SessionTime` | timedelta | absolute session time each sector split landed |
| `SpeedI1`, `SpeedI2` | float, km/h | speed-trap readings at the two intermediate points |
| `SpeedFL` | float, km/h | speed trap at the finish line |
| `SpeedST` | float, km/h | speed trap on the designated "speed trap" straight |
| `IsPersonalBest` | bool | best of session for that driver, at that point |
| `Compound` | str | SOFT/MEDIUM/HARD/INTERMEDIATE/WET/UNKNOWN — **relative** label, re-picked every event (see `data/reference/compound_allocation.json`) |
| `TyreLife` | float | cumulative laps on this physical tyre set, carried over from a prior session if not fresh — the tyre-age signal |
| `FreshTyre` | bool | was the set new when fitted |
| `Team` | str | constructor |
| `LapStartTime` | timedelta | session time the lap began |
| `LapStartDate` | datetime | wall-clock timestamp — **came back 100% null** in every session pulled here (see below) |
| `TrackStatus` | str | digit code(s) active during the lap: 1 AllClear, 2 Yellow, 4 SC, 5 Red, 6 VSC, 7 VSC-ending; concatenated if it changed mid-lap |
| `Position` | float | track position at lap end |
| `Deleted` | bool | stewards deleted this lap time |
| `DeletedReason` | str | free text, e.g. `TRACK LIMITS AT TURN 4 LAP 31` |
| `FastF1Generated` | bool | row was synthesized by FastF1 rather than read straight off the feed |
| `IsAccurate` | bool | FastF1's own consistency check — this is the trust flag the cleaning pipeline filters on |

## The four other raw tables a session carries

- **`weather_data`** — sampled independently of laps, roughly every 60s: `AirTemp`,
  `TrackTemp` (°C), `Humidity` (%), `Pressure` (mbar), `WindSpeed` (m/s),
  `WindDirection` (deg), `Rainfall` (bool). Track temp, not air temp, is the one that
  matters physically for degradation.
- **`results`** — final classification: identity fields, `Position` /
  `ClassifiedPosition` (can be `R`=retired, `D`=disqualified instead of a number),
  `GridPosition`, `Q1/Q2/Q3` (quali only), `Status` (`Finished`, `+1 Lap`, or a
  retirement reason), `Points`, `Laps`.
- **`track_status`** — a sparse **event log** (one row per status *change*, not per
  lap): when a yellow/SC/VSC/red period started or cleared.
- **`race_control_messages`** — the official FIA log: flags, DRS enable/disable,
  investigations, penalties — free text plus `Category`/`Flag`/`Scope`/`Lap`.

## Cross-session numbers (all 8 pulls)

| session | raw laps | drivers | max stint | pit stops | deleted | not-`IsAccurate` | no `LapTime` | compounds | max tyre life | rain seen | track temp °C |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| monza FP2 | 454 | 20 | 7 | 94 | 0 | 183 | 99 | MEDIUM, SOFT | 22 | no | 41.6–48.0 |
| **monza R** | 1008 | 20 | 4 | 31 | 13 | 81 | 0 | HARD, MEDIUM, SOFT | 43 | no | 43.5–54.6 |
| silverstone FP2 | 477 | 20 | 7 | 82 | 0 | 178 | 92 | HARD, INTERMEDIATE, MEDIUM, SOFT | 23 | yes | 25.0–32.7 |
| **silverstone R** | 960 | 19 | 5 | 46 | 9 | 110 | 0 | HARD, INTERMEDIATE, MEDIUM, SOFT | 28 | yes | 20.7–37.9 |
| zandvoort FP2 | 589 | 20 | 7 | 81 | 0 | 163 | 66 | HARD, MEDIUM, SOFT | 26 | no | 29.8–31.9 |
| **zandvoort R** | 1426 | 20 | 3 | 26 | 4 | 72 | 0 | HARD, MEDIUM, SOFT | 57 | no | 27.5–32.4 |
| barcelona FP2 | 571 | 20 | 6 | 91 | 0 | 181 | 73 | HARD, MEDIUM, SOFT | 25 | no | 41.2–44.2 |
| **barcelona R** | 1310 | 20 | 4 | 42 | 14 | 104 | 0 | HARD, MEDIUM, SOFT | 37 | yes | 38.3–43.6 |

Full table: [raw_session_summary.csv](raw_session_summary.csv).

### What jumps out

1. **Practice sessions have far more stints (6–7) than races (3–5)** — teams
   deliberately cycle through compounds in FP2 to gather data, which is exactly why
   the hackathon brief calls practice data noisier: more, shorter, more heterogeneous
   runs to strip signal from.

2. **`LapTime` is null on 66–99 laps per FP2 session, but on *zero* race laps.**
   Practice is full of installation laps, aborted runs and laps a driver simply
   backs out of that never produce a time; races essentially don't have this problem
   because every lap is either timed or clearly marked pit in/out. This is the raw
   version of what the processed data's `quality.json` reports as `no_lap_time`.

3. **`IsAccurate` fails far more in practice (~35–40% of laps) than in race
   (~8–11%)** — practice laps routinely include out-laps on old tyres, drivers
   backing off mid-lap, and track-limits excursions that never get formally deleted
   but that FastF1's own check still distrusts. This is the single biggest reason
   the processed retention rate for FP2 sessions (31–46%, see the earlier report)
   is roughly half that of races (64–95%).

4. **Deletions (`Deleted`=True) only appear in races here (4–14 laps), never in the
   four FP2 pulls** — stewards were not policing track limits as strictly, or laps
   were not being pushed hard enough to trigger the same violations, in these
   particular practice sessions.

5. **`Rainfall` reads `True` at some point in 3 of the 4 races** (Silverstone,
   Barcelona) **but 0 of the 4 practice sessions**, even though all four races still
   ran mostly on slicks. The sensor is genuinely more sensitive than "the track is
   wet" — a few drops trip it without changing compound choice. Silverstone's race
   also actually used INTERMEDIATE tyres for part of the field, which is the case
   where the flag corresponds to a real compound decision; Barcelona's `True` did
   not coincide with any non-slick compound being used, so there the flag is noise,
   not signal — worth checking `Rainfall` against `Compound`, not reading it alone.

6. **`TyreLife` maxes out at 57 laps (Zandvoort race)** — one driver ran a single
   set for more than a third of the race distance, the longest degradation curve in
   this set of sessions.

7. **`LapStartDate` came back entirely null in all 8 pulls.** FastF1 can populate it
   from the session's real-world start offset, but that path wasn't triggered by
   this load configuration (`laps=True, telemetry=False, weather=True,
   messages=True`). `LapStartTime` (session-relative) is fully populated and is what
   the pipeline actually uses — don't depend on `LapStartDate` being present.

8. **`track_status` is nearly empty for every session pulled** (typically a single
   `AllClear` row) — none of these four 2024 events had a safety car or red flag
   during the FP2/R sessions grabbed here. That means the diversity of
   confounders in this particular 8-session sample skews towards *fuel + traffic +
   track evolution*, not *safety-car resets* — a real limitation of first-looking at
   only these 4 events, and worth keeping in mind if the model needs to be validated
   against a chaotic race too.

## Reconciling raw vs. processed

Raw Monza Race is 1008 laps; the processed `data/demo/2024-monza-R.parquet` has
921. The gap (87 laps) is accounted for exactly by the exclusion cascade already
documented in `data/demo/2024-monza-R.quality.json`: 61 pit in/out laps, 20 flagged
inaccurate, 2 slow/safety-car outliers, 4 laps orphaned into a too-short run. Nothing
is lost silently — every one of those 87 laps is still sitting in
`data/raw_explore/sessions/2024-monza-R/laps.parquet` if you want to look at exactly
which ones.

## Caveats on this raw pull specifically

- Only 4 events × 2 sessions were pulled (not the full season corpus) — this is a
  first look, not the evidence base the model is trained on.
- `telemetry=False` was used (car position/speed traces), so this does **not**
  include the per-corner GPS/speed data that feeds the physics/3D layer — that's a
  much heavier pull (tens of MB per session) and a separate concern from the lap
  table.
- Timezone/session-time fields (`Time`, `LapStartTime`, sector session times) are
  all relative to session start, not wall-clock, except where noted.
