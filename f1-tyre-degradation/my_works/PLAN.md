# Build Plan — F1 Tyre Degradation Isolation

Source of truth for phase content: `my_works/PROMPT.md`. This file tracks status only;
`my_works/PROGRESS.md` is the append-only log updated after every phase.

| Phase | What | Status |
|---|---|---|
| 1 | Data acquisition — FastF1 laps/weather/testing, 2022–2026, era-tagged | **REWORKING** (see below) |
| 3 | Synthetic simulator + recovery test | **IN PROGRESS** — not yet passing all 4 thresholds |
| 5 (synthetic) | NAM heads + assembly, fit on synthetic only | Built; tuning against Phase 3 |
| 2 | Real-data cleaning + feature construction | Not started (`clean.py` missing) |
| 4 | Baselines (linear/polynomial/LightGBM) | Skeleton written (`src/models/baselines.py`), untested |
| 5 (real) | Fit NAM on real data | Not started — blocked on Phase 3 passing + Phase 2 |
| 6 | Race validation (practice→race) | Skeleton written (`src/validate_race.py`), untested |
| 7 | Figures, strategy tool | Strategy tool written (`src/strategy.py`), smoke-tested; figures not started |

## What changed in the PROMPT.md revision (vs. the version Phase 1/3/5-synthetic were built against)

1. **Three tyre eras, not one corpus**: `GE1` (2022–mid2023), `GE2` (mid2023–2025), `R26` (2026).
   Train on GE1+GE2, fine-tune on R26 with an **era embedding added to `TyreHead`'s conditioning
   vector**. Do not pool naively.
2. **Download scope widened**: all of 2022–2026 (to date), sessions FP1/FP2/FP3/**SQ/Sprint**/Q/R
   — previously FP1–FP3/Q/R only.
3. **Pre-season testing sessions are now in scope** (`fastf1.get_testing_session`), tagged
   `session_type='TESTING'`, kept in the training set — called out as the highest-quality data for
   this task (empty track, high fuel, deliberate long constant-setup runs).
4. **`session.load(..., messages=True)`** — was `messages=False` in the version Phase 1 was built
   against.
5. **Retry policy**: exponential backoff, 3 attempts, before marking a session failed — in addition
   to (not instead of) the rate-limit-specific backoff already implemented from the first pass
   (FastF1's provider caps at 500 calls/hour; that needs a longer wait than 3 short retries).
6. **Expected-yield acceptance table** added (Section 3.5): >80% of expected sessions present,
   testing present every season, total clean laps > 80,000, per-(circuit, compound, era) stint-count
   table produced, cells under 15 stints flagged for widened uncertainty.
7. New features: `era`, `session_type` — added to Phase 2's feature list.
8. **CLAUDE.md workflow rule (new)**: one phase at a time, stop and report at each phase's
   acceptance gate before continuing, plan mode before any phase touching more than one file.

## Carried over from the first pass (already resolved, still valid)

- Rate-limit backoff fix for `download.py` (FastF1: 500 calls/hour).
- Two real bugs found and fixed via the Phase 3 recovery test:
  - `CircuitHead`/`EntryHead` bias embeddings need a data-informed warm start (a ~75–110s
    constant does not converge from a zero init in any realistic epoch budget under Adam).
  - `EvoHead` was over-fitting and contaminating the recovered tyre slope, because
    `session_progress` and `tyre_age` are collinear within a stint. Fixed by (a) constraining
    `EvoHead` to a 2-parameter-per-circuit saturating exponential instead of a free MLP, and
    (b) keying `EntryHead`'s embedding per (event, driver, **stint**) instead of per (event,
    driver), since fuel-offset drift between a driver's stints was leaking into the evolution term.
- Recovery test not yet reliably passing all four thresholds (slope tolerance, cliff tolerance,
  evolution correlation, negative control) — still being tuned; see `PROGRESS.md` for the exact
  numbers from each attempt.
