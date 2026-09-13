# Progress Log

Append-only. One entry per phase checkpoint (or significant sub-step within a long phase).

---

## 2026-09-12 — Session 1: scaffold, Phase 1 (v1), Phase 3 + Phase 5 (synthetic)

**Scaffold**: `f1-tyre-degradation/` created under the `Trackshift` repo, on branch
`feature/nam-tyre-degradation`. Python 3.11 venv, all deps installed (`torch` 2.14, `fastf1` 3.8.3,
`lightgbm` 4.7 — needed `brew install libomp` for LightGBM to import on macOS).

**Phase 1 (v1, against the original PROMPT.md)**: `src/data/download.py` written — resumable,
per-session try/except, errors logged. First run hit FastF1's 500-calls/hour rate limit within
~13 minutes and then burned through the rest of the schedule in seconds, marking 196 sessions as
false "failed" (every subsequent call failed instantly once the quota was hit). Fixed with a
retry-with-backoff wrapper (`RATE_LIMIT_BACKOFF_S=300`, up to 20 retries) around both the schedule
fetch and the per-session load. Restarted; reached **68 sessions cached** (2022–2025, FP1-3/Q/R only)
before being stopped to rework against the revised PROMPT.md (see below) — those 68 sessions are
still valid data under the new spec, just incomplete (missing testing sessions, Sprint/SQ, 2026).

**Phase 3 (simulator) + Phase 5 (synthetic)**: built `src/sim/simulator.py`, `src/models/heads.py`,
`src/models/nam.py`, `src/models/losses.py`, `src/models/dataset.py`, `src/train.py`,
`src/evaluate.py`. `tests/test_head_allowlists.py` passes (9/9) — the per-head column allowlist is
enforced at `NAM.__init__` and refuses construction if any head is wired outside its permitted set.

Recovery test (`tests/test_recovery.py`) debugging, in order:

1. **First real bug**: `CircuitHead`/`EntryHead` bias embeddings start at zero; circuit base pace is
   ~75-110s. Adam's per-step update is roughly LR-sized regardless of gradient magnitude, so climbing
   from 0 to ~100 needs on the order of 100/lr steps — tens of thousands at lr=1e-3, far more than
   any reasonable epoch budget. Fixed with `NAM.warm_start_circuit_bias`, called from `train.fit`
   before the training loop, seeding each circuit's bias from its observed mean lap time.
   Val Huber loss dropped from ~0.5-0.6 to ~0.06-0.09 immediately after this fix.

2. **Second real bug**: recovered tyre slope was ~0.05-0.06 s/lap too steep, uniformly across all
   three compounds (a near-constant additive offset, not noise) — and got WORSE with more training,
   ruling out "just needs more epochs." Traced to `EvoHead`: a free 2-layer MLP recovered an
   evolution amplitude 3-5x too large. Root cause: within one stint, `session_progress` and
   `tyre_age` move together almost linearly (the same collinearity the sibling TyreMind project's
   README documents as "the second collinearity, not anticipated," resolved there with a "saturating
   basis + amplitude prior"). Two fixes applied together:
   - Replaced `EvoHead`'s free MLP with a 2-parameter-per-circuit saturating exponential
     (`-softplus(amplitude) * (1 - exp(-t/softplus(tau)))`), matching the true generative shape and
     removing the excess capacity that was absorbing tyre signal.
   - Keyed `EntryHead`'s embedding per **(event, driver, stint)** instead of per (event, driver).
     Section 1.2's claim that a per-stint fuel-offset constant "is absorbed by b_entry" is only true
     for a single-stint driver; with 2-4 stints per driver (as specified), the true remaining fuel at
     each stint's start differs stint to stint, and a single event-driver constant can only capture
     the average, leaving a residual that correlates with session_progress (early stints = more
     fuel = higher offset; late stints = less fuel = lower offset) — which is exactly what a
     session-progress-conditioned `EvoHead` can and did partially absorb as spurious "evolution."

   After both fixes, one run (40 synthetic sessions, patience 20) got very close: SOFT diff 0.018,
   MEDIUM 0.006, HARD 0.010 (tolerance 0.015), evolution correlation 0.93-0.997. A follow-up run with
   a larger synthetic corpus (80 sessions) and more patience regressed (diffs 0.027-0.034,
   negative-control slope ~0.017 against a 0.01 limit) — non-monotonic with more data/epochs, which
   is itself a signal that something is still not fully identified rather than just under-trained.
   **Not yet resolved** — paused here when the user interrupted to update the prompt files.

**Phase 4 (baselines)**: `src/models/baselines.py` written (linear, degree-2/3 polynomial, LightGBM +
SHAP dependence helper) — not yet run against real or synthetic data.

**Phase 6 (race validation)**: `src/validate_race.py` written (fuel-correct race pace from a known
start-fuel/race-distance, per-stint empirical slope, predicted-vs-observed comparison) — not yet run.

**Phase 7 (strategy)**: `src/strategy.py` written and smoke-tested (DP over stint-length splits for
1-3 stops, one-vs-two-stop crossover pit loss) — figures not started.

**Session paused**: user sent an interrupt, then updated the prompt files —
`CLAUDE_CODE_PROMPT.md` → renamed `PROMPT.md` with the changes summarised in `PLAN.md`, and added
`CLAUDE.md` (workflow rules: one phase at a time, plan mode before multi-file phases, stop and
report at each acceptance gate). Stale download process (against the old, narrower Phase 1 scope)
killed at 68 sessions cached. `PLAN.md` and this file created per `CLAUDE.md`'s requirement.

---

## 2026-09-12 — Phase 1 rework (plan-mode approved)

**Open question resolved**: could not confirm the exact mid-2023 Pirelli construction-change
race from available sources (contradictory reporting — one source's timeline was internally
impossible). Asked the user; decided to simplify to a 2-way split (`GE1`=2022, `GE2`=2023-2025,
`R26`=2026) rather than assert an unconfirmed mid-season boundary.

**Verified against the installed `fastf1` (3.8.3) before writing code, not assumed**:
- `fastf1.get_event_schedule(year, include_testing=True)` exposes each event's own
  `Session1..Session5` name columns (e.g. `"Practice 1"`, `"Sprint Qualifying"`, `"Sprint"`) —
  confirmed 2023 uses `EventFormat="sprint_shootout"`/`"Sprint Shootout"`, 2024+ uses
  `"sprint_qualifying"`/`"Sprint Qualifying"` for the same concept.
- `fastf1.get_session(year, round, <full session name>)` accepts the schedule's own session-name
  strings directly — confirmed on both a conventional and a sprint weekend.
- Pre-season testing is **not** reachable via `get_session(year, 0, ...)` (raises `ValueError:
  Cannot get testing event by round number!`) — must use
  `fastf1.get_testing_session(year, test_number, session_number)`, which has no enumeration API, so
  the grid must be probed. Confirmed a nonexistent (test_number, session_number) combination raises
  `ValueError` cleanly and immediately (not a slow timeout), so probing 3×3 per year is cheap even
  when most of the grid doesn't exist.
- 2026 schedule **is** available (25 rows incl. testing) — confirmed 2 pre-season tests × 3 sessions
  each exist for 2026 (test 3 correctly comes back not-applicable).

**`src/data/download.py` rewritten**:
- Sessions enumerated per-event from `Session1..Session5` instead of a fixed list (correctly
  handles sprint weekends — no more assuming FP2/FP3 exist where they don't, or missing
  Sprint/SQ where they do).
- `messages=True` (was `False`) in every `session.load(...)` call.
- Two-layer retry: `ValueError` (deterministic "doesn't exist") never retried; other transient
  errors get 3 short exponential backoffs (1s/2s/4s); a rate-limit error escalates to the existing
  300s/20-retry wait. Composed, not replacing the rate-limit fix from the first pass.
- `era` and `session_type` tagged into every session's `meta.json`.
- Testing-session probe grid (`testing_probe.test_numbers × session_numbers`) added as a second
  pass after the regular per-event loop.
- Summary report extended: sessions per era, laps per era, testing-sessions-present-per-year.

**Smoke-tested before the full run** (per the plan's verification step): a 2-event slice
(Bahrain 2024 conventional + Chinese GP 2024 sprint) downloaded correctly with the right
era/session_type tags on every session including `SQ`/`SPRINT`; the 2026 testing probe found
exactly 2 real tests (6 sessions) and correctly marked the 3rd test's slots `not_applicable`;
re-running the same testing probe against itself came back 100% `skipped_cached` (0 re-downloads),
confirming resumability holds under the new code. (These smoke-test sessions are genuine, useful
downloads, not thrown away — they count toward the corpus.)

**Relaunched the full background download** (2022-2026 regular sessions + pre-season testing,
all in `nohup`, same pattern as before) — running healthily as of this entry.

**Phase 1 acceptance, rescoped** (Section 3.5's "clean laps" and stint-count-table columns need
Phase 2's `clean.py`, which doesn't exist yet — deferred there, not silently dropped, as flagged in
the plan):
- ✅ Resumability: confirmed zero re-downloads on rerun.
- ⏳ `data/raw/` > 80% of expected sessions: not yet — download just relaunched, will report once
  substantially further along (full corpus is 2022-2026, hours unattended, as the prompt expects).
- ⏳ Testing sessions present for every season: confirmed structurally correct for 2026 (2/3 test
  slots real); full per-season confirmation pending the complete run.
- ⏳ Total clean laps > 80,000: blocked on Phase 2 (`clean.py` not built).
- ⏳ Per-(circuit, compound, era) stint-count table: blocked on Phase 2.

**Not done in this entry, deliberately** (per `CLAUDE.md`'s one-phase-at-a-time rule): the era
embedding change to `TyreHead`'s conditioning vector (Phase 5) and the still-open Phase 3 recovery
test gap (last recorded state: slope diffs 0.027-0.034 s/lap against a 0.015 tolerance, negative
control ~0.017 against a 0.01 limit) — both are separate phases from this one and will be picked up
next, each through its own plan/report cycle.

---

## 2026-09-12 — Phase 1 throughput: parallelised across two networks

User asked why a ~2GB download was taking hours, and to find a faster way. Diagnosed: the
bottleneck is never bandwidth -- it's FastF1's 500-calls-per-hour cap, and each session costs
~7-9 calls, so the full ~540-session corpus needs ~4,000-4,500 calls regardless of payload size --
an ~8-9 hour floor even at gigabit speed. Confirmed directly from the log: 8 rate-limit hits in
that run alone, each forcing a 300s sleep (40 minutes spent purely waiting, not transferring).

Since the limit is per-network (not a global shared budget, near-certainly, given it's enforced by
a community-run mirror rather than a per-account API key), split the corpus in half across two
independent downloaders with **zero overlapping work**:
- **Local machine**: `config/default.yaml` `data.seasons` narrowed to `[2022, 2023]` (was
  `[2022, 2023, 2024, 2025, 2026]`). Restart with the same resumable `download.py` -- the existing
  2024/2026 sessions already on disk from before the split stay put as a bonus, just outside this
  machine's ongoing scope now.
- **Colab**: `notebooks/colab_download.ipynb`'s `SEASONS` changed to `[2024, 2025, 2026]`.

Both relaunched; local confirmed healthy post-restart. Expected effect: roughly halves wall-clock
time to a complete corpus, assuming the per-network assumption holds (flagged as an assumption, not
a certainty, when explaining this to the user). To return to a single-machine setup later, restore
`seasons: [2022, 2023, 2024, 2025, 2026]` in whichever config is doing the work and merge the other
side's `data/raw/` folder in first (plain directory merge, no collision risk -- every session lives
in its own uniquely-named folder).

---

## 2026-09-13 — Phase 3 recovery test: two more real bugs found and fixed, one open

Resumed the actual blocker per user's request ("resume proposed work, bottleneck if it has one").
Reverted the synthetic config to the last known-good point (`n_sessions: 40`,
`early_stop_patience: 20` -- the 80-session/patience-30 run had regressed, non-monotonically,
which was itself a clue something was still unstable rather than just under-trained).

**Test-harness bug (not a model bug)**: `tests/test_recovery.py` hardcoded `vocabs.circuit.mapping[0]`
in three places, assuming raw circuit id 0 always survives the train/val split. It doesn't always
(seed-dependent) -- the negative-control test hit a bare `KeyError`. Fixed all three call sites to
use `next(iter(vocabs.circuit.mapping.values()))` / the first few actually-present keys instead.

**Real bug found via direct curve inspection**: with the harness bug fixed, slope recovery was
excellent (diffs 0.0004-0.0088, all compounds) but cliff detection still failed for MEDIUM
(detected age 15, true 22). Printed the fitted curve's raw per-lap increments directly rather than
just the aggregate slope metric, and found the actual shape: wildly jagged, swinging between ~0.0002
and ~0.5+ s/lap from one age to the next, when the true curve is a flat 0.055 until the cliff. The
*average* over the slope-check's window happened to land close to true, which is exactly why the
slope test passed while the curve itself was nonsense -- and why cliff detection, which needs the
real shape, failed. Root cause: `TyreHead`'s conditioning vector (Section 1.3's own specified
architecture) carries no age information at all -- the MLP emits all `max_age` increments at once
from `(compound, circuit, track_temp)` alone, so nothing stops the network from fitting each age's
output column almost independently to noise. **Fixed** by adding a second-difference smoothness
penalty on the raw increments (`TyreHead.increments_for`, `NAM.tyre_smoothness_penalty`,
`tyre.smoothness_weight` in config, wired into `train.py`'s loss) -- a natural prior for a
physically slowly-varying wear curve, in the same spirit as the traffic and evolution penalties
already in the codebase. Result: **cliff detection now passes cleanly for all three compounds**
(detected within 2 laps of truth, tolerance is 3).

**With the smoothness fix**, slope recovery regressed slightly (HARD diff 0.0179, tolerance 0.015)
and the negative control still failed (~0.02-0.027 s/lap of phantom degradation where truth is
exactly zero, limit 0.01). Diagnosed by comparing the fitted `EvoHead` against ground truth directly
(not just correlation): recovered evolution *amplitude* is consistently ~2-3x too large versus truth
(e.g. -1.48 vs -0.56) even though the *shape* correlation is excellent (0.93-0.997) -- confirming an
over-strong evolution estimate is what forces the tyre curve to compensate with extra slope,
including phantom slope with no true tyre effect at all.

**Two counter-measures tried on the amplitude bias, both made things WORSE, not better** (evidence
against continuing down that path, not just an untried knob): an L2 shrinkage penalty toward zero
(`evo_shrinkage_weight` swept 0.1-1.0 against the *low-capacity* `EvoHead` this time, not the old
free-MLP version it was previously and unsuccessfully tried against -- same negative result: every
metric got worse, because the true amplitude is itself nonzero, so shrinking toward zero overshoots
past it); and a hard sigmoid ceiling on the amplitude (fixed the bias directly but broke cliff
detection for HARD outright, detected age 10 vs true 30). Both reverted; neither is in the code.

**A separate, more fundamental bug found while investigating**: `torch.manual_seed(cfg.seed)` is
called inside `train.fit()`, but `NAM(nam_cfg)` -- which initialises every head's weights from the
GLOBAL torch RNG state -- is constructed *before* `fit()` is ever called. So the seed was never
actually controlling model initialisation, in any script, all session. Confirmed directly: two
identical CPU runs (same seed, same data, same config) produced different results end to end (a
detected cliff of 31 vs 33 laps). This retroactively casts doubt on some of the earlier "X made it
worse" conclusions in this log, since those comparisons were partly confounded by uncontrolled
random init -- though the *shape* of the findings (smoothness fixes jaggedness; shrinkage and hard
caps both backfire on amplitude) held up again after the fix, so the conclusions above stand.
**Fixed** in `tests/test_recovery.py::_fit_on` by moving `torch.manual_seed(seed)` before model
construction. Any future real-data training entrypoint needs the same ordering.

**With both fixes** (smoothness penalty + correct seeding), a smoothness-weight re-sweep
(now on a reproducible basis) found `smoothness_weight=4.0` clearly better than the earlier `1.0`:
cliff detection passes for **all three compounds** (within 2 laps of truth, tolerance 3), and 2 of 3
slopes pass cleanly (MEDIUM, HARD). Locked in as the new config default.

**Still open, honestly**: SOFT's slope is consistently ~0.02 s/lap too steep (tolerance 0.015)
across every smoothness weight tested (1.0-8.0), and the negative control still fails, though
improved (~0.018-0.02 vs the ~0.01 limit, down from ~0.02-0.027 before the seeding fix). Both track
the same residual: SOFT has the steepest true slope and earliest true cliff, and its excess closely
matches the negative control's phantom-slope magnitude, both pointing back to the same
not-yet-resolved evolution-amplitude bias that smoothness (which only fixes jaggedness, not a
constant offset) doesn't touch. **2 of 4 acceptance criteria pass outright** (cliff detection,
evolution-shape correlation); slope recovery is very close (2 of 3 compounds clear tolerance); the
negative control is the furthest from passing. Per the standing rule, thresholds have not been
relaxed to paper over this -- reporting it as open rather than declaring Phase 3 complete.

---

## 2026-09-13 — Semi-synthetic validation on REAL covariates: reveals a bigger gap

User's request: stop both downloads, use whatever real data is on disk now (207 sessions, 2022 +
partial 2023), and build a stronger validation -- train on the REAL, as-occurred confounders (real
stint lengths/pit strategy, real weather, real traffic, real driver/circuit identities), inject only
a KNOWN synthetic tyre-wear effect on top (reality has no wear label to check against), and see how
the same architecture does against real-world complexity instead of the simplified simulator.

**Built** (plan-mode approved):
- `src/data/clean.py` -- real-session filter cascade per `PROMPT.md` Section 4 (drop: no lap time,
  not green flag, pit in/out, deleted, unknown/wet compound, `TyreLife` null/>45, traffic <2s gap,
  per-stint outliers, stints <4 laps), plus real feature extraction (`tyre_age`, `lap_in_stint`
  ported from the sibling TyreMind project's gap-aware fix, `session_progress`, weather joined by
  nearest timestamp, `gap_ahead_s` from real lap-start-time ordering). Run on all 40 real 2022-2023
  race sessions on disk: 10 dropped whole for rain, **13,388 real laps retained from 30 sessions**
  (30.7% retention -- the traffic-gap filter alone removes ~27% of all laps, by far the largest cut,
  which is the filter list working as specified, not a bug).
- `src/sim/semi_synthetic.py` -- injects the SAME known per-compound wear curves used in the
  pure-simulator test (for direct comparability) plus synthetic fuel/evolution/condition/traffic
  effect functions onto the real `tyre_age`/`lap_in_stint`/`session_progress`/weather/`gap_ahead_s`
  values, using the REAL per-circuit median lap time (not an invented constant) as circuit base pace.
- `tests/test_recovery_real_covariates.py` -- the same four checks as `test_recovery.py`, run
  against this dataset. Additive, not a replacement for the pure-simulator gate.

**Result: all four checks fail, and by a wider margin than the pure-simulator version.**

| Check | Pure simulator (last measured) | Real covariates |
|---|---|---|
| Slope (max diff, tol 0.015) | 0.021 | **0.102** (all 3 compounds ~2-3x true value) |
| Cliff detection | pass, all 3 within 2 laps | SOFT not detected at all |
| Evolution correlation (min 0.90) | 0.93-0.997 | 0.88-0.98 (still mostly OK) |
| Negative control (limit 0.01) | 0.019 | **0.105-0.110** (~10x the limit) |

Validation loss during training was also far higher than on synthetic data (best val Huber ~1.69 vs
~0.06-0.14), and early stopping fired after only 17 epochs -- the model is not fitting the real data
well at all, not just misattributing one effect to another. Most likely cause, not yet confirmed:
13,388 rows spread across 22 real circuits x 3 compounds x many drivers leaves very few samples per
(circuit, compound) cell -- `PROMPT.md` Section 3.5 itself warns that cells under 15 stints need
widened uncertainty, and with only 30 sessions total this is likely happening for most cells. This
is exactly the kind of result the user's proposed test exists to surface: the architecture's
near-pass on simplified synthetic data was optimistic relative to real-world difficulty, and this
would not have been visible without building this exact comparison.

**Not yet root-caused or fixed** -- this entry reports the finding, per the standing report-before-
continuing discipline, rather than immediately chasing another fix.

---

## 2026-09-13 — Root-caused via user's diagnostic methodology: a real, mechanical bug found and partially fixed

User pushed back hard, correctly, on the "sparse data" hypothesis above: sparsity produces variance
(curves scattered around truth, inconsistent across compounds), not the observed signature (all
three compounds biased the SAME direction, plus a systematically positive negative control). Laid
out a precise diagnostic order: matched-sample-size subsample first (cheap, discriminates cleanly
between "data quantity" and "real covariate structure" as the cause), then an oracle ablation and a
check of how often real `TyreLife` starts above zero if sample size doesn't fully explain it.

**Step 1 -- matched-sample-size synthetic** (13,388 rows, 22 circuits, 30 sessions, same config, to
match the real corpus exactly): confirmed val loss is genuinely monotone (not noisy early stopping
from a thin 4-event validation split, which was my working guess). Result was a clean split verdict:
- Negative control: matched-size synthetic reproduces real's magnitude almost exactly (0.10-0.125 vs
  real's 0.105-0.110, both stopping ~epoch 16-17) -- **sample size alone fully explains this one.**
- Main slope error: matched-size synthetic only reached 0.017-0.045, while real covariates were at
  0.07-0.10 -- **sample size does NOT explain this one.** Something specific to real covariate
  structure was adding a second, separate error on top.

**Step 2 -- found the real-covariate-specific mechanism, per the user's steer to check how `TyreLife`
starts**: measured directly on the real corpus -- **99.8% of real stints do not survive filtering
with their true first lap; 55% are missing their first 3+ laps (max observed: 39 laps truncated).**
Root cause: this project's own traffic filter (Section 4 -- drop any lap with a <2s gap to the car
ahead) disproportionately removes exactly the laps right after a pit stop, when cars bunch up. The
existing `lap_in_stint` calculation (ported from the sibling TyreMind project, whose own pipeline
never drops laps for traffic at all, only uses it as a continuous feature) computed its zero-point
from `TyreLife.min()` of whatever survived filtering -- valid where TyreMind needed it (a safety car
dropping laps from the MIDDLE of a run), silently wrong here, where the HEAD of the run is what's
usually missing. `tyre_age` itself (raw `TyreLife`) stays correct throughout; `fuel_kg_rel` (built
from the corrupted `lap_in_stint`) does not -- for the majority of real stints, it was computed
relative to a fabricated "start" many laps late.

**Fixed**: verified directly against RAW (pre-filter) data that 100% of fresh-tyre stints have
`TyreLife == 1` on their true first lap (38/38 checked, zero exceptions) -- so `lap_in_stint =
TyreLife - 1` is correct and, critically, immune to filtering by construction, unlike the group-min
approach. Replaced in `src/data/clean.py::_lap_in_stint`.

**Effect of the fix -- real, but not a full resolution**: re-ran the real-covariate suite.

| | Before this fix | After |
|---|---|---|
| SOFT slope diff (tol 0.015) | 0.102 (over) | 0.014 (essentially passes) |
| MEDIUM slope diff | 0.077 (over) | 0.023 (now UNDER, direction flipped) |
| HARD slope diff | 0.069 (over) | 0.025 (now UNDER, direction flipped) |
| Best epoch (main fit) | 17 | **169** -- trains ~10x longer before plateauing |
| Negative control (limit 0.01) | 0.105-0.110 | 0.126-0.129 -- slightly worse |

The bias flipping direction for two of three compounds, and training running an order of magnitude
longer before early stopping, are both strong independent confirmations that this was a real,
substantial mechanical error, not a coincidence -- but it over-corrected rather than fully resolving,
and the negative control moved slightly the wrong way. **Something else is still contributing.**
User's next diagnostic step (oracle ablation -- fix the true evolution curve as a known offset and
see whether the negative control drops toward zero, isolating whether the remaining leak is in
`EvoHead`) has not yet been run as of this entry.

---

## 2026-09-13 — L1 penalty on tyre increments: helped, then shown to be an overfit choice

**Scrubbed-tyre hypothesis checked and eliminated**: user suspected the `lap_in_stint = TyreLife - 1`
fix might mishandle scrubbed (non-fresh) tyres, disproportionately biasing Medium/Hard. Measured
directly: Soft actually has the HIGHEST scrubbed fraction (53.6%) of the three, while Medium (13.9%)
and Hard (16.9%) have the lowest -- the opposite of what the hypothesis predicts (Soft is the one
passing). Not the explanation; noted and moved on, per the user's own instruction to check cheaply
and not chase it further if it didn't hold up.

**Added `NAM.tyre_l1_penalty`** (`tyre.l1_weight` in config, wired into `train.py`) -- an L1 penalty
on the tyre head's raw increments, the direct countermeasure for a monotone (softplus-based)
estimator's one-sided bias: every increment is >= 0 by construction, so noise can only ever
accumulate upward, never cancel.

**Swept on matched-size synthetic** (13,388 rows, 22 circuits, same harness as the sample-size
experiment). Non-monotonic, with a clear failure mode past a threshold:

| L1 | Max slope error (tol 0.015) | Negative control (informal target 0.01) |
|---|---:|---:|
| 0 | 0.050 | 0.049 |
| 0.1 | 0.016 | 0.035 |
| **0.3** | **0.0058** | 0.023 |
| 1.0 | 0.032 (undershoots) | 0.095 (worse) |
| 3.0-10.0 | 0.44-0.45 (broken) | 0.022-0.066 |

At L1 >= 3, slope error explodes in the positive direction even as the negative control partially
improves -- L1's classic sparsity behaviour: instead of shrinking every lap's increment a little, it
concentrates wear into a few large spikes. Since the slope metric averages the first 8 laps, an early
spike inflates it badly even though the overall curve is nonsense.

**L1=0.3 looked like a real fix -- slope passed cleanly, negative control markedly improved. It was
not.** Per the user's insistence on testing generalisation before trusting a tuned value: generated a
FRESH synthetic realisation (different true slopes {0.075, 0.060, 0.040}, different cliffs
{18, 25, 33}, different seed, matched sample size) and ran L1=0.3 on it UNCHANGED. Result:

| | Tuned realisation | **Fresh realisation** |
|---|---:|---:|
| Max slope error | 0.0058 | **0.0264** (fails tolerance) |
| Cliff detection | 3/3 passed | **2/3 not detected at all** (Medium, Hard) |
| Negative control | 0.023 | **0.094** (back to baseline) |

Confirms the user's specific prediction: L1 suppresses exactly the large increments a cliff consists
of, so a value chosen to look good on one realisation actively breaks cliff detection on another.
This was fitting the validation metric, not a real fix. **Reverted to `l1_weight: 0.0`** in
`config/default.yaml`, with the finding documented inline so it isn't silently re-tried later without
the same generalisation check.

**Reframing the negative control** (per the user): the 0.01 acceptance number was never derived from
anything, just a plan-time placeholder. The scientifically honest framing is to MEASURE the untuned
estimator's positive bias floor under real sample-size constraints (not chase it to zero by tuning),
then state which true degradation rates are distinguishable from it.

**First 8-seed measurement (before the fix below) surfaced something bigger than a noise floor**:
results were bimodal, not a tight spread -- 6 of 8 seeds clustered at 0.025-0.035, but 2 of 8 (seeds
4 and 7) blew up to 0.18-0.52, an order of magnitude worse. Reporting `mean +/- std` (0.114 +/- 0.159)
over this would have hidden the real finding: this is a convergence failure ~25% of the time, not a
stable noise level.

**Root-caused, per the user's diagnosis-before-fix approach**: on the two failed seeds, `best_epoch`
was 1 and 6 (training essentially never started) and `EntryHead`'s embedding had barely moved from
its zero init (std 0.005-0.018 vs 0.25-0.37 on healthy seeds) -- but evolution-curve correlation was
FINE on the failed seeds (0.96-0.99, if anything better than the healthy ones), ruling out the user's
first-guess mechanism (`EvoHead` winning a competition against `EntryHead`). Root cause found instead
in `TyreHead`'s initialisation: `softplus(0) = ln(2) ~= 0.69`, not 0, so with the MLP's ordinary
small-random init, every one of the 45 increments starts around 0.69s regardless of luck -- a ~31s
cumulative curve baked in before any training happens at all. On an unlucky draw, this substantial
wrong prior is apparently enough to send the optimiser into a state that 20+ epochs of patience never
recovers from.

**Fixed** with a one-line, well-targeted change: `nn.init.constant_(self.mlp.net[-1].bias, -5.0)` in
`TyreHead.__init__` (`src/models/heads.py`), so `softplus(-5) ~= 0.0067` and the curve starts flat,
pushed up by evidence rather than pulled down from an already-large wrong value. A separate warmup
mechanism (freeze `TyreHead` for N epochs, `tyre.tyre_warmup_epochs` in config) was also implemented
per the user's suggestion, wired in and tested, but proved unnecessary -- left in the code, off by
default, in case real data (noisier than synthetic) needs it later.

**Result: re-ran the exact same 8 seeds. Zero catastrophic failures.**

| | Before the fix | After |
|---|---|---|
| Seeds converging normally | 6 / 8 | **8 / 8** |
| Failed-seed best_epoch | 1, 6 | (no failures) |
| Detection floor | 0.114 +/- 0.159 (bimodal, meaningless as a single number) | **0.022 +/- 0.013** (tight, max 0.043) |

**This is now the honest number to report**, per the user's framing: at n~13,000 laps, the estimator
has a positive bias floor of ~0.022 +/- 0.013 s/lap (max observed 0.043 across 8 seeds). Consequence,
stated plainly rather than left implicit: Hard's true slope (0.035 s/lap) sits inside this floor --
not reliably distinguishable from noise at this sample size. Medium (0.055) is at the edge. Soft
(0.09) is comfortably above it, ~3-4x the floor. **Soft and Medium's recovered curves are credible at
this data volume; Hard's is marginal and should be reported as such, not asserted with the same
confidence.**

**Verified this isn't specific to the matched-size harness**: re-ran the original, larger-scale
`tests/test_recovery.py` (n_sessions=40, the project's standing Phase 3 gate) with the same fix.
Still technically fails the file's strict thresholds (slope 0.0156 vs 0.015, cliff 5 vs 3 laps for
Hard, negative control 0.031 vs 0.01) -- but critically, `best_epoch=58`, not 1: no collapse, just
the same honest floor showing up at a different sample size, consistent with the 8-seed measurement
above (0.031 is within its observed max of 0.043). The strict thresholds have not been loosened
(per the standing rule); the finding is reported instead of hidden behind them.

**Per the user's explicit timebox**: stopping tuning here. This was the last open Phase 3 item;
moving to real data / Phase 6 / figures next.

---

## 2026-09-13 — `FINDINGS.md` written; figures started per user's priority order (figures before Phase 6)

Wrote `FINDINGS.md` (project root) capturing the three findings above in presentation-ready form,
including the methodology note on pattern-vs-attribution (correctly inferring a shared upstream
cause from all-compounds-failing-identically, incorrectly attributing it to `EvoHead` before the
real cause -- the tyre head's own init -- was found).

Loaded the `dataviz` skill before writing any chart code (validated categorical palette, fixed
slot assignment, one-axis rule, etc.) rather than picking colors by eye.

**Figure 5 (additive decomposition, the user's stated highest priority)** --
`experiments/figure_decomposition.py`: fits on real covariates, picks the longest real stint on
disk (40 laps, HARD), and stacks the five effect heads (fuel, tyre, evolution, conditions, traffic)
positive-up/negative-down around the circuit+driver baseline, with actual and predicted lap time
overlaid. Rendered and visually verified (no label collisions/overflow after two layout passes --
first attempt had the subtitle overlapping the title, fixed by moving both to figure-level text and
reserving top margin explicitly rather than trusting `tight_layout` alone). Saved to
`outputs/figure5_additive_decomposition.png`.

**Detection-floor chart** -- `experiments/figure_detection_floor.py`: the three true wear rates as
bars against the measured floor band (mean/std/max from the 8-seed measurement). Sequential (one
hue) encoding since this is a magnitude comparison against a threshold, not a categorical
comparison. Saved to `outputs/figure_detection_floor.png`.

**Figure 1 (degradation curves with uncertainty bands)** -- `experiments/figure_degradation_curves.py`:
8-seed ensemble at matched real-world scale (13,388 rows, 22 circuits, the same harness used
throughout today), median + 12.5-87.5 percentile band per compound, true curves overlaid dashed.
Running as of this entry (8 fresh fits, ~10-12 min) -- result and visual check pending.

---

## 2026-09-13 (cont.) -- Figure 5 critique, resolved

User caught two real reading errors on the first Figure 5 (HARD compound): (1) suspected L1=0.3
was suppressing the cliff -- checked config, L1 is 0.0, inactive, not the cause; (2) read a
correctly-detected cliff (bend in the cumulative curve at lap 31 vs. true 30) as a missed one --
on a cumulative chart a cliff is a bend, not a break. Both corrected before acting further.

Ran a smoothness-weight sweep (4.0/1.0/0.3/0.0) on the exact stint Figure 5 used anyway, since the
underlying mechanism (a penalty discouraging large increments) was worth checking independent of
which specific penalty was suspected. Result: not the cause for Hard (cliff detected within 1 lap
of true at every setting, closest at the current weight 4.0) or Medium (stable ~23 vs true 22 at
every setting). Soft is a real, separate problem: cliff detected at 23-25 vs. true 14 regardless of
smoothness weight.

**Coverage check** (`experiments/figure_tyre_age_coverage.py`, real 2022-2023 corpus): tested
whether sparse real low-tyre-age data explains Soft's failure. It does not -- Soft stints reach
their own cliff age (14) more reliably (77% of 183 stints) than Medium/Hard reach theirs (33%/37%),
and per-age lap density near the cliff is comparable across all three compounds. Hypothesis ruled
out, not confirmed -- written up in FINDINGS.md #4 as an honest "cause not yet identified" rather
than asserting the checked-and-eliminated explanation.

**Figure 5 regenerated with MEDIUM** (not Hard: true rate sits inside the measured detection floor;
not Soft: cliff detection currently broken there) -- `experiments/figure_decomposition.py` now
selects the longest MEDIUM stint specifically rather than the longest stint overall. New figure
shows the tyre-wear band visibly steepening around lap 22-25, matching Medium's correctly-detected
cliff (23 vs true 22). Two cosmetic fixes applied at the same time: legend moved from an in-axes
upper-left block (covering data, 1 column x 8 rows) to a 4-column strip below the plot; explicit
`ax.margins(x=0.02, y=0.12)` added since the default 5% autoscale margin wasn't enough headroom for
the stacked bands' largest extent at the oldest tyre age, which was clipping against the axes frame.

Verified this stint is genuinely the driver's first stint of the race (no prior pit stop), so the
fuel-reset concern raised alongside the original critique doesn't apply to this specific figure --
though the general risk (fuel resets every stint by design) remains real and already documented.

FINDINGS.md updated with finding #4 (Soft cliff failure, coverage hypothesis ruled out) and the
real-data validation table's cliff-detection row corrected to name Soft specifically rather than
"one compound."

**Next**: Phase 6 (race validation), per the standing priority order -- coverage check and Figure 5
were the two items ahead of it.

---

## 2026-09-13 (cont.) -- Phase 6: first real-data-only validation

Built `experiments/phase6_race_validation.py`: fits NAM on real 2022-2023 practice sessions
(FP1/FP2/FP3, 12,956 laps, 90 event-sessions, `src/data/clean.py` + a small schema-rename adapter,
no synthetic injection), validates per-event against that same event's real race slope
(`src/validate_race.py`, already built earlier, run for the first time here). Added `lap_number` to
`clean.py`'s output columns (needed for `session_lap`/`race_distance` in the fuel-correction step) --
additive, non-breaking.

Result: best-fit slope of predicted vs observed wear rate = 0.23 (79 points, 30 races), 0.20 with
the two most extreme events removed -- confirms the compression bias (findings #5) a third,
fully-independent way, on real data never shown to the model in training. Matches the user's stated
prediction before running it.

Two sub-patterns separated out rather than folded into the headline number: (a) 29% of points have
near-zero predicted wear despite real observed wear -- real per-cell practice data sparsity, the
real-data instance of finding #3's coverage caveat; (b) two events (2022 R4, 2023 R16) over-predicted
across all three compounds simultaneously -- confirmed as track_temp extrapolation (2022 R4 raced at
16.6C after ~31C practice, 14C outside the tyre head's trained range), not a wear-model failure, and
confirmed not responsible for the headline slope (removing them: 0.232 -> 0.202).

FINDINGS.md updated with finding #6 covering all of the above.

**Status**: Phase 6 complete for the first time with real data. Remaining Phase 7 work: fuel-
sensitivity table, baseline comparison figure. No trained checkpoint has ever been persisted to disk
-- every fit in every experiment script is ephemeral, retrained each run.

---

## 2026-09-13 (cont.) -- Calibration attempted, and it's more nuanced than "fixed" or "didn't work"

User pushed on the Phase 6 slope=0.23 headline: check correlation separately from slope (weak
correlation makes the slope number less meaningful), then attempt an actual calibration factor with
a held-out check, same discipline as every other tuned value in this project.

Checked correlation first: pooled Pearson r=0.11 (p=0.33, not significant), per-compound ~0. Found
that slope=0.23 is *exactly* r * sd(predicted)/sd(observed) = 0.11 * 2.10 -- i.e. largely a
mechanical consequence of weak correlation + predicted values being 2.1x noisier than observed, not
independent evidence of a clean "1/5th magnitude" bias. This changes how the number should be
reported.

Attempted a ratio-of-means calibration factor (robust to the regression-attenuation problem a
least-squares scale factor inherits). On all 79 points: 1.22 (barely any bias). Excluding the two
already-diagnosed track-temp-extrapolation outlier events (2022 R4, 2023 R16): 2.04. Checked
stability via 200 random 60/40 event splits: WITH the outliers included, the needed correction on
the held-out 40% swings 0.6x-3.8x depending on the split -- unusable, and the instability traces
directly to whether those 2 high-leverage events land in the fit half or the held-out half. WITHOUT
them (28 events), the correction needed is >1 in 99% of splits, bootstrap 95% CI on the full-sample
estimate = [1.25, 3.87] (excludes 1 -- real, but wide given only 28 usable events).

Applied the 2.04x factor: local best-fit slope (excl. outliers) moves from 0.20 to 0.41 -- real
improvement, not a full fix, since scaling a mean bias can't repair weak point-level correlation.

New scripts: `experiments/phase6_calibration_check.py` (stability figure,
`outputs/figure_calibration_instability.png`), `experiments/figure_calibration_corrected.py`
(raw-vs-corrected scatter, `outputs/figure6b_calibration_corrected.png`).

FINDINGS.md #6 extended with the full correction -- this is a good example of the project's own
stated discipline catching an overstated number before it went in the deck: "0.23" read as "5x
compression," properly decomposed it's "weak correlation + noisy predictions," and the real,
defensible number is "~2x average under-prediction, direction confirmed, magnitude honestly
uncertain at current data volume."

---

## 2026-09-13 (cont.) -- Phase 6 reframed: characterization, not validation; stint-level check

User caught a bigger error than the calibration number: reading a weak-correlation regression slope
(r=0.11, p=0.33) as evidence of event-level predictive validity. Corrected framing: "Phase 6
confirms the compression bias a third way" survives; "Phase 6 validates the model" does not.

Ran the suggested cheap check: stint-level correlation instead of event-level aggregation
(`experiments/phase6_stint_level_check.py`, 1,044 individual real stints vs. 79 aggregated
event-compound points). Result splits cleanly by compound: Hard r=0.207 (p<0.0001, n=480) -- real
signal event-aggregation was hiding. Medium r=-0.042 (n=381), Soft r=-0.038 (n=183) -- flat even
with an order of magnitude more data, ruling out sample size as their explanation and pointing back
to the already-diagnosed structural issues (Medium's post-cliff undershoot, Soft's missed cliff).

Wrote the r-decomposition catch into FINDINGS.md as a standing methodology note (parallel to finding
#2's pattern-vs-attribution lesson): never report a fitted slope as a "bias factor" without its
correlation alongside it, since slope = r * sd_y/sd_x can be driven entirely by a variance mismatch
even at near-zero correlation.

FINDINGS.md #6 substantially restructured: headline reframed from "confirmed a third way" (unqualified)
to explicit two-claims split (compression bias survives; event-level predictive validity does not),
stint-level table added, "one root cause" framing corrected to "two root causes" (data-volume
ceiling for Hard; distinct structural bugs for Medium/Soft) since collapsing them would understate
the second.

**Next**: baseline comparison (LightGBM vs NAM -- MAE vs slope-recovery tradeoff), per priority order.
Now carries more narrative weight since Phase 6's predictive-validation claim was walked back --
it's the remaining evidence the structural constraints buy something a plain accuracy-optimizing
model wouldn't.

---

## 2026-09-13 (cont.) -- Baseline comparison run for the first time

Ran `src/models/baselines.py` for the first time (built earlier, never exercised). Found and fixed a
real bug before running: `compound` was missing from `CONTINUOUS_COLUMNS`/`CATEGORICAL_COLUMNS`
entirely -- no baseline could have distinguished Soft/Medium/Hard at all. Added it to
`CATEGORICAL_COLUMNS`, threaded through `_design_matrix`, `LinearBaseline.implied_tyre_curve`, and
`lightgbm_shap_tyre_curve` (all now take/use a `compound` argument). Also fixed a LightGBM dtype
error (raw string columns rejected -- added `prepare_lgb_features`/`category_map` for consistent
train/val/grid categorical encoding) and a macOS OpenMP mutex crash (`OMP_NUM_THREADS=1
KMP_DUPLICATE_LIB_OK=TRUE` needed to run lightgbm in this sandboxed environment).

New script: `experiments/baseline_comparison.py`, same matched-scale synthetic data as Figure 1
(13,388 rows, 22 circuits). Result: LightGBM wins MAE by ~30% (2.71s vs NAM 3.88s vs Linear 4.31s)
but its recovered per-compound curve reports the IDENTICAL slope (0.0107) for all three compounds --
compound is in its feature set and it still doesn't differentiate them in the early-age window --
and detects a cliff at lap 11 for all three vs true 15/23/31. Visually plateaus hard after ~lap 35
for every compound (trees can't extrapolate past their training tyre-age range). NAM recovers all
three slopes/cliffs within a few percent / a few laps. Clean, exactly-as-predicted demonstration of
the module's own thesis: accuracy and causal attribution are different objectives.

FINDINGS.md #7 added. Figure: `outputs/figure7_baseline_comparison.png`.

**Next**: fuel sensitivity table (fast, defensive rather than affirmative per user's framing), then
deck assembly per the agreed headline/sequencing.

---

## 2026-09-13 (cont.) -- Three more review rounds: degenerate-cell check, Figure 7 correction, Figure 6b axes

**Degenerate-cell hypothesis (the ~hour check).** User spotted the dense row of near-zero
predictions in the Figure 6 scatter and proposed it explains r=0.11: split degenerate (<0.01) vs
not, check if non-degenerate correlation is meaningfully higher, check what predicts degeneracy
(expected: stint count). Tested properly:
- Event-level split: non-degenerate correlation is WORSE (0.11 -> 0.014 pooled), not better.
- Real race-stint count: nearly identical between degenerate/non-degenerate (13.8 vs 12.9) -- no.
- Real practice-lap density per (circuit,compound) cell: not predictive either (r=0.10, p=0.39),
  if anything backwards (degenerate cells average MORE practice laps).
- Re-checked at stint level (n=1044, much better powered): excluding degenerate-flagged cells
  actually WEAKENS Hard's signal (0.207->0.139) and worsens Medium (-0.042->-0.127).
Conclusion: real phenomenon (30/79 cells, 414/1044 stints), but does not resolve into a data-volume
screening rule under either measure tested. Reported as an open question, not forced into the nicer
story. FINDINGS.md #6 extended with this.

**Figure 7 wording was wrong, figure was right.** User caught that "identical 0.011 slope" was
described as "ignores compound" when the figure clearly shows 3 different curves (Soft 6.8s, Medium
2.5s, Hard 1.1s). Checked the booster's actual per-lap increments directly: ages 1-5 are EXACTLY zero
for all 3 compounds, ages 6-7 bit-for-bit identical (0.046, 0.040) -- no split on compound has
happened yet in that window at all; differentiation starts ~age 10. Corrected FINDINGS.md #7 to the
precise claim: no tyre-age response for 10 laps regardless of compound, then discrete uneven steps,
then a hard plateau past ~lap 35 (structurally can't extrapolate past the practice tyre-age range --
the actual strategy question). No figure regeneration needed, only the write-up was wrong.

**Figure 6b axes fixed.** Multiplying by 2.04 pushed 2 known outlier events to ~1.1, autoscaling
crushed all 73 normal points into a corner. Fixed: both panels now clip to a shared 0-0.35 range,
outlier events drawn as triangle markers pinned to the top edge with a compact off-scale-values text
box (was 6 overlapping arrow labels, unreadable -- fixed). Also fixed a real labeling bug: legend
said "73 non-outlier events" (point count) when it meant 28 (event count) -- and added explicit
"all 79 points" vs "28 non-outlier events" labels to Figure 6 and 6b respectively so the 0.23 vs 0.20
numbers don't read as a typo.

All three: `experiments/figure_calibration_corrected.py`, `experiments/phase6_race_validation.py`
(one label line), FINDINGS.md #6/#7 updated, figures regenerated.
