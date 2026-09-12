# Pre-registration — exp32, is the traffic effect a mechanism or an artefact?

**Written before the experiment was run. Committed before the results existed.**

The only things known at the time of writing are (i) the finding to be explained,
restated in §1, and (ii) structural facts about the data — how many stints
survive the filters, that the OpenF1 timestamps align to FastF1 lap windows, that
a per-MJ slope is computable. No path coefficient, no mediated proportion and no
quartile contrast had been computed when this document was written.

---

## 1. The finding this experiment exists to explain

A judge asked: *if there is traffic, how does your model predict degradation?*
Measured across real stints, the answer is uncomfortable:

| | mean fitted degradation slope |
|---|---|
| clear air (bottom traffic quartile) | **+0.1037 s/lap** |
| heavy traffic (top traffic quartile) | **+0.0703 s/lap** |
| difference | **−0.0335 s/lap**, t = −6.32, p = 4e−10 |

It holds inside every compound (HARD ρ = −0.278, MEDIUM −0.178, SOFT −0.186).
**Stints in traffic show less measured degradation.** Two explanations, with
opposite consequences:

**(A) Real mechanism.** In traffic you cannot push. Less frictional energy goes
through the tyre, so it genuinely wears less. The wear-energy literature in
`research/papers/02_tyre_wear_physics/` models wear rate as proportional to
frictional power dissipated in the contact patch, not to distance travelled. If
(A) holds, traffic is a **rate modifier we should model**.

**(B) Measurement artefact.** A car stuck behind another is pace-limited by the
car ahead, not by its own tyre. The tyre keeps degrading; the lap time cannot
show it. The degradation is **masked, not reduced**. If (B) holds, every
lap-time-based degradation method — ours included — **systematically
under-reports degradation in traffic**, and that is a limitation paragraph in any
paper built on lap times.

Both are publishable. This document fixes, in advance, what would make us say
each one.

## 2. Why this is answerable now and was not before

`data/telemetry/*.parquet` carries 202 sessions of per-lap frictional energy and
load, reduced from FastF1's ~4 Hz car and position streams by
`scripts/build_telemetry.py`. Energy is therefore **observed**, not assumed.

That converts the question into a mediation question with a clean prediction on
each side:

> If (A) is true, energy **mediates** traffic → degradation. Traffic reduces
> energy, energy drives wear, and controlling for energy should collapse the
> traffic coefficient.
>
> If (B) is true, traffic still predicts lower measured degradation **after
> energy is partialled out**, because the masking lives in the lap time rather
> than in the physics.

## 3. Sample, unit of analysis, filters

**Unit:** one stint = (session, driver, `run_id`). **Sample:** race sessions
(`session == "R"`) in `data/season` that have a joinable telemetry file, joined
on `(driver, session_lap)`. Lap tables are read through
`tyremind.data.corpus.read_lap_table` and telemetry through
`read_telemetry_table`; no direct parquet reads.

Filters, fixed here:

* at least **10 laps** in the stint;
* at least **6 laps of spread** in `tyre_age` (`np.ptp`);
* `|slope| ≤ 0.5` s/lap — beyond that is a damaged car, a wet patch or a failed
  fit, not a tyre;
* every `np.polyfit` call wrapped in `try/except (np.linalg.LinAlgError, ValueError)`,
  finiteness tested with `np.isfinite`, never `np.isnan`;
* a stint whose fit raises or returns a non-finite slope is **dropped and
  counted**, never silently skipped.

A structural pilot (counts only, no effect estimates) put this at **3,222 stints
across 83 race sessions**, of which 2,940 sit in a (session, driver) group with
at least two stints. That is the power available and it was known before any
coefficient was.

## 4. Variables

**Outcome.** `slope_per_lap` — OLS slope of fuel-corrected lap time on
`tyre_age`, fuel correction `+0.081 × lap_in_run` s, matching the rest of the
ladder. Units s/lap. Higher = more measured degradation.

**Second outcome (the energy clock, §7).** `slope_per_mj` — the same corrected
lap times regressed on **cumulative frictional energy** within the stint
(`energy_mj_total`, cumulated lagged so the first lap is 0 MJ). Units s/MJ.

**Treatment.**
* *Derived:* `traffic_index`, mean over the stint. Available for every stint.
  Higher = more traffic.
* *Measured (confirmatory):* from `data/reference/openf1_intervals/*.parquet`,
  70 races of real ~4 Hz gaps. Each interval sample is assigned to the lap whose
  `[LapStartDate, LapStartDate + LapTime)` window contains it, using FastF1's
  cached session (offline) for lap windows and for the driver-number →
  abbreviation map. Samples flagged `interval_lapped` are discarded as ambiguous
  (a lapped car ahead is a different state from a gap). A lap needs **≥ 3 usable
  samples**; a stint needs ≥ 5 such laps. The primary measured regressor is
  **`close_fraction`** — the share of samples with gap ≤ 1.0 s — because it is
  bounded, monotone in traffic and directionally comparable to the derived index.
  `mean_interval` (seconds, direction reversed) is reported as a sensitivity.

**Mediator (primary).** `energy_mj_total`, mean per lap over the stint.

**Mediator block (the generous version of (A)).** All six effort channels
jointly: `energy_mj_total`, `mean_abs_lateral_g`, `p95_lateral_g`,
`mean_abs_long_g`, `loaded_fraction`, `full_throttle_fraction`,
`braking_fraction`, `mean_speed_kmh`. If (A) is true in any form these should
carry the effect, so (A) is given its best shot rather than only its narrowest.

**Covariates, fixed here.** Compound dummies, stint length in laps, mean tyre age
of the stint. All continuous variables are z-scored over the analysis sample so
coefficients are comparable; the mediated proportion is scale-free either way.

## 5. The two specifications

**S1 — pooled, session × compound fixed effects.** Every variable is demeaned
within its (session, compound) cell before the regressions, so circuit,
compound, race conditions and field-wide effects cannot produce the result.

**S2 — within driver, within race.** Every variable is demeaned within its
(session, driver) group; only groups with ≥ 2 stints contribute. The contrast is
then between *the same driver's own stints in the same race*, so **car pace
cannot explain the effect**. Compound dummies stay in the regression.

S2 is the specification the finding must survive. S1 is reported alongside for
readability and because it has more stints.

## 6. Mediation — estimator and decision rule

Four regressions per specification, all OLS:

| path | regression | coefficient |
|---|---|---|
| **a** | mediator ~ traffic + covariates | traffic → energy |
| **b**, **c′** | slope ~ mediator + traffic + covariates | energy → slope; direct traffic effect |
| **c** | slope ~ traffic + covariates | total traffic effect |

Indirect effect = **a·b**. Mediated proportion = **a·b / c**. Confidence
intervals by **cluster bootstrap resampling whole sessions**, 2,000 replicates,
percentile method at 95%, because stints within a race are not independent. For
the mediator *block*, the indirect effect is `c − c′` (the sum of products over
mediators), bootstrapped the same way.

**Direction expected before running:** c < 0 (the finding, restated). If c is not
negative and significant in S1, the finding has not replicated on this sample and
the experiment reports that instead of an explanation.

Decision rule, fixed now:

* **(A) REAL MECHANISM is supported** if *all* of:
  1. `a` CI excludes 0 and `a < 0` — traffic really does reduce energy;
  2. `b` CI excludes 0 and `b > 0` — energy really does predict degradation;
  3. indirect `a·b` CI excludes 0 and is negative;
  4. mediated proportion ≥ **0.50** with bootstrap lower bound > **0.25**;
  5. the direct effect `c′` either has a CI covering 0, or `|c′| < 0.5·|c|`.

* **(B) MEASUREMENT ARTEFACT is supported** if:
  1. `c′` CI excludes 0 and is negative, and
  2. `|c′| ≥ 0.7·|c|` — equivalently mediated proportion ≤ **0.30** with
     bootstrap upper bound < **0.50**.

  Traffic then predicts lower measured degradation with energy held fixed, which
  is what masking looks like and what a real wear-rate mechanism cannot produce.

* **INCONCLUSIVE** — and it will be called inconclusive, not spun — if any of:
  * the `b` path's CI covers 0, or `b < 0`. **This is a live risk and it is
    declared now.** exp25 found the naive stint-level energy→degradation
    correlation to be *negative*, because high-energy laps are fast laps and fast
    laps happen on fresh tyres. If the mediator does not predict the outcome in
    the right direction under the fixed specifications, mediation is not
    identified and no mediated proportion may be quoted from it;
  * the mediated proportion lands between 0.30 and 0.50, or its CI spans that
    band;
  * S1 and S2 disagree in *direction* on the direct effect.

* If both the indirect effect and the direct effect are significant and the
  mediated proportion is between 0.30 and 0.50, the honest reading is **partial
  mediation — both are true in part** — and it is reported as that.

## 7. The energy clock — a second, independent discriminator

Mediation depends on a linear model. The energy clock does not, and it makes the
same distinction from a different direction.

Refit each stint's corrected lap time against **cumulative energy** instead of
tyre age, giving `slope_per_mj` in s/MJ. Then compare the clear-air and heavy
traffic quartiles on both clocks.

* **(A) predicts:** traffic lowers energy per lap, so the per-lap slope falls —
  but the tyre's loss *per megajoule* is unchanged. `slope_per_mj` should show
  **no significant difference** between quartiles, and Spearman(traffic,
  `slope_per_mj`) should be ≈ 0.
* **(B) predicts:** the masking is in the lap time, so it survives a change of
  clock. `slope_per_mj` should be **significantly lower in traffic too**, with a
  similar relative deficit.

The relative deficit — (clear − traffic) / clear — is computed on both clocks and
reported side by side. This test is confirmatory, not exploratory; its direction
is fixed here.

## 8. Model uncertainty in traffic

Fit `tyremind.models.ssm.tyre_ssm.fit_tyre_ssm` on every race session and take
the per-lap posterior `rate_sd` from `degradation()`. A pilot on 16 sessions gave
mean sd 0.0274 in clear sessions against 0.0421 in busy ones — the right
direction, not significant at that n.

Pre-registered test: regress `rate_sd` on lap `traffic_index` after demeaning
both within (session, driver) and controlling for `tyre_age`, with a cluster
bootstrap over sessions. **Direction expected: positive** — the model should
report *wider* intervals when the lap time is less informative about the tyre.
The session-level pilot contrast (mean `rate_sd` in the top vs bottom traffic
tercile of sessions) is re-run at full scale as the pilot's direct replication.

This is a property of our estimator, not evidence for (A) or (B). A widening
interval under (B) is the model partially protecting the user; a widening
interval under (A) would be harder to justify. Reported either way.

## 9. Exploratory — declared as exploratory

Not part of any decision rule:

* R² of the stint fit, clear vs traffic. Under (B) the lap time is partly set by
  the car ahead, so tyre age should explain less of it.
* `full_throttle_fraction` and `braking_fraction` by quartile — the pace-margin
  signature.
* The derived-vs-measured traffic agreement at lap level, per race.
* Per-compound breakdowns of every headline number.

## 10. Known limitations, stated up front

1. **Energy is estimated, not measured.** `build_telemetry.py` reduces 4 Hz car
   data to a per-corner frictional-energy proxy under a load model. If that proxy
   is attenuated, mediation is attenuated with it, which biases the test
   **towards (B)**. This asymmetry is declared now and must be repeated next to
   any (B) conclusion.
2. **A stint-mean mediator cannot see within-stint timing.** Traffic that arrives
   late in a stint affects the fitted slope differently from traffic throughout,
   and a mean over the stint erases that.
3. **Traffic is not randomly assigned.** Slow cars are in traffic. S2 removes the
   between-car part of that; it does not remove a driver choosing to manage the
   tyre on the race day they happened to be stuck.
4. **Mediation assumes no unmeasured mediator–outcome confounding.** A variable
   that both lowers energy and lowers measured degradation — a cool track, a lift
   phase — would masquerade as mediation and inflate the (A) reading.
5. **`slope_per_mj` inherits the same lap times.** The energy clock changes the
   x-axis, not the y-axis. It cannot rescue a lap time that was never a
   measurement of the tyre.

## 11. Fixed before running

* Sample, filters, unit of analysis: §3.
* Variables and their direction: §4.
* Specifications S1 and S2, and the covariate set: §5.
* Paths, estimator, bootstrap scheme (2,000 replicates, clustered on session,
  percentile CI): §6.
* The thresholds 0.50 / 0.25 for (A), 0.30 / 0.50 / 0.7 for (B), and the
  inconclusive band: §6.
* The energy-clock prediction on each hypothesis: §7.
* The sign expected of the uncertainty test: §8.

Any deviation is recorded in `experiments/results/exp32_traffic_mechanism.json`
under `deviations`, with a reason, and repeated in the reported summary.
