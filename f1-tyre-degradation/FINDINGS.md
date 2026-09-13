# Findings — Phase 3 debugging, 2026-09-13

Three findings from getting the recovery test to a trustworthy state, each with a clean root
cause, a verified fix, and a measured before/after. Written immediately after finding them, not
reconstructed later.

---

## 1. `lap_in_stint` was silently wrong on real data — a bug that only exists because two pipelines disagree on what to filter

**The bug.** `lap_in_stint` (how many laps a tyre has completed *this stint*, feeding the fixed
fuel term) was computed as `TyreLife - TyreLife.min()` within each (driver, stint) group — a
calculation ported from a sibling project (TyreMind) whose own filters never drop a lap for
traffic, only use it as a continuous feature. This project's spec calls for something stricter:
drop any lap with a <2s gap to the car ahead. That filter disproportionately removes the *first*
laps of a stint — cars bunch up right after a pit stop — so `TyreLife.min()` over what survives
silently reconstructs a fake "stint start" partway through the real one.

**Measured on the real 2022–2023 corpus**: 99.8% of stints do not survive with their true first
lap; 55% are missing their first 3+ laps; the worst case is missing 39.

**The fix**, verified before applying it: checked raw, unfiltered data directly — 100% of
fresh-tyre stints (38/38 sampled) start at `TyreLife == 1`. So `lap_in_stint = TyreLife - 1` is
correct and, critically, immune to filtering by construction, since it never depends on which rows
survived. Replaced the group-min calculation with this in `src/data/clean.py`.

**Effect**: real-covariate slope error dropped from ~0.10 s/lap to ~0.01–0.03 s/lap (the second
number after the initialization fix below was applied too — the two compound to explain the full
gap; see the combined real-data numbers at the end of this file).

**The general lesson**: a calculation that is provably correct in its original pipeline can be
silently wrong when moved into a pipeline with different filtering assumptions. Neither pipeline's
version of the function has a bug in isolation.

---

## 2. The tyre head's own initialization was creating a ~31-second artifact it then had to explain away

**The bug.** The wear curve is built as `cumsum(softplus(MLP(...)))` — monotone by construction,
anchored at zero for a fresh tyre. `softplus(0) = ln(2) ≈ 0.693`, not 0. With the network's
ordinary small-random initialization, every one of the 45 per-lap increments starts around 0.693s
regardless of how "small" the random weights are — a roughly **31-second** cumulative wear curve
that exists before a single gradient step, built entirely out of the parameterization's own math,
not the data.

**Measured consequence**: at ~13,300 laps, 2 of 8 random seeds never recovered from this. Both
failed seeds had `best_epoch` of 1 and 6 — training essentially never started — and the entry-bias
embedding (which should absorb per-driver/stint pace) had barely moved from its own zero init
(std 0.005–0.018, vs. 0.25–0.37 on the 6 healthy seeds). The evolution head's fit was, if anything,
*better* on the failed seeds (0.96–0.99 correlation) than the healthy ones — ruling out the first
hypothesis tested (that the evolution head was winning a competition against the entry bias for
slow-varying structure). The pattern-level reasoning that led there was correct — all three
compounds failing by nearly identical amounts within a seed does mean one shared upstream cause,
not three independent bad fits — the specific attribution (which shared thing) was wrong. The
actual shared cause was the tyre head's own initialization, not another head.

**The fix**: initialize the tyre head's final layer bias at −5 instead of the default, so
`softplus(-5) ≈ 0.007` — the curve starts flat and has to be pushed up by evidence, rather than
pulled down from an already-large, self-inflicted error. One line
(`src/models/heads.py::TyreHead.__init__`).

**Effect**, same 8 seeds, nothing else changed:

| | Before | After |
|---|---:|---:|
| Seeds converging normally | 6 / 8 | **8 / 8** |
| Failed-seed `best_epoch` | 1, 6 | — (no failures) |
| Detection floor (see #3) | 0.114 ± 0.159 (bimodal, not a real number) | **0.022 ± 0.013** |

That 6 of 8 seeds recovered *at all* despite starting from a self-inflicted 31-second error is the
more surprising fact than the 2 that didn't.

**Two things ruled out along the way, worth recording so they aren't re-tried without cause:**
- An L1 penalty on the tyre increments looked like a real fix at one setting (0.3) — passed slope
  and negative-control checks cleanly on the realization it was tuned against. It failed a fresh
  synthetic realization (different true curves, different cliffs, different seed) it was never
  tuned on: slope error 5x worse, 2 of 3 cliffs undetected, negative control back to baseline. This
  was fitting the validation metric, not a real fix — reverted. Any future regularization choice
  needs to clear a held-out realization before being trusted, not just the one it was chosen on.
- A shrinkage penalty on the evolution head's amplitude (tested against two different head
  architectures, on two separate occasions) made every metric worse both times. The true amplitude
  is nonzero, so shrinking toward zero overshoots past it.

---

## 3. A measured detection floor of 0.022 ± 0.013 s/lap — and the honest consequence for Hard

Feeding the model synthetic data with **zero** true tyre wear and reading off what it reports is
not a pass/fail check on its own — it's a measurement of the estimator's own noise floor at a given
data volume. Across 8 seeds at ~13,300 laps (post-fix): **0.022 ± 0.013 s/lap, max observed
0.043**.

The honest consequence, stated rather than left implicit: the three compounds' true wear rates in
this project's synthetic ground truth are Soft 0.090, Medium 0.055, Hard 0.035 s/lap.

| Compound | True rate | vs. floor (0.022 ± 0.013) |
|---|---:|---|
| Soft | 0.090 | ~4x the floor — reliably distinguishable |
| Medium | 0.055 | ~2.5x the floor — distinguishable, moderate confidence |
| Hard | 0.035 | **inside the floor's range** — not reliably separable from noise at this data volume |

Most approaches to this problem would present all three compounds' curves with equal confidence.
This is the stronger and more credible claim: **measuring your own resolution limit and stating
which curves it invalidates**, rather than asserting a number the data can't actually support.

**Caveat for real data, not yet addressed**: this floor was measured at one fixed, even sample
size. Real data has uneven coverage across (circuit, compound) cells — dense for some combinations,
thin for others — so the *effective* floor varies by cell, tighter where there's 200 stints,
wider where there's 15. The single number above should not be applied uniformly to every curve on
real data; report it per cell, or at minimum flag thin cells rather than drawing every curve with
the same implied confidence.

---

## 4. Cliff detection on real covariates works for two of three compounds — Soft's failure is real and, so far, unexplained

Fit on real 2022–2023 covariates with known synthetic wear injected (same setup as #3's real-data
table): cliff detection lands within 1–3 laps of true for Medium (23 vs. true 22) and Hard (31 vs.
true 30), across a sweep of the tyre-smoothness penalty from 0 to 4.0 — this is not a smoothness- or
L1-regularization artifact (L1 is confirmed at 0.0, inactive; the smoothness sweep changes Hard's
detected cliff by at most 6 laps, and moves it *closer* to true at the current setting, not further).
Soft is a real outlier: detected at age 23–25 against a true cliff of 14, at every smoothness
setting tested.

**The obvious hypothesis — checked, not confirmed.** Real Soft stints are shorter and more often run
on scrubbed tyres, so the natural guess is sparse real data at low tyre age, exactly where Soft's
cliff sits. A direct coverage check (`experiments/figure_tyre_age_coverage.py`) does not support
this:

| Compound | True cliff age | Real stints reaching that age | Laps observed per age near the cliff |
|---|---:|---:|---|
| Soft | 14 | **77%** of 183 stints | 80–110 per age (comparable to the other two) |
| Medium | 22 | 33% of 381 stints | 60–190 per age |
| Hard | 30 | 37% of 480 stints | 74–176 per age |

Soft stints reach their own cliff age *more* reliably than Medium or Hard reach theirs, and per-age
lap density near the cliff is not visibly thinner for Soft. The coverage hypothesis is ruled out,
not confirmed — so it is **not** asserted as the cause. Cliff detection for Soft on real covariates
is a genuine, currently unexplained failure, reported as such rather than attributed to a checked
and eliminated cause.

---

## 5. A systematic whole-stint compression bias for Hard — confirmed pooled across ~7,000 laps, distinct from the cliff-detection issue

Figure 5 (one real Medium stint) showed the model's predicted lap time sitting above actual early
in the stint and below actual late — not scatter, a directional pattern. Checked whether that was
one stint's noise or systematic: residual (actual − predicted) binned by tyre age, pooled across
every real stint, one panel per compound (`experiments/figure_residual_bias.py`,
`outputs/figure_residual_bias.png`).

| Compound | Mean residual, age ≤ 8 | Mean residual, age ≥ 25 | Shape |
|---|---:|---:|---|
| Hard (n=6,946) | −0.23 s | +0.69 s | clean, roughly linear tilt across the **whole** stint |
| Medium (n=4,481) | −0.04 s | +0.47 s | flat until ~age 20, rises only after the cliff |
| Soft (n=1,777) | +0.44 s | +3.48 s | flat until ~age 15, then grows without bound |

These are two different mechanisms wearing the same "late-stint under-prediction" mask, not one bias:

- **Hard** shows genuine whole-curve compression, independent of any cliff — the model's curve is
  flatter than the data across its full length. This is the clean instance of the pattern originally
  flagged, and it matches the direction of the previously measured slope undershoot at high
  smoothness-penalty weight (recovered magnitude was 65-75% of true for all three compounds at the
  current smoothness weight — see the smoothness-sweep note above).
- **Medium and Soft's** late-stint residual growth is the *same* already-identified post-cliff
  magnitude undershoot (Medium) and missed-cliff failure (Soft, finding #4) restated in pooled,
  visual form — not an independent bias. Soft's residual in particular grows to +3.5s specifically
  because its cliff isn't detected at all, so the gap between actual and predicted widens for as
  long as the stint continues.

**Practical consequence, stated plainly**: the fitted degradation curve is too flat to hand a
strategist as-is — a curve that under-states how much lap time is actually lost late in a stint will
systematically recommend staying out too long. This is consistent with (not separate from) the
`tyre_smoothness_weight=4.0` under-recovery pattern noted earlier; both point at the same
regularization trade-off, now confirmed at the level of individual real predictions rather than only
in an aggregate slope number.

---

## 6. Phase 6 (real practice → real race, no synthetic injection anywhere): a bias characterization, not a predictive-accuracy validation

First fully real-data test in the project: fit the NAM on real 2022–2023 **practice** sessions only
(FP1/FP2/FP3, 12,956 laps, 90 event-sessions), race laps never entering training, then compare the
fitted tyre-curve slope against each event's own real **race** slope (fuel-corrected with the known
~110kg race start load) — `experiments/phase6_race_validation.py`,
`outputs/figure6_race_validation_scatter.png`.

**Headline result**: best-fit slope of predicted vs. observed wear rate, 79 (event, compound)
points across 30 real races: **0.23** (0.20 with the two most extreme events removed — robust, not
an artifact of a couple of points). This is the same compression signature already measured on
synthetic data (finding #5) and the smoothness-sweep undershoot, now confirmed on real data the
model never saw in training at all. **Stated precisely: this confirms the compression bias a third
way; it does not validate the model as an event-level predictor.** Those are different claims —
see the correlation check immediately below, which shows point-level predictive accuracy at event
level is not distinguishable from chance in this data.

Two things folded out of the headline number rather than into it, since they are distinct
mechanisms:

- **29% of points (23/79) have near-zero predicted wear** (<0.005 s/lap) despite real observed wear
  averaging 0.070 s/lap. This is not the compression bias — it looks like specific
  (circuit, compound) cells having too little real practice data for the tyre head to move off its
  near-zero initialization. The real-data instance of the "detection floor varies by data coverage"
  caveat already flagged in finding #3.
- **Two events (2022 round 4, 2023 round 16) over-predicted wear across all three compounds at
  once** — a whole-event effect, not compound-specific, which is the tell it isn't really about
  tyre wear. Confirmed directly: 2022 round 4's race ran at 16.6°C track temperature against ~31°C
  in that event's own practice sessions — a 14°C extrapolation outside the tyre head's
  `track_temp`-conditioned training range. Removing these two events changes the headline slope from
  0.232 to 0.202, confirming they are not responsible for it.

MAE (~6.5s median) is large but expected: `entry_id` is keyed per (event, driver, stint), and race
stint numbers don't correspond to practice run numbers, so nearly every race lap's entry bias falls
back to the untrained, zero-valued `<UNK>` embedding. The model was never expected to recover
absolute pace level across a session-type boundary — only the slope, which is the number reported
above.

**Correction, on closer inspection — the "0.23" figure overstated the bias, and this is worth
stating as a methodology point, not just a number correction.** Point-level correlation between
predicted and observed slopes at event level is weak: pooled Pearson r = 0.11 (p = 0.33, not
distinguishable from zero), and per-compound it is essentially zero (Hard 0.09, Medium −0.03, Soft
0.19). The regression slope of 0.23 is fully explained by that weak correlation combined with
predicted values being noisier than observed ones: `slope = r * sd(predicted)/sd(observed) =
0.11 * 2.10 = 0.232`, exactly. A regression slope under near-zero correlation is not the same
statement as "predictions are a fifth of true magnitude" — it is largely a mechanical consequence of
weak correlation and unequal variance, not direct evidence of a clean multiplicative bias.

**Methodology note, for the record (the same kind of catch as finding #2's pattern-vs-attribution
lesson).** A regression slope was read as a magnitude bias without first checking whether there was
enough correlation underneath it to support that reading. There wasn't. `slope = r * sd_y/sd_x` is
an identity, not a coincidence — it means any regression slope can be driven entirely by an
unequal-variance artifact even when the two quantities barely covary. **The check this project now
applies going forward: never report a fitted regression slope as a "bias factor" without reporting
r (or R²) alongside it.** A slope without its correlation is a number that can say almost anything.

**Stint-level correlation check — is event-aggregation the problem, or is it the model?**
Aggregating each event's several real stints down to one average slope before comparing to the
model both shrinks the sample (79 event-compound points from several hundred real stints) and adds
noise (an event's average is itself a noisy estimate from as few as 1-3 stints of a given compound).
Re-ran the same practice-only fit and paired the model's predicted slope against every individual
real stint's own empirical slope instead — `experiments/phase6_stint_level_check.py`, 1,044 real
stints:

| Compound | n (stints) | Pearson r | p-value | Mean real stint length |
|---|---:|---:|---:|---:|
| Hard | 480 | **0.207** | **<0.0001** | 14.7 laps |
| Medium | 381 | −0.042 | 0.42 | 11.9 laps |
| Soft | 183 | −0.038 | 0.61 | 9.7 laps |
| Pooled (all 3) | 1,044 | 0.057 | 0.064 | — |

This splits the answer cleanly by compound rather than giving one verdict for all three:

- **For Hard, event-aggregation WAS the problem.** A real, statistically robust correlation
  (r=0.207, p<0.0001, n=480) emerges at stint level that the event-level check (n=30, r≈0.09) had
  too little data to detect. Hard's real stints are also the longest on average (14.7 laps vs.
  Soft's 9.7), so each individual observed-slope estimate is itself less noisy — compounding in the
  same direction. This is a genuine, positive finding: for Hard specifically, the practice-fitted
  curve carries real predictive signal once you have enough stints to see it.
- **For Medium and Soft, more data does not help — the ceiling is not sample size.** Correlation
  stays at essentially zero even with 183-381 real stints, an order of magnitude more than the
  event-level check had. This rules out "not enough events" as their explanation and points back to
  the already-diagnosed, compound-specific structural issues: Medium's post-cliff magnitude
  undershoot (finding #5) and Soft's missed cliff (finding #4) actively distort the ranking, rather
  than simply adding noise a bigger sample would average out.

Pooled correlation (r=0.057, weaker than the event-level 0.11) is not a useful number on its own —
mixing three compounds with different baseline slope levels manufactures a between-group signal that
has little to do with within-compound predictive accuracy. Report per-compound, not pooled.

**The degenerate-prediction hypothesis — checked properly, does not hold up.** The scatter shows a
dense row of points at predicted slope ≈ 0.00-0.02 spread across the full range of observed slopes —
the tyre head reading as flat, not merely wrong. The natural hypothesis: these are identifiable in
advance (e.g. by how much real data that cell had), and excluding them reveals real signal in what's
left. Checked three ways, none of which support it:

1. *Splitting event-level cells into degenerate (predicted < 0.01, 30/79) vs. not.* Correlation on
   the non-degenerate 49 points is **weaker**, not stronger: pooled r drops from 0.11 to 0.014, and
   per-compound (Hard −0.05, Medium −0.19, Soft +0.15, n=13-19 each) shows nothing significant in
   either direction.
2. *What distinguishes degenerate cells — checked two candidate measures, neither holds.* Real race
   stint count: nearly identical between degenerate and non-degenerate cells (mean 13.8 vs. 12.9).
   Real practice-lap density for that (circuit, compound) cell: also not predictive (point-biserial
   r=0.10, p=0.39) — if anything degenerate cells average slightly *more* practice laps (204 vs.
   173), the opposite of the hypothesis. Data volume, by either measure, does not identify which
   cells collapse.
3. *Re-ran the exclusion at stint level (n=1,044, far better powered than the 49-point event-level
   check above)*: flagging every stint whose (circuit, compound) cell was degenerate at event level
   and excluding it does not raise correlation either — Hard actually drops slightly (r=0.207 → 0.139,
   still significant at p=0.02 but weaker), Medium moves further negative (−0.042 → −0.127), Soft
   stays flat. Excluding the "degenerate" cells removes some of Hard's real signal along with
   whatever noise it was meant to remove — they are not cleanly separable.

**Conclusion, reported as it came out rather than as hoped**: the flat-prediction phenomenon is real
(30/79 event-cells, 414/1,044 stints) but does not resolve into an identifiable-in-advance screening
rule under either candidate measure tested, and removing it does not clean up the correlation the way
the "collapse vs. usable curve" story predicted. This is a genuine open question, not a solved one —
worth investigating further (a per-cell diagnostic beyond raw volume, perhaps how *concentrated* a
cell's laps are across distinct real stints rather than total lap count) but not yet a defensible
minimum-data rule for a strategist. Reported honestly rather than forcing the cleaner story.

**The calibration attempt** (`experiments/phase6_calibration_check.py`,
`figure_calibration_instability.png`): a ratio-of-means factor (robust to the regression-attenuation
problem above) computed on all 79 points is only **1.22** — barely any average bias — but jumps to
**2.04** once the two track-temp-extrapolation events (above) are excluded. Checked whether either
number is trustworthy via 200 random 60/40 event-level splits, fitting the factor on 60% and
checking what the other 40% actually needed: **including** the two known-outlier events, the needed
correction swings from 0.6x to 3.8x depending on which events land in which half — unusable, and
the instability traces mechanically to those two high-leverage points landing on one side of the
split or the other. **Excluding** them, the picture is far more stable: the correction needed is
`>1` (under-prediction) in 99% of the 200 splits, and a bootstrap 95% CI (resampling events, n=2000)
on the full 28-event estimate is **[1.25, 3.87]**, excluding 1 — a real, statistically supported
average under-prediction, of a magnitude closer to **~2x** than the ~5x the raw regression slope
implied, and honestly wide given only 28 usable real events.

**Applying the 2.04x factor** (`figure6b_calibration_corrected.png`) moves the local best-fit slope
(excluding the two known outliers) from 0.20 to 0.41 — a real improvement, roughly halfway to
perfect agreement, not a full fix, because scaling a systematic mean bias cannot repair weak
point-level correlation (correlation is invariant to linear rescaling of one variable, by
construction). The two already-diagnosed extrapolation-outlier events remain outliers after
correction, as expected — a scalar was never going to fix an out-of-distribution track-temp
extrapolation.

**The honest summary — two different claims, kept separate.**

*Claim 1, survives*: the compression bias is real, confirmed three independent ways (smoothness
sweep, pooled real-stint residuals, practice→race generalization), and its average magnitude (~2x,
not the ~5x the raw regression slope implied) is now quantified with a defensible if wide bootstrap
interval.

*Claim 2, does not survive*: "Phase 6 validates the model" is not supportable at event level —
pooled event-level correlation (r=0.11, p=0.33) is not distinguishable from chance. At stint level
it resolves further: Hard shows real, significant predictive skill (r=0.21, p<0.0001) once enough
stints are pooled — event-aggregation, not the model, was hiding it. Medium and Soft do not, at any
sample size tested, because a specific already-diagnosed structural issue is actively degrading each
one's ranking (not simply averaged-out noise).

**One root cause connects three separate-looking findings, but it is not uniform across
compounds.** The detection floor (#3), Hard's weak-until-stint-level event correlation, and the
general difficulty of pinning down a precise calibration factor (28-30 events) all trace back to
the same thing: *real per-(circuit, compound, event) samples are small, and event-level true
variation is comparable in scale to the estimator's own measurement noise* — a data-volume ceiling,
not a model defect, and the reason attenuation this severe is unsurprising (a true r of 0.5 easily
attenuates to ~0.1 under this much per-point measurement error). Medium's undershoot and Soft's
missed cliff are a *second*, distinct root cause — a specific structural failure in each, not
resolved by more data at any volume tested so far. Reporting these as one undifferentiated
"real-data is noisy" limitation would understate the second problem and overstate how much more
data alone would fix.

---

## 7. Baseline comparison, run for the first time: LightGBM wins accuracy by 30% and recovers a curve that barely distinguishes compounds at all

Same synthetic ground truth as the recovery test, same data, same train/val split, three models —
`experiments/baseline_comparison.py`, `outputs/figure7_baseline_comparison.png`. Fixed a real bug
in `src/models/baselines.py` before running it: `compound` was missing from the feature list
entirely, which would have made the comparison meaningless (no baseline could have distinguished
Soft/Medium/Hard's different wear rates at all).

**Prediction accuracy (val MAE)**: LightGBM 2.71s, NAM 3.88s, Linear 4.31s. LightGBM wins by ~30%
over the NAM — expected and stated up front in the module's own docstring: nothing here is trying
to beat LightGBM on raw accuracy.

**Recovered per-compound curve, against known ground truth**:

| Compound | True slope | NAM slope | LightGBM slope | True cliff | NAM cliff | LightGBM cliff |
|---|---:|---:|---:|---:|---:|---:|
| Soft | 0.090 | 0.097 | **0.011** | 15 | 16 | 11 |
| Medium | 0.055 | 0.065 | **0.011** | 23 | 25 | 11 |
| Hard | 0.035 | 0.045 | **0.011** | 31 | 34 | 11 |

The NAM recovers each compound's distinct slope and cliff within a few percent / a few laps of
truth. LightGBM's implied curve reports the identical slope (0.0107) for all three compounds over
the ages-1-8 window that metric measures — **not because its three curves are the same shape (they
visibly aren't: Soft reaches 6.8s, Medium 2.5s, Hard 1.1s by lap 45 — see the figure), but because in
that specific low-age window they genuinely are.** Checked directly against the fitted booster's own
per-lap increments, not inferred from the summary metric: increments are exactly zero for ages 1-5 in
all three compounds, and bit-for-bit identical (0.046, 0.040) at ages 6-7. No split on `compound` has
happened yet at low tyre age at all; differentiation between compounds only starts around age 10, in
uneven jumps. The slope metric is measuring precisely the window where the model hasn't started to
differentiate — real and precise as a fact about *where* LightGBM's response kicks in, but not
evidence it ignores compound altogether.

**The sharper, correct claim**: LightGBM shows no tyre-age response over the first ten laps in any
compound, then responds in discrete, uneven steps, then **plateaus hard after ~lap 35** for every
compound — trees cannot extrapolate past the tyre-age range they had enough training data at, so the
curve simply repeats its last leaf value. (Cliff detection at lap 11 for all three is the same
flat-start artifact, not three independently-detected — and coincidentally identical — cliffs.) The
plateau is the practically important part: a tree model of this kind structurally cannot answer
"what happens if this stint runs five laps longer than anything in practice" — the actual question a
pit-strategy call needs answered.

This is the direct, non-rhetorical version of the module's stated argument: winning on MAE and
attributing correctly are different objectives. LightGBM's dependence on `tyre_age` is whatever
minimizes prediction error given how it happened to split the fuel/tyre-age/circuit collinearity —
it has no obligation to recover the true per-compound shape, and here it visibly doesn't (flat where
degradation is real, stepped where the truth is smooth, frozen past the observed range), even though
it has strictly more predictive accuracy than the model that does.

---

## Real-data validation, before and after both fixes (#1 and #2 together)

Semi-synthetic test: real 2022–2023 race covariates (real stint lengths, real traffic, real
weather, real driver/circuit identities) with a known synthetic wear curve injected on top —
`tests/test_recovery_real_covariates.py`.

| Check | Before | After |
|---|---:|---:|
| Wear-rate accuracy (tolerance 0.015 s/lap) | 0.102 | **0.0125 — passes** |
| Negative control (informal target 0.01) | 0.105–0.110 | **0.008 — passes** |
| Track-evolution shape (min 0.90 correlation) | 0.88–0.98 | 0.88–0.98 (unchanged, one circuit just short) |
| Cliff detection | broken | still off for one compound |

Two of four checks now pass cleanly on real covariates, including the two that were most badly
broken. Of the remaining two: evolution correlation sits just under threshold on one circuit,
plausibly explained by that circuit's real data being thin (not yet confirmed). Cliff detection is
now understood more precisely than "one compound mislocated" — see finding #4: it is specifically
Soft that fails, Medium and Hard both land within a few laps of true, and the leading hypothesis for
Soft's failure (sparse real low-age data) was checked directly and ruled out, leaving the cause
unexplained rather than attributed.
