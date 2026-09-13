# Pre-registration — exp33, kerb exposure and tyre degradation

**Written before the degradation analysis was run. Committed before those results
existed.** §3 declares exactly what *had* already been run when this was written,
because a pre-registration that hides a pilot is worse than no pre-registration.

---

## 1. The question, and why we have no answer

A judge asked:

> When a car hits the kerbs, how does tyre degradation happen and what is the
> effect?

TyreMind does not model kerb strikes at all. A big one becomes an outlier lap the
MAD filter removes, or it ends the stint. That is an honest gap, and this
experiment is the attempt to close it.

The physics is not in dispute. The wear models in
`research/papers/02_tyre_wear_physics/` make wear rate proportional to frictional
power dissipated in the contact patch, and a kerb is a harder, ridged, higher-slip
surface than asphalt with a vertical input on top. Riding kerbs should cost tyre
life. What our literature review could not find is anyone who has **measured** it
from public data.

Nine hypotheses in this project have been refuted so far, including exp25, where
every significant correlation between measured frictional energy and degradation
had the wrong sign and nothing survived at circuit level. A null result here is
entirely plausible and is a perfectly good outcome.

## 2. What is being measured, and what is not

Neither measure below sees a kerb. Detecting an actual strike needs the vertical
accelerometer channel, which is team-private exactly like tyre temperature and
pressure. These are the two proxies public data can support.

**Measure A — lateral deviation from the racing line.** For every position sample
of every lap, the perpendicular distance to the session's fastest lap's racing
line, summarised per lap as the fraction of lap distance beyond a threshold. Built
by `scripts/build_kerb_exposure.py`.

- The reference is a **racing line, not a surveyed centreline**: it is where the
  fastest lap of that session went. Deviation is relative to "what a fast lap
  did", so a driver running a different line for fuel, traffic or car balance
  registers deviation without touching anything.
- `reference_stability_m` — the median lateral distance between that reference and
  one built from the next-fastest lap by a different driver — is recorded per
  session as the honest floor under every threshold.
- The position feed samples at ~4.2 Hz, so a 0.3 s strike can fall between two
  samples. Only *sustained* wide running could survive that.
- **The threshold is a choice.** Every threshold in `THRESHOLDS_M` = (1.0, 1.5,
  2.0, 2.5, 3.0, 4.0) m is measured and the sensitivity is reported. The headline
  threshold is fixed **now** at **2.0 m** — roughly a car's half-width plus a
  margin — and no other threshold may be promoted to headline after the results
  are seen.

**Measure B — track-limits excursions.** Every lap race control deleted for
exceeding track limits, with the corner, read from FastF1's `Deleted` and
`DeletedReason` lap columns. These are adjudicated facts, not inferences.

- An excursion is **the subset of wide running that a monitored corner caught**,
  at the two or three corners per circuit that are monitored. A driver riding a
  kerb inside the white line generates nothing.
- It is entangled with **how hard the driver was pushing**, which has its own
  effect on degradation. The within-driver contrasts in §5 exist because of this,
  and they do not remove it entirely. Stated now, not after.

## 3. What had already been run when this was written

A measurement pilot, on five sessions, before this document. It is declared
because it changed the design.

The pilot compared each car's position against the racing line and found on-track
lateral deviation concentrated in the first few centimetres, and it found that at
the 2024 Austrian Grand Prix **all 190 pairs of cars reach a minimum separation of
0.00 m**. Two Formula 1 cars cannot occupy the same point. The pilot's reading is
that the public position feed reports position as distance along a single path and
discards the lateral coordinate, which would make Measure A unmeasurable in
principle rather than merely noisy.

**Nothing relating either measure to degradation has been computed.** The pilot
looked only at the position feed. §4 turns the pilot's reading into a gate with
numbers attached, fixed before the corpus-wide diagnostic was run; §5 and §6 are
untouched by it.

## 4. The measurement-validity gate for Measure A — fixed now

Measured over all 84 race sessions, pooling every moving on-track position sample
within 10 m of the racing line (beyond that is the pit lane, a genuinely different
path, and including it would be evidence for the feed that the feed has not
earned).

Measure A is **valid** only if all three hold:

| Gate | Condition | Why |
|---|---|---|
| **G1** | corpus-median session `dev_p99_m` ≥ **2.0 m** | If the 99th percentile of lateral deviation is under a car's half-width, the feed cannot resolve a car off the line at all. |
| **G2** | corpus-median `dev_p99_m` ≥ **5 ×** corpus-median `reference_stability_m` | Deviation must be larger than two fast drivers' disagreement about the line, or it is measuring the disagreement. |
| **G3** | fewer than **25%** of sessions have any car pair within **0.05 m** | Real position data never puts two cars at the same point. |

If any gate fails, Measure A is **declared unmeasurable**, and that is reported as
the primary finding of this experiment — a negative result about the data, not
about kerbs. Its degradation analysis still runs, and is reported, in the role
assigned in §6.

## 5. The hypotheses, the metrics and the directions

**Sessions.** Races only (84 of them). Practice is excluded because the fuel
correction below is a race quantity and practice run structure is not stint
structure.

**Stints.** Matching the rest of the ladder: lap tables read through
`tyremind.data.corpus.read_lap_table`; fuel correction **0.081 s/lap** applied to
`lap_in_run`; slope fitted by least squares of corrected lap time on `tyre_age`;
a stint kept only with **≥ 10 laps** and **≥ 6 laps of tyre-age spread**; slopes
with **|slope| > 0.5 s/lap** dropped as a damaged car, a wet patch or a failed fit.
Laps that are pit in- or out-laps, that are lap 1, or that are not run under green
(`track_status != "1"`) are excluded before the fit and before the exposure
average, because all three are off-line for reasons that have nothing to do with
kerbs.

**Per-stint exposure.**

- `exposure_frac_2m0` — mean over the stint's laps of the fraction of lap distance
  more than 2.0 m from the racing line. (Measure A, headline.)
- `limits_rate` — deleted-for-track-limits laps in the stint ÷ laps in the stint.
  (Measure B, headline.)
- `any_limits` — `limits_rate > 0`.

**H1 (primary).** Stints with more kerb exposure show **larger** fuel-corrected
degradation slopes. Direction: **positive**. More exposure, more seconds per lap
lost per lap of tyre age.

**The tests, in the order they are decisive.**

| | Test | Contrast | Direction |
|---|---|---|---|
| **T1** | Spearman ρ(`limits_rate`, slope), pooled | none | positive |
| **T2** | Spearman ρ on circuit-demeaned values | within circuit | positive |
| **T3** | Spearman ρ on (driver × circuit)-demeaned values, groups of ≥ 3 stints | within driver *and* circuit | positive |
| **T4** | Wilcoxon signed-rank on the per-(driver, circuit) difference in mean slope between stints with an excursion and stints without | paired within driver and circuit | positive |
| **T5** | Leave-one-circuit-out MAE: compound label alone vs compound label + exposure, Wilcoxon paired against the label-only baseline, as exp09 does | out-of-sample, unseen venue | exposure lowers MAE |

T1 is reported first and believed least. exp25 is the precedent: its stint-level
correlations reached significance with the wrong sign, because high-energy laps
are fast laps and fast laps happen on fresh tyres. Kerb exposure is confounded with
circuit — some circuits have more kerbs and more monitored corners — and with
driving style, which is why T2 and T3 exist and why T3 and T4 carry the verdict.

**Negative control.** T2 and T3 are also run on `exposure_frac_2m0`. If §4's gate
fails, Measure A provably contains no information about where the car was, so
these must come out null. A significant result from a measure known to be
information-free would mean the demeaning has not removed what it is supposed to
remove, and **the whole design would be discounted accordingly** — including T1–T5.
That is the point of running it.

## 6. What counts as support, and what counts as a null

Fixed now.

**H1 is supported** only if **all three** hold:

1. **T3** is positive with p < 0.05;
2. **T4** is positive with p < 0.05 and rests on **≥ 20** (driver, circuit) pairs;
3. **T5** shows exposure lowering out-of-sample MAE against the compound-label
   baseline, with Wilcoxon p < 0.05.

Anything short of all three is a **null**, and is reported as "kerb exposure, as
public data can measure it, does not predict degradation" — not as "a trend".

**H1 is refuted with a wrong sign** if T2 or T3 is significant and **negative**.
That is the exp25 outcome and it gets the same treatment: reported plainly, with
the confound named.

**If the §4 gate fails**, the headline finding is the measurement result, and the
Measure A analysis is demoted to the negative control described in §5. Measure B
still carries H1 on its own. This is written here so that a failed gate cannot be
presented afterwards as though Measure B had been the plan all along.

**Multiplicity.** Six thresholds × two contrasts is twelve tests on Measure A.
Significance for Measure A is judged at Bonferroni-corrected α = 0.05/12 ≈ 0.0042.
Measure B has one headline metric and its five tests are the pre-declared sequence
above, judged at α = 0.05 each with the conjunction in §6 as the real bar.

## 7. Power — stated before, not after

The smallest correlation detectable at 80% power and α = 0.05 is approximately
ρ = 2.8/√(N−3). The experiment reports this number for every test it runs,
computed from that test's own N, **next to the result**, so a null can be read as
"no effect larger than ρ" rather than as "no effect".

T4's power is unknown until the pairs are counted. If fewer than 20 (driver,
circuit) pairs have both an excursion stint and a clean stint, T4 is reported as
**underpowered**, not as a null, and H1 cannot be supported (§6.2) — it also cannot
be refuted by T4 alone.

**A structural power limit that no sample size fixes.** One track-limits excursion
is one corner on one lap of a ~20-lap stint: order 0.1% of the stint's cornering.
If the effect of kerb contact is proportional to the contact, the effect on a
stint slope is far below the noise floor of a slope fitted from 20 lap times. The
only way this experiment sees anything is if an excursion is a **marker for a
style** — a driver who gets caught once was running wide on many laps that were not
caught — rather than a cause in itself. That interpretation is declared now so that
a positive result is read for what it would be: evidence about drivers who use the
kerbs, not a measurement of one kerb strike's cost.

## 8. Known limitations — stated up front

1. **Exposure, not impact.** Neither measure observes contact. No public channel
   does.
2. **The reference line is a racing line.** Measure A's zero is one driver's fast
   lap, not the track's centre.
3. **Track-limits enforcement is not uniform.** Corners monitored, and the appetite
   to police them, vary by circuit and by year. Within-circuit contrasts absorb the
   circuit part; they do not absorb a mid-race change of enforcement.
4. **Deleted laps are fast laps.** A lap run wide is often quicker. That biases a
   stint's level, and it could bias the slope if excursions cluster early or late in
   a stint. The experiment reports the mean lap-in-stint position of excursions so
   the direction of that bias is visible.
5. **Wet races are not excluded separately.** The |slope| > 0.5 filter and the
   green-flag filter remove the worst of them; exp09's `dry_session` flag is not
   applied here, because it is keyed to (year, round) and this experiment is keyed
   to sessions. Declared as a deviation from exp09's practice.
6. **This cannot answer the judge's question about a single strike.** It can answer
   whether drivers who go off the track degrade differently. Those are different
   questions and the report must not merge them.

## 9. Fixed before running

- Both measures, their thresholds, and the 2.0 m headline: §2.
- The measurement-validity gate and its three numeric conditions: §4.
- Stint filters, fuel correction, slope method: §5.
- Tests, directions, the conjunction that counts as support, and the Bonferroni
  correction: §5 and §6.
- The negative control and what a positive negative-control result would mean: §5.
- The power statement and the underpowered-not-null rule: §7.

Any deviation must be recorded in the result file as a deviation, with a reason.
