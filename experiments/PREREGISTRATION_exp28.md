# Pre-registration — exp28, the value experiment

**Written before the experiment was run. Committed before the results existed.**

---

## 1. The question

Every number this project reports is in seconds per lap of degradation error. A
sporting director does not buy seconds per lap. The question this experiment
answers is:

> Replaying real races, how much race time and how many track positions would a
> team have gained or lost by acting on each model's pit recommendation instead of
> the stop it actually made?

and, as one sentence:

> "Using the naive estimate instead of ours would have cost X seconds and Y
> positions per race."

X and Y are whatever comes out. If the sign is against us, the sentence is
reported with the sign it has.

## 2. The replay, and why it is causal

For each race, each real pit stop that survives the exclusion rules below is
replayed:

1. A **decision lap** is chosen one third of the way into the stint that ended in
   that stop — the same rule exp22 uses. Deciding at the stop itself is hindsight;
   deciding on lap one is deciding on no evidence.
2. Each model is fitted on **an expanding prefix of the race** ending at or before
   the decision lap. Four checkpoints, at 15%, 30%, 50% and 70% of race distance;
   a decision uses the latest checkpoint at or before it. **No lap after the
   decision reaches the fit.** A stop whose decision lap falls before the first
   checkpoint is recorded as unscorable, not silently dropped.

   This is stricter than exp22, which fits every model on the whole race. The
   deviation is deliberate and in our disfavour: a value claim that depends on
   hindsight is not a value claim.
3. The model recommends a pit lap via `tyremind.models.pit_decision.recommend_pit_lap`,
   the same optimiser for every model, so the comparison is between degradation
   estimates and not between nine pieces of strategy code.
4. The recommended lap and the lap actually taken are both priced by the
   simulator, from the decision lap to the flag, under a **referee** (Section 4)
   that is the same for every model.

## 3. The scoring

Both candidate laps are simulated by `tyremind.simulate.race.simulate_strategy`
with **common random numbers** (identical seed), so the paired difference is not
swamped by Monte Carlo noise. The compound fitted is the one the driver actually
fitted at that stop, for both candidates, so the comparison isolates the *lap*.

    value_s(model, stop) = E[time | actual lap] - E[time | recommended lap]

Positive means following the model would have been faster. This is **signed**.
`strategy_regret` clamps at zero, which is right for a product surface — it will
not tell a user their stop was better than the model's — and wrong for a
benchmark, where a model that recommends worse laps must be allowed to score
negative. The clamped `regret_s` is recorded alongside for continuity with
`/api/session/{id}/regret`, and the deviation is noted in the result file.

**Positions.** Seconds are converted with a *per-race measured* seconds-per-position:
each driver's median lap time projected over race distance, sorted, and the median
gap between adjacent cars taken. The project's shipped value model
(`explain/business.py`) currently assumes a flat 8.0 s; both the measured figure
and the result under the 8.0 s assumption are reported, so the shipped assumption
becomes checkable.

Positions are the **softer** of the two numbers and are labelled so. Real
finishing gaps vary by an order of magnitude between the front and the midfield,
and a median gap cannot represent that.

**Unit for the headline sentence:** one car, one race — the sum of `value_s` over
the scored stops of a single driver in a single race, averaged over all
(race, driver) pairs. Per-stop figures are reported too.

**Only a driver's final stop carries the headline.** `simulate_strategy` prices one
stop and then runs to the flag. For a driver's last stop that is exactly what
happened, so both candidate laps are priced correctly. For an earlier stop it is
not — a long, unbroken final stint gets charged to whichever candidate boxed
sooner, which systematically punishes the earlier lap and, since every model on
this ladder recommends earlier than the teams did, would systematically punish
every model for a fault of the scorer. Earlier stops are still replayed, recorded
and reported, under a label saying exactly that.

For the same reason `recommend_pit_lap` is called with `max_remaining_stops=1`:
the optimiser and the counterfactual must price the same number of stints, or a
model is charged for a second stop the counterfactual never lets it take. Applied
identically to every rung.

**The headline sentence is a paired two-model comparison** — ours against the
naive rung, on the final stops *both* answered. That is a much larger and better
matched sample than the six-way intersection, which is used for the ranking table
and is reported separately.

## 4. The referee, and the circularity it is there to avoid

The counterfactual "what would that lap have cost" needs a degradation truth. No
public data contains one. Whichever estimate is used, the model closest to it in
structure is flattered — which is exactly the disease exp27 exists to treat, and
it would be absurd to reintroduce it here.

So two referees, both fitted on the **full** race (the referee scores; it does not
compete):

- **Referee A — median per-run Theil–Sen slope** of fuel-corrected lap time on
  tyre age, taken within each run so the run intercept and driver pace cancel,
  then the median across runs of that compound. Robust, model-free, and **not a
  rung on the ladder**. It is structurally closest to the *fuel-corrected*
  baseline, which is a competitor and not us.
- **Referee B — the race-fitted TyreMind rate.** Structurally closest to *us*.

**Referee A is the headline.** Referee B is reported as a sensitivity check. If
our advantage exists only under Referee B, that is circularity and the result
file and the report must say so in those words.

Referee assumptions, identical for every model and both referees: no compound
pace offset (`base_pace_s = 0`), the simulator's default cliff at 25 laps with
severity 0.004, and the default 21 s pit loss. These are held fixed so the
counterfactual world differs between models only through the lap they chose.

## 5. The models

`tyremind.models.literature.extended_ladder()` is fitted; the six rungs that
expose a degradation parameter are scored: Naive, Fuel-corrected, Pooled,
TyreMind, Heilmeier, Cappello & Hoegh. ARIMA, LightGBM and the MLP return an
empty `compound_rates()` by construction and are **reported as having no
degradation parameter**, the finding exp19 already recorded. They cannot
recommend a pit lap at all, which is itself the point of including them in the
ladder.

A model that declines to answer (`PitRecommendation.reason` non-empty — a flat
cost curve, or a rate indistinguishable from zero) is recorded as declining. Its
answer rate is reported beside its value, and the headline comparison is computed
on the stops **every** model answered, for the same reason exp22 uses a like-for-like
subset: declining is legitimate, being graded on an easier subset is not.

## 6. The data

The 12 most recent races in `data/season/` by `tyremind.data.corpus.sessions`
ordering — the same set exp19 Leg A and exp22 use, so the three results are
readable against each other. Loading via `read_lap_table`/`load_frames`.

Stops excluded by `tyremind.models.pit_decision.excluded_stops`: stops at or
before lap 3 (damage, not wear) and stops inside a safety-car window (nearly free,
so taken whatever the tyre is doing). Exclusion counts are reported.

## 7. What counts as a win, and what counts as a loss

- **The headline claim stands** only if, under Referee A and on the final stops
  both models answered, TyreMind's mean `value_s` is **greater than the naive
  model's by more than one standard error of the paired difference**.
- **If the paired difference is within one standard error, it is a tie**, and the
  sentence becomes "using the naive estimate instead of ours would have cost
  nothing measurable" — which, given exp14 found the naive method produces a
  physically impossible negative rate in 74% of races, would itself be a finding
  worth reporting.
- **If TyreMind's mean `value_s` is negative**, following our recommendation would
  have been slower than what the teams actually did, and we report that sentence
  first, before any other result in this experiment.
- **If the ranking flips between Referee A and Referee B**, no claim is made from
  either, and the disagreement is the finding.
- **If fewer than 20 final stops are answered by both models**, the experiment is
  reported as underpowered and no claim is made.

## 8. Known limitations — stated up front

1. **The lap a team chose is not the optimal lap.** It is what a professional
   strategist decided with information we do not have — a damaged floor, a rival's
   undercut, a tyre allocation. "Value against what happened" is not "value
   against the optimum", and it can be negative for good reasons.
2. **The simulator prices one stop and runs to the flag.** Mitigated by scoring
   only final stops in the headline (Section 3) and by `max_remaining_stops=1`,
   but not removed: the absolute expected times are still not race times, only the
   difference between two candidate laps is meaningful.
   Restricting to final stops also uses knowledge of the whole race to decide
   *which* stops to score. That is a scoring choice, applied identically to every
   model, and it never reaches any model's fit.
3. **Track position is not simulated.** The simulator has a dirty-air term but no
   overtaking model, so an undercut that gains a place by clearing traffic is not
   represented. Positions are derived from seconds, not raced for.
4. **The referee is an estimate.** Section 4 is the mitigation, not a cure.
5. **Safety cars are sampled, not replayed.** The simulator draws them at a fixed
   per-lap rate; the actual safety cars of that race are not reproduced. Stops
   *inside* a real safety-car window are excluded for exactly this reason.

## 9. Fixed before running

- Metrics, directions, referees and win conditions: this document.
- Checkpoints: 15%, 30%, 50%, 70% of race distance.
- Simulation: 5000 sims per strategy, seed 7 for both candidates (common random
  numbers), the simulator's shipped defaults elsewhere.
- Race set: `limit=12`, `min_laps=200`, deterministic by corpus ordering.

Any deviation must be recorded in the result file as a deviation, with a reason.
