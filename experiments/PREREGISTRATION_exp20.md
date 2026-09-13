# Pre-registration — exp20, an independently *shaped* truth engine

**Written before the experiment was run. Committed before the results existed.**

---

## 1. The objection this experiment exists to answer

Our headline result is exp19 Leg C: TyreMind recovers a known degradation rate to
0.0037 s/lap, against 0.0062 for the next best rung and 0.0158 for the closest
published model (Cappello & Hoegh). It is measured on
`tyremind.data.synthetic` — a generator **we wrote**. The strongest thing a judge
can say about it is:

> You graded your own homework.

`PREREGISTRATION_exp19.md` §6.1 already concedes this and calls the mitigation
partial. exp29 swept the generator's *parameters* away from our assumptions and
the ranking survived, but §6.1 of that document concedes the remaining problem in
one sentence: sweeping a parameter does not escape the generator's **structural
form**. `synthetic` builds a lap time as

    lap_time = base + driver + rate(compound) * age + cliff + fuel*lap + track + traffic + noise

which is, term for term, the algebraic shape `tyremind.models.ssm.tyre_ssm`
assumes. A generator that shares the estimator's functional form cannot falsify
it.

The scoped fix was exp20: capture true tyre wear from a commercial F1 game's UDP
telemetry. That needs the game running and is not available here, and pretending
otherwise would be worse than not doing it.

**What this experiment does instead** is build a second truth engine —
`tyremind.data.physics_truth` — whose *data-generating form is not our
estimator's*, out of the physics this repository already implements from the
published literature (Archard and energy-dissipation wear, `physics/wear.py`; a
reduced two-state TRT thermal model, `physics/thermal.py`; load transfer and
frictional power, `physics/dynamics.py`; all cited in
`research/LITERATURE_REVIEW.md` §4–5).

## 2. What makes it independent, and what does not

**Independent in form.** In `physics_truth` the tyre's time loss is not a
parameter and is not written down anywhere. Lap time is the line integral
`∮ ds / v(s)` of a quasi-steady-state speed profile; grip enters through the
cornering limit; grip is a nonlinear function of accumulated wear; accumulated
wear is the time integral of `physics.wear.energy_wear_rate` over a lap's
frictional power and estimated tread temperature. The "true degradation rate" is
obtained by differencing the simulated clean lap times with respect to tyre age.
The estimator's functional form is therefore **not** the data-generating form,
and the test can fail.

Four specific mis-specifications our estimator faces here and does not face on
`synthetic`:

1. Lap time is **not additive** in its causes. Fuel, track evolution, traffic and
   wear all act through the same cornering-speed bottleneck and therefore
   interact. The generator records a `decomposition_residual` per lap measuring
   exactly how much the additive decomposition fails by.
2. **Fuel acts through mass**, not through a slope.
3. **Traffic acts through downforce loss**, not through a coefficient.
4. **Degradation bends** because grip falls nonlinearly in accumulated wear and
   because the thermal state carries across laps in a run — not because a cliff
   term was added to a lap-time equation.

**Not independent in authorship.** We wrote this generator too. This is **not**
validation against a third party's tyre model, and no result from it may be
presented as if it were. What it removes is the functional-form advantage. It
does not remove self-reference. §6 lists what remains.

## 3. The data

**Arm P — physics.** `tyremind.data.physics_truth.generate_session` at default
configuration, seeds `20260901 + i`.

**Arm S — synthetic.** `tyremind.data.synthetic.generate_session` at default
configuration, **the same seeds**, so the two generators are read side by side at
matched seed counts. Arm S is a re-measurement of exp19 Leg C, not a new claim.

No real telemetry is read. If a future version of this experiment reads a cached
lap table it must go through `tyremind.data.corpus.read_lap_table`.

## 4. The models

`tyremind.models.literature.extended_ladder()` — all nine rungs. The rungs that
expose a degradation parameter are scored: Naive, Fuel-corrected, Pooled,
TyreMind, Heilmeier, Cappello & Hoegh. ARIMA, LightGBM and the MLP return an
empty `compound_rates()` by construction and are reported as **having no
degradation parameter**, which is the exp19 finding, not a gap. They are detected
by introspection rather than by a hard-coded name list, so a rung that later
gains a rate is picked up automatically.

No model's priors are retuned for either arm, including ours.

## 5. The metrics, their directions, and what counts as a LOSS

Scoring is `tyremind.models.evaluation.score_rate_recovery`, called directly, so
the numbers are computed by the same function exp19 used.

### Truth targets

When degradation is nonlinear there is no single number that is "the true
degradation rate", and the rungs do not all report the same quantity: TyreMind
and Cappello & Hoegh report a *baseline* rate, Heilmeier reports a *linear
slope*, the regression rungs report an OLS slope. Three targets are therefore
scored, **computed by the same four lines of arithmetic on both arms** from each
generator's own per-lap `true_tyre` and `true_rate` columns:

| Target | Definition | The rung it is the fair target for |
|---|---|---|
| `fresh` | mean instantaneous rate over the first 5 laps of tyre age | a model whose parameter means "baseline" |
| **`mean_instantaneous`** (**primary**) | lap-weighted mean instantaneous rate over the laps actually run | "what the tyre cost per lap on average" |
| `ols_slope` | least-squares slope of true cumulative tyre loss on tyre age | a model reporting a straight-line slope |

On Arm S these three nearly coincide, because degradation there is a declared
constant with a mild cliff bolted on; `fresh` should reproduce
`GroundTruth.compound_rates` — the number exp19 Leg C scored against — to within
rounding, and the declared value is recorded in the result file so that link
stays auditable. On Arm P they do not coincide, and the gap between them is a
property of the simulated tyre rather than of the experiment.

Arm P has no declared baseline rate to score against, because no baseline rate
exists in it. That is the point of the arm.

**If the ranking differs between targets, all three rankings are reported and
none is quoted alone.** See §8 — this clause is an amendment.

### Metrics

| Metric | Direction |
|---|---|
| **Rate MAE** against the truth target | lower better |
| Rate bias | closer to 0 better |
| Coverage of the 95% interval on the rate | **closer to 0.95 better**, not higher-better |
| Paired margin over the runner-up, with its standard error | reported next to every ranking |

Every comparison is **paired on `(seed, compound)`**: the same generated session
goes to every model. Between-seed variance dwarfs the between-model difference —
exp29 recorded a four-seed pilot flipping a ranking that eight seeds reversed —
so an unpaired comparison would be reporting noise.

**A win requires beating the runner-up by more than one standard error of the
paired difference.** Anything less is recorded as a **TIE**, and a tie is a tie.

### What counts as a loss — declared now

- **The exp19 Leg C claim survives** only if, on Arm P, TyreMind has the lowest
  rate MAE **and** beats the runner-up by more than one paired standard error.
- **If TyreMind is not first on Arm P**, the finding is that our exp19 margin was
  substantially an artefact of a generator shaped like our estimator. That is
  reported as the headline of this experiment, exp19 Leg C is re-presented with
  the caveat attached, and the name of the model that beat us is in the summary
  table. This sentence is written here so that outcome stays reportable.
- **If TyreMind is first but the margin is inside one standard error**, we say
  "comparable", not "better".
- **The survival ratio** `paired_margin(Arm P) / paired_margin(Arm S)` is reported
  whether it is above or below 1. A ratio below 1 means part of our published
  advantage came from the generator's shape.

### Pre-declared expectations

Declared in advance so a shrink cannot later be presented as a win:

- **We expect the margin to shrink on Arm P.** Our estimator is additive and
  linear-in-age; Arm P is neither. If the margin does not shrink, that is a
  surprise and gets said as one.
- **We expect every model's absolute MAE to be worse on Arm P than on Arm S**,
  ours included, because every rung in the ladder is additive.
- **We expect coverage to degrade on Arm P**, because a mis-specified mean is not
  repaired by a correctly propagated variance.

## 6. Known limitations — stated up front

1. **We wrote this generator.** It removes the functional-form advantage. It does
   not remove self-reference, and it is not third-party validation. Any
   presentation of these results that drops this sentence is misrepresenting
   them.
2. **Two constants are calibrated, not derived.** `thermal_power_scale` supplies
   the unknown constant that `frictional_power_proxy` declares it is missing;
   `reference_life_laps` puts a scale on `physics.wear`'s arbitrary units. They
   fix the *scale* of degradation, not its shape, not its compound ordering (that
   comes from `wear.COMPOUND_WEAR_FACTOR`), and not the quantity being scored,
   which is a derivative of a simulated lap-time curve. Setting a tyre life is a
   weaker assumption than setting a rate. It is still an assumption.
3. **The grip law `mu(W)/mu_0 = 1 - aW - bW^3` is declared, not fitted.** No
   public data exists to fit it against. A different exponent would move the
   emergent rates.
4. **The circuit is invented**, and the quasi-steady-state driver is on the
   friction limit everywhere it can be. Neither is a claim about a real venue or
   a real driver.
5. **`fuel_burn_kg_per_lap` is set so the emergent fuel gain lands near the
   0.081 s/lap the ladder hard-codes.** This is a concession *against* our
   interest: three rungs assume that constant and would otherwise be penalised
   for a mis-specified number rather than for their structure. The emergent value
   is reported in the result file so the concession is auditable.
6. **Temperature affects wear but not grip** in this generator. Adding a thermal
   grip term would create a confounder no rung in the ladder can see, which would
   measure something other than functional form.
7. **One configuration.** exp29 swept eleven regimes of the other generator;
   this sweeps none of this one. Interactions between arm-P structure and
   arm-P parameters are not measured.
8. **The identification of degradation against fuel is weak on both arms.** A
   design with per-run intercepts makes `(every compound slope += δ, fuel
   slope += δ)` an exactly null direction, and an oracle regression with run
   intercepts blows up on *both* generators. What pins it is either the absence
   of run intercepts (the regression rungs) or an informative fuel prior (ours,
   0.081 ± 0.016 s/lap). Our estimator therefore has a real advantage here that
   is about priors rather than about functional form, and any margin that shrinks
   when the fuel constant is wrong should be read that way. This is why §6.5
   matches the emergent fuel gain to the constant the other rungs hard-code.

## 7. Fixed before running

- Generators, models, metrics, directions, win conditions, expectations: this
  document.
- Seeds: `20260901 + i`, identical across both arms and every model.
- Truth targets: as tabulated in §5.
- Tie rule: within one standard error of the paired difference.
- Scoring function: `tyremind.models.evaluation.score_rate_recovery`.

Any deviation must be recorded in the result file as a deviation, with a reason.

## 8. Amendments — recorded rather than hidden

**Amendment 1, made after a two-seed pilot and before the full run.** The
original §5 named one primary target (`mean_instantaneous`) and one sensitivity
target (`ols_slope`). The pilot showed that the *ranking flips between them*:
TyreMind led on the primary target and Heilmeier led on the sensitivity target,
because the emergent degradation on Arm P is convex and a straight-line slope is
a different quantity from an average instantaneous rate.

Having seen that, the tempting move is to pick the target we win on. The
amendment does the opposite:

- A third target, `fresh`, is added, so all three natural readings of "the
  degradation rate" are on the record.
- **The primary target is unchanged** (`mean_instantaneous`), so the headline
  cannot have been chosen after the fact.
- A rule is added that if the ranking differs between targets, all three
  rankings are reported and none is quoted alone.

The pilot's numbers are not carried into the result file; the full run
re-measures everything. This amendment is declared here, in the result file's
`targets` block, and in the experiment's docstring.
