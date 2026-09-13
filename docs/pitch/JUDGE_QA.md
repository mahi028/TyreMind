# TyreMind — Judge Q&A

**The 25 hardest questions, and the answers we can defend with a file.**

Rules for whoever is answering:

1. **Concede first, then explain.** Every one of these has a real concession in it.
   Lead with the concession. You get believed on the second half only if you gave
   away the first half.
2. **Name the file.** Every number below points at something in
   `experiments/results/`. If you cannot name the file, do not say the number.
3. **"We have not tested that yet. It is on our list."** is a complete answer and
   costs nothing. A guess costs everything.

Companions: `docs/pitch/STORY.md`, `docs/pitch/DEMO_SCRIPT.md`,
`research/HEAD_TO_HEAD.md`.

---

# A. "You lose"

## Q1. Why not just use the pooled regression? It beats you.

**It does beat us, at forecasting.** `exp19_field_comparison.json`, 12 real 2025
races: pooled regression CRPS **0.4088**, ours **0.6484**. We are **4th of 9**.

**And we wrote down that we expected to lose before we ran it.**
`experiments/PREREGISTRATION_exp19.md`, committed before the results existed:

> *"We claim a win on Leg A only if TyreMind has the lowest CRPS. **We expect not
> to.** … If that holds, we report it as the headline of Leg A and explain the
> two-task distinction — we do not bury it."*

**Now ask what the pooled regression is for.** `exp22_pit_stop_validation.json` —
274 real pit stops, scored against what professional strategists actually did. On
the 49 stops every model answered:

| Model | Error vs the real call (laps) |
|---|---|
| **TyreMind** | **5.92** ± 0.52 |
| Cappello & Hoegh | 5.98 ± 0.52 |
| Fuel-corrected regression | 6.31 ± 0.72 |
| Heilmeier additive | 7.55 ± 0.65 |
| Naive | 10.86 ± 0.67 |
| **Pooled regression** | **11.12** ± 0.70 ← **last** |

**The best forecaster is the worst decider, by 5.2 laps.**

Why: a pooled regression is good precisely *because* it soaks fuel, track, tyre,
driver and traffic into one flexible fit. That is what makes it unable to itemise a
single one of them. It predicts the total on the bill and cannot tell you what any
line item was.

**And this is not our idea.** Pitwall (arXiv:2607.06495), an independently published
production system, reports the same phenomenon and names it *"calibration-optimal is
not decision-optimal"*. We replicated it in a different setting and pre-registered
the prediction.

> **The line:** *"If you want a lap-time forecaster, use the pooled regression and we
> will help you set it up. Problem 3 asks for the tyre component isolated from the
> confounders. That is a different job, and on that job the pooled regression comes
> last."*

---

## Q2. So you are 4th on forecasting and tied on pit stops. What do you actually win?

**Three things, and we will name the tie first.**

**The tie:** we are 5.92 laps and Cappello & Hoegh are 5.98, with standard errors of
0.52. That is one-tenth of a standard error apart. Our own pre-registration says:

> *"A tie is a tie. If TyreMind and Cappello & Hoegh are within one standard error on
> a metric, we say 'comparable', not 'better'."*

So we say comparable. Fuel-corrected regression at 6.31 makes it a three-way tie.

**What we actually win:**

1. **Degradation-rate recovery — the quantity the problem statement asks for.**
   `exp19` Leg C: ours **0.0037 s/lap**, best of nine. Closest published model
   (Cappello & Hoegh) **0.0158** — we are **4.3× better**. Three of the nine models
   (ARIMA, LightGBM, MLP) have **no degradation parameter at all** and cannot answer
   the question in any form.
2. **Calibration.** On that same leg our 95% interval covers the truth **100%** of
   the time; theirs **37.5%**. Live, `exp13`: **95.2% over 69,206 laps**, 0.2
   percentage points off target. Nobody else in this field measures this.
3. **The identifiability bound.** `exp18`: 6.0% of the fuel/tyre separation is
   data-driven, we sit at 1.38× the Cramér–Rao floor, and **no paper in our
   28-paper review states this number.**

---

## Q3. Your pit-stop hit rate is 24%. A published Bi-LSTM reports F1 0.81. You lose badly.

**Correct, and that number is in our own result file because we put it there.**
`exp22_pit_stop_validation.json`, field `literature_benchmark`: Sasikumar et al.,
Frontiers in AI 2025, Bi-LSTM with SMOTE balancing — precision 0.77, recall 0.86,
F1 0.81. Our hit rate within ±2 laps is **24.0%**.

**They are not solving the same problem.** They train a supervised classifier
directly on pit labels. We run an untrained expected-cost optimisation over a
calibrated degradation distribution and **never see a pit label at all**. Their model
learns what strategists do. Ours computes what the tyre implies and is then scored
against what strategists did.

**Two consequences we accept:**

- A classifier trained on the outcome should win an outcome-matching metric. It does.
- Our number is a *ceiling statement*, not a benchmark win: it tells you how far an
  untrained physical argument gets you toward a professional's call.

**What our table is actually for** is the *relative* ranking under identical scoring
— and there, the model that forecasts best comes last. That comparison is
apples-to-apples because every rung sees the same data and the same cost function.

---

## Q4. You lose on NASA C-MAPSS too. Why is that slide even in the deck?

`exp07_cross_domain.json`: RMSE **22.67** cycles on all 100 FD001 test engines.
Published deep models reach **15.98** (CNN-LSTM-Attention), 16.22, 17.60. **We are
30–40% worse on point accuracy.**

**It is in the deck because the claim is transfer, not state of the art.** Those
models are purpose-built for turbofans, trained on turbofans, tuned on turbofans.
Ours is an **F1 tyre model with not one line changed**. Tyre stint becomes engine
life, tyre age becomes flight cycles. It should not work at all.

**Two details worth more than the RMSE:**

- **44% of predictions are early** — the conservative side for maintenance. On a
  maintenance problem, the direction of the error is a design property, not an
  accident.
- The estimated degradation rate came out at **0.00202** against a prior mean of
  **0.004**. The data moved the estimate by a **factor of two, away from the prior**.
  That is the cleanest available answer to "your prior is doing the work".

**And what is genuinely competitive is the uncertainty.** The 2025
uncertainty-aware Inception-BiLSTM paper presents 93.5–95.2% interval coverage as a
*new contribution*, described as *"previously unattainable in CMAPSS literature"*.
Our conformal layer reaches **95.2% over 69,206 laps** with a distribution-free
guarantee rather than a learned aleatoric head.

---

# B. "Your evidence is weak"

## Q5. Your synthetic data is your own generator. Isn't your headline result circular?

**Partly, yes. We wrote that down before we ran the experiment.**
`experiments/PREREGISTRATION_exp19.md` §6, "Known biases in this design — stated up
front":

> *"**Simulator bias on Leg C.** Our synthetic generator shares structural
> assumptions with our estimator, which advantages us. Partially mitigated:
> Student-t(5) observation noise where our estimator assumes Gaussian, and 25%
> scrubbed sets. **Not fully mitigated.**"*

**What we did to blunt it:** the generator emits Student-t(5) noise while our
estimator assumes Gaussian — so the estimator is deliberately given the wrong noise
model — and 25% of sets are scrubbed. Ground truth is never shown to any model.

**What we did not do:** author an independent truth engine. It is scoped as exp20
and it is not done. Until it exists, **Leg C is suggestive, not decisive**, and we
label it that way on every slide.

**Now the part that survives the objection entirely.** Our strongest claim about real
data uses **no synthetic data and no ground truth**: `exp14_naive_failure_rate.json`,
77 real races — the standard method reports at least one compound getting *faster*
as it wore in **74.0%** of them, and **53.4%** of 208 individual compound-stints.
You do not need the right answer to know that answer is impossible.

And `exp22` — 274 real pit stops against real strategists — has no synthetic
component at all.

> **The line:** *"Strike the synthetic leg entirely and we still have a
> falsification on 77 real races and a decision test on 274 real pit stops."*

---

## Q6. Your practice-to-race result compares your model to your own model.

**Yes. It is circular and we say so in the document before anyone asks.**

`exp03_practice_to_race.json`, 42 events, 94 compound comparisons: TyreMind MAE
**0.0807** vs naive **0.1471** — 45% tighter, with a systematic bias of **+0.0268**
that we report rather than tune away.

**The "actual" it scores against is our own model fitted to the race session.** There
is no measured race degradation rate in existence. So this is a **consistency test,
not a ground-truth test**, and we never present the 45% as validation.

**What it is still worth:** a model producing nonsense in practice would not agree
with itself on Sunday, and the naive method demonstrably does not agree with itself.
But it flatters us, because the reference shares our assumptions.

**The fix is pre-registered, and the first run is already in the repo — going against
us.** `experiments/PREREGISTRATION_exp27.md` fixes the metric before any number was
visible: score every model against **its own** race-derived estimate, and make the
primary score a **skill score against each model's own climatology**, so that a model
returning a constant rate scores exactly zero and cannot win by being uninformative.

`exp27_decircularised_practice_to_race.json`, first partial run — **2 events, 24
comparisons, 4 scored per model**:

- **Every model scores negative skill.** Ours is **−4.81** (SE **2.89**).
- `win_claimed: false`, `practice_uninformative: true`.
- We rank **3rd of 6** de-circularised, having ranked 4th on the circular metric — so
  the circularity was **not** flattering us on rank, which is worth knowing.

**Three things to say about it:**

1. It is **our** experiment, pre-registered with its losing condition written down in
   advance, and we ran it knowing it could go against us.
2. It is **2 events against a pre-registered 49**, with a standard error larger than
   half the point estimate. Nothing is concluded yet in either direction.
3. **If it holds at full scale, the finding is that Friday does not predict Sunday for
   anyone** — not that TyreMind fails. That reframes what a practice-derived number is
   for: a **within-session decomposition**, which is exactly what Problem 3 asks for,
   rather than a cross-session forecast.

**What it does not touch:** exp14, exp18, exp26, exp19, exp22, exp12, exp13. None of
those depend on practice-to-race transfer.

> **The line:** *"It shows we agree with ourselves. It is not proof, we would not let
> you quote it as proof, and the de-circularised replacement is in the repo and
> currently negative for every model in the ladder including ours."*

---

## Q7. There is no ground truth anywhere. How do you know any of your numbers are right?

**We don't, on the magnitude. We know three things without it.**

Measured tyre wear has never been public. Pirelli measures it, the FIA and the teams
see it, nobody else does. We searched Pirelli, the FIA, Kaggle, HuggingFace, OpenF1
and Ergast (`docs/data_doc.md` §3.1).

**Route 1 — impossibility.** You do not need the right answer to identify a wrong
one. Rubber does not regenerate. `exp14`: 74.0% of 77 real races. No ground truth
required.

**Route 2 — build a world where truth exists, then hide it.** `exp19` Leg C. Caveated
under Q5.

**Route 3 — go where truth is published.** `exp07`, NASA C-MAPSS, run-to-failure
labels, unmodified code. Caveated under Q4.

**Route 4 — score against real human decisions.** `exp22`, 274 real pit stops. The
reference is what a professional strategist did on the day.

No single route is sufficient. The four fail in different directions, which is the
point of having four.

---

## Q8. Isn't the naive baseline a strawman you built so you could knock it down?

**The best evidence that it is not comes from someone else's paper.**

Cappello & Hoegh (arXiv:2512.00640, Nov 2025) — the closest published prior art, our
exact model class — put a **half-normal prior on the degradation rate to force it
positive**, on the stated grounds that *"a negative overall degradation rate would
be"* implausible.

**They only need that prior because the unconstrained estimate goes negative.** A
published team hit the same failure and patched it with an assumption. We
**measured** how often it happens instead: 74.0% of races, 53.4% of stints (`exp14`).
A prior hides the symptom; an experiment quantifies it.

**Second answer:** the naive fit is not the only baseline we beat. `exp19` runs
**nine** rungs including the field-standard Heilmeier additive model, ARIMA(2,1,2)
(Cappello & Hoegh's own chosen baseline), LightGBM, an MLP, and Cappello & Hoegh
themselves. The models were fixed in the pre-registration and none could be added or
removed after results were seen.

**Third answer, and we volunteer it:** the naive fit *does* get the sign right
sometimes — Barcelona, in our demo set. That is not the method working. That is the
confounders happening not to swamp the signal there.

---

## Q9. Your coverage is 100% when you designed for 95%. That is miscalibrated too.

**Correct, and our own scoring rules say so.** `PREREGISTRATION_exp19.md` fixes the
direction of every metric in advance and explicitly states:

> *"**Coverage of the 95% interval** — **closer to 0.95 better**. *Not*
> higher-is-better. An interval covering 100% is as miscalibrated as one covering
> 61%."*

**The honest reading of 24/24.** With 24 estimates, a perfect score gives an exact
one-sided 95% lower bound of **0.883**. So the data is consistent with a true
coverage anywhere from about 88% upward. It does not prove 95%, and we would not
claim it does.

**And notice that coverage alone does not rank the table** — Heilmeier's additive
model also hits 100% coverage on the same leg while being **3× less accurate**
(0.0112 vs our 0.0037). Coverage is a constraint, not a score.

**The calibration claim rests on the large-sample number, not this one.** `exp13`:
adaptive conformal, **95.2% over 69,206 scored laps**, 0.2 percentage points off
target. `exp12`: **94.7%** offline over 94 comparisons. Those are the numbers to
quote.

**And we found our own miscalibration first.** Our model's native Gaussian interval
claimed 95% and delivered **75.5%**. `exp16` goes further and diagnoses the *shape*:
Gaussian intervals sit 0.189 from the calibration diagonal, adaptive conformal at
**0.005**.

---

## Q10. Twelve races and twenty-four estimates is a small sample.

**For the benchmark, yes. For the corpus, no.**

The corpus is **203 sessions, 91,867 clean laps, 84 event-seasons, 2022–2025**, all
downloaded by us from the official timing feed.

Sample sizes scale with what each experiment costs:

| Experiment | n |
|---|---|
| Live interval calibration (`exp13`) | 20 sessions, **69,206 laps** |
| Degradation regimes (`exp17`) | **2,827 stints**, 77 races |
| Telemetry energy (`exp25`) | **2,219 stints**, 24 circuits, 84 sessions |
| Driver effects (`exp15`) | **1,458 driver-races**, 31 drivers |
| Sector identifiability (`exp26`) | **913 stints**, 40 races, 25 drivers |
| Naive failure rate (`exp14`) | 77 races, 208 compound-stints |
| Circuit transfer, LOCO (`exp09`) | 193 stints, **26 circuits** |
| Pit-stop validation (`exp22`) | **274 real stops**, 14 sessions |
| Identifiability bound (`exp18`) | **520 runs**, 11 sessions |
| Field comparison (`exp19`) | 12 races + 8 synthetic seeds |

`exp19` is small because each rung fits nine models with rolling-origin folds. We
report its standard errors and we call ties ties.

**For scale context:** the closest published prior art fits **one driver in one
race**.

---

# C. "The method"

## Q11. Why not deep learning?

**We tested it. It came last of nine.**

`exp19`, lap-time forecasting on 12 real races: the neural network (MLP) scored CRPS
**1.9442** — worst in the table by a factor of three — and took **37.4 seconds** to
fit against the pooled regression's 0.02 s.

**And LightGBM is the sharper cautionary tale.** It forecasts well — CRPS 0.4826,
second place — and its 95% interval covers **62.0%** of the time. It is confidently
wrong by 33 percentage points. `exp16` shows why: five of six models are **U-shaped**
on the PIT histogram, which is plain over-confidence.

**But the real reason is structural, not a leaderboard.** Look at the
`models_without_degradation_parameter` field in `exp19_field_comparison.json`:
**ARIMA, LightGBM and the neural network are in it.** They forecast the lap time and
have **no term that corresponds to tyre degradation at all**. They cannot answer
Problem 3 in any format, at any accuracy. You cannot ask a black box "how much of
that lap was the tyre" and get a number it is committed to.

The literature says the same thing. Cappello & Hoegh §1: deep-learning methods
*"often lack interpretability and explicit uncertainty quantification — features that
are crucial in operational race environments."*

**Three reasons for state-space, positively stated:**

1. Tyre condition is a **latent state**. A Kalman filter is the standard tool for
   tracking something invisible through noisy observations.
2. It produces uncertainty as a first-class output, not an afterthought.
3. It **explains itself** — it says how much was fuel and how much was tyre, which
   is the entire deliverable.

**And the economics.** 12.66 s per session on a laptop, 2.6 MB for four seasons, no
GPU, no training run, no cloud bill, runs fully offline.

---

## Q12. Why a Kalman filter and not MCMC or full Bayesian inference?

**One operational reason and one honest concession.**

**The reason: a sampler has no incremental mode.** On a pit wall the estimate must
update every lap, forward-only, with no access to the future. Our online update runs
in well under a millisecond and the cost is **flat in session length** — lap 60 costs
what lap 1 cost. You cannot re-run an MCMC chain every lap during a race.

**The concession:** the Cappello & Hoegh rung in our benchmark is fitted **MAP, not
MCMC** — same model, different inference. That is declared in the pre-registration as
a known bias, with the consequence spelled out in advance:

> *"Where the gap between us and them is within one standard error, the inference
> difference is a plausible explanation and we will say so."*

On pit-stop timing the gap **is** within one standard error (5.92 vs 5.98, SE 0.52),
so we say it: that is a tie, and inference choice is a plausible part of the
explanation.

**What we get instead of sampling:** distribution-free calibration. Conformal
prediction gives a finite-sample coverage guarantee **without** assuming the model is
correct. Bayesian credible intervals are exactly right *if the model is right*. Ours
are right because we measured them against outcomes — 95.2% over 69,206 laps.

---

## Q13. If only 6% of the separation is data-driven, are you not just reporting your prior?

**This is the best question on the list, and it has three answers.**

**First — the honest restatement.** `exp18_identifiability_bound.json` is us proving
a limit *on ourselves*. Within a stint, tyre age and laps of fuel burned advance one
for one. 520 runs out of 520 exactly collinear, correlation **1.000**, Fisher
information singular in **11 of 11** sessions. The only escape is curvature — a tyre
curve bends, fuel burn-off never does — and that channel carries **6.0%** of the
information. So yes: **94% of the level comes from physics.** That is a property of
the problem, not of our model. **Every other system in this field has exactly the
same 6%. They just have not measured it.**

**Second — we tested whether the prior determines the answer.**

- `exp02_prior_sensitivity.json`: 8 synthetic configurations × 6 prior variants, plus
  5 real sessions. Fuel prior moved ±1 standard deviation, track prior ±1 sd, and a
  deliberately wide-prior setting. **The degradation estimate is stable across all of
  them.** The priors regularise the fit; they do not determine it.
- `exp07`, NASA: prior mean 0.004, data-driven estimate **0.00202**. The data moved
  it by a **factor of two, away from the prior.** A prior that was driving the answer
  could not do that.

**Third — and this is new — part of the bound is escapable, and we found the
escape.** `exp26_sector_identifiability.json`, 913 stints, 40 races, 25 drivers.

A lap is timed in **three sectors**. The three sectors load the tyre differently while
the fuel mass is identical across all three. So sectors give three equations where
lap time gives one.

- All three sectors **depart significantly from time-proportionality** (S1
  p = 5.5e-10, S2 p = 8.8e-4, S3 p = 0.033). Degradation has a *shape* across the
  lap; it is not the lap time scaled.
- That shape is **stable, not noise**: split-half correlation 0.638 over 14,000
  splits, 77.0% sign agreement, permutation p = 0.0 against a null of −0.007.
- **The direction of tyre loading is identified from data alone, with no prior**
  (median rank-one fraction 0.870).
- **The overall level still is not** — three equations, four unknowns after
  normalisation.

> **The line:** *"The 6% bound applies to the level. The shape escapes it. We found a
> crack in our own impossibility result and we are reporting both halves."*

---

## Q14. Why not use richer data — sector times, telemetry — to break the collinearity?

**We did both. One worked partially, one failed, and we report both.**

**Sector times — partially worked.** Q13, third answer. `exp26`. Direction
identified without a prior; level still not.

**Telemetry energy — failed, twice, at two different scales.** The physically
appealing idea is that tyres wear from *energy*, not from laps, so accumulated energy
should be the better clock.

- `exp04_energy_clock.json`, 2024 Interlagos, 4 stints: the energy clock won 1 stint
  of 4, mean R² gain **−0.0008**. Recorded verdict: *"no meaningful difference"*. Why:
  energy per lap has a coefficient of variation of only **2.3%**, so "energy so far"
  and "laps so far" are nearly the same number.
- `exp25_telemetry_energy.json` redid it properly at scale — **2,219 stints, 24
  circuits, 84 sessions**, with real telemetry: total frictional energy ρ = −0.040,
  p = 0.060. `supported: false`. Even the significant correlates point the **wrong
  way** (loaded fraction ρ = −0.153 — *more* loading associated with *less*
  degradation), which is a sign of confounding by circuit, not a mechanism.

**Circuit geometry — failed.** `exp09_circuit_transfer.json`, leave-one-circuit-out
over 26 circuits: geometry features −3.3%, p = 0.222, no detectable effect. Thermal
features **actively harmful**, p = 0.0012.

**What we would use if we had it:** team-internal channels — tyre surface and carcass
temperature, pressures, load cells, slip angles. That is the Mercedes/Sulsters data
asymmetry, and it is a **data gap, not a method gap**. See `docs/data_doc.md` §2.3.

---

## Q15. Your intervals are 2.3× wider than the alternative. Why is that a good thing?

**Because the narrow one was lying.**

`exp12_conformal_intervals.json`, 94 comparisons across 42 events:

| Interval | Claimed | Delivered | Median half-width |
|---|---|---|---|
| Our model's own Gaussian | 95% | **75.5%** | ±0.119 |
| **Absolute conformal (shipped)** | 95% | **94.7%** | ±0.276 |

The narrow interval is not more precise. It is **wrong one time in four while
claiming to be wrong one time in twenty**.

**Say it as a decision, not a statistic.** A strategist told "±0.12, and it is right
95% of the time" makes a pit call at the edge of that band and it silently fails one
time in four. They will never know why. A strategist told "±0.28, and that really is
95%" can compute the cost of being wrong — which is their actual job.

**And this is where the entire product positioning lives.** `docs/PRODUCT_PLAN.md`:
the primary user is the **race strategist**, because calibrated uncertainty only has
value to someone who makes decisions under uncertainty. Give a driver an interval and
you have made their job harder. Give a strategist a point estimate and you have made
theirs impossible.

---

## Q16. You invented your own traffic index. Isn't that just a free parameter?

**It is derived, it is documented, and we tested whether it mattered — and it mostly
did not.**

`traffic_index` is derived by us from timing-loop crossings (`docs/data_doc.md`
§1.1.4, §2.2). It is a **proxy**, and the column is labelled as derived, not
measured, everywhere it appears — including on the dashboard, where the confounders
panel states for each term whether it is assumed or measured.

**We tested it as a driver of our practice-to-race bias.** `exp10_bias_mechanism.json`
ran nine candidate explanations with Benjamini–Hochberg false-discovery control, so
that looking nine times does not manufacture a finding. Traffic gap: ρ = **−0.167**,
p = 0.109. **It does not survive.** Only one of nine did — practice stint length.

**What it buys us anyway** is structural, not statistical: the closest published prior
art absorbs traffic into i.i.d. observation noise — their ε term explicitly covers
*"driver mistakes and the presence of other cars"*. Lumping a systematic confounder
into i.i.d. noise biases the degradation estimate. We carry it as an explicit
covariate even where it turns out small, because a term you can inspect is better
than a term hidden inside the residual.

**The honest next step, scoped:** replace the derived index with **measured** intervals
from OpenF1. Not yet done. We must not swap the column silently or the experiment is
destroyed.

---

## Q17. Wet weather, safety cars, red flags, out-laps — what do you exclude, and does it matter?

**Seven filters, and every session ships its own receipt.**

Each session writes a `.quality.json` alongside its data recording exactly what was
removed and why. For Monza 2024 we received 1,008 laps and kept **921**; we can show
where the other 87 went, by category. The dashboard renders this as a donut labelled
*"Laps thrown out before analysis, and why"*, and prints a data-quality score out of
100 on screen.

**Wet running is excluded corpus-wide**, so every model in the benchmark is evaluated
on the same restricted regime — that constraint is in the pre-registration, applied
equally to all nine rungs rather than tuned for us.

**We also name the sessions we excluded, in the result file, by name.** `exp03`
lists them: 2023 Australian GP (FP2 50% on wets), 2024 Canadian GP (race 67% wets),
2024 Japanese GP (FP2 46% wets), 2025 Australian GP (race 81% wets), 2025 British GP
(race 74% wets). Two more are recorded as **failures rather than silently dropped**:
2023 Canadian GP (tyre age *decreased* inside a run — a timing-feed inconsistency we
refuse to paper over) and 2024 Mexico City GP (no compound had 8+ laps in both FP2
and the race).

**Does it matter?** Yes, and here is the limit stated plainly: **we have not built a
wet regime.** It is scoped and not done. A wet race is currently out of scope for this
system, and we would tell a customer that before they bought it.

---

## Q18. What happens at a circuit you have never seen? Your own experiment says transfer fails.

**It does, and we published that as a negative result.**

`exp09_circuit_transfer.json`, leave-one-circuit-out over **26 circuits**, 193 stints:

| Feature family | MAE | vs baseline | p | Verdict |
|---|---|---|---|---|
| Label mean (baseline) | 0.0380 | — | — | — |
| Geometry only | 0.0393 | −3.3% | 0.222 | **No detectable effect** |
| Thermal only | 0.0386 | −1.5% | **0.0012** | **Actively hurts** |
| Everything | 0.0401 | −5.5% | 0.056 | No effect |

**Note the careful wording**: geometry is "no detectable effect", *not* "harmful" — at
p = 0.22 we cannot distinguish it from zero. An earlier draft of ours said "harmful"
and that was wrong; it is corrected in `docs/data_doc.md` §exp09. Thermal features
*are* significantly harmful, most likely because our only thermal input is track
temperature from a single weather mast, which is a poor proxy for what the tyre
surface experiences. **A noisy feature is worse than no feature.**

**So what do we actually do at a new circuit?** We fit it from its own session. The
model needs **one session at the track**, not a prior calibration from a different
track. A practice session is 12.66 seconds of compute.

**And this is the sharpest structural criticism we have of a single-circuit
calibration approach.** A five-parameter formula tuned on Bahrain has no evidence it
travels, and our LOCO test says circuit transfer genuinely fails. We are not asserting
that about anyone's system — we are reporting the measurement and letting it apply
where it applies.

**What we would need to do better:** track-surface characterisation — macro-texture,
aggregate type, resurfacing history. It is not published anywhere we could find
(`docs/data_doc.md` §3.6).

---

# D. "The competition"

## Q19. Cappello & Hoegh published your model class a month ago. What is left for you?

**Cite them first. Always.** arXiv:2512.00640, November 2025, Montana State. Bayesian
state-space model for F1 tyre degradation on FastF1 data. **Same model class,
published first.** We implemented their model as a benchmark rung rather than
describing it.

**We should also say what they do better than us:** their skewed-t observation model
is a better idea than our Gaussian — driver errors really are one-sided — and we
intend to adopt it as a variant. Their Bayesian workflow is clean and honest about
data poverty.

**Four things we have that they do not:**

| | Them | Us |
|---|---|---|
| **Scale** | One driver, one race (Hamilton, 2025 Austria) | 203 sessions, 91,867 laps, 84 event-seasons, 4 seasons — roughly 200× |
| **Positivity** | Imposed by a half-normal prior, because the unconstrained rate goes negative | **Measured**: `exp14`, 74.0% of 77 races, 53.4% of 208 stints |
| **Uncertainty** | Assumed — Bayesian credible intervals are correct *if the model is* | **Measured**: ours claimed 95% and delivered 75.5%; fixed to 94.7% offline and 95.2% over 69,206 laps live. Theirs covers **37.5%** on our recovery leg. |
| **Identifiability** | *"we lean on moderately strong priors since we are relatively data poor"* — never quantified | **Quantified**: only **6.0%** data-driven, 1.38× the Cramér–Rao floor. Their result rests on a prior doing **94%** of the work, and the paper does not say so **because it does not know**. |

That last row is our single biggest contribution over the closest prior art.

**And where we tie, we say tie:** pit-stop timing, 5.92 vs 5.98, one-tenth of a
standard error. Comparable.

---

## Q20. How is this different from the system the previous team built?

**Say this one carefully. Their metrics are good. We are keeping them. We are
changing what calculates them.**

We have not seen their code, so every claim below is a **structural** ceiling plus
**our own** measurement of its cost. No adjectives.

| Their design | The structural ceiling | Our measurement |
|---|---|---|
| Five-parameter formula, calibrated on **one circuit** (Bahrain V3) | No evidence a single-circuit calibration travels | `exp09`: LOCO over 26 circuits. Circuit transfer genuinely fails. Geometry p = 0.22; thermal actively harmful, p = 0.0012. |
| **Threshold** alerts on point estimates | A threshold with no interval is a coin flip near the line | `exp12`: our own model claimed 95%, delivered 75.5%. We only know because we checked. |
| **18 rules + a 17-step ladder** | Cannot say "I don't know". Cannot weigh the cost of being wrong. The supersede logic is the tell — rules collide, so rules arbitrate rules. | Replaced by expected-cost minimisation over a calibrated distribution — **and we keep all 18 rules as a veto layer**. When model and rules disagree, the engineer sees both. |
| No negative results | Untested features silently degrade a model | Eight published negatives, including `exp09` where adding a feature made predictions **significantly worse**. |

**Their five metrics, our versions:**

| Their metric | Ours |
|---|---|
| Grip Level — "real-time traction coefficient" | Normalised **grip index 0–1**. We **refuse to print a μ value**: true μ needs load-cell and slip channels that are team-private. |
| Tread Remaining — "wear rate & remaining depth" | **% of usable life, with an interval**, anchored to the measured cliff at 71.9% through a stint (`exp17`, 2,827 stints). **Never millimetres.** |
| Degradation Rate — 5-param Bahrain formula | State-space, 4 seasons, calibrated interval, **with the identifiability bound stated**. |
| Tyre Energy — "laps of useful life" | Conformal RUL — and the same code already does it on jet engines. |
| Puncture Risk — "threshold-based and probabilistic" | **A structural exposure index, explicitly not a probability.** See Q25. |

**The line for the last two rows:**

> *"They report remaining depth in millimetres and a traction coefficient. Neither is
> measurable from public data. We report percentage of life remaining and a normalised
> grip index — because we would rather be right than precise."*

**And their four "future opportunities":** live telemetry integration is
**substantially built** (95.2% over 69,206 laps, self-correcting live); full vehicle
health monitoring is **already demonstrated** (NASA turbofans, plus 151 vehicle
failure events collected — nine times the tyre-failure count, so that is where a
hazard model actually belongs); driver-in-the-loop and multi-compound simulation are
scoped, not built, and we say so.

---

## Q21. Pitwall already built the live strategy agent. Why do you exist?

**Pitwall is a better race-outcome simulator than anything we will build, and we say
so.** arXiv:2607.06495: N = 2,000 Monte Carlo continuations per lap, calibrated on 126
races, held out on 2025–2026, winner-in-top-3 **90.3%** over 155 backtests, held-out
Brier **0.0745**, ran live at two 2026 Grands Prix, trilingual briefings with a
claim-level verifier. **We do not compete on race-outcome prediction and we will not
pretend to.**

**Now read what their paper says about their own tyre layer.** They use a
compound-pace normalisation, and report that it

> *"demonstrably fixes pathological live pit calls **worsens** finishing order
> prediction, because the raw, **confounded** per-race fit carries genuine
> race-specific signal."*

**The word is theirs: confounded.** Their tyre layer is a per-race pace fit that mixes
tyre with everything else, and they route around it with two paths rather than solving
it.

> **The positioning:** *"Pitwall is a better race-outcome simulator than anything we
> will build, and it is honest that its tyre layer is confounded. We are not competing
> with Pitwall. **We are the component its own paper says it is missing.**"*

**And they validate our central methodological claim.** Their finding
*"calibration-optimal is not decision-optimal"* is the thing we replicated
independently and pre-registered. A third-party critique of the field standard is
worth far more than our own — including this, on the field-standard simulators:

> *"These simulators are calibrated to reproduce race **times**; **none report
> probability calibration of race outcomes**, none model the box-now-vs-later
> counterfactual under common random numbers, and none operate against a live timing
> feed."* — Pitwall §2

We have been asserting "nobody calibrates" for weeks. We can now cite someone else
saying it.

---

# E. "The business"

## Q22. Who pays for this, and what stops a team building it in a fortnight?

**Three markets, in ascending order of size.**

| Level | Customer | Value |
|---|---|---|
| **Team** | 10 F1 teams | One wrong pit call costs positions; positions cost millions. `exp22` puts the error in laps; at a 21-second pit loss, seconds follow directly. |
| **Series / broadcast** | F1, F2, F3, WEC, IMSA, Formula E | Insight graphics. **Pitwall already competes here — do not lead with it.** |
| **Industrial** | Fleet maintenance, wind, rail, mining | Domain-agnostic and already demonstrated on turbofans. The strategist persona becomes the maintenance planner. **The largest market, and nobody in our 28-paper review addresses it.** |

**Could a team build it in a fortnight?** They could build the estimator. Two things
take longer than a fortnight:

1. **Knowing what is impossible.** `exp18` and `exp26` are not implementation. They
   are the result of asking what is knowable *before* asking what is accurate. The
   closest published team has the identical model and does not know its prior is doing
   94% of the work.
2. **Eight negative results.** Every one of those is a fortnight a team would spend
   building something that does not work — the energy clock, C1–C5 compound identity,
   circuit-geometry transfer, thermal features, depth matching, driver style, puncture
   probability. We already spent it. Knowing which seven roads are dead is worth more
   than the eighth road.

**And the cost structure is the moat at the industrial end.** 12.66 s per session on a
laptop, 2.6 MB for four seasons, no GPU, no training run, **runs fully offline**. A
maintenance planner on a mine site with no connectivity can run this. A deep model
cannot.

---

## Q23. What happens when it is wrong, live?

**Four layers, in order.**

1. **The interval corrects itself during the race.** `exp13`: adaptive conformal
   inference updates its own α online — `α_{t+1} = α_t + γ(α − err_t)`, γ = 0.02. It
   lands at **95.2% over 69,206 laps**, 0.2 points from target. It does not need to be
   right at the start; it needs to notice and correct. It does.
2. **The screen shows its own live coverage while running.** No other system in our
   review displays this. A strategist can audit the tool mid-race.
3. **The rules keep a veto.** All 18 rules from the existing engine are retained as a
   safety layer alongside expected-cost minimisation. When the model and the rules
   disagree, **the engineer sees both**, and the system does not silently pick.
4. **The model is allowed to refuse.** Applicability falls off as a projection reaches
   past what the session contains, and the UI prints that column. "I don't know" is a
   valid output. An 18-rule expert system structurally cannot produce it.

**And the layer we would not ship without:** any generated sentence is decomposed into
typed factual claims and each is checked against the state object; an unverified claim
means the sentence is discarded for a provably faithful template. Pitwall's own
finding is the reason — the identical generator **fabricates drivers, gaps and tyre
compounds when the grounding state is sparse**, and replicating across four base
models showed this is a property of instruction adherence, **not of scale**. Read that
in our context: the agent hallucinates exactly when data is thin, which is early in a
stint, which is exactly when a strategist needs it. A bigger model does not fix it.
Only the verifier does.

---

## Q24. Why should anyone trust a system that admits it does not know 94% of the answer?

**Invert the question. Every system in this field has the same 6%. We are the only
one that knows it.**

The collinearity between tyre age and fuel burn is a property of the *physics of a
stint*, not of our model. Anyone fitting lap times inside a stint faces it. The
difference is that we measured it — 520 runs out of 520, correlation 1.000, singular
Fisher information in 11 of 11 sessions — and everyone else resolves it with an
unexamined assumption.

**Three reasons that admission is the product, not a weakness:**

1. **It is actionable.** We sit at **1.38× the Cramér–Rao floor**. That tells a buyer
   how much room is left above us — 38%, and no more. Nobody else in this field can
   tell you when to stop investing in a better estimator.
2. **It points at what to go and measure.** If 94% is physics, then the way to improve
   is better physics inputs, not more lap times. `exp26` is that argument executed:
   sector times recover the *shape* without a prior. The dashboard makes it a feature
   — "What would most reduce the uncertainty" ranks the top three signals by how much
   each would cut the error.
3. **It is the only claim here that is provable.** "Best in the world" is not
   checkable in this room. "Our 95% interval covered 95.2% of 69,206 laps" is.

> **The line:** *"We are not the only system that is 94% assumption. We are the only
> one that knows it, has measured it, and tells you which 6% is real."*

---

## Q25. Your puncture risk is not a probability. Why ship something weaker than what already exists?

**Because we went looking for the labels and they are not there.**

We ran the collection properly — `scripts/build_failure_labels.py`, the Pirelli era
2011–2022, **5,057 result rows**:

| Cause group | Events |
|---|---|
| Competing risks (Collision 190, Accident 102, Collision damage 45, …) | 351 |
| **Vehicle: Gearbox 54, Brakes 50, Suspension 37, Transmission 8, Driveshaft 2** | **151** |
| Wheel assembly (Wheel 16, Wheel nut 8) | 24 |
| **Tyre proper: Puncture 12, Tyre 5** | **17** |

Two further limits, both measured: **detailed retirement causes stop entirely after
2022** (from 2023 the public results carry only Finished / Lapped / Retired), and
**race control messages contain no punctures** — three 2024 races checked, **0
tyre-related messages out of 249**.

**Seventeen positive labels in twelve seasons, with competing risks twenty times more
common.** No validated puncture *probability* can be calibrated from public data — by
us or by anyone. So we ship a **structural exposure index**, labelled an index, never
a percentage.

**This is a strength, and it is also the sharpest respectful question available about
any competing system that ships a "probabilistic puncture risk score":**

> *"We went looking for the labels needed to calibrate a puncture probability. There
> are seventeen in twelve seasons. So we report exposure rather than probability — and
> we would genuinely like to know what a probability here is calibrated against."*

**And the payoff.** The same search found **151 vehicle failures** with workable
per-family sizes — nine times the tyre count. **The "full vehicle health monitoring"
item listed as future work is better supported by public data than the puncture metric
that already ships.** That is where the hazard model belongs, and the machinery is the
same.

---

# F. Two questions we want to be asked

## Q26. What failed?

**Say this proudly. Eight hypotheses refuted, several of them our own.**

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Energy through the tyre beats lap count as a clock | ✗ | `exp04`: 1 stint of 4, R² gain −0.0008. Energy per lap varies only 2.3% lap-to-lap. |
| …and it will hold up with real telemetry at scale | ✗ | `exp25`: 2,219 stints, 24 circuits. ρ = −0.040, p = 0.060. `supported: false`. |
| True Pirelli compound (C1–C5) beats the HARD/MEDIUM/SOFT label | ✗ | `exp08`: **6.4% worse**, won 22 of 58, Wilcoxon p = 0.096. We built the dataset by hand and the simpler feature won. |
| Circuit geometry transfers degradation to a new track | ✗ | `exp09`: LOCO over 26 circuits. Geometry p = 0.222. Thermal **actively harmful**, p = 0.0012. |
| Temperature explains our practice-to-race bias | ✗ | `exp10`: ρ = −0.022, **p = 0.835**. The explanation every fan offers first explains nothing. One of nine candidates survived BH correction. |
| **Matching stint depth fixes the bias — our own idea** | ✗ | `exp11`: bias +0.0268 → **+0.0286** (6.9% worse), MAE 0.0807 → 0.0918 (13.8% worse), paired t = −2.46, **p = 0.0159**. |
| Driver style is a usable feature | ✗ | `exp15`: 1,458 driver-races. Raw driver effects reverse sign in 5% of splits. Teammate contrasts *are* stable — but adding driver identity made prediction **0.88% worse**. |
| Puncture probability from public data | ✗ | Q25. 17 labels in 12 seasons. |

**Number six is the one to dwell on.** `exp10` found a correlation — shorter practice
stints, more bias. We derived a mechanism from it, designed an intervention that could
kill the mechanism, and it killed it, significantly. Correlation did not survive
contact with intervention.

**That is the difference between a dashboard and a research project.**

**One more, which is a negative inside a positive.** `exp17` established the tyre
cliff is real — 337 stints, median at 71.9% through a stint, severity +0.202 s/lap.
Then we asked whether fitting it improves forecasting: across 2,683 forecasts the
broken-stick model was **26.5% worse** than a straight line, helping only on warm-up
stints (+30.7%, p = 3e-18). So the cliff is worth **showing** a strategist and not
worth **fitting** for a forecast. We report both halves.

**And two mistakes we made and corrected, because someone will find them:**

- We first reported a cliff at **37%** through the stint. That was not a cliff — it
  was tyre **warm-up** pooled in with it, bending the curve the opposite way. Once
  warm-up was classified separately the cliff moved to its physically sensible place
  at ~72%.
- We first reported per-driver rates differing by ~0.00001 s/lap. That was not a fact
  about drivers — the driver-variance parameter had been pinned at its optimiser
  floor, so the model was **incapable** of expressing a driver difference. We were
  measuring our own code, not the sport. An implausibly tiny effect is usually a bug,
  not a finding.

---

## Q27. What would change your mind, and what is next?

**Four things, in priority order, each of which could go against us.**

1. **An independently authored truth engine (exp20).** Validate against a tyre model
   we did not write. This is the experiment most capable of embarrassing us, and it is
   first on the list for exactly that reason.
2. **Finish the de-circularised practice-to-race test (exp27).** Pre-registered, first
   partial run already in the repo, and **already going against us** — every model
   scores negative skill on 2 of the specified 49 events. Q6. Our 45% may not survive
   at all.
3. **Reproduce a single-circuit calibration and test whether it travels (exp21).** We
   claim it will not, on the strength of `exp09`. That claim is falsifiable and we
   should falsify it properly rather than infer it.
4. **Quantify the business value in seconds and positions per race (exp28).** Right
   now the value argument runs through pit-lap error and a 21-second pit loss. It
   should run through finishing positions.

**Also committed:** adopt Cappello & Hoegh's skewed-t observation model as a variant —
their idea, sound reasoning, cheap test; build a wet regime; replace the derived
traffic index with measured OpenF1 intervals.

> **The closing line:** *"Everyone in this field validates lap-time prediction. Almost
> nobody checks whether the reason is right — and the reason is the actual question,
> because many wrong decompositions add up to the same right total. We built that
> test, we pre-registered what would count as losing, we lost the leg we predicted we
> would lose, and we put it on the same slide as the one we won."*

---

# Appendix — the number card

Keep this where you can see it. Everything here is in `experiments/results/`.

| Claim | Number | File |
|---|---|---|
| Corpus | 203 sessions · 91,867 laps · 84 event-seasons · 2022–2025 | `data/season/` |
| Naive method gives an impossible answer | **74.0%** of 77 races · 53.4% of 208 stints | `exp14` |
| Degradation-rate recovery, ours | **0.0037 s/lap**, best of 9 | `exp19` |
| Closest published model | 0.0158 — **4.3× worse** | `exp19` |
| Interval coverage, recovery leg | ours **100%** · theirs **37.5%** · pooled 83.3% | `exp19` |
| Lap-time forecasting | **4th of 9** — pooled 0.4088, ours 0.6484 | `exp19` |
| LightGBM interval | claims 95%, delivers **62.0%** | `exp19` |
| Pit timing, like-for-like (49 common stops) | ours **5.92** · C&H 5.98 · pooled **11.12 (last)** | `exp22` |
| Real pit stops scored | **274**, 14 sessions, 93 excluded | `exp22` |
| Live interval coverage | **95.2%** over **69,206 laps**, 0.2 pts off target | `exp13` |
| Offline interval coverage | 94.7% (was 75.5% before conformal) | `exp12` |
| Calibration-diagonal gap | Gaussian 0.189 → adaptive **0.005** | `exp16` |
| Fuel/tyre collinearity | **520/520**, r = **1.000**, 11/11 singular | `exp18` |
| Data-driven share of the separation | **6.0%** | `exp18` |
| Distance from the Cramér–Rao floor | **1.38×** (0.02215 vs 0.01607) | `exp18` |
| Sector loading direction identified **without a prior** | median rank-one fraction **0.870**, 913 stints | `exp26` |
| Degradation regimes | linear **55.9%** · recovery 20.6% · cliff 11.9% · warm-up 11.6%, of 2,827 stints | `exp17` |
| The cliff | median **71.9%** through a stint, severity +0.202 s/lap | `exp17` |
| Circuit-direction physics check | **7 of 8** circuits | `exp06` |
| NASA C-MAPSS transfer | RMSE **22.67** (published best 15.98), 44% early | `exp07` |
| Fit cost | **12.66 s** per session, no GPU, fully offline | `exp19` |
| Neural network | CRPS 1.9442, **last of 9**, 37.4 s to fit | `exp19` |
| Practice→race, circular (do not quote as validation) | 0.0807 vs naive 0.1471, bias +0.0268 | `exp03` |
| Practice→race, de-circularised — **currently negative for every model** | ours skill **−4.81** (SE 2.89), 2 of a pre-registered 49 events | `exp27` |
| Tyre failure labels in public data | **17** in 12 seasons | `scripts/build_failure_labels.py` |
| Vehicle failure labels | **151** — 9× the tyre count | same |
