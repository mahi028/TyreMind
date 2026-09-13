# TyreMind — The Story

**The narrative arc for the TrackShift Innovation Challenge, Problem 3.**

Every number on this page points at a file in `experiments/results/`. If a claim
has no file, it is not on this page.

Companions: `docs/pitch/DEMO_SCRIPT.md` (the live run), `docs/pitch/JUDGE_QA.md`
(the hard questions), `research/HEAD_TO_HEAD.md` (evidence vs every competitor).

---

## The one sentence

> **A lap time is a bill with no itemisation. TyreMind itemises it — and it is the
> only system in this field that can tell you how much of that itemisation is
> evidence and how much is assumption.**

If you have thirty seconds, say that and stop.

---

## The one slide

| | |
|---|---|
| **Problem** | Strip fuel, traffic and track evolution out of a practice session and recover the true tyre degradation curve. |
| **Why it is hard** | The confounder is the same size as the signal. Fuel is worth ~0.081 s/lap. Tyre degradation is 0.03–0.08 s/lap. |
| **What goes wrong** | You do not get a slightly wrong answer. You get the wrong **sign**. On 77 real Grands Prix the standard method says tyres get *faster* as they wear in **74%** of races. |
| **What we built** | A state-space estimator with a measured — not assumed — uncertainty interval, a pre-registered nine-model benchmark, and a stated bound on what is knowable at all. |
| **The headline** | Degradation-rate recovery **0.0037 s/lap**, best of nine models, **4.3× better** than the closest published model. Our 95% interval covers the truth **100%** of the time. Theirs covers **37.5%**. |
| **The honest part** | We are **4th of 9** at plain lap-time forecasting, and we say so on the same slide. |

---

# Act 1 — The bill with no itemisation

You paid 95 seconds for that lap.

How much of it was the tyre? How much was fuel? How much was traffic? How much was
the track rubbering in?

Nobody sends you an itemised bill. The timing loop gives you one number, and that
number is a sum. Every strategy decision in Formula 1 — box now, box in three,
stay out — is a bet on one line item of a bill that was never itemised.

That is Problem 3, stated the way a pit wall experiences it.

---

# Act 2 — The confounder is the same size as the signal

This is why the problem is not a data-cleaning exercise.

- Fuel burn-off makes the car **faster** at roughly **0.081 s/lap**.
- Tyre degradation makes the car **slower** at roughly **0.03–0.08 s/lap**.

The thing hiding the answer is the same size as the answer.

So the failure mode is not "a bit noisy". The failure mode is **sign inversion**.

**We measured it.** `exp14_naive_failure_rate.json`, 77 real dry races, 208
compound-stints:

| | Failures | Rate |
|---|---|---|
| Races where the standard method reports at least one compound getting **faster** as it wore | 57 / 77 | **74.0%** |
| Individual compound-stints reported negative | 111 / 208 | **53.4%** |

In three out of four real Grands Prix, fitting a straight line through lap time
against tyre age claims that rubber regenerates.

**Why this is the strongest thing we own.** It needs no ground truth. It needs no
comparison to our model. It needs one fact everyone in the room already agrees
with: *tyres do not get younger*. You can falsify a method without knowing the
right answer, as long as you know which answers are impossible.

**And say the limit out loud:** this proves the standard method is broken. It does
**not** prove our number is right. Those are two different claims and we never let
them blur.

---

# Act 3 — There is no answer key, so we validated three ways

Measured tyre wear is the one thing on Earth you cannot download. Pirelli measures
every tyre after every session. That data goes to Pirelli, the FIA and the teams.
It has never been public. We searched Pirelli, the FIA, Kaggle, HuggingFace,
OpenF1 and Ergast. It does not exist.

So "just validate against the truth" is not available. Here is what we did instead.

### Route 1 — Impossibility (needs no truth at all)

Act 2. 74% of races. `exp14`.

### Route 2 — Build a world where truth exists, then hide it

`exp19_field_comparison.json`, Leg C. Eight synthetic seeds, 24 scored estimates.
Nine models, all fitted blind, truth never shown to any of them.

| Model | Rate MAE (s/lap) ↓ | Bias | 95% interval coverage |
|---|---|---|---|
| **TyreMind state-space** | **0.0037** | +0.0015 | **100%** |
| Pooled regression | 0.0062 | −0.0011 | 83.3% |
| Heilmeier additive (linear tyre) | 0.0112 | −0.0103 | 100% |
| Cappello & Hoegh state-space | 0.0158 | −0.0158 | **37.5%** |
| Fuel-corrected regression | 0.0212 | −0.0187 | 54.2% |
| Naive (lap time vs tyre age) | 0.0725 | −0.0725 | **0%** |
| ARIMA(2,1,2) · LightGBM · Neural network (MLP) | **no degradation parameter** | — | — |

Three of the nine models **cannot answer the question at all**. That is reported as
the finding, not as a gap in the table.

Read the naive row carefully. MAE 0.0725 and bias −0.0725 are the same number. It
is not noisy. It is wrong in the same direction every single time — always
*under*-stating degradation, because fuel gain masks tyre loss. That one row is the
whole problem in four digits.

Read the Cappello & Hoegh row too. It is the closest published prior art
(arXiv:2512.00640, Nov 2025), and it is the second-most-confident model in the
table while being 4.3× less accurate: it claims 95% and delivers **37.5%**.

**The caveat we volunteer before anyone asks:** the synthetic generator is ours.
Shared structural assumptions flatter us. Partial mitigations are in the
pre-registration — Student-t(5) observation noise where our estimator assumes
Gaussian, and 25% scrubbed sets — and they are partial, not complete. The real fix
is a truth engine we did not write. It is scoped as exp20 and not yet done. Until
it is, Leg C is **suggestive, not decisive**, and we label it that way every time.

### Route 3 — Take the same code somewhere truth *is* published

`exp07_cross_domain.json`. NASA C-MAPSS FD001. Unmodified F1 code. Tyre stint
becomes engine life, tyre age becomes flight cycles, lap time becomes a sensor
channel.

All 100 test engines, 12 sensor channels. **RMSE 22.67 cycles**, MAE 17.94, 44% of
predictions early (the safe side for maintenance).

**Published deep models reach 15.98.** We are 30–40% worse on point accuracy and we
put that number on the slide. **The claim is transfer, not state of the art.** A
tyre model pointed at a jet engine with no retuning should not work at all, and it
does.

One detail worth the ten seconds: the estimated degradation rate came out at
**0.00202** against a prior mean of **0.004**. The data moved the estimate by a
factor of two *away* from the prior. That is the clean answer to "your prior is
doing the work."

---

# Act 4 — The insight the pitch is built around

**Forecasting and explaining are different jobs.**

We ran a nine-model benchmark and **wrote down what would count as a win before we
ran it** (`experiments/PREREGISTRATION_exp19.md`, committed before the results
existed). The pre-registration says, in advance:

> *"We claim a win on Leg A only if TyreMind has the lowest CRPS. **We expect not
> to.** … If that holds, we report it as the headline of Leg A and explain the
> two-task distinction — we do not bury it."*

We lost, exactly as predicted. Here is the losing table, `exp19`, Leg A, 12 real
2025 races, rolling-origin folds:

| Model | CRPS ↓ | 95% coverage (target 0.95) | Fit |
|---|---|---|---|
| Pooled regression | **0.4088** | 87.0% | 0.02 s |
| LightGBM | 0.4826 | **62.0%** | 0.55 s |
| ARIMA(2,1,2) | 0.5489 | 74.5% | 5.73 s |
| **TyreMind state-space** | 0.6484 ← **4th of 9** | 81.9% | 12.66 s |
| Heilmeier additive | 0.7159 | 74.6% | 0.06 s |
| Cappello & Hoegh | 0.8388 | **41.9%** | 1.25 s |
| Naive | 0.8697 | 77.3% | 0.01 s |
| Fuel-corrected regression | 0.8715 | 77.2% | 0.01 s |
| Neural network (MLP) | 1.9442 ← last | 69.7% | 37.38 s |

Now put that next to what happens when a decision is on the line.
`exp22_pit_stop_validation.json` — **274 real pit stops**, 14 sessions, scored
against what professional strategists actually did. Like-for-like on the 49 stops
every model answered:

| Model | Error vs the strategist's real call (laps) ↓ |
|---|---|
| **TyreMind state-space** | **5.92** ± 0.52 |
| Cappello & Hoegh state-space | 5.98 ± 0.52 |
| Fuel-corrected regression | 6.31 ± 0.72 |
| Heilmeier additive | 7.55 ± 0.65 |
| Naive | 10.86 ± 0.67 |
| **Pooled regression** | **11.12** ± 0.70 ← **last** |

**The model that forecasts lap times best comes last at deciding when to stop.**

It is not a paradox. A pooled regression is a good forecaster because it soaks
everything — fuel, track, tyre, driver, traffic — into one flexible fit. That is
exactly what makes it a bad explainer. It predicts the total on the bill very well
and cannot itemise a single line.

**And we are not the first to find this.** Pitwall (arXiv:2607.06495), an
independent published production system, reports the same phenomenon and names it:

> *"calibration-optimal is not decision-optimal"*

They found it in race-outcome simulation. We found it separately in tyre
degradation, and we pre-registered the prediction before running the test.

**That turns our Leg A loss from an embarrassment into a replicated finding.**

And note the top two rows of the pit table: we and the closest published prior art
are **0.06 laps apart with a standard error of 0.52**. That is a tie, and our own
pre-registration obliged us to call it one:

> *"A tie is a tie. If TyreMind and Cappello & Hoegh are within one standard error
> on a metric, we say 'comparable', not 'better'."*

So we say comparable. The decisive gap in that table is not at the top. It is the
**5.2 laps** between the best forecaster and the best explainer.

---

# Act 5 — The result no paper in the field has

This is the technical centre of the pitch. `exp18_identifiability_bound.json`.

**The algebra.** Inside a single stint, tyre age goes up by one each lap. Laps of
fuel burned also goes up by one each lap. Write age as `a = a₀ + f`:

```
lap_time ≈ intercept + β·a − φ·f          [β = tyre effect, φ = fuel effect]
         = intercept + β·(a₀ + f) − φ·f
         = (intercept + β·a₀) + (β − φ)·f
```

The stint's intercept swallows `β·a₀`. The lap counter `f` only ever appears
multiplied by `(β − φ)`. **From one stint, only the difference is identifiable.
Never the two separately.**

This is not a sample-size problem. Adding a million laps changes nothing. It is
exact algebra.

**What we measured**, across 520 runs in 11 sessions:

| Measure | Value |
|---|---|
| Runs exactly collinear | **520 / 520 = 100%** |
| Mean within-run correlation (tyre age vs laps in run) | **1.000** |
| Sessions with a singular Fisher information matrix | **11 / 11** |
| Sessions unidentified without a prior | 10 / 11 |
| Cramér–Rao lower bound (best precision physically possible) | 0.01607 |
| Our model's reported sd | 0.02215 |
| **How close to the theoretical optimum** | **1.38×** |
| **Share of the separation that is data-driven** | **6.0%** |

**There is exactly one escape, and it is small.** A tyre curve *bends* — it warms
up early and falls off late. Fuel burn-off is a straight line and never bends. That
curvature is the only channel through which the data can tell the two apart, and it
carries **6.0%** of the information. The other **94% comes from the physical
prior.**

**Why this wins a technical judge.** Every entry in this competition will claim its
model separates fuel from tyre. We can state **exactly how much of that separation
is evidence and how much is assumption**, prove the evidence part is 6%, and show
our estimator sits within **1.38× of the information-theoretic floor** — so we can
tell you our answer *and* how much room is left above us.

**No paper in our 28-paper review states this number.** Cappello & Hoegh have the
identical model structure and resolve the problem with a half-normal prior that
forces the degradation rate positive, because the unconstrained estimate goes
negative. They never quantify what that prior contributes. We measured how often the
unconstrained estimate goes negative — 74% of races — and we measured what the prior
is doing — 94% of the work.

### The new result: part of the bound is escapable

`exp26_sector_identifiability.json`, 913 stints, 40 races, 25 drivers.

A lap time is one number. But a lap is timed in **three sectors**, and the three
sectors load the tyre differently while the fuel mass is identical across all three.
So sectors give you three equations where lap time gives you one.

What we found:

- All three sectors **depart significantly from time-proportionality** — S1
  p = 5.5e-10, S2 p = 8.8e-4, S3 p = 0.033. Degradation is not simply "the lap
  time, scaled". It has a shape across the lap.
- That shape is **stable, not noise**: split-half correlation 0.638 over 14,000
  splits, sign agreement 77.0%, permutation p = 0.0 against a null mean of −0.007.
- **The direction of tyre loading is identified from data alone, with no prior**
  (median rank-one fraction 0.870).
- **The overall level is still not** — three sector equations, four unknowns after
  normalisation. Given a measured fuel loading, the level problem conditions at 6.45.

**So the honest statement is:** the 6% bound applies to the *level*. The *shape*
escapes it. We found a real crack in our own impossibility result and we report both
halves.

---

# Act 6 — The field-standard model is the wrong shape

The field-standard race simulator is Heilmeier et al. Their tyre term, quoted
verbatim from the paper:

> **`t_tire(a, c) = k₀(c) + k₁(c)·a`**

A straight line, two coefficients per compound.

We measured 2,827 real stints across 77 races with broken-stick regression and BIC
selection, so an extra parameter has to earn its place
(`exp17_degradation_cliff.json`):

| Regime | Stints | Share | What it is |
|---|---|---|---|
| Linear | 1,580 | **55.9%** | Steady wear, no break |
| Recovery | 582 | 20.6% | Degradation *slows* later in the stint |
| Cliff | 337 | 11.9% | Degradation sharply accelerates |
| Warm-up | 328 | 11.6% | Fast improvement early, then settles |

A breakpoint model beat the straight line on BIC in **1,247 stints (44.1%)**.

**The field-standard model is the wrong shape for 44% of real stints.**

The cliff is real and lands where physics says it should: median at **71.9%**
through the stint, median severity **+0.202 s/lap** on top of the existing rate.

**And we report the negative alongside it.** Fitting the break does *not* improve
forecasting. Across 2,683 forecasts the broken-stick model was **26.5% worse** than
a straight line. It only helps on warm-up stints (+30.7%, p = 3e-18). So the cliff
is worth *showing* a strategist and not worth *fitting* for a forecast. We say both.

---

# Act 7 — Does any of it change a decision?

A degradation number that does not change a pit call is a science project.

`exp22_pit_stop_validation.json`: **274 real pit stops**, 14 sessions, 93 excluded
and listed, 21.0 s pit loss, ±2 lap hit tolerance. The reference is not our model
and not a simulator. It is **what professional race strategists actually did on the
day**.

The ranking is in Act 4. Two things to say about it.

**First, the honest scoring rule.** Cappello & Hoegh's method is per-driver by
design and returns no answer on thin sessions. We scored "no answer" as **no
answer**, never as an error — that rule is in the pre-registration, written before
we knew it would cost us. On all answered stops our MAE is 6.42 over 246 stops with
28 unanswered; theirs is 7.39 over all 274 with none unanswered. On the 49 stops
every model answered, we are tied.

**Second, and this is the ceiling we volunteer.** Our hit rate within ±2 laps is
**24.0%**. A published supervised Bi-LSTM classifier trained directly on pit labels
(Sasikumar et al., Frontiers in AI 2025) reports precision 0.77, recall 0.86,
F1 0.81. **We do not beat that and we are not trying to.** They train a classifier
on the outcome. We run an untrained expected-cost optimisation over a calibrated
degradation distribution and never see a pit label. Different problems. The number
is in our result file under `literature_benchmark` because we put it there, not
because someone found it.

---

# Act 8 — We are the only system here that knows how wrong it is

Every model in this field ships an error bar. Almost nobody checks whether the bar
is telling the truth.

**We checked ours, and it was lying.** `exp12_conformal_intervals.json`, 94
comparisons across 42 events:

| Interval | Claimed | Delivered | Median half-width |
|---|---|---|---|
| Our model's own Gaussian | 95% | **75.5%** | ±0.119 |
| Consistency score | 95% | 79.8% | ±0.139 |
| **Absolute score (what we ship)** | 95% | **94.7%** | ±0.276 |

The honest interval is **2.3× wider**. We shipped the wide one. A strategist told
"±0.12, right 75% of the time" will make a bad pit call and never know why.

**Then we did it live.** `exp13_lap_time_calibration.json`, 20 sessions,
**69,206 scored laps**, adaptive conformal inference updating its own α online:

| Method | Coverage | Off target by |
|---|---|---|
| Gaussian | 75.9% | 19.1 pts |
| Split conformal (absolute) | 87.6% | 7.4 pts |
| **Adaptive conformal** | **95.2%** | **0.2 pts** |

**0.2 percentage points from target across 69,206 laps.** The interval corrects
itself while the race is running, and the screen shows its own live coverage. No
other system in our review displays that.

`exp16_calibration_shape.json` goes one level further and asks not "is the coverage
wrong" but "wrong *how*": Gaussian intervals sit 0.189 from perfect calibration on
the reliability diagonal; adaptive conformal sits at **0.005**.

**This is the claim that beats "best in the world", because this one is provable.**

---

# Act 9 — Where we lose

Put this slide in the deck. A team that names its own losses is believed on
everything else.

| We lose to | On what | By how much | Why we accept it |
|---|---|---|---|
| Pooled regression | lap-time CRPS | 0.4088 vs our 0.6484 — we are **4th of 9** | Better forecaster, worse explainer. Comes **last** on pit timing (11.12 vs our 5.92). Independently replicated by Pitwall as *calibration-optimal ≠ decision-optimal*, and we pre-registered the prediction. |
| Cappello & Hoegh | pit-stop timing | 5.98 vs our 5.92 — a **three-way tie** with fuel-corrected regression at 6.31 | One standard error apart. Our own rules say a tie is a tie. |
| Cappello & Hoegh | publication priority | They published the model class first, Nov 2025 | We cite them, implement them as a baseline rung, and beat them on scale (203 sessions vs one race), on calibration (100% vs 37.5%) and on identifiability (they never quantify their prior). |
| CNN-LSTM / BiLSTM | C-MAPSS FD001 RMSE | 15.98 vs our 22.67 | Purpose-built for turbofans. Ours is unmodified tyre code. **Transfer, not SOTA.** |
| Sasikumar et al. Bi-LSTM | pit-stop classification | F1 0.81 vs our 24% hit rate | They train on pit labels. We never see one. Different problems, reported side by side anyway. |
| Pitwall | race-outcome calibration | Brier 0.0745 over 155 backtests | We do not model race outcomes at all. **We are the layer their paper says is confounded.** |
| Sulsters / Mercedes | data access | Team-internal telemetry | A data gap, not a method gap. Not reproducible by anyone outside the team, and we say so rather than approximate it. |

**Three more things we will not claim.**

1. **Puncture probability is impossible from public data.** We went looking for the
   labels. Across the entire Pirelli era, 2011–2022, 5,057 result rows: **17 tyre
   failures** (Puncture 12, Tyre 5). Detailed retirement causes stop entirely after
   2022. Three 2024 races checked for race-control messages: **0 tyre-related
   messages out of 249**. So no probability is calibratable from public data — by us
   or by anyone. **We ship a structural exposure index, labelled an index, never a
   percentage.** The same search found **151 vehicle failures** (Gearbox 54,
   Brakes 50, Suspension 37) — nine times the tyre count — so that is where a hazard
   model actually belongs.
2. **The synthetic win is on a generator we wrote.** Suggestive, not decisive. Act 3.
3. **Our practice-to-race test is circular, and the de-circularised replacement is
   not going our way.** `exp03` reports 0.0807 vs naive 0.1471 (45% tighter) over 42
   events — but the "actual" it scores against is our own model's race fit. It shows
   we agree with ourselves. It is **not** ground truth and must never be presented as
   such. We pre-registered the fix (`experiments/PREREGISTRATION_exp27.md`): score
   every model against **its own** race-derived estimate, with a skill score against
   each model's own climatology so that a constant model scores exactly zero and
   cannot win. **The first partial run is in the repo and the early answer is
   negative for everybody** — see the box below.

---

### ⚠ Live result — exp27, and how to handle it on stage

`exp27_decircularised_practice_to_race.json` is **in the repo and judges can open
it.** Know what it says before they do.

It is the de-circularised version of exp03. Primary metric is a **skill score
against each model's own climatology** — zero or below means the practice session
told that model nothing its own average did not already tell it.

**On the first run — 2 events, 24 comparisons, 4 scored per model — every model
scores negative skill.** Ours is −4.81 (SE 2.89). The verdict field reads
`win_claimed: false`, `practice_uninformative: true`. We rank 3rd of 6 on the
de-circularised metric, having ranked 4th on the circular one.

**Three things to say, in this order:**

1. **"That is our experiment, pre-registered, and we ran it knowing it could go
   against us."** The pre-registration fixes the metric and the losing condition in
   advance, precisely so a bad result cannot be rescued by changing the score.
2. **"It is 2 events. The pre-registration specifies 49."** The standard error is
   2.89 against a point estimate of −4.81. Nothing is concluded yet, in either
   direction, and we will not pretend otherwise while the full run is pending.
3. **"If it holds at 49 events, the finding is that a Friday practice session does
   not predict Sunday for *anyone* — and that is a more important result than our
   45%."** It would not be a TyreMind failure. It would be a statement about the
   sport, and it would immediately reframe what a practice-derived number is for:
   a **within-session** decomposition, which is what Problem 3 actually asks for,
   rather than a cross-session forecast.

**What this does NOT touch:** exp14 (74% of 77 real races), exp18 and exp26
(identifiability), exp19 (the nine-model benchmark), exp22 (274 real pit stops),
exp12/exp13 (calibration). None of those depend on practice-to-race transfer.

**Never say the 45% is validation.** Under any circumstances.

---

# Act 10 — Eight of our own ideas, killed by our own tests

This is the slide that makes the rest of the deck believable.

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 1 | Energy through the tyre is a better clock than lap count | ✗ Refuted | exp04: won 1 stint of 4, mean R² gain −0.0008. Energy per lap varies only 2.3% lap-to-lap, so "energy so far" and "laps so far" are the same number. |
| 2 | ...and it will hold up with real telemetry at scale | ✗ Refuted again | exp25: 2,219 stints, 24 circuits, 84 sessions. Total frictional energy ρ = −0.040, p = 0.060. `supported: false`. |
| 3 | The true Pirelli compound (C1–C5) beats the HARD/MEDIUM/SOFT label | ✗ Refuted | exp08: true-compound grouping was **6.4% worse** (LOEO MAE 0.0506 vs 0.0475), won 22 of 58, Wilcoxon p = 0.096. We built the dataset by hand and the simpler feature won. |
| 4 | Circuit geometry transfers degradation knowledge to a new track | ✗ Refuted | exp09, leave-one-circuit-out over 26 circuits: geometry −3.3%, p = 0.222 (no effect). **Thermal features actively hurt, p = 0.0012.** |
| 5 | Temperature explains our practice-to-race bias | ✗ Refuted | exp10, nine candidates with Benjamini–Hochberg FDR control: temperature gap ρ = −0.022, **p = 0.835**. The explanation every fan offers first explains nothing. One of nine survived. |
| 6 | Matching stint depth will fix the bias — **our own idea** | ✗ Refuted | exp11: bias went from +0.0268 to +0.0286 (6.9% *worse*), MAE 0.0807 → 0.0918 (13.8% worse), paired t = −2.46, **p = 0.0159**. Correlation did not survive contact with intervention. |
| 7 | Driver style is a usable feature | ✗ Refuted | exp15, 1,458 driver-races: raw driver effects have a split-half 5th percentile of **−0.079** — in 5% of splits the effect reverses. Teammate contrasts *are* stable (+0.031 at the 5th percentile), but adding driver identity made prediction **0.88% worse**. A small real effect that does not help. |
| 8 | Puncture probability can be built from public data | ✗ Refuted | exp23: 17 tyre failures in 12 seasons. Act 9. |

**Number 6 is the one to show a judge who asks about rigour.** We had a plausible
mechanism derived from our own data, we designed a test that could kill it, and it
killed it. That is the difference between a dashboard and a research project.

---

# Act 11 — Why this is worth money

**It is not an F1 model. It is a degradation engine that happens to be pointed at
tyres.**

The same unmodified code runs on NASA turbofan engines (`exp07`). The abstraction
is: *a hidden state that only decays, observed through a signal contaminated by
confounders the same size as the decay.* That describes tyres, jet engines, wind
turbine gearboxes, rail wheelsets, mining haul trucks and battery packs.

| Level | Customer | Value |
|---|---|---|
| **Team** | 10 F1 teams | One wrong pit call costs positions. Positions cost millions. `exp22` puts the error in laps; at a 21 s pit loss the seconds follow directly. |
| **Series / broadcast** | F1, F2, F3, WEC, IMSA, Formula E | Insight graphics. Note Pitwall already competes here — do not lead with it. |
| **Industrial** | Fleet maintenance, wind, rail, mining | Domain-agnostic, already demonstrated on turbofans. The strategist persona becomes the maintenance planner. **Largest market, and nobody in our 28-paper review addresses it.** |

**The cost story, which matters more than it sounds.** 12.66 s to fit a session on a
laptop. 2.6 MB for four seasons. No GPU. No training run. No cloud bill. **Runs
fully offline** — you can clone it and use it on a plane. The neural network in our
own benchmark took 37.38 s and came **last of nine**.

---

# Act 12 — Building on the existing TrackShift engine

**Say this carefully. The mentors in the room supervised that system.**

We are not saying their approach is bad. We have not seen their code. We are saying
what an architecture structurally cannot do, and attaching our own measurement of
the cost.

| Their design | The structural ceiling | Our measurement |
|---|---|---|
| Five-parameter formula, calibrated on **Bahrain V3** — one circuit | No evidence a single-circuit calibration travels | exp09, LOCO over 26 circuits: circuit transfer genuinely fails. Geometry p = 0.22, thermal actively harmful p = 0.0012. |
| **Threshold** alerts on point estimates | A threshold with no interval is a coin flip near the line | exp12: our own model claimed 95% and delivered 75.5%. We only know that because we checked. |
| **18 rules + a 17-step ladder** | Cannot say "I don't know", cannot weigh the cost of being wrong. The supersede logic is the tell: rules collide, so rules arbitrate rules. | Replaced by expected-cost minimisation over a calibrated distribution — **with all 18 rules kept as a veto layer.** When model and rules disagree, the engineer sees both. |
| No negative results | Untested features silently degrade a model | Act 10. Eight of them, including exp09 where a feature made predictions significantly worse. |

**Their five metrics, our versions:**

| Their metric | Ours |
|---|---|
| Grip Level — "real-time traction coefficient" | Normalised **grip index 0–1**. We refuse to print a μ value: true μ needs load-cell and slip channels that are team-private. |
| Tread Remaining — "remaining depth" | **% of usable life, with an interval**, anchored to the measured cliff at 71.9% through the stint. **Never millimetres** — no public measurement of F1 tread depth exists to anchor against. |
| Degradation Rate — 5-param Bahrain formula | State-space, 4 seasons, calibrated interval, **with the identifiability bound stated**. |
| Tyre Energy — "laps of useful life" | Conformal RUL — and the same code already does this on jet engines. |
| Puncture Risk — "threshold-based and probabilistic" | **A structural exposure index, explicitly not a probability.** 17 labels in 12 seasons. |

**The line to deliver on the last two rows:**

> *"They report remaining depth in millimetres and a traction coefficient. Neither
> is measurable from public data. We report percentage of life remaining and a
> normalised grip index — because we would rather be right than precise."*

And their four "future opportunities":

| Their future item | Our status |
|---|---|
| Live telemetry integration | 🟢 **Substantially built** — online filter, self-correcting interval, 95.2% over 69,206 laps |
| Driver-in-the-loop | 🟡 Physics layer exists; the closed loop is scoped, not built |
| Multi-compound strategy simulation | 🟡 Simulator exists; needs uncertainty propagated through it |
| Full vehicle health monitoring | 🟢 **Already demonstrated** — NASA turbofans, plus **151 vehicle failure events** collected. Better supported by public data than the puncture metric that already ships. |

---

# The close

> **"Everyone in this field validates lap-time prediction. Almost nobody checks
> whether the *reason* is right — and the reason is the actual question, because
> many wrong decompositions add up to the same right total.**
>
> **We built that test. We pre-registered what would count as losing. We ran it, we
> lost the leg we predicted we would lose, and we reported it on the same slide as
> the leg we won.**
>
> **We are not claiming to be the best tyre model in the world. We are claiming to
> be the only one in this room that can prove how wrong it is — 95.2% coverage over
> 69,206 laps, 6% of the separation from data and 94% from physics, and within
> 1.38× of the best any estimator could ever do.**
>
> **That claim is provable. 'Best in the world' is not."**

---

## Appendix — how this maps to the judging weights

| Weight | Criterion | Where we are strongest | Say this |
|---|---|---|---|
| **30%** | Technical depth | **Act 5** — the identifiability bound. Cramér–Rao, 6% data-driven, 1.38× optimal, and the sector-level escape in exp26. **No paper in a 28-paper review states this.** | Lead every technical conversation here. |
| **25%** | Problem fit | **Acts 1–2** — the problem is *isolating* tyre degradation from confounders, and exp14 proves the standard approach fails at that specific job in 74% of races. | We are solving the stated problem, not an adjacent one. |
| **20%** | Working prototype | **`DEMO_SCRIPT.md`** — 8 offline sessions, 12.66 s per fit, live self-correcting interval, 0.22 ms per online update, no network required. | Run it, don't describe it. |
| **15%** | Real-world applicability | **Act 7** (274 real pit stops vs professional strategists) and **Act 11** (same code on turbofans, industrial market). | Decisions, not metrics. |
| Remainder | Presentation | **Acts 9 and 10** — we name our losses before anyone asks. | The honest framing *is* the persuasive one. |
