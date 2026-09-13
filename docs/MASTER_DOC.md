# TyreMind — The Complete Explanation

**For the team. Read this and you can answer anything.**

Written in plain words. Every number here was read from a result file, not from
memory. Where we lose, it says so — that is deliberate, and §14 explains why it
is the strongest thing in the document.

---

## Contents

| Part | What it answers |
|---|---|
| 1 | What problem are we solving? |
| 2 | What data do we have? Real and synthetic |
| 3 | The orchestration, step by step, with the maths |
| 4 | The maths explained as if you have never seen it |
| 5 | Every paper we read, what they did, how we differ |
| 6 | The four questions the mentors asked |
| 7 | How we compute confidence |
| 8 | How we test and validate |
| 9 | Every model we ran, and where we won |
| 10 | Why we did not train anything |
| 11 | Is it adaptive, feasible, scalable? |
| 12 | The physics, and the test on another industry |
| 13 | Hard questions with answers |
| 14 | Everything we got wrong |

---

# PART 1 — THE PROBLEM

## Say this first

> A lap time is a bill with no itemisation. You paid 95 seconds. How much was the
> tyre? How much was fuel? How much was traffic?

Four things move a lap time at once:

| Cause | Direction | Roughly worth |
|---|---|---|
| Tyre wearing out | **slower** | 0.03–0.08 s/lap |
| Fuel burning off (car lighter) | **faster** | 0.081 s/lap |
| Track rubbering in | faster | ~0.9 s over a session |
| Traffic | slower | up to ~1.3 s |

**The thing hiding the answer is the same size as the answer.** Fuel is worth
about as much as the tyre and pushes the opposite way.

## Why that matters more than it sounds

If you ignore fuel you do not get a slightly wrong answer. **You get the wrong
sign.** We measured it: across **77 real Grands Prix**, the standard simple method
says the tyre got *faster* as it wore in **74% of races** and in **53.4% of 208
individual stints**.

Tyres do not regenerate. That is not a small error; it is an impossible answer.
And you do not need to know the true wear rate to know a negative one is wrong —
which is why this is our most defensible single result (**exp14**).

## What the competition asks for

> *"Create a predictive model that strips out external noise from practice
> sessions to generate clean tyre performance degradation curves. Includes
> post-race validation tools."*

---

# PART 2 — THE DATA

## 2.1 Real data

| Source | Size | What it is |
|---|---|---|
| **Season corpus** | **203 sessions, 91,867 clean laps**, 2022–2025 | The main dataset. Lap times, compound, tyre age |
| **Telemetry** | **202 sessions, 147,550 laps**, 27 features each | Per-corner energy, load, lateral g, throttle, braking |
| **OpenF1 intervals** | **70 races** | Measured gap to the car ahead at ~4 Hz |
| **Track geometry** | 6 circuits | Real 3D racing lines with elevation |
| **Demo sessions** | 10, committed to git | So a clean clone runs offline |
| **NASA C-MAPSS** | 4 datasets | Jet engines, for the cross-industry test |
| **Research papers** | 31 PDFs | Everything published that we could find |

**Where it comes from:** FastF1 → F1's official live timing API. The same feed
that powers the F1 TV timing screens. We downloaded it ourselves.

### The eight columns that matter

```
driver · session_lap · run_id · tyre_age · lap_time · compound · traffic_index · lap_in_run
```

**Only four of those come from the feed.** The rest we derive:

- `run_id` — a new tyre set. We detect it from compound changes and age resets.
- `traffic_index` — 0 to 1, how blocked the lap was. **We invented this.**
- `lap_in_run` — laps since the stint began. **This is the fuel proxy.**

### `tyre_age` vs `lap_in_run` — do not confuse these

- `tyre_age` = laps **this set of tyres** has done. Can start above zero if a
  used set is fitted.
- `lap_in_run` = laps since **this stint** started. Proportional to fuel burned.

Conflating them is the single most common error in this problem, and it caused a
real bug in our own code (§14).

### How we validated the traffic index

We invented it from lap **start times** — two cars crossing the line 0.6 s apart
are 0.6 s apart for the next lap. No positional data needed.

Then we checked it against OpenF1's **measured** 4 Hz gaps, race by race:

| | |
|---|---|
| Races checked | **66** |
| Median correlation with the measured gap | **ρ = −0.798** |
| Races with the correct (negative) sign | **66 of 66** |
| Weakest single race | ρ = −0.512 |
| Largest p-value of any race | 3×10⁻¹⁹ |
| Stint-level agreement (2,441 stints) | **ρ = 0.912** |

Negative is correct: a small measured gap means heavy traffic means a high index.
**Not one race points the wrong way, and the worst race is still strong.**

## 2.2 Cleaning — seven filters, every one logged

A raw session is not clean. It has in-laps, out-laps, safety-car laps, and laps
where the driver gave up on a corner.

| # | Filter | Why |
|---|---|---|
| 1 | `no_lap_time` | Nothing to model |
| 2 | `pit_in_out_lap` | Slowed by the pit lane, not the tyre |
| 3 | `flagged_inaccurate` | Upstream timing glitch |
| 4 | `unknown_compound` | Cannot assign to a tyre set |
| 5 | `no_tyre_age` | Same reason. Found when 70 laps of one race had nulls |
| 6 | `wet_compound` | Wet wear is a different physical process |
| 7 | `slow_lap_safety_car_or_traffic` | Median + 3×MAD |
| 8 | `run_too_short` | Cannot fit a slope through 3 points |

**Every session ships a receipt.** Monza 2024: API gave 1,008 laps, we kept 921,
and the file says exactly where the other 87 went.

**The outlier rule**, plainly: take the median lap time, compute the median
absolute deviation, multiply by **1.4826** (the constant that makes MAD
comparable to a standard deviation), drop anything slower than median + 3×that.
We use median and MAD rather than mean and standard deviation because a
safety-car period inflates a standard deviation so much that the safety-car laps
stop looking like outliers and hide themselves.

## 2.3 Synthetic data — and why we need it

**The honest core of the whole project:**

> On real data we have the lap time but **not** the true wear rate. Pirelli
> measures it, the FIA and teams see it, the public never does. So if our model
> says 0.06 s/lap there is nothing to check it against.

In synthetic data **we chose the true rate**, so we can check exactly.

**We searched properly** for a public source of measured wear: Pirelli's media
site, FIA documents, Kaggle, Hugging Face, academic replication packages, OpenF1,
Ergast, and the timing schema itself. **It does not exist.**

### Two generators, deliberately

**Generator 1 — `synthetic.py`.** Latent state plus fuel, track, traffic. Made
hostile on purpose: **Student-t(5) noise** where our estimator assumes Gaussian,
and **25% scrubbed sets** so tyre age does not start at zero.

**Generator 2 — `physics_truth.py`.** Because generator 1 shares our estimator's
*shape*, and a generator shaped like the estimator cannot falsify it. Here lap
time comes from a **line integral over a speed profile**; fuel acts through
**mass**; traffic through **downforce loss**; degradation bends because grip falls
non-linearly as wear feeds back into next lap's speed. **There is no rate
parameter anywhere** — the true rate is a counterfactual (the same lap on an
unworn tyre), differenced.

**The caveat we always state: we wrote both.** Generator 2 removes the
functional-form advantage. It removes nothing else. It is not third-party
validation.

---

# PART 3 — THE ORCHESTRATION

```
     their timing feed  (4 fields: lap_time, compound, tyre_age, lap_number)
              |
   [1] INGEST         save_session() — a parquet drops in and is served
              |
   [2] CLEAN          7 filters, every one logged, receipt per session
              |
   [3] FEATURES       run_id, traffic_index, lap_in_run  (all derived)
              |
   [4] FIT            Kalman filter forward -> RTS smoother backward
                      6 parameters, maximum likelihood, ~12 s
              |
   [5] CALIBRATE      conformal — where "95%" is made to mean 95%
              |
   [6] STATE          one TyreState object, every field carrying its own sd
              |
      +-------+--------+
      |                |
   [7a] ORACLE      [7b] DECISION
   probabilities    pit optimiser + 18 rules as a veto
      |                |
      +-------+--------+
              |
   [8] VERIFIER       every sentence -> typed claims -> checked against state
              |
   [9] AGENT          role-adapted language, 4 surfaces
              |
   [10] ROUTER        each question answered by the model measured best at it
```

**Steps 5, 8 and 10 are what no competitor has.** Everything else is standard.

---

# PART 4 — THE MATHS, EXPLAINED SIMPLY

## 4.1 The observation equation

For driver `d` on session lap `t`, running tyre set `r`:

```
y[d,t] = α[r]            run intercept: car pace, setup, starting fuel
       − A·g(t)          track evolution, shared by the whole field
       + θ[t]            small residual track wobble
       + s[d,t]          ← THE TYRE. What we want. Invisible.
       − φ·L[d,t]        fuel burn-off, L = laps completed this run
       + γ·TI[d,t]       traffic
       + ε               noise
```

**In words:** the lap time is the car's baseline, minus what the track gives back,
plus what the tyre costs, minus what the fuel gives back, plus what traffic costs,
plus randomness.

## 4.2 The state equation — the important bit

The tyre's condition is a **hidden state** that evolves in **tyre age**:

```
s[d, a+1]    = s[d,a] + rate[d,a] + noise     how much performance is lost so far
rate[d, a+1] = rate[d,a] + noise              how fast it is being lost right now
```

**`rate` is the product.** Seconds per lap the tyre is costing, right now.

**Why this matters:** because `rate` is free to drift, **a cliff needs no special
machinery.** It appears as `rate` accelerating. A recovery appears as `rate`
falling. **We never tell the model what shape degradation has.**

Everyone else does. Heilmeier forces a straight line. Cappello & Hoegh's best
extension forces the rate to grow linearly. We measured **2,827 real stints** and
only **55.9% are straight**:

| Regime | Stints | Share |
|---|---|---|
| Linear | 1,580 | 55.9% |
| Recovery | 582 | 20.6% |
| Cliff | 337 | 11.9% |
| Warm-up | 328 | 11.6% |

**The field-standard model is the wrong shape for 44% of real stints.**

## 4.3 Three things that look identical — and how each is pinned

This is the honest heart of the method.

### Problem 1 — fuel and tyre are perfectly confounded

Within a stint, tyre age and laps-of-fuel-burned advance together. Write
`a = a₀ + f`:

```
y = c + β·a − φ·f
  = c + β(a₀ + f) − φ·f
  = (c + β·a₀) + (β − φ)·f
```

The run intercept swallows `β·a₀`, and the slope you can actually see is
**`β − φ`**, not `β`. **More data does not help. It is algebra.**

**Pinned by:** a physical prior on `φ` = 0.030 s/kg × 2.7 kg/lap = **0.081 s/lap**.
And the prior's own uncertainty is carried through, so it *widens* every interval
we publish.

### Problem 2 — track evolution vs a uniform shift in all rates

Shift every rate by `c` and the track slope by `−c`, and the run intercept absorbs
the difference exactly.

**Pinned by:** modelling track evolution as a **saturating curve** `A(1−e^{−kL})`
rather than a free trend. Rubber deposition genuinely saturates; tyre wear does
not. That shape difference is what separates them — physically correct *and*
identifying.

### Problem 3 — tyre age vs session lap

**Pinned by data, and this is our best idea.**

> Every car pits at a different time. On lap 30, one car is on 3-lap-old tyres and
> another on 25-lap-old tyres. **The grid staggers itself.**
>
> Watch one car and you cannot do this. Watch twenty and it solves itself —
> **for free, with no assumption at all.**

**Cappello & Hoegh fit one driver in one race. They structurally cannot use this.**

### What it costs, measured (exp18)

| | |
|---|---|
| Runs exactly collinear | **520 / 520** |
| Within-run correlation | **1.000** |
| Sessions with singular information matrix | **11 / 11** |
| Cramér–Rao bound (best possible) | 0.0161 |
| Our reported sd | 0.0221 → **1.38× optimal** |
| **Share of the answer that is data, not prior** | **6.0%** |

**No paper in our review states this number.** Cappello & Hoegh have the identical
structure and resolve it with a prior whose contribution they never quantify.

## 4.4 Kalman filter and smoother, in plain words

**Forward (filter):** guess the tyre state → see the actual lap → correct the
guess. Lap by lap. This is what a pit wall knows in real time.

**Backward (smoother):** run it again backwards so every lap benefits from what
happened later. Better, but only available after the session.

It is the same machinery GPS uses to track your position from noisy satellite
signals. Our tyre state is the position; the lap times are the signals.

## 4.5 The six parameters — the entire model

```
log_q_track        how much the track wobbles lap to lap
log_q_level        how much the tyre state drifts
log_q_rate         how much the rate itself drifts   ← this is what makes a cliff possible
log_obs_sd         lap-to-lap noise
log_stint_rate_sd  spread of rates across stints
log_track_shape    how fast track evolution saturates
```

Fitted by maximum likelihood (L-BFGS-B) on each session. **Six numbers. No
training set. No weights. No checkpoint.**

Everything else — the fuel coefficient, traffic coefficient, track amplitude,
per-compound baselines — is a **state with zero process noise**, which for a
constant *is* the Bayesian posterior. That makes physical priors enter as
`P₀` and propagate automatically.

---

# PART 5 — THE PAPERS

31 PDFs in `research/papers/`, eleven topic folders, every citation in `INDEX.json`.

## 5.1 The closest prior art — read this twice

### Cappello & Hoegh (arXiv:2512.00640, November 2025, Montana State)

**They published our model class, one month before we built ours.**

```
Observation:  y_t = α_t + γ·fuel_t + ε_t ,   ε_t ~ N(0, σ²)  [or Skewed-t]
Process:      α_{t+1} = (1 − I_pit)(α_t + ν) + I_pit·α_reset + η_t
Extension 2:  ν_{t+1} = (1 − I_pit)(ν_t + β[c]) + I_pit·ν_reset
Priors:       ν ~ N⁺(0.05, 0.1²)   σ_ε ~ N⁺(0.3, 0.1²)   α_reset ~ N(69, 0.1²)
```

They write: *"To the best of our knowledge, there are no examples in the
literature that apply state-space models to tyre degradation in Formula 1."*

**Say this before anyone asks.** And then say why it helps us: independent
researchers chose the same architecture for the same stated reason — that deep
learning *"lacks interpretability and explicit uncertainty quantification,
features crucial in operational race environments."* Our design choice is no
longer our opinion.

| | Them | Us |
|---|---|---|
| Scope | **1 driver, 1 race** | **203 sessions, 91,867 laps, 4 seasons** |
| Traffic | absorbed into noise | explicit, validated term |
| Track evolution | ❌ none | ✅ saturating curve |
| Negative rates | **prevented** by a prior | **measured**: 74% of races |
| Uncertainty | Bayesian, assumed correct | conformal, **measured** |
| Identifiability | not addressed | **bounded: 6% data-driven** |
| Inference | Stan / MCMC | Kalman + L-BFGS-B |

### Pitwall (arXiv:2607.06495, July 2026)

A production system: ingests live F1 timing, runs Monte Carlo (N=2,000 per lap),
speaks calibrated briefings in three languages. Winner-in-top-3 **90.3%** over 155
backtests, Brier **0.0745**. Ran live at two 2026 Grands Prix.

**So "AI agent explains race strategy" is not novel.** Know that.

**But their tyre layer, in their own words:** a *"compound-pace normalization"*
that *"worsens finishing order prediction, because the raw, **confounded**
per-race fit carries genuine race-specific signal."*

> **They use the word "confounded". That is our problem statement, named as an
> unresolved tension inside the best system in the field. We are the layer they
> say they are missing.**

**And their central finding, which we replicated independently:**
*"calibration-optimal is not decision-optimal."* They split into an oracle path
and a decision path because the two criteria disagree. We found the same in exp05
before reading them. That turns our lap-time loss from an embarrassment into a
replicated result.

## 5.2 The rest

| Paper | What they do | The maths | How we differ |
|---|---|---|---|
| **Heilmeier et al. 2018/2020** — the field standard | Race strategy simulation | `t_lap = t_base + t_tire(a,c) + t_fuel(l) + t_car + t_driver + t_grid + t_pit`, with **`t_tire = k₀(c) + k₁(c)·a`** | Their tyre term is a **straight line**. Only 55.9% of real stints are. No track-evolution term, no traffic in the lap time, no uncertainty on `k₁` |
| **Farroni/Timpone TRT** | Thermal tyre model | Fourier heat conduction on a 3-D tyre, `ρc∂T/∂t = ∇·(k∇T) + q̇` | We implement a reduced 2-state version. Explains why our thermal features hurt: the driver is *contact-patch* temperature; we only have a trackside mast |
| **Archard / Braghin / energy-wear** | Physical wear | `V = K·F·L/H`; modern form `ẇ ∝ τ·v_slip` | Implemented both. **We tested it and it fails on public data** (exp25, exp34) |
| **Sulsters/Todd 2025** | LSTM/GRU/TFT for tyre energy | Deep learning | They had **Mercedes' internal data**. Not a like-for-like competitor — a demonstration of what private channels buy |
| **Gibbs & Candès 2021** | Adaptive conformal | `α_{t+1} = α_t + γ(α − err_t)` | We implement **their** rule. Cited, not claimed |
| **Lei et al. 2018** | Split conformal | `ceil((n+1)(1−α))` order statistic | Same |
| **Sasikumar et al. 2025** | Pit-stop prediction | Bi-LSTM + SMOTE | Precision 0.77, recall 0.86, **F1 0.81**. Trained on pit labels; ours never sees one |

## 5.3 Are we copying?

**Honest answer: the ingredients are textbook and cited. The recipe and four parts
are ours.**

| Ours | In any paper we read? |
|---|---|
| Run stagger across the field as the identifying experiment | ❌ Single-car models cannot |
| Saturating track curve as the identifying device | ❌ |
| **Quantifying data vs prior — 6%, CRB 1.38×** | ❌ **None** |
| Measured (not assumed) interval coverage for degradation | ❌ None |
| Falsification without ground truth — 74% | ❌ None |
| Four regimes measured, not assumed | ❌ None |
| Same code transferred to jet engines | ❌ None |

**"We invented state-space tyre modelling" would not survive. This list does.**

---

# PART 6 — THE FOUR QUESTIONS

> *"Lap 1 took 90 s, lap 2 took 92 s, lap 3 took 95 s. Where is the time going?"*

| # | Question | Endpoint | What comes back |
|---|---|---|---|
| 1 | Where is the time going? | `/decompose-run` | tyre +0.196, fuel −0.078, track −0.033, traffic 0.000, **residual −0.080** — each with its own sd |
| 2 | How fast is he losing it? | `/degradation` | MEDIUM 0.198 ± 0.022 s/lap |
| 3 | After 3 laps? 15? | `/projection` | 20-lap horizon, `loss` and `loss_sd`, band widening |
| 4 | When do I box? | `/pit-window` | lap + **calibrated windows** + cost sweep |
| 5 | Can I trust it? | `/trust` | four methods agreeing within 0.030 s/lap |

**Use `/decompose-run`, never `/decompose`.** Single-lap attribution leaves 73% of
the gap as residual; the per-lap run view sits around 0.08 s.

**And show the residual.** It is what we cannot explain, and hiding it would claim
more than we can support.

---

# PART 7 — HOW WE COMPUTE CONFIDENCE

## 7.1 The problem with the model's own error bars

Our model said 95%. We checked. **It was right 75.5% of the time.**

That is lying with mathematics, and it is what every competing system ships —
because nobody checks.

## 7.2 Conformal prediction, in plain words

Instead of trusting the model, **measure how wrong it has been and use that**.

Take past predictions, compute how far each missed, take the
`ceil((n+1)(1−α))`-th biggest miss. That is your interval. The `+1` accounts for
the new point itself.

**It assumes nothing about the shape of the errors.** That matters here because
our errors are skewed — teams pit earlier than pure degradation implies, for track
position we do not model.

## 7.3 What we measured

| Quantity | Claimed | **Delivered** | Sample |
|---|---|---|---|
| Degradation interval (exp12) | 95% | **94.7%** | 94 comparisons |
| Live lap-time interval (exp13) | 95% | **95.2%** | **69,206 laps** |
| Pit window, 90% (exp30) | 90% | **90.8%** | held-out real stops |
| Pit window, 80% | 80% | **83.1%** | |
| Pit window, 50% | 50% | **53.8%** | |

**Before conformal calibration the pit window claimed 31.9% and delivered 24.0%.**

## 7.4 The claim that actually sells

Anyone can be calibrated by widening until they are always right. **Being
calibrated *and* narrow is the product.**

For the **same 90% guarantee**, the window each model needs:

| Model | Half-width | Stops it could answer |
|---|---|---|
| **TyreMind** | **13 laps** | 246 |
| Cappello & Hoegh | 17 laps | 274 |
| Pooled regression | 20 laps | 137 |
| Naive | 22 laps | 119 |
| Heilmeier | 26 laps | 222 |
| Fuel-corrected | 27 laps | 271 |

**40% narrower than naive for an identical promise** — and naive earned its 22
laps on only the 119 easiest stops, while we earned 13 on 246.

Our own held-out re-check lands on 13.3 laps at 90.8% coverage, so the width is
honest against stops it was never fitted on.

## 7.5 The pit probability is a frequency, not a formula

We simulate 1,200 races per candidate lap **under common random numbers**. Draw
*i* read across all candidates is one coherent race; whichever lap wins that draw
would have been right. **Count the winners.**

That is a real probability, not a softmax dressed up as one.

Live example — Monza lap 20:
```
box lap 39  |  9% on that exact lap  |  51% on the window 37–42  |  0% within 3 laps
```

**The 9% is the honest number.** Picking one lap from thirty on a nearly flat cost
curve is genuinely unknowable. **The window is what a strategist acts on.**

---

# PART 8 — HOW WE VALIDATE

## Three independent legs, and two need no synthetic data at all

| Leg | Ground truth | What it proves |
|---|---|---|
| **A. Synthetic** | We set it | The estimator is correct |
| **B. Real, label-free** | None needed | The naive method is broken (74% impossible) |
| **C. Real, real labels** | **274 actual pit stops** | We agree with professional strategists |

## Pre-registration

Before running the nine-model comparison we wrote down the models, the metrics,
the directions, and **what counts as a loss** — then committed it. It says:

> *"We claim a win on Leg A only if TyreMind has the lowest CRPS. **We expect not
> to.**"*

We were right; we lost that one. **The pre-registration is worth more than the
win** would have been.

## The false-positive floor (exp33)

We built a negative control — a measure that provably carries no information — and
**it was not null**. 4 of 12 pre-registered contrasts survived Bonferroni at
|ρ| = 0.105, p < 10⁻⁷.

**Demeaning within circuit and driver does not immunise this corpus.** So we now
read correlation-style findings against an empirical floor of **ρ ≈ 0.10**. Our
kerb measure's largest was 0.0085 — an order of magnitude below it, so that null
is a null against a calibrated noise level rather than a nominal α.

**Building a control that should have been null, finding it wasn't, and
recalibrating everything else is the most methodologically careful thing in this
project.**

---

# PART 9 — EVERY MODEL, AND WHERE WE WON

## The nine-model ladder

1. Naive · 2. Fuel-corrected · 3. Pooled regression · 4. LightGBM ·
5. Neural network (MLP) · 6. **TyreMind** · 7. ARIMA(2,1,2) ·
8. **Heilmeier** (field standard) · 9. **Cappello & Hoegh** (closest paper)

Rungs 7–9 are **other people's published methods, rebuilt faithfully.** None is
crippled; ambiguous choices were resolved in the published model's favour.

## 🥇 Recovering the degradation rate — the thing the problem asks for

| Model | Rate MAE | Coverage |
|---|---|---|
| **TyreMind** | **0.0037** | **100%** |
| Pooled regression | 0.0062 | 83% |
| Heilmeier | 0.0112 | 100% |
| **Cappello & Hoegh** | 0.0158 | **38%** |
| Naive | 0.0725 | **0%** |

**1.7× the next model. 4.3× the closest published paper.**

## 🥇 Robustness — 11 generator regimes, 32 seeds

**8 wins, 1 tie, 2 losses.** Biggest margin **+0.0115 ± 0.0007** on `no_scrubbed`,
which is exp18's collinear worst case — exactly where the model should help most.

**Both losses are the track-evolution axis**, and one (`strong_track`, six standard
errors) was not predicted. That is the clearest architectural weakness we have.

## 🥇 Independent truth engine — the generator shaped unlike us

| Target definition | Winner |
|---|---|
| Fresh-tyre baseline | Cappello & Hoegh |
| **Stint average (pre-registered primary)** | **TyreMind, +0.0086 ± 0.0017** |
| Best straight line | Fuel-corrected |

**Our margin survives and is 2.6× the synthetic one.** But note: *"the degradation
rate"* is three different numbers on a non-linear generator, and a different model
wins each. exp19's clean sweep was partly an artefact of our generator making all
three coincide.

## 🥇 Value in seconds and positions

> **"Using the naive estimate instead of ours would have cost 25.9 seconds and
> 5.26 positions per car per race."**

**Qualification that must travel with it:** our own value is **+1.03 ± 0.88 s** —
that interval contains zero. **We are as good as professional strategists, not
better.**

## 🥇 Confidence calibration — best of six

| Model | Claimed | Delivered | Gap |
|---|---|---|---|
| **TyreMind** | 31.9% | 24.0% | **−7.9%** |
| Pooled | 24.3% | 15.3% | −8.9% |
| Cappello & Hoegh | 38.7% | 23.0% | −15.8% |
| **Naive** | **47.3%** | **2.5%** | **−44.8%** |

Naive claims 47% confidence and delivers 2.5%. **That is the most damning number
we have about the standard method** — and it needs no wear data at all.

## 🥈 and 4th — where we lose, stated plainly

| Task | Result |
|---|---|
| **Lap-time forecasting** | **4th of 9.** Pooled 0.4088, us 0.6484 |
| **Pit timing** | **3-way tie.** 5.92 / 5.98 / 6.31, SEs ~0.6 |
| **Practice → race skill** | **2nd** — and **every model is negative** |
| **NASA C-MAPSS RMSE** | **22.7** vs published best 15.98 |

## 10.1 The router — the answer to all of it

No single model wins everything. **So we ship an orchestration that routes.**

```
forecast_lap_time          -> Pooled regression          (we route AWAY)
recover_degradation_rate   -> TyreMind
pit_timing                 -> TyreMind                   (TIE, marked as one)
practice_to_race           -> Cappello & Hoegh           (we route AWAY)
confidence                 -> TyreMind                   (TIE)
```

**5 tasks, 3 models. We route away from ourselves twice.**

Every route cites a result file, and the code **refuses to build a route without
evidence**. The tests re-read those files, so a re-run that changes a winner fails
the build rather than leaving the table asserting fiction.

> **"We tested nine models on five tasks. Three different models won. So we route
> each question to the one the evidence says is best — including away from our own
> model twice. Competitors ship one model and claim it wins everything."**

---

# PART 10 — WHY WE DID NOT TRAIN ANYTHING

## What we fit vs what we train

| Model | Method | Trained? |
|---|---|---|
| **TyreMind** | L-BFGS-B over **6 log-variances**, per session | ❌ **Fitted** |
| Pooled / Heilmeier / ARIMA / Cappello & Hoegh | least squares / MLE / MAP | fitted |
| **LightGBM, MLP** | gradient boosting, backprop | ✅ **trained** |

**The only two trained models are the two we built as baselines to beat.**

## Why fitting is better here

| | |
|---|---|
| **No train/test leakage possible** | There is no training set to leak from |
| **Works at a brand-new circuit, session one** | Nothing to retrain |
| **No retraining when regulations change** | 2026 cars arrive; we fit them that day |
| **12 s on a laptop, no GPU** | The MLP took 36 s and came **last of nine** |
| **Auditable** | Six interpretable parameters, not a weight matrix |

## "Why not deep learning?"

> "We did. It is rung 5 and it came **last of nine** — CRPS 1.94 against our 0.65,
> with a 95% interval covering 70%. LightGBM did better and its interval covers
> **62%**. And the bottleneck is not capacity: exp18 proves only **6%** of the
> fuel/tyre separation is recoverable from data at all. **No amount of GPU changes
> a singular information matrix.**"

---

# PART 11 — ADAPTIVE, FEASIBLE, SCALABLE

## "Is your pipeline adaptive to any kind of data?"

**Yes, and here is the precise claim.** We need four fields:

```
lap_time · compound · tyre_age · lap_number
```

A team with real telemetry gives us **strictly more** than we need, so we fit
their feed by construction. **There is no integration project.**

**Proven across domains:** the same unmodified code runs on NASA turbofan engines.
Tyre stint → engine life, tyre age → flight cycles, lap time → sensor channel,
laps-to-cliff → remaining useful life. **RMSE 22.7 cycles, 44% of predictions
conservative.**

**Where it does not adapt, honestly:**
- Wet running — different physics, excluded and reported as excluded
- Kerb strikes — impact, not gradual wear. Not modelled (Part 13)
- Traffic-heavy stints — under-reports by ~25% (Part 13)

## Feasible

| | |
|---|---|
| Fit time | **~12 s per session on a laptop** |
| Hardware | **No GPU** |
| Data size | 2.6 MB for four seasons |
| Internet | **None needed** — clone and run offline |
| Dependencies | numpy, scipy, pandas |

## Scalable

- Per-session fit is **independent** — 200 sessions is 200 parallel jobs
- The live path is a Kalman update per lap: **milliseconds**
- Adding a circuit needs **no retraining**
- 27 experiments reproduce from scripts in the repo

## Adoptable

- **488 automated tests**, including structural tests that fail the build if
  someone bypasses the repairing loader
- Every session ships a data-quality receipt
- Every route cites its evidence file
- Runs fully offline for a demo with no connectivity

---

# PART 12 — PHYSICS, AND THE TEST ON ANOTHER INDUSTRY

## What we implemented from the literature

| Module | Physics |
|---|---|
| `dynamics.py` | Path curvature, lateral/longitudinal acceleration, aero downforce `½ρC_L Av²`, per-corner load transfer, frictional power proxy |
| `thermal.py` | Two-state tread/carcass thermal model, working window, temperature-wear multiplier (reduced TRT) |
| `wear.py` | Archard `V = K·F·L/H` as an ablation baseline; energy-dissipation wear as the model in use |

## The physics sanity check that costs us nothing to show

**exp06:** we recover **circuit direction** — clockwise vs anticlockwise — from
per-corner tyre energy alone. **7 of 8 circuits correct.** It uses no wear data.
If the energy model were wrong it would get this wrong.

## The cross-industry test

**NASA C-MAPSS turbofan engines**, the most-cited benchmark in predictive
maintenance. Same code, no domain tuning.

**RMSE 22.7 cycles, MAE 17.9, all 100 test engines, 44% conservative.**

**State this honestly:** published deep models reach **15.98**. **We are not state
of the art and never claimed to be.** The claim is *transfer*: unmodified tyre code
on jet engines, in the competitive range. What *is* competitive is the uncertainty
— a 2025 paper presents 93.5–95.2% coverage as a new contribution; our conformal
layer reaches 95.2% over 69,206 laps.

## And the physics that did NOT work — two experiments

**exp25 and exp34** both tested whether measured frictional energy predicts
degradation. **Both refuted.**

- exp25: every significant stint-level correlation had the **wrong sign**; nothing
  survived at circuit level
- exp34: **147,550 measured laps**, every feature makes circuit transfer worse
  (−5.3%, p < 0.001), and a ridge sweep converges on the baseline **from above** at
  every penalty — the signature of noise

**Wear ∝ frictional power is the literature's model. It does not survive the
circuit boundary in public F1 data.** Two independent designs, same answer.

---

# PART 13 — HARD QUESTIONS

### "If there is traffic, how do you predict degradation?"

**We ran the experiment.** 3,220 stints. Traffic stints show *less* measured
degradation — 0.0938 s/lap clear vs 0.0693 in traffic, p = 2×10⁻¹⁰.

**Two explanations, and we distinguished them:**

- **(A) Real** — less pushing, less energy, less wear
- **(B) Artefact** — pace-limited by the car ahead, so the tyre's decline cannot
  reach the lap time

**It is (B), and (A) fails at its first step:** traffic *raises* frictional energy
(69.7 → **73.0 MJ/lap**, p = 1×10⁻¹¹) and raises braking. Following cars brake and
accelerate **more**, not less.

Three more facts: the deficit survives an energy clock (29.0% per MJ vs 26.1% per
lap); controlling for effort makes the traffic coefficient *larger* (suppression,
not mediation); and tyre age explains far less of the lap time in traffic
(**R² 0.649 → 0.449**).

> **Every lap-time-based method under-reports degradation in traffic by roughly a
> quarter. Ours included.**

**And the part that is ours to fix:** the model does not widen its interval when
this happens (0.0262 vs 0.0271, p = 0.88). We tried a fix — heteroscedastic
observation noise — and **it did not work**: the optimiser refits the base noise
level and absorbs the scaling. Reverted. The fix has to live at the reporting
layer, not inside the filter.

### "When a car hits the kerbs, how does the tyre degrade?"

**We do not model it, and we found out we cannot from public data.**

We tried to build kerb exposure from position telemetry and discovered the
**FastF1 position feed has no lateral coordinate.** It reports distance along a
single path. Proof needing no reference line: **15,581 of 15,922 car pairs, in
100% of sessions, are reported at the same coordinate at some instant.** Two F1
cars are 1.8 m wide.

The one usable proxy — stewards' track-limits adjudications — is **null on every
pre-registered test** (pooled ρ = −0.003).

**What we do instead:** a kerb strike shows up in the **residual band** as an
unexplained jump. We display it rather than absorbing it into a wear number a
strategist would then act on.

### "Why not just use the pooled regression, if it wins?"

> "We tested it — it is rung 3 of our own ladder and it does win lap-time
> forecasting, 0.41 against our 0.65, and we report that.
>
> Then we scored the same nine models on **274 real pit stops**. The pooled
> regression came **last** at 11.1 laps with a 6% hit rate, and could only answer
> **half** the stops. We came first-equal at 5.9 with 24%.
>
> On recovering the rate it is 0.0062 against our 0.0037, with 83% coverage
> instead of 95%. **It forecasts well and decides badly.** We are building a
> strategy tool, not a lap-time predictor."

**And volunteer this:** if a judge says "use both", **that is our architecture.**
The router keeps rung 3 and sends lap-time forecasting to it.

### "Your mentors said regression never works. Why does pooled win?"

**They were right, and pooled proves them right.** They meant *naive* regression:
`lap_time ~ tyre_age`, two parameters, no confounders. That model has rate MAE
**0.0725**, coverage **0%**, and gives impossible answers in 74% of races.

The pooled regression is naive regression **with the confounders added** — it is
our own rung 3. It works *because* it models fuel, track and traffic. **That is
the mentors' point demonstrated.**

What it still cannot do is bend. Its tyre term is a straight line, and only 56% of
stints are.

### "Your synthetic data is your own. Why should we trust it?"

**You are right, and we wrote that down before running.** The pre-registration
says the generator shares our estimator's assumptions and that the result is
*partially* mitigated, not fully.

**Three answers:**
1. Two of three validation legs need **no synthetic data** — the 74%
   impossible-answer result and the 274 real pit stops.
2. We built a **second generator** with a different functional form, and the
   margin survives (+0.0086 ± 0.0017).
3. **A rigged setup would win everything. We lose lap-time forecasting and report
   it.**

### "How do you know it is the best?"

**We do not claim that.** We claim: best at recovering the degradation rate (4.3×
the closest paper), best calibrated of six, and the only system here that knows
how wrong it is. **On forecasting we are 4th and on pit timing we are tied.**

---

# PART 14 — EVERYTHING WE GOT WRONG

**Read this part. It is the strongest thing in the document.**

## Twelve refuted hypotheses

| # | Idea | Outcome |
|---|---|---|
| 1 | Energy clock beats lap clock | No difference — energy varies only 2.3% lap to lap |
| 2 | True C1–C5 beats the weekend label | **6.4% worse** |
| 3 | Circuit geometry transfers | No effect, p = 0.22 |
| 4 | Temperature explains our bias | ρ = −0.022, p = 0.835 |
| 5 | Matching stint depth fixes the bias | **Significantly worse**, p = 0.016 |
| 6 | Driver style is a usable feature | 0.88% worse |
| 7 | Telemetry energy predicts degradation | Wrong sign, nothing at circuit level |
| 8 | Sector times break the collinearity | Identify *where*, not *how much* |
| 9 | Practice predicts race | **Negative skill for every model** |
| 10 | Measured energy transfers between circuits | Significantly harmful, p < 0.001 |
| 11 | Kerb exposure predicts degradation | Null, and not measurable from public data |
| 12 | Heteroscedastic noise fixes the traffic defect | **Made intervals narrower.** Reverted |

**Three of these killed our own ideas with our own tests.**

## Real bugs we found in our own work

| Bug | Consequence |
|---|---|
| **Fuel counter** counted surviving rows, not laps completed | Miami HARD understated by **72%**. Shanghai — zero gaps — unchanged, which is the control that proved the diagnosis |
| Five experiments bypassed the repairing loader | Old numbers looked plausible. Fixed with a **structural test that reads source code** |
| Two pit recommenders disagreed | The product served **lap 20**; the validated method said **lap 11**; the driver boxed on **lap 11** |
| `for row, probability in zip(...)` shadowed a Series | Latent `KeyError`, silent until something read it |
| Benchmark rigged **in our favour** | We raised on null input, competitors returned NaN. Fixed so all fail identically |
| exp29 smoke test committed as the real result | 3 seeds masquerading as 32 |
| **Three thresholds chosen without asking what magnitude matters** | exp26's `p05 > 0` (impossible), its permutation test (wrong reference), exp34's `any(v < baseline)` (no tolerance). **All three printed verdicts contradicting the table above them** |

## Claims we retracted

- *"exp03 flatters us"* — **backwards.** It flattered the **naive** comparator
- *"The FastF1 community uses the broken method"* — no evidence, retracted
- *"Circuit geometry is harmful"* — it is **no effect**; only thermal is harmful
- *"Published C-MAPSS results are 12–25, so we are competitive"* — too generous.
  We are at the weak end
- *"Our interval widens in traffic"* — did not replicate. It is a null

## Why this is the strongest section

> **A team that publishes twelve refuted hypotheses and seven of its own bugs is a
> team whose surviving claims can be believed.**

No competing write-up in our literature review reports a single negative result.

---

## The one-paragraph position

> *"Everyone can build a dashboard showing a degradation number. We built one that
> is right, that tells you how confident it is and is actually right that often,
> that proves the standard method is broken in 74% of real races, and that states
> mathematically how much of the answer is data and how much is assumption — 6%
> and 94%. Then we ran the same code on jet engines to show it is not an F1 trick.
> And where we lose, we route the question to whoever is better."*
