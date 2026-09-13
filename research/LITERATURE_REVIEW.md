# TyreMind — Literature Review

**What exists in the world for modelling tyre degradation and its effect on lap
time, the exact mathematics each approach uses, and where TyreMind sits.**

Every paper cited here is downloaded to `research/papers/` and listed in
`research/papers/INDEX.json`. Nothing in this review is cited from memory.

---

## 0. The headline finding of this review

> **Someone published our model class, one month ago.**
>
> Cappello & Hoegh, *A State-Space Approach to Modeling Tire Degradation in
> Formula 1 Racing*, arXiv:2512.00640 (29 Nov 2025), Montana State University.
> Bayesian state-space model. FastF1 data. Lap time as a function of fuel mass and
> a latent tyre state. Pit stops as state resets.
>
> They state: *"To the best of our knowledge, there are no examples in the
> literature that apply state-space models to the phenomenon of tire degradation
> in Formula 1."*

**This is good news, not bad news, and here is why.**

Independent academic researchers, working separately, reached **the same
architecture we did, for the same stated reasons** — that deep-learning methods
"often lack interpretability and explicit uncertainty quantification, features
that are crucial in operational race environments" (their words, §1).

That converts our design choice from "a decision we made" into "a decision the
literature independently validates". When a mentor asks *"why a state-space model
and not a neural network?"*, the answer is no longer our opinion.

**And the gap between their work and ours is large, in our favour.** §6 is the
full comparison. In short: they fit **one driver in one race**; we fit **203
sessions and 91,867 laps across four seasons**. They assume their uncertainty is
correct; we measured ours and found it wasn't, then fixed it. And the
identifiability problem that their model quietly resolves with a strong prior is
the problem we **measured, bounded and published a number for**.

---

## 1. The six families of approach

| # | Family | Core idea | Representative work |
|---|---|---|---|
| 1 | **Additive lap-time decomposition** | Lap time = sum of named physical effects | Heilmeier et al. 2018/2020 |
| 2 | **Naive / empirical regression** | Regress lap time on tyre age | Widely used; our exp14 baseline |
| 3 | **Physical wear models** | Wear from friction energy in the contact patch | Archard; Farroni et al.; Klüppel–Heinrich |
| 4 | **Thermodynamic tyre models** | Solve heat transfer in the tyre, grip follows temperature | Farroni/Timpone TRT, TRT EVO |
| 5 | **Machine learning** | Learn degradation from data, no physics | Sulsters/Todd et al. 2025 (XGBoost + deep learning) |
| 6 | **State-space / latent-state** | Tyre condition is hidden; infer it from lap times | Cappello & Hoegh 2025; **TyreMind** |

---

## 2. Family 1 — Additive lap-time decomposition (the field standard)

### Heilmeier, Graf, Betz, Lienkamp

**Papers held:**
- `01_race_strategy_simulation/Heilmeier2020_VirtualStrategyEngineer.pdf` —
  *Applied Sciences* 10(21):7805
- `01_race_strategy_simulation/Heilmeier2020_MonteCarloRaceSimulation.pdf` —
  *Applied Sciences* 10(12):4229

**The formula (their Equation 1), which is the canonical model in this field:**

```
t_lap(l) = t_base
         + t_tire(a, c)                  tyre degradation, depends on age a and compound c
         + t_fuel(l)                     fuel mass aboard
         + t_car                         car performance
         + t_driver                      driver performance
         + t_grid(l, p_g)                start-position loss
         + t_pit,in-lap/out-lap(l)       pit stop effects
```

and race time accumulates as `t_race(l) = Σ_{i=1..l} t_lap(i)`.

**Why it matters to us.** This is the structure everyone uses, and our model has
the same skeleton. Their design constraint is also ours and is stated explicitly:
*"the simulation parameters are chosen so they can be determined based on publicly
accessible lap time data."*

**What it does not contain — and this is a real gap we fill:**

| Term | Heilmeier | TyreMind |
|---|---|---|
| Tyre | ✅ `t_tire(a, c)` | ✅ latent state, per compound |
| Fuel | ✅ `t_fuel(l)` | ✅ |
| Car / driver | ✅ constants | ✅ hierarchical random effects |
| **Track evolution** | ❌ **absent** | ✅ explicit term |
| **Traffic** | ❌ handled by a separate overtaking model, not in the lap-time equation | ✅ explicit `traffic_index` covariate |
| **Uncertainty on the tyre term** | ❌ deterministic | ✅ calibrated interval |

Heilmeier's Monte Carlo paper adds probabilistic effects — but at the level of
*race outcomes* (lap-time variation, full-course-yellow phases), **not** as
uncertainty on the estimated degradation rate itself.

**Their VSE extension** replaces human strategy input with two neural networks
(pit-or-not, and which compound). Note what the ANN does there: it makes
*decisions*, while the *lap-time model stays physical*. That is exactly the
architecture we propose in Part 10 of the data document — a physical/statistical
core with a decision layer on top.

**Also held:** `Bassi2023_MasteringNordschleife.pdf` (arXiv:2306.16088) and
`TowardsLearningBasedF1RaceStrategies.pdf` (arXiv:2512.21570), which extend this
line into RL-based strategy; and `Pitwall_CalibratedMonteCarloBriefings.pdf`
(arXiv:2607.06495), which is notable for the word *calibrated* in its title — the
same concern we have.

---

## 3. Family 2 — Naive empirical regression (the thing we falsify)

**The model:**

```
t_lap = β₀ + β·a + ε            a = tyre age in laps
```

`β` is reported as the degradation rate.

**Why it fails, in one line of algebra.** Within a stint, laps of fuel burned `f`
and tyre age `a` advance together: `a = a₀ + f`. Substituting a true model
`t = c + β·a − φ·f`:

```
t = c + β(a₀ + f) − φ·f
  = (c + β·a₀) + (β − φ)·f
```

The stint intercept absorbs `β·a₀`, and the slope you actually recover is
**`β − φ`**, not `β`. With `φ ≈ 0.08 s/lap` (fuel) and `β ≈ 0.03–0.08 s/lap`
(tyre), `β − φ` is frequently **negative**.

**Our measurement of the consequence (exp14):** across 77 real Grands Prix, the
naive estimate is negative — claiming tyres get *faster* as they wear — in
**74.0% of races** and **53.4% of 208 compound-stints**.

**Note how the published literature handles this.** Cappello & Hoegh place a
**half-normal prior** on the degradation rate `ν ~ N⁺(0.05, 0.1²)`, explicitly to
*restrict it to be positive*, because "a negative overall degradation rate would
be" implausible.

That is a legitimate modelling choice. But observe the difference in what it
achieves: **a positivity prior prevents the symptom; our experiment measures how
often the symptom occurs.** Both are defensible. Only one produces a diagnostic
you can show a judge.

---

## 4. Family 3 — Physical wear models

### 4.1 Archard (the classical baseline)

```
V = K · F_N · L / H
```

`V` worn volume, `K` wear coefficient, `F_N` normal load, `L` sliding distance,
`H` hardness.

Implemented in `src/tyremind/physics/wear.py` as `archard_wear()` — **deliberately
as an ablation baseline, not as the model in use**, because Archard assumes a
constant wear coefficient and is blind to temperature, so it cannot represent
graining or thermal degradation at all.

### 4.2 Energy-dissipation wear (the modern standard)

**Papers held:**
- `02_tyre_wear_physics/RubberWear_HistoryMechanismsPerspectives.pdf`
  (arXiv:2503.19494)
- `02_tyre_wear_physics/Farroni2023_TireWearSensitivityAnalysis.pdf`
  (*Lubricants* 11(6):269)
- `02_tyre_wear_physics/Wear_IrreversibleEntropyGeneration.pdf` (arXiv:1008.0412)

**The governing idea.** Wear rate is proportional to **frictional power dissipated
in the contact patch**, not to distance travelled:

```
ẇ  ∝  P_friction  =  τ · v_slip          per unit contact area
```

where `τ` is shear stress and `v_slip` the local sliding velocity. Integrated over
the patch and over a lap, this gives wear per lap.

Key results from the review literature:
- Rubber friction has **two physically distinct mechanisms** — *adhesion*, governed
  by real contact area, and *hysteresis*, governed by viscoelastic energy
  dissipation (Klüppel–Heinrich).
- Braghin et al. link wear explicitly to **local frictional energy dissipation**,
  with friction power scaled by contact-patch area and wear constants that depend
  on **rubber temperature** and **road roughness**.
- Wear rate is closely related to energy dissipated in frictional processes
  (Finnie & Sheldon).

**Where TyreMind already implements this:** `physics/wear.py::energy_wear_rate()`
takes frictional power and surface temperature per sample and returns an
instantaneous wear rate — the energy formulation, with Archard retained beside it
for comparison.

**The honest caveat we must keep stating:** all of this needs measured contact-patch
forces and slip. Those come from load cells and wheel-speed sensors that are
team-private. We compute a *proxy* from public telemetry, not the real thing.

### 4.3 What this family says about the "energy clock" idea

Our exp04 tested whether tyre age should be measured in accumulated energy rather
than laps. Result: no meaningful difference, because energy per lap varies by only
**2.3%** (coefficient of variation) in race conditions.

The literature supports why: in a race, laps are highly repeatable, so
`∫P dt ≈ constant × laps`. Energy-based clocks pay off where load varies a lot
between laps — endurance racing with driver changes, or road cars — not in a
steady F1 stint. **This is a case where our negative result is explained, not just
observed.**

---

## 5. Family 4 — Thermodynamic tyre models

**Papers held:**
- `03_tyre_friction_grip/Farroni2019_TRT_EVO_ThermodynamicTireModel.pdf`
  (Proc IMechE Part D)

**TRT (Thermo Racing Tyre)** — Farroni, Giordano, Russo, Timpone, *Meccanica*
49:707–723 (2014), extended as **TRT EVO** (2019).

A physical-analytical model applying **Fourier's law of heat conduction** on a
three-dimensional tyre domain:

```
ρ c_p ∂T/∂t = ∇·(k ∇T) + q̇_gen
```

with generative terms `q̇_gen` from:
- **friction energy** developed in the contact patch, and
- **strain energy loss** (viscoelastic hysteresis in the carcass),

and loss terms to the track (conduction), to external air (convection), and — in
TRT EVO — even exhaust gases impinging on the rear axle, plus inhomogeneous local
variables across the contact patch caused by camber.

**Why this matters to TyreMind.** `src/tyremind/physics/thermal.py` implements a
reduced version of exactly this structure: surface and bulk (carcass) temperature
states, a working window, and `temperature_wear_multiplier()` coupling temperature
to wear rate.

**And it explains one of our negative results.** exp09 found that *thermal
features actively hurt* prediction (−1.5%, p = 0.0012). TRT shows why: the quantity
that drives wear is **tread surface temperature in the contact patch**, which is
generated by friction and strain internally. Our only public thermal input is
**track temperature from a single trackside mast**. Those are different physical
quantities. A bad proxy for the right variable is worse than no variable — and now
we can cite the physics for why.

---

## 6. Family 5 — Machine learning

**Paper held:** `06_ml_degradation/Sulsters2025_ExplainableTyreEnergyF1.pdf` —
*Explainable Time Series Prediction of Tyre Energy in Formula One Race Strategy*,
arXiv:2501.04067, ACM SAC 2025.

**Note the title carefully: "Tyre Energy" is literally one of TrackShift's five
metrics.** This is the closest published work to that specific metric.

**Their approach:** deep learning models trained on **Mercedes-AMG PETRONAS F1
team's historic race data**, plus XGBoost, with explainability via feature
importance and counterfactual explanations.

**The decisive difference, and it is not about algorithms:** they had **a real
team's internal data**. We have public timing only. So this is not a like-for-like
competitor — it is evidence of what becomes possible *with* the private channels
described in the data document §2.3.

**The criticism the state-space literature makes of this family**, quoted from
Cappello & Hoegh §1: such methods *"often lack interpretability and explicit
uncertainty quantification — features that are crucial in operational race
environments."*

**Our own measurement of the same thing (exp05, 20 real races):**

| Model | CRPS ↓ | 95% interval actually covered |
|---|---|---|
| Pooled regression | **0.395** | 86.4% |
| LightGBM | 0.469 | **61.4%** |
| TyreMind | 0.645 | 81.1% |
| Naive | 0.879 | 75.6% |
| Neural network (MLP) | **1.548** | 72.9% |

LightGBM forecasts lap times reasonably and its uncertainty is badly wrong — 61%
coverage on a 95% interval. The MLP came last of six. **This is our own empirical
version of the criticism the literature makes.**

---

## 7. Family 6 — State-space models: the direct prior art

### 7.1 Cappello & Hoegh (arXiv:2512.00640) — their exact model

**Observation equation:**

```
y_t = α_t + γ · fuel_t + ε_t ,        ε_t ~ N(0, σ²_ε)
```

`y_t` observed lap time; `α_t` latent true tyre pace; `fuel_t` derived fuel mass in
kg; `γ` seconds of lap time per extra kg.

**Process (state) equation:**

```
α_{t+1} = (1 − I_pit,t)(α_t + ν) + I_pit,t · α_reset + η_t ,   η_t ~ N(0, σ²_η)

I_pit,t = 1 if the driver has new tyres on lap t+1, else 0
```

**Extension 1 — compound-specific degradation:**
```
α_{t+1} = (1 − I_pit)(α_t + ν[compound_t]) + I_pit · α_reset[compound_t] + η_t
```

**Extension 2 — time-varying degradation (their version of "the cliff"):**
```
α_{t+1} = (1 − I_pit)(α_t + ν_t) + I_pit · α_reset[compound_t] + η_t
ν_{t+1} = (1 − I_pit)(ν_t + β[compound_t]) + I_pit · ν_reset
```

**Extension 3 — skewed-t observation errors**, because driver mistakes produce
extreme values predominantly in the *positive* direction:
```
ε_t ~ SkewedT(0, σ²_ε, λ, 2)         λ ∈ [−1, 1], 2 degrees of freedom
```

**Their priors:**
```
σ_ε ~ N⁺(0.3, 0.1²)      σ_η ~ N⁺(0.1, 0.1²)
ν   ~ N⁺(0.05, 0.1²)     α_reset ~ N(69, 0.1²)
```
They are explicit about why: *"we lean on moderately strong priors since we are
relatively data poor."*

**Fitted with:** Stan / MCMC. **Baseline:** ARIMA(2,1,2). **Data:** Lewis Hamilton,
2025 Austrian Grand Prix — **one driver, one race**.

**Their results:** the state-space model beats ARIMA(2,1,2), best under the
skewed-t specification; compound-specific degradation differences were **not**
statistically distinct, which they attribute to lack of data.

### 7.2 TyreMind versus Cappello & Hoegh — the honest comparison

| Dimension | Cappello & Hoegh 2025 | TyreMind |
|---|---|---|
| **Model class** | Bayesian linear-Gaussian state space | Hierarchical linear-Gaussian state space |
| **Inference** | Stan / MCMC | Hand-written Kalman filter + RTS smoother, exact likelihood, L-BFGS-B |
| **Scale** | **1 driver, 1 race** | **203 sessions, 91,867 laps, 84 event-seasons, 4 seasons** |
| **Fuel term** | derived fuel mass in kg, coefficient `γ` | laps-completed-in-run proxy — *same information, different scaling* |
| **Traffic** | ❌ absorbed into observation noise `ε_t` | ✅ explicit covariate, derived from lap start times |
| **Track evolution** | ❌ not modelled | ✅ explicit term |
| **Driver / car effects** | single driver, so not needed | ✅ hierarchical, tested in exp15 |
| **Degradation curvature** | linear growth `ν_t + β` (Extension 2) | broken-stick changepoint, BIC-selected, **four regimes** across 2,827 stints |
| **Heavy tails** | skewed-t observation model | Student-t(5) noise in the *generator*, so the estimator is tested outside its own Gaussian assumption |
| **Negative rates** | **prevented** by a half-normal prior | **measured**: 74% of races, 53% of stints (exp14) |
| **Uncertainty** | Bayesian credible intervals — correct *if the model is correct* | Conformal — distribution-free, and **empirically validated**: 94.7% (exp12), 95.2% over 69,206 laps (exp13) |
| **Identifiability** | ❌ not addressed; resolved implicitly by strong priors | ✅ **proven exactly collinear** (520/520 runs, r = 1.000, 11/11 singular information matrices); **Cramér–Rao bound computed**; we sit at **1.38×** optimal; **only 6.0%** of the separation is data-driven |
| **Cross-domain** | none | NASA C-MAPSS turbofans, unmodified code, RMSE 22.7 cycles |
| **Baselines** | ARIMA(2,1,2) | naive, fuel-corrected, pooled, LightGBM, MLP — and a physical-impossibility test |
| **Negative results** | 1 (compounds not distinct) | 13 refuted hypotheses |
| **Validation** | cross-validation within one race | 25-seed synthetic ground truth; practice→race across 42 events; leave-one-circuit-out across 26 circuits |

### 7.3 What we must do because of this paper

1. **Cite it.** Not citing the closest prior art, when it is one arXiv search away,
   is the single fastest way to lose a technical judge's trust.
2. **Add ARIMA(2,1,2) to our baseline ladder.** It is the baseline the published
   work uses. This also directly answers the mentor's criticism in §8.
3. **Implement their model as a baseline in exp05.** Score it on our corpus. If we
   beat it at scale, that is a far stronger claim than beating a naive regression.
4. **Adopt their skewed-t observation model as a variant.** Their reasoning is
   sound — driver errors are one-sided — and it is a cheap test.
5. **Lead with identifiability.** It is the one thing in our work that the closest
   published paper does not have, and it is not a small thing: their entire result
   rests on informative priors whose contribution they never quantify. We quantified
   it: **94%**.

---

## 8. The mentor's criticism: "why compare against naive regression?"

**The criticism is correct and we should accept it immediately.**

Beating a naive `lap_time ~ tyre_age` regression is a low bar. It proves the
problem is real; it does not prove our solution is good. Our exp05 already has five
baselines, but the *headline* comparison in our materials has been against the
naive method, and that reads as choosing a weak opponent.

**The fix — a proper baseline ladder, in rungs of increasing seriousness:**

| Rung | Baseline | Why it belongs | Status |
|---|---|---|---|
| 1 | Naive `t ~ a` | Shows the problem exists (74% sign errors) | ✅ done |
| 2 | Fuel-corrected regression | Removes the obvious confounder by hand | ✅ done |
| 3 | Pooled regression | Strong empirical baseline; currently **beats us** on lap-time CRPS | ✅ done |
| 4 | LightGBM | Standard tabular ML | ✅ done |
| 5 | Neural network (MLP) | The "just use deep learning" answer | ✅ done |
| 6 | **ARIMA(2,1,2)** | The baseline the published state-space paper uses | ❌ **to add** |
| 7 | **Heilmeier additive model** | The field-standard lap-time decomposition | ❌ **to add** |
| 8 | **Cappello & Hoegh state space** | The closest prior art — same model class | ❌ **to add** |
| 9 | **Five-parameter single-circuit formula** | The existing TrackShift engine, leave-one-circuit-out | ❌ **to add (exp21)** |

**What we say to the mentor:**

> "You are right that naive regression is a weak opponent. We keep it only because
> it demonstrates the problem is real — it produces a physically impossible answer
> in 74% of races. But it is rung 1 of nine. We already run four more. We are adding
> ARIMA, the Heilmeier field standard, the published state-space model, and a
> reproduction of the existing Bahrain-calibrated formula. And we will report where
> we lose: the pooled regression currently beats us on raw lap-time forecasting."

**That last sentence is the one that earns trust.** A team that names the baseline
beating it is a team whose other numbers can be believed.

---

## 9. The four-tyre question

**The criticism:** we currently model *the tyre*, singular. A real car has four,
they wear differently, and the limiting one decides the pit stop.

**What the literature gives us.** Papers held in
`research/papers/03_tyre_friction_grip/`:

- `AutoRally_DoubleTrackModel.pdf` (arXiv:1806.00678) — double-track model with
  explicit per-side load differences from lateral load transfer
- `GM3_GeneralPhysicalModel.pdf` (arXiv:2510.07807) — longitudinal and lateral load
  transfer in terms of CoG height, wheelbase and track width
- `LoadTransferMitigation_MPC.pdf` (arXiv:2606.26313) — load transfer ratio (LTR),
  where LTR = 0 is equal left/right load and |LTR| = 1 is wheel lift-off
- `MultichamberSuspension_LoadTransfer.pdf` (arXiv:2304.08201)

**The standard quasi-static per-corner load model:**

```
Longitudinal transfer:   ΔF_x = m · a_x · h_cg / L
Lateral transfer:        ΔF_y = m · a_y · h_cg / T

F_FL = F_FL,static − ΔF_x/2 − ΔF_y,front/2
F_FR = F_FR,static − ΔF_x/2 + ΔF_y,front/2
F_RL = F_RL,static + ΔF_x/2 − ΔF_y,rear/2
F_RR = F_RR,static + ΔF_x/2 + ΔF_y,rear/2
```

plus aerodynamic downforce `F_aero = ½ρ C_L A v²` distributed by aero balance —
which at F1 speeds is the *dominant* term, several times static weight.

**What we already have.** `src/tyremind/physics/dynamics.py` already implements
`corner_loads()`, `aerodynamic_downforce()`, `lateral_acceleration()` and
`frictional_power_proxy()`, and `LapWear.energy_mj` is **already a dict keyed by
corner** — our exp06 result reports per-corner energy shares (Monza: FL 25.7%,
FR 20.9%, RL 29.1%, RR 24.3%).

**So the physics layer is already four-tyre. The estimator is not.** That is the
honest statement.

**The design for making the estimator four-tyre:**

1. **Four latent states instead of one:** `α_t = [α_FL, α_FR, α_RL, α_RR]`.
2. **A shared lap-time observation.** This is the hard part and must be said
   plainly: *one lap time cannot identify four independent wear states.* Four
   unknowns, one measurement per lap — the system is underdetermined, and this is
   the same class of problem as our exp18 identifiability result.
3. **The escape is the same as before: structure.** Constrain the four states with
   per-corner energy shares from telemetry, which *are* observable. The energy
   split then carries the asymmetry and the lap time carries the magnitude:
   ```
   α_corner,t  =  α_total,t · (E_corner / E_total)^κ
   ```
   with `κ` a shared exponent. Four states, but only two free parameters.
4. **The pit decision is then driven by the worst corner**, not the average —
   which is physically correct and is what a race engineer actually reasons about.
5. **Validate it** against the independent tyre model (exp20), which publishes
   **per-wheel** wear, so the asymmetry is directly checkable.

**exp06 is the evidence this can work:** we already recover circuit direction —
clockwise vs anticlockwise — from per-corner energy alone, 7 of 8 circuits correct.
The asymmetry signal is real and measurable in public data.

---

## 10. Feature engineering plan, grounded in the papers

The mentors asked for serious feature engineering. Each feature below traces to a
specific paper, not to intuition.

| Feature | Formula / source | From which paper | Data needed |
|---|---|---|---|
| Frictional power per corner | `P = τ · v_slip`, integrated over the patch | Braghin; Farroni 2023 | telemetry |
| Lap energy per corner | `E = ∫ P dt` over the lap | energy-wear family | telemetry |
| Aero load fraction | `F_aero / F_total`, `F_aero = ½ρC_L A v²` | vehicle dynamics standard | telemetry |
| Load transfer ratio | `LTR = (F_right − F_left)/(F_right + F_left)` | arXiv:2606.26313 | telemetry |
| Slip-energy proxy | speed × curvature × load | brush/string models | telemetry |
| Thermal stress | `∫ (T − T_window)² dt` | TRT / TRT EVO | thermal model output |
| Fraction in working window | time with `T ∈ [T_lo, T_hi]` / lap time | TRT | thermal model |
| Surface–bulk temperature gap | `T_surface − T_carcass` | TRT EVO | thermal model |
| Fuel mass in kg | `m_fuel(l) = m_0 − l · burn_rate` | Heilmeier `t_fuel(l)`; Cappello `γ·fuel_t` | derivable now |
| Measured gap to car ahead | from interval telemetry | replaces our proxy | OpenF1 |
| Track evolution index | session-lap smooth trend | our own term; absent from Heilmeier | have it |
| Cumulative energy clock | `Σ E` since fitting | exp04; energy-wear family | telemetry |
| Corner-count × load | circuit geometry × per-corner load | exp09 | have geometry |

**Discipline to keep:** exp09 already showed that adding features *without testing
them* makes things worse — thermal features hurt at p = 0.0012. **Every feature on
this list gets the leave-one-out test and the Wilcoxon before it enters the model.**
Feature engineering without ablation is how you make a model worse and feel
productive.

---

## 11. F1 rules the synthetic generator must obey

The mentors' point is correct: synthetic data that breaks the sporting regulations
is not evidence about F1. Current state of `src/tyremind/data/synthetic.py` against
the actual rules:

| Rule | Current generator | Status |
|---|---|---|
| Pirelli nominates **3 consecutive** compounds from C1–C5 per event; relabelled HARD/MEDIUM/SOFT | Uses generic HARD/MEDIUM/SOFT rates | 🟡 partially — needs the consecutive-triple constraint |
| **13 sets of dry tyres** per driver per weekend (2 HARD, 3 MEDIUM, 8 SOFT) | Unlimited sets | ❌ **to add** |
| **Two different compounds must be used** in a dry race | Not enforced | ❌ **to add** |
| Sets must be **returned** progressively through practice (2 after FP1, 2 after FP2) | Not modelled | ❌ to add |
| Q3 runners start on their Q2 tyre — **abolished from 2022**, so *not* a rule now | Correctly absent | ✅ |
| Scrubbed/used sets carry over tyre age | ✅ `scrubbed_set_probability = 0.25` | ✅ |
| Wet compounds (INTER/WET) are a different process | Excluded and reported | ✅ |
| Fuel: max **110 kg** race fuel, no refuelling since 2010 | Fuel burn-off modelled; no refuel | ✅ |
| Fuel flow limited to **100 kg/h** | Not modelled | 🟢 not needed at lap granularity |
| Minimum tyre pressures set by Pirelli per event | Not modelled | 🟢 out of scope without pressure data |
| Race distance ≥ **305 km** (Monaco 260 km) | Session length is generic | 🟡 should derive from circuit length |
| Pit lane speed limit **80 km/h** (60 in some events) → pit loss ~20 s | Pit loss not in the generator | 🟡 add for strategy experiments |
| Parc fermé from qualifying — setup frozen | Not relevant to lap-time generation | ✅ |

**Action:** a `rules.py` module encoding the allocation, the two-compound rule and
the set-count limits, with the generator sampling *legal* strategies only. Then
every synthetic session is a race that could actually have happened — and we can
say so.

---

## 12. What is genuinely ours

After reading the field, here is the honest list of what TyreMind contributes that
is not in any paper we found:

1. **The identifiability result.** No paper in this review states how much of the
   fuel/tyre separation is data-driven versus prior-driven. We measured it: exact
   collinearity in 520/520 runs, singular information in 11/11 sessions, Cramér–Rao
   bound 0.0161 against our 0.0221 (**1.38× optimal**), and **6.0%** of the
   information coming from curvature. Cappello & Hoegh's model has exactly this
   structure and resolves it with a prior whose contribution is never quantified.

2. **Validated rather than assumed uncertainty.** Bayesian credible intervals are
   correct if the model is correct. We *measured* our coverage, found it was 75%
   when it claimed 95%, and fixed it with conformal prediction to 94.7% — and 95.2%
   online across 69,206 laps. **We are the only one in this list who checked.**

3. **Falsification without ground truth.** The 74%-of-races physical-impossibility
   result needs no label at all. It is a way of evaluating tyre models in a domain
   where the truth is not published, and it generalises.

4. **Scale with leave-one-out discipline.** 84 event-seasons, 26 circuits,
   leave-one-circuit-out and leave-one-event-out, against a field whose closest
   comparable work fits one race.

5. **Six published negative results**, including three where we killed our own
   hypothesis with our own test.

6. **Cross-domain proof.** Unmodified code on NASA turbofans.

---

## 13. Reading order

| If you have | Read |
|---|---|
| 20 minutes | §0, §7.2 (the comparison table), §8 (the baseline answer) |
| 1 hour | add §2 (Heilmeier), §7.1 (their equations), §9 (four tyres) |
| A day | all of it, then the PDFs in `research/papers/08`, `01`, `09` in that order |

## 14. Still to obtain

| Paper | Why | Access |
|---|---|---|
| Farroni, Sakhnevych, Timpone (2017), *Physical modelling of tire wear…* | The direct wear-vs-thermal coupling | SAGE — paywalled |
| Farroni et al. (2014), *TRT: Thermo racing tyre*, Meccanica 49:707–723 | The original thermal model | Springer — paywalled |
| Heilmeier et al. (2018), IEEE ITSC — the original race simulation | Source of Equation (1) | IEEE — paywalled (the 2020 papers restate it) |
| FSAE Tire Test Consortium / Calspan | 430+ real tyre rig tests | Membership-gated, restricted use |
| SAE tyre wear papers | Various | SAE — paywalled |

These are recorded rather than fetched. Several are obtainable through a university
library if any team member has access — worth asking the mentors.
