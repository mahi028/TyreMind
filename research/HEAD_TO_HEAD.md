# TyreMind vs the Field — Evidence Table

**Every claim here is either a direct quotation from a paper in `research/papers/`
or a number from `experiments/results/`. Nothing is asserted from memory.**

All 28 papers have been read; extracted text is in `research/notes/`.

---

## 0. Three corrections to earlier statements

Read these first. All three are things I told the team that were wrong or too
generous, and a mentor with the papers open would catch them.

### 0.1 We are **not** competitive on C-MAPSS point accuracy

I previously said published FD001 results "sit roughly in the 12–25 RMSE band, so
we are within the competitive range." That is technically true and materially
misleading. Here is the actual table, from
`07_prognostics_rul/UncertaintyAwareRUL_TurbofanAleatoric.pdf`:

| Method | FD001 RMSE |
|---|---|
| CNN-LSTM-Attention | **15.98** |
| Uncertainty-aware Inception-BiLSTM (2025) | 16.22 |
| Bidirectional LSTM | 17.60 |
| **TyreMind (unmodified F1 code)** | **22.67** |

**We are roughly 30–40% worse than specialised deep models on point accuracy.**

**Say it this way instead:** *"We are not the best RUL model and never claimed to
be — those models are purpose-built for turbofans and ours is a tyre model we did
not change a line of. The claim is transfer, not state of the art. What is
competitive is our uncertainty: that 2025 paper presents 93.5–95.2% interval
coverage as a new contribution — 'previously unattainable in CMAPSS literature' —
and our conformal layer reaches 95.2% over 69,206 laps with a distribution-free
guarantee rather than a learned aleatoric head."*

That framing is honest and still strong. The other one collapses under one
question.

### 0.2 The puncture labels do not exist — a claim I made and then disproved

I said real failure labels were "the breakthrough here" and quoted **Puncture 41,
Tyre 55, Wheel 88**. Those are **all-time counts back to 1950**, not modern F1. I
also said race control messages would multiply the positive class.

Both were wrong. We then ran the collection properly
(`scripts/build_failure_labels.py`, Pirelli era 2011–2022, **5,057 result rows**):

| Cause group | Events |
|---|---|
| Competing risks (Collision 190, Accident 102, Collision damage 45, …) | 351 |
| **Vehicle: Gearbox 54, Brakes 50, Suspension 37, Transmission 8, Driveshaft 2** | **151** |
| Wheel assembly (Wheel 16, Wheel nut 8) | 24 |
| **Tyre proper: Puncture 12, Tyre 5** | **17** |

Plus two further limits, both measured:

- **Detailed causes stop after 2022.** From 2023 the public results carry only
  `Finished` / `Lapped` / `Retired`.
- **Race control messages contain no punctures.** Three 2024 races checked:
  **0 tyre-related messages out of 249.**

**Conclusion: a validated puncture *probability* cannot be built from public data
— by us or by anyone.** We ship a **structural exposure index**, labelled as an
index and never as a percentage.

**Why this is a strong result.** It is the sharpest respectful question available
about any competing system, including the existing engine's "probabilistic
puncture risk scoring":

> *"We went looking for the labels needed to calibrate a puncture probability.
> There are seventeen in twelve seasons. So we report exposure rather than
> probability — and we would genuinely like to know what a probability here is
> calibrated against."*

**And the payoff.** The vehicle group has **151 events, nine times the tyre count**,
with workable per-family sizes. **The "full vehicle health monitoring" item the
current system lists as future work is better supported by public data than the
puncture metric it already ships.**

---

### 0.3 "Nobody calibrates" is true — but now I can cite someone saying it

I have been asserting this. It turns out a published paper states it directly, of
the field-standard simulators including Heilmeier's:

> *"These simulators are calibrated to reproduce race **times**; **none report
> probability calibration of race outcomes**, none model the box-now-vs-later
> counterfactual under common random numbers, and none operate against a live
> timing feed."* — Pitwall, arXiv:2607.06495, §2

Cite that rather than claiming it ourselves. A third-party critique of the field
standard is worth far more than our own.

---

## 1. Heilmeier et al. — the field-standard race simulation

**Papers:** `01_race_strategy_simulation/Heilmeier2020_VirtualStrategyEngineer.pdf`,
`…MonteCarloRaceSimulation.pdf`

### Their model, exactly

```
t_lap(l) = t_base + t_tire(a,c) + t_fuel(l) + t_car + t_driver + t_grid + t_pit
```

and the tyre term, their Equation (6), quoted verbatim from the paper:

> *"…the **linear tire degradation model**. Thus, the time loss `t_tire` is
> determined by tire age `a` and two coefficients `k₀` and `k₁`, which are
> dependent on the tire compound `c`:*
>
> **`t_tire(a, c) = k₀(c) + k₁(c)·a`**"

### Where they are genuinely good — say this before criticising

- The additive decomposition is the right skeleton and we use the same one.
- Their parameters are deliberately identifiable from public lap-time data — the
  same constraint we work under.
- The Monte Carlo paper handles safety cars and probabilistic race effects
  properly.
- The VSE keeps the **lap-time model physical** and puts the neural network only in
  the *decision* layer. That is the correct architecture and we copied it.

### Where it fails — with proof

| # | Failure | Proof | Source |
|---|---|---|---|
| 1 | **The tyre model is linear. Real degradation is not linear in 44% of stints.** | exp17: 2,827 stints, 77 races, BIC-selected broken-stick. **Linear 1,580 (55.9%)**, recovery 582 (20.6%), **cliff 337 (11.9%)**, warm-up 328 (11.6%). A breakpoint model beat the straight line on BIC in 1,247 stints (44.1%). | our data |
| 2 | **No track-evolution term** in Equation (1). | Track evolution is a real, measurable session effect; omitting it pushes its variance into `t_car`/`t_driver`/residual. | their Eq. (1) |
| 3 | **No traffic term** in the lap-time equation — traffic is handled by a separate overtaking model. | Our exp10 tested traffic gap as a bias driver: ρ = −0.167, p = 0.109. Small, but it is *in* the lap time, not only in overtakes. | their Eq. (1) |
| 4 | **No uncertainty on `k₁`.** The degradation coefficient is a point estimate. | Our exp12: our own model claimed 95% and delivered **75.5%**. Nobody knows a coefficient's error until they measure it. | our data |
| 5 | **Never validated for outcome calibration.** | *"none report probability calibration of race outcomes"* | Pitwall §2 |

### What we do instead — validated

| Their component | Ours | Validation |
|---|---|---|
| `k₀(c) + k₁(c)·a`, linear | Latent state with BIC-selected changepoint; four regimes | exp17, 2,827 stints |
| point estimate of `k₁` | conformal interval | exp12 94.7%; exp13 **95.2% / 69,206 laps** |
| no track evolution | explicit term | in the model |
| no traffic in lap time | explicit `traffic_index` covariate | exp10 |
| `k₁` fitted, identifiability unexamined | identifiability **bounded** | exp18: 520/520 collinear, CRB 0.0161 vs 0.0221 = **1.38×**, **6.0%** data-driven |

---

## 2. Cappello & Hoegh — the closest prior art

**Paper:** `08_state_space_filtering/StateSpaceApproach_TireDegradationF1.pdf`
(arXiv:2512.00640, Nov 2025)

### Their model

```
Observation:  y_t = α_t + γ·fuel_t + ε_t ,          ε_t ~ N(0, σ²_ε)  [or Skewed-T]
Process:      α_{t+1} = (1−I_pit)(α_t + ν) + I_pit·α_reset + η_t
Extension 2:  ν_{t+1} = (1−I_pit)(ν_t + β[c]) + I_pit·ν_reset
Priors:       ν ~ N⁺(0.05, 0.1²)   σ_ε ~ N⁺(0.3, 0.1²)   α_reset ~ N(69, 0.1²)
```

### Where they are genuinely good

- **Same model class as ours, published first.** Say so plainly.
- The **skewed-t observation model** is a better idea than our Gaussian: driver
  errors really are one-sided. We should adopt it as a variant.
- Clean Bayesian workflow, honest about data poverty.

### Where it fails — with proof

| # | Failure | Proof |
|---|---|---|
| 1 | **One driver, one race** (Hamilton, 2025 Austria). | Our corpus: **203 sessions, 92,326 laps, 84 event-seasons, 4 seasons.** Roughly 200× the data. |
| 2 | **Traffic is absorbed into observation noise** — `ε_t` covers "driver mistakes and the presence of other cars". | Lumping a systematic confounder into i.i.d. noise biases `ν`. We model it explicitly. |
| 3 | **No track-evolution term.** | Same omission as Heilmeier. |
| 4 | **Positivity is imposed, not measured.** A half-normal prior forces `ν > 0` because "a negative overall degradation rate would be" implausible. | Our exp14 **measures** how often the unconstrained estimate goes negative: **74.0% of 77 races, 53.4% of 208 stints.** A prior hides the symptom; an experiment quantifies it. |
| 5 | **Identifiability never addressed.** They state they "lean on moderately strong priors since we are relatively data poor" — without quantifying what the priors contribute. | exp18: **only 6.0%** of the fuel/tyre separation is data-driven. Their result rests on a prior doing **94%** of the work, and the paper does not say so because it does not know. **This is our single biggest contribution over the closest prior art.** |
| 6 | **Uncertainty assumed, not validated.** Bayesian credible intervals are correct *if the model is correct*. | We measured ours: claimed 95%, delivered 75.5%, fixed to 94.7%. |
| 7 | **Compound differences not statistically distinct** — they attribute it to lack of data. | Our exp08 at 197 estimates / 79 events / 4 seasons: true C1–C5 identity is **6.4% worse** than the weekend label, Wilcoxon p = 0.096. So it is not only a data problem — the effect may genuinely not be there. |

---

## 3. Pitwall — the agent, and the best system in this review

**Paper:** `01_race_strategy_simulation/Pitwall_CalibratedMonteCarloBriefings.pdf`

**Their numbers:** N = 2,000 Monte Carlo continuations per lap; calibrated on 126
races (2018–2024); held out on 2025–2026; winner-in-top-3 **90.3%** over 155
backtests; held-out Brier **0.0745**; live at two 2026 Grands Prix; trilingual.

### Where they are genuinely good — better than us, and we should say so

- **Outcome calibration.** They calibrate race-outcome probabilities and hold out
  whole seasons. We do not do this at all.
- **The faithfulness architecture.** Every sentence decomposed into typed claims,
  each verified against state; 81.9% retained, rest fall back to templates.
- **Production reality.** It ran live, twice.
- **The two-path insight** (§0 of the product plan).

**Do not claim to beat Pitwall on race-outcome prediction. We do not compete there.**

### Where the gap is — in their own words

> *"a compound-pace normalization that demonstrably fixes pathological live pit
> calls **worsens** finishing order prediction, because the raw, **confounded**
> per-race fit carries genuine race-specific signal."*

**Their tyre layer is, by their own description, confounded** — a per-race pace fit
that mixes tyre with everything else, or a "feasibility-capped H5 tyre model". They
route around the problem with two paths rather than solving it.

**That is precisely the layer we build.** The honest positioning:

> *"Pitwall is a better race-outcome simulator than anything we will build, and it
> is honest that its tyre layer is confounded. We are not competing with Pitwall —
> we are the component it says it is missing."*

### What we replicated independently

| | Pitwall | TyreMind exp05 |
|---|---|---|
| Finding | calibration-optimal ≠ decision-optimal | pooled regression best CRPS (0.395), TyreMind best rate MAE (0.0041) |
| Response | oracle path / decision path | same split adopted in our architecture |

---

## 4. Sulsters / Todd et al. — deep learning on team data

**Paper:** `06_ml_degradation/Sulsters2025_ExplainableTyreEnergyF1.pdf`

Deep learning (RNN, LSTM, GRU, Temporal Fusion Transformer) plus XGBoost and
linear regression, trained on **Mercedes-AMG PETRONAS historic race data**,
objective = minimise RMSE of tyre energies. Explainability via feature importance
and counterfactuals.

### Where they are good

They have a real team's internal data. **This is not a like-for-like competitor —
it is a demonstration of what becomes possible with the private channels listed in
`docs/data_doc.md` §2.3.** Do not pretend to beat it; note the data asymmetry.

### The structural criticism — quoted from the state-space literature

> *deep-learning methods "often lack interpretability and explicit uncertainty
> quantification — features that are crucial in operational race environments."*
> — Cappello & Hoegh §1

**Our own measurement of exactly that (exp05, 20 real races):**

| Model | CRPS ↓ | 95% interval actual coverage |
|---|---|---|
| Pooled regression | **0.395** | 86.4% |
| LightGBM | 0.469 | **61.4%** ← claims 95% |
| TyreMind | 0.645 | 81.1% |
| Neural network (MLP) | **1.548** ← last of six | 72.9% |

LightGBM's interval is wrong by 34 percentage points. That is the criticism made
concrete on our data.

---

## 5. TrackShift's existing engine

We have not seen the code, so every row is a **structural** claim plus **our own**
measurement of the cost. No adjectives.

| # | Their design | The structural ceiling | Our measurement |
|---|---|---|---|
| 1 | Five-parameter formula, **Bahrain V3** — one circuit | No evidence a single-circuit calibration travels | exp09, LOCO over 26 circuits: geometry **−3.3%, p = 0.22** (no effect); thermal features **actively hurt, p = 0.0012** |
| 2 | **Threshold-based** alerts on point estimates | A threshold with no interval is a coin flip near the line | exp12: our own model claimed 95%, delivered **75.5%** |
| 3 | **18 rules + 17-step ladder** | Cannot say "I don't know", cannot weigh the cost of being wrong. Supersede logic is the tell: rules collide, so rules arbitrate rules | Replaced by expected-cost minimisation — **with all 18 rules retained as a veto layer** |
| 4 | No negative results | Untested features silently degrade a model | exp09 proved it: thermal features made predictions significantly worse |
| 5 | "Tread remaining **depth**" | No public measurement of F1 tread depth exists to anchor millimetres against | data_doc §3.1 — searched Pirelli, FIA, Kaggle, HuggingFace, OpenF1, Ergast |
| 6 | "Real-time **traction coefficient**" | True μ needs load-cell and slip channels that are team-private | data_doc §2.3 |

**Rows 5 and 6 are the ones to handle carefully.** We are not saying their numbers
are wrong — we are saying the quantity is not measurable from public data, so we
report % of usable life and a normalised grip index instead. *"We would rather be
right than precise."*

---

## 6. Where we lose — put this slide in the deck

A team that names its own losses is believed on everything else.

| We lose to | On what | By how much | Why we accept it |
|---|---|---|---|
| Pooled regression | lap-time CRPS | 0.395 vs our 0.645 | It is a better *forecaster* and a worse *explainer*. It cannot say how much of the lap was tyre. Independently replicated by Pitwall as calibration-optimal ≠ decision-optimal. |
| CNN-LSTM / BiLSTM | C-MAPSS FD001 RMSE | 15.98–17.60 vs our 22.67 | Purpose-built for turbofans. Ours is unmodified tyre code. The claim is transfer, not SOTA. |
| Pitwall | race-outcome calibration | Brier 0.0745, 155 backtests | We do not model race outcomes at all. Different layer. |
| Sulsters/Mercedes | data access | team-internal channels | Not a method gap. A data gap. |
| Cappello & Hoegh | publication priority | they were first, Nov 2025 | Cite them. Beat them on scale, calibration and identifiability. |

---

## 7. What is genuinely ours — after reading all 28 papers

| # | Contribution | Evidence | Present in any paper read? |
|---|---|---|---|
| 1 | **Identifiability bound on fuel-vs-tyre separation** | 520/520 exactly collinear, r = 1.000, 11/11 singular information; CRB 0.0161 vs model 0.0221 = **1.38× optimal**; **6.0%** data-driven | ❌ **None.** Cappello & Hoegh have the identical model structure and never quantify what their prior contributes. |
| 2 | **Measured, not assumed, interval coverage for degradation** | 94.7% offline; 95.2% over 69,206 laps online | ❌ None for tyre degradation. Pitwall calibrates *outcomes*; the RUL literature calibrates *engines*. |
| 3 | **Falsification without ground truth** | 74.0% of 77 races physically impossible | ❌ None. Cappello & Hoegh prevent it with a prior instead. |
| 4 | **Four degradation regimes, measured** | 2,827 stints; linear only 55.9% | ❌ None. Heilmeier assumes linear; Cappello & Hoegh allow linearly growing. |
| 5 | **Scale + leave-one-out discipline** | 84 event-seasons, 26 circuits, LOCO + LOEO | 🟡 Heilmeier is multi-race but not LOCO-validated for degradation |
| 6 | **Six published negative results** | three killed our own hypothesis | ❌ None publishes this many |
| 7 | **Cross-domain transfer of the same code** | NASA turbofans | ❌ None |

---

## 8. What this review changes in our plan

| # | Action | Why |
|---|---|---|
| 1 | **Add Heilmeier `k₀(c) + k₁(c)·a` as a baseline** | It is the field standard. Beating it is worth more than beating naive regression, and exp17 predicts we win on the 44% of stints that are not linear. |
| 2 | **Add ARIMA(2,1,2)** | The baseline the closest prior art uses. |
| 3 | **Add Cappello & Hoegh's model as a baseline** | Same class. Scale should decide it. |
| 4 | **Adopt the skewed-t observation model as a variant** | Their idea, sound reasoning, cheap test. |
| 5 | **Correct the C-MAPSS claim everywhere** | §0.1. It currently overstates. |
| 6 | **Quote Pitwall on "none report probability calibration"** | Third-party critique beats self-assertion. |
| 7 | **Stop competing with Pitwall; position as its missing layer** | They say their tyre fit is confounded. That sentence is our market. |
| 8 | **Lead every technical conversation with identifiability** | It is the only thing on the list no paper has. |

---

## 9. The one-paragraph position

> *"The field's standard race simulator models tyre degradation as a straight line
> with two coefficients per compound. We measured 2,827 real stints and only 56% of
> them are straight. The closest published model to ours — a Bayesian state-space
> model from November — fits one driver in one race and forces the degradation rate
> positive with a prior, because the unconstrained estimate goes negative; we
> measured that it goes negative in 74% of real races, and we proved that only 6%
> of the fuel-versus-tyre separation is recoverable from the data at all, with the
> other 94% coming from physics assumptions nobody else has quantified. The best
> live agent in the field says, in its own paper, that its tyre layer is
> confounded. That layer is what we built, and unlike everyone in this review, we
> checked whether our error bars were telling the truth: they said 95%, they were
> delivering 75%, and now they deliver 95.2% across 69,206 laps."*
