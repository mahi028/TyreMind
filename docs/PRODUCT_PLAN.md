# TyreMind — Product Plan

**From existing data → live data → AI agent → dashboard.**
Who it talks to, what it says, why it is better, and what we build in what order.

Companion to `research/LITERATURE_REVIEW.md` (the science) and `docs/data_doc.md`
(the data and experiments).

---

## 1. Two systems already exist that we must position against

Before designing anything, this is what is already out there. Both were found in
this week's literature search and both are downloaded to `research/papers/`.

### 1.1 The model: Cappello & Hoegh (arXiv:2512.00640, Nov 2025)

Bayesian state-space model for F1 tyre degradation on FastF1 data. **Our model
class, published.** One driver, one race. Full comparison in the literature review
§7.2. Short version: same architecture, 200× less data, no traffic or track
evolution term, uncertainty assumed rather than measured, and the identifiability
problem resolved by a strong prior whose contribution is never quantified.

### 1.2 The agent: "Pitwall" (arXiv:2607.06495, Jul 2026) — **read this section twice**

A production system that ingests the official F1 live-timing stream, maintains a
probabilistic race state, runs vectorised Monte Carlo (N = 2,000 per-lap
continuations), and **verbalises calibrated strategy briefings in three
languages**. Calibrated on 126 races (2018–2024), held out on 2025–2026.
Winner-in-top-3 **90.3%** over 155 backtests, held-out Brier **0.0745**. Ran live
at the 2026 Austrian and British Grands Prix.

**So "AI agent that explains race strategy" is not a novel product idea. It exists
and it works.** We must know that before we pitch it.

**But look at what their tyre layer actually is.** They use *"a compound-pace
normalization"*, and they report that it *"demonstrably fixes pathological live pit
calls"* yet *"worsens finishing order prediction, because the raw, confounded
per-race fit carries genuine race-specific signal."*

**The word they use is "confounded". That is our problem statement, in their
paper, named as an unresolved tension.** Pitwall is a superb race-outcome
simulator sitting on top of a tyre model that is admittedly confounded. That is
precisely the gap we fill.

### 1.3 The finding we independently replicated — lead with this

Pitwall's central methodological claim:

> *"calibration-optimal is not decision-optimal"* — they gate every component on
> two independent held-out tests (outcome calibration vs decision fidelity), the
> two criteria **repeatedly disagree**, and they route each component down the path
> it improves: an **oracle path** for probabilities, a **decision path** for
> recommendations.

**We found the same thing, separately, in exp05:**

| Model | CRPS (forecast quality) ↓ | Degradation-rate MAE (decision quality) ↓ |
|---|---|---|
| Pooled regression | **0.395** ← best | 0.0068 |
| TyreMind | 0.645 | **0.0041** ← best |
| Naive | 0.879 | 0.0748 (coverage **0%**) |

The model that forecasts lap times best is **not** the model that recovers the
decision-relevant quantity best. We reported that as an awkward result. It is not
awkward — it is an independently published finding about this class of system, and
we replicated it in a different setting.

**This is the strongest single thing we can say to a technical judge**, because it
shows we understand *why* being third on CRPS is not a weakness.

**Architectural consequence we adopt:** two paths, exactly as Pitwall does.

```
                  ┌─→ ORACLE PATH    → probabilities, scored on calibration (Brier, coverage)
   estimator ─────┤
                  └─→ DECISION PATH  → recommendations, scored on decision fidelity (regret)
```

---

## 2. THE DECISION: who does the agent talk to?

You asked us to decide. Here is the decision and the reasoning.

### The candidates, scored honestly

| Candidate | What they need | Can they act on a probability? | Verdict |
|---|---|---|---|
| **The driver** | "Box this lap." Nothing else. | No — cognitively saturated at 300 km/h | ❌ **Never.** One voice channel exists (the race engineer) and it exists for safety reasons. Inserting an AI into it is a regulatory and safety non-starter, and a driver cannot act on "61% probability". |
| **Tyre / pit crew engineer** | One imperative, ~30 s ahead: "mediums, box now" | No | ❌ **Not a product.** They execute a decision already made. Real need, but it is one line of output, not a system. |
| **Race strategist (pit wall)** | Recommendation + confidence + cost of being wrong + what would change it | **Yes. This is their entire job.** | ✅ **PRIMARY USER** |
| **Race engineer** | The one-line translation to relay to the driver | Partially | ✅ Secondary rendering |
| **Broadcast / fans** | Narrative | No | 🟡 Third rendering — but Pitwall already owns this and it is a crowded media-rights business |

### The decision

> **Primary user: the race strategist.**
> **Architecture: one verified state, four renderings.**

### Why the strategist, in one sentence

**Our entire technical edge is calibrated uncertainty — and calibrated uncertainty
only has value to someone who makes decisions under uncertainty.** That is the
strategist, and nobody else on the list. Give a driver an interval and you have
made their job harder; give a strategist a point estimate and you have made theirs
impossible.

### The four renderings of the same state

| Surface | Audience | Example output | Latency |
|---|---|---|---|
| **Strategist console** | Pit wall | *"Box lap 34. Expected gain 2.1 s over lap 36, 95% interval [−0.4, +4.6]. Driven by MEDIUM degradation 0.112 s/lap [0.067, 0.157]. Rule 14 (puncture risk) does not fire. If degradation is at the top of the interval, box lap 32 instead."* | < 2 s |
| **Engineer line** | Race engineer → driver | *"Two more laps then box. Tyres are going at the top of our expected range."* | < 2 s |
| **Crew command** | Tyre crew | *"MEDIUM. Box lap 34. Stand by from lap 32."* | 30 s lead |
| **Post-race audit** | Team debrief, sporting director | *"Lap 34 recommended, lap 37 taken, cost 2.8 s and one position."* | offline |

**One state, four voices. Never four models.** Divergence between surfaces is the
classic failure of this kind of system.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ L0  DATA                                                            │
│     Phase 1: 203 historical sessions, 91,867 laps   ← WE ARE HERE    │
│     Phase 2: live timing stream                                      │
├─────────────────────────────────────────────────────────────────────┤
│ L1  ESTIMATOR    state-space + Kalman filter / RTS smoother          │
│                  latent tyre state, per corner (§9 of lit review)    │
├─────────────────────────────────────────────────────────────────────┤
│ L2  CALIBRATION  split conformal (offline) + adaptive conformal (live)│
│                  → 94.7% offline, 95.2% over 69,206 laps live        │
├──────────────────────────┬──────────────────────────────────────────┤
│ L3a ORACLE PATH          │ L3b DECISION PATH                        │
│     calibrated probs     │     expected-cost pit optimiser           │
│     scored on coverage   │     + 18-rule safety layer (veto)         │
│                          │     scored on regret                      │
├──────────────────────────┴──────────────────────────────────────────┤
│ L4  VERIFIER     every generated sentence → typed claims → checked    │
│                  against state. Unverified claim → template fallback. │
├─────────────────────────────────────────────────────────────────────┤
│ L5  AGENT        role-adapted language for 4 personas                 │
├─────────────────────────────────────────────────────────────────────┤
│ L6  SURFACES     strategist console · engineer line · crew command ·  │
│                  post-race audit                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### L4 — the verifier is not optional, and here is the evidence

Pitwall's most important engineering finding, which we take as a design rule:

> Fine-tuning the generator on richer, more natural targets improves fluency — **yet
> the identical model fabricates drivers, gaps and tyre compounds when the
> grounding state is sparse.** Replicating across four base models showed this is a
> property of the base model's instruction adherence, **not of scale**.

Read that again in our context: **the agent hallucinates exactly when data is
thin — which is early in a stint, which is exactly when a strategist needs it.**

So the rule:

1. Every sentence the agent produces is decomposed into **typed factual claims**
   (rate, interval, lap number, compound, gap, rule verdict).
2. Each claim is **checked against the state object** that prompted it.
3. A sentence with any unverified claim is **discarded** and replaced by a
   **provably faithful template**.
4. We log the verification rate. Pitwall retained 81.9% of model-written
   candidates; ours should report its own number, whatever it is.

**A bigger model does not fix this. Only the verifier does.**

### What the agent must never do

- Invent a number that is not in the state object.
- Give a recommendation without its interval.
- Speak to the driver.
- Report a rate without saying which compound and how many laps it rests on.
- Say "confident" when coverage has not been measured for that quantity.

---

## 4. Phase 1 — existing data (build this now)

Everything here runs on the 203 sessions we already hold. No live feed needed.

| # | Build | Depends on |
|---|---|---|
| 1 | Four-corner estimator — energy-share-constrained latent states | telemetry download |
| 2 | Conformal RUL, "laps of useful life", with interval | have it |
| 3 | Puncture hazard model on real failure labels | label pull |
| 4 | Grip index from telemetry | telemetry download |
| 5 | Tread remaining as **% of usable life** (never mm) | 1, 2 |
| 6 | Expected-cost pit optimiser (decision path) | 2 |
| 7 | 18-rule safety layer, running alongside, with veto | 6 |
| 8 | State object — the single source of truth the verifier checks against | 1–7 |
| 9 | Verifier — typed claim extraction and checking | 8 |
| 10 | Agent — four renderings | 9 |
| 11 | Dashboard — five metric cards, **every one with an interval** | 8 |
| 12 | Post-race audit / replay | 6 |

**The gate between Phase 1 and Phase 2:** the agent must achieve a measured
verification rate on historical races, replayed lap by lap, before any live feed is
connected. If it hallucinates on a replay it will hallucinate live, faster.

## 5. Phase 2 — live

| # | Build |
|---|---|
| 1 | Live timing ingestion replacing the replay source |
| 2 | Adaptive conformal running online (**already built** — 95.2% over 69,206 laps) |
| 3 | Late / out-of-order / missing frame handling |
| 4 | Running coverage display — the system audits itself on screen, live |
| 5 | Latency budget: state → recommendation → verified sentence in < 2 s |

**The differentiator to demo:** the interval **corrects itself during the race**
and the screen **shows its own accuracy while running**. No other system in this
review displays its live coverage.

---

## 6. Why our approach beats the current TrackShift engine

### 6.1 First — a warning about how to say this

You will be presenting to the mentors who supervised that system. **"Your approach
is bad" is both a social risk and an evidential risk**, because we have not seen
their code or their data.

The strong version is not the aggressive version. It is:

> *"Here is what that architecture structurally cannot do, and here is our
> measurement of the cost."*

Every row below is a **measured** claim from our own experiments, not an assertion
about their work.

### 6.2 The four structural ceilings

| # | Their design | The ceiling | Our measurement |
|---|---|---|---|
| 1 | Five-parameter formula, **Bahrain V3 calibration** — one circuit | Single-circuit calibration has no evidence it travels | exp09: leave-one-circuit-out over 26 circuits — geometry gives **−3.3%, p = 0.22** (no effect), thermal features **actively hurt, p = 0.0012**. Circuit transfer genuinely fails. |
| 2 | **Threshold-based** alerts on point estimates | A threshold with no interval is a coin flip near the boundary | exp12: our own model claimed 95% and delivered **75%**. We only know that because we checked. |
| 3 | **18 rules + 17-step ladder** — an expert system | Cannot express "I don't know", cannot trade off the cost of being wrong, cannot learn from outcomes. Supersede logic is the tell: rules collide, so you add rules to arbitrate rules. | Our replacement is expected-cost minimisation over a calibrated distribution — **and we keep all 18 rules as a veto layer** |
| 4 | No negative results published | Untested features silently degrade a model | exp09 proved this happens: adding thermal features made predictions **worse**, significantly |

### 6.3 What we add that is not in their list *or* in the literature

| # | Contribution | Evidence |
|---|---|---|
| 1 | **The identifiability bound** | 520/520 runs exactly collinear, r = 1.000, 11/11 singular information matrices, Cramér–Rao 0.0161 vs our 0.0221 = **1.38× optimal**, and **only 6.0%** of the fuel/tyre separation is data-driven. **No paper in our review states this number.** Cappello & Hoegh's model has the identical structure and resolves it with a prior they never quantify. |
| 2 | **Validated, not assumed, uncertainty** | 94.7% offline, **95.2% over 69,206 laps** online. Bayesian credible intervals are right *if the model is right*. We checked. |
| 3 | **Falsification without ground truth** | 74% of 77 races give a physically impossible answer. Needs no label. Generalises to any domain where truth is unpublished. |
| 4 | **Scale with leave-one-out discipline** | 84 event-seasons, 26 circuits, LOCO + LOEO — against a field whose closest comparable work fits **one race** |
| 5 | **Six published negative results** | Three of them killed our own hypothesis with our own test |
| 6 | **Cross-domain proof** | NASA turbofans, unmodified code, RMSE 22.7 cycles. Directly delivers their "full vehicle health monitoring" future item. |
| 7 | **Independent replication of a published methodological finding** | calibration-optimal ≠ decision-optimal — §1.3 |

### 6.4 Their five metrics, our version

| Their metric | Their version | Ours |
|---|---|---|
| Grip Level | "real-time traction coefficient" | Normalised **grip index 0–1**, with an ablation test. **We refuse to print a μ value** — it needs load-cell and slip data that is team-private. |
| Tread Remaining | "wear rate & remaining depth" | **% of usable life + interval**, anchored to the measured cliff (exp17: 71.9% through a stint). **Never millimetres** — no public measurement exists to anchor against. |
| Degradation Rate | 5-param Bahrain formula | State-space, 4 seasons, calibrated interval, **with the identifiability bound stated** |
| Tyre Energy | "laps of useful life" | Conformal RUL — **and the same code already does this on jet engines** |
| Puncture Risk | "threshold-based and probabilistic" | **A structural exposure index, explicitly not a probability.** We collected the labels: 17 tyre failures in 12 seasons (2011-2022), and no cause detail at all after 2022. No probability is calibratable from public data -- by anyone. |

**The honest line on Tread Remaining and Grip is a strength, not a retreat:**

> *"They report remaining depth in millimetres and a traction coefficient. Neither
> is measurable from public data. We report percentage of life remaining and a
> normalised grip index, because we would rather be right than precise."*

### 6.5 Their four "future opportunities" — status

| Their future item | Our status |
|---|---|
| Live telemetry integration | 🟢 **Substantially built** — live filter + self-correcting interval, 95.2% over 69,206 laps |
| Driver-in-the-loop | 🟡 Physics exists (`frictional_power_proxy`, `lap_energy`); closed loop is exp34 |
| Multi-compound strategy simulation | 🟡 Simulator exists; needs uncertainty propagated through it (exp29) |
| Full vehicle health monitoring | 🟢 **Already proven** — NASA turbofans + **151 vehicle failure events** (Gearbox 54, Brakes 50, Suspension 37) collected. Nine times the tyre-failure count, so this is where the hazard model belongs. |

---

## 7. The business case

| Level | Customer | Value |
|---|---|---|
| **Team** | 10 F1 teams | One wrong pit call costs positions; championship positions cost millions. exp28 will quantify this in seconds and positions per race. |
| **Series / broadcast** | F1, F2, F3, WEC, IMSA, Formula E | Insight graphics. **Note: Pitwall already competes here** — do not lead with it. |
| **Industrial** | Fleet maintenance, wind, rail, mining | The engine is domain-agnostic and **already proven on turbofans**. The strategist persona becomes the maintenance planner. **This is the largest market and the one nobody else in this review addresses.** |

**Cost story:** 12 s to fit a session on a laptop. 2.6 MB for four seasons. No GPU,
no training run, no cloud bill. Runs fully offline. The neural network in our own
comparison took 36 s and came last.

---

## 8. Honest risks

| Risk | Mitigation |
|---|---|
| **Pitwall already built the agent** | Do not pitch the agent as novel. Pitch the **substrate**: their tyre layer is, by their own word, *confounded*. Ours is not. |
| **Cappello & Hoegh published the model class** | Cite it, implement it as a baseline, and lead with identifiability — the thing they do not have. |
| **Agent hallucination** | The verifier (L4). Non-negotiable, gated before live. |
| **Four-tyre estimator is underdetermined** | One lap time cannot identify four states. Constrain with observable energy shares — lit review §9. Say the limitation out loud. |
| **Overclaiming against the mentors' own system** | Use measured ceilings, never adjectives. §6.1. |

---

## 9. Immediate next actions

1. **Start the telemetry download** — blocks the four-corner estimator and the grip index. Overnight.
2. **Pull failure labels** — half a day, unblocks puncture risk.
3. **Add ARIMA(2,1,2) + Heilmeier + Cappello–Hoegh baselines** — directly answers the mentor criticism that our baseline is too weak.
4. **Define the state object** — everything downstream (verifier, agent, dashboard) depends on this one schema, so it is the real critical path.
5. **Build the verifier before the agent.** In that order. Always.
