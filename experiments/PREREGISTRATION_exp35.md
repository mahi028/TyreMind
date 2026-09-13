# Pre-registration — exp35: the 4-corner physics-informed continuous-discrete EKF

Written **before** the estimator exists. Committed before it is run.

The point of writing this first is that the proposed model is large, expensive and
attractive, and those three properties together are how a team talks itself into
shipping something that does not work. The thresholds below are chosen now, while
no result is known.

---

## 1. What is being proposed

Replace the 2-state scalar latent tyre model with a 10-state per-corner
physics-informed continuous-discrete Extended Kalman Filter:

```
x(t) = [w_FL w_FR w_RL w_RR  T_FL T_FR T_RL T_RR  m_fuel  Δ_track]^T
```

- `w_i ∈ [0,1]` normalised mechanical wear at corner `i`
- `T_i` bulk tread temperature, °C
- `m_fuel` fuel mass, kg
- `Δ_track` track evolution progress

Between lap crossings the state follows an SDE driven by kinematics measured
from telemetry; at each lap crossing a scalar lap time updates it.

## 2. Why this could work

Three honest reasons, all of which predate this document:

1. **The per-corner inputs already exist and are measured, not modelled.** The
   telemetry corpus carries `energy_mj_{fl,fr,rl,rr}` and
   `mean_load_n_{fl,fr,rl,rr}` for 202 sessions and 147,550 laps. The corners are
   driven by data, not by a symmetry assumption.
2. **Corner asymmetry is real and we have already measured it.** exp06 recovered
   circuit rotation direction from per-corner energy alone on 7 of 8 circuits.
   Whatever else is true, the left/right load split carries signal.
3. **Circuit transfer has a principled reason to improve.** A compound's wear
   coefficient and thermal window are properties of rubber, not of a circuit. If
   the circuit enters only through `F_z,i(t)`, which is computed rather than
   fitted, then a compound parameter learned at Silverstone is meaningful at
   Monza in a way a fitted per-circuit slope is not.

## 3. Why it may well fail — stated now, not after

1. **exp18 measured the identifiability ceiling and it is low.** Within a run,
   tyre age and fuel are collinear at ρ = 1.000, the information matrix is
   singular in 11 of 11 sessions, and only **6.0%** of the recovered rate is
   data-driven. Adding eight more latent states to the same scalar observation
   per lap cannot increase the information available. **I predict the data-driven
   share falls below 6.0%, not above it.**
2. **The four corner wears are not separately identifiable from a scalar lap
   time** unless the sensitivity coefficients `c_i` differ. If `c_FL = c_FR` the
   model can trade FL wear against FR wear with no change in likelihood. The only
   thing breaking that degeneracy is measured left/right load asymmetry.
3. **The energy-wear hypothesis has already been refuted twice on this corpus.**
   exp25 found every significant stint-level correlation carried the wrong sign,
   and exp34 found every measured feature made leave-one-circuit-out prediction
   *worse*, at p < 0.001, with a ridge sweep converging on the baseline from
   above. `dw/dt ∝ (μ F_z v_slip)^α` is the relationship those two experiments
   failed to find. A structured dynamical model is a genuinely different
   hypothesis from a linear correlation, which is why this is worth testing — but
   the prior is not favourable and pretending otherwise would be dishonest.

## 4. Pre-registered predictions

Recorded before any run. Numbered so they can be scored individually.

| # | Prediction | Confidence |
|---|---|---|
| P1 | Data-driven share of the degradation posterior **falls below** exp18's 6.0% | high |
| P2 | The 4-corner model does **not** beat the 2-state model on exp19's rate-recovery MAE | moderate |
| P3 | Corner-level wear states are **not** separately identifiable: the smallest eigenvalue of the 4-corner Fisher block is within numerical noise of zero on at least half of sessions | high |
| P4 | Left/right *aggregate* asymmetry **is** identifiable where circuit rotation is strong, because exp06 already showed that signal exists | moderate |
| P5 | Circuit transfer does **not** improve over the compound-label baseline that beat every feature in exp34 | moderate |

**If P1–P3 and P5 all hold, this architecture is refuted as a replacement and
the honest outcome is to keep the 2-state model.** P4 holding while the rest fail
would mean the useful part is a left/right split, not a four-corner one.

## 5. What counts as success — thresholds fixed now

The proposal only replaces the shipped model if **all four** of these hold:

- **S1 — Recovery.** Rate MAE on exp19's synthetic benchmark is lower than the
  2-state model's **0.0037 s/lap** by a margin of at least **2 standard errors**
  over the same 8 seeds. A margin inside the noise is a tie and a tie is a loss,
  because the incumbent is simpler.
- **S2 — Calibration survives.** 95% interval coverage stays within **[0.90,
  0.99]** after conformal calibration, measured the same way exp12 measured it.
- **S3 — Transfer.** Leave-one-circuit-out error beats the compound-label
  baseline by at least **1%** with **p < 0.05** on a paired Wilcoxon. This is the
  exact test and the exact threshold exp34 used to refute the energy features, so
  the two results are directly comparable.
- **S4 — Cost.** Fit time stays under **60 s per session** on the demo laptop. The
  current model takes about 12 s. A model that needs a cluster is not the same
  product, whatever its error.

**Falsification is the expected outcome and is a publishable result either way.**
If the model fails S1–S4 it does not ship, and exp35 joins the refuted-hypothesis
table as the thirteenth entry.

## 6. What will be reported regardless of outcome

- The measured data-driven share, against exp18's 6.0%
- The Fisher eigenvalue spectrum of the 4-corner block, per session
- Rate MAE against every rung of the existing ladder on identical seeds
- Leave-one-circuit-out error against the exp34 baseline, same design
- Wall-clock fit time
- Every parameter that had to be fixed by prior rather than fitted, and why

## 7. Deviations

Any departure from this document gets written into the result file's
`deviations` field with a reason, as in exp32 and exp34.
