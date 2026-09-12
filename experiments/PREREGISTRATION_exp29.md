# Pre-registration — exp29, generator robustness

**Written before the experiment was run. Committed before the results existed.**

---

## 1. The objection this experiment exists to answer

Our strongest published result — exp19 Leg C, rate MAE 0.0037 against 0.0062 for
the next best — is measured on synthetic sessions **we wrote**. The strongest
thing a judge can say about it is:

> Of course your estimator recovers the truth. You built the generator to match
> your estimator's assumptions.

`PREREGISTRATION_exp19.md` §6.1 already concedes the point and calls the
mitigation partial. This experiment measures how big the problem is by sweeping
the generator **away** from our assumptions and checking whether the ranking
survives.

If it does not survive, that is the finding, and it is reported as the finding.

## 2. The regimes — fixed before running

Each regime changes `SessionConfig` along one axis from the shipped defaults.
Every other parameter, including the seed schedule, is held constant.

| Regime | Change | What it attacks |
|---|---|---|
| `as_shipped` | — | The exp19 baseline, re-measured here |
| `heavy_tails` | `observation_noise_df = 2.5` | Our estimator assumes Gaussian noise. t(2.5) has no finite variance. |
| `triple_noise` | `observation_noise_sd = 0.48` | Signal-to-noise: degradation is ~0.07 s/lap under 0.48 s of scatter |
| `gaussian_noise` | `observation_noise_df = 1e6` | The *opposite* direction: noise exactly as our estimator assumes. The regime most favourable to us, included so the best case is on the record too. |
| `severe_cliff` | `cliff_severity = 0.020` | Strong non-linearity in age. Heilmeier cannot represent it at all; neither can a linear rate. |
| `no_cliff` | `cliff_severity = 0.0` | Perfectly linear degradation — the shape the *competitors* assume |
| `all_scrubbed` | `scrubbed_set_probability = 1.0` | Tyre age fully decoupled from laps-in-run |
| `no_scrubbed` | `scrubbed_set_probability = 0.0` | Tyre age and laps-in-run collinear — the exp18 worst case |
| `heavy_traffic` | `traffic_probability = 0.55`, `traffic_coefficient = 2.2` | A confounder Cappello & Hoegh absorb into i.i.d. noise, and we model |
| `flat_track` | `track_evolution_total = 0.0` | A confounder we model and the others do not. **Our machinery becomes dead weight.** |
| `strong_track` | `track_evolution_total = 2.5` | The same confounder, amplified |

Eleven regimes. No regime may be added or removed after results are seen. A
regime that fails to generate is reported as failed.

## 3. The models

`extended_ladder()` is instantiated, and the six rungs that expose a degradation
parameter are scored: Naive, Fuel-corrected, Pooled, TyreMind, Heilmeier,
Cappello & Hoegh. ARIMA, LightGBM and the MLP return an empty `compound_rates()`
by construction — the finding exp19 recorded — and are reported as having no
degradation parameter rather than being quietly dropped. They are not fitted here,
because fitting a model that cannot produce the quantity being scored only costs
time; that omission is declared now rather than discovered later.

## 4. The metric, and the power problem

**Primary: rate MAE** against the true baseline degradation rate per compound
(`GroundTruth.compound_rates`), lower better — the same target exp19 Leg C used,
so the numbers are readable against it. Rate bias and 95% interval coverage
(closer to 0.95 better, not higher-better) are reported alongside. As a
sensitivity, MAE against the **lap-weighted mean true instantaneous rate** is also
reported, since the cliff makes the two targets differ.

**On power.** A pilot at 4 seeds put Pooled ahead of us in the `as_shipped`
regime at 0.0049 against 0.0053, while exp19 at 8 seeds has us ahead at 0.0037
against 0.0062. A ranking that flips with seed count is a ranking inside the
noise, and reporting it either way would be reporting noise. So:

- **32 seeds per regime**, `20260901 + i`, identical across every regime and
  every model, so every comparison is **paired**: the same generated session is
  handed to all six models.
- The headline statistic is the **paired difference in absolute rate error**
  between two models, with its standard error over the 32 × 3 = 96 (seed,
  compound) pairs. Paired, because the between-seed variance is far larger than
  the between-model difference, and an unpaired comparison would hide a real
  effect.
- A regime is called for a model only if it beats the runner-up by **more than
  one standard error of the paired difference**. Otherwise the regime is recorded
  as a **tie**, not a win.

## 5. What counts as a win, and what counts as a loss

Declared now.

- **The headline claim — "TyreMind recovers the true rate best" — survives** only
  if TyreMind has the lowest rate MAE in a **majority of the eleven regimes**,
  counting ties as not-wins for anybody.
- **Any regime we lose is reported by name**, with the margin and its standard
  error, in the result file and in the summary table. There is no aggregate that
  hides a lost regime.
- **If TyreMind loses more regimes than it wins**, the finding is that our
  synthetic advantage is an artefact of the generator's default settings, and
  exp19 Leg C must be re-presented with that caveat attached. This sentence is
  written here so that outcome stays reportable.
- **`flat_track` is expected to be a loss** and is pre-declared as one. A model
  that explicitly represents track evolution must pay for that machinery when
  there is no track evolution to represent. If we win it anyway, that is a
  surprise and gets said; if we lose it, it is a cost of our architecture and gets
  its own paragraph rather than a footnote.
- **`gaussian_noise` is expected to be a win** and is pre-declared as one, so the
  regime most favourable to us cannot be quoted as though it were neutral.

## 6. Known limitations — stated up front

1. **This is still our generator.** Sweeping one parameter at a time does not
   escape the structural form of the simulator: additive terms, a single
   degradation rate per compound, exponential track evolution. A genuinely
   independent truth engine is exp20 and does not exist yet. Every result here
   carries that caveat, exactly as exp19 Leg C does.
2. **One axis at a time.** Interactions between regimes are not measured. Eleven
   single-axis regimes at 32 seeds are affordable; a full factorial is not.
3. **The estimators are unchanged.** No model gets its priors retuned per regime,
   including ours. That is fair, and it also means a regime can be lost for the
   wrong reason — a mis-specified prior rather than a mis-specified model — and we
   cannot tell the two apart from this experiment alone.
4. **Coverage is measured against the point truth**, and three of the six rungs
   report a standard error rather than a posterior, so their intervals are not
   comparable in kind. Reported, not ranked.

## 7. Fixed before running

- Regimes, metrics, directions and win conditions: this document.
- Seeds: `20260901 + i` for `i` in `0..31`, identical across regimes and models.
- Truth target: `GroundTruth.compound_rates`; mean-instantaneous as sensitivity.
- Tie rule: within one standard error of the paired difference.

Any deviation must be recorded in the result file as a deviation, with a reason.
