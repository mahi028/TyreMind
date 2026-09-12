# Pre-registration — exp27, de-circularising the practice-to-race test

**Written before the experiment was run. Committed before the results existed.**

This file exists for the same reason `PREREGISTRATION_exp19.md` does: to fix the
metrics, the directions and the losing conditions before any number is visible,
so that a disappointing result cannot be rescued by choosing a different score.

---

## 1. The problem with exp03, stated plainly

`exp03_practice_to_race` asks whether a Friday degradation curve predicts Sunday.
It answers by fitting **our** state-space model to practice, fitting **our** model
to the race, and scoring the first against the second. The naive comparator in
that experiment is then scored against **our race fit as well**
(`validation.py`, `CompoundComparison.naive_error`).

That is circular in a way that flatters us. The reference against which every
method is graded is one particular method's own output. A competitor is penalised
both for its practice estimate being wrong *and* for its race estimate differing
from ours — and we are penalised only once. It is the weakest evidence in the
project.

## 2. The fix

Each model is scored against **its own** race-derived estimate.

    error(model, event, compound) = rate_practice(model) - rate_race(model)

Every model is now judged on the same question — *does this method, applied to
Friday, agree with the same method applied to Sunday?* — and no method's output
is the yardstick for anyone else.

The old circular error is computed **as well**, on the same comparisons, so that
the size and direction of the distortion is measurable rather than asserted.

## 3. The catch this creates, and the metric that closes it

Self-consistency is trivially gameable. A model that returns the same constant
degradation rate for every event scores a perfect zero. Reporting self-consistency
MAE alone would replace one flattering metric with another — one that flatters the
*least informative* model instead of ours.

So the **primary metric is a skill score against each model's own climatology**:

    skill(model) = 1 - MAE_self(model) / MAE_constant(model)

where `MAE_constant` is the error that model would have made by ignoring practice
entirely and predicting the mean of its own race-derived rates across all scored
comparisons. Higher is better; 1.0 is perfect; **0 or below means the practice
session told that model nothing it did not already know from its own average**.

A constant model scores exactly 0, by construction. It cannot win.

## 4. The models — all nine, fixed

`tyremind.models.literature.extended_ladder()`, unchanged and in its shipped
order: Naive, Fuel-corrected, Pooled, LightGBM, MLP, TyreMind, ARIMA(2,1,2),
Heilmeier, Cappello & Hoegh.

Three of them — ARIMA, LightGBM, MLP — have no degradation parameter and return
an empty `compound_rates()`. They are **reported as "no degradation parameter"**,
which is the finding exp19 already recorded, and are not scored. They are still
fitted, so that a failure to fit would be visible rather than assumed.

No model may be added or removed after results are seen. A model that raises is
recorded as **failed** with its exception text, not dropped.

## 5. The data

Every event in `data/season/corpus.json` that has both an FP2 and a race session:
49 events across 2022–2025. Selection is by the manifest, not by hand.

**Dry only.** `data/reference/session_conditions.json` marks six sessions as not
dry. An event is excluded if *either* of its two sessions is wet, and the
exclusion is listed in the result file with its wet-lap fraction. This is the
same rule exp03 applies, applied to both sides.

**Per-compound threshold.** A compound is compared only when it has at least 8
laps in both sessions, matching `validate_practice_to_race`'s
`min_laps_per_compound`. Compounds below the threshold are recorded as skipped
with the reason, never scored as an error.

Loading is via `tyremind.data.corpus.read_lap_table` / `load_frames`, never
`pd.read_parquet`, so the `lap_in_run` fuel repair is applied.

## 6. The metrics, with directions

| Metric | Direction | Note |
|---|---|---|
| **Skill vs own climatology** | **higher better** | Primary. Cannot be won by being constant. |
| Self-consistency MAE, s/lap | lower better | Secondary, and gameable — read next to the spread. |
| Bias (signed mean error) | closer to 0 better | Persistent sign = practice systematically mis-states race degradation. |
| Coverage of the 95% interval | **closer to 0.95 better** | Not higher-is-better. Combined sd of both sides, as in `CompoundComparison.covered`. |
| SD of the model's own race rates | reported | The denominator of the skill score. Near zero exposes a degenerate model. |
| Spearman correlation, practice vs own race rate | reported | Whether the practice estimate tracks event-to-event variation at all. |
| Circular MAE (vs TyreMind's race fit) | reported | The old exp03 metric, kept so the distortion is measurable. |

## 7. What counts as a win, and what counts as a loss

Declared now.

- **We claim a win** only if TyreMind has the highest skill score *and* its
  coverage is within 10 points of 0.95.
- **If TyreMind's skill score is ≤ 0**, the finding is that our practice fit
  carries no event-specific information beyond our own average, and we report
  exactly that sentence.
- **A tie is a tie.** Models within one standard error of the best on skill are
  reported as comparable, not worse.
- **If de-circularising changes our rank** — if we look better under the old
  circular metric than under the new one — that difference is the headline of
  this experiment, whichever way it goes. The point of exp27 is to measure how
  much exp03 was flattering us, and a large flattering effect is a *successful*
  experiment with an unflattering result.
- **If every model scores ≤ 0 on skill**, the finding is that no method on the
  ladder transfers Friday to Sunday at the compound-rate level, and exp03's claim
  needs withdrawing. That outcome is written down here so it stays reportable.

## 8. Known limitations — stated up front

1. **Self-consistency is not accuracy.** No public data contains measured tyre
   wear, so "the race rate" is a model output for every model, including ours.
   This experiment measures agreement between two applications of one method. It
   cannot say which method is *right*; only the synthetic legs can.
2. **Each model's race fit carries its own bias.** A method biased in the same
   direction on Friday and Sunday scores well here. That is a real property —
   a consistently biased estimate is still useful for *deltas* — but it is not a
   correctness claim, and the skill score does not repair it.
3. **Practice and race differ in fuel, traffic, track state and driving style at
   once.** This remains an out-of-distribution transfer, not a holdout split.
4. **49 events is not a large sample** once split by compound, and the standard
   errors are reported rather than elided.

## 9. Fixed before running

- Metrics, directions and win conditions: this document.
- Model set: `extended_ladder()`, all nine.
- Event set: every FP2+R pair in the manifest, dry both sides.
- Compound threshold: 8 laps each side.
- No random seeds are involved; both fits are deterministic given the data.

Any deviation must be recorded in the result file as a deviation, with a reason.
