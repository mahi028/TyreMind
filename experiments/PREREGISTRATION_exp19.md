# Pre-registration — exp19, the nine-rung field comparison

**Written before the experiment was run. Committed before the results existed.**

The point of this file is to remove a specific way of cheating: running a
comparison, seeing that we lose on some metric, and then quietly choosing a
different metric. Everything below — the models, the data, the metrics, and what
counts as winning and losing — is fixed here in advance. The result file must be
readable against this document with no room to move.

---

## 1. The question

Does TyreMind beat the published field at (a) forecasting lap times on real data,
and (b) recovering a true degradation rate on data where the truth is known?

These are **two different questions** and we expect **different winners**. Pitwall
(arXiv:2607.06495) reports the same phenomenon and calls it *calibration-optimal is
not decision-optimal*. Our exp05 found it independently. We are not permitted to
report only the one we win.

## 2. The models — all nine, fixed

| # | Model | Origin |
|---|---|---|
| 1 | Naive (lap time vs tyre age) | the strawman; shows the problem is real |
| 2 | Fuel-corrected regression | paddock standard |
| 3 | Pooled regression | strongest non-state-space baseline |
| 4 | LightGBM | standard tabular ML |
| 5 | Neural network (MLP) | "just use deep learning" |
| 6 | **TyreMind state-space** | ours |
| 7 | ARIMA(2,1,2) | Cappello & Hoegh's own baseline |
| 8 | Heilmeier additive (linear tyre) | field standard, Eq. (1) and (6) |
| 9 | Cappello & Hoegh state-space | closest published prior art |

No model may be added or removed after results are seen. If a rung fails to fit it
is reported as **failed**, not dropped.

## 3. The data

**Leg A — real.** Sessions drawn from `data/season/`, ordered most-recent-season
first. `--limit` fixes the count; the selection rule is deterministic and is in
`tyremind.data.corpus.sessions()`. No cherry-picking of circuits.

**Leg C — synthetic.** `tyremind.data.synthetic` with fixed seeds. Ground truth is
never shown to any model.

**Leg B — the papers' own data** is scoped but *not* part of exp19. Cappello &
Hoegh's case study (Hamilton, 2025 Austrian GP) is reproducible and is exp21.
Sulsters/Todd used Mercedes-internal data and is **not reproducible by anyone
outside the team** — we will say so rather than approximate it.

## 4. The metrics — fixed, and each with a stated direction

### Leg A: lap-time forecasting on real data

Chronological (rolling-origin) folds. No future information reaches a fold's fit.

| Metric | Direction | Why this one |
|---|---|---|
| **CRPS** | lower better | Scores the whole predictive distribution, not just the point. A model cannot win by being confidently wrong. |
| MAE | lower better | Plain point accuracy, for readability |
| **Coverage of the 95% interval** | **closer to 0.95 better** | *Not* higher-is-better. An interval covering 100% is as miscalibrated as one covering 61%. |
| Interval width | reported, not ranked | Only meaningful next to coverage |
| Fit seconds | reported | The feasibility/economy claim |

**Lap time is fully observed.** This leg needs no ground truth and no synthetic
data. It is a genuine supervised comparison.

### Leg C: degradation recovery on synthetic data

| Metric | Direction |
|---|---|
| **Rate MAE** vs the true rate | lower better |
| Rate bias | closer to 0 better |
| **Coverage of the 95% interval on the rate** | closer to 0.95 better |

Models with no degradation parameter (ARIMA) report **null**, which is the finding,
not a gap.

## 5. What counts as a win, and what counts as a loss

Declared now, so it cannot be renegotiated later.

- **We claim a win on Leg C** only if TyreMind has the lowest rate MAE *and*
  coverage within 10 points of 0.95.
- **We claim a win on Leg A** only if TyreMind has the lowest CRPS. **We expect not
  to.** exp05 already showed the pooled regression ahead on CRPS (0.395 vs 0.645).
  If that holds, we report it as the headline of Leg A and explain the two-task
  distinction — we do not bury it.
- **A tie is a tie.** If TyreMind and Cappello & Hoegh are within one standard
  error on a metric, we say "comparable", not "better".
- **If we lose both legs**, the finding is that our model is not better and we say
  so, then investigate why. This is written down precisely so that outcome stays
  reportable.

## 6. Known biases in this design — stated up front

1. **Simulator bias on Leg C.** Our synthetic generator shares structural
   assumptions with our estimator, which advantages us. Partially mitigated:
   Student-t(5) observation noise where our estimator assumes Gaussian, and 25%
   scrubbed sets. **Not fully mitigated.** The real fix is an independently
   authored truth engine (exp20), and until that exists Leg C results carry this
   caveat in every presentation of them.
2. **MAP not MCMC** for the Cappello & Hoegh rung. Same model, different inference.
   Where the gap between us and them is within one standard error, the inference
   difference is a plausible explanation and we will say so.
3. **Their model is per-driver by design.** On thin sessions it may return no rate
   for a compound. That is a property of the published method, not a handicap we
   imposed — but it must be reported as "no estimate", never scored as an error.
4. **Sessions are dry-only.** Wet running is excluded corpus-wide, so every model
   is evaluated on the same restricted regime.

## 7. Fixed before running

- Metrics, directions, and win conditions: **this document**
- Random seeds: fixed in the script
- Fold count: 4 chronological folds
- Session selection: deterministic, by `--limit`

Any deviation from this document must be recorded in the results file as a
deviation, with the reason.
