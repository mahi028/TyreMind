/**
 * The pit-lap optimiser, ported line for line from `tyremind/models/pit_decision.py`.
 *
 * Why port it rather than call an endpoint. There are two pit recommenders in
 * this repository and they do not agree:
 *
 *   - `recommend_pit_lap` (this one) is the analytic sweep that **exp22 scored**
 *     across 274 real stops in 14 races. Every aggregate this view quotes --
 *     24% within two laps, 12% within one, the belief the model placed on the
 *     lap that actually happened -- was produced by this function and no other.
 *   - `/api/session/{id}/pit-window` is a Monte-Carlo race simulator. It is a
 *     fine estimator and it has its own screen, but it has never been scored on
 *     the 246-stop benchmark. On this very case it answers lap 20 where the
 *     validated optimiser answers lap 11.
 *
 * Showing one model's recommendation beside another model's track record is the
 * exact dishonesty this view exists to avoid, so the view runs the optimiser
 * that owns the track record. Its only inputs are the compound rates from
 * `/api/session/{id}`, which are the same fitted rates exp22 fed it, so the
 * number on screen is reproduced live rather than replayed from a file.
 *
 * The port is checked against the Python for the demonstrated case: SOFT at age
 * 8, deciding on lap 5 of a 57-lap race, returns lap 11 with 0.09619 on it, to
 * every digit Python prints.
 */

/** Seconds lost in a stop, including the pit-lane delta. `DEFAULT_PIT_LOSS_S`. */
export const DEFAULT_PIT_LOSS_S = 21.0

/**
 * Python's `round` is half-to-even; JavaScript's is half-away-from-zero.
 *
 * It matters. The stint split `int(round(laps_after / n_stints))` lands on a
 * .5 often enough that the two conventions produce different cost curves, and a
 * different cost curve is a different recommended lap. This is the only place
 * the port could have silently diverged from the validated model.
 */
function pyRound(value: number): number {
  const floor = Math.floor(value)
  const diff = value - floor
  if (diff > 0.5) return floor + 1
  if (diff < 0.5) return floor
  return floor % 2 === 0 ? floor : floor + 1
}

/** Seconds lost to degradation over `nLaps`, starting at tyre age `startAge`. */
export function degradationLoss(rate: number, startAge: number, nLaps: number): number {
  if (nLaps <= 0) return 0
  const n = nLaps
  return rate * (n * startAge + (n * (n + 1)) / 2)
}

export interface PitRecommendation {
  /** Recommended session lap to box on. */
  lap: number
  /** Probability mass on that lap. Usually small, and that is the honest part. */
  confidence: number
  /** Lap to probability, over every feasible lap. */
  distribution: Map<number, number>
  /** Lap to expected total remaining race time, seconds. */
  costCurve: Map<number, number>
  /** Laps whose cost is within one second of the optimum. */
  window: [number, number] | null
  /** Probability mass inside that window. The number a strategist acts on. */
  windowConfidence: number | null
  /** Empty when the recommendation is well posed; otherwise why it is not. */
  reason: string
}

export interface PitInputs {
  currentRate: number
  currentRateSd: number
  freshRate: number
  currentAge: number
  decisionLap: number
  finalLap: number
  pitLossS?: number
  minStintLaps?: number
  maxRemainingStops?: number
  flatCurveToleranceS?: number
}

/**
 * Sweep every feasible pit lap and return the cheapest, with a distribution.
 *
 * The distribution is a softmax over negative cost at a temperature set by the
 * model's own uncertainty about the rate: a model that does not know the rate
 * cannot claim to know the lap. That is why the confidence here is single
 * figures rather than the 90% a strategy tool is usually tempted to print.
 */
export function recommendPitLap(input: PitInputs): PitRecommendation {
  const pitLossS = input.pitLossS ?? DEFAULT_PIT_LOSS_S
  const minStintLaps = input.minStintLaps ?? 3
  const maxRemainingStops = input.maxRemainingStops ?? 3
  const flatCurveToleranceS = input.flatCurveToleranceS ?? 3.0

  const empty = (lap: number, costCurve: Map<number, number>, reason: string): PitRecommendation => ({
    lap,
    confidence: 0,
    distribution: new Map(),
    costCurve,
    window: null,
    windowConfidence: null,
    reason,
  })

  // Rubber does not regenerate. A negative estimated rate is a failed estimate,
  // not a tyre that improves, and feeding it to the optimiser yields "stay out
  // forever" -- the Monaco failure the Python comments describe at length.
  const rateIsNoise = input.currentRate <= 0 || input.currentRate < input.currentRateSd
  const currentRate = Math.max(input.currentRate, 0)
  const freshRate = Math.max(input.freshRate, 0)

  const remaining = input.finalLap - input.decisionLap
  if (remaining < 2 * minStintLaps) {
    return empty(input.decisionLap, new Map(), 'too few laps remain to choose between stops')
  }

  const candidates: number[] = []
  for (let lap = input.decisionLap + 1; lap <= input.finalLap - minStintLaps; lap += 1) {
    candidates.push(lap)
  }
  if (candidates.length === 0) return empty(input.decisionLap, new Map(), 'no feasible pit lap')

  const costs = new Map<number, number>()
  for (const lap of candidates) {
    const lapsBefore = lap - input.decisionLap
    const lapsAfter = input.finalLap - lap
    const onCurrent = degradationLoss(currentRate, input.currentAge, lapsBefore)

    let bestForLap: number | null = null
    for (let nStints = 1; nStints <= maxRemainingStops; nStints += 1) {
      if (lapsAfter < nStints * minStintLaps) break
      const perStint = lapsAfter / nStints
      const total =
        onCurrent + nStints * pitLossS + nStints * degradationLoss(freshRate, 0, pyRound(perStint))
      if (bestForLap === null || total < bestForLap) bestForLap = total
    }
    if (bestForLap !== null && Number.isFinite(bestForLap)) costs.set(lap, bestForLap)
  }

  if (costs.size === 0) return empty(input.decisionLap, costs, 'no feasible stop plan')

  let best = candidates[0]
  let bestCost = Infinity
  for (const [lap, cost] of costs) {
    if (cost < bestCost) {
      bestCost = cost
      best = lap
    }
  }

  if (rateIsNoise) {
    return empty(
      best,
      costs,
      'degradation is not distinguishable from zero on this tyre, so the tyre is not what decides this stop',
    )
  }

  const spread = Math.max(...costs.values()) - bestCost
  if (spread < flatCurveToleranceS) {
    return empty(
      best,
      costs,
      `degradation does not determine this stop: the best and worst laps differ by ${spread.toFixed(2)} s, ` +
        `below the ${flatCurveToleranceS} s threshold. Track position or regulation is deciding, not the tyre.`,
    )
  }

  const horizon = Math.max(remaining / 2, 1)
  const temperature = Math.max((input.currentRateSd * horizon * horizon) / 2, 1e-3)
  const lapsInPlay = [...costs.keys()]
  const values = lapsInPlay.map((lap) => -(costs.get(lap) as number) / temperature)
  const top = Math.max(...values)
  const weights = values.map((v) => Math.exp(v - top))
  const sum = weights.reduce((a, b) => a + b, 0)
  const distribution = new Map<number, number>()
  lapsInPlay.forEach((lap, i) => distribution.set(lap, weights[i] / sum))

  // The one-second window. `/pit-window` reports the same construction under
  // the name `window_within_1s`; it is computed here from this optimiser's own
  // cost curve so the window and the recommendation come from one model.
  const within = lapsInPlay.filter((lap) => (costs.get(lap) as number) <= bestCost + 1.0)
  const window: [number, number] | null = within.length
    ? [Math.min(...within), Math.max(...within)]
    : null
  const windowConfidence = within.length
    ? within.reduce((total, lap) => total + (distribution.get(lap) ?? 0), 0)
    : null

  return {
    lap: best,
    confidence: distribution.get(best) ?? 0,
    distribution,
    costCurve: costs,
    window,
    windowConfidence,
    reason: '',
  }
}

/**
 * The fresh-tyre rate exp22 uses: the mean of the other compounds the model has
 * an estimate for. The same choice for every model on the ladder, which is what
 * makes the benchmark a comparison of degradation estimates rather than of
 * strategy code.
 */
export function freshRateFor(compound: string, rates: Record<string, { rate: number }>): number {
  const others = Object.entries(rates)
    .filter(([name]) => name !== compound)
    .map(([, value]) => value.rate)
  if (!others.length) return rates[compound]?.rate ?? 0
  return others.reduce((a, b) => a + b, 0) / others.length
}
