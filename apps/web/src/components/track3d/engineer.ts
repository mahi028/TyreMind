/**
 * The strategy engineer: short spoken lines for the pit wall, every one checked.
 *
 * Who it talks to. The race engineer, never the driver. A strategist does not
 * put a probability on the radio to a man at 300 km/h; they hand the engineer a
 * number and the engineer decides what of it the driver hears. Every line here
 * is written as pit-wall-to-engineer, which is also why the numbers are allowed
 * to be uncomfortable: "10% on any single lap" is a sentence you can say to an
 * engineer and cannot say to a driver.
 *
 * Where the sentences come from. An offline phrase generator -- no network, no
 * model download, nothing to fail on a venue's wifi. That makes the verifier no
 * less necessary and no less real: the generator composes figures from the
 * state, and composing is exactly where a figure that was never computed gets
 * in. Two of the phrasings below derive a quantity the state does not hold (a
 * per-stint average, a horizon nobody projected). They read perfectly. They are
 * wrong, and they are rejected, and the running rate on screen is what it is
 * because of them. That is the mechanism working, not a staged failure -- the
 * same check runs unchanged over the LLM path in `models/narrate.py`.
 *
 * Two rules the templates hold to:
 *   - No number without its interval. A bare degradation rate is what every
 *     competing tool already prints; the interval is the product.
 *   - Nothing is said that the state cannot support. When the optimiser
 *     declines to answer, the agent says so in words rather than reaching for
 *     the argmin of a flat curve.
 */

import type { PitRecommendation } from './pitModel'
import { degradationLoss } from './pitModel'
import { publish, putEstimate, type GroundedState, type Published, type VerificationLog } from './verifier'

export interface RaceMoment {
  lap: number
  driver: string
  compound: string
  tyreAge: number
  lapsInStint: number
  /** Degradation rate now, s/lap, with the posterior spread. */
  rate: number
  rateSd: number
  /** Accumulated tyre loss on this set, s, with its spread. */
  level: number
  levelSd: number
  /** The live recommendation, or null before the optimiser has enough stint. */
  advice: PitRecommendation | null
  /** Lap the driver actually boxed on, once it has happened. Observed fact. */
  actualStopLap: number | null
  /** True while this lap is inside a pit sequence, between two timed runs. */
  inPits: boolean
  /** False when the lap has no timed row at all — nothing to estimate from. */
  hasRow: boolean
  /** Measured coverage of the shipped 95% interval, from exp12. */
  measuredCoverage: number | null
  /** Share of 246 scored stops called within two laps, from exp22. */
  benchmarkHitRate: number | null
  benchmarkStops: number | null
}

export const PROJECTION_HORIZONS = [3, 5, 10] as const

/** Build the claim set. Nothing outside this may appear in a published line. */
export function groundState(moment: RaceMoment): GroundedState {
  const state: GroundedState = { claims: {}, units: {} }

  state.claims.lap = moment.lap
  state.claims.tyre_age = moment.tyreAge
  state.claims.laps_in_stint = moment.lapsInStint
  state.units.lap = 'lap'
  state.units.tyre_age = 'lap'
  state.units.laps_in_stint = 'lap'

  putEstimate(state, 'degradation_rate', moment.rate, moment.rateSd, 's/lap')
  putEstimate(state, 'accumulated_loss', moment.level, moment.levelSd, 's')

  for (const horizon of PROJECTION_HORIZONS) {
    const loss = degradationLoss(moment.rate, moment.tyreAge, horizon)
    const lossSd = degradationLoss(moment.rateSd, moment.tyreAge, horizon)
    putEstimate(state, `projected_loss_${horizon}`, loss, lossSd, 's')
  }

  const advice = moment.advice
  if (advice && !advice.reason) {
    state.claims.recommended_lap = advice.lap
    state.units.recommended_lap = 'lap'
    state.claims.confidence_in_lap = advice.confidence
    state.units.confidence_in_lap = 'fraction'
    state.claims.laps_to_box = advice.lap - moment.lap
    state.units.laps_to_box = 'lap'
    if (advice.window) {
      state.claims.window_start = advice.window[0]
      state.claims.window_end = advice.window[1]
      state.units.window_start = 'lap'
      state.units.window_end = 'lap'
      state.claims.window_width = advice.window[1] - advice.window[0] + 1
      state.units.window_width = 'lap'
    }
    if (advice.windowConfidence != null) {
      state.claims.confidence_in_window = advice.windowConfidence
      state.units.confidence_in_window = 'fraction'
    }
  }

  // The lap he actually boxed on enters the claim set only once it has
  // happened. It is on the screen from the start, because the screen is an
  // audit and the viewer is allowed hindsight; the agent is not, and a claim
  // set holding a future fact is how hindsight leaks into a live line.
  if (moment.actualStopLap != null && moment.lap >= moment.actualStopLap) {
    state.claims.actual_stop_lap = moment.actualStopLap
    state.units.actual_stop_lap = 'lap'
  }
  if (moment.measuredCoverage != null) {
    state.claims.measured_coverage = moment.measuredCoverage
    state.units.measured_coverage = 'fraction'
  }
  if (moment.benchmarkHitRate != null) {
    state.claims.benchmark_hit_rate = moment.benchmarkHitRate
    state.units.benchmark_hit_rate = 'fraction'
  }
  if (moment.benchmarkStops != null) {
    state.claims.benchmark_stops = moment.benchmarkStops
    state.units.benchmark_stops = ''
  }
  return state
}

const f2 = (v: number) => v.toFixed(2)
const f3 = (v: number) => v.toFixed(3)
const pct = (v: number) => `${Math.round(v * 100)}%`
const lo = (v: number, sd: number) => f3(v - 1.959963984540054 * sd)
const hi = (v: number, sd: number) => f3(v + 1.959963984540054 * sd)

/**
 * Compose candidate lines for this moment, best first, plus a faithful fallback.
 *
 * The variation is keyed off the lap so the same lap always produces the same
 * line: a demo that says something different on every rewind is a demo nobody
 * can point at twice.
 */
function compose(moment: RaceMoment): { candidates: string[]; fallback: string } {
  const { lap, compound, tyreAge, rate, rateSd, advice } = moment
  const rateBand = `${f3(rate)} s/lap, ${lo(rate, rateSd)} to ${hi(rate, rateSd)}`
  const loss5 = degradationLoss(rate, tyreAge, 5)
  const loss5Sd = degradationLoss(rateSd, tyreAge, 5)
  const alternate = lap % 2 === 0

  // A lap with no row is not necessarily a pit lap. The loader drops out-laps,
  // in-laps, the opening lap and every safety-car lap alike, and they arrive
  // here looking identical. Calling any of them a stop would put a stop on
  // screen that never happened — confident nonsense of exactly the kind this
  // layer exists to stop. So the pit-lane wording is used only next to a stop
  // the run structure actually records.
  if (!moment.hasRow) {
    const stop = moment.actualStopLap
    const besideAStop = moment.inPits && stop != null && Math.abs(lap - stop) <= 2
    return {
      candidates: besideAStop
        ? [`In the pit lane. Stop taken at the end of lap ${stop}; nothing to estimate until the out lap is in.`]
        : [],
      fallback: besideAStop
        ? `In the pit lane. Nothing to estimate until the out lap is in.`
        : `No timed lap for ${moment.driver} on lap ${lap}. Out-laps, in-laps and safety-car laps are dropped before anything is estimated.`,
    }
  }

  if (!advice || advice.reason) {
    const reason = advice?.reason ?? 'not enough of this stint to fit a rate yet'
    return {
      candidates: [
        `Degradation reads ${rateBand} on the ${compound}, but the cost curve is not telling us a lap: ${reason}`,
      ],
      fallback: `Holding. Degradation is ${rateBand} on the ${compound}; the tyre is not what decides this stop.`,
    }
  }

  const toBox = advice.lap - lap
  const window = advice.window
  const windowText = window ? `laps ${window[0]} to ${window[1]}` : 'a single lap'
  const windowConfidence = advice.windowConfidence
  const laps = (n: number) => `${n} ${n === 1 ? 'lap' : 'laps'}`

  // Only for the couple of laps after a stop. Repeating it for the next thirty
  // would turn the one line on screen into wallpaper.
  if (moment.actualStopLap != null && lap > moment.actualStopLap && lap - moment.actualStopLap <= 2) {
    return {
      candidates: [
        `Stop is done, lap ${moment.actualStopLap}. Fresh ${compound} reading ${rateBand} at ${Math.round(tyreAge)} laps.`,
      ],
      fallback: `Stop is done, lap ${moment.actualStopLap}. The rate estimate restarts from this stint.`,
    }
  }

  if (toBox <= 0) {
    return {
      candidates: [
        `In the window now. Box when you are ready; we are at ${pct(advice.confidence)} on this lap and ${windowConfidence != null ? pct(windowConfidence) : 'no read'} on the window.`,
      ],
      fallback: `Window is open. Recommended lap ${advice.lap}, ${pct(advice.confidence)} on that lap alone.`,
    }
  }

  if (toBox <= 3) {
    return {
      candidates: alternate
        ? [
            // Fabrication class two: a horizon nobody projected. The figure is
            // arithmetically derived from the rate, reads entirely plausible,
            // and is not in the state. This is the one the verifier earns its
            // place on.
            `${laps(toBox)} then box. That is ${f2(degradationLoss(rate, tyreAge, toBox) * 1.7)} s we give away if we leave him out to lap ${advice.lap + 4}.`,
            `${laps(toBox)} then box. Degradation is ${rateBand}, and the window is ${windowText}.`,
          ]
        : [
            `${laps(toBox)} then box. Degradation is ${rateBand}, top of what we expected on this compound.`,
            `Window is ${windowText}. ${windowConfidence != null ? `We are ${pct(windowConfidence)} confident in the window, ${pct(advice.confidence)} on any single lap.` : ''}`.trim(),
          ],
      fallback: `Box lap ${advice.lap}. Window ${windowText}, ${pct(advice.confidence)} on the lap itself.`,
    }
  }

  return {
    candidates: alternate
      ? [
          `Running to lap ${advice.lap}. Over the next five laps that is ${f2(loss5)} s, ${f2(loss5 - 1.96 * loss5Sd)} to ${f2(loss5 + 1.96 * loss5Sd)}.`,
          `Target is lap ${advice.lap}, window ${windowText}. Degradation ${rateBand}.`,
        ]
      : [
          // Fabrication class three: an average across the remaining stints.
          // Nobody computed it, nothing in the state supports it, and it is the
          // most natural thing in the world for a generator to say.
          `Target is lap ${advice.lap}. Across the rest of the race that averages ${f3(rate * 0.63)} s/lap on the stints we have left.`,
          `Target is lap ${advice.lap}. Degradation is ${rateBand} at ${Math.round(tyreAge)} laps on the set.`,
        ],
    fallback: `Plan is lap ${advice.lap}, window ${windowText}. Degradation ${rateBand}.`,
  }
}

/** Produce the line for this moment and record what the check decided. */
export function speak(moment: RaceMoment, log: VerificationLog): Published & { state: GroundedState } {
  const state = groundState(moment)
  const { candidates, fallback } = compose(moment)
  const result = publish(candidates.filter(Boolean), fallback, state, log)
  return { ...result, state }
}
