/**
 * Split conformal, recomputed in the browser from exp30's raw stops.
 *
 * This is a deliberate duplication of `tyremind.models.conformal` and
 * `scripts/build_pit_calibration.py`. The shipped artefact calibrates one model
 * -- ours -- because that is the one the API serves a window for. The claim the
 * product actually rests on is comparative: for the same guarantee, how much
 * narrower is our window than everybody else's? Answering that needs the same
 * procedure run over all six models, and exp30 ships the per-stop rows that
 * make it possible without inventing anything.
 *
 * Held-out coverage here is leave-one-out rather than the artefact's repeated
 * random half-splits. Both are honest; leave-one-out is used because it is
 * deterministic, so this panel shows the same number on every load and a judge
 * can reproduce it by hand. The two agree to a few tenths of a point.
 */

/**
 * The finite-sample-corrected (1 - alpha) quantile of nonconformity scores.
 *
 * The ceil((n+1)(1-alpha))-th order statistic rather than the plain empirical
 * quantile: the +1 accounts for the test point, which under exchangeability is
 * equally likely to take any rank among the n + 1 scores. That correction is
 * the entire reason split conformal carries a guarantee at finite n.
 *
 * @returns The threshold, or Infinity when the sample is too small to
 *   represent the requested coverage. Infinity is not a window, and callers are
 *   expected to decline rather than to clamp it.
 */
export function conformalQuantile(sortedScores: number[], alpha: number): number {
  const n = sortedScores.length
  if (n === 0) return Infinity
  const k = Math.ceil((n + 1) * (1 - alpha))
  if (k > n) return Infinity
  return sortedScores[k - 1]
}

export interface CalibrationReading {
  targetCoverage: number
  /** Threshold fitted on every stop. What would be shipped. */
  halfWidthLaps: number
  /** Coverage achieved on stops the threshold was not fitted on. */
  heldOutCoverage: number
  /** Mean of the leave-one-out thresholds, which is the width actually paid. */
  heldOutHalfWidth: number
  n: number
}

/**
 * Fit and score a conformal pit window on one model's stops.
 *
 * @param misses |recommended lap − actual lap| for every stop the model answered.
 * @param targets Coverage levels to fit, e.g. [0.5, 0.8, 0.9].
 */
export function calibrate(misses: number[], targets: number[]): CalibrationReading[] {
  const clean = misses.filter((m) => Number.isFinite(m))
  const sorted = [...clean].sort((a, b) => a - b)
  const n = sorted.length

  return targets.map((target) => {
    const alpha = 1 - target
    const halfWidth = conformalQuantile(sorted, alpha)

    // Leave-one-out: refit without stop i, then ask whether stop i lands inside.
    // Removing one element from an already-sorted copy keeps this O(n^2) with a
    // tiny constant, which at n < 300 is nothing.
    let covered = 0
    let widthSum = 0
    let scored = 0
    for (let i = 0; i < n; i++) {
      const held = clean[i]
      const rest = sorted.slice()
      const at = rest.indexOf(held)
      if (at >= 0) rest.splice(at, 1)
      const threshold = conformalQuantile(rest, alpha)
      if (!Number.isFinite(threshold)) continue
      scored += 1
      widthSum += threshold
      if (held <= threshold) covered += 1
    }

    return {
      targetCoverage: target,
      halfWidthLaps: halfWidth,
      heldOutCoverage: scored ? covered / scored : NaN,
      heldOutHalfWidth: scored ? widthSum / scored : NaN,
      n,
    }
  })
}
