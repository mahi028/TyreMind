/**
 * exp30 — what a confidence number was worth, before and after.
 *
 * Two findings, and the second is the one that sells.
 *
 * **Before.** The pit window shipped as two laps either side of the
 * recommendation with a probability read off a softmax nobody had validated. It
 * claimed 31.9% and delivered 24.0% on 246 real stops. Overconfident by 7.9
 * points, with the error worst where a strategist is most likely to act
 * without hedging.
 *
 * **After.** Split conformal replaces both choices. Claiming 90% now delivers
 * about 90%, claiming 80% delivers about 83%, claiming 50% delivers about 52%.
 *
 * But being calibrated is not the differentiator, because anyone can be
 * calibrated by widening: a window of ±30 laps covers everything and decides
 * nothing. The differentiator is calibrated *and narrow*, which is why the
 * width comparison gets more room on this screen than the reliability figures
 * do. For the same 90% guarantee we need 13 laps where Cappello & Hoegh's
 * published model needs 17 and Heilmeier's needs 26.
 */

import { useMemo } from 'react'
import { Caption, Fact, Note } from '../briefing/ui'
import { calibrate, type CalibrationReading } from './conformal'
import { OURS, type Exp30 } from './types'

const TARGETS = [0.5, 0.8, 0.9]
const HEADLINE_TARGET = 0.9

export function Exp30Calibration({ exp30 }: { exp30?: Exp30 }) {
  const perModel = useMemo(() => {
    if (!exp30?.rows) return []
    const byModel = new Map<string, number[]>()
    for (const row of exp30.rows) {
      const miss = Math.abs(row.recommended - row.actual)
      if (!Number.isFinite(miss)) continue
      const bucket = byModel.get(row.model)
      if (bucket) bucket.push(miss)
      else byModel.set(row.model, [miss])
    }
    return [...byModel.entries()]
      // Fewer than forty stops cannot represent a 90% quantile with a
      // finite-sample correction, and a model that thin is excluded rather than
      // shown with an infinite window.
      .filter(([, misses]) => misses.length >= 40)
      .map(([model, misses]) => ({ model, readings: calibrate(misses, TARGETS) }))
      .sort((a, b) => headline(a.readings).halfWidthLaps - headline(b.readings).halfWidthLaps)
  }, [exp30])

  if (!exp30?.summary) {
    return (
      <section className="plate p-5">
        <h2 className="text-[18px] font-semibold text-ink">Is the confidence number real?</h2>
        <p className="mt-2 text-[12.5px] text-ink-faint">
          Run <span className="num">experiments/exp30_pit_confidence_calibration.py</span> to
          populate this. Nothing is shown in its place.
        </p>
      </section>
    )
  }

  const ours = exp30.summary.find((r) => r.model === OURS)
  const oursCalibrated = perModel.find((r) => r.model === OURS)
  const best = perModel[0]

  return (
    <section className="plate">
      <header className="border-b border-line px-5 py-4">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <h2 className="text-[22px] leading-tight font-semibold tracking-[-0.02em] text-ink">
            We were overconfident. We measured it, and fixed it.
          </h2>
          <span className="num text-[11px] text-ink-faint">
            exp30 · {exp30.n_sessions} races · {ours?.n ?? '—'} real stops
          </span>
        </div>
        <div className="rule-accent mt-3" />
      </header>

      {/* --- before and after --------------------------------------------- */}
      {ours && (
        <div className="grid gap-px bg-line lg:grid-cols-2">
          <div className="bg-surface px-5 py-5">
            <Caption>Before · the window we used to ship</Caption>
            <ClaimVsDelivered
              claimed={ours.mean_window_mass}
              delivered={ours.observed_window_hit}
              width={`± ${exp30.window_half_width_laps} laps`}
            />
            <p className="mt-3 max-w-[54ch] text-[12px] leading-relaxed text-ink-dim">
              Two laps either side of the recommendation, with the probability read off a softmax
              whose temperature was chosen because it was dimensionally sensible. Neither the width
              nor the temperature had ever been checked against an outcome. Checked, they were
              wrong by{' '}
              <span className="num text-alert">
                {Math.abs(ours.calibration_gap * 100).toFixed(1)}
              </span>{' '}
              points in the direction that gets a strategist hurt.
            </p>
          </div>

          <div className="bg-surface px-5 py-5">
            <Caption>After · conformal, fitted on held-out stops</Caption>
            {oursCalibrated ? (
              <div className="space-y-2.5">
                {[...oursCalibrated.readings].reverse().map((r, i) => (
                  <div
                    key={r.targetCoverage}
                    className="settle grid grid-cols-[64px_1fr_auto] items-center gap-3"
                    style={{ ['--i' as string]: i }}
                  >
                    <span className="num text-[15px] text-ink-dim">
                      {Math.round(r.targetCoverage * 100)}%
                    </span>
                    <div className="relative h-5 bg-raised">
                      <div
                        className="absolute top-0 bottom-0 w-px bg-line-bright"
                        style={{ left: `${r.targetCoverage * 100}%` }}
                      />
                      <div
                        className="grow-x absolute top-0 bottom-0 left-0"
                        style={{
                          width: `${Math.min(100, r.heldOutCoverage * 100)}%`,
                          background: `color-mix(in oklab, var(--color-good) ${
                            55 + i * 12
                          }%, transparent)`,
                          ['--i' as string]: i,
                        }}
                      />
                    </div>
                    <span className="num text-right text-[15px] text-ink">
                      {(r.heldOutCoverage * 100).toFixed(1)}%
                      <span className="ml-1.5 text-[10.5px] text-ink-faint">
                        ± {r.heldOutHalfWidth.toFixed(0)} laps
                      </span>
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-[12px] text-ink-faint">
                exp30 served no per-stop rows for {OURS}, so nothing is recomputed here.
              </p>
            )}
            <p className="mt-3 max-w-[54ch] text-[12px] leading-relaxed text-ink-dim">
              The claim is the hairline; the bar is what was delivered on stops the threshold was
              not fitted on. The width beside each is the price of that guarantee, in laps, and it
              is printed every time the coverage is.
            </p>
          </div>
        </div>
      )}

      {/* --- the width comparison ----------------------------------------- */}
      <div className="border-t border-line px-5 py-5">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <h3 className="text-[17px] font-semibold tracking-tight text-ink">
            Anyone can be calibrated by widening. This is the part that is hard.
          </h3>
          <span className="text-[11px] text-ink-faint">
            same {Math.round(HEADLINE_TARGET * 100)}% guarantee, every model
          </span>
        </div>
        <p className="mt-2 max-w-[92ch] text-[13px] leading-relaxed text-ink-dim">
          A window of ± 30 laps covers almost every stop and decides nothing. So the comparison
          that matters is not whether a model is calibrated, it is how many laps it has to give
          away to <em>get</em> calibrated. Every model below is put through the same split-conformal
          procedure on its own stops, at the same {Math.round(HEADLINE_TARGET * 100)}% target.
        </p>

        <WidthComparison perModel={perModel} />

        {best && ours && oursCalibrated && (
          <div className="mt-5 grid gap-5 sm:grid-cols-3">
            <Fact
              label="Our width at 90%"
              value={headline(oursCalibrated.readings).halfWidthLaps.toFixed(0)}
              unit="laps"
              foot={`Delivering ${(headline(oursCalibrated.readings).heldOutCoverage * 100).toFixed(
                1,
              )}% on held-out stops.`}
            />
            <Fact
              label="Nearest published model"
              value={widthOf(perModel, 'Cappello & Hoegh state-space')}
              unit="laps"
              foot="Cappello & Hoegh, for the same guarantee."
            />
            <Fact
              label="Lift on the chosen lap"
              value={`${ours.lift.toFixed(2)}×`}
              foot={`Probability we place on the lap the driver actually chose, against the ${(
                ours.uniform_chance * 100
              ).toFixed(1)}% a uniform guess would place.`}
            />
          </div>
        )}
      </div>

      {/* --- the reliability curve we had before -------------------------- */}
      {exp30.reliability_tyremind?.length > 0 && (
        <div className="border-t border-line px-5 py-5">
          <Caption>Where the old overconfidence lived</Caption>
          <div className="mt-1 space-y-2.5">
            {exp30.reliability_tyremind.map((bin, i) => (
              <div
                key={i}
                className="settle grid grid-cols-[minmax(120px,auto)_1fr_minmax(120px,auto)] items-center gap-3"
                style={{ ['--i' as string]: i }}
              >
                <span className="num text-[12px] text-ink-dim">
                  claimed {(bin.claimed_low * 100).toFixed(0)}–{(bin.claimed_high * 100).toFixed(0)}%
                </span>
                <div className="relative h-6 bg-raised">
                  <div
                    className="hatch absolute top-0 bottom-0"
                    style={{
                      left: `${bin.observed * 100}%`,
                      width: `${Math.max(0, (bin.mean_claimed - bin.observed) * 100)}%`,
                      ['--hatch-color' as string]: 'var(--color-alert)',
                    }}
                  />
                  <div
                    className="grow-x absolute top-0 bottom-0 left-0"
                    style={{
                      width: `${bin.observed * 100}%`,
                      background: 'color-mix(in oklab, var(--color-ink-dim) 45%, transparent)',
                      ['--i' as string]: i,
                    }}
                  />
                  <div
                    className="absolute top-0 bottom-0 w-[2px]"
                    style={{
                      left: `${bin.mean_claimed * 100}%`,
                      background: 'var(--color-alert)',
                    }}
                  />
                </div>
                <span className="num text-right text-[12px] text-ink">
                  {(bin.observed * 100).toFixed(1)}%
                  <span className="ml-1 text-[10.5px] text-ink-faint">of {bin.n}</span>
                </span>
              </div>
            ))}
          </div>
          <Note>
            The orange tick is what the old window claimed in that band; the solid bar is what it
            delivered; the hatching between them is the shortfall. It is drawn as hatching rather
            than as a solid colour on purpose &mdash; it is not a quantity to be proud of. This is
            the figure the conformal correction above replaces.
          </Note>
        </div>
      )}
    </section>
  )
}

function headline(readings: CalibrationReading[]): CalibrationReading {
  return readings.find((r) => r.targetCoverage === HEADLINE_TARGET) ?? readings[readings.length - 1]
}

function widthOf(
  perModel: { model: string; readings: CalibrationReading[] }[],
  model: string,
): string {
  const entry = perModel.find((m) => m.model === model)
  return entry ? headline(entry.readings).halfWidthLaps.toFixed(0) : '—'
}

/** Claimed against delivered, as one mark. The gap between them is the point. */
function ClaimVsDelivered({
  claimed,
  delivered,
  width,
}: {
  claimed: number
  delivered: number
  width: string
}) {
  return (
    <div>
      <div className="flex items-baseline gap-4">
        <div>
          <div className="num text-[42px] leading-none font-medium text-ink-dim">
            {(claimed * 100).toFixed(1)}
            <span className="ml-0.5 text-[16px] text-ink-faint">%</span>
          </div>
          <div className="mt-1 text-[11px] text-ink-faint">claimed</div>
        </div>
        <span className="text-[22px] text-ink-faint">→</span>
        <div>
          <div className="num text-[42px] leading-none font-medium text-alert">
            {(delivered * 100).toFixed(1)}
            <span className="ml-0.5 text-[16px] text-ink-faint">%</span>
          </div>
          <div className="mt-1 text-[11px] text-ink-faint">delivered, on {width}</div>
        </div>
      </div>
      <div className="relative mt-3.5 h-6 bg-raised">
        <div
          className="grow-x absolute top-0 bottom-0 left-0"
          style={{
            width: `${delivered * 100}%`,
            background: 'color-mix(in oklab, var(--color-alert) 45%, transparent)',
          }}
        />
        <div
          className="hatch absolute top-0 bottom-0"
          style={{
            left: `${delivered * 100}%`,
            width: `${Math.max(0, (claimed - delivered) * 100)}%`,
            ['--hatch-color' as string]: 'var(--color-alert)',
          }}
        />
        <div
          className="absolute top-0 bottom-0 w-[2px]"
          style={{ left: `${claimed * 100}%`, background: 'var(--color-ink)' }}
        />
      </div>
    </div>
  )
}

/**
 * Half-width at the headline guarantee, every model on one ruler.
 *
 * Sorted narrowest first, with the delivered coverage printed beside each bar,
 * because a narrow window is only a virtue if it still covers what it promised.
 * A bar with a short length and a coverage figure well under the target would
 * be a failure, not a win, and the two numbers have to sit together for that to
 * be visible.
 */
function WidthComparison({
  perModel,
}: {
  perModel: { model: string; readings: CalibrationReading[] }[]
}) {
  if (!perModel.length) {
    return (
      <p className="mt-3 text-[12px] text-ink-faint">
        exp30 served no per-stop rows, so no widths can be recomputed.
      </p>
    )
  }
  const widest = Math.max(...perModel.map((m) => headline(m.readings).halfWidthLaps))

  return (
    <div className="mt-4 space-y-2">
      {perModel.map((entry, i) => {
        const r = headline(entry.readings)
        const ours = entry.model === OURS
        const colour = ours ? 'var(--color-alert)' : 'var(--color-ink-faint)'
        return (
          <div
            key={entry.model}
            className="settle grid grid-cols-[minmax(150px,1.2fr)_2.4fr_minmax(150px,auto)] items-center gap-3"
            style={{ ['--i' as string]: i }}
          >
            <span
              className={`truncate text-[13px] ${ours ? 'font-medium text-ink' : 'text-ink-dim'}`}
              title={entry.model}
            >
              {entry.model}
            </span>
            <div className="relative h-7 bg-raised">
              <div
                className="grow-x absolute top-0 bottom-0 left-0 flex items-center justify-end pr-2"
                style={{
                  width: `${(r.halfWidthLaps / widest) * 100}%`,
                  background: `color-mix(in oklab, ${colour} ${ours ? 55 : 26}%, transparent)`,
                  borderRight: `2px solid ${colour}`,
                  ['--i' as string]: i,
                }}
              >
                <span className="num text-[13px] font-medium text-ink">
                  ± {r.halfWidthLaps.toFixed(0)}
                </span>
              </div>
            </div>
            <span className="num text-right text-[11.5px] text-ink-dim">
              {(r.heldOutCoverage * 100).toFixed(1)}% delivered
              <span className="ml-1 text-ink-faint">· {r.n} stops</span>
            </span>
          </div>
        )
      })}
      <p className="pt-1 text-[11.5px] leading-relaxed text-ink-faint">
        Half-width in laps, fitted on every stop a model answered; coverage is leave-one-out, so
        each stop is scored against a threshold fitted without it. Models answering fewer than
        forty stops are left out entirely &mdash; a 90% quantile with a finite-sample correction
        does not exist below that, and the honest output there is no window rather than a wide one.
      </p>
    </div>
  )
}
