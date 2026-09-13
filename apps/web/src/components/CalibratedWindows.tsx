/**
 * The pit window a strategist should actually act on.
 *
 * Two things are drawn here and they are not the same thing.
 *
 * The **simulation window** (`window_within_1s`) is a cost statement: every lap
 * inside it finishes within a second of the best. It says nothing about whether
 * the driver will box there.
 *
 * The **calibrated windows** are a coverage statement: a width fitted by split
 * conformal on real stops, carrying the coverage it actually achieved on stops
 * it was not fitted on. exp30 measured what the old hand-picked window was
 * worth -- it claimed 31.9% and delivered 24.0% -- so every level here prints
 * the claim and the measurement side by side rather than the claim alone. The
 * gap between them is the whole product of that experiment and it is not going
 * to be hidden behind a single reassuring percentage.
 *
 * The width is the price of the guarantee, and it is stated in laps at every
 * level, because a reader who sees "90% confident" without "± 13 laps" has been
 * told the easy half.
 */

import type { CalibratedWindow, PitWindow } from '../lib/api'

const LEVEL_COLOUR: Record<string, string> = {
  '0.5': 'var(--color-fuel)',
  '0.8': 'var(--color-medium)',
  '0.9': 'var(--color-good)',
}

function levelColour(target: number): string {
  return LEVEL_COLOUR[String(target)] ?? 'var(--color-good)'
}

/**
 * The refusal, rendered as a sentence.
 *
 * `/pit-window` returns `declined: true` with a reason when degradation is not
 * what decides the stop -- the rate indistinguishable from zero, or the best and
 * worst laps costing the same. Drawing an empty chart there would present the
 * absence of a recommendation as a recommendation of nothing. The reason is the
 * answer, so it gets the space the chart would have had.
 */
export function DeclinedWindow({ window: w }: { window: PitWindow }) {
  // The optimiser declines for two different reasons and they deserve
  // different words. One is about the tyre not being the deciding term; the
  // other is about there being no decision left to make. Printing the
  // degradation explanation under "too few laps remain" would be answering a
  // question the model did not ask.
  const reason = (w.reason ?? '').toLowerCase()
  const outOfRoad = reason.includes('too few laps') || reason.includes('no feasible')

  return (
    <div
      className="plate p-5"
      style={{ borderColor: 'color-mix(in oklab, var(--color-medium) 45%, transparent)' }}
    >
      <div className="flex items-baseline gap-2.5">
        <span
          className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
          style={{ background: 'var(--color-medium)' }}
        />
        <span className="text-[13px] font-semibold tracking-tight text-ink">
          No pit lap recommended
        </span>
      </div>

      <p className="mt-3 max-w-[64ch] text-[15px] leading-relaxed text-ink">
        {w.reason ? sentence(w.reason) : 'The optimiser declined to name a lap.'}
      </p>

      <p className="mt-3 max-w-[72ch] text-[12.5px] leading-relaxed text-ink-dim">
        {outOfRoad ? (
          <>
            This is the model declining, not the model failing. There is no longer a choice
            between stops to make: whatever is left of the race is shorter than the time it takes
            for a fresh set to pay back the pit lane. Any lap this screen named would be a lap
            picked out of a flat curve.
          </>
        ) : (
          <>
            This is the model declining, not the model failing. TyreMind only prices the part of a
            pit decision that runs through tyre degradation. When the degradation term does not
            separate the candidate laps, the stop is being decided by something it does not
            measure&nbsp;&mdash; track position, a safety car, an undercut on the car
            ahead&nbsp;&mdash; and naming a lap anyway would be inventing precision.
          </>
        )}
      </p>

      <p className="mt-3 max-w-[72ch] text-[11.5px] leading-relaxed text-ink-faint">
        Declining has a cost, and it is measured rather than waved away. Answer rate is part of how
        the router scores models: <span className="num">exp22</span> grades every model on the
        stops all of them answered, precisely so that a model which declines the hard ones cannot
        be flattered by its own silence. The answer rates are on the &ldquo;Which model
        answers&rdquo; screen, next to the errors they belong to.
      </p>
    </div>
  )
}

/** Capitalise the optimiser's lowercase reason without touching the rest. */
function sentence(text: string): string {
  const trimmed = text.trim()
  if (!trimmed) return trimmed
  return trimmed[0].toUpperCase() + trimmed.slice(1) + (/[.!?]$/.test(trimmed) ? '' : '.')
}

/**
 * Nested calibrated windows on a lap axis.
 *
 * @param windows The `calibrated_windows` list, widest guarantee last.
 * @param centre The recommended lap the windows are built around.
 * @param fromLap Left edge of the axis: the lap the decision is being made on.
 * @param totalLaps Right edge: the flag.
 * @param actualLap The lap the driver really boxed on, when it is known. Drawn
 *   as a separate mark, never merged into the model's own marks.
 */
export function CalibratedWindows({
  windows,
  centre,
  fromLap,
  totalLaps,
  actualLap,
}: {
  windows: CalibratedWindow[]
  centre: number
  fromLap: number
  totalLaps: number
  actualLap?: number | null
}) {
  if (!windows.length) {
    return (
      <p className="max-w-[68ch] text-[12.5px] leading-relaxed text-ink-faint">
        No calibrated window is being served. The conformal thresholds live in{' '}
        <span className="num">data/reference/pit_calibration.json</span>, built from exp30 by{' '}
        <span className="num">scripts/build_pit_calibration.py</span>. Rather than fall back to a
        width nobody measured, the API serves none.
      </p>
    )
  }

  // Widest first, so narrower levels paint on top of it.
  const ordered = [...windows].sort((a, b) => b.target_coverage - a.target_coverage)
  const widest = ordered[0]

  // The axis spans the decision lap to the flag, padded to the widest window so
  // a window that runs off the end of the race is visibly clipped rather than
  // silently rescaled.
  const lo = Math.min(fromLap, widest.low)
  const hi = Math.max(totalLaps, widest.high)
  const span = hi - lo || 1
  const pct = (lap: number) => ((lap - lo) / span) * 100
  const clamp = (v: number) => Math.min(100, Math.max(0, v))

  return (
    <div>
      <div className="relative" style={{ height: 30 * ordered.length + 26 }}>
        {ordered.map((w, i) => {
          const left = clamp(pct(w.low))
          const right = clamp(pct(w.high))
          const colour = levelColour(w.target_coverage)
          return (
            <div
              key={w.target_coverage}
              className="absolute right-0 left-0"
              style={{ top: i * 30, height: 24 }}
            >
              <div className="absolute inset-0 bg-raised" />
              <div
                className="band grow-x absolute top-0 bottom-0"
                style={{
                  left: `${left}%`,
                  width: `${Math.max(right - left, 0.8)}%`,
                  ['--band-color' as string]: colour,
                  ['--i' as string]: ordered.length - 1 - i,
                }}
              />
              <div
                className="absolute top-1/2 -translate-y-1/2 px-2 text-[11px] whitespace-nowrap"
                style={{ left: `${left}%`, color: colour }}
              >
                <span className="num font-medium">{Math.round(w.target_coverage * 100)}%</span>
              </div>
            </div>
          )
        })}

        {/* The recommendation. One vertical line through every band, because all
            three windows are centred on the same lap and a per-band tick would
            suggest otherwise. */}
        <div
          className="absolute top-0 w-[2px]"
          style={{
            left: `${clamp(pct(centre))}%`,
            height: 30 * ordered.length - 6,
            background: 'var(--color-alert)',
          }}
        />
        <div
          className="absolute text-[11px] whitespace-nowrap"
          style={{
            left: `${clamp(pct(centre))}%`,
            top: 30 * ordered.length,
            transform: 'translateX(-50%)',
            color: 'var(--color-alert)',
          }}
        >
          <span className="num">L{centre}</span> recommended
        </div>

        {actualLap != null && (
          <>
            <div
              className="ping absolute top-0 h-2.5 w-2.5 rounded-full"
              style={{
                left: `${clamp(pct(actualLap))}%`,
                transform: 'translate(-50%, -50%)',
                background: 'var(--color-ink)',
                ['--ping-color' as string]: 'var(--color-ink)',
              }}
            />
            <div
              className="absolute text-[11px] whitespace-nowrap text-ink"
              style={{
                left: `${clamp(pct(actualLap))}%`,
                top: 30 * ordered.length + 13,
                transform: 'translateX(-50%)',
              }}
            >
              <span className="num">L{actualLap}</span> actual
            </div>
          </>
        )}
      </div>

      <div className="mt-1 flex justify-between text-[10.5px] text-ink-faint">
        <span className="num">lap {lo}</span>
        <span className="num">lap {hi}</span>
      </div>

      <table className="mt-4 w-full text-[12.5px]">
        <thead>
          <tr className="border-b border-line text-[10.5px] text-ink-faint">
            <th className="py-1.5 text-left font-normal">we claim</th>
            <th className="text-right font-normal">we measured</th>
            <th className="text-right font-normal">width</th>
            <th className="text-right font-normal">laps</th>
            <th className="text-right font-normal">held-out stops</th>
          </tr>
        </thead>
        <tbody>
          {[...ordered].reverse().map((w, i) => {
            const colour = levelColour(w.target_coverage)
            const measured = w.measured_coverage
            // Signed, and not rounded away. Delivering more than claimed is
            // conservative and delivering less is overconfidence, and a reader
            // deciding whether to hedge needs to know which one this is.
            const gap = measured == null ? null : measured - w.target_coverage
            return (
              <tr
                key={w.target_coverage}
                className="settle border-b border-line/50"
                style={{ ['--i' as string]: i }}
              >
                <td className="py-2 text-left">
                  <span className="flex items-center gap-2">
                    <span
                      className="inline-block h-2.5 w-2.5 shrink-0"
                      style={{ background: colour }}
                    />
                    <span className="num text-ink">{Math.round(w.target_coverage * 100)}%</span>
                  </span>
                </td>
                <td className="num text-right text-ink">
                  {measured == null ? '—' : `${(measured * 100).toFixed(1)}%`}
                  {gap != null && (
                    <span
                      className="ml-1.5 text-[10.5px]"
                      style={{
                        color: gap >= 0 ? 'var(--color-good)' : 'var(--color-alert)',
                      }}
                    >
                      {gap >= 0 ? '+' : '−'}
                      {Math.abs(gap * 100).toFixed(1)}
                    </span>
                  )}
                </td>
                <td className="num text-right text-ink-dim">± {w.half_width_laps.toFixed(0)}</td>
                <td className="num text-right text-ink-dim">
                  {w.low}–{w.high}
                </td>
                <td className="num text-right text-ink-faint">{w.n_calibration ?? '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>

      <p className="mt-3 max-w-[80ch] text-[11.5px] leading-relaxed text-ink-faint">
        The claimed column is a target. The measured column is what that width delivered on real
        stops it was <em>not</em> fitted on, averaged over repeated splits of exp30&rsquo;s{' '}
        <span className="num">{widest.n_calibration ?? '—'}</span> stops. They are printed
        separately because the previous window&rsquo;s claim and delivery were nine points apart
        and nobody knew until the two were measured against each other.
      </p>
    </div>
  )
}
