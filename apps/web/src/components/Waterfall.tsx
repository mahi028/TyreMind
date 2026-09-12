/**
 * Decomposition of one lap into its causes.
 *
 * Signed bars from a common zero rather than a stacked waterfall — the useful
 * comparison is between magnitudes, not the running total. A race engineer needs
 * to see which term is biggest and whether the tyre term differs from zero.
 */

import type { Decomposition } from '../lib/api'
import { compoundColour, signed, TERM_COLOUR } from '../lib/api'
import { EstimateTag } from './primitives'

export function Waterfall({ decomposition }: { decomposition: Decomposition }) {
  const terms = [
    ...decomposition.contributions,
    {
      key: 'residual',
      label: 'Unexplained',
      seconds: decomposition.residual,
      sd: 0,
      ci95: [decomposition.residual, decomposition.residual] as [number, number],
      is_tyre: false,
    },
  ]

  const extent = Math.max(
    ...terms.map((t) => Math.abs(t.ci95[0])),
    ...terms.map((t) => Math.abs(t.ci95[1])),
    Math.abs(decomposition.observed_delta),
    0.1,
  )
  const domain: [number, number] = [-extent * 1.1, extent * 1.1]
  const toPct = (v: number) => ((v - domain[0]) / (domain[1] - domain[0])) * 100

  const tyreColour = compoundColour(decomposition.compound)
  const slower = decomposition.observed_delta > 0

  return (
    <div>
      {/* Header: total delta + tyre contribution */}
      <div className="mb-4 flex flex-wrap items-end justify-between gap-4 border-b border-line pb-3.5">
        <div>
          <div className="label-caps mb-1">
            {decomposition.driver} · lap {decomposition.session_lap} vs lap{' '}
            {decomposition.reference_lap}
          </div>
          <div className="num text-[30px] leading-none font-semibold tracking-[-0.02em] text-ink">
            {signed(decomposition.observed_delta)}
            <span className="ml-1.5 text-[12px] font-normal text-ink-faint">s</span>
          </div>
          <div className="mt-1 text-[11px] text-ink-dim">
            {slower ? 'slower' : 'faster'} than reference
          </div>
        </div>
        <div className="text-right">
          <div className="label-caps mb-1">of which the tyre is</div>
          <div
            className="num text-[30px] leading-none font-semibold tracking-[-0.02em]"
            style={{ color: tyreColour }}
          >
            {signed(decomposition.tyre_seconds)}
            <span className="ml-1.5 text-[12px] font-normal text-ink-faint">s</span>
          </div>
          <div className="mt-1 text-[11px] text-ink-dim">
            age {decomposition.tyre_age.toFixed(0)} laps · {decomposition.compound}
          </div>
        </div>
      </div>

      {/* Bar rows */}
      <div className="space-y-2">
        {terms.map((term) => {
          const colour =
            term.is_tyre
              ? tyreColour
              : TERM_COLOUR[term.key] ?? 'var(--color-residual)'
          const lo    = toPct(Math.min(term.ci95[0], term.ci95[1]))
          const hi    = toPct(Math.max(term.ci95[0], term.ci95[1]))
          const barLo = toPct(Math.min(0, term.seconds))
          const barHi = toPct(Math.max(0, term.seconds))

          return (
            <div
              key={term.key}
              className="grid items-center gap-3"
              style={{ gridTemplateColumns: '120px 1fr 80px' }}
            >
              {/* Label */}
              <div
                className={`text-[12px] ${
                  term.is_tyre ? 'font-semibold text-ink' : 'text-ink-dim'
                }`}
              >
                {term.label}
              </div>

              {/* Bar track */}
              <div className="relative h-[18px] rounded-pill bg-raised">
                {/* Zero reference */}
                <div
                  className="absolute top-0 bottom-0 w-px"
                  style={{
                    left: `${toPct(0)}%`,
                    background: 'var(--color-line-bright)',
                  }}
                />
                {/* Uncertainty beam */}
                {term.sd > 0 && (
                  <div
                    className="beam absolute top-0 bottom-0"
                    style={{
                      left: `${lo}%`,
                      width: `${Math.max(hi - lo, 0.5)}%`,
                      ['--beam-color' as string]: colour,
                    }}
                  />
                )}
                {/* Solid bar */}
                <div
                  className="absolute top-1 bottom-1 rounded-[1px]"
                  style={{
                    left:       `${barLo}%`,
                    width:      `${Math.max(barHi - barLo, 0.4)}%`,
                    background: colour,
                    opacity:    term.is_tyre ? 1 : 0.75,
                  }}
                />
              </div>

              {/* Value */}
              <div
                className={`num text-right text-[12.5px] ${
                  term.is_tyre ? 'font-medium text-ink' : 'text-ink-dim'
                }`}
              >
                {signed(term.seconds)}
              </div>
            </div>
          )
        })}
      </div>

      <KeyInsight decomposition={decomposition} />
    </div>
  )
}

function KeyInsight({ decomposition }: { decomposition: Decomposition }) {
  const { observed_delta, tyre_seconds, confounder_seconds, tyre_share } = decomposition
  const gotFaster  = observed_delta <= 0
  const tyreIsWorse = tyre_seconds > 0.05

  return (
    <div
      className="mt-5 rounded-md border-l-2 py-2 pl-3.5 pr-2"
      style={{
        borderLeftColor: 'var(--color-alert)',
        background: 'color-mix(in oklab, var(--color-alert) 4%, transparent)',
      }}
    >
      <div className="mb-1.5 flex items-center gap-2">
        <span className="label-caps" style={{ color: 'var(--color-alert)' }}>
          What this means
        </span>
        <EstimateTag />
      </div>
      <p className="max-w-[62ch] text-[13px] leading-relaxed text-ink">
        {gotFaster && tyreIsWorse ? (
          <>
            The car was{' '}
            <strong>{Math.abs(observed_delta).toFixed(2)} s faster</strong> than the
            reference lap, so the stopwatch says the tyre is fine. It is not: the
            tyre lost{' '}
            <strong style={{ color: compoundColour(decomposition.compound) }}>
              {tyre_seconds.toFixed(2)} s
            </strong>
            , and the gain came from elsewhere, mostly fuel burn-off worth{' '}
            {Math.abs(confounder_seconds).toFixed(2)} s. Reading pace alone would
            miss a degrading tyre completely.
          </>
        ) : tyre_share > 1 ? (
          <>
            The tyre cost{' '}
            <strong style={{ color: compoundColour(decomposition.compound) }}>
              {tyre_seconds.toFixed(2)} s
            </strong>
            , more than the{' '}
            <strong>{Math.abs(observed_delta).toFixed(2)} s</strong> the stopwatch
            shows, because conditions gave{' '}
            {Math.abs(confounder_seconds).toFixed(2)} s back. The slowdown on
            screen understates the tyre.
          </>
        ) : Number.isFinite(tyre_share) && tyre_share >= 0 ? (
          <>
            Only <strong>{(tyre_share * 100).toFixed(0)}%</strong> of this slowdown
            is the tyre. The remaining{' '}
            {((1 - tyre_share) * 100).toFixed(0)}% comes from conditions that
            changed around the car, not from the rubber on it.
          </>
        ) : (
          <>
            The car matched the reference lap, but the underlying terms did not
            cancel by accident. The tyre contributed {signed(tyre_seconds, 2)} s and
            everything else {signed(confounder_seconds, 2)} s.
          </>
        )}
      </p>
    </div>
  )
}
