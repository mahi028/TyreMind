/**
 * The belief, not the pick.
 *
 * A pit recommendation printed as one lap number is the thing every strategy
 * tool already ships, and it is the reason those tools cannot be audited: a
 * single lap hides whether the model was nearly certain or nearly indifferent.
 * These charts draw the whole distribution the optimiser produced over every
 * feasible lap, so the width of the belief is on screen next to its centre.
 *
 * The marker for the lap the driver actually took is the point of the whole
 * view. It is observed fact, drawn in a colour nothing modelled is allowed to
 * use, and it never moves.
 *
 * Plain SVG on a uniform viewBox — the aspect ratio is preserved rather than
 * stretched, because a chart whose type is squashed horizontally is the first
 * thing a reader distrusts. These redraw once a lap, so a charting library
 * would only be bringing its own opinions about axes to a screen read from two
 * metres.
 */

import { useMemo } from 'react'
import { useThemeColours } from '../../lib/theme'

/** Drawing width of the viewBox. Height is given per chart as an aspect. */
const W = 800

export interface BeliefChartProps {
  distribution: Map<number, number>
  recommendedLap: number
  window: [number, number] | null
  /** The lap the driver actually boxed on. Observed, not modelled. */
  actualStop: number | null
  /** Where the car is now, if the replay is running. */
  currentLap?: number | null
  height?: number
}

export function BeliefChart({
  distribution,
  recommendedLap,
  window,
  actualStop,
  currentLap = null,
  height = 220,
}: BeliefChartProps) {
  const colours = useThemeColours()
  const laps = useMemo(() => [...distribution.keys()].sort((a, b) => a - b), [distribution])

  const first = laps[0] ?? 0
  const last = laps[laps.length - 1] ?? 1
  const span = Math.max(1, last - first + 1)
  const padTop = 14
  const padBottom = 34
  const plot = height - padTop - padBottom
  const bandWidth = W / span
  const x = (lap: number) => ((lap - first) / span) * W
  const peak = laps.length ? Math.max(...distribution.values()) : 1

  const ticks = useMemo(() => {
    const step = span > 40 ? 10 : span > 18 ? 5 : 2
    const out: number[] = []
    for (let lap = Math.ceil(first / step) * step; lap <= last; lap += step) out.push(lap)
    return out
  }, [first, last, span])

  if (!laps.length) {
    return (
      <div className="flex h-[140px] items-center justify-center px-6 text-center text-[12.5px] text-ink-dim">
        No distribution on this stint — the optimiser declined to answer, which is
        an output and not a gap.
      </div>
    )
  }

  const probabilityOnActual = actualStop != null ? (distribution.get(actualStop) ?? 0) : null

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${height}`} className="w-full" role="img"
        aria-label="Probability that each lap is the right lap to box on">
        {window && (
          <rect
            x={x(window[0])}
            y={padTop - 6}
            width={Math.max(bandWidth, x(window[1]) - x(window[0]) + bandWidth)}
            height={plot + 6}
            fill={colours.alert}
            opacity={0.1}
          />
        )}

        {laps.map((lap) => {
          const p = distribution.get(lap) ?? 0
          const h = peak > 0 ? (p / peak) * plot : 0
          const isPick = lap === recommendedLap
          const isActual = lap === actualStop
          return (
            <rect
              key={lap}
              x={x(lap) + bandWidth * 0.14}
              y={padTop + plot - h}
              width={Math.max(1.5, bandWidth * 0.72)}
              height={Math.max(h, 1)}
              fill={isActual ? colours.good : isPick ? colours.alert : colours.fuel}
              opacity={isActual || isPick ? 0.95 : 0.5}
              // When our pick and his stop are the same lap the bar can only be
              // one colour, and the coincidence is the whole event. Outline it.
              stroke={isPick && isActual ? colours.alert : 'none'}
              strokeWidth={isPick && isActual ? 4 : 0}
            />
          )
        })}

        {currentLap != null && currentLap >= first && currentLap <= last && (
          <line
            x1={x(currentLap) + bandWidth / 2}
            x2={x(currentLap) + bandWidth / 2}
            y1={padTop - 6}
            y2={padTop + plot}
            stroke={colours.ink}
            strokeWidth={1.5}
            opacity={0.45}
            strokeDasharray="4 4"
          />
        )}

        {actualStop != null && actualStop >= first && actualStop <= last && (
          <>
            <line
              x1={x(actualStop) + bandWidth / 2}
              x2={x(actualStop) + bandWidth / 2}
              y1={padTop - 10}
              y2={padTop + plot}
              stroke={colours.good}
              strokeWidth={2.5}
            />
            <circle cx={x(actualStop) + bandWidth / 2} cy={padTop - 10} r={4} fill={colours.good} />
          </>
        )}

        <line x1={0} x2={W} y1={padTop + plot} y2={padTop + plot} stroke={colours.line} strokeWidth={1.5} />

        {ticks.map((lap) => (
          <text
            key={lap}
            x={x(lap) + bandWidth / 2}
            y={padTop + plot + 22}
            textAnchor="middle"
            fill={colours.inkFaint}
            style={{ fontSize: 15, fontFamily: 'var(--font-mono)' }}
          >
            {lap}
          </text>
        ))}
      </svg>

      <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-ink-faint">
        <Key colour={colours.alert} label={`our lap ${recommendedLap}`} />
        {window && <Key colour={colours.alert} faint label={`window ${window[0]}–${window[1]}`} />}
        {actualStop != null && (
          <Key
            colour={colours.good}
            label={`he boxed lap ${actualStop}${
              probabilityOnActual != null
                ? ` · we had ${(probabilityOnActual * 100).toFixed(1)}% on it`
                : ''
            }`}
          />
        )}
        <span className="ml-auto">probability each lap is the right lap</span>
      </div>
    </div>
  )
}

function Key({ colour, label, faint = false }: { colour: string; label: string; faint?: boolean }) {
  return (
    <span className="flex items-center gap-1.5">
      <span
        className="inline-block h-2 w-2.5"
        style={{ background: colour, opacity: faint ? 0.25 : 1 }}
      />
      {label}
    </span>
  )
}

/**
 * How the call moved as evidence arrived.
 *
 * Every lap carries its own recommendation, computed from the filtered estimate
 * available at that lap — so this is the model changing its mind in public. The
 * horizontal green lines are where the driver actually stopped. Where the
 * orange converges onto green, that is the claim; where it does not, that is on
 * the same chart at the same size, which is the only way the first half means
 * anything.
 */
export function CallTrajectory({
  calls,
  stops,
  currentLap,
  finalLap,
  height = 200,
}: {
  calls: { lap: number; recommended: number | null; confidence: number }[]
  stops: number[]
  currentLap: number
  finalLap: number
  height?: number
}) {
  const colours = useThemeColours()
  const padTop = 10
  const padBottom = 26
  const plot = height - padTop - padBottom
  const x = (lap: number) => (lap / finalLap) * W
  const y = (lap: number) => padTop + plot - (lap / finalLap) * plot

  const answered = calls.filter((c) => c.recommended != null)
  const refused = calls.filter((c) => c.recommended == null)

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${height}`} className="w-full" role="img"
        aria-label="Recommended pit lap at each lap of the race, against the laps the driver actually stopped">
        {stops.map((stop) => (
          <g key={stop}>
            <line x1={0} x2={W} y1={y(stop)} y2={y(stop)} stroke={colours.good}
              strokeWidth={1.5} opacity={0.5} strokeDasharray="6 5" />
            <circle cx={x(stop)} cy={y(stop)} r={5} fill={colours.good} />
          </g>
        ))}

        {answered.map((call) => (
          <circle
            key={call.lap}
            cx={x(call.lap)}
            cy={y(call.recommended as number)}
            r={1.6 + 7 * call.confidence}
            fill={colours.alert}
            opacity={0.7}
          />
        ))}

        {refused.map((call) => (
          <rect key={`none-${call.lap}`} x={x(call.lap) - 1.5} y={padTop + plot - 7}
            width={3} height={7} fill={colours.inkFaint} opacity={0.8} />
        ))}

        <line x1={x(currentLap)} x2={x(currentLap)} y1={padTop} y2={padTop + plot}
          stroke={colours.ink} strokeWidth={1.5} opacity={0.5} />
        <line x1={0} x2={W} y1={padTop + plot} y2={padTop + plot} stroke={colours.line} strokeWidth={1.5} />

        <text x={4} y={padTop + 12} fill={colours.inkFaint}
          style={{ fontSize: 14, fontFamily: 'var(--font-mono)' }}>
          lap {finalLap}
        </text>
        <text x={4} y={padTop + plot - 6} fill={colours.inkFaint}
          style={{ fontSize: 14, fontFamily: 'var(--font-mono)' }}>
          lap 1
        </text>
      </svg>
      <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-ink-faint">
        <Key colour={colours.alert} label="our call that lap · dot size is confidence" />
        <Key colour={colours.good} label="where he actually stopped" />
        <span className="ml-auto">ticks on the floor are laps we refused to answer</span>
      </div>
    </div>
  )
}
