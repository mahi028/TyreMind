/**
 * The hero chart: confounders lifted off a stint one at a time.
 *
 * The product's whole claim is that observed pace ≠ tyre degradation. The most
 * direct way to make that land is to show it happening — start from lap times,
 * remove fuel, then track, then traffic, and let the tyre curve be what remains.
 */

import { useEffect, useMemo, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import type { DecompositionRow } from '../lib/api'
import { compoundColourResolved } from '../lib/api'
import { useThemeColours } from '../lib/theme'

const STEPS = [
  { key: 'none',    label: 'Lap times as driven',     removes: null as string | null },
  { key: 'fuel',    label: 'Fuel removed',             removes: 'fuel'               },
  { key: 'track',   label: 'Track evolution removed',  removes: 'track'              },
  { key: 'traffic', label: 'Traffic removed',          removes: 'traffic'            },
] as const

export function PeelAway({
  rows,
  compound,
  autoPlay = true,
}: {
  rows: DecompositionRow[]
  compound: string
  autoPlay?: boolean
}) {
  const [step, setStep] = useState(0)

  useEffect(() => {
    if (!autoPlay || step >= STEPS.length - 1) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const timer = setTimeout(() => setStep((s) => s + 1), reduced ? 400 : 1600)
    return () => clearTimeout(timer)
  }, [step, autoPlay])

  const series = useMemo(() => {
    const removed = STEPS.slice(1, step + 1)
      .map((s) => s.removes)
      .filter(Boolean) as string[]
    return rows.map((row) => {
      let value = row.observed_delta
      for (const term of removed) {
        value -= row[term as 'fuel' | 'track' | 'traffic'] ?? 0
      }
      return [row.tyre_age, value]
    })
  }, [rows, step])

  const tyreOnly = useMemo(
    () => rows.map((row) => [row.tyre_age, row.tyre ?? 0]),
    [rows],
  )

  const colour = compoundColourResolved(compound)
  const atEnd  = step === STEPS.length - 1
  const c      = useThemeColours()

  const option = {
    animation: true,
    animationDuration: 0,
    animationDurationUpdate: 600,
    animationEasingUpdate: 'cubicInOut',
    grid: { left: 48, right: 16, top: 20, bottom: 36 },
    xAxis: {
      type: 'value',
      name: 'tyre age (laps)',
      nameLocation: 'middle',
      nameGap: 24,
      nameTextStyle: { color: c.inkFaint, fontSize: 10, fontFamily: 'Inter, sans-serif' },
      axisLine: { lineStyle: { color: c.line } },
      axisLabel: { color: c.inkFaint, fontSize: 10.5, fontFamily: 'Inter, sans-serif' },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      name: 'seconds vs stint start',
      nameLocation: 'middle',
      nameGap: 36,
      nameTextStyle: { color: c.inkFaint, fontSize: 10, fontFamily: 'Inter, sans-serif' },
      axisLine: { show: false },
      axisLabel: {
        color: c.inkFaint,
        fontSize: 10.5,
        fontFamily: 'Inter, sans-serif',
        formatter: (v: number) => v.toFixed(1),
      },
      splitLine: { lineStyle: { color: c.line, type: 'dashed' as const, dashOffset: 3 } },
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: c.surface,
      borderColor: c.line,
      borderWidth: 1,
      borderRadius: 4,
      padding: [8, 12],
      textStyle: { color: c.ink, fontSize: 11.5, fontFamily: 'Inter, sans-serif' },
      valueFormatter: (v: number) =>
        `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(3)} s`,
    },
    series: [
      {
        id: 'tyre',
        name: 'True tyre degradation',
        type: 'line',
        data: tyreOnly,
        smooth: 0.25,
        symbol: 'none',
        lineStyle: { color: colour, width: 2, type: 'dashed' as const },
        z: 3,
      },
      {
        id: 'peel',
        name: STEPS[step].label,
        type: 'line',
        data: series,
        smooth: 0.2,
        symbol: 'circle',
        symbolSize: 3.5,
        itemStyle: { color: atEnd ? colour : c.inkDim },
        lineStyle: { color: atEnd ? colour : c.inkDim, width: 2 },
        z: 4,
      },
    ],
  }

  return (
    <div>
      {/* Step buttons */}
      <div className="mb-3 flex flex-wrap items-center gap-1.5">
        {STEPS.map((s, i) => (
          <button
            key={s.key}
            onClick={() => setStep(i)}
            className={`
              rounded-pill border px-2.5 py-1 text-[11px] transition-colors duration-150
              ${i === step
                ? 'border-alert/60 bg-alert-dim text-ink'
                : i < step
                  ? 'border-line text-ink-dim hover:border-line-bright hover:text-ink'
                  : 'border-line text-ink-faint hover:border-line-bright hover:text-ink-dim'
              }
            `}
          >
            {s.label}
          </button>
        ))}
      </div>

      <ReactECharts option={option} style={{ height: 300 }} />

      <p className="mt-3 max-w-[68ch] text-[12px] leading-relaxed text-ink-dim">
        {atEnd ? (
          <>
            With fuel, track evolution and traffic removed, the solid line is the
            tyre; the dashed line is the model&rsquo;s degradation estimate, and the
            gap between them is lap-to-lap driver noise.
          </>
        ) : (
          <>
            The grey line is what the stopwatch saw, not the tyre yet: each step
            removes one cause that changed lap time without the tyre changing at
            all.
          </>
        )}
      </p>
    </div>
  )
}
