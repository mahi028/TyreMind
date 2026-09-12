/**
 * Charts for the five-question briefing.
 *
 * Every one of them draws a band. That is not a stylistic preference: the
 * product's whole claim is that its intervals are honest, and a line without a
 * band is indistinguishable from a competitor's line. Where a band is absent
 * below, it is because the series *is* a probability and therefore already a
 * statement about spread.
 *
 * These are separate from `components/charts.tsx` for the same reason the
 * primitives are: this screen is read off a projector. Axis labels run at 12-13px
 * rather than 10, lines at 3px rather than 2, and the grids are looser. The
 * shared charts are correct at desk distance and illegible at four metres.
 */

import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { useThemeColours } from '../../lib/theme'
import type { DecompositionRow, PitWindow, ProjectionResult } from '../../lib/api'

/**
 * Axis, tooltip and legend styling shared by the briefing charts.
 *
 * Animation stays off for the same reason it is off in the shared chart set:
 * these are re-initialised with `notMerge` whenever the lap scrubber moves, and
 * an interrupted entry animation leaves a clipped line that looks exactly like
 * a truncated series. The motion on this screen lives in the HTML layer, where
 * it cannot be mistaken for data.
 */
function useAxis() {
  const c = useThemeColours()

  return useMemo(() => {
    const axisLabel = { color: c.inkDim, fontSize: 12 }
    const nameTextStyle = { color: c.inkFaint, fontSize: 11.5 }
    return {
      c,
      base: {
        animation: false,
        textStyle: { fontFamily: 'Archivo, ui-sans-serif, system-ui, sans-serif' },
        tooltip: {
          backgroundColor: c.surface,
          borderColor: c.lineBright,
          textStyle: { color: c.ink, fontSize: 13 },
          extraCssText: 'border-radius:0;',
        },
        legend: {
          top: 0,
          textStyle: { color: c.inkDim, fontSize: 12 },
          itemWidth: 16,
          itemHeight: 3,
          itemGap: 18,
        },
      },
      xAxis: (name: string) => ({
        name,
        nameLocation: 'middle' as const,
        nameGap: 28,
        nameTextStyle,
        axisLabel,
        axisTick: { lineStyle: { color: c.line } },
        axisLine: { lineStyle: { color: c.lineBright } },
        splitLine: { show: false },
      }),
      yAxis: (name: string) => ({
        name,
        nameTextStyle,
        axisLabel,
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: c.raised } },
      }),
    }
  }, [c])
}

/** A hidden series carrying the lower edge, plus the visible shaded thickness. */
function bandSeries(
  name: string,
  lower: (number | null)[],
  thickness: (number | null)[],
  colour: string,
  opacity = 0.16,
) {
  return [
    {
      name: `${name} base`,
      type: 'line' as const,
      stack: `band-${name}`,
      symbol: 'none',
      lineStyle: { opacity: 0 },
      data: lower,
      silent: true,
      tooltip: { show: false },
      legendHoverLink: false,
      z: 1,
    },
    {
      name,
      type: 'line' as const,
      stack: `band-${name}`,
      symbol: 'none',
      lineStyle: { opacity: 0 },
      areaStyle: { color: colour, opacity },
      data: thickness,
      silent: true,
      tooltip: { show: false },
      legendHoverLink: false,
      z: 1,
    },
  ]
}

const seconds = (v: number | null) =>
  v == null || !Number.isFinite(v) ? '—' : `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(3)} s`

// ---------------------------------------------------------------------------
// Question 1 — where is the time going
// ---------------------------------------------------------------------------

/**
 * Every lap of the stint, split into the causes that produced it.
 *
 * `residual` is stacked alongside the four causes in its own colour and legended
 * "Unexplained". Folding it into the others would make the bars add up and would
 * be a lie about how much of the lap the model accounts for; an engineer who
 * later discovers a hidden residual stops believing the rest of the screen, and
 * they would be right to.
 *
 * Mixed-sign terms stack same-sign by default, so the top of the positive stack
 * is not the total. The dashed line is therefore drawn from `observed_delta`
 * directly rather than inferred from the bar heights.
 */
export function ContributionStack({ rows }: { rows: DecompositionRow[] }) {
  const { base, xAxis, yAxis, c } = useAxis()

  const option = useMemo(() => {
    const terms = [
      { key: 'tyre', label: 'Tyre', colour: c.alert, opacity: 1 },
      { key: 'fuel', label: 'Fuel burn-off', colour: c.fuel, opacity: 0.85 },
      { key: 'track', label: 'Track evolution', colour: c.track, opacity: 0.85 },
      { key: 'traffic', label: 'Traffic', colour: c.traffic, opacity: 0.85 },
      { key: 'residual', label: 'Unexplained', colour: c.residual, opacity: 0.9 },
    ] as const

    return {
      ...base,
      legend: { ...base.legend, data: [...terms.map((t) => t.label), 'Observed lap time'] },
      grid: { left: 64, right: 22, top: 36, bottom: 48 },
      xAxis: {
        type: 'category',
        data: rows.map((r) => r.tyre_age),
        ...xAxis('tyre age (laps)'),
        axisLabel: {
          ...xAxis('').axisLabel,
          interval: (i: number) => rows.length <= 16 || i % Math.ceil(rows.length / 14) === 0,
        },
      },
      yAxis: { type: 'value', ...yAxis('seconds vs the first lap of the stint') },
      tooltip: {
        ...base.tooltip,
        trigger: 'axis',
        axisPointer: { type: 'shadow', shadowStyle: { color: c.raised, opacity: 0.5 } },
        valueFormatter: seconds,
      },
      series: [
        ...terms.map((t) => ({
          name: t.label,
          type: 'bar',
          stack: 'terms',
          barCategoryGap: '22%',
          data: rows.map((r) => (r[t.key] as number | undefined) ?? 0),
          itemStyle: {
            color: t.colour,
            opacity: t.opacity,
            // The unexplained band is outlined as well as filled, so it stays
            // findable at a glance in a bar that is mostly other colours.
            borderColor: t.key === 'residual' ? c.inkFaint : 'transparent',
            borderWidth: t.key === 'residual' ? 1 : 0,
          },
        })),
        {
          name: 'Observed lap time',
          type: 'line',
          data: rows.map((r) => r.observed_delta),
          symbol: 'none',
          lineStyle: { color: c.ink, width: 2.4, type: 'dashed' },
          z: 9,
        },
      ],
    }
  }, [rows, base, xAxis, yAxis, c])

  return <ReactECharts option={option} style={{ height: 330 }} notMerge />
}

/**
 * The part of the lap the model cannot account for, against the noise it should
 * be lost in.
 *
 * The shaded band is the combined 95% spread of the four fitted terms. Residual
 * inside the band is a model that has explained the lap to within its own
 * stated precision. Residual outside it is the model admitting something is
 * happening that it has no term for -- which is a finding, not a failure, and
 * is worth more on screen than a tidier chart would be.
 */
export function ResidualBand({ rows }: { rows: DecompositionRow[] }) {
  const { base, xAxis, yAxis, c } = useAxis()

  const option = useMemo(() => {
    const spread = rows.map((r) =>
      Math.sqrt(
        (r.tyre_sd ?? 0) ** 2 +
          (r.fuel_sd ?? 0) ** 2 +
          (r.track_sd ?? 0) ** 2 +
          (r.traffic_sd ?? 0) ** 2,
      ),
    )
    const half = spread.map((s) => 1.96 * s)

    return {
      ...base,
      legend: { ...base.legend, data: ['Unexplained'] },
      grid: { left: 64, right: 22, top: 34, bottom: 48 },
      xAxis: {
        type: 'category',
        data: rows.map((r) => r.tyre_age),
        ...xAxis('tyre age (laps)'),
        axisLabel: {
          ...xAxis('').axisLabel,
          interval: (i: number) => rows.length <= 16 || i % Math.ceil(rows.length / 14) === 0,
        },
      },
      yAxis: { type: 'value', ...yAxis('seconds') },
      tooltip: {
        ...base.tooltip,
        trigger: 'axis',
        valueFormatter: seconds,
      },
      series: [
        ...bandSeries(
          'model noise (95%)',
          half.map((h) => -h),
          half.map((h) => 2 * h),
          c.inkFaint,
          0.14,
        ),
        {
          name: 'Unexplained',
          type: 'line',
          data: rows.map((r) => r.residual),
          symbol: 'none',
          lineStyle: { color: c.residual, width: 3 },
          areaStyle: { color: c.residual, opacity: 0.2 },
          z: 5,
          markLine: {
            symbol: 'none',
            silent: true,
            lineStyle: { color: c.lineBright, width: 1 },
            label: { show: false },
            data: [{ yAxis: 0 }],
          },
        },
      ],
    }
  }, [rows, base, xAxis, yAxis, c])

  return <ReactECharts option={option} style={{ height: 220 }} notMerge />
}

// ---------------------------------------------------------------------------
// Question 2 — how fast is he losing performance
// ---------------------------------------------------------------------------

export interface RateRow {
  session_lap: number
  tyre_age: number
  rate: number
  rate_sd: number
  level: number
  level_sd: number
}

/**
 * The degradation rate as the filter estimates it, lap by lap.
 *
 * These rows come from the *filtered* pass -- each lap's estimate uses only the
 * laps up to and including it, which is what a pit wall actually has. The
 * smoothed pass uses the whole stint and is the better estimate, but plotting it
 * lap by lap would show the model already knowing on lap two what it only
 * learned on lap twenty. It is drawn instead as the flat `reference` line, so
 * the two are visibly different objects.
 *
 * The band is the posterior at that moment. On a long run it collapses as
 * evidence accumulates; on a short one it can widen, because a handful of noisy
 * laps is genuinely less informative than the prior about where the rate sits.
 * Both are shown as they come.
 *
 * @param which `rate` answers "how fast is he losing it"; `level` answers "how
 *   much has he lost so far". They are the same posterior, differentiated.
 */
export function RateRibbon({
  rows,
  which,
  colour,
  reference,
}: {
  rows: RateRow[]
  which: 'rate' | 'level'
  colour: string
  reference?: { value: number; label: string }
}) {
  const { base, xAxis, yAxis, c } = useAxis()

  const option = useMemo(() => {
    const mean = rows.map((r) => (which === 'rate' ? r.rate : r.level))
    const sd = rows.map((r) => (which === 'rate' ? r.rate_sd : r.level_sd))
    const half = sd.map((s) => 1.96 * s)
    const label = which === 'rate' ? 'Degradation rate' : 'Performance lost'

    return {
      ...base,
      legend: { show: false },
      grid: { left: 62, right: 20, top: 18, bottom: 46 },
      xAxis: {
        type: 'category',
        data: rows.map((r) => r.tyre_age),
        ...xAxis('tyre age (laps)'),
        axisLabel: {
          ...xAxis('').axisLabel,
          interval: (i: number) => rows.length <= 16 || i % Math.ceil(rows.length / 12) === 0,
        },
      },
      yAxis: {
        type: 'value',
        ...yAxis(which === 'rate' ? 's/lap' : 'seconds vs fresh'),
        axisLabel: {
          ...yAxis('').axisLabel,
          formatter: (v: number) => v.toFixed(which === 'rate' ? 2 : 1),
        },
      },
      tooltip: {
        ...base.tooltip,
        trigger: 'axis',
        formatter: (params: { dataIndex: number }[]) => {
          const i = params[0]?.dataIndex ?? 0
          return (
            `<b>tyre age ${rows[i].tyre_age}</b> · race lap ${rows[i].session_lap}<br/>` +
            `${label} ${mean[i].toFixed(3)}<br/>` +
            `<span style="color:${c.inkFaint}">95% ${(mean[i] - half[i]).toFixed(3)} — ` +
            `${(mean[i] + half[i]).toFixed(3)}</span>`
          )
        },
      },
      series: [
        ...bandSeries(
          'band',
          mean.map((m, i) => m - half[i]),
          half.map((h) => 2 * h),
          colour,
          0.2,
        ),
        {
          name: label,
          type: 'line',
          data: mean,
          symbol: 'none',
          lineStyle: { color: colour, width: 3 },
          z: 5,
          markLine: reference
            ? {
                symbol: 'none',
                silent: true,
                lineStyle: { color: c.ink, type: 'dashed', width: 1.4 },
                label: {
                  color: c.inkDim,
                  fontSize: 11,
                  formatter: reference.label,
                  position: 'insideStartTop',
                },
                data: [{ yAxis: reference.value }],
              }
            : undefined,
        },
      ],
    }
  }, [rows, which, colour, reference, base, xAxis, yAxis, c])

  return <ReactECharts option={option} style={{ height: 230 }} notMerge />
}

// ---------------------------------------------------------------------------
// Question 3 — what happens after 3 laps, after 15
// ---------------------------------------------------------------------------

/**
 * Forward projection, with the band allowed to open.
 *
 * The widening is the honest part and it is never clamped. A 20-lap projection
 * whose band is the same width as a 1-lap projection is a chart that has been
 * tidied into a falsehood: extrapolating an uncertain *rate* compounds that
 * uncertainty linearly, and the picture should show it.
 *
 * Breach probability rides a second axis, because "how much will he lose" and
 * "will he fall off the cliff" are different questions with different units and
 * the second one is usually the one that decides the stop.
 */
export function ProjectionFan({
  projection,
  colour,
  marks,
}: {
  projection: ProjectionResult
  colour: string
  marks: number[]
}) {
  const { base, xAxis, yAxis, c } = useAxis()

  const option = useMemo(() => {
    const h = projection.horizon
    const half = projection.loss_sd.map((s) => 1.96 * s)
    const life = projection.competitive_life_laps
    const lower = projection.competitive_life_lower
    const upper = projection.competitive_life_upper

    return {
      ...base,
      legend: {
        ...base.legend,
        data: ['Performance lost', 'Chance of breaching the threshold'],
      },
      grid: { left: 64, right: 66, top: 36, bottom: 48 },
      xAxis: {
        type: 'category',
        data: h,
        ...xAxis('laps from now'),
      },
      yAxis: [
        {
          type: 'value',
          ...yAxis('seconds lost'),
          axisLabel: { ...yAxis('').axisLabel, formatter: (v: number) => v.toFixed(1) },
        },
        {
          type: 'value',
          min: 0,
          max: 1,
          ...yAxis(''),
          splitLine: { show: false },
          axisLabel: {
            ...yAxis('').axisLabel,
            color: c.traffic,
            formatter: (v: number) => `${Math.round(v * 100)}%`,
          },
        },
      ],
      tooltip: {
        ...base.tooltip,
        trigger: 'axis',
        formatter: (params: { dataIndex: number }[]) => {
          const i = params[0]?.dataIndex ?? 0
          return (
            `<b>in ${h[i]} lap${h[i] === 1 ? '' : 's'}</b><br/>` +
            `losing ${projection.loss[i].toFixed(2)} s<br/>` +
            `<span style="color:${c.inkFaint}">95% ${(projection.loss[i] - half[i]).toFixed(2)} — ` +
            `${(projection.loss[i] + half[i]).toFixed(2)} s</span><br/>` +
            `<span style="color:${c.traffic}">past the ${projection.threshold_s} s threshold: ` +
            `${Math.round(projection.breach_probability[i] * 100)}%</span>`
          )
        },
      },
      series: [
        ...bandSeries(
          'band',
          projection.loss.map((v, i) => v - half[i]),
          half.map((v) => 2 * v),
          colour,
          0.2,
        ),
        {
          name: 'Chance of breaching the threshold',
          type: 'line',
          yAxisIndex: 1,
          data: projection.breach_probability,
          symbol: 'none',
          lineStyle: { color: c.traffic, width: 2, type: 'dashed' },
          z: 4,
        },
        {
          name: 'Performance lost',
          type: 'line',
          data: projection.loss,
          symbol: 'none',
          lineStyle: { color: colour, width: 3.4 },
          z: 6,
          markLine: {
            symbol: 'none',
            silent: true,
            lineStyle: { color: c.alert, type: 'dotted', width: 1.6 },
            label: {
              color: c.alert,
              fontSize: 11.5,
              formatter: `${projection.threshold_s} s — no longer competitive`,
              position: 'insideEndTop',
            },
            data: [{ yAxis: projection.threshold_s }],
          },
          // The competitive-life estimate is shaded across its own interval
          // rather than drawn as a single lap. On a flat stint that interval is
          // most of the horizon, and saying so is the answer.
          markArea:
            Number.isFinite(lower) && Number.isFinite(upper)
              ? {
                  silent: true,
                  itemStyle: { color: c.alert, opacity: 0.09 },
                  label: {
                    show: true,
                    color: c.inkFaint,
                    fontSize: 11,
                    position: 'insideTop',
                    formatter: `competitive life ${life} laps (${lower}–${upper})`,
                  },
                  data: [
                    [
                      { xAxis: String(Math.max(h[0], lower)) },
                      { xAxis: String(Math.min(h[h.length - 1], upper)) },
                    ],
                  ],
                }
              : undefined,
          markPoint: {
            symbol: 'circle',
            symbolSize: 10,
            itemStyle: { color: c.ink },
            label: { show: true, position: 'top', color: c.ink, fontSize: 12 },
            // Each callout carries its own literal label. A callback formatter
            // reading `p.value` throws here: a markPoint specified by `xAxis`/
            // `yAxis` is a *coordinate* item and has no `value`, so the callback
            // gets undefined and takes the whole chart down at render time.
            data: marks
              .filter((m) => h.includes(m))
              .map((m) => {
                const loss = projection.loss[h.indexOf(m)]
                return {
                  xAxis: String(m),
                  yAxis: loss,
                  label: { formatter: `${loss.toFixed(2)} s` },
                }
              }),
          },
        },
      ],
    }
  }, [projection, colour, marks, base, xAxis, yAxis, c])

  return <ReactECharts option={option} style={{ height: 340 }} notMerge />
}

// ---------------------------------------------------------------------------
// Question 4 — when do I box
// ---------------------------------------------------------------------------

/**
 * The cost of every possible pit lap, and how confident the simulation is in
 * each of them.
 *
 * Two things share this chart because they are the same decision seen twice.
 * The curve says what each lap costs; the bars say how often that lap actually
 * came out fastest across the simulated races. A sharp curve with a flat
 * probability distribution means the *shape* is confident and the *lap* is not,
 * which is the situation the single recommended lap most badly misrepresents.
 *
 * Times are drawn relative to the best available action, including staying out.
 * Absolute race time is a four-digit number whose interesting variation lives in
 * the last two digits.
 */
export function PitSweep({ window: w, colour }: { window: PitWindow; colour: string }) {
  const { base, xAxis, yAxis, c } = useAxis()

  const option = useMemo(() => {
    const sweep = w.sweep
    const bestStop = Math.min(...sweep.map((r) => r.expected_time))
    const baseline = Math.min(bestStop, w.stay_out_expected_time)
    const laps = sweep.map((r) => r.pit_lap)
    const rel = (v: number) => v - baseline

    const peakProbability = Math.max(...sweep.map((r) => r.probability_optimal), 0.02)

    return {
      ...base,
      legend: {
        ...base.legend,
        data: ['Expected cost', 'Chance this lap is the right one'],
      },
      grid: { left: 66, right: 66, top: 36, bottom: 48 },
      xAxis: { type: 'category', data: laps, ...xAxis('box on lap') },
      yAxis: [
        {
          type: 'value',
          ...yAxis('seconds vs the best option'),
          axisLabel: { ...yAxis('').axisLabel, formatter: (v: number) => v.toFixed(0) },
        },
        {
          type: 'value',
          min: 0,
          max: Math.min(1, peakProbability * 2.6),
          ...yAxis(''),
          splitLine: { show: false },
          axisLabel: {
            ...yAxis('').axisLabel,
            color: c.good,
            formatter: (v: number) => `${Math.round(v * 100)}%`,
          },
        },
      ],
      tooltip: {
        ...base.tooltip,
        trigger: 'axis',
        formatter: (params: { dataIndex: number }[]) => {
          const i = params[0]?.dataIndex ?? 0
          const r = sweep[i]
          return (
            `<b>box on lap ${r.pit_lap}</b><br/>` +
            `expected +${rel(r.expected_time).toFixed(2)} s<br/>` +
            `<span style="color:${c.inkFaint}">range +${rel(r.best_case).toFixed(2)} — ` +
            `+${rel(r.downside).toFixed(2)} s</span><br/>` +
            `<span style="color:${c.good}">fastest in ${(r.probability_optimal * 100).toFixed(1)}% ` +
            `of ${w.n_sims} simulated races</span>`
          )
        },
      },
      series: [
        {
          name: 'Chance this lap is the right one',
          type: 'bar',
          yAxisIndex: 1,
          data: sweep.map((r) => r.probability_optimal),
          itemStyle: { color: c.good, opacity: 0.45 },
          barCategoryGap: '35%',
          z: 2,
        },
        ...bandSeries(
          'band',
          sweep.map((r) => rel(r.best_case)),
          sweep.map((r) => rel(r.downside) - rel(r.best_case)),
          colour,
          0.16,
        ),
        {
          name: 'Expected cost',
          type: 'line',
          data: sweep.map((r) => rel(r.expected_time)),
          symbol: 'none',
          smooth: 0.2,
          lineStyle: { color: colour, width: 3.4 },
          z: 6,
          markArea: w.window_within_1s
            ? {
                silent: true,
                itemStyle: { color: c.good, opacity: 0.1 },
                label: {
                  show: true,
                  color: c.good,
                  fontSize: 11.5,
                  position: 'insideTop',
                  formatter: `window ${w.window_within_1s[0]}–${w.window_within_1s[1]}`,
                },
                data: [
                  [
                    { xAxis: String(w.window_within_1s[0]) },
                    { xAxis: String(w.window_within_1s[1]) },
                  ],
                ],
              }
            : undefined,
          markPoint: {
            symbol: 'circle',
            symbolSize: 12,
            itemStyle: { color: c.good },
            label: {
              show: true,
              position: 'bottom',
              color: c.good,
              fontSize: 12,
              formatter: `lap ${w.optimum_lap}`,
            },
            data: [{ xAxis: String(w.optimum_lap), yAxis: rel(w.optimum_expected_time) }],
          },
          markLine: {
            symbol: 'none',
            silent: true,
            lineStyle: { color: c.inkFaint, type: 'dotted', width: 1.4 },
            label: {
              color: c.inkFaint,
              fontSize: 11.5,
              formatter: 'stay out — never stop again',
              position: 'insideEndTop',
            },
            data: [{ yAxis: rel(w.stay_out_expected_time) }],
          },
        },
      ],
    }
  }, [w, colour, base, xAxis, yAxis, c])

  return <ReactECharts option={option} style={{ height: 340 }} notMerge />
}

// ---------------------------------------------------------------------------
// Question 5 — can I trust it
// ---------------------------------------------------------------------------

/**
 * Four independent methods, each with its own posterior, against the consensus.
 *
 * The argument this makes is not "the number is right" -- nothing on a screen
 * can make that argument. It is narrower and checkable: perturb the two priors
 * the identification argument leans on, and the answer moves by less than its
 * own uncertainty. When it does not, `disagreement_flagged` comes back true and
 * the panel says so rather than averaging the disagreement away.
 */
export function ConsensusSpread({
  estimates,
  consensus,
  consensusSd,
  colour,
}: {
  estimates: Record<string, { mean: number; sd: number }>
  consensus: number
  consensusSd: number
  colour: string
}) {
  const { base, xAxis, c } = useAxis()

  const option = useMemo(() => {
    const names = Object.keys(estimates)
    const rows = names.map((n) => estimates[n])
    const all = [
      ...rows.flatMap((r) => [r.mean - 1.96 * r.sd, r.mean + 1.96 * r.sd]),
      consensus - 1.96 * consensusSd,
      consensus + 1.96 * consensusSd,
    ]
    const pad = (Math.max(...all) - Math.min(...all)) * 0.08 || 0.01
    const lo = Math.min(...all) - pad
    const hi = Math.max(...all) + pad

    const categories = [...names, 'consensus']

    return {
      ...base,
      legend: { show: false },
      grid: { left: 132, right: 30, top: 14, bottom: 48 },
      xAxis: {
        type: 'value',
        min: lo,
        max: hi,
        ...xAxis('degradation rate (s/lap)'),
        axisLabel: { ...xAxis('').axisLabel, formatter: (v: number) => v.toFixed(3) },
      },
      yAxis: {
        type: 'category',
        data: categories,
        axisLabel: {
          color: c.inkDim,
          fontSize: 12,
          formatter: (v: string) => (v === 'consensus' ? '{strong|consensus}' : v),
          rich: { strong: { color: c.ink, fontSize: 12.5, fontWeight: 600 } },
        },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { show: false },
      },
      tooltip: {
        ...base.tooltip,
        formatter: (p: { dataIndex: number }) => {
          const i = p.dataIndex
          const r = i < rows.length ? rows[i] : { mean: consensus, sd: consensusSd }
          return (
            `<b>${categories[i]}</b><br/>${r.mean.toFixed(4)} s/lap<br/>` +
            `<span style="color:${c.inkFaint}">95% ${(r.mean - 1.96 * r.sd).toFixed(4)} — ` +
            `${(r.mean + 1.96 * r.sd).toFixed(4)}</span>`
          )
        },
      },
      series: [
        {
          // One horizontal 95% interval per method, with a tick at the mean.
          type: 'custom',
          // Literal styles rather than `api.style()`, which ECharts 6 deprecates
          // and warns about on every render.
          renderItem: (
            params: { dataIndex: number },
            api: {
              value: (i: number) => number
              coord: (p: [number, number]) => [number, number]
            },
          ) => {
            const isConsensus = params.dataIndex === rows.length
            const y = api.coord([api.value(0), params.dataIndex])[1]
            const [x1] = api.coord([api.value(1), params.dataIndex])
            const [x2] = api.coord([api.value(2), params.dataIndex])
            const [xm] = api.coord([api.value(0), params.dataIndex])
            const tone = isConsensus ? colour : c.inkDim
            const thickness = isConsensus ? 14 : 8

            return {
              type: 'group',
              children: [
                {
                  // The 95% interval.
                  type: 'rect',
                  shape: { x: x1, y: y - thickness / 2, width: x2 - x1, height: thickness },
                  style: { fill: tone, opacity: isConsensus ? 0.3 : 0.2 },
                },
                {
                  // The mean, ticked through it.
                  type: 'rect',
                  shape: { x: xm - 1.5, y: y - thickness / 2 - 3, width: 3, height: thickness + 6 },
                  style: { fill: tone, opacity: 1 },
                },
              ],
            }
          },
          encode: { x: [0, 1, 2] },
          data: [
            ...rows.map((r) => [r.mean, r.mean - 1.96 * r.sd, r.mean + 1.96 * r.sd]),
            [consensus, consensus - 1.96 * consensusSd, consensus + 1.96 * consensusSd],
          ],
        },
      ],
    }
  }, [estimates, consensus, consensusSd, colour, base, xAxis, c])

  return (
    <ReactECharts
      option={option}
      style={{ height: 40 + (Object.keys(estimates).length + 1) * 40 }}
      notMerge
    />
  )
}
