/**
 * Live session monitor.
 *
 * Streams through the online estimator over WebSocket, one lap at a time, with
 * no access to future data. The per-update timing is shown because the real-time
 * claim should be visible, not captioned.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { LiveState } from '../lib/api'
import { compoundColour, fixed, signed } from '../lib/api'
import { Beam, CompoundChip, Panel, Stat } from './primitives'
import { Explainer } from './Explainer'
import ReactECharts from 'echarts-for-react'
import { useThemeColours } from '../lib/theme'

interface Frame {
  type: string
  index?: number
  observation?: {
    driver: string
    session_lap: number
    lap_time: number
    compound: string
    tyre_age: number
    traffic_index: number
  }
  state?: LiveState
  compound_rates?: Record<string, { mean: number; sd: number }>
  total_laps?: number
  performance?: {
    total_laps_processed: number
    n_states: number
    mean_update_ms: number
    p95_update_ms: number
    max_update_ms: number
  }
  detail?: string
}

type Status = 'idle' | 'connecting' | 'streaming' | 'complete' | 'error'

export function LiveMonitor({ sessionId }: { sessionId: string }) {
  const [status, setStatus] = useState<Status>('idle')
  const [states, setStates] = useState<Record<string, LiveState>>({})
  const [rates, setRates] = useState<Record<string, { mean: number; sd: number }>>({})
  const [progress, setProgress] = useState({ done: 0, total: 0 })
  const [feed, setFeed] = useState<Frame[]>([])
  const [perf, setPerf] = useState<Frame['performance']>()
  const [history, setHistory] = useState<{ lap: number; sd: number; rate: number }[]>([])
  const [error, setError] = useState('')
  const [speed, setSpeed] = useState(0.05)
  const socketRef = useRef<WebSocket | null>(null)

  const stop = useCallback(() => {
    socketRef.current?.close()
    socketRef.current = null
  }, [])

  useEffect(() => stop, [stop])
  useEffect(() => {
    stop()
    setStatus('idle')
    setStates({})
    setRates({})
    setFeed([])
    setPerf(undefined)
    setHistory([])
    setProgress({ done: 0, total: 0 })
  }, [sessionId, stop])

  const start = useCallback(() => {
    stop()
    setStatus('connecting')
    setStates({})
    setFeed([])
    setPerf(undefined)
    setHistory([])
    setError('')

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const socket = new WebSocket(
      `${protocol}//${location.host}/ws/replay/${sessionId}?speed=${speed}`,
    )
    socketRef.current = socket

    socket.onmessage = (event) => {
      const frame: Frame = JSON.parse(event.data)
      if (frame.type === 'start') {
        setStatus('streaming')
        setProgress({ done: 0, total: frame.total_laps ?? 0 })
      } else if (frame.type === 'lap' && frame.state) {
        setStates((prev) => ({ ...prev, [frame.state!.driver]: frame.state! }))
        if (frame.compound_rates) setRates(frame.compound_rates)
        setProgress((p) => ({ ...p, done: (frame.index ?? 0) + 1 }))
        setFeed((prev) => [frame, ...prev].slice(0, 9))
        const index = frame.index ?? 0
        if (index % 4 === 0 && frame.compound_rates) {
          const entries = Object.values(frame.compound_rates)
          if (entries.length) {
            setHistory((prev) => [
              ...prev,
              {
                lap: frame.state!.session_lap,
                sd: entries.reduce((a, e) => a + e.sd, 0) / entries.length,
                rate: entries.reduce((a, e) => a + e.mean, 0) / entries.length,
              },
            ])
          }
        }
      } else if (frame.type === 'complete') {
        setStatus('complete')
        setPerf(frame.performance)
      } else if (frame.type === 'error') {
        setStatus('error')
        setError(frame.detail ?? 'stream failed')
      }
    }
    socket.onerror = () => {
      setStatus('error')
      setError('WebSocket connection failed. Is the API running?')
    }
    socket.onclose = () => setStatus((s) => (s === 'streaming' ? 'complete' : s))
  }, [sessionId, speed, stop])

  const ordered = Object.values(states).sort((a, b) => b.degradation_rate - a.degradation_rate)
  const pct = progress.total ? (progress.done / progress.total) * 100 : 0
  const isStreaming = status === 'streaming'

  return (
    <div className="space-y-4">
      <Explainer id="live" question="What does 'live' actually mean here?">
        <p>
          Laps are fed to the model one at a time, in order, with{' '}
          <strong>no access to anything that happens later</strong>, exactly like a
          pit wall during a session. Watch the intervals narrow as laps arrive: the
          model knows almost nothing early on and says so, then tightens up as
          evidence accumulates.
        </p>
      </Explainer>

      {/* Control panel */}
      <Panel
        title="Live session monitor"
        aside={
          <span className="flex items-center gap-1.5">
            <span
              className={`h-1.5 w-1.5 rounded-full ${isStreaming ? 'pulse-live' : ''}`}
              style={{
                background: isStreaming
                  ? 'var(--color-good)'
                  : status === 'complete'
                    ? 'var(--color-fuel)'
                    : 'var(--color-ink-ghost)',
              }}
            />
            <span className="text-[10.5px]">
              {isStreaming
                ? `lap ${progress.done} of ${progress.total}`
                : status === 'complete'
                  ? 'session complete'
                  : status === 'connecting'
                    ? 'connecting…'
                    : 'not running'}
            </span>
          </span>
        }
      >
        <div className="mb-4 flex flex-wrap items-center gap-3">
          <button
            onClick={isStreaming ? stop : start}
            className={`rounded-pill border px-3.5 py-1.5 text-[12px] font-medium transition-colors duration-150 ${
              isStreaming
                ? 'border-line text-ink-dim hover:border-line-bright hover:text-ink'
                : 'border-alert text-alert hover:bg-alert/10'
            }`}
          >
            {isStreaming ? 'Stop stream' : 'Start live replay'}
          </button>

          <label className="flex items-center gap-2 text-[11px] text-ink-faint">
            Pace
            <input
              type="range"
              min={0}
              max={0.3}
              step={0.01}
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              disabled={isStreaming}
              className="w-28"
            />
            <span className="num w-12 text-ink-dim">
              {speed === 0 ? 'max' : `${(speed * 1000).toFixed(0)} ms`}
            </span>
          </label>
        </div>

        {/* Progress bar */}
        <div className="h-0.5 w-full overflow-hidden rounded-pill bg-raised">
          <div
            className="h-full rounded-pill transition-[width] duration-200"
            style={{
              width: `${pct}%`,
              background: 'var(--color-alert)',
            }}
          />
        </div>

        {error && (
          <p className="mt-3 text-[12px]" style={{ color: 'var(--color-danger)' }}>
            {error}
          </p>
        )}

        {perf && (
          <div className="mt-4 grid grid-cols-2 gap-5 border-t border-line pt-4 sm:grid-cols-4">
            <Stat label="Laps processed"  value={String(perf.total_laps_processed)} />
            <Stat label="Model states"    value={String(perf.n_states)} />
            <Stat label="Mean update"     value={perf.mean_update_ms.toFixed(2)} unit="ms" tone="warm" />
            <Stat label="Worst update"    value={perf.max_update_ms.toFixed(2)}  unit="ms" />
          </div>
        )}
      </Panel>

      <div className="grid gap-4 lg:grid-cols-[1.35fr_1fr]">
        {/* State table */}
        <Panel title="Tyre state, as known right now" aside="filtered, no future data">
          {ordered.length === 0 ? (
            <div className="py-8 text-center text-[12px] text-ink-faint">
              Start the replay to watch the estimator build its picture lap by lap.
            </div>
          ) : (
            <div className="space-y-px">
              {/* Column headers */}
              <div
                className="grid items-center gap-2 pb-1.5"
                style={{ gridTemplateColumns: '54px 74px 44px 1fr 72px' }}
              >
                {['Car', 'Compound', 'Age', 'Degradation rate (s/lap)', 'Health'].map((h) => (
                  <span key={h} className="label-caps">{h}</span>
                ))}
              </div>
              {ordered.map((state) => (
                <div
                  key={state.driver}
                  className="grid items-center gap-2 border-t border-line-subtle py-1.5"
                  style={{ gridTemplateColumns: '54px 74px 44px 1fr 72px' }}
                >
                  <span className="num text-[12.5px] font-semibold text-ink">
                    {state.driver}
                  </span>
                  <CompoundChip compound={state.compound} />
                  <span className="num text-right text-[12px] text-ink-dim">
                    {state.tyre_age.toFixed(0)}
                  </span>
                  <div className="flex items-center gap-2">
                    <Beam
                      mean={state.degradation_rate}
                      sd={state.degradation_rate_sd}
                      domain={[-0.05, 0.3]}
                      colour={compoundColour(state.compound)}
                      zero
                      height={12}
                    />
                    <span className="num w-14 shrink-0 text-right text-[12px] text-ink">
                      {fixed(state.degradation_rate)}
                    </span>
                  </div>
                  <span
                    className="num text-right text-[12.5px] font-medium"
                    style={{
                      color:
                        state.health_index > 60
                          ? 'var(--color-good)'
                          : state.health_index > 25
                            ? 'var(--color-warn)'
                            : 'var(--color-danger)',
                    }}
                  >
                    {state.health_index.toFixed(0)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </Panel>

        <div className="space-y-4">
          {/* Compound baselines */}
          <Panel title="Compound baselines" aside="pooled across the field">
            {Object.keys(rates).length === 0 ? (
              <div className="py-4 text-[12px] text-ink-faint">Waiting for laps.</div>
            ) : (
              <div className="space-y-3">
                {Object.entries(rates).map(([compound, r]) => (
                  <div key={compound}>
                    <div className="mb-1 flex items-center justify-between">
                      <CompoundChip compound={compound} />
                      <span className="num text-[12.5px] text-ink">
                        {fixed(r.mean)}
                        <span className="ml-1 text-[10px] text-ink-faint">
                          ± {r.sd.toFixed(3)}
                        </span>
                      </span>
                    </div>
                    <Beam
                      mean={r.mean}
                      sd={r.sd}
                      domain={[-0.05, 0.25]}
                      colour={compoundColour(compound)}
                      zero
                      height={10}
                    />
                  </div>
                ))}
              </div>
            )}
          </Panel>

          {/* Convergence chart */}
          <Panel title="Confidence, as laps arrive" aside="uncertainty collapsing">
            <ConvergenceChart history={history} />
          </Panel>

          {/* Incoming feed */}
          <Panel title="Incoming laps">
            {feed.length === 0 ? (
              <div className="py-4 text-[12px] text-ink-faint">No laps yet.</div>
            ) : (
              <div className="space-y-px">
                {feed.map((frame, i) => (
                  <div
                    key={`${frame.index}-${i}`}
                    className="flex items-center justify-between border-t border-line-subtle py-1 text-[11.5px]"
                    style={{ opacity: 1 - i * 0.09 }}
                  >
                    <span className="num text-ink-dim">
                      L{frame.observation?.session_lap}{' '}
                      <span className="text-ink">{frame.observation?.driver}</span>
                    </span>
                    <span className="num text-ink-dim">
                      {frame.observation?.lap_time.toFixed(3)}s
                    </span>
                    <span
                      className="num text-[11px]"
                      style={{
                        color:
                          Math.abs(frame.state?.innovation_z ?? 0) > 2.5
                            ? 'var(--color-danger)'
                            : 'var(--color-ink-faint)',
                      }}
                      title="prediction error in standard deviations"
                    >
                      z {signed(frame.state?.innovation_z ?? 0, 1)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </Panel>
        </div>
      </div>
    </div>
  )
}

function ConvergenceChart({
  history,
}: {
  history: { lap: number; sd: number; rate: number }[]
}) {
  const c = useThemeColours()

  if (history.length < 3) {
    return (
      <div className="py-8 text-center text-[12px] text-ink-faint">
        Start the replay to watch the model become confident.
      </div>
    )
  }

  const option = {
    animation: false,
    grid: { left: 44, right: 44, top: 24, bottom: 32 },
    xAxis: {
      type: 'category',
      data: history.map((h) => h.lap),
      name: 'session lap',
      nameLocation: 'middle',
      nameGap: 20,
      nameTextStyle: { color: c.inkFaint, fontSize: 10, fontFamily: 'Inter, sans-serif' },
      axisLine: { lineStyle: { color: c.line } },
      axisLabel: { color: c.inkFaint, fontSize: 10, fontFamily: 'Inter, sans-serif' },
    },
    yAxis: [
      {
        type: 'value',
        name: 'rate',
        nameTextStyle: { color: c.inkFaint, fontSize: 10, fontFamily: 'Inter, sans-serif' },
        axisLine: { show: false },
        axisLabel: {
          color: c.inkFaint,
          fontSize: 10,
          fontFamily: 'Inter, sans-serif',
          formatter: (v: number) => v.toFixed(2),
        },
        splitLine: { lineStyle: { color: c.line, type: 'dashed' as const } },
      },
      {
        type: 'value',
        name: '± sd',
        nameTextStyle: { color: c.inkFaint, fontSize: 10, fontFamily: 'Inter, sans-serif' },
        axisLine: { show: false },
        axisLabel: {
          color: c.inkFaint,
          fontSize: 10,
          fontFamily: 'Inter, sans-serif',
          formatter: (v: number) => v.toFixed(3),
        },
        splitLine: { show: false },
      },
    ],
    tooltip: {
      trigger: 'axis',
      backgroundColor: c.surface,
      borderColor: c.line,
      borderWidth: 1,
      borderRadius: 4,
      padding: [8, 12],
      textStyle: { color: c.ink, fontSize: 11, fontFamily: 'Inter, sans-serif' },
    },
    series: [
      {
        name: 'Degradation estimate',
        type: 'line',
        data: history.map((h) => h.rate),
        symbol: 'none',
        smooth: 0.3,
        lineStyle: { color: c.alert, width: 2 },
      },
      {
        name: 'Uncertainty',
        type: 'line',
        yAxisIndex: 1,
        data: history.map((h) => h.sd),
        symbol: 'none',
        smooth: 0.3,
        lineStyle: { color: c.fuel, width: 1.4, type: 'dashed' as const },
        areaStyle: { color: c.fuel, opacity: 0.1 },
      },
    ],
  }

  const first = history[0].sd
  const last  = history[history.length - 1].sd

  return (
    <>
      <ReactECharts option={option} style={{ height: 190 }} notMerge />
      <p className="mt-2 max-w-[46ch] text-[11px] leading-relaxed text-ink-faint">
        Uncertainty fell from ±{first.toFixed(3)} to ±{last.toFixed(3)} s/lap over{' '}
        {history[history.length - 1].lap - history[0].lap} laps: evidence accumulating,
        not the model changing its mind.
      </p>
    </>
  )
}
