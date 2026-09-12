/**
 * The circuit, the car, and the split the stopwatch cannot show you.
 *
 * This is the screen the whole product exists to justify. A stint's lap times
 * frequently get FASTER while the tyre underneath is getting worse, because
 * fuel burn-off is worth about 0.081 s/lap and degradation is worth 0.03-0.08.
 * The confounder is the same size as the signal, so reading pace alone gets the
 * sign wrong. On the Monza race stint below that happens on 31 of 41 laps.
 *
 * Two deliberate restraints:
 *
 *   - The track is not coloured by the model. The estimator produces a state
 *     per lap, not per metre, and painting the tarmac with it would claim a
 *     spatial resolution that does not exist.
 *   - The car's position within a lap is animation, not measurement. The lap it
 *     is on is data; where it sits between two corners at any instant is not,
 *     and nothing is read off it.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  api,
  compoundColour,
  compoundColourResolved,
  signed,
  type DecompositionRow,
  type RunRow,
} from '../lib/api'
import {
  geometrySlug,
  loadTrackGeometry,
  toWorldFrame,
  type TrackFrame,
  type TrackGeometryFile,
} from '../lib/geometry'
import { RaceTrack3D } from './RaceTrack3D'
import { Beam, CompoundChip, ErrorNote, Panel, RunChip } from './primitives'
import { Explainer } from './Explainer'

/** Seconds of wall clock per simulated lap at 1x. */
const LAP_SECONDS = 4

export function RaceView({
  sessionId,
  circuit,
  runs,
  selected,
  onSelect,
}: {
  sessionId: string
  circuit: string
  runs: RunRow[]
  selected: RunRow | null
  onSelect: (r: RunRow) => void
}) {
  const [geometry, setGeometry] = useState<TrackGeometryFile | null>(null)
  const [geometryError, setGeometryError] = useState('')
  const [rows, setRows] = useState<DecompositionRow[]>([])
  const [fitting, setFitting] = useState(false)
  const [error, setError] = useState('')

  const [lapIndex, setLapIndex] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [rotating, setRotating] = useState(true)
  const [progress, setProgress] = useState(0)

  const slug = geometrySlug(circuit)

  useEffect(() => {
    if (!slug) {
      setGeometry(null)
      setGeometryError(`No exported geometry for ${circuit}.`)
      return
    }
    setGeometryError('')
    loadTrackGeometry(slug)
      .then(setGeometry)
      .catch((e) => setGeometryError(String(e.message ?? e)))
  }, [slug, circuit])

  useEffect(() => {
    if (!selected) return
    setFitting(true)
    setError('')
    setRows([])
    setLapIndex(0)
    api
      .decomposeRun(sessionId, selected.driver, selected.run_id)
      .then((r) => setRows(r.rows))
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setFitting(false))
  }, [sessionId, selected])

  // One clock drives both the car's trip round the lap and the advance through
  // the stint, so the lap counter and the animation cannot drift apart.
  const lapCount = rows.length
  const rafRef = useRef(0)
  useEffect(() => {
    if (!playing || lapCount === 0) return
    let last = performance.now()
    const tick = (now: number) => {
      const delta = (now - last) / 1000
      last = now
      setProgress((p) => {
        const next = p + delta / LAP_SECONDS
        if (next >= 1) setLapIndex((i) => (i + 1) % lapCount)
        return next % 1
      })
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [playing, lapCount])

  const frame: TrackFrame | null = useMemo(
    () => (geometry ? toWorldFrame(geometry) : null),
    [geometry],
  )

  const row = rows[Math.min(lapIndex, Math.max(rows.length - 1, 0))] ?? null
  const scrub = useCallback((value: number) => {
    setPlaying(false)
    setLapIndex(value)
  }, [])

  if (error) return <ErrorNote error={error} />

  return (
    <div className="space-y-4">
      <Explainer id="race" question="What is this screen showing?">
        <p>
          The real racing line, with its real elevation, and one stint being
          driven round it. Beside it: what the stopwatch recorded on that lap,
          and what the tyre actually did underneath it. They frequently
          disagree, and that disagreement is the product.
        </p>
      </Explainer>

      <div className="grid gap-4 xl:grid-cols-[1.45fr_1fr]">
        <Panel
          title={geometry ? `${geometry.circuit}, ${geometry.event}` : circuit}
          aside={
            geometry
              ? `${(geometry.lap_length_m / 1000).toFixed(2)} km · ${geometry.elevation_gain_m.toFixed(1)} m climbed`
              : undefined
          }
        >
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <button
              onClick={() => setPlaying((p) => !p)}
              className="rounded-pill border border-alert bg-alert-dim px-3 py-1 text-[11.5px] font-medium text-alert transition-colors duration-150 hover:bg-alert/10"
            >
              {playing ? 'Pause' : 'Play'}
            </button>
            <button
              onClick={() => setRotating((r) => !r)}
              className="rounded-pill border border-line px-3 py-1 text-[11.5px] text-ink-dim transition-colors duration-150 hover:border-line-bright hover:text-ink"
            >
              {rotating ? 'Stop camera' : 'Orbit camera'}
            </button>
            {row && (
              <span className="num ml-auto text-[11.5px] text-ink-dim">
                lap {row.session_lap} · tyre age {row.tyre_age.toFixed(0)}
              </span>
            )}
          </div>

          <div className="h-[420px] w-full overflow-hidden rounded-md border border-line">
            {frame ? (
              <RaceTrack3D
                frame={frame}
                progress={progress}
                carColour={compoundColourResolved(selected?.compound ?? '')}
                rotating={rotating}
              />
            ) : (
              <div className="flex h-full items-center justify-center px-6 text-center text-[12px] text-ink-faint">
                {geometryError ||
                  'Loading the racing line…'}
              </div>
            )}
          </div>

          {geometry && (
            <p className="mt-2 text-[10.5px] leading-relaxed text-ink-faint">
              Fastest lap of {geometry.session_id}, resampled to{' '}
              {geometry.n_points} points. Elevation is real, from the positioning
              feed, drawn {frame?.elevationExaggeration}x taller than life so a{' '}
              {geometry.elevation_gain_m.toFixed(1)} m climb over{' '}
              {(geometry.lap_length_m / 1000).toFixed(2)} km is visible at all.
              This is the racing line, not the kerbs.
            </p>
          )}

          {rows.length > 1 && (
            <div className="mt-3">
              <input
                type="range"
                min={0}
                max={rows.length - 1}
                value={Math.min(lapIndex, rows.length - 1)}
                onChange={(e) => scrub(Number(e.target.value))}
                className="w-full"
                aria-label="Scrub through the stint"
              />
              <div className="mt-0.5 flex justify-between text-[10px] text-ink-faint">
                <span>lap {rows[0].session_lap}</span>
                <span>lap {rows[rows.length - 1].session_lap}</span>
              </div>
            </div>
          )}
        </Panel>

        <div className="space-y-4">
          <Panel title="Pick a stint" aside={`${runs.length} in this session`}>
            <div className="flex flex-wrap gap-1.5">
              {runs.slice(0, 14).map((run) => (
                <RunChip
                  key={run.run_id}
                  run={run}
                  selected={selected?.run_id === run.run_id}
                  onSelect={() => onSelect(run)}
                />
              ))}
            </div>
          </Panel>

          {fitting ? (
            <Panel title="This lap" aside="fitting">
              <FittingNote />
            </Panel>
          ) : row ? (
            <LapReadout row={row} />
          ) : null}
        </div>
      </div>

      {rows.length > 0 && <StintSignSummary rows={rows} />}
    </div>
  )
}

/**
 * The model takes seconds on a cold cache. Saying what it is doing beats a
 * spinner: the wait is the fit, and the fit is the product.
 */
function FittingNote() {
  return (
    <div className="flex items-center gap-3 py-6">
      <span
        className="ring-spin inline-block h-5 w-5 rounded-full border-2 border-line"
        style={{ borderTopColor: 'var(--color-alert)' }}
        aria-hidden
      />
      <div>
        <div className="text-[12.5px] text-ink">Separating the tyre from its confounders</div>
        <div className="text-[11px] text-ink-faint">
          Kalman filter and RTS smoother over the whole field, about 12 s per session
        </div>
      </div>
    </div>
  )
}

/** One lap, taken apart, with every term carrying its interval. */
function LapReadout({ row }: { row: DecompositionRow }) {
  const stopwatchFaster = row.observed_delta < 0
  const tyreWorse = (row.tyre ?? 0) > 0.05
  const disagree = stopwatchFaster && tyreWorse

  const terms = [
    { key: 'tyre', label: 'Tyre', value: row.tyre, sd: row.tyre_sd, colour: compoundColour(row.compound) },
    { key: 'fuel', label: 'Fuel burn-off', value: row.fuel, sd: row.fuel_sd, colour: 'var(--color-fuel)' },
    { key: 'track', label: 'Track evolution', value: row.track, sd: row.track_sd, colour: 'var(--color-track)' },
    { key: 'traffic', label: 'Traffic', value: row.traffic, sd: row.traffic_sd, colour: 'var(--color-traffic)' },
    // Constraint: the residual is never folded into the others to make the bars
    // add up. An engineer who finds a hidden residual stops trusting the rest.
    { key: 'residual', label: 'Unexplained', value: row.residual, sd: 0, colour: 'var(--color-residual)' },
  ].filter((t) => t.value != null)

  const extent = Math.max(
    0.25,
    ...terms.map((t) => Math.abs(t.value as number) + 1.96 * (t.sd ?? 0)),
    Math.abs(row.observed_delta),
  )

  return (
    <Panel title="This lap, taken apart" aside={<CompoundChip compound={row.compound} />}>
      <div className="mb-4 grid grid-cols-2 gap-4">
        <div>
          <div className="label-caps mb-1">The stopwatch</div>
          <div
            className="num text-[24px] leading-none font-semibold"
            style={{ color: stopwatchFaster ? 'var(--color-good)' : 'var(--color-ink)' }}
          >
            {signed(row.observed_delta, 2)}
            <span className="ml-1 text-[11px] font-normal text-ink-faint">s</span>
          </div>
          <div className="mt-1 text-[10.5px] text-ink-faint">
            {stopwatchFaster ? 'faster than the stint start' : 'slower than the stint start'}
          </div>
        </div>
        <div>
          <div className="label-caps mb-1">The tyre</div>
          <div
            className="num text-[24px] leading-none font-semibold"
            style={{ color: compoundColour(row.compound) }}
          >
            {signed(row.tyre ?? 0, 2)}
            <span className="ml-1 text-[11px] font-normal text-ink-faint">s</span>
          </div>
          <div className="mt-1 text-[10.5px] text-ink-faint">
            ± {(1.96 * (row.tyre_sd ?? 0)).toFixed(2)} at 95%
          </div>
        </div>
      </div>

      {disagree && (
        <div
          className="mb-4 rounded-md border-l-2 py-2.5 pl-3.5 pr-3"
          style={{
            borderLeftColor: 'var(--color-alert)',
            background: 'color-mix(in oklab, var(--color-alert) 5%, transparent)',
          }}
        >
          <div className="mb-1 text-[11px] font-semibold text-alert">
            The stopwatch has the sign wrong
          </div>
          <p className="text-[12px] leading-relaxed text-ink-dim">
            The lap was {Math.abs(row.observed_delta).toFixed(2)} s quicker, and the
            tyre still lost {(row.tyre ?? 0).toFixed(2)} s. Fuel burn-off is worth{' '}
            {Math.abs(row.fuel ?? 0).toFixed(2)} s here and is covering it. Reading
            pace alone would call this tyre healthy.
          </p>
        </div>
      )}

      <div className="space-y-2">
        {terms.map((t) => (
          <div key={t.key} className="grid items-center gap-2" style={{ gridTemplateColumns: '96px 1fr 58px' }}>
            <span
              className={`text-[11.5px] ${t.key === 'tyre' ? 'font-medium text-ink' : 'text-ink-dim'}`}
            >
              {t.label}
            </span>
            <Beam
              mean={t.value as number}
              sd={t.sd ?? 0}
              domain={[-extent, extent]}
              colour={t.colour}
              zero
              height={12}
            />
            <span
              className={`num text-right text-[11.5px] ${t.key === 'tyre' ? 'text-ink' : 'text-ink-dim'}`}
            >
              {signed(t.value as number, 2)}
            </span>
          </div>
        ))}
      </div>

      <p className="mt-3 text-[10.5px] leading-relaxed text-ink-faint">
        Bars run from a common zero and carry their 95% interval. Unexplained is
        the part the model cannot account for, shown rather than absorbed.
      </p>
    </Panel>
  )
}

/**
 * How often this stint's pace lied about its tyre.
 *
 * A single lap where the two disagree reads as a curiosity. The count over the
 * whole stint is the argument.
 */
function StintSignSummary({ rows }: { rows: DecompositionRow[] }) {
  const disagreeing = rows.filter((r) => r.observed_delta < 0 && (r.tyre ?? 0) > 0.05)
  if (disagreeing.length === 0) return null

  const worst = disagreeing.reduce((a, b) => ((b.tyre ?? 0) > (a.tyre ?? 0) ? b : a))

  return (
    <Panel title="How often the pace lied about the tyre" aside="this stint, real laps">
      <div className="flex flex-wrap items-baseline gap-x-6 gap-y-2">
        <div>
          <div className="num text-[26px] leading-none font-semibold text-alert">
            {disagreeing.length}
            <span className="ml-1 text-[13px] font-normal text-ink-faint">
              of {rows.length}
            </span>
          </div>
          <div className="mt-1 text-[11px] text-ink-faint">
            laps got quicker while the tyre was going away
          </div>
        </div>
        <p className="max-w-[62ch] flex-1 text-[12px] leading-relaxed text-ink-dim">
          Worst case is lap {worst.session_lap}: {Math.abs(worst.observed_delta).toFixed(2)} s
          quicker on the stopwatch, {(worst.tyre ?? 0).toFixed(2)} s lost by the tyre.
          Fitting a straight line through lap time against tyre age, which is the
          standard approach, reports the tyre improving on laps like these.
        </p>
      </div>
    </Panel>
  )
}
