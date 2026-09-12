/**
 * The opening screen: what this session showed, in terms anyone can read.
 *
 * Leads with the comparison against the naive method because on real races that
 * method reports NEGATIVE degradation, a claim any reader immediately recognises
 * as impossible.
 */

import { useEffect, useState } from 'react'
import {
  api,
  compoundColour,
  fixed,
  type SessionSummary,
  type RunRow,
} from '../lib/api'
import { DegradationCurves, QualityBreakdown, type CompoundCurve } from './charts'
import { Beam, CompoundChip, ErrorNote, Loading, Panel, REGIME_TONE, Stat } from './primitives'
import { Explainer, Term } from './Explainer'

export function Overview({
  sessionId,
  onOpenExplain,
}: {
  sessionId: string
  onOpenExplain: () => void
}) {
  const [summary, setSummary] = useState<SessionSummary | null>(null)
  const [runs, setRuns] = useState<RunRow[]>([])
  const [error, setError] = useState('')

  useEffect(() => {
    setSummary(null)
    setError('')
    Promise.all([api.summary(sessionId), api.runs(sessionId)])
      .then(([s, r]) => {
        setSummary(s)
        setRuns(r)
      })
      .catch((e) => setError(String(e.message ?? e)))
  }, [sessionId])

  if (error) return <ErrorNote error={error} />
  if (!summary) return <Loading what="this session" />

  const compounds = Object.entries(summary.compounds)
  const naiveNegative = compounds.filter(
    ([, c]) => c.naive_estimate != null && c.naive_estimate < 0,
  )
  const longest = runs[0]

  const curves: CompoundCurve[] = Object.entries(summary.compounds)
    .map(([compound, estimate]) => ({
      compound,
      rate: estimate.degradation_rate,
      sd: estimate.degradation_rate_sd,
      maxAge: Math.max(0, ...runs.filter((r) => r.compound === compound).map((r) => r.end_age)),
    }))
    .filter((c) => c.maxAge >= 5)

  return (
    <div className="space-y-4">
      <Explainer id="overview" question="What is this, in one paragraph?">
        <p>
          A Formula 1 tyre gets slower as it wears. Teams need to know{' '}
          <em>how much</em> slower, per lap, so they can decide when to pit.
          The obvious way to measure it, watching lap times climb, is wrong.
        </p>
        <p>
          Lap times move for several reasons at once: fuel burn-off, track
          rubber, and time lost to{' '}
          <Term word="dirty air" meaning="Turbulence behind another car, which removes downforce from the car following." />
          . <strong>TyreMind separates these and reports only the tyre.</strong>
        </p>
      </Explainer>

      {/* Naive vs TyreMind comparison */}
      {naiveNegative.length > 0 && (
        <Panel title="Why the obvious method fails" aside="this session, real data">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-start">
            <div className="lg:w-[44%]">
              <p className="mb-3 max-w-[52ch] text-[13px] leading-relaxed text-ink">
                The standard approach, fitting a straight line through lap
                time against tyre age, reports that tyres got{' '}
                <strong style={{ color: 'var(--color-danger)' }}>faster</strong> the
                longer they ran on this race.
              </p>
              <p className="mb-4 max-w-[52ch] text-[12px] leading-relaxed text-ink-dim">
                That&rsquo;s not subtle: fuel burn-off makes the car quicker by
                about 0.08 s a lap, larger than the tyre&rsquo;s own
                degradation, so the tyre effect comes out with the wrong sign.
              </p>
              <button
                onClick={onOpenExplain}
                className="rounded-pill border border-alert px-3.5 py-1.5 text-[12px] font-medium text-alert transition-colors duration-150 hover:bg-alert/10"
              >
                See where the lap time actually went →
              </button>
            </div>

            <div className="flex-1 space-y-2.5">
              {compounds.map(([compound, estimate]) => {
                const colour = compoundColour(compound)
                const naiveBad = (estimate.naive_estimate ?? 0) < 0
                return (
                  <div
                    key={compound}
                    className="rounded-md border border-line bg-raised/40 p-3 shadow-card"
                  >
                    <div className="mb-2.5 flex items-center justify-between">
                      <CompoundChip compound={compound} />
                      <span className="text-[10px] text-ink-faint">
                        {estimate.laps} laps
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      {/* Naive */}
                      <div>
                        <div className="label-caps mb-1">Lap time vs age</div>
                        <div
                          className="num text-[19px] leading-tight font-semibold"
                          style={{
                            color: naiveBad
                              ? 'var(--color-danger)'
                              : 'var(--color-ink-dim)',
                          }}
                        >
                          {fixed(estimate.naive_estimate)}
                        </div>
                        <div className="text-[10px] text-ink-faint">
                          {naiveBad ? 'negative, impossible' : 's/lap'}
                        </div>
                      </div>
                      {/* TyreMind */}
                      <div>
                        <div className="label-caps mb-1">TyreMind</div>
                        <div
                          className="num text-[19px] leading-tight font-semibold"
                          style={{ color: colour }}
                        >
                          {fixed(estimate.degradation_rate)}
                        </div>
                        <div className="text-[10px] text-ink-faint">
                          ± {estimate.degradation_rate_sd.toFixed(3)} s/lap
                        </div>
                      </div>
                    </div>
                    <div className="mt-2.5">
                      <Beam
                        mean={estimate.degradation_rate}
                        sd={estimate.degradation_rate_sd}
                        domain={[-0.08, 0.28]}
                        colour={colour}
                        zero
                        height={10}
                      />
                    </div>

                    {estimate.race_projection && (
                      <div className="mt-2.5 border-t border-line pt-2.5">
                        <div className="flex items-baseline justify-between">
                          <span className="label-caps">Projected race rate</span>
                          <span className="num text-[13px] text-ink">
                            {fixed(estimate.race_projection.expected_race_rate)}
                          </span>
                        </div>
                        <div className="mt-0.5 text-[10px] text-ink-faint">
                          {Math.round(estimate.race_projection.confidence * 100)}% interval{' '}
                          <span className="num">
                            {fixed(estimate.race_projection.interval[0])} to{' '}
                            {fixed(estimate.race_projection.interval[1])}
                          </span>
                        </div>
                        {/*
                          Provenance stays on the card, not in a tooltip: an
                          interval is only worth what the evidence behind it
                          is worth.
                        */}
                        <div className="mt-0.5 text-[10px] text-ink-faint">
                          calibrated on {estimate.race_projection.calibrated_on.events} events,{' '}
                          {estimate.race_projection.calibrated_on.seasons.join('/')}
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        </Panel>
      )}

      {/* Degradation curves */}
      {curves.length > 0 && runs.length > 0 && (
        <Panel
          title="What the model actually estimates"
          aside="one curve per compound, with uncertainty"
        >
          <div className="grid gap-5 lg:grid-cols-[1.5fr_1fr]">
            <DegradationCurves curves={curves} />
            <div className="space-y-3 prose-data">
              <p>
                Each line is how much performance a compound has lost after a
                given number of laps, with fuel, track and traffic removed.
                The shaded band is the 95% range.
              </p>
              <p>
                <strong>Is it straight?</strong> A curve that steepens is a
                tyre falling off its cliff.{' '}
                <strong>Where does the band flare?</strong> Where the session
                ran out of laps on that compound.
              </p>
              <CurveShapeSummary runs={runs} />
              <p className="text-[11.5px] text-ink-faint">
                Each line stops at the oldest tyre age that compound reached
                this session.
              </p>
            </div>
          </div>
        </Panel>
      )}

      {/* Confounders + session stats */}
      <div className="grid gap-4 lg:grid-cols-[1.15fr_1fr]">
        <Panel title="What moved the lap times" aside="estimated for this session">
          <div className="space-y-4">
            <Effect
              label="Fuel burn-off"
              value={`${fixed(summary.confounders.fuel_slope.mean)} s/lap`}
              colour="var(--color-fuel)"
              plain="The car is heavy at the start and light at the end. Every lap it burns fuel, gets lighter, and goes quicker."
              caveat="Set by physics, not measured: fuel and tyre wear both change smoothly with laps, so timing data can't tell them apart."
            />
            <Effect
              label="Track evolution"
              value={`${fixed(summary.confounders.track_evolution.mean, 2)} s over the session`}
              colour="var(--color-track)"
              plain="Cars leave rubber on the racing line. That rubber adds grip, so the circuit itself gets faster as the session goes on."
              caveat="Partly assumed. Modelled as a saturating curve because rubber build-up genuinely saturates."
            />
            <Effect
              label="Traffic"
              value={`${fixed(summary.confounders.traffic.mean, 2)} s when worst`}
              colour="var(--color-traffic)"
              plain="Following another car costs downforce and therefore lap time. Estimated from when each car started its lap."
              caveat="Measured from data: traffic comes and goes independently of tyre age."
            />
          </div>
        </Panel>

        <div className="space-y-4">
          {/* Session stats */}
          <Panel
            title="This session"
            aside={`quality ${(summary.quality.quality_score ?? 0).toFixed(0)}/100`}
          >
            <div className="grid grid-cols-2 gap-5 sm:grid-cols-4 lg:grid-cols-2">
              <Stat label="Laps analysed" value={String(summary.n_laps)} />
              <Stat label="Cars" value={String(summary.n_drivers)} />
              <Stat label="Tyre sets" value={String(summary.n_runs)} />
              <Stat
                label="Longest stint"
                value={longest ? String(longest.laps) : '-'}
                unit="laps"
              />
            </div>

            {summary.quality.exclusions && (
              <div className="mt-4 border-t border-line pt-3">
                <div className="label-caps mb-2">Laps removed before analysis</div>
                <QualityBreakdown
                  retained={summary.n_laps}
                  exclusions={summary.quality.exclusions}
                  labels={EXCLUSION_PLAIN}
                />
                <p className="mt-1.5 max-w-[44ch] text-[11px] leading-relaxed text-ink-faint">
                  Pit laps, safety-car laps and scruffy laps say nothing about the
                  tyre and would corrupt the estimate.
                </p>
              </div>
            )}
          </Panel>

          {/* Model diagnostics */}
          <Panel title="How sure is it?" aside="model diagnostics">
            <div className="grid grid-cols-2 gap-5">
              <Stat
                label="Unexplained scatter"
                value={summary.diagnostics.observation_noise_sd.toFixed(3)}
                unit="s"
                hint="driver variation not modelled"
              />
              <Stat
                label="States tracked"
                value={String(summary.diagnostics.n_states)}
                hint="one per car per tyre set"
              />
            </div>
            <p className="mt-3 max-w-[46ch] prose-data text-[11.5px]">
              Every number carries a{' '}
              <Term
                word="credible interval"
                meaning="A range the true value probably falls in, given the data and the model's assumptions."
              />
              . A wide bar means the model is genuinely unsure: that's
              information, not a defect.
            </p>
          </Panel>
        </div>
      </div>
    </div>
  )
}

function Effect({
  label,
  value,
  colour,
  plain,
  caveat,
}: {
  label: string
  value: string
  colour: string
  plain: string
  caveat: string
}) {
  return (
    <div
      className="border-l-2 pl-3.5"
      style={{ borderLeftColor: colour }}
    >
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-[13px] font-medium text-ink">{label}</span>
        <span className="num text-[13px] text-ink">{value}</span>
      </div>
      <p className="mt-1 max-w-[56ch] text-[12px] leading-relaxed text-ink-dim">{plain}</p>
      <p className="mt-0.5 max-w-[56ch] text-[11px] leading-relaxed text-ink-faint">{caveat}</p>
    </div>
  )
}

const EXCLUSION_PLAIN: Record<string, string> = {
  no_lap_time:                    'No timed lap completed',
  pit_in_out_lap:                 'Entering or leaving the pit lane',
  flagged_inaccurate:             'Timing data flagged unreliable',
  unknown_compound:               'Tyre compound not recorded',
  wet_compound:                   'Wet-weather tyre (different physics)',
  slow_lap_safety_car_or_traffic: 'Far too slow: safety car, flags or heavy traffic',
  run_too_short:                  'Stint too short to show a trend',
}

/**
 * How the individual stints in this session actually behaved.
 *
 * The curve above is the model's pooled estimate per compound; this is what the
 * stints underneath it did one at a time, and they do not all do the same thing.
 * A stint that holds and then falls away and one that fades evenly can share a
 * degradation rate and mean completely different things to whoever is deciding
 * when to pit, which is the entire reason a rate is not a curve.
 *
 * Shapes are fitted per stint on de-confounded lap time, and a stint is only
 * called a cliff if the tyre was already degrading before the break. The same
 * arithmetic with the tyre getting faster beforehand is a warm-up, which is the
 * opposite event.
 */
function CurveShapeSummary({ runs }: { runs: RunRow[] }) {
  const shaped = runs.filter((r) => r.curve)
  if (shaped.length < 3) return null

  const counts = shaped.reduce<Record<string, number>>((acc, r) => {
    const regime = r.curve!.regime
    acc[regime] = (acc[regime] ?? 0) + 1
    return acc
  }, {})

  // The longest stint that is not merely linear: the most informative single
  // example on offer, and worth naming rather than summarising away.
  const notable = shaped
    .filter((r) => r.curve!.regime !== 'linear')
    .sort((a, b) => b.laps - a.laps)[0]

  return (
    <div className="border-t border-line pt-3">
      <div className="mb-2 text-[11px] font-semibold text-ink">
        What the individual stints did
      </div>
      <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1">
        {['linear', 'warm-up', 'cliff', 'recovery'].map((regime) =>
          counts[regime] ? (
            <span key={regime} className="text-[11.5px]">
              <span className={`num ${REGIME_TONE[regime]}`}>{counts[regime]}</span>{' '}
              <span className="text-ink-faint">{regime}</span>
            </span>
          ) : null,
        )}
      </div>
      {notable && (
        <p className="text-[11.5px] leading-relaxed text-ink-faint">
          Longest non-linear stint:{' '}
          <span className="text-ink-dim">
            {notable.driver}, {notable.compound.toLowerCase()}, {notable.laps} laps:
          </span>{' '}
          {notable.curve!.description}
        </p>
      )}
    </div>
  )
}
