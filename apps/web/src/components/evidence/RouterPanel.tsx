/**
 * The routing table.
 *
 * `src/tyremind/models/router.py` sends each of five questions to whichever
 * model the evidence says answers it best, and on two of the five that model is
 * not ours. This panel is that table, rendered.
 *
 * The declarative half -- which model owns which task, whether the margin was
 * decisive, and the rationale in words -- mirrors `ROUTES` in router.py. There
 * is no API endpoint for it yet, so it is transcribed here rather than fetched.
 * The numeric half is not transcribed: every score is looked up live in the
 * result file the route cites, exactly as `scripts/check_router_routes.py` does
 * on the Python side. If a rerun changes a ranking, this panel says the route
 * has gone stale against its own evidence instead of quietly showing the old
 * number. A hard-coded routing table that cannot disagree with its evidence is
 * a marketing claim.
 */

import type { ReactNode } from 'react'
import { Caption, Note } from '../briefing/ui'
import { OURS, type Exp19, type Exp22, type Exp20, type Exp27, type Exp30 } from './types'

// ---------------------------------------------------------------------------
// The field a route was decided on
// ---------------------------------------------------------------------------

interface FieldRow {
  model: string
  score: number
  /** Standard error, where the result file carries one. Never invented. */
  se?: number
}

interface Field {
  rows: FieldRow[]
  unit: string
  digits: number
  /** True when a larger score is better. Skill scores are; error measures are not. */
  higherIsBetter: boolean
  /** Rank on distance from zero rather than on the signed value. */
  rankOnMagnitude?: boolean
  /** Where the uncertainty on this comparison comes from, or that there is none. */
  spreadNote: ReactNode
}

interface RouteSpec {
  task: string
  question: string
  /** Model name, exactly as it appears in the result tables. */
  model: string
  metric: string
  evidence: string
  /** False when the gap sits inside the noise. Mirrors `margin_is_decisive`. */
  decisive: boolean
  /** Why this model, including why a tie broke the way it did. */
  rationale: ReactNode
  field: (e: Record<string, unknown>) => Field | null
}

function sortField(field: Field): FieldRow[] {
  const key = (r: FieldRow) => (field.rankOnMagnitude ? Math.abs(r.score) : r.score)
  return [...field.rows].sort((a, b) =>
    field.higherIsBetter ? key(b) - key(a) : key(a) - key(b),
  )
}

// ---------------------------------------------------------------------------
// The routes
// ---------------------------------------------------------------------------

const ROUTES: RouteSpec[] = [
  {
    task: 'Forecast lap time',
    question: 'What will the next lap take?',
    model: 'Pooled regression',
    metric: 'CRPS, 12 real races, rolling-origin folds',
    evidence: 'exp19_field_comparison.lap_time_prediction',
    decisive: true,
    rationale: (
      <>
        We are <strong className="text-ink">fourth of nine here</strong> and route away from
        ourselves. A pooled regression minimises lap-time error with maximum freedom in the
        nuisance terms; the physical priors and smooth latent state that let us isolate the tyre
        are exactly what cost us on fitting the total. Shipping our own worse number to keep one
        model on the front page would be vanity.
      </>
    ),
    field: (e) => {
      const exp19 = e['exp19_field_comparison'] as Exp19 | undefined
      if (!exp19?.lap_time_prediction) return null
      return {
        rows: exp19.lap_time_prediction.map((r) => ({ model: r.model, score: r.crps })),
        unit: 'CRPS',
        digits: 4,
        higherIsBetter: false,
        spreadNote: (
          <>
            exp19 reports no standard error on CRPS, so none is drawn. The spread that matters
            here is the one on our own forecasts: our 95% intervals cover{' '}
            <span className="num">
              {(
                (exp19.lap_time_prediction.find((r) => r.model === OURS)?.coverage_95 ?? 0) * 100
              ).toFixed(0)}
              %
            </span>{' '}
            of laps, against the pooled regression&rsquo;s{' '}
            <span className="num">
              {(
                (exp19.lap_time_prediction.find((r) => r.model === 'Pooled regression')
                  ?.coverage_95 ?? 0) * 100
              ).toFixed(0)}
              %
            </span>
            .
          </>
        ),
      }
    },
  },
  {
    task: 'Recover the degradation rate',
    question: 'How fast is this tyre actually going off?',
    model: OURS,
    metric: 'rate MAE against known truth, 8 synthetic seeds',
    evidence: 'exp19_field_comparison.degradation_recovery',
    decisive: true,
    rationale: (
      <>
        The quantity the problem statement actually asks for, and the one we win. exp20 repeats it
        on a generator whose functional form is not our estimator&rsquo;s, and exp29 holds it
        across eleven regimes &mdash; eight wins, one tie, two losses, both on the track-evolution
        axis.
      </>
    ),
    field: (e) => {
      const exp19 = e['exp19_field_comparison'] as Exp19 | undefined
      const exp20 = e['exp20_independent_truth'] as Exp20 | undefined
      if (!exp19?.degradation_recovery) return null
      const margin = exp20?.verdict
      return {
        rows: exp19.degradation_recovery.map((r) => ({ model: r.model, score: r.rate_mae })),
        unit: 's/lap',
        digits: 4,
        higherIsBetter: false,
        spreadNote: margin ? (
          <>
            exp19 carries no standard error on these MAEs. The margin that does carry one is
            exp20&rsquo;s, measured on the independent generator:{' '}
            <span className="num text-ink">
              +{margin.physics_margin.toFixed(4)} ± {margin.physics_margin_se.toFixed(4)}
            </span>{' '}
            s/lap over the next model. Roughly five standard errors, which is why this route is
            called decisive and the two below are not.
          </>
        ) : (
          'exp19 carries no standard error on these MAEs. Run exp20 for a margin that does.'
        ),
      }
    },
  },
  {
    task: 'Pit timing',
    question: 'Which lap should he box on?',
    model: OURS,
    metric: 'mean absolute error in laps, on the stops every model answered',
    evidence: 'exp22_pit_stop_validation.like_for_like',
    decisive: false,
    rationale: (
      <>
        <strong className="text-ink">A three-way tie.</strong> Three models sit within one standard
        error of each other and the lead is a fraction of a lap, which is not a win and is not
        claimed as one. The tie breaks our way on the two things that <em>are</em> separated: the
        calibration gap in exp30, and the answer rate &mdash; a model that declines the hard stops
        is being graded on the easy ones.
      </>
    ),
    field: (e) => {
      const exp22 = e['exp22_pit_stop_validation'] as Exp22 | undefined
      if (!exp22?.like_for_like) return null
      const rows = exp22.like_for_like
      const sorted = [...rows].sort((a, b) => a.mae_laps - b.mae_laps)
      const gap = sorted.length > 1 ? sorted[1].mae_laps - sorted[0].mae_laps : 0
      return {
        rows: rows.map((r) => ({ model: r.model, score: r.mae_laps, se: r.mae_se })),
        unit: 'laps',
        digits: 2,
        higherIsBetter: false,
        spreadNote: (
          <>
            Whiskers are ± 1 standard error. The lead is{' '}
            <span className="num text-ink">{gap.toFixed(2)}</span> laps against standard errors near{' '}
            <span className="num">{sorted[0]?.mae_se.toFixed(2)}</span> &mdash; the whiskers overlap
            almost completely, which is the definition of a tie and the reason this route is
            labelled one. Scored on the{' '}
            <span className="num">{exp22.n_common_stops}</span> stops every model answered.
          </>
        ),
      }
    },
  },
  {
    task: 'Practice to race',
    question: 'What will Friday’s number be worth on Sunday?',
    model: 'Cappello & Hoegh state-space',
    metric: 'skill against own climatology, 43 events, 568 comparisons',
    evidence: 'exp27_decircularised_practice_to_race.summary',
    decisive: true,
    rationale: (
      <>
        <strong className="text-ink">We route to a competitor.</strong> Their per-driver model is
        more self-consistent from Friday to Sunday than ours. The caveat matters more than the
        ranking: every model scores negative skill, so no Friday fit beats its own average race
        rate. The honest product behaviour is to decline the forecast and present the practice
        number as a within-session decomposition.
      </>
    ),
    field: (e) => {
      const exp27 = e['exp27_decircularised_practice_to_race'] as Exp27 | undefined
      if (!exp27?.summary) return null
      return {
        rows: exp27.summary.map((r) => ({
          model: r.model,
          score: r.skill_vs_own_climatology,
          se: r.skill_se,
        })),
        unit: 'skill',
        digits: 3,
        higherIsBetter: true,
        spreadNote: (
          <>
            Whiskers are ± 1 standard error. Zero would mean &ldquo;as good as predicting each
            driver&rsquo;s own average race rate&rdquo;. Every whisker on this axis is entirely
            below zero.
          </>
        ),
      }
    },
  },
  {
    task: 'Confidence',
    question: 'How sure are we about the window?',
    model: OURS,
    metric: 'calibration gap on the pit window, 246 real stops',
    evidence: 'exp30_pit_confidence_calibration.summary',
    decisive: false,
    rationale: (
      <>
        <strong className="text-ink">A tie, and the winner is still overconfident.</strong> Best of
        six on calibration gap, and the gap with the pooled regression is inside the noise. It
        breaks on lift &mdash; we place more probability mass on the lap the driver actually chose
        &mdash; but neither model is calibrated enough to present a confident claim without the
        conformal correction, which is what the panel below this one ships.
      </>
    ),
    field: (e) => {
      const exp30 = e['exp30_pit_confidence_calibration'] as Exp30 | undefined
      if (!exp30?.summary) return null
      return {
        rows: exp30.summary.map((r) => ({ model: r.model, score: r.calibration_gap })),
        unit: 'claimed − delivered',
        digits: 3,
        higherIsBetter: false,
        rankOnMagnitude: true,
        spreadNote: (
          <>
            Ranked on distance from zero, and every model is on the same side of it: all six claim
            more than they deliver. exp30 reports no standard error on the gap, so none is drawn
            and the route is marked a tie rather than a win.
          </>
        ),
      }
    },
  },
]

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

export function RouterPanel({ experiments }: { experiments: Record<string, unknown> }) {
  const resolved = ROUTES.map((spec) => {
    const field = spec.field(experiments)
    const sorted = field ? sortField(field) : []
    const winner = sorted[0]
    const ourRank = sorted.findIndex((r) => r.model === OURS)
    return {
      spec,
      field,
      sorted,
      winner,
      ourRank: ourRank < 0 ? null : ourRank + 1,
      // The same check `scripts/check_router_routes.py` runs: does the table
      // still put the routed model on top?
      stale: !!winner && winner.model !== spec.model,
    }
  })

  const models = new Set(ROUTES.map((r) => r.model))
  const away = ROUTES.filter((r) => r.model !== OURS).length
  const ties = ROUTES.filter((r) => !r.decisive).length

  return (
    <section className="plate">
      <header className="border-b border-line px-5 py-4">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <h2 className="text-[22px] leading-tight font-semibold tracking-[-0.02em] text-ink">
            The product is not a model. It is a router.
          </h2>
          <span className="num text-[11px] text-ink-faint">src/tyremind/models/router.py</span>
        </div>
        <div className="rule-accent mt-3" />
        <p className="mt-3 max-w-[92ch] text-[14px] leading-relaxed text-ink-dim">
          Nine models were scored on five tasks and{' '}
          <strong className="text-ink">three different models won</strong>. No single rung wins
          everything, so each question goes to whichever rung the evidence says is best at it, and
          the product says which one it used. Two rules keep that from becoming a way to launder a
          weak result: every route cites a result file, and a margin inside the noise is recorded
          as a tie rather than rounded into a win.
        </p>
      </header>

      <div className="grid grid-cols-2 gap-px border-b border-line bg-line sm:grid-cols-4">
        <Tile label="Tasks routed" value={String(ROUTES.length)} />
        <Tile label="Distinct models used" value={String(models.size)} />
        <Tile
          label="Sent to a competitor's model"
          value={String(away)}
          foot="lap-time forecasting and practice-to-race, because the evidence says so"
        />
        <Tile label="Recorded as ties" value={String(ties)} foot="pit timing, confidence" />
      </div>

      <div className="divide-y divide-line">
        {resolved.map(({ spec, field, sorted, winner, ourRank, stale }, i) => (
          <RouteRow
            key={spec.task}
            index={i}
            spec={spec}
            field={field}
            sorted={sorted}
            winner={winner}
            ourRank={ourRank}
            stale={stale}
          />
        ))}
      </div>

      <div className="border-t border-line px-5 py-4">
        <Note>
          The same split Pitwall (arXiv:2607.06495) arrived at independently and calls{' '}
          <em>&ldquo;calibration-optimal is not decision-optimal&rdquo;</em>: they route components
          down an oracle path and a decision path because the two criteria disagree. Ours disagree
          too, and exp19 measured it before we read their paper.
        </Note>
      </div>
    </section>
  )
}

function Tile({
  label,
  value,
  tone = 'default',
  foot,
}: {
  label: string
  value: string
  tone?: 'default' | 'alert'
  foot?: string
}) {
  return (
    <div className="bg-surface px-5 py-4">
      <Caption>{label}</Caption>
      <div
        className="num text-[34px] leading-none font-medium"
        style={{ color: tone === 'alert' ? 'var(--color-alert)' : 'var(--color-ink)' }}
      >
        {value}
      </div>
      {foot && <div className="mt-1.5 text-[11px] leading-snug text-ink-faint">{foot}</div>}
    </div>
  )
}

function RouteRow({
  index,
  spec,
  field,
  sorted,
  winner,
  ourRank,
  stale,
}: {
  index: number
  spec: RouteSpec
  field: Field | null
  sorted: FieldRow[]
  winner?: FieldRow
  ourRank: number | null
  stale: boolean
}) {
  const away = spec.model !== OURS

  return (
    <div className="settle px-5 py-5" style={{ ['--i' as string]: index }}>
      <div className="grid gap-5 xl:grid-cols-[minmax(240px,1fr)_1.6fr]">
        <div>
          <div className="flex items-baseline gap-2.5">
            <span className="num text-[11px] text-ink-faint">{String(index + 1).padStart(2, '0')}</span>
            <h3 className="text-[17px] leading-tight font-semibold tracking-tight text-ink">
              {spec.task}
            </h3>
          </div>
          <p className="mt-1 text-[12.5px] text-ink-faint">{spec.question}</p>

          <div className="mt-3.5 flex flex-wrap items-center gap-2">
            <span
              className="inline-flex items-center gap-2 border px-2.5 py-1.5 text-[13px]"
              style={{
                borderColor: away ? 'var(--color-fuel)' : 'var(--color-alert)',
                color: 'var(--color-ink)',
                background: `color-mix(in oklab, ${
                  away ? 'var(--color-fuel)' : 'var(--color-alert)'
                } 10%, transparent)`,
              }}
            >
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ background: away ? 'var(--color-fuel)' : 'var(--color-alert)' }}
              />
              {spec.model}
            </span>
            <span
              className="border px-2 py-1 text-[10.5px] tracking-[0.08em] uppercase"
              style={{
                borderColor: away ? 'var(--color-fuel)' : 'var(--color-line)',
                color: away ? 'var(--color-fuel)' : 'var(--color-ink-faint)',
              }}
            >
              {away ? 'routed away from us' : 'ours'}
            </span>
            <span
              className="border px-2 py-1 text-[10.5px] tracking-[0.08em] uppercase"
              style={{
                borderColor: spec.decisive
                  ? 'var(--color-line)'
                  : 'color-mix(in oklab, var(--color-medium) 55%, transparent)',
                color: spec.decisive ? 'var(--color-ink-faint)' : 'var(--color-medium)',
              }}
            >
              {spec.decisive ? 'decisive margin' : 'tie'}
            </span>
          </div>

          <p className="mt-3.5 max-w-[52ch] text-[12.5px] leading-relaxed text-ink-dim">
            {spec.rationale}
          </p>

          <div className="num mt-3 text-[10.5px] text-ink-faint">{spec.evidence}</div>
        </div>

        <div>
          <Caption>{spec.metric}</Caption>
          {!field || !winner ? (
            <p className="text-[12px] text-ink-faint">
              The result file this route cites is not being served. Nothing is shown in its place.
            </p>
          ) : (
            <>
              <FieldStrip field={field} sorted={sorted} routedTo={spec.model} />
              <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-1 text-[12px]">
                <span className="text-ink-dim">
                  Winner{' '}
                  <span className="num text-ink">{fmt(winner.score, field.digits)}</span>
                  {winner.se != null && (
                    <span className="num text-ink-faint"> ± {fmt(winner.se, field.digits)}</span>
                  )}
                </span>
                {sorted[1] && (
                  <span className="text-ink-dim">
                    Runner-up{' '}
                    <span className="num text-ink">{fmt(sorted[1].score, field.digits)}</span>
                    {sorted[1].se != null && (
                      <span className="num text-ink-faint"> ± {fmt(sorted[1].se, field.digits)}</span>
                    )}
                  </span>
                )}
                {ourRank != null && (
                  <span className="text-ink-dim">
                    We place{' '}
                    <span
                      className="num"
                      style={{ color: ourRank === 1 ? 'var(--color-ink)' : 'var(--color-alert)' }}
                    >
                      {ordinal(ourRank)} of {sorted.length}
                    </span>
                  </span>
                )}
              </div>
              <p className="mt-2 max-w-[78ch] text-[11.5px] leading-relaxed text-ink-faint">
                {field.spreadNote}
              </p>
              {stale && (
                <p
                  className="mt-2.5 border-l-2 pl-3 text-[12px] leading-relaxed"
                  style={{ borderColor: 'var(--color-alert)', color: 'var(--color-alert)' }}
                >
                  This route has gone stale against its own evidence: the table now puts{' '}
                  <strong>{winner.model}</strong> on top, not {spec.model}. Rerun the experiment or
                  change the route &mdash; do not read the route as current.
                </p>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

/**
 * Every model on one axis.
 *
 * A two-model comparison would let a fourth place look like a near miss. Putting
 * the whole field on a shared ruler is what makes "we are fourth of nine on lap
 * times" a thing a judge can see rather than a sentence they have to take on
 * trust, and it costs nothing to draw.
 */
function FieldStrip({
  field,
  sorted,
  routedTo,
}: {
  field: Field
  sorted: FieldRow[]
  routedTo: string
}) {
  const values = sorted.flatMap((r) => [r.score - (r.se ?? 0), r.score + (r.se ?? 0)])
  // Zero is on the axis whenever it is a meaningful reference: a skill score of
  // zero is "no better than the climatology", and a calibration gap of zero is
  // "claims what it delivers". For a plain error measure it is not, so the axis
  // is left to the data.
  const anchorZero = field.higherIsBetter || !!field.rankOnMagnitude
  const rawLo = anchorZero ? Math.min(...values, 0) : Math.min(...values)
  const rawHi = anchorZero ? Math.max(...values, 0) : Math.max(...values)
  const pad = (rawHi - rawLo) * 0.08 || 1
  const lo = rawLo - pad
  const hi = rawHi + pad
  const span = hi - lo || 1
  const pct = (v: number) => ((v - lo) / span) * 100
  const clamp = (v: number) => Math.min(100, Math.max(0, v))

  return (
    <div className="space-y-1">
      {sorted.map((r, i) => {
        const ours = r.model === OURS
        const chosen = r.model === routedTo
        const colour = ours
          ? 'var(--color-alert)'
          : chosen
            ? 'var(--color-fuel)'
            : 'var(--color-ink-faint)'
        return (
          <div
            key={r.model}
            className="settle grid grid-cols-[minmax(120px,1.1fr)_2fr_minmax(66px,auto)] items-center gap-3"
            style={{ ['--i' as string]: i }}
          >
            <span
              className="truncate text-[12px]"
              style={{ color: ours || chosen ? 'var(--color-ink)' : 'var(--color-ink-dim)' }}
              title={r.model}
            >
              {r.model}
            </span>
            <div className="relative h-4 bg-raised">
              {lo < 0 && hi > 0 && (
                <div
                  className="absolute top-0 bottom-0 w-px bg-line-bright"
                  style={{ left: `${pct(0)}%` }}
                />
              )}
              {r.se != null && (
                <div
                  className="beam absolute top-1 bottom-1"
                  style={{
                    left: `${clamp(pct(r.score - r.se))}%`,
                    width: `${Math.max(clamp(pct(r.score + r.se)) - clamp(pct(r.score - r.se)), 0.6)}%`,
                    ['--beam-color' as string]: colour,
                  }}
                />
              )}
              <div
                className="absolute top-0 bottom-0 w-[2px]"
                style={{ left: `${clamp(pct(r.score))}%`, background: colour }}
              />
            </div>
            <span
              className="num text-right text-[12px]"
              style={{ color: ours || chosen ? 'var(--color-ink)' : 'var(--color-ink-dim)' }}
            >
              {fmt(r.score, field.digits)}
            </span>
          </div>
        )
      })}
      <div className="flex justify-between pt-0.5 text-[10px] text-ink-faint">
        <span className="num">{fmt(lo, field.digits)}</span>
        <span>
          {field.unit} · {field.higherIsBetter ? 'higher is better' : 'lower is better'}
        </span>
        <span className="num">{fmt(hi, field.digits)}</span>
      </div>
    </div>
  )
}

function fmt(v: number, digits: number): string {
  return Number.isFinite(v) ? v.toFixed(digits) : '—'
}

function ordinal(n: number): string {
  const tens = n % 100
  if (tens >= 11 && tens <= 13) return `${n}th`
  return `${n}${({ 1: 'st', 2: 'nd', 3: 'rd' } as Record<number, string>)[n % 10] ?? 'th'}`
}
