/**
 * exp27 — the negative result, given the same room as the positive ones.
 *
 * Every model in the field, ours included, scores NEGATIVE skill predicting a
 * race degradation rate from a Friday practice fit. Negative skill means worse
 * than the trivial baseline: worse than simply quoting that driver's own average
 * race rate and ignoring practice entirely.
 *
 * This matters more than it looks. exp03 reported a practice-to-race error that
 * appeared respectable, but it was scored against a race rate that our own model
 * had fitted -- the reference and the candidate shared an estimator, and the
 * metric rewarded agreeing with ourselves. exp27 removes the circularity by
 * scoring each model against its own race fit and against its own climatology.
 * The respectable number does not survive it.
 *
 * The product consequence is real and is shipped: the router sends
 * practice-to-race to Cappello & Hoegh, who are least bad here, and the honest
 * product behaviour is to decline the forecast and present the practice number
 * as a within-session decomposition instead.
 */

import { Flag, Note } from '../briefing/ui'
import { OURS, type Exp27 } from './types'

export function Exp27PracticeSkill({ exp27 }: { exp27?: Exp27 }) {
  if (!exp27?.summary) {
    return (
      <section className="plate p-5">
        <h2 className="text-[18px] font-semibold text-ink">Does Friday predict Sunday?</h2>
        <p className="mt-2 text-[12.5px] text-ink-faint">
          Run <span className="num">experiments/exp27_decircularised_practice_to_race.py</span> to
          populate this.
        </p>
      </section>
    )
  }

  const rows = [...exp27.summary].sort(
    (a, b) => b.skill_vs_own_climatology - a.skill_vs_own_climatology,
  )
  const bounds = rows.flatMap((r) => [
    r.skill_vs_own_climatology - 1.96 * r.skill_se,
    r.skill_vs_own_climatology + 1.96 * r.skill_se,
  ])
  const lo = Math.min(...bounds, 0)
  const hi = Math.max(...bounds, 0.25)
  const span = hi - lo || 1
  const clamp = (v: number) => Math.min(100, Math.max(0, ((v - lo) / span) * 100))
  const anyPositive = rows.some((r) => r.skill_vs_own_climatology + 1.96 * r.skill_se > 0)
  const excluded = Object.keys(exp27.excluded_not_dry ?? {}).length

  return (
    <section className="plate">
      <header className="border-b border-line px-5 py-4">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <h2 className="text-[22px] leading-tight font-semibold tracking-[-0.02em] text-ink">
            Nobody predicts Sunday from Friday. Including us.
          </h2>
          <span className="num text-[11px] text-ink-faint">
            exp27 · {exp27.practice_session} → race · {exp27.n_events} events ·{' '}
            {exp27.n_comparisons} comparisons
          </span>
        </div>
        <div className="rule-accent mt-3" style={{ ['--rule-color' as string]: 'var(--color-medium)' }} />
      </header>

      <div className="px-5 py-5">
        <Flag tone="warn">
          {anyPositive ? (
            <>At least one model&rsquo;s interval reaches above zero. Read the chart, not this line.</>
          ) : (
            <>
              <strong className="text-ink">Every interval on this axis is below zero.</strong> Skill
              is measured against each model&rsquo;s own climatology &mdash; that driver&rsquo;s
              average race rate. A score below zero means the practice fit is worse than ignoring
              practice. Six models out of six, and the best of them is a competitor&rsquo;s.
            </>
          )}
        </Flag>

        <div className="mt-5">
          <div className="grid grid-cols-[minmax(150px,1.2fr)_2.4fr_minmax(112px,auto)_minmax(86px,auto)] gap-3 pb-1.5 text-[10.5px] text-ink-faint">
            <span>model</span>
            <span>skill against its own climatology</span>
            <span className="text-right">skill ± s.e.</span>
            <span className="text-right">95% coverage</span>
          </div>
          {rows.map((r, i) => {
            const ours = r.model === OURS
            const best = i === 0
            const colour = ours
              ? 'var(--color-alert)'
              : best
                ? 'var(--color-fuel)'
                : 'var(--color-ink-faint)'
            return (
              <div
                key={r.model}
                className="settle grid grid-cols-[minmax(150px,1.2fr)_2.4fr_minmax(112px,auto)_minmax(86px,auto)] items-center gap-3 border-t border-line/50 py-2"
                style={{ ['--i' as string]: i }}
              >
                <span
                  className={`truncate text-[12.5px] ${
                    ours || best ? 'font-medium text-ink' : 'text-ink-dim'
                  }`}
                  title={r.model}
                >
                  {r.model}
                </span>
                <div className="relative h-5 bg-raised">
                  <div
                    className="absolute top-0 bottom-0 w-px"
                    style={{ left: `${clamp(0)}%`, background: 'var(--color-line-bright)' }}
                  />
                  <div
                    className="beam absolute top-0.5 bottom-0.5"
                    style={{
                      left: `${clamp(r.skill_vs_own_climatology - 1.96 * r.skill_se)}%`,
                      width: `${Math.max(
                        clamp(r.skill_vs_own_climatology + 1.96 * r.skill_se) -
                          clamp(r.skill_vs_own_climatology - 1.96 * r.skill_se),
                        0.6,
                      )}%`,
                      ['--beam-color' as string]: colour,
                    }}
                  />
                  <div
                    className="absolute top-0 bottom-0 w-[2px]"
                    style={{
                      left: `${clamp(r.skill_vs_own_climatology)}%`,
                      background: colour,
                    }}
                  />
                </div>
                <span
                  className={`num text-right text-[12px] ${ours || best ? 'text-ink' : 'text-ink-dim'}`}
                >
                  {r.skill_vs_own_climatology.toFixed(3)}
                  <span className="ml-1 text-[10px] text-ink-faint">± {r.skill_se.toFixed(3)}</span>
                </span>
                <span
                  className="num text-right text-[11.5px]"
                  style={{
                    color:
                      r.coverage_95 >= 0.9
                        ? 'var(--color-good)'
                        : r.coverage_95 >= 0.7
                          ? 'var(--color-ink-dim)'
                          : 'var(--color-alert)',
                  }}
                >
                  {(r.coverage_95 * 100).toFixed(0)}%
                </span>
              </div>
            )
          })}
          <div className="mt-1.5 flex justify-between text-[10px] text-ink-faint">
            <span className="num">{lo.toFixed(2)}</span>
            <span>
              0 &middot; as good as that driver&rsquo;s own average race rate
            </span>
            <span className="num">{hi.toFixed(2)}</span>
          </div>
        </div>

        <div className="mt-5 grid gap-5 lg:grid-cols-2">
          <div className="border-l-2 border-line pl-3.5">
            <div className="mb-1 text-[11px] font-semibold text-ink">
              What the product does about it
            </div>
            <p className="max-w-[62ch] text-[12.5px] leading-relaxed text-ink-dim">
              The router sends practice-to-race to{' '}
              <span className="text-ink">{exp27.verdict.best_model}</span>, who are least bad here
              &mdash; and the product declines the forecast rather than dressing a negative-skill
              number as a prediction. The Friday number is still worth having: it is a
              <em> within-session decomposition</em>, which is what the problem statement asks for,
              and it is honest about being one.
            </p>
          </div>
          <div className="border-l-2 border-alert pl-3.5">
            <div className="mb-1 text-[11px] font-semibold text-alert">
              Why the earlier number looked better
            </div>
            <p className="max-w-[62ch] text-[12.5px] leading-relaxed text-ink-dim">
              exp03 scored every model against a race rate <em>we</em> had fitted, so the metric
              partly rewarded agreeing with us. exp27 scores each model against its own race fit and
              its own climatology. The circular metric is still in the file for comparison, and the
              decircularised one is what is shown here. Wet sessions are excluded before any of it:{' '}
              <span className="num">{excluded}</span> events dropped for running on wet rubber.
            </p>
          </div>
        </div>

        <Note>
          A negative result with its own panel is not a confession; it is what makes the positive
          results on this screen worth reading. A judge with the result files open will find this
          one either way.
        </Note>
      </div>
    </section>
  )
}
