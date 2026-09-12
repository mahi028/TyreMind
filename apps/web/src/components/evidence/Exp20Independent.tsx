/**
 * exp20 — the headline, tested on a generator that is not our own model.
 *
 * The obvious objection to any synthetic recovery result is that the generator
 * and the estimator share a functional form, so the estimator is being asked to
 * invert an equation it already knows. exp20 removes that: the physics arm
 * builds lap times from an energy-and-temperature process in which no
 * degradation rate appears as a parameter at all, and the "true rate" has to be
 * read back out of the simulated truth afterwards.
 *
 * Two findings, and the second is the interesting one.
 *
 * The margin survives. On the pre-registered primary target we are first, by
 * about five standard errors.
 *
 * **But the ranking depends on which definition of "the rate" you score.** Read
 * the truth three defensible ways and a different model wins each time. That is
 * not a caveat buried under the win, it is the more useful result: it says the
 * field's disagreement is partly about what the quantity *is*, and a leaderboard
 * quoting one number per model is hiding a choice somebody made. So all three
 * are on screen, with their definitions.
 */

import { Caption, Note } from '../briefing/ui'
import { OURS, type Exp20 } from './types'

export function Exp20Independent({ exp20 }: { exp20?: Exp20 }) {
  if (!exp20?.verdict) {
    return (
      <section className="plate p-5">
        <h2 className="text-[18px] font-semibold text-ink">Does it hold on a truth we did not write?</h2>
        <p className="mt-2 text-[12.5px] text-ink-faint">
          Run <span className="num">experiments/exp20_independent_truth.py</span> to populate this.
        </p>
      </section>
    )
  }

  const v = exp20.verdict
  const targetKeys = Object.keys(exp20.targets)
  const primary = exp20.readings.physics?.[v.primary_target]

  return (
    <section className="plate">
      <header className="border-b border-line px-5 py-4">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <h2 className="text-[22px] leading-tight font-semibold tracking-[-0.02em] text-ink">
            The margin survives a generator that is not ours
          </h2>
          <span className="num text-[11px] text-ink-faint">
            exp20 · {exp20.n_seeds} seeds · pre-registered
          </span>
        </div>
        <div className="rule-accent mt-3" />
      </header>

      <div className="grid gap-px bg-line lg:grid-cols-[1fr_1.35fr]">
        <div className="bg-surface px-5 py-5">
          <Caption>Margin over the next model, primary target</Caption>
          <div className="flex items-baseline gap-2">
            <span className="num text-[48px] leading-none font-medium text-alert">
              +{v.physics_margin.toFixed(4)}
            </span>
            <span className="num text-[16px] text-ink-faint">± {v.physics_margin_se.toFixed(4)}</span>
          </div>
          <div className="mt-1 text-[11.5px] text-ink-faint">
            s/lap, paired over {primary?.paired_margin_best_over_runner_up.n_paired ?? '—'}{' '}
            compound-seeds · 95%{' '}
            <span className="num">
              {(v.physics_margin - 1.96 * v.physics_margin_se).toFixed(4)} to{' '}
              {(v.physics_margin + 1.96 * v.physics_margin_se).toFixed(4)}
            </span>
          </div>

          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <div>
              <Caption>On the synthetic arm</Caption>
              <div className="num text-[22px] leading-none text-ink-dim">
                +{v.synthetic_margin.toFixed(4)}
              </div>
              <p className="mt-1.5 text-[11px] leading-snug text-ink-faint">
                Where the generator and the estimator do share a form.
              </p>
            </div>
            <div>
              <Caption>Survival ratio</Caption>
              <div className="num text-[22px] leading-none text-ink">
                {v.survival_ratio.toFixed(2)}×
              </div>
              <p className="mt-1.5 text-[11px] leading-snug text-ink-faint">
                The margin is larger on the harder arm, not smaller. The pre-registration predicted
                the opposite.
              </p>
            </div>
          </div>

          <div className="mt-4 border-l-2 border-alert pl-3.5">
            <div className="mb-1 text-[11px] font-semibold text-alert">
              This is not third-party validation
            </div>
            <p className="max-w-[50ch] text-[12px] leading-relaxed text-ink-dim">
              {exp20.honest_framing}
            </p>
          </div>
        </div>

        <div className="bg-surface px-5 py-5">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h3 className="text-[15px] font-semibold tracking-tight text-ink">
              A different model wins each definition of &ldquo;the rate&rdquo;
            </h3>
            {v.physics_ranking_depends_on_target && (
              <span
                className="border px-2 py-0.5 text-[10px] tracking-[0.08em] uppercase"
                style={{
                  borderColor: 'color-mix(in oklab, var(--color-medium) 55%, transparent)',
                  color: 'var(--color-medium)',
                }}
              >
                ranking is target-dependent
              </span>
            )}
          </div>

          <div className="mt-3 space-y-3">
            {targetKeys.map((key, i) => {
              const physicsWinner = v.physics_best_by_target[key]
              const syntheticWinner = v.synthetic_best_by_target[key]
              const reading = exp20.readings.physics?.[key]
              const isPrimary = key === v.primary_target
              const oursRow = reading?.table.find((r) => r.model === OURS)
              return (
                <div
                  key={key}
                  className="settle border-l-2 pl-3.5"
                  style={{
                    ['--i' as string]: i,
                    borderColor: isPrimary ? 'var(--color-alert)' : 'var(--color-line)',
                  }}
                >
                  <div className="flex flex-wrap items-baseline gap-2">
                    <span className="num text-[12.5px] text-ink">{key}</span>
                    {isPrimary && (
                      <span className="text-[10px] tracking-[0.08em] text-alert uppercase">
                        pre-registered primary
                      </span>
                    )}
                  </div>
                  <p className="mt-1 max-w-[62ch] text-[11.5px] leading-relaxed text-ink-faint">
                    {exp20.targets[key]}
                  </p>
                  <div className="mt-1.5 flex flex-wrap gap-x-6 gap-y-1 text-[12px]">
                    <span className="text-ink-dim">
                      Wins on physics:{' '}
                      <span
                        style={{
                          color:
                            physicsWinner === OURS ? 'var(--color-alert)' : 'var(--color-fuel)',
                        }}
                      >
                        {physicsWinner}
                      </span>
                    </span>
                    <span className="text-ink-faint">
                      on synthetic: <span className="text-ink-dim">{syntheticWinner}</span>
                    </span>
                    {oursRow && (
                      <span className="text-ink-faint">
                        we score{' '}
                        <span className="num text-ink-dim">
                          {oursRow.rate_mae.toFixed(4)} ± {oursRow.rate_mae_se.toFixed(4)}
                        </span>
                      </span>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {primary && (
        <div className="border-t border-line px-5 py-5">
          <Caption>
            The primary target in full &mdash; {v.primary_target}, physics arm
          </Caption>
          <ModelTable rows={primary.table} />
        </div>
      )}

      <div className="border-t border-line px-5 py-4">
        <Note>
          Every row carries its own standard error, and the coverage column is the share of 95%
          intervals that actually contained the truth. A model can hold a respectable error and
          still be badly calibrated &mdash; the pooled regression&rsquo;s intervals cover none of
          the truths on this arm &mdash; and a product that routes on error alone would never see
          it.
        </Note>
      </div>
    </section>
  )
}

function ModelTable({
  rows,
}: {
  rows: { model: string; n: number; rate_mae: number; rate_mae_se: number; coverage_95: number }[]
}) {
  const sorted = [...rows].sort((a, b) => a.rate_mae - b.rate_mae)
  const hi = Math.max(...sorted.map((r) => r.rate_mae + 1.96 * r.rate_mae_se))
  const clamp = (v: number) => Math.min(100, Math.max(0, (v / (hi || 1)) * 100))

  return (
    <div className="mt-2">
      <div className="grid grid-cols-[minmax(150px,1.2fr)_2.2fr_minmax(112px,auto)_minmax(78px,auto)] gap-3 pb-1.5 text-[10.5px] text-ink-faint">
        <span>model</span>
        <span>rate MAE against the truth, s/lap</span>
        <span className="text-right">MAE ± s.e.</span>
        <span className="text-right">95% coverage</span>
      </div>
      {sorted.map((r, i) => {
        const ours = r.model === OURS
        const colour = ours ? 'var(--color-alert)' : 'var(--color-ink-faint)'
        return (
          <div
            key={r.model}
            className="settle grid grid-cols-[minmax(150px,1.2fr)_2.2fr_minmax(112px,auto)_minmax(78px,auto)] items-center gap-3 border-t border-line/50 py-1.5"
            style={{ ['--i' as string]: i }}
          >
            <span
              className={`truncate text-[12.5px] ${ours ? 'font-medium text-ink' : 'text-ink-dim'}`}
              title={r.model}
            >
              {r.model}
            </span>
            <div className="relative h-4 bg-raised">
              <div
                className="beam absolute top-0.5 bottom-0.5"
                style={{
                  left: `${clamp(r.rate_mae - 1.96 * r.rate_mae_se)}%`,
                  width: `${Math.max(
                    clamp(r.rate_mae + 1.96 * r.rate_mae_se) -
                      clamp(r.rate_mae - 1.96 * r.rate_mae_se),
                    0.6,
                  )}%`,
                  ['--beam-color' as string]: colour,
                }}
              />
              <div
                className="absolute top-0 bottom-0 w-[2px]"
                style={{ left: `${clamp(r.rate_mae)}%`, background: colour }}
              />
            </div>
            <span className={`num text-right text-[12px] ${ours ? 'text-ink' : 'text-ink-dim'}`}>
              {r.rate_mae.toFixed(4)}
              <span className="ml-1 text-[10px] text-ink-faint">
                ± {r.rate_mae_se.toFixed(4)}
              </span>
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
    </div>
  )
}
