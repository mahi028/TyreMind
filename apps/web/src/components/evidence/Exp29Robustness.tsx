/**
 * exp29 — the same contest, eleven times, with the generator changed each time.
 *
 * A single synthetic corpus proves that a model wins on that corpus. exp29 asks
 * whether the win is a property of the model or of the corpus, by re-running the
 * whole field across eleven regimes that each break a different assumption:
 * heavy-tailed noise, triple observation noise, a severe cliff, no cliff at all,
 * heavy traffic, every set scrubbed, no set scrubbed, a flat track and a strong
 * one.
 *
 * Eight wins, one tie, two losses. Both losses are on the track-evolution axis,
 * which is not a coincidence and is stated rather than averaged away: track
 * evolution and a uniform shift in every degradation rate are the collinearity
 * this model resolves by assumption, so a generator that sets track evolution to
 * zero or to two and a half seconds is attacking exactly the assumption. Losing
 * there is the model behaving as its own identifiability argument predicts.
 */

import { Caption, Fact, Note } from '../briefing/ui'
import { OURS, type Exp29, type Exp29Regime } from './types'

export function Exp29Robustness({ exp29 }: { exp29?: Exp29 }) {
  if (!exp29?.verdict) {
    return (
      <section className="plate p-5">
        <h2 className="text-[18px] font-semibold text-ink">Does the win survive a different world?</h2>
        <p className="mt-2 text-[12.5px] text-ink-faint">
          Run <span className="num">experiments/exp29_generator_robustness.py</span> to populate
          this.
        </p>
      </section>
    )
  }

  const v = exp29.verdict
  const outcome = (regime: string): 'win' | 'tie' | 'loss' =>
    v.wins.includes(regime) ? 'win' : v.ties.includes(regime) ? 'tie' : 'loss'

  const ordered = [...exp29.per_regime].sort((a, b) => {
    const rank = { win: 0, tie: 1, loss: 2 } as const
    return rank[outcome(a.regime)] - rank[outcome(b.regime)] || a.regime.localeCompare(b.regime)
  })

  return (
    <section className="plate">
      <header className="border-b border-line px-5 py-4">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <h2 className="text-[22px] leading-tight font-semibold tracking-[-0.02em] text-ink">
            {v.wins.length} wins, {v.ties.length} tie, {v.losses.length} losses
          </h2>
          <span className="num text-[11px] text-ink-faint">
            exp29 · {v.n_regimes} generator regimes · {exp29.n_seeds} seeds each
          </span>
        </div>
        <div className="rule-accent mt-3" />
        <p className="mt-3 max-w-[92ch] text-[13.5px] leading-relaxed text-ink-dim">
          The same field, re-run against eleven generators that each break a different assumption.
          Both losses land on the track-evolution axis, which is the one collinearity this model
          resolves by assumption rather than by data &mdash; so a generator that removes track
          evolution entirely, or triples it, is attacking the assumption directly. Losing there is
          the model doing what its own identifiability argument says it will do.
        </p>
      </header>

      <div className="grid grid-cols-3 gap-px border-b border-line bg-line">
        <div className="bg-surface px-5 py-4">
          <Fact label="Regimes won" value={String(v.wins.length)} foot="best rate MAE in the field" />
        </div>
        <div className="bg-surface px-5 py-4">
          <Fact
            label="Tied"
            value={String(v.ties.length)}
            foot="margin inside the standard error"
          />
        </div>
        <div className="bg-surface px-5 py-4">
          <Caption>Lost</Caption>
          <div className="num text-[26px] leading-none font-medium text-alert">
            {v.losses.length}
          </div>
          <div className="mt-1.5 text-[11px] leading-snug text-ink-faint">
            {v.losses.join(', ') || '—'}
          </div>
        </div>
      </div>

      <div className="divide-y divide-line/60">
        {ordered.map((regime, i) => (
          <RegimeRow key={regime.regime} regime={regime} outcome={outcome(regime.regime)} index={i} />
        ))}
      </div>

      <div className="border-t border-line px-5 py-4">
        <Note>
          A margin is a win only when it exceeds its own standard error. The{' '}
          <span className="num">{v.ties.join(', ') || 'tied'}</span> regime is recorded as a tie
          rather than counted into the eight, and the two losses are named rather than described as
          &ldquo;mixed results&rdquo;.
        </Note>
      </div>
    </section>
  )
}

function RegimeRow({
  regime,
  outcome,
  index,
}: {
  regime: Exp29Regime
  outcome: 'win' | 'tie' | 'loss'
  index: number
}) {
  const margin = regime.paired_margin_best_over_runner_up
  const colour =
    outcome === 'win'
      ? 'var(--color-good)'
      : outcome === 'tie'
        ? 'var(--color-medium)'
        : 'var(--color-alert)'

  const overrides = Object.entries(regime.overrides)
  // Our error against the best in the field, on one shared ruler per row. A
  // regime is only interesting for how far apart the two are.
  const worst = Math.max(...regime.table.map((r) => r.rate_mae + 1.96 * r.rate_mae_se))
  const clamp = (v: number) => Math.min(100, Math.max(0, (v / (worst || 1)) * 100))

  return (
    <div
      className="settle grid gap-4 px-5 py-3.5 lg:grid-cols-[minmax(210px,1fr)_1.5fr_minmax(190px,auto)]"
      style={{ ['--i' as string]: index }}
    >
      <div>
        <div className="flex items-center gap-2.5">
          <span
            className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
            style={{ background: colour }}
          />
          <span className="num text-[13px] text-ink">{regime.regime}</span>
          <span
            className="border px-1.5 py-0.5 text-[9.5px] tracking-[0.08em] uppercase"
            style={{ borderColor: colour, color: colour }}
          >
            {outcome}
          </span>
        </div>
        <div className="mt-1 pl-5 text-[11px] text-ink-faint">
          {overrides.length === 0
            ? 'the generator as shipped'
            : overrides.map(([k, val]) => `${k} = ${val}`).join(' · ')}
        </div>
      </div>

      <div className="space-y-1">
        {regime.table.slice(0, 3).map((r) => {
          const ours = r.model === OURS
          const rowColour = ours ? 'var(--color-alert)' : 'var(--color-ink-faint)'
          return (
            <div key={r.model} className="grid grid-cols-[minmax(120px,1fr)_2fr_64px] items-center gap-2">
              <span
                className={`truncate text-[11px] ${ours ? 'text-ink' : 'text-ink-dim'}`}
                title={r.model}
              >
                {r.model}
              </span>
              <div className="relative h-3 bg-raised">
                <div
                  className="beam absolute top-0 bottom-0"
                  style={{
                    left: `${clamp(r.rate_mae - 1.96 * r.rate_mae_se)}%`,
                    width: `${Math.max(
                      clamp(r.rate_mae + 1.96 * r.rate_mae_se) -
                        clamp(r.rate_mae - 1.96 * r.rate_mae_se),
                      0.6,
                    )}%`,
                    ['--beam-color' as string]: rowColour,
                  }}
                />
                <div
                  className="absolute top-0 bottom-0 w-[2px]"
                  style={{ left: `${clamp(r.rate_mae)}%`, background: rowColour }}
                />
              </div>
              <span className={`num text-right text-[10.5px] ${ours ? 'text-ink' : 'text-ink-dim'}`}>
                {r.rate_mae.toFixed(4)}
              </span>
            </div>
          )
        })}
      </div>

      <div className="text-[11.5px] leading-snug">
        <div className="text-ink-dim">
          Won by <span className="text-ink">{regime.winner}</span>
        </div>
        {margin?.margin != null && margin.margin_se != null && (
          <div className="num mt-0.5 text-ink-faint">
            margin {margin.margin.toFixed(4)} ± {margin.margin_se.toFixed(4)}
            <span className="ml-1.5" style={{ color: margin.decisive ? colour : 'var(--color-medium)' }}>
              {margin.decisive ? 'decisive' : 'inside the noise'}
            </span>
          </div>
        )}
        <div className="mt-0.5 text-ink-faint">
          we place{' '}
          <span className="num" style={{ color: regime.tyremind_rank === 1 ? 'var(--color-ink)' : 'var(--color-alert)' }}>
            {regime.tyremind_rank} of {regime.table.length}
          </span>
        </div>
      </div>
    </div>
  )
}
