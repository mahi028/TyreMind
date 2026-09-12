/**
 * exp28 — what the estimate is worth in seconds.
 *
 * This panel carries the only number in the product denominated in the unit a
 * team actually cares about, and it comes with a qualification that is not
 * optional and is not in the small print.
 *
 * **The number.** Against a referee that knows how the race went, calling the
 * final stop off a naive lap-time-versus-tyre-age slope instead of ours costs
 * about 26 seconds and five positions per car per race, on the stops both
 * methods answered, with a standard error small enough that the sign is not in
 * doubt.
 *
 * **The qualification.** Our own value against what the team actually did is
 * +1.03 ± 0.88 s. That interval contains zero. We are as good as professional
 * race strategists, and the data does not say we are better. Showing the 26
 * seconds without showing the 1.03 would be choosing a comparison that flatters
 * us -- the naive estimator is a straw man that no team uses, and the reason it
 * is on screen at all is that it is what a tyre model looks like when nobody
 * has separated fuel from rubber.
 */

import { Caption, Fact, Flag, Note } from '../briefing/ui'
import { OURS, type Exp28 } from './types'

export function Exp28Value({ exp28 }: { exp28?: Exp28 }) {
  if (!exp28?.verdict) {
    return (
      <section className="plate p-5">
        <h2 className="text-[18px] font-semibold text-ink">What is it worth, in seconds?</h2>
        <p className="mt-2 text-[12.5px] text-ink-faint">
          Run <span className="num">experiments/exp28_value_experiment.py</span> to populate this.
        </p>
      </section>
    )
  }

  const v = exp28.verdict
  const a = v.paired_referee_a
  const b = v.paired_referee_b
  const ladder = exp28.referee_a_theil_sen?.all_answered_final_stops ?? []
  const ourRow = ladder.find((r) => r.model === OURS)

  // Is our own value separable from zero at 95%? The answer is no, and the
  // panel is built around saying so rather than around hiding it.
  const ourValue = ourRow?.value_s_per_car_race ?? v.tyremind_value_s_per_car_race
  const ourSe = ourRow?.value_se_per_stop ?? NaN
  const ourSeparable = Number.isFinite(ourSe) && Math.abs(ourValue) > 1.96 * ourSe

  return (
    <section className="plate">
      <header className="border-b border-line px-5 py-4">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <h2 className="text-[22px] leading-tight font-semibold tracking-[-0.02em] text-ink">
            What the estimate is worth, in seconds
          </h2>
          <span className="num text-[11px] text-ink-faint">
            exp28 · {exp28.n_sessions} races · {exp28.n_final_stops} final stops ·{' '}
            {exp28.n_sims.toLocaleString('en-GB')} sims
          </span>
        </div>
        <div className="rule-accent mt-3" />
      </header>

      <div className="grid gap-px bg-line lg:grid-cols-[1.15fr_1fr]">
        {/* --- the money number ------------------------------------------- */}
        <div className="bg-surface px-5 py-5">
          <Caption>Against the naive estimator, per car per race</Caption>
          <div className="flex flex-wrap items-baseline gap-x-5 gap-y-1">
            <span className="num text-[54px] leading-none font-medium text-alert">
              {v.gap_seconds_per_car_race.toFixed(1)}
              <span className="ml-1 text-[18px] text-ink-faint">s</span>
            </span>
            <span className="num text-[30px] leading-none font-medium text-ink">
              {v.gap_positions_per_car_race.toFixed(2)}
              <span className="ml-1 text-[13px] text-ink-faint">positions</span>
            </span>
          </div>

          {/* The interval, drawn. A headline this large without its spread
              beside it would be the one number on screen that does not carry
              its own uncertainty. */}
          <div className="mt-4">
            <SignedInterval
              mean={a.mean_difference_s_per_car_race}
              se={a.se_s_per_car_race}
              unit="s"
            />
            <div className="num mt-1.5 text-[11.5px] text-ink-faint">
              {a.mean_difference_s_per_car_race.toFixed(2)} ± {a.se_s_per_car_race.toFixed(2)} s
              (1 s.e.) over {a.n_paired_stops} paired stops · 95%{' '}
              {(a.mean_difference_s_per_car_race - 1.96 * a.se_s_per_car_race).toFixed(1)} to{' '}
              {(a.mean_difference_s_per_car_race + 1.96 * a.se_s_per_car_race).toFixed(1)}
            </div>
          </div>

          <p className="mt-4 max-w-[58ch] text-[12.5px] leading-relaxed text-ink-dim">
            Positions are converted at the rate <em>measured in these races</em> &mdash;{' '}
            <span className="num text-ink">
              {v.seconds_per_position_measured_median.toFixed(1)} s
            </span>{' '}
            per position, the median across the field &mdash; not at the{' '}
            <span className="num">{exp28.assumed_seconds_per_position} s</span> rule of thumb. At
            the rule of thumb the same gap is worth{' '}
            <span className="num text-ink">
              {(v.gap_seconds_per_car_race / exp28.assumed_seconds_per_position).toFixed(2)}
            </span>{' '}
            positions, and both conversions are shown because the choice of divisor moves the
            headline by 40%.
          </p>

          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <Fact
              label="Referee A · Theil–Sen race fit"
              value={`${a.mean_difference_s_per_car_race.toFixed(1)} s`}
              foot={`± ${a.se_s_per_car_race.toFixed(2)} over ${a.n_paired_stops} stops`}
            />
            <Fact
              label="Referee B · TyreMind race fit"
              value={`${b.mean_difference_s_per_car_race.toFixed(1)} s`}
              foot={`± ${b.se_s_per_car_race.toFixed(2)}. Two independent referees, ${
                v.referees_disagree ? 'and they disagree' : 'and they agree on the sign'
              }.`}
            />
          </div>
        </div>

        {/* --- the qualification ------------------------------------------ */}
        <div className="bg-surface px-5 py-5">
          <Caption>Our own value, against what the team actually did</Caption>
          <div className="flex items-baseline gap-2">
            <span className="num text-[44px] leading-none font-medium text-ink">
              {ourValue >= 0 ? '+' : '−'}
              {Math.abs(ourValue).toFixed(2)}
              <span className="ml-1 text-[15px] text-ink-faint">s</span>
            </span>
            {Number.isFinite(ourSe) && (
              <span className="num text-[15px] text-ink-faint">± {ourSe.toFixed(2)}</span>
            )}
          </div>

          <div className="mt-4">
            <SignedInterval mean={ourValue} se={ourSe} unit="s" colour="var(--color-good)" />
          </div>

          <div className="mt-4">
            <Flag tone="warn">
              {ourSeparable ? (
                <>
                  This interval excludes zero, which would be a claim of improvement over the
                  strategists. Read the numbers, not this sentence.
                </>
              ) : (
                <>
                  <strong className="text-ink">This interval contains zero.</strong> On the stops we
                  answered, our recommendation is not distinguishable from the one a professional
                  race strategist actually made. We are as good as they are. We are not measurably
                  better, and this experiment cannot be used to say otherwise.
                </>
              )}
            </Flag>
          </div>

          <p className="mt-4 max-w-[54ch] text-[12.5px] leading-relaxed text-ink-dim">
            Which is what makes the {v.gap_seconds_per_car_race.toFixed(0)}-second figure
            interpretable rather than impressive. The naive estimator is not a competitor&rsquo;s
            product &mdash; it is what a tyre model degrades into when nobody separates fuel burn
            from rubber, and it is on the screen because that separation is the entire thesis. The
            distance between us and a real pit wall is zero. The distance between us and{' '}
            <em>not doing the separation</em> is {v.gap_seconds_per_car_race.toFixed(0)} seconds.
          </p>
        </div>
      </div>

      {/* --- the whole ladder --------------------------------------------- */}
      {ladder.length > 0 && (
        <div className="border-t border-line px-5 py-5">
          <Caption>Every model, valued the same way</Caption>
          <ValueLadder rows={ladder} />
        </div>
      )}

      {exp28.deviations?.length > 0 && (
        <div className="border-t border-line px-5 py-4">
          <Caption>Deviations from the pre-registration</Caption>
          <ul className="mt-1 space-y-1.5">
            {exp28.deviations.map((d, i) => (
              <li
                key={i}
                className="border-l-2 border-line pl-3 text-[11.5px] leading-relaxed text-ink-faint"
              >
                {d}
              </li>
            ))}
          </ul>
          <Note>
            Listed because the experiment was pre-registered before it was run. Two of these are in
            our disfavour: models are refitted on expanding prefixes rather than on the whole race,
            and the headline counts only final stops, where the simulator&rsquo;s one-stop pricing
            is actually correct.
          </Note>
        </div>
      )}
    </section>
  )
}

/**
 * A value, an interval, and a zero line.
 *
 * The zero line is the load-bearing mark. Every row in this experiment is a
 * signed difference against a referee, so whether the interval crosses zero is
 * the finding, and a bar chart without that line would hide it.
 */
function SignedInterval({
  mean,
  se,
  unit,
  colour = 'var(--color-alert)',
}: {
  mean: number
  se: number
  unit: string
  colour?: string
}) {
  // The axis is derived from the estimate rather than fixed, and always
  // includes zero: a hand-picked domain can clip an interval on a rerun, and
  // clipping the end of an interval is exactly the failure this whole product
  // is arguing against.
  const spreadFull = Number.isFinite(se) ? 1.96 * se : 0
  const rawLo = Math.min(0, mean - spreadFull)
  const rawHi = Math.max(0, mean + spreadFull)
  const pad = (rawHi - rawLo) * 0.16 || 1
  const lo = rawLo - pad
  const hi = rawHi + pad
  const span = hi - lo || 1
  const clamp = (v: number) => Math.min(100, Math.max(0, ((v - lo) / span) * 100))
  const spread = Number.isFinite(se) ? 1.96 * se : 0
  const left = clamp(mean - spread)
  const right = clamp(mean + spread)
  const crossesZero = mean - spread <= 0 && mean + spread >= 0

  return (
    <div>
      <div className="relative h-7 bg-raised">
        <div
          className="absolute top-0 bottom-0 w-px"
          style={{ left: `${clamp(0)}%`, background: 'var(--color-line-bright)' }}
        />
        <div
          className="beam beam-open absolute top-0 bottom-0"
          style={{
            left: `${left}%`,
            width: `${Math.max(right - left, 0.8)}%`,
            ['--beam-color' as string]: crossesZero ? 'var(--color-medium)' : colour,
          }}
        />
        <div
          className="absolute top-0 bottom-0 w-[2px]"
          style={{
            left: `${clamp(mean)}%`,
            background: crossesZero ? 'var(--color-medium)' : colour,
          }}
        />
      </div>
      <div className="num mt-1 flex justify-between text-[10px] text-ink-faint">
        <span>
          {lo.toFixed(1)} {unit}
        </span>
        <span style={{ color: 'var(--color-line-bright)' }}>0</span>
        <span>
          {hi.toFixed(1)} {unit}
        </span>
      </div>
    </div>
  )
}

function ValueLadder({
  rows,
}: {
  rows: {
    model: string
    n_stops: number
    value_s_per_car_race: number
    value_se_per_stop: number
    mean_abs_error_laps: number
    answer_rate: number
  }[]
}) {
  const sorted = [...rows].sort((a, b) => b.value_s_per_car_race - a.value_s_per_car_race)
  const bounds = sorted.flatMap((r) => [
    r.value_s_per_car_race - 1.96 * r.value_se_per_stop,
    r.value_s_per_car_race + 1.96 * r.value_se_per_stop,
  ])
  const lo = Math.min(...bounds, 0)
  const hi = Math.max(...bounds, 0)
  const span = hi - lo || 1
  const clamp = (v: number) => Math.min(100, Math.max(0, ((v - lo) / span) * 100))

  return (
    <div className="mt-2">
      <div className="grid grid-cols-[minmax(150px,1.2fr)_2.2fr_minmax(80px,auto)_minmax(84px,auto)_minmax(72px,auto)] gap-3 pb-1.5 text-[10.5px] text-ink-faint">
        <span>model</span>
        <span>value vs the referee, seconds per car per race</span>
        <span className="text-right">value</span>
        <span className="text-right">lap error</span>
        <span className="text-right">answered</span>
      </div>
      {sorted.map((r, i) => {
        const ours = r.model === OURS
        const spread = 1.96 * r.value_se_per_stop
        const crossesZero =
          r.value_s_per_car_race - spread <= 0 && r.value_s_per_car_race + spread >= 0
        const colour = crossesZero
          ? 'var(--color-medium)'
          : r.value_s_per_car_race > 0
            ? 'var(--color-good)'
            : 'var(--color-alert)'
        return (
          <div
            key={r.model}
            className="settle grid grid-cols-[minmax(150px,1.2fr)_2.2fr_minmax(80px,auto)_minmax(84px,auto)_minmax(72px,auto)] items-center gap-3 border-t border-line/50 py-1.5"
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
                className="absolute top-0 bottom-0 w-px bg-line-bright"
                style={{ left: `${clamp(0)}%` }}
              />
              <div
                className="beam absolute top-0.5 bottom-0.5"
                style={{
                  left: `${clamp(r.value_s_per_car_race - spread)}%`,
                  width: `${Math.max(
                    clamp(r.value_s_per_car_race + spread) -
                      clamp(r.value_s_per_car_race - spread),
                    0.6,
                  )}%`,
                  ['--beam-color' as string]: colour,
                }}
              />
              <div
                className="absolute top-0 bottom-0 w-[2px]"
                style={{ left: `${clamp(r.value_s_per_car_race)}%`, background: colour }}
              />
            </div>
            <span className="num text-right text-[12px]" style={{ color: colour }}>
              {r.value_s_per_car_race >= 0 ? '+' : '−'}
              {Math.abs(r.value_s_per_car_race).toFixed(1)}
            </span>
            <span className="num text-right text-[11.5px] text-ink-dim">
              {r.mean_abs_error_laps.toFixed(1)} L
            </span>
            <span className="num text-right text-[11.5px] text-ink-faint">
              {(r.answer_rate * 100).toFixed(0)}%
            </span>
          </div>
        )
      })}
      <p className="mt-3 max-w-[92ch] text-[11.5px] leading-relaxed text-ink-faint">
        Amber bands cross zero: those models are not separable from the strategist&rsquo;s own
        call, in either direction. The <span className="num">answered</span> column matters as much
        as the value &mdash; a model that answers 38% of stops is being scored on the ones it found
        easy, which is why the headline comparison is paired stop-for-stop rather than taken from
        these column averages.
      </p>
    </div>
  )
}
