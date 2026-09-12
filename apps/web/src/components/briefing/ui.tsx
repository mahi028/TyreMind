/**
 * Display primitives for the five-question briefing.
 *
 * These exist separately from `components/primitives.tsx` for one reason: scale.
 * The working views are read at desk distance on a laptop; this screen is read
 * from the back of a room off a projector, so the type runs roughly twice as
 * large and the marks are correspondingly heavier. Sharing one component and
 * branching on a `size` prop was tried and produced a component whose every
 * dimension was a ternary.
 *
 * The rule they all enforce: an estimate is drawn as an interval and annotated
 * with a number, never printed as a number with a spread appended as an
 * afterthought. `Interval` has no mode that omits the band, deliberately -- if
 * a panel wants a bare figure it has to reach past this file to get one.
 */

import { useEffect, useRef, useState, type ReactNode } from 'react'

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

/**
 * One of the five questions.
 *
 * The question is the heading, phrased the way a race engineer says it out
 * loud. The endpoint that answers it is printed in the corner: a judge with the
 * repository open can check that the number on screen came from the API and not
 * from a fixture, and that check should take two seconds, not a code read.
 */
export function Question({
  index,
  question,
  endpoint,
  answer,
  children,
}: {
  index: number
  question: string
  endpoint: string
  answer?: ReactNode
  children: ReactNode
}) {
  const ref = useReveal<HTMLElement>()

  return (
    <section ref={ref} className="reveal border border-line bg-surface">
      <header className="flex flex-wrap items-start gap-x-5 gap-y-2 border-b border-line px-5 py-4">
        <span
          className="num shrink-0 text-[34px] leading-none font-bold text-line-bright tabular-nums"
          aria-hidden
        >
          {String(index).padStart(2, '0')}
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="text-[22px] leading-tight font-semibold tracking-[-0.02em] text-ink sm:text-[26px]">
            {question}
          </h2>
          {answer && (
            <p className="mt-1.5 max-w-[72ch] text-[14px] leading-snug text-ink-dim">{answer}</p>
          )}
        </div>
        <code className="num shrink-0 border border-line px-2 py-1 text-[10.5px] text-ink-faint">
          {endpoint}
        </code>
      </header>
      <div className="p-5">{children}</div>
    </section>
  )
}

/** Reveal-on-scroll. One observer per node, disconnected after it fires once. */
function useReveal<T extends HTMLElement>() {
  const ref = useRef<T>(null)

  useEffect(() => {
    const node = ref.current
    if (!node) return
    if (typeof IntersectionObserver === 'undefined') {
      node.classList.add('seen')
      return
    }
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            entry.target.classList.add('seen')
            observer.disconnect()
          }
        }
      },
      { rootMargin: '0px 0px -12% 0px', threshold: 0.06 },
    )
    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  return ref
}

/** A small caption above a block. Never competes with the figure below it. */
export function Caption({ children }: { children: ReactNode }) {
  return (
    <div className="mb-2 text-[11px] tracking-[0.09em] text-ink-faint uppercase">{children}</div>
  )
}

export function Note({ children }: { children: ReactNode }) {
  return (
    <p className="max-w-[84ch] text-[12.5px] leading-relaxed text-ink-dim">{children}</p>
  )
}

// ---------------------------------------------------------------------------
// The interval
// ---------------------------------------------------------------------------

export interface IntervalProps {
  label: string
  mean: number
  sd: number
  /**
   * The value range the track represents. Supplied by the caller rather than
   * derived from the estimate, because a self-scaled beam is always the same
   * width and therefore says nothing: the point of drawing it is to see this
   * spread against a fixed ruler and against its neighbours.
   */
  domain: [number, number]
  unit?: string
  digits?: number
  colour?: string
  /** Draw a reference line at zero, where the sign of the estimate is a fact. */
  zero?: boolean
  size?: 'hero' | 'md' | 'sm'
  /** Extra line under the interval, for context the number cannot carry. */
  foot?: ReactNode
  /** Suppress the count-up. Used inside lists that re-render on every scrub. */
  still?: boolean
}

const VALUE_CLASS = {
  hero: 'text-[46px] sm:text-[56px]',
  md: 'text-[30px]',
  sm: 'text-[20px]',
} as const

const TRACK_HEIGHT = { hero: 26, md: 18, sm: 13 } as const

/**
 * An estimate, drawn.
 *
 * Three marks, and all three are load-bearing:
 *
 *  - the **beam** spans the 95% interval, so its width is the uncertainty;
 *  - the **tick** is the mean, so the reader can see where inside its own
 *    interval the estimate sits;
 *  - the printed **range** underneath states the endpoints, because a judge
 *    reading a figure off a projector will want to quote it.
 *
 * The beam animates open from the tick rather than fading in. That reads as a
 * spread growing out of a point estimate, which is precisely the claim.
 */
export function Interval({
  label,
  mean,
  sd,
  domain,
  unit,
  digits = 3,
  colour = 'var(--color-alert)',
  zero = false,
  size = 'md',
  foot,
  still = false,
}: IntervalProps) {
  const [lo, hi] = domain
  const span = hi - lo || 1
  const pct = (v: number) => ((v - lo) / span) * 100
  const clamp = (v: number) => Math.min(100, Math.max(0, v))

  const low = mean - 1.96 * sd
  const high = mean + 1.96 * sd
  const left = clamp(pct(low))
  const right = clamp(pct(high))
  const width = Math.max(right - left, 0.7)
  const tick = clamp(pct(mean))
  // The beam opens from wherever the mean sits inside it, not from its middle,
  // so an asymmetric clamp at the edge of the domain still grows from the tick.
  const origin = width > 0 ? `${clamp(((tick - left) / width) * 100)}%` : '50%'

  const shown = useCountUp(mean, still)
  const finite = Number.isFinite(mean)

  return (
    <div>
      <Caption>{label}</Caption>
      <div className="flex items-baseline gap-2">
        <span className={`num leading-none font-medium text-ink ${VALUE_CLASS[size]}`}>
          {finite ? shown.toFixed(digits) : '—'}
        </span>
        <span className="num text-[13px] leading-none text-ink-faint">
          ± {Number.isFinite(sd) ? sd.toFixed(digits) : '—'}
        </span>
        {unit && <span className="text-[12px] leading-none text-ink-faint">{unit}</span>}
      </div>

      <div
        className="relative mt-2.5 w-full bg-raised"
        style={{ height: TRACK_HEIGHT[size] }}
        role="img"
        aria-label={
          finite
            ? `${label}: ${mean.toFixed(digits)} ${unit ?? ''}, 95% interval ${low.toFixed(
                digits,
              )} to ${high.toFixed(digits)}`
            : `${label}: not available`
        }
      >
        {zero && lo < 0 && hi > 0 && (
          <div
            className="absolute top-0 bottom-0 w-px bg-line-bright"
            style={{ left: `${pct(0)}%` }}
          />
        )}
        <div
          className="beam beam-open absolute top-0 bottom-0"
          style={{
            left: `${left}%`,
            width: `${width}%`,
            ['--beam-color' as string]: colour,
            ['--beam-origin' as string]: origin,
          }}
        />
        <div
          className="absolute top-0 bottom-0 w-[2px]"
          style={{ left: `${tick}%`, background: colour }}
        />
      </div>

      <div className="num mt-1.5 flex items-baseline justify-between text-[11.5px] text-ink-faint">
        <span>95% {finite ? low.toFixed(digits) : '—'}</span>
        <span aria-hidden>—</span>
        <span>{finite ? high.toFixed(digits) : '—'}</span>
      </div>
      {foot && <div className="mt-1.5 text-[11.5px] leading-snug text-ink-dim">{foot}</div>}
    </div>
  )
}

/**
 * A compact interval for list rows: label on the left, beam in the middle,
 * value on the right. Used where a dozen of them are stacked and a full
 * `Interval` each would be a wall.
 */
export function IntervalRow({
  label,
  mean,
  sd,
  domain,
  digits = 3,
  colour = 'var(--color-ink-dim)',
  emphasis = false,
}: {
  label: ReactNode
  mean: number
  sd: number
  domain: [number, number]
  digits?: number
  colour?: string
  emphasis?: boolean
}) {
  const [lo, hi] = domain
  const span = hi - lo || 1
  const clamp = (v: number) => Math.min(100, Math.max(0, ((v - lo) / span) * 100))
  const left = clamp(mean - 1.96 * sd)
  const right = clamp(mean + 1.96 * sd)
  const width = Math.max(right - left, 0.7)

  return (
    <div className="grid grid-cols-[minmax(96px,auto)_1fr_auto] items-center gap-3">
      <span
        className={`truncate text-[12.5px] ${emphasis ? 'font-medium text-ink' : 'text-ink-dim'}`}
      >
        {label}
      </span>
      <div className="relative h-3 bg-raised">
        <div
          className="beam absolute top-0 bottom-0"
          style={{
            left: `${left}%`,
            width: `${width}%`,
            ['--beam-color' as string]: colour,
          }}
        />
        <div
          className="absolute top-0 bottom-0 w-[2px]"
          style={{ left: `${clamp(mean)}%`, background: colour }}
        />
      </div>
      <span
        className={`num text-right text-[12.5px] tabular-nums ${
          emphasis ? 'text-ink' : 'text-ink-dim'
        }`}
      >
        {mean.toFixed(digits)}
        <span className="ml-1 text-[10.5px] text-ink-faint">± {sd.toFixed(digits)}</span>
      </span>
    </div>
  )
}

/**
 * A probability, drawn as a filled proportion rather than printed.
 *
 * Probabilities are the one quantity on this screen that does not get an
 * interval, because they already are one -- a bar at 53% is the spread. Printing
 * "0.535" next to it would be the same fact twice.
 */
export function Probability({
  label,
  value,
  tone = 'alert',
  foot,
}: {
  label: string
  value: number
  tone?: 'alert' | 'good' | 'dim'
  foot?: ReactNode
}) {
  const colour =
    tone === 'good'
      ? 'var(--color-good)'
      : tone === 'dim'
        ? 'var(--color-ink-dim)'
        : 'var(--color-alert)'
  const pct = Math.min(100, Math.max(0, value * 100))

  return (
    <div>
      <Caption>{label}</Caption>
      <div className="num text-[30px] leading-none font-medium text-ink">
        {pct.toFixed(pct < 10 ? 1 : 0)}
        <span className="ml-0.5 text-[14px] text-ink-faint">%</span>
      </div>
      <div className="mt-2 h-2.5 w-full bg-raised">
        <div
          className="h-full transition-[width] duration-700 ease-out"
          style={{ width: `${pct}%`, background: colour }}
        />
      </div>
      {foot && <div className="mt-1.5 text-[11.5px] leading-snug text-ink-faint">{foot}</div>}
    </div>
  )
}

/** A fact with no uncertainty attached because it is a count, not an estimate. */
export function Fact({
  label,
  value,
  unit,
  foot,
}: {
  label: string
  value: string
  unit?: string
  foot?: ReactNode
}) {
  return (
    <div>
      <Caption>{label}</Caption>
      <div className="num text-[26px] leading-none font-medium text-ink">
        {value}
        {unit && <span className="ml-1 text-[12px] text-ink-faint">{unit}</span>}
      </div>
      {foot && <div className="mt-1.5 text-[11.5px] leading-snug text-ink-faint">{foot}</div>}
    </div>
  )
}

/** Loud, and only used when the model itself has raised a hand. */
export function Flag({ tone, children }: { tone: 'warn' | 'ok'; children: ReactNode }) {
  const warn = tone === 'warn'
  return (
    <div
      className="flex items-start gap-2.5 border px-3.5 py-2.5 text-[12.5px] leading-snug"
      style={{
        borderColor: warn ? 'color-mix(in oklab, var(--color-alert) 45%, transparent)' : 'var(--color-line)',
        background: warn ? 'color-mix(in oklab, var(--color-alert) 8%, transparent)' : 'transparent',
        color: 'var(--color-ink-dim)',
      }}
    >
      <span
        className="mt-[5px] inline-block h-2 w-2 shrink-0 rounded-full"
        style={{ background: warn ? 'var(--color-alert)' : 'var(--color-good)' }}
      />
      <span>{children}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Motion
// ---------------------------------------------------------------------------

/**
 * Count a figure up to its value on change.
 *
 * Purely presentational, with one functional benefit: when the lap scrubber
 * moves, a number that travels makes it obvious *which* figures on screen
 * depend on the lap and which do not.
 */
function useCountUp(target: number, still: boolean): number {
  const [value, setValue] = useState(still ? target : 0)
  const from = useRef(still ? target : 0)

  useEffect(() => {
    if (still || !Number.isFinite(target)) {
      setValue(target)
      return
    }
    if (
      typeof window === 'undefined' ||
      window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    ) {
      setValue(target)
      return
    }

    const start = performance.now()
    const origin = from.current
    const duration = 520
    let frame = 0

    const step = (now: number) => {
      const t = Math.min(1, (now - start) / duration)
      // Same easing curve as the beam, so the number and its band settle together.
      const eased = 1 - Math.pow(1 - t, 3)
      setValue(origin + (target - origin) * eased)
      if (t < 1) frame = requestAnimationFrame(step)
      else from.current = target
    }

    frame = requestAnimationFrame(step)
    return () => cancelAnimationFrame(frame)
  }, [target, still])

  return value
}
