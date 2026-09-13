/**
 * The business case: the gap in motorsport this fills, and how the industry
 * takes it up.
 *
 * Not a closing flourish. It is the argument that a real market exists and that
 * adoption is cheap, which is a different claim from "our model is accurate" and
 * needs its own evidence. The shape is: here is who currently cannot do this at
 * all, here is what it costs them to start, here is the order they come in.
 *
 * Written to be *read across a room*, not read closely. The rest of the
 * application argues by evidence and can afford paragraphs; this section argues
 * by contrast and cannot. So the rules here are numbers first, at most one line
 * of prose under each, and every figure either measured by one of our own
 * experiments or a plain property of the software.
 *
 * **Nothing here is a market projection.** Revenue multiples, addressable-market
 * figures and prize-money estimates are the easiest numbers in the world to put
 * on a slide and the easiest for a panel to puncture, and this project's whole
 * position is that its numbers survive being checked. So the value claim is the
 * one exp28 measured -- seconds and positions per car per race against the
 * method teams would otherwise use -- and the cost claims are facts about the
 * program: it fits in twelve seconds, it needs no GPU, and it needs no data
 * anyone has to buy.
 *
 * The comparison that does the selling is the asymmetry: the cost column is
 * essentially zero and the value column is measured in positions.
 */

const CORNERSTONE = [
  {
    value: '25.9',
    unit: 'seconds',
    line: 'lost per car, per race, by using the standard method instead of ours',
    tone: 'var(--color-alert)',
  },
  {
    value: '5.3',
    unit: 'places',
    line: 'what those seconds are worth on track, at the rate measured in these races',
    tone: 'var(--color-alert)',
  },
  {
    value: '12',
    unit: 'seconds',
    line: 'to analyse a full race session, on an ordinary laptop',
    tone: 'var(--color-good)',
  },
  {
    value: '0',
    unit: 'to run',
    line: 'no GPU, no cloud bill, no data to license, no internet',
    tone: 'var(--color-good)',
  },
]

const COST_ROWS: { label: string; ours: string; usual: string; good: boolean }[] = [
  { label: 'Hardware', ours: 'Any laptop', usual: 'GPU server', good: true },
  { label: 'Time to analyse a session', ours: '12 seconds', usual: 'Hours of training', good: true },
  { label: 'Data you must buy', ours: 'None — public timing', usual: 'Licensed telemetry', good: true },
  { label: 'New circuit', ours: 'Works immediately', usual: 'Retrain the model', good: true },
  { label: 'New car regulations', ours: 'Works the same day', usual: 'Rebuild the dataset', good: true },
  { label: 'Internet during a session', ours: 'Not required', usual: 'Cloud round-trip', good: true },
]

const BUYERS = [
  {
    tier: '01',
    who: 'Junior formulae and club racing',
    why: 'They race on the same public timing feed and have no tyre modelling department. Nothing to integrate, nothing to buy.',
    proof: 'Needs 4 data fields',
    accent: 'var(--color-alert)',
  },
  {
    tier: '02',
    who: 'Formula 1 and endurance teams',
    why: 'They already hold far more data than we ask for, so it fits their feed on day one. What they gain is the calibration layer.',
    proof: 'Fits any richer feed',
    accent: 'var(--color-soft)',
  },
  {
    tier: '03',
    who: 'Anything that wears out while it works',
    why: 'Aviation, wind, rail, mining. We proved the transfer by running the same engine, unchanged, on NASA jet engines.',
    proof: '100 of 100 engines scored',
    accent: 'var(--color-good)',
  },
]

export function BusinessCase() {
  return (
    <section
      className="border border-line bg-surface"
      style={{ borderTopWidth: 3, borderTopColor: 'var(--color-alert)' }}
    >
      <div className="border-b border-line px-6 py-5">
        <div className="text-[11px] tracking-[0.22em] text-alert uppercase">
          The business
        </div>
        <h2 className="mt-2 max-w-[30ch] text-[28px] leading-tight font-bold tracking-[-0.02em] text-ink sm:text-[34px]">
          Ten teams in the world can do this today. Everybody else races blind.
        </h2>
        <p className="mt-3 max-w-[78ch] text-[14px] leading-relaxed text-ink-dim">
          Tyre strategy decides races at every level of motorsport. Modelling it properly needs a
          department, licensed data and a lot of money, so only the top of Formula 1 has one.
          Everyone below runs on instinct and a stopwatch.{' '}
          <strong className="text-ink">
            We put the same capability on a laptop, using timing data that is already public.
          </strong>
        </p>
      </div>

      {/* ── the gap, as two columns ──────────────────────────────────────── */}
      <div className="grid gap-px bg-line sm:grid-cols-2">
        <div className="bg-surface px-6 py-5">
          <div className="text-[11px] tracking-[0.16em] uppercase" style={{ color: 'var(--color-ink-faint)' }}>
            Who can do this now
          </div>
          <div className="mt-2.5 flex items-baseline gap-3">
            <span className="num text-[40px] leading-none font-semibold text-ink">~10</span>
            <span className="text-[13px] text-ink-dim">Formula 1 teams</span>
          </div>
          <p className="mt-3 max-w-[46ch] text-[12.5px] leading-relaxed text-ink-faint">
            In-house tyre models, private telemetry, dedicated staff. The maths exists, it is just
            locked at the top of one championship.
          </p>
        </div>
        <div
          className="bg-surface px-6 py-5"
          style={{ borderLeft: '3px solid var(--color-alert)' }}
        >
          <div className="text-[11px] tracking-[0.16em] text-alert uppercase">
            Who cannot, and races anyway
          </div>
          <div className="mt-2.5 flex items-baseline gap-3">
            <span
              className="num text-[40px] leading-none font-semibold"
              style={{ color: 'var(--color-alert)' }}
            >
              Everyone
            </span>
            <span className="text-[13px] text-ink-dim">else</span>
          </div>
          <p className="mt-3 max-w-[46ch] text-[12.5px] leading-relaxed text-ink-dim">
            F2, F3, F4, regional single-seaters, GT and endurance privateers, club racing. Same
            compounds, same pit decisions, same public timing feed. No model at all.
          </p>
        </div>
      </div>

      {/* ── the four numbers ─────────────────────────────────────────────── */}
      <div className="grid gap-px bg-line sm:grid-cols-2 xl:grid-cols-4">
        {CORNERSTONE.map((c) => (
          <div key={c.line} className="bg-surface px-5 py-6">
            <div className="flex items-baseline gap-2">
              <span
                className="num text-[44px] leading-none font-semibold tracking-[-0.03em]"
                style={{ color: c.tone }}
              >
                {c.value}
              </span>
              <span className="text-[13px] text-ink-faint">{c.unit}</span>
            </div>
            <p className="mt-3 text-[12.5px] leading-snug text-ink-dim">{c.line}</p>
          </div>
        ))}
      </div>

      <div className="grid gap-px bg-line xl:grid-cols-[1fr_1fr]">
        {/* ── cost comparison ───────────────────────────────────────────── */}
        <div className="bg-surface px-6 py-5">
          <div className="text-[11px] tracking-[0.16em] text-ink-faint uppercase">
            What it takes to run
          </div>
          <div className="mt-4">
            <div className="grid grid-cols-[1fr_auto_auto] gap-x-4 pb-2 text-[10px] tracking-wider text-ink-faint uppercase">
              <span />
              <span className="text-right">TyreMind</span>
              <span className="w-[110px] text-right">A trained model</span>
            </div>
            {COST_ROWS.map((r) => (
              <div
                key={r.label}
                className="grid grid-cols-[1fr_auto_auto] items-center gap-x-4 border-t border-line/60 py-2.5"
              >
                <span className="text-[12.5px] text-ink-dim">{r.label}</span>
                <span
                  className="text-right text-[12.5px] font-medium"
                  style={{ color: 'var(--color-good)' }}
                >
                  {r.ours}
                </span>
                <span className="w-[110px] text-right text-[12px] text-ink-faint">{r.usual}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── adoption ladder ───────────────────────────────────────────── */}
        <div className="bg-surface px-6 py-5">
          <div className="text-[11px] tracking-[0.16em] text-ink-faint uppercase">
            Who adopts it, in order
          </div>
          <div className="mt-4 space-y-3">
            {BUYERS.map((b) => (
              <div
                key={b.tier}
                className="border border-line bg-raised/30 px-4 py-3.5"
                style={{ borderLeftWidth: 3, borderLeftColor: b.accent }}
              >
                <div className="flex items-baseline justify-between gap-3">
                  <span className="flex items-baseline gap-2.5">
                    <span className="num text-[12px]" style={{ color: b.accent }}>
                      {b.tier}
                    </span>
                    <span className="text-[13.5px] font-semibold text-ink">{b.who}</span>
                  </span>
                  <span
                    className="num shrink-0 border px-1.5 py-0.5 text-[9.5px] tracking-wide"
                    style={{ borderColor: b.accent, color: b.accent }}
                  >
                    {b.proof}
                  </span>
                </div>
                <p className="mt-1.5 text-[12px] leading-relaxed text-ink-dim">{b.why}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── how little it takes to switch on ─────────────────────────────── */}
      <div className="border-t border-line px-6 py-5">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <div className="text-[11px] tracking-[0.16em] text-ink-faint uppercase">
            Everything we need from a customer
          </div>
          <span className="text-[12px] text-ink-faint">
            They already have all four. There is no integration project.
          </span>
        </div>
        <div className="mt-3.5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {[
            ['Lap time', 'Every timing screen has it'],
            ['Tyre type', 'Shown on every broadcast'],
            ['Tyre age', 'Laps on the current set'],
            ['Lap number', 'Where they are in the race'],
          ].map(([field, note]) => (
            <div key={field} className="border border-line bg-raised/40 px-4 py-3">
              <div className="text-[14px] font-semibold text-ink">{field}</div>
              <div className="mt-1 text-[11.5px] text-ink-faint">{note}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── how it reaches a team ────────────────────────────────────────── */}
      <div className="border-t border-line px-6 py-5">
        <div className="text-[11px] tracking-[0.16em] text-ink-faint uppercase">
          How a team runs it
        </div>
        <div className="mt-3.5 grid gap-3 lg:grid-cols-3">
          {[
            {
              step: 'Install',
              line: 'It runs on the team&rsquo;s own machine, in the garage. Nothing to provision.',
              stat: 'One command',
            },
            {
              step: 'Point it at the session',
              line: 'The public timing feed is enough. A team with its own telemetry gives us more than we need.',
              stat: '4 fields',
            },
            {
              step: 'Nothing leaves the garage',
              line: 'No cloud, no upload, no vendor sees a team&rsquo;s data. In motorsport that is not a feature, it is the condition of sale.',
              stat: 'Fully offline',
            },
          ].map((b) => (
            <div
              key={b.step}
              className="border border-line bg-raised/30 px-4 py-3.5"
              style={{ borderTop: '2px solid var(--color-fuel)' }}
            >
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-[13.5px] font-semibold text-ink">{b.step}</span>
                <span className="num text-[11px]" style={{ color: 'var(--color-fuel)' }}>
                  {b.stat}
                </span>
              </div>
              <p
                className="mt-1.5 text-[12px] leading-relaxed text-ink-dim"
                dangerouslySetInnerHTML={{ __html: b.line }}
              />
            </div>
          ))}
        </div>
      </div>

      {/* ── what the industry gets out of it ─────────────────────────────── */}
      <div
        className="border-t border-line px-6 py-6"
        style={{ background: 'color-mix(in oklab, var(--color-alert) 6%, transparent)' }}
      >
        <p className="max-w-[80ch] text-[17px] leading-snug font-semibold text-ink">
          Every series below Formula 1 makes the same pit decisions on worse information. The
          capability already exists. It has never been cheap enough to reach them.
        </p>
        <p className="mt-2.5 max-w-[80ch] text-[12.5px] leading-relaxed text-ink-dim">
          The 25.9 seconds is measured against the method teams use today, across real races, and
          converted to places at the rate we measured in those same races. Our lead over a
          professional strategist&rsquo;s own call is smaller and we report that too, on the
          evidence screen. We would rather you checked.
        </p>
      </div>
    </section>
  )
}
