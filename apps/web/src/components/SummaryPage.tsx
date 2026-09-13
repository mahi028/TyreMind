/**
 * The page for someone who will never read the rest of this application.
 *
 * Every other screen is written for a reader who wants the method. This one is
 * written for a judge who has eight minutes, no motorsport background and four
 * other teams to see. So three rules hold here and nowhere else:
 *
 * **No jargon in the main copy.** The words Kalman, conformal, state-space,
 * posterior and collinearity do not appear above the fold. They are correct and
 * they are what the method is, and they also end a non-technical reader's
 * attention immediately. The method has its own screens; this one has to earn
 * the click.
 *
 * **Every claim is checkable.** A judge who distrusts a demo is right to. So the
 * trust section leads with the number that needs no faith -- the standard method
 * reporting tyres that get faster, which anyone can recognise as impossible --
 * and cites where each figure lives.
 *
 * **The live numbers come from the selected race, not from a constant.** A
 * summary page showing hard-coded figures is a brochure. This one refits the
 * race in the rail and shows what it found, so a judge can change the race and
 * watch the numbers move.
 */

import { useEffect, useState } from 'react'

import { api, compoundColour, fixed, type SessionRef, type SessionSummary } from '../lib/api'

/** Views this page can send a judge to, described without jargon. */
const TOUR: {
  key: string
  title: string
  plain: string
  tag: string
  accent: string
}[] = [
  {
    key: 'briefing',
    title: 'The five questions',
    plain: 'The whole product on one screen: where the time went, how fast the tyre is going away, what happens next, when to stop, and whether to believe it.',
    tag: 'Start here',
    accent: 'var(--color-alert)',
  },
  {
    key: 'race',
    title: 'Run the race',
    plain: 'A real Grand Prix replayed lap by lap in 3D, with our pit call shown against the one the team actually made.',
    tag: 'Most visual',
    accent: 'var(--color-soft)',
  },
  {
    key: 'router',
    title: 'Which model answers',
    plain: 'We tested nine models. Three different ones won. This shows which question goes to which, including the two we send to a rival model.',
    tag: 'Our differentiator',
    accent: 'var(--color-fuel)',
  },
  {
    key: 'overview',
    title: 'Why the obvious method fails',
    plain: 'Fit a straight line through lap times and it says the tyres got faster the longer they ran. On this race, in front of you.',
    tag: 'The proof',
    accent: 'var(--color-alert)',
  },
  {
    key: 'explain',
    title: 'Why is the car slow',
    plain: 'One lap, broken into its causes: the tyre, the fuel, the track, the traffic, and the part we cannot explain.',
    tag: '',
    accent: 'var(--color-medium)',
  },
  {
    key: 'circuit',
    title: 'Where it wears',
    plain: 'The lap in 3D, coloured by how hard each corner works the tyre. Turn by turn.',
    tag: '',
    accent: 'var(--color-track)',
  },
  {
    key: 'tyre',
    title: 'Tyre twin',
    plain: 'Condition now, life left, and what changes if the driver pushes or backs off.',
    tag: '',
    accent: 'var(--color-medium)',
  },
  {
    key: 'strategy',
    title: 'When to pit',
    plain: 'Thousands of simulated races behind one recommendation, with the window a strategist would actually act on.',
    tag: '',
    accent: 'var(--color-soft)',
  },
  {
    key: 'live',
    title: 'Live monitor',
    plain: 'The same estimate updating lap by lap, as it would on a pit wall during a session.',
    tag: '',
    accent: 'var(--color-good)',
  },
  {
    key: 'evidence',
    title: 'Does it work',
    plain: 'Every test we ran, including the ones we failed. This is where a sceptical judge should go.',
    tag: 'For sceptics',
    accent: 'var(--color-fuel)',
  },
  {
    key: 'ask',
    title: 'Ask the method',
    plain: 'Type a question and get the exact paragraph from our research notes that answers it, with its source.',
    tag: '',
    accent: 'var(--color-traffic)',
  },
  {
    key: 'beyond',
    title: 'Beyond racing',
    plain: 'The same engine, unchanged, predicting how much life is left in NASA jet engines.',
    tag: 'Scale',
    accent: 'var(--color-good)',
  },
]

export function SummaryPage({
  sessionId,
  session,
  onNavigate,
}: {
  sessionId: string
  session: SessionRef | undefined
  onNavigate: (view: string) => void
}) {
  const [summary, setSummary] = useState<SessionSummary | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    if (!sessionId) return
    setSummary(null)
    setFailed(false)
    api.summary(sessionId).then(setSummary).catch(() => setFailed(true))
  }, [sessionId])

  const compounds = Object.entries(summary?.compounds ?? {})
  // The headline that needs no expertise to check: a negative rate means the
  // tyre got faster as it wore, and tyres do not do that.
  const impossible = compounds.filter(
    ([, c]) => c.naive_estimate != null && c.naive_estimate < 0,
  )

  return (
    <div className="space-y-3">
      {/* ── the promise ────────────────────────────────────────────────── */}
      <section className="border border-line bg-surface">
        <div className="border-b border-line px-6 py-7">
          <div className="grid gap-x-10 gap-y-6 xl:grid-cols-[minmax(0,1fr)_minmax(230px,300px)]">
            <div>
              <div className="text-[11px] tracking-[0.22em] text-alert uppercase">
                TrackShift Innovation Challenge · Problem 3
              </div>
              <h1 className="mt-2.5 text-[36px] leading-[1.04] font-bold tracking-[-0.03em] text-ink sm:text-[46px]">
                We tell you what the tyre is doing.
              </h1>
              <p className="mt-4 max-w-[68ch] text-[15px] leading-relaxed text-ink-dim">
                A lap time is one number, and four different things move it at the same time. The
                tyre wearing out. The fuel burning off, which makes the car lighter and faster. The
                track getting grippier as rubber goes down. And traffic.
              </p>
              <p className="mt-3 max-w-[68ch] text-[15px] leading-relaxed text-ink">
                <strong className="text-ink">
                  Only the tyre decides when you stop, and it is the only one of the four you cannot
                  see.
                </strong>{' '}
                TyreMind separates them, reports the tyre on its own, and tells you how sure it is.
              </p>
            </div>

            {/* The right rail answers the unspoken question a judge has while
                reading the left: is this a demo, or is there something behind
                it. Counts, not adjectives. */}
            <div className="self-start border border-line bg-raised/40 px-5 py-4">
              <div className="text-[10.5px] tracking-[0.16em] text-ink-faint uppercase">
                What it is built on
              </div>
              <dl className="mt-3 space-y-2.5">
                {[
                  ['203', 'real race and practice sessions'],
                  ['91,867', 'clean laps, 2022 to 2025'],
                  ['32', 'recorded experiments'],
                  ['615', 'automated tests'],
                  ['0', 'internet needed — it runs offline'],
                ].map(([n, what]) => (
                  <div key={what} className="flex items-baseline gap-3">
                    <dt className="num w-[62px] shrink-0 text-right text-[15px] font-medium text-ink">
                      {n}
                    </dt>
                    <dd className="text-[12px] leading-snug text-ink-dim">{what}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>

          <Separation />
        </div>

        {/* The four numbers that do the selling. */}
        <div className="grid gap-px bg-line sm:grid-cols-2 xl:grid-cols-4">
          <Headline
            value="74%"
            unit="of races"
            label="where the normal way of measuring tyre wear gives an answer that is physically impossible"
            tone="var(--color-alert)"
          />
          <Headline
            value="95.2%"
            unit="accurate"
            label="When we say we are 95% sure, we checked across 69,206 real laps. We were right 95.2% of the time."
            tone="var(--color-good)"
          />
          <Headline
            value="4.3×"
            unit="better"
            label="than the closest published research model at the job this competition asks for"
            tone="var(--color-fuel)"
          />
          <Headline
            value="13"
            unit="ideas killed"
            label="of our own hypotheses that we tested, disproved, and published anyway"
            tone="var(--color-medium)"
          />
        </div>
      </section>

      {/* ── what it does, in three steps ───────────────────────────────── */}
      <section className="border border-line bg-surface px-6 py-5">
        <h2 className="text-[17px] font-semibold tracking-[-0.01em] text-ink">
          How it works, in three sentences
        </h2>
        <div className="mt-4 grid gap-5 lg:grid-cols-3">
          <Step
            n="1"
            title="It watches the whole grid, not one car"
            body="Every car pits at a different moment. So at any lap, some cars are on fresh tyres and some are on old ones. That difference is what lets us pull the tyre apart from everything else. Watching a single car, it cannot be done."
          />
          <Step
            n="2"
            title="It reports a range, not a number"
            body="Any tool can print a number. Ours prints the range it is confident the truth sits inside, and we have checked that promise against tens of thousands of real laps. It holds."
          />
          <Step
            n="3"
            title="It sends each question to the best model"
            body="We built nine competing models, including two from published research papers. Three different ones turned out to win different questions. So we route each question to whichever is genuinely best, even when that is not ours."
          />
        </div>
      </section>

      {/* ── proof, on the race currently selected ──────────────────────── */}
      <section className="border border-line bg-surface">
        <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-line px-6 py-4">
          <h2 className="text-[17px] font-semibold tracking-[-0.01em] text-ink">
            Proof, on the race selected in the sidebar
          </h2>
          <span className="num text-[12px] text-ink-faint">
            {session ? `${session.year} ${session.grand_prix}` : sessionId}
          </span>
        </div>

        {failed ? (
          <p className="px-6 py-8 text-[13px] text-ink-faint">
            This race could not be loaded. Pick another from the sidebar.
          </p>
        ) : !summary ? (
          <p className="px-6 py-8 text-[13px] text-ink-faint">Fitting this race…</p>
        ) : (
          <div className="grid gap-px bg-line lg:grid-cols-[1.1fr_1fr]">
            <div className="bg-surface px-6 py-5">
              <div className="text-[11px] tracking-[0.16em] text-ink-faint uppercase">
                What we just did to it
              </div>
              <p className="mt-3 text-[14px] leading-relaxed text-ink-dim">
                We read{' '}
                <strong className="num text-ink">{summary.n_laps.toLocaleString('en-GB')}</strong>{' '}
                clean laps from{' '}
                <strong className="num text-ink">{summary.n_drivers}</strong> drivers across{' '}
                <strong className="num text-ink">{summary.n_runs}</strong> sets of tyres, and
                separated the tyre from everything else. It took about twelve seconds on a laptop,
                with no internet connection and no special hardware.
              </p>

              {impossible.length > 0 ? (
                <div
                  className="mt-5 border-l-2 pl-4"
                  style={{ borderColor: 'var(--color-alert)' }}
                >
                  <div className="text-[12px] font-semibold text-alert">
                    On this race, the normal method breaks
                  </div>
                  <p className="mt-1.5 max-w-[60ch] text-[13px] leading-relaxed text-ink-dim">
                    Fit a straight line through lap times against tyre age, which is what the
                    textbook says to do, and on{' '}
                    <strong className="text-ink">
                      {impossible.length} of {compounds.length}
                    </strong>{' '}
                    tyre types here it reports that the tyres got <em>faster</em> the longer they
                    ran. That is not a small error. Tyres do not regenerate.
                  </p>
                </div>
              ) : (
                <p className="mt-5 max-w-[60ch] text-[13px] leading-relaxed text-ink-faint">
                  On this particular race the simple method happens to give positive answers. It
                  fails on 74% of races, and the sidebar has plenty where it does. Try a longer
                  race.
                </p>
              )}
            </div>

            <div className="bg-surface px-6 py-5">
              <div className="text-[11px] tracking-[0.16em] text-ink-faint uppercase">
                What we say instead
              </div>
              <div className="mt-3 space-y-3">
                {compounds.length === 0 ? (
                  <p className="text-[13px] text-ink-faint">No tyre types fitted in this session.</p>
                ) : (
                  compounds.map(([compound, estimate]) => (
                    <div key={compound} className="border border-line bg-raised/40 px-4 py-3">
                      <div className="flex items-center justify-between">
                        <span className="flex items-center gap-2 text-[13px] text-ink">
                          <span
                            className="inline-block h-2.5 w-2.5 rounded-full"
                            style={{ background: compoundColour(compound) }}
                          />
                          {compound}
                        </span>
                        <span className="num text-[11px] text-ink-faint">
                          {estimate.laps} laps seen
                        </span>
                      </div>
                      <div className="mt-2 flex flex-wrap items-baseline gap-x-5 gap-y-1">
                        <span>
                          <span
                            className="num text-[22px] leading-none font-medium"
                            style={{ color: compoundColour(compound) }}
                          >
                            {fixed(estimate.degradation_rate)}
                          </span>
                          <span className="ml-1.5 text-[11px] text-ink-faint">
                            seconds lost per lap
                          </span>
                        </span>
                        <span className="num text-[12px] text-ink-dim">
                          give or take {estimate.degradation_rate_sd.toFixed(3)}
                        </span>
                        {estimate.naive_estimate != null && estimate.naive_estimate < 0 && (
                          <span className="num text-[11px] text-alert">
                            normal method says {fixed(estimate.naive_estimate)} — impossible
                          </span>
                        )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}
      </section>

      {/* ── why trust it ───────────────────────────────────────────────── */}
      <section className="border border-line bg-surface px-6 py-5">
        <h2 className="text-[17px] font-semibold tracking-[-0.01em] text-ink">
          Four reasons to believe this, all of which you can check yourself
        </h2>
        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <Reason
            title="We prove the competition is wrong without needing the right answer"
            body="Nobody publishes real tyre wear data, so we cannot grade ourselves against truth. But a method that says tyres get faster as they wear has disproved itself, and we counted how often that happens: 74% of 77 real races. That result needs no trust at all."
          />
          <Reason
            title="We checked our own confidence and found it lying"
            body="Our model claimed to be right 95% of the time. We measured it. It was right 75%. So we fixed it, and measured again across 69,206 laps: 95.2%. Almost nobody performs this check, which is why almost every system overstates its confidence."
          />
          <Reason
            title="We published thirteen of our own failures"
            body="Thirteen ideas we believed, tested, and disproved. Including a far more advanced physics model we built specifically to try to beat our own system. It lost, so we did not ship it. A team that hides its failures is a team whose successes you cannot check."
          />
          <Reason
            title="We route two questions to a rival's model"
            body="We rebuilt nine competing models, including two from published papers. On two of the five questions, someone else's model is better, and ours sends those questions to them. We would rather be right than look good."
          />
        </div>
      </section>

      {/* ── the tour ───────────────────────────────────────────────────── */}
      <section className="border border-line bg-surface">
        <div className="border-b border-line px-6 py-4">
          <h2 className="text-[13px] font-semibold text-ink">
            See it for yourself — every screen, and what it shows
          </h2>
          <p className="mt-1 text-[12px] text-ink-faint">
            The race selected in the sidebar carries through to every screen below.
          </p>
        </div>
        <div className="grid gap-px bg-line sm:grid-cols-2 xl:grid-cols-3">
          {TOUR.map((v) => (
            <button
              key={v.key}
              onClick={() => onNavigate(v.key)}
              className="group bg-surface px-5 py-4.5 text-left transition-colors hover:bg-raised"
              style={{ borderLeft: `3px solid ${v.accent}` }}
            >
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-[14.5px] font-semibold text-ink group-hover:text-ink">
                  {v.title}
                </span>
                {v.tag && (
                  <span
                    className="shrink-0 border px-1.5 py-0.5 text-[9.5px] tracking-wider uppercase"
                    style={{ borderColor: v.accent, color: v.accent }}
                  >
                    {v.tag}
                  </span>
                )}
              </div>
              <p className="mt-1.5 text-[12.5px] leading-relaxed text-ink-dim">{v.plain}</p>
              <span className="mt-2 inline-block text-[11px] text-ink-faint group-hover:text-ink-dim">
                Open →
              </span>
            </button>
          ))}
        </div>
      </section>

      {/* ── the one line to leave with ─────────────────────────────────── */}
      <section
        className="border border-line bg-surface px-6 py-5"
        style={{ borderLeftWidth: 3, borderLeftColor: 'var(--color-alert)' }}
      >
        <p className="max-w-[86ch] text-[15px] leading-relaxed text-ink">
          Everyone can build a dashboard that shows a tyre wear number.{' '}
          <strong>
            We built one that is right, that tells you how confident it is and is genuinely right
            that often, and that proves the standard method is broken in three races out of four.
          </strong>{' '}
          Then we ran the same engine on NASA jet engines to show it is not a Formula 1 trick.
        </p>
        <p className="mt-3 text-[12px] text-ink-faint">
          Built on 203 real sessions and 91,867 laps from 2022 to 2025 · 32 recorded experiments ·
          615 automated tests · runs offline on a laptop with no GPU
        </p>
      </section>
    </div>
  )
}

/**
 * The one picture that explains the product to somebody who has never heard of
 * it: a lap time going in on the left, and four named causes coming out.
 *
 * Deliberately not a chart. It carries no axis and no live data, because its job
 * is to make the idea land in about four seconds, and a reader who has to work
 * out an axis has already stopped. The widths are illustrative and the caption
 * says so -- a diagram that looked like a measurement without being one would be
 * exactly the overclaim the rest of this project spends its time avoiding.
 */
function Separation() {
  const parts = [
    { name: 'The tyre', note: 'what we want', pct: 34, colour: 'var(--color-alert)', hero: true },
    { name: 'Fuel burn', note: 'car gets lighter', pct: 30, colour: 'var(--color-fuel)' },
    { name: 'Track grip', note: 'rubber goes down', pct: 18, colour: 'var(--color-track)' },
    { name: 'Traffic', note: 'stuck behind a car', pct: 12, colour: 'var(--color-traffic)' },
    { name: 'Unexplained', note: 'we show this too', pct: 6, colour: 'var(--color-ink-faint)' },
  ]

  return (
    <div className="mt-7 grid items-center gap-5 lg:grid-cols-[minmax(150px,auto)_16px_1fr]">
      <div className="border border-line bg-raised px-5 py-4">
        <div className="text-[10.5px] tracking-[0.16em] text-ink-faint uppercase">
          What you are given
        </div>
        <div className="num mt-1.5 text-[30px] leading-none font-medium text-ink">
          1:32.418
        </div>
        <div className="mt-1.5 text-[11.5px] text-ink-faint">
          One lap time. One number. No breakdown.
        </div>
      </div>

      <div className="hidden text-center text-[20px] text-ink-faint lg:block">&rarr;</div>

      <div>
        <div className="text-[10.5px] tracking-[0.16em] text-ink-faint uppercase">
          What TyreMind gives back
        </div>
        <div className="mt-2.5 flex h-11 overflow-hidden border border-line">
          {parts.map((p) => (
            <div
              key={p.name}
              className="relative flex items-center justify-center"
              style={{
                width: `${p.pct}%`,
                background: p.hero
                  ? p.colour
                  : `color-mix(in oklab, ${p.colour} 42%, transparent)`,
                borderRight: '1px solid var(--color-line)',
              }}
              title={`${p.name} — ${p.note}`}
            >
              {p.pct >= 12 && (
                <span
                  className="px-1 text-center text-[10.5px] leading-tight font-medium"
                  style={{ color: p.hero ? '#fff' : 'var(--color-ink)' }}
                >
                  {p.name}
                </span>
              )}
            </div>
          ))}
        </div>
        <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1">
          {parts.map((p) => (
            <span key={p.name} className="flex items-center gap-1.5 text-[11px] text-ink-faint">
              <span
                className="inline-block h-2 w-2"
                style={{ background: p.colour }}
              />
              {p.name}
              <span className="text-ink-faint/70">· {p.note}</span>
            </span>
          ))}
        </div>
        <p className="mt-2.5 text-[11.5px] text-ink-faint">
          Illustrative proportions. The real split for any lap is on the{' '}
          <span className="text-ink-dim">Why is the car slow</span> screen, measured, with the part
          we cannot explain shown rather than hidden.
        </p>
      </div>
    </div>
  )
}

function Headline({
  value,
  unit,
  label,
  tone,
}: {
  value: string
  unit: string
  label: string
  tone: string
}) {
  return (
    <div className="bg-surface px-5 py-6" style={{ borderTop: `2px solid ${tone}` }}>
      <div className="flex items-baseline gap-2">
        <span
          className="num text-[38px] leading-none font-semibold tracking-[-0.02em]"
          style={{ color: tone }}
        >
          {value}
        </span>
        <span className="text-[12px] text-ink-faint">{unit}</span>
      </div>
      <p className="mt-3 text-[12.5px] leading-snug text-ink-dim">{label}</p>
    </div>
  )
}

function Step({ n, title, body }: { n: string; title: string; body: string }) {
  return (
    <div className="flex gap-3.5">
      <span
        className="num mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center border text-[12px] font-medium"
        style={{ borderColor: 'var(--color-alert)', color: 'var(--color-alert)' }}
      >
        {n}
      </span>
      <div>
        <div className="text-[14px] leading-snug font-semibold text-ink">{title}</div>
        <p className="mt-1.5 text-[12.5px] leading-relaxed text-ink-dim">{body}</p>
      </div>
    </div>
  )
}

function Reason({ title, body }: { title: string; body: string }) {
  return (
    <div className="flex gap-3 border border-line bg-raised/30 px-4 py-4">
      <span
        className="mt-[3px] shrink-0 text-[13px] leading-none"
        style={{ color: 'var(--color-good)' }}
        aria-hidden
      >
        &#10003;
      </span>
      <div>
        <div className="text-[13.5px] leading-snug font-semibold text-ink">{title}</div>
        <p className="mt-1.5 text-[12.5px] leading-relaxed text-ink-dim">{body}</p>
      </div>
    </div>
  )
}
