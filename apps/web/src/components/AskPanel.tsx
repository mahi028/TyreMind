/**
 * Ask the project about itself.
 *
 * The honest answer to "why should I trust this?" is spread across a research
 * audit, a model card, a limitations register and seven experiment result files.
 * Nobody reads all of that. This makes it answerable in one line.
 *
 * **It retrieves; it does not generate.** Every panel here is a passage lifted
 * verbatim from a document or an experiment output, with its source attached.
 * A generated summary would be smoother and would remove the one property that
 * makes this useful — that every claim can be checked against the file it came
 * from.
 */

import { useCallback, useEffect, useState } from 'react'
import { Panel } from './primitives'
import { Explainer } from './Explainer'
import { CorpusComposition, RankFusion } from './charts'

interface AskResult {
  query: string
  results: {
    text: string
    source: string
    heading: string
    kind: string
    citation: string
    score: number
    lexical_rank: number | null
    semantic_rank: number | null
  }[]
  corpus: {
    n_passages: number
    n_sources: number
    by_kind: Record<string, number>
    by_source: { source: string; n_passages: number }[]
    embedding_backend: string
    retrieval: string
  }
  note: string
}

/** Questions that show the retrieval working on both of its halves. */
const SUGGESTED = [
  'What does this not measure?',
  'How accurate is it against a known answer?',
  'What happens when it rains?',
  'CRPS',
  'Does it work on jet engines?',
  'Which assumptions carry the result?',
  'Where does a simpler model beat you?',
]

export function AskPanel() {
  const [query, setQuery] = useState('')
  const [result, setResult] = useState<AskResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const run = useCallback((q: string) => {
    if (!q.trim()) return
    setBusy(true)
    setError('')
    fetch(`/api/ask?q=${encodeURIComponent(q)}&k=5`)
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json()).detail ?? r.statusText)
        return r.json()
      })
      .then(setResult)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setBusy(false))
  }, [])

  useEffect(() => {
    run(SUGGESTED[0])
    setQuery(SUGGESTED[0])
  }, [run])

  return (
    <div className="space-y-4">
      <Explainer id="ask" question="What is this?">
        <p>
          A search over TyreMind&rsquo;s own research audit, model card, limitations
          register and every recorded experiment result, so a question about the
          method has an answer with a source attached.
        </p>
        <p>
          <strong>It retrieves rather than generates.</strong> Every passage below
          is lifted verbatim from a file in this repository, with its source
          attached.
        </p>
      </Explainer>

      <Panel
        title="Ask about the method"
        aside={
          result
            ? `${result.corpus.n_passages} passages · ${result.corpus.n_sources} files`
            : undefined
        }
      >
        <form
          onSubmit={(e) => {
            e.preventDefault()
            run(query)
          }}
          className="flex gap-2"
        >
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. how do you know the degradation is real?"
            className="min-w-0 flex-1 rounded-sm border border-line bg-raised px-3 py-2 text-[13px] text-ink placeholder:text-ink-ghost focus:border-alert focus:outline-none transition-colors duration-150"
          />
          <button
            type="submit"
            disabled={busy}
            className="shrink-0 rounded-pill border border-alert px-4 py-2 text-[12px] font-medium text-alert transition-colors duration-150 hover:bg-alert/10 disabled:opacity-40"
          >
            {busy ? 'Searching…' : 'Search'}
          </button>
        </form>

        <div className="mt-3 flex flex-wrap gap-1.5">
          {SUGGESTED.map((s) => (
            <button
              key={s}
              onClick={() => {
                setQuery(s)
                run(s)
              }}
              className="rounded-pill border border-line px-2 py-1 text-[11px] text-ink-dim transition-colors duration-150 hover:border-line-bright hover:text-ink"
            >
              {s}
            </button>
          ))}
        </div>

        {error && (
          <p className="mt-3 text-[12px]" style={{ color: 'var(--color-danger)' }}>
            {error}
          </p>
        )}
      </Panel>

      {result && (
        <div className="grid gap-4 xl:grid-cols-[1.55fr_1fr]">
        <Panel
          title={`Passages matching “${result.query}”`}
          aside={`${result.results.length} results`}
        >
          {result.results.length === 0 ? (
            <p className="py-6 text-center text-[12.5px] text-ink-faint">
              Nothing in the documentation matches closely enough to show. An empty
              result beats a confident citation of an irrelevant passage.
            </p>
          ) : (
              <div className="space-y-4">
              {result.results.map((hit, i) => (
                <article key={i} className="rounded-md border-l-2 pl-3.5" style={{ borderLeftColor: 'var(--color-line-bright)' }}>
                  <div className="mb-1 flex flex-wrap items-baseline gap-2">
                    <span
                      className="rounded-pill px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-[0.06em]"
                      style={{
                        background: hit.kind === 'result'
                          ? 'color-mix(in oklab, var(--color-good) 15%, transparent)'
                          : 'var(--color-raised)',
                        color: hit.kind === 'result'
                          ? 'var(--color-good)'
                          : 'var(--color-ink-faint)',
                      }}
                    >
                      {hit.kind === 'result' ? 'measured result' : 'documentation'}
                    </span>
                    <span className="num text-[10.5px] text-ink-faint">{hit.source}</span>
                    {hit.heading && (
                      <span className="text-[10.5px] text-ink-dim">· {hit.heading}</span>
                    )}
                  </div>
                  <p className="max-w-[86ch] text-[12.5px] leading-relaxed text-ink">
                    {hit.text}
                  </p>
                  <div className="mt-1 flex gap-3 text-[10px] text-ink-faint">
                    <span>keyword rank {hit.lexical_rank != null ? hit.lexical_rank + 1 : 'not ranked'}</span>
                    <span>meaning rank {hit.semantic_rank != null ? hit.semantic_rank + 1 : 'not ranked'}</span>
                  </div>
                </article>
              ))}
            </div>
          )}

          <div className="mt-5 border-t border-line pt-3">
            <div className="mb-1 text-[11px] text-ink-faint">How the search works</div>
            <p className="max-w-[80ch] text-[11.5px] leading-relaxed text-ink-dim">
              Two retrievers with different weaknesses run in parallel, merged by
              rank. <strong>Keyword search</strong> finds exact terms like
              &ldquo;CRPS&rdquo; but misses paraphrase. <strong>Meaning search</strong>{' '}
              finds a passage about calibration when you ask &ldquo;how sure is
              it&rdquo;, but misses rare exact tokens. The ranks above show which
              retriever found each passage.
            </p>
            <p className="mt-1.5 max-w-[80ch] text-[11px] leading-relaxed text-ink-faint">
              {result.corpus.retrieval}. Embeddings: {result.corpus.embedding_backend}
            </p>
          </div>
        </Panel>

        <div className="space-y-4">
          {result.results.length > 0 && (
            <Panel title="Which retriever found what" aside="the case for fusing two">
              <RankFusion hits={result.results} />
              <p className="mt-2 max-w-[46ch] text-[11.5px] leading-relaxed text-ink-dim">
                Each numbered dot is one answer, placed by where the two
                retrievers ranked it. On the dashed diagonal, both agreed. Off it,
                one retriever found the passage and the other nearly missed it.
              </p>
              <p className="mt-1.5 max-w-[46ch] text-[11px] leading-relaxed text-ink-faint">
                Merged by Reciprocal Rank Fusion, which scores a passage by its
                <em> position</em> in each list rather than its raw score: BM25
                relevance and cosine similarity aren&rsquo;t on a common scale, but
                ranks are comparable.
              </p>
            </Panel>
          )}

          {result.corpus.by_source?.length > 0 && (
            <Panel
              title="What the corpus is made of"
              aside={`${result.corpus.by_kind.result ?? 0} passages are measured results`}
            >
              <CorpusComposition bySource={result.corpus.by_source} />
              <p className="mt-2 max-w-[46ch] text-[11.5px] leading-relaxed text-ink-faint">
                Green files are recorded experiment output: numbers this system
                measured and wrote down. Grey is prose. Nothing outside this
                repository is indexed, so an answer can always be traced to a
                file you can open.
              </p>
            </Panel>
          )}
        </div>
        </div>
      )}

      <Panel title="Why a retrieval index and an MCP server, and not a chatbot">
        <div className="grid gap-5 md:grid-cols-3">
          <div>
            <div className="mb-1.5 text-[12.5px] font-medium text-ink">
              RAG, without the G
            </div>
            <p className="max-w-[44ch] text-[12px] leading-relaxed text-ink-dim">
              The retrieval half of retrieval-augmented generation is the half that
              carries the trust: it finds the paragraph, you read the paragraph. A
              language model is used in exactly one place here, rewriting an
              already-computed explanation into plainer English, with every number
              fixed before it is called.
            </p>
          </div>
          <div>
            <div className="mb-1.5 text-[12.5px] font-medium text-ink">
              MCP, so an agent can use the model
            </div>
            <p className="max-w-[44ch] text-[12px] leading-relaxed text-ink-dim">
              The Model Context Protocol exposes the estimator as seven tools an
              assistant can call: get a degradation rate, explain a lap, project
              tyre life, price a strategy, check the fit, search this corpus. Seven,
              deliberately, since tool bloat degrades agent selection.
            </p>
          </div>
          <div>
            <div className="mb-1.5 text-[12.5px] font-medium text-ink">
              Every tool is read-only
            </div>
            <p className="max-w-[44ch] text-[12px] leading-relaxed text-ink-dim">
              There is no write path: an agent cannot change a fit, a threshold or a
              stored result, so the worst outcome of a confused agent is a confused
              answer, never a corrupted one. Every tool returns its uncertainty
              alongside its estimate.
            </p>
          </div>
        </div>
      </Panel>
    </div>
  )
}
