/**
 * The claim verifier, ported from `tyremind/models/verifier.py`.
 *
 * The failure mode this guards against is not clumsy phrasing. It is a fluent
 * sentence citing a number that was never computed. Pitwall (arXiv:2607.06495)
 * measured that precisely: fine-tuning a generator on richer targets improves
 * fluency *and* the same model then fabricates drivers, gaps and compounds when
 * the grounding state is sparse -- across four base models, so it is a property
 * of instruction adherence rather than of scale. The state is sparsest early in
 * a stint, which is exactly when a strategist is reading.
 *
 * So every sentence the engineer agent wants to say is decomposed into typed
 * claims, each claim is checked against the numbers the state actually holds,
 * and a sentence with any unsupported claim is discarded in favour of a
 * template that is faithful by construction. The rate at which that happens is
 * shown on screen: a verification rate that quietly falls is the signal the
 * generator has started inventing, and it is invisible unless something counts.
 *
 * The unit matters as much as the magnitude. Python's comment records the case:
 * a written "0.999 s/lap" once verified clean against a `projected_loss_3_high`
 * of 0.9924 -- a different quantity in different units that happened to land
 * nearby. With two dozen claims spanning several orders of magnitude, an
 * invented figure has a real chance of being numerically close to something, so
 * the unit is what makes the check mean anything.
 */

/** Matches "0.198 s/lap", "62%", "lap 34", "4.3 seconds". Mirrors `_NUMBER`. */
const NUMBER =
  /(?<lead>lap\s+)?(?<sign>[-+]?)(?<number>\d+(?:\.\d+)?)\s*(?<unit>%|s\/lap|s\b|seconds?\b|laps?\b)?/gi

const RELATIVE_TOLERANCE = 0.02
const ABSOLUTE_TOLERANCE = 0.005

/** Which written units may satisfy which stored units. Mirrors `_COMPATIBLE`. */
const COMPATIBLE: Record<string, string[]> = {
  's/lap': ['s/lap'],
  s: ['s'],
  second: ['s'],
  seconds: ['s'],
  '%': ['%', 'fraction'],
  lap: ['lap'],
  laps: ['lap'],
}

function isClose(written: number, computed: number): boolean {
  return (
    Math.abs(written - computed) <=
    Math.max(RELATIVE_TOLERANCE * Math.max(Math.abs(written), Math.abs(computed)), ABSOLUTE_TOLERANCE)
  )
}

export interface Claim {
  text: string
  number: number
  unit: string
  supported: boolean
  matchedTo: string
}

export interface Verdict {
  sentence: string
  claims: Claim[]
  supported: boolean
  unsupported: Claim[]
}

/**
 * Everything a generated sentence is allowed to cite, with the unit of each.
 *
 * The TypeScript twin of `TyreState.numeric_claims()` and `claim_units()`. A
 * sentence quoting a figure that is not in here has invented it.
 */
export interface GroundedState {
  claims: Record<string, number>
  units: Record<string, string>
}

/** Add an estimate and, when it has one, its 95% interval. */
export function putEstimate(
  state: GroundedState,
  name: string,
  value: number | null | undefined,
  sd: number | null | undefined,
  unit: string,
): void {
  if (value == null || !Number.isFinite(value)) return
  state.claims[name] = value
  state.units[name] = unit
  if (sd != null && Number.isFinite(sd)) {
    state.claims[`${name}_low`] = value - 1.959963984540054 * sd
    state.claims[`${name}_high`] = value + 1.959963984540054 * sd
    state.units[`${name}_low`] = unit
    state.units[`${name}_high`] = unit
  }
}

function candidates(number: number, unit: string, state: GroundedState): string[] {
  const written = (unit || '').toLowerCase().replace(/\.$/, '')
  const allowed = COMPATIBLE[written]
  const matches: string[] = []
  for (const [name, value] of Object.entries(state.claims)) {
    const stored = state.units[name] ?? ''
    if (allowed !== undefined && !allowed.includes(stored)) continue
    // A confidence stored as 0.51 is displayed as 51%, so a percentage is
    // matched both as written and as a fraction.
    if (written === '%' && isClose(number / 100, value)) matches.push(name)
    else if (isClose(number, value)) matches.push(name)
  }
  return matches
}

/** Pull every numeric assertion out of a sentence and check each one. */
export function extractClaims(sentence: string, state: GroundedState): Claim[] {
  const claims: Claim[] = []
  NUMBER.lastIndex = 0
  for (const match of sentence.matchAll(NUMBER)) {
    const groups = match.groups ?? {}
    if (!groups.number) continue
    const number = Number(`${groups.sign ?? ''}${groups.number}`)
    // "lap 34" is a lap number even though it carries no unit token.
    const unit = groups.lead ? 'lap' : (groups.unit ?? '').trim()
    const matched = candidates(number, unit, state)
    claims.push({
      text: match[0].trim(),
      number,
      unit,
      supported: matched.length > 0,
      matchedTo: matched[0] ?? '',
    })
  }
  return claims
}

/** Check one sentence against one state. */
export function verify(sentence: string, state: GroundedState): Verdict {
  const claims = extractClaims(sentence, state)
  const unsupported = claims.filter((c) => !c.supported)
  return {
    sentence,
    claims,
    // A sentence with no numeric claims is publishable: qualitative language
    // cannot invent a figure.
    supported: unsupported.length === 0,
    unsupported,
  }
}

/** Running record of how often generated text survived checking. */
export class VerificationLog {
  checked = 0
  published = 0
  fellBack = 0
  rejected: { claim: Claim; sentence: string }[] = []

  get verificationRate(): number {
    return this.checked ? this.published / this.checked : Number.NaN
  }

  record(verdict: Verdict): void {
    this.checked += 1
    if (verdict.supported) {
      this.published += 1
    } else {
      this.fellBack += 1
      for (const claim of verdict.unsupported) {
        this.rejected.push({ claim, sentence: verdict.sentence })
      }
      this.rejected = this.rejected.slice(-40)
    }
  }
}

export interface Published {
  text: string
  verdict: Verdict | null
  /** True when a candidate was rejected and the template answered instead. */
  usedFallback: boolean
}

/**
 * Return the first candidate whose every claim checks out, else the fallback.
 *
 * The fallback is not verified because it is built from the state by
 * construction -- template substitution cannot invent a figure it was not
 * given. Verifying it would only test the template engine.
 */
export function publish(
  candidateSentences: string[],
  fallback: string,
  state: GroundedState,
  log: VerificationLog,
): Published {
  for (const candidate of candidateSentences) {
    const verdict = verify(candidate, state)
    log.record(verdict)
    if (verdict.supported) return { text: candidate, verdict, usedFallback: false }
  }
  return { text: fallback, verdict: null, usedFallback: true }
}
