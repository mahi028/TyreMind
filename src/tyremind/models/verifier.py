"""Check generated language against the state that prompted it, claim by claim.

The agent layer turns a `TyreState` into sentences a strategist reads. The
failure mode that matters is not clumsy phrasing -- it is a fluent sentence
citing a number that was never computed.

Pitwall (arXiv:2607.06495) documents this precisely, and their finding is the
reason this module exists and is built before the generator rather than after:
fine-tuning a generator on richer targets improves fluency, *and the identical
model then fabricates drivers, gaps and tyre compounds when the grounding state
is sparse*. Replicating across four base models showed it is a property of
instruction adherence, **not of scale**. A bigger model does not fix it.

Read that in this setting: the state is thinnest early in a stint, which is
exactly when a strategist is reading. The generator will hallucinate at the worst
possible moment, and only a verifier prevents it.

So every sentence is decomposed into typed claims, each claim is checked against
`TyreState.numeric_claims()`, and a sentence with any unsupported claim is
discarded in favour of a template that is faithful by construction. The
verification rate is logged rather than assumed -- Pitwall retained 81.9% of
model-written candidates, and ours reports whatever it actually achieves.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from tyremind.models.state import TyreState

#: Numbers written with units or as percentages, e.g. "0.198 s/lap", "62%",
#: "lap 34", "4.3 seconds". The unit is captured so a figure cannot be checked
#: against a quantity measured in something else.
_NUMBER = re.compile(
    r"(?P<lead>lap\s+)?"
    r"(?P<sign>[-+]?)"
    r"(?P<number>\d+(?:\.\d+)?)"
    r"\s*"
    r"(?P<unit>%|s/lap|s\b|seconds?\b|laps?\b)?",
    re.IGNORECASE,
)

#: Tolerances for matching a written figure to a computed one. Generous enough
#: that rounding for display is not a violation, tight enough that a different
#: number is.
RELATIVE_TOLERANCE = 0.02
ABSOLUTE_TOLERANCE = 0.005


@dataclass(frozen=True)
class Claim:
    """One factual assertion extracted from a sentence."""

    text: str
    number: float
    unit: str
    supported: bool
    matched_to: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "number": self.number,
            "unit": self.unit,
            "supported": self.supported,
            "matched_to": self.matched_to,
        }


@dataclass
class Verdict:
    """Whether a sentence may be published, and why not when it may not."""

    sentence: str
    claims: list[Claim] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        """A sentence with no numeric claims is publishable.

        Qualitative language cannot invent a figure, so it needs no grounding --
        which is also why the fallback templates are written to lean on
        quantities that are always present.
        """
        return all(claim.supported for claim in self.claims)

    @property
    def unsupported(self) -> list[Claim]:
        return [claim for claim in self.claims if not claim.supported]

    def to_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "supported": self.supported,
            "claims": [c.to_dict() for c in self.claims],
        }


def _close(written: float, computed: float) -> bool:
    return math.isclose(written, computed,
                        rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE)


#: Which written units may satisfy which stored units. A figure in seconds per
#: lap cannot be checked against a figure in seconds, however close the digits.
_COMPATIBLE: dict[str, set[str]] = {
    "s/lap": {"s/lap"},
    "s": {"s"},
    "second": {"s"},
    "seconds": {"s"},
    "%": {"%", "fraction"},
    "lap": {"lap"},
    "laps": {"lap"},
}


def _candidates(number: float, unit: str, claims: dict[str, float],
                units: dict[str, str]) -> list[str]:
    """Which computed quantities a written figure could legitimately be.

    Two filters, and the unit one is load-bearing. A percentage is matched both
    as written and as a fraction, because a confidence stored as 0.51 is
    displayed as 51%. Everything else must agree on what it measures.

    An unrecognised or absent unit falls back to magnitude alone, which is the
    lenient case -- so templates should carry units, and the fallback text does.
    """
    written = (unit or "").lower().rstrip(".")
    allowed = _COMPATIBLE.get(written)

    matches = []
    for name, value in claims.items():
        stored = units.get(name, "")
        if allowed is not None and stored not in allowed:
            continue
        if written == "%" and _close(number / 100.0, value):
            matches.append(name)
        elif _close(number, value):
            matches.append(name)
    return matches


def extract_claims(sentence: str, state: TyreState) -> list[Claim]:
    """Pull every numeric assertion out of a sentence and check each one."""
    computed = state.numeric_claims()
    units = state.claim_units()
    claims: list[Claim] = []

    for match in _NUMBER.finditer(sentence):
        raw = match.group("number")
        if raw is None:
            continue
        number = float(match.group("sign") + raw)
        unit = (match.group("unit") or "").strip()
        # "lap 34" is a lap number even though it carries no unit token.
        if match.group("lead"):
            unit = "lap"

        matched = _candidates(number, unit, computed, units)

        claims.append(Claim(
            text=match.group(0).strip(),
            number=number,
            unit=unit,
            supported=bool(matched),
            matched_to=matched[0] if matched else "",
        ))
    return claims


def verify(sentence: str, state: TyreState) -> Verdict:
    """Check one sentence against one state."""
    return Verdict(sentence=sentence, claims=extract_claims(sentence, state))


@dataclass
class VerificationLog:
    """Running record of how often generated text survived checking.

    Reported rather than assumed. A verification rate that quietly falls is the
    signal that the generator has started inventing, and it is invisible unless
    something counts.
    """

    checked: int = 0
    published: int = 0
    fell_back: int = 0
    rejected_claims: list[Claim] = field(default_factory=list)

    @property
    def verification_rate(self) -> float:
        return self.published / self.checked if self.checked else float("nan")

    def record(self, verdict: Verdict) -> None:
        self.checked += 1
        if verdict.supported:
            self.published += 1
        else:
            self.fell_back += 1
            self.rejected_claims.extend(verdict.unsupported)

    def to_dict(self) -> dict:
        return {
            "checked": self.checked,
            "published": self.published,
            "fell_back": self.fell_back,
            "verification_rate": (None if math.isnan(self.verification_rate)
                                  else self.verification_rate),
            "rejected": [c.to_dict() for c in self.rejected_claims[:20]],
        }


def publish(
    candidates: list[str],
    fallback: str,
    state: TyreState,
    log: VerificationLog | None = None,
) -> tuple[str, Verdict | None]:
    """Return the first candidate whose every claim checks out, else the fallback.

    The fallback is not verified because it is built from the state by
    construction -- template substitution cannot invent a figure it was not
    given. Verifying it would only test the template engine.

    Args:
        candidates: Generated sentences, best first.
        fallback: A provably faithful template.
        state: The state that prompted them.
        log: Optional running record.

    Returns:
        The published sentence and the verdict that admitted it, or the fallback
        and None.
    """
    for candidate in candidates:
        verdict = verify(candidate, state)
        if log is not None:
            log.record(verdict)
        if verdict.supported:
            return candidate, verdict
    return fallback, None
