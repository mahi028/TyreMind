"""The single source of truth every surface reads and every claim is checked against.

Four surfaces need the same facts: the strategist console, the one-line relay to
the race engineer, the crew's box command, and the post-race audit. If each one
calls the API separately and assembles its own picture, they drift -- the console
says box on lap 34 while the relay says two more laps, and nobody can tell which
is wrong. That divergence is the classic failure of this kind of system and it is
very hard to debug after the fact, because every component is individually
correct.

So there is one object. It is built once per (session, driver, lap), every
surface renders *from* it, and the verifier checks generated language *against*
it. A sentence citing a number not in here is, by definition, invented.

**Every quantity carries its uncertainty.** That is not a style preference. A
degradation rate without an interval is the thing every competing system already
ships; the interval is the product. `Estimate` therefore has no constructor path
that produces a bare number, and `to_dict` always emits `sd` and `ci95`.

**Unknown is a value, not a gap.** `Estimate.unknown()` exists so that "we cannot
answer this" travels through the system as data and reaches the screen as words,
rather than becoming a null that a template renders as zero. The Monaco case --
where degradation is not what decides the stop -- has to survive all the way to
the strategist, and it only does that if there is somewhere to put it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

#: Multiplier for a 95% interval under a Gaussian. Stated once so that a reader
#: can find every place an interval is formed by searching for it.
Z95 = 1.959963984540054


def _finite(value: float | None) -> float | None:
    """JSON has no NaN or Infinity.

    A bare NaN is not valid JSON, strict parsers reject it, and one such field
    poisons an entire frame rather than only itself. This has already happened
    once on the live WebSocket, which is why it is a helper and not a habit.
    """
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class Estimate:
    """A number that knows how sure it is, and what it rests on.

    Attributes:
        value: The estimate. `None` means genuinely unknown -- see `unknown()`.
        sd: Standard deviation. `None` when the quantity has no meaningful
            uncertainty (a lap number that was observed, say) or when unknown.
        unit: Displayed unit, e.g. "s/lap". Kept with the number so a surface
            cannot label it wrongly.
        basis: What the estimate rests on, in words a strategist can read:
            "MEDIUM, 12 laps, 4 stints". Shown next to the number, because a
            rate from four laps and a rate from forty are not the same claim.
        calibrated: Whether the interval has been conformally calibrated and its
            coverage measured. **False does not mean bad -- it means unchecked**,
            and the difference has to be visible or the whole argument collapses.
        reason: When `value` is None, why. Rendered verbatim.
    """

    value: float | None
    sd: float | None = None
    unit: str = ""
    basis: str = ""
    calibrated: bool = False
    reason: str = ""

    @classmethod
    def unknown(cls, reason: str, unit: str = "") -> Estimate:
        """An answer of "we do not know", carried as a value.

        Used where the honest output is a refusal: a flat pit-cost curve, a rate
        indistinguishable from zero, a compound with too few laps. The reason
        travels to the screen.
        """
        return cls(value=None, sd=None, unit=unit, reason=reason)

    @property
    def known(self) -> bool:
        return self.value is not None

    @property
    def ci95(self) -> tuple[float, float] | None:
        if self.value is None or self.sd is None:
            return None
        return (self.value - Z95 * self.sd, self.value + Z95 * self.sd)

    def to_dict(self) -> dict[str, Any]:
        interval = self.ci95
        return {
            "value": _finite(self.value),
            "sd": _finite(self.sd),
            "ci95": [_finite(interval[0]), _finite(interval[1])] if interval else None,
            "unit": self.unit,
            "basis": self.basis,
            "calibrated": self.calibrated,
            "known": self.known,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Contribution:
    """One named cause of a lap-time change, with its share and its uncertainty.

    `is_tyre` exists so a surface can colour the tyre band differently without
    string-matching the label, and `is_residual` so the unexplained part can be
    rendered as its own visible band. Folding the residual into the other causes
    to make the bars add up is the one presentation choice that would make this
    dishonest: it claims we explain more than we do.
    """

    key: str
    label: str
    seconds: Estimate
    is_tyre: bool = False
    is_residual: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "seconds": self.seconds.to_dict(),
            "is_tyre": self.is_tyre,
            "is_residual": self.is_residual,
        }


@dataclass(frozen=True)
class PitAdvice:
    """When to box, how sure, and what it costs to be wrong.

    Attributes:
        recommended_lap: The lap to box on, or None when the tyre is not what
            decides this stop.
        confidence_in_lap: Share of simulated races in which that exact lap was
            fastest. Usually low -- picking one lap out of thirty on a flat cost
            curve is genuinely uncertain -- and that is the honest number.
        window: Laps within a second of the optimum.
        confidence_in_window: Share of simulations won by any lap in the window.
            **This is the number a strategist acts on**, not the single-lap one.
        cost_of_being_late: Seconds lost by boxing at the far end of the window.
        alternative_compound: What would be fitted.
        reason: Why there is no recommendation, when there is none.
    """

    recommended_lap: int | None
    confidence_in_lap: float | None
    window: tuple[int, int] | None
    confidence_in_window: float | None
    cost_of_being_late: Estimate
    alternative_compound: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_lap": self.recommended_lap,
            "confidence_in_lap": _finite(self.confidence_in_lap),
            "window": list(self.window) if self.window else None,
            "confidence_in_window": _finite(self.confidence_in_window),
            "cost_of_being_late": self.cost_of_being_late.to_dict(),
            "alternative_compound": self.alternative_compound,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TrustReport:
    """Whether this estimate should be believed, with the evidence attached.

    Attributes:
        methods_agree_within: Spread across independent estimation methods, s/lap.
        n_methods: How many were run.
        disagreement_flagged: True when they do not agree, which is a warning to
            show prominently rather than a number to average away.
        measured_coverage: What our 95% interval actually covered when checked.
            The single most important number in the system.
        coverage_sample: How many observations that was measured on.
        explanation: Prose written for display.
    """

    methods_agree_within: float | None
    n_methods: int
    disagreement_flagged: bool
    measured_coverage: float | None
    coverage_sample: int
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "methods_agree_within": _finite(self.methods_agree_within),
            "n_methods": self.n_methods,
            "disagreement_flagged": self.disagreement_flagged,
            "measured_coverage": _finite(self.measured_coverage),
            "coverage_sample": self.coverage_sample,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class TyreState:
    """Everything known about one car's tyre at one lap. The contract.

    Answers the five questions the product exists for:

    1. Where is the time going?      -> `contributions`
    2. How fast is it degrading?     -> `degradation_rate`
    3. What happens in N laps?       -> `projected_loss`
    4. When do I box?                -> `pit`
    5. Should I believe this?        -> `trust`

    Attributes:
        session_id: Which session.
        driver: Three-letter code.
        lap: Session lap this state describes.
        compound: Tyre in use.
        tyre_age: Laps on this set.
        laps_in_stint: Laps since the stint began -- the fuel proxy. Not the same
            as `tyre_age`, and conflating the two is the most common error in
            this problem.
        regime: Measured curve shape: linear, warm-up, cliff, recovery, or "" when
            the stint is too short to fit one.
        degradation_rate: Seconds per lap lost to the tyre, now.
        contributions: The lap-time decomposition, residual included.
        projected_loss: Horizon in laps to cumulative seconds lost.
        life_remaining_pct: Share of usable life left, anchored to the measured
            cliff onset rather than a hand-set threshold.
        pit: The recommendation.
        trust: Why to believe it.
        rule_verdicts: Output of the deterministic safety rules, which run
            alongside the model and may veto it. Disagreement between the two is
            shown, never resolved silently.
        generated_at: When this state was built.
    """

    session_id: str
    driver: str
    lap: int
    compound: str
    tyre_age: float
    laps_in_stint: float

    regime: str
    degradation_rate: Estimate
    contributions: list[Contribution]
    projected_loss: dict[int, Estimate]
    life_remaining_pct: Estimate
    pit: PitAdvice
    trust: TrustReport

    rule_verdicts: list[dict[str, Any]] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def tyre_seconds(self) -> Estimate:
        for contribution in self.contributions:
            if contribution.is_tyre:
                return contribution.seconds
        return Estimate.unknown("no tyre contribution in this decomposition", "s")

    @property
    def unexplained_seconds(self) -> Estimate:
        for contribution in self.contributions:
            if contribution.is_residual:
                return contribution.seconds
        return Estimate.unknown("no residual reported", "s")

    def numeric_claims(self) -> dict[str, float]:
        """Every number a generated sentence is allowed to cite.

        The verifier checks language against this. A sentence quoting a figure
        that is not in here has invented it, which is the failure mode that
        matters: a fluent generator fabricates most readily when the grounding
        state is thin, and the state is thinnest early in a stint, which is
        exactly when a strategist is reading.
        """
        claims: dict[str, float] = {
            "lap": float(self.lap),
            "tyre_age": float(self.tyre_age),
            "laps_in_stint": float(self.laps_in_stint),
        }

        def put(name: str, estimate: Estimate) -> None:
            if estimate.value is None:
                return
            claims[name] = float(estimate.value)
            if estimate.sd is not None:
                interval = estimate.ci95
                if interval:
                    claims[f"{name}_low"], claims[f"{name}_high"] = interval

        put("degradation_rate", self.degradation_rate)
        put("life_remaining_pct", self.life_remaining_pct)
        for contribution in self.contributions:
            put(f"contribution_{contribution.key}", contribution.seconds)
        for horizon, estimate in self.projected_loss.items():
            put(f"projected_loss_{horizon}", estimate)
        if self.pit.recommended_lap is not None:
            claims["recommended_lap"] = float(self.pit.recommended_lap)
        if self.pit.confidence_in_lap is not None:
            claims["confidence_in_lap"] = float(self.pit.confidence_in_lap)
        if self.pit.confidence_in_window is not None:
            claims["confidence_in_window"] = float(self.pit.confidence_in_window)
        if self.pit.window:
            claims["window_start"] = float(self.pit.window[0])
            claims["window_end"] = float(self.pit.window[1])
        if self.trust.measured_coverage is not None:
            claims["measured_coverage"] = float(self.trust.measured_coverage)
        return claims

    def claim_units(self) -> dict[str, str]:
        """The unit of every quantity in `numeric_claims`.

        Without this the verifier matches a number against anything of similar
        magnitude regardless of what it measures. A sentence claiming "0.999
        s/lap" once verified clean against `projected_loss_3_high` of 0.9924 --
        a different quantity in different units that happened to be numerically
        close. With twenty-eight claims spanning several orders of magnitude, an
        invented figure has a real chance of landing near one of them, so the
        unit is what makes the check mean anything.
        """
        units: dict[str, str] = {
            "lap": "lap",
            "tyre_age": "lap",
            "laps_in_stint": "lap",
            "recommended_lap": "lap",
            "window_start": "lap",
            "window_end": "lap",
            "confidence_in_lap": "fraction",
            "confidence_in_window": "fraction",
            "measured_coverage": "fraction",
        }

        def tag(name: str, unit: str) -> None:
            units[name] = unit
            units[f"{name}_low"] = unit
            units[f"{name}_high"] = unit

        tag("degradation_rate", self.degradation_rate.unit or "s/lap")
        tag("life_remaining_pct", "%")
        for contribution in self.contributions:
            tag(f"contribution_{contribution.key}", contribution.seconds.unit or "s")
        for horizon, estimate in self.projected_loss.items():
            tag(f"projected_loss_{horizon}", estimate.unit or "s")
        return units

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "driver": self.driver,
            "lap": int(self.lap),
            "compound": self.compound,
            "tyre_age": _finite(self.tyre_age),
            "laps_in_stint": _finite(self.laps_in_stint),
            "regime": self.regime,
            "degradation_rate": self.degradation_rate.to_dict(),
            "contributions": [c.to_dict() for c in self.contributions],
            "projected_loss": {
                str(h): e.to_dict() for h, e in sorted(self.projected_loss.items())
            },
            "life_remaining_pct": self.life_remaining_pct.to_dict(),
            "pit": self.pit.to_dict(),
            "trust": self.trust.to_dict(),
            "rule_verdicts": self.rule_verdicts,
            "generated_at": self.generated_at,
        }
