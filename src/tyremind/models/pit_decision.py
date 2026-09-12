"""Turn any degradation model into a pit-stop recommendation, so all of them can be scored.

Every rung on the ladder estimates a degradation rate. None of them, on its own,
says which lap to box on. This module supplies the missing step once, identically
for every model, so that a comparison of pit recommendations is a comparison of
*degradation estimates* and not of nine different pieces of strategy code.

The optimisation is the standard one. Staying out costs the degradation you keep
accumulating; boxing costs the pit-lane delta now and buys a fresh tyre for the
laps that remain. For a stint starting at tyre age `a0` with rate `r`, the time
lost to degradation over `n` further laps is the sum of an arithmetic series:

    loss(n) = sum_{i=1..n} r * (a0 + i)  =  r * (n*a0 + n*(n+1)/2)

so total race time from the decision lap, if we box after `k` more laps, is

    T(k) = loss_current(k) + pit_loss + loss_fresh(remaining - k)

which is quadratic in `k` and has a single interior minimum. Heilmeier et al.
solve the same problem analytically; here it is swept, because a sweep also
yields the *shape* of the cost curve, and the shape is what a confidence over
laps is made of.

**On confidence.** A strategist asking "how sure are you about lap 34" wants a
distribution over laps, not a single number. We produce one by turning the cost
curve into a softmax over -T(k), scaled by the model's own uncertainty about the
rate. A model that is unsure about the rate gets a flat distribution across many
laps; a model that is confident gets a peaked one. That is the honest mapping:
uncertainty about degradation *is* uncertainty about when to stop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Typical time lost in a Formula 1 pit stop, in seconds, including the pit lane
#: delta. Circuit-specific in reality (Monaco is ~16 s, Singapore ~26 s); a
#: single value is used so that every model is optimising against the same cost
#: and the comparison isolates the degradation estimate.
DEFAULT_PIT_LOSS_S = 21.0


def degradation_loss(rate: float, start_age: float, n_laps: int) -> float:
    """Seconds lost to degradation over `n_laps`, starting at tyre age `start_age`."""
    if n_laps <= 0:
        return 0.0
    n = float(n_laps)
    return float(rate * (n * start_age + n * (n + 1.0) / 2.0))


@dataclass(frozen=True)
class PitRecommendation:
    """A recommended pit lap and how sure the model is about it.

    Attributes:
        lap: Recommended session lap to box on.
        confidence: Probability mass on the recommended lap.
        distribution: Lap to probability, over every feasible lap.
        cost_curve: Lap to expected total remaining time, seconds.
        reason: Empty when the recommendation is well posed; otherwise why it
            is not, so the caller can exclude it rather than score noise.
    """

    lap: int
    confidence: float
    distribution: dict[int, float]
    cost_curve: dict[int, float]
    reason: str = ""


def recommend_pit_lap(
    *,
    current_rate: float,
    current_rate_sd: float,
    fresh_rate: float,
    current_age: float,
    decision_lap: int,
    final_lap: int,
    pit_loss_s: float = DEFAULT_PIT_LOSS_S,
    min_stint_laps: int = 3,
    max_remaining_stops: int = 3,
    flat_curve_tolerance_s: float = 3.0,
) -> PitRecommendation:
    """Sweep every feasible pit lap and return the cheapest, with a distribution.

    Args:
        current_rate: Degradation rate of the tyre now on the car, s/lap.
        current_rate_sd: The model's uncertainty about that rate. Drives how
            peaked the confidence distribution is.
        fresh_rate: Degradation rate of the compound that would be fitted.
        current_age: Tyre age at the decision lap.
        decision_lap: The lap we are deciding on.
        final_lap: Last lap of the race.
        pit_loss_s: Time cost of the stop.
        min_stint_laps: Shortest stint worth fitting a tyre for.
        max_remaining_stops: Most further stints to price. A two-stopper's first
            stop is far earlier than a one-stopper's only stop, so this must be
            searched rather than assumed.
        flat_curve_tolerance_s: If the best and worst candidate laps differ by
            less than this, the tyre is not what decides the stop and the model
            declines to answer.

    Returns:
        A recommendation. `reason` is non-empty when no meaningful choice exists.
    """
    # Rubber does not regenerate. A negative estimated rate is not a small tyre
    # gain, it is an estimate that has failed, and feeding it to the optimiser
    # produces "stay out forever".
    #
    # Monaco 2025 is where this surfaced. Our own MEDIUM estimate came out at
    # -0.0291 +- 0.0414 s/lap on the lowest-degradation circuit of the year, and
    # the optimiser dutifully recommended lap 42 against an actual median of 27 --
    # a 31-lap mean error that accounted for our entire deficit on this benchmark.
    #
    # exp14 measures the naive method producing a negative rate in 74% of races.
    # We are far rarer, but we are not immune, and criticising a method for a
    # failure mode we share without guarding against it would be indefensible.
    # Clamping at zero is the same physical constraint Cappello & Hoegh impose
    # through a half-normal prior -- their choice looks better here than our
    # earlier write-up allowed, and this applies to every model identically.
    rate_is_noise = current_rate <= 0.0 or current_rate < current_rate_sd
    current_rate = max(current_rate, 0.0)
    fresh_rate = max(fresh_rate, 0.0)

    remaining = final_lap - decision_lap
    if remaining < 2 * min_stint_laps:
        return PitRecommendation(decision_lap, 0.0, {}, {},
                                 reason="too few laps remain to choose between stops")

    candidates = list(range(decision_lap + 1, final_lap - min_stint_laps + 1))
    if not candidates:
        return PitRecommendation(decision_lap, 0.0, {}, {}, reason="no feasible pit lap")

    # Cost of boxing on `lap`, optimised over how many stops remain after it.
    #
    # The first version of this assumed exactly one stop left and ran the fresh
    # tyre to the flag. That is structurally wrong for a two-stopper and it
    # showed: on 274 real stops the mean error was +4.8 laps for drivers who
    # stopped once and +12.2 for drivers who stopped twice. Treating a
    # two-stopper's first stop as if it were their only stop pushes the
    # recommendation far too late, because the model is pricing in a stint that
    # will never be run.
    #
    # For a constant rate, splitting R remaining laps across n stints of equal
    # length minimises the total quadratic loss, so only the split count needs
    # searching, not the split itself.
    costs = {}
    for lap in candidates:
        laps_before = lap - decision_lap
        laps_after = final_lap - lap
        on_current = degradation_loss(current_rate, current_age, laps_before)

        best_for_lap = None
        for n_stints in range(1, max_remaining_stops + 1):
            if laps_after < n_stints * min_stint_laps:
                break
            per_stint = laps_after / n_stints
            # n_stints fresh stints means n_stints - 1 further stops after this one.
            total = (
                on_current
                + n_stints * pit_loss_s
                + n_stints * degradation_loss(fresh_rate, 0.0, int(round(per_stint)))
            )
            if best_for_lap is None or total < best_for_lap:
                best_for_lap = total
        costs[lap] = best_for_lap if best_for_lap is not None else float("inf")

    costs = {k: v for k, v in costs.items() if np.isfinite(v)}
    if not costs:
        return PitRecommendation(decision_lap, 0.0, {}, {},
                                 reason="no feasible stop plan")

    best = min(costs, key=costs.get)

    # A flat cost curve means degradation does not determine this stop.
    #
    # Monaco 2025 is the case that exposed this: 78 laps of the lowest-energy
    # circuit on the calendar, near-zero degradation, overtaking essentially
    # impossible, and a mandatory two-stop rule the model knows nothing about.
    # The cost curve there is almost level, so its argmin is noise -- and the
    # optimiser reported it with the same confidence it reports a real minimum.
    # Mean error across those 29 stops was 31 laps against 9 laps elsewhere.
    #
    # Refusing to answer is the correct behaviour, not a way of dodging a hard
    # circuit. The whole claim of this project is that a number comes with an
    # honest statement of whether it means anything, and a recommendation drawn
    # from a level curve does not.
    spread = max(costs.values()) - costs[best]
    if rate_is_noise:
        return PitRecommendation(
            int(best), 0.0, {}, {k: float(v) for k, v in costs.items()},
            reason=("degradation is not distinguishable from zero on this tyre, so "
                    "the tyre is not what decides this stop"),
        )
    if spread < flat_curve_tolerance_s:
        return PitRecommendation(
            int(best), 0.0, {}, {k: float(v) for k, v in costs.items()},
            reason=(f"degradation does not determine this stop: the best and worst "
                    f"laps differ by {spread:.2f} s, below the {flat_curve_tolerance_s} s "
                    f"threshold. Track position or regulation is deciding, not the tyre."),
        )


    # Confidence: softmax over negative cost. The temperature is the time the
    # rate uncertainty is worth over a typical stint, so a model that does not
    # know the rate cannot claim to know the lap.
    horizon = max(remaining / 2.0, 1.0)
    temperature = max(current_rate_sd * horizon * horizon / 2.0, 1e-3)
    laps_in_play = list(costs)
    values = np.array([-costs[lap] / temperature for lap in laps_in_play], dtype=float)
    values -= values.max()
    weights = np.exp(values)
    weights /= weights.sum()
    distribution = {lap: float(w) for lap, w in zip(laps_in_play, weights)}

    return PitRecommendation(
        lap=int(best),
        confidence=float(distribution[best]),
        distribution=distribution,
        cost_curve={k: float(v) for k, v in costs.items()},
    )


def actual_pit_laps(lap_table: pd.DataFrame) -> dict[str, list[int]]:
    """Laps on which each driver actually boxed, read off the run structure.

    A new `run_id` means a new set of tyres, so the last lap of every run except
    the driver's final one is a stop. This is observed fact, not a model output,
    and it is the ground truth exp22 scores against.

    Runs are ordered by lap rather than by id: `run_id` is an identifier, and
    sorting identifiers numerically would silently reorder a driver's race.
    """
    stops: dict[str, list[int]] = {}
    for driver, block in lap_table.groupby("driver"):
        ordered = block.sort_values("session_lap")
        run_last = ordered.groupby("run_id")["session_lap"].max().sort_values()
        laps = run_last.tolist()
        if len(laps) > 1:
            stops[str(driver)] = [int(lap) for lap in laps[:-1]]
    return stops


def safety_car_laps(lap_table: pd.DataFrame, *, missing_share: float = 0.5) -> set[int]:
    """Session laps the whole field is missing from -- the fingerprint of a safety car.

    The loader removes safety-car laps as outliers, so they are absent from the
    table rather than flagged. Distinguishing them from an ordinary pit stop is a
    matter of *how many cars* lost the lap: a safety car takes the lap away from
    everybody at once, a pit stop takes two laps away from one driver.

    This replaces a per-driver gap test that was wrong in a way that silenced the
    entire experiment. The loader also drops every pit in-lap and out-lap, so
    there is a gap either side of *every* stop, and a per-driver test therefore
    excluded all of them -- 45 out of 45 on the first run.
    """
    drivers = lap_table["driver"].nunique()
    if drivers == 0:
        return set()
    counts = lap_table.groupby("session_lap")["driver"].nunique()
    if counts.empty:
        return set()
    full_range = range(int(counts.index.min()), int(counts.index.max()) + 1)
    return {
        lap for lap in full_range
        if counts.get(lap, 0) < missing_share * drivers
    }


def excluded_stops(lap_table: pd.DataFrame, stops: list[int], *,
                   early_lap_cutoff: int = 3,
                   field_lap_table: pd.DataFrame | None = None) -> set[int]:
    """Stops that should not be scored, because degradation did not drive them.

    Two exclusions, both necessary or the benchmark measures the wrong thing:

    * **Very early stops.** A stop in the opening laps is damage, a puncture or a
      first-lap incident, not a worn tyre.
    * **Stops inside a safety-car window.** A safety-car stop is nearly free, so
      teams take it whatever the tyre is doing. No degradation model should be
      expected to predict one, and scoring them measures race luck.

    Args:
        lap_table: This driver's laps.
        stops: Candidate stop laps.
        early_lap_cutoff: Stops at or before this lap are dropped.
        field_lap_table: The whole session, needed to detect safety cars. When
            omitted only the early-lap rule applies, which is the safe default --
            under-excluding is visible in the results, over-excluding is not.

    Returns:
        The laps to drop.
    """
    dropped = {lap for lap in stops if lap <= early_lap_cutoff}
    if field_lap_table is None:
        return dropped

    sc = safety_car_laps(field_lap_table)
    for lap in stops:
        if any(neighbour in sc for neighbour in (lap - 1, lap, lap + 1, lap + 2)):
            dropped.add(lap)
    return dropped
