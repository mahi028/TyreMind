"""Phase 7 — strategy tool. Turns a fitted degradation curve into a decision,
so it is load-bearing rather than decorative: cumulative time lost vs. stint
length, and the optimal pit-stop count/laps for a given race distance and pit
loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cumulative_time_lost(curve: np.ndarray, stint_length: int) -> float:
    """Total seconds lost to degradation across a stint of `stint_length` laps
    on a fresh tyre, i.e. `sum_{k=1}^{stint_length} curve[k]` -- NOT `curve[stint_length]`,
    which is only the LAST lap's instantaneous cost. A strategist cares about
    the total time given back over the whole stint.
    """
    return float(np.sum(curve[1 : stint_length + 1]))


def _min_degradation_cost(curve: np.ndarray, race_distance: int, n_stints: int,
                            max_stint_len: int) -> tuple[float, list[int]] | None:
    """Dynamic program: minimum total degradation cost (sum of per-stint
    `cumulative_time_lost`) splitting `race_distance` laps into exactly
    `n_stints` stints, each at most `max_stint_len` laps (the curve's domain).

    Returns `(cost, stint_lengths)` or None if infeasible (e.g. race_distance
    too long for `n_stints` stints of at most `max_stint_len` laps each).
    """
    if n_stints <= 0 or race_distance <= 0:
        return None
    if race_distance > n_stints * max_stint_len:
        return None

    # dp[laps_left][stints_left] = (best_cost, first_choice_length)
    dp: dict[tuple[int, int], tuple[float, int]] = {}

    def solve(laps_left: int, stints_left: int) -> float:
        if stints_left == 1:
            if laps_left > max_stint_len or laps_left <= 0:
                return float("inf")
            return cumulative_time_lost(curve, laps_left)
        key = (laps_left, stints_left)
        if key in dp:
            return dp[key][0]
        best_cost = float("inf")
        best_len = 1
        max_len_here = min(max_stint_len, laps_left - (stints_left - 1))
        for length in range(1, max_len_here + 1):
            remainder_cost = solve(laps_left - length, stints_left - 1)
            if remainder_cost == float("inf"):
                continue
            cost = cumulative_time_lost(curve, length) + remainder_cost
            if cost < best_cost:
                best_cost = cost
                best_len = length
        dp[key] = (best_cost, best_len)
        return best_cost

    total_cost = solve(race_distance, n_stints)
    if total_cost == float("inf"):
        return None

    # Reconstruct the stint lengths from the memo table.
    lengths = []
    laps_left, stints_left = race_distance, n_stints
    while stints_left > 1:
        _, length = dp[(laps_left, stints_left)]
        lengths.append(length)
        laps_left -= length
        stints_left -= 1
    lengths.append(laps_left)
    return total_cost, lengths


@dataclass
class StrategyOption:
    n_pit_stops: int
    stint_lengths: list[int]
    degradation_cost_s: float
    pit_cost_s: float

    @property
    def total_extra_time_s(self) -> float:
        return self.degradation_cost_s + self.pit_cost_s


def evaluate_strategies(curve: np.ndarray, race_distance: int, pit_loss_s: float, *,
                          max_pit_stops: int = 3) -> list[StrategyOption]:
    """Every feasible strategy from a 0-stop (impossible unless the whole race
    fits under `max_stint_len`) up to `max_pit_stops` pit stops, ranked.

    Args:
        curve: `(max_age + 1,)` cumulative degradation curve, single compound
            assumed for every stint (a multi-compound search is a larger
            combinatorial problem, out of scope here).
        race_distance: Total laps.
        pit_loss_s: Time lost per pit stop, seconds.
        max_pit_stops: Search up to this many stops (default covers 0-3 stops).

    Returns:
        Feasible `StrategyOption`s sorted by total extra time, best first.
    """
    max_stint_len = len(curve) - 1
    options: list[StrategyOption] = []
    for n_stops in range(0, max_pit_stops + 1):
        n_stints = n_stops + 1
        result = _min_degradation_cost(curve, race_distance, n_stints, max_stint_len)
        if result is None:
            continue
        cost, lengths = result
        options.append(
            StrategyOption(
                n_pit_stops=n_stops, stint_lengths=lengths,
                degradation_cost_s=cost, pit_cost_s=n_stops * pit_loss_s,
            )
        )
    return sorted(options, key=lambda o: o.total_extra_time_s)


def one_vs_two_stop_crossover_pit_loss(curve: np.ndarray, race_distance: int) -> float | None:
    """The pit-loss value at which a one-stop and a two-stop strategy cost the
    same. Below it, two-stop wins (the extra stop is cheap enough to be worth
    the fresher tyres); above it, one-stop wins.

    Returns:
        The crossover pit loss in seconds, or None if either strategy is
        infeasible for this `race_distance`/curve combination.
    """
    max_stint_len = len(curve) - 1
    one_stop = _min_degradation_cost(curve, race_distance, 2, max_stint_len)
    two_stop = _min_degradation_cost(curve, race_distance, 3, max_stint_len)
    if one_stop is None or two_stop is None:
        return None
    one_stop_cost, _ = one_stop
    two_stop_cost, _ = two_stop
    # one_stop_cost + 1*pit_loss == two_stop_cost + 2*pit_loss  =>  pit_loss == one_stop_cost - two_stop_cost
    return one_stop_cost - two_stop_cost
