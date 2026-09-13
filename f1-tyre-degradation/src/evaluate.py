"""Shared evaluation utilities: cliff detection, curve comparison, MAE. Used by
the Phase 3 recovery test, Phase 4 baseline comparison and Phase 6 race
validation, so the definition of "cliff" and "slope" is the same everywhere.
"""

from __future__ import annotations

import numpy as np


def detect_cliff(curve: np.ndarray, *, window: tuple[int, int] = (1, 8),
                   multiplier: float = 2.5, sustain: int = 3) -> int | None:
    """First age where the per-lap increment exceeds `multiplier` times the
    median increment over `window`, sustained for `sustain` consecutive laps.

    Args:
        curve: `(max_age + 1,)` cumulative degradation curve, `curve[0] == 0`.
        window: `(start, end)` inclusive ages defining the "normal" pre-cliff
            increment, used as the reference median.
        multiplier: How many times the reference median an increment must
            exceed to count as "cliff-like".
        sustain: How many consecutive ages must clear that bar before it's
            reported as a real cliff, not one noisy lap.

    Returns:
        The first age of the sustained run, or None if no cliff is detected.
    """
    increments = np.diff(curve)
    lo, hi = window
    reference = np.median(increments[lo - 1 : hi])  # increments[k-1] is the k-th increment
    if reference <= 0:
        reference = 1e-6
    threshold = multiplier * reference

    is_cliff_like = increments > threshold
    run_length = 0
    for age, flag in enumerate(is_cliff_like, start=1):
        run_length = run_length + 1 if flag else 0
        if run_length >= sustain:
            return age - sustain + 1
    return None


def recovered_slope(curve: np.ndarray, *, window: tuple[int, int] = (1, 8)) -> float:
    """Mean per-lap increment over `window` -- the recovered linear-phase slope."""
    increments = np.diff(curve)
    lo, hi = window
    return float(np.mean(increments[lo - 1 : hi]))


def mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(actual))))
