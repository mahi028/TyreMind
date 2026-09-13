"""Phase 2 — feature construction and splitting utilities shared between the
synthetic recovery test and real-data training.

Splitting: split by EVENT, never by row (Section 4 "Scaling and splits"). Laps
from the same stint appearing in both train and val is trivial leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_by_event(
    df: pd.DataFrame, *, val_fraction: float = 0.15, seed: int = 42, event_col: str = "event_id"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random split of whole EVENTS into train/val, never splitting a stint.

    Args:
        df: Lap table with an `event_col` identifying which laps belong to the
            same event/session.
        val_fraction: Fraction of *events* (not rows) held out for validation.
        seed: Fixed seed so the split is reproducible.
        event_col: Column identifying the event.

    Returns:
        `(train_df, val_df)`, each with a fresh `RangeIndex`.
    """
    events = df[event_col].unique()
    rng = np.random.default_rng(seed)
    shuffled = events.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_events = set(shuffled[:n_val].tolist())
    train_mask = ~df[event_col].isin(val_events)
    return df[train_mask].reset_index(drop=True), df[~train_mask].reset_index(drop=True)


def split_train_val_test(
    df: pd.DataFrame, *, train_seasons: list[int], test_season: int,
    val_fraction: float = 0.15, seed: int = 42,
    season_col: str = "season", event_col: str = "event_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The real-data three-way split (Section 4):
    train = all `train_seasons` events except a random val fraction of them;
    val   = that held-out fraction;
    test  = all of `test_season`, untouched until Phase 6.
    """
    train_pool = df[df[season_col].isin(train_seasons)]
    test_df = df[df[season_col] == test_season].reset_index(drop=True)
    train_df, val_df = split_by_event(train_pool, val_fraction=val_fraction, seed=seed, event_col=event_col)
    return train_df, val_df, test_df
