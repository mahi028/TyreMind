"""Categorical encoding and tensor-batch construction, shared by the synthetic
recovery test and real-data training.

Unseen categoricals at test time map to a learned `<UNK>` embedding: every
vocab reserves index `len(vocab)` for exactly that, and `encode` below maps
anything not in the fitted vocab to it rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch


def _entry_key(df: pd.DataFrame) -> pd.Series:
    """`b_entry` is keyed at (event, driver, STINT) granularity, not just
    (event, driver). See Section 1.2: a driver's true remaining fuel at the
    start of a stint differs stint to stint (no refuelling since 2010), but
    `fuel_kg_rel` resets to 0 at every pit stop and carries no absolute-level
    information. Section 1.2 argues that offset "is absorbed by b_entry" -- true
    only if a driver has a single stint per event. With multiple stints, a
    single event-driver constant is the MEAN across stints, leaving a residual
    that is systematically larger early in a race (heavy fuel) and smaller late
    (light fuel) -- and because that residual correlates with session_progress,
    it was observed leaking into `EvoHead` as a spurious, over-large
    "evolution" effect, which in turn biased the recovered tyre slope upward by
    a near-constant amount across every compound. Keying the entry embedding
    per stint removes the mismatch at its source rather than papering over it
    downstream.
    """
    return (
        df["event_id"].astype(str) + "_" + df["driver_id"].astype(str) + "_" + df["stint_id"].astype(str)
    )


@dataclass
class Vocab:
    """A fitted (train-only) id -> dense-index mapping with a reserved <UNK>."""

    mapping: dict

    @classmethod
    def fit(cls, values: pd.Series) -> "Vocab":
        uniques = sorted(values.dropna().unique().tolist())
        return cls(mapping={v: i for i, v in enumerate(uniques)})

    @property
    def unk_index(self) -> int:
        return len(self.mapping)

    @property
    def size(self) -> int:
        return len(self.mapping)  # UNK is the +1 handled by the embedding table

    def encode(self, values: pd.Series) -> np.ndarray:
        return values.map(lambda v: self.mapping.get(v, self.unk_index)).to_numpy(dtype=np.int64)


@dataclass
class Vocabs:
    circuit: Vocab
    entry: Vocab
    compound: Vocab

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Vocabs":
        entry_key = _entry_key(df)
        return cls(
            circuit=Vocab.fit(df["circuit_id"]),
            entry=Vocab.fit(entry_key),
            compound=Vocab.fit(df["compound"]),
        )


FEATURE_COLUMNS = [
    "circuit_id", "tyre_age", "compound", "track_temp", "session_progress",
    "fuel_kg_rel", "air_temp", "wind_speed", "humidity", "gap_ahead_s",
]


def make_batch(df: pd.DataFrame, vocabs: Vocabs, device: torch.device) -> dict[str, torch.Tensor]:
    """Build the tensor dict `NAM.forward` expects from a lap-table dataframe."""
    entry_key = _entry_key(df)
    return {
        "circuit_id": torch.tensor(vocabs.circuit.encode(df["circuit_id"]), device=device),
        "entry_id": torch.tensor(vocabs.entry.encode(entry_key), device=device),
        "compound": torch.tensor(vocabs.compound.encode(df["compound"]), device=device),
        "tyre_age": torch.tensor(df["tyre_age"].to_numpy(dtype=np.float32), device=device),
        "track_temp": torch.tensor(df["track_temp"].to_numpy(dtype=np.float32), device=device),
        "air_temp": torch.tensor(df["air_temp"].to_numpy(dtype=np.float32), device=device),
        "wind_speed": torch.tensor(df["wind_speed"].to_numpy(dtype=np.float32), device=device),
        "humidity": torch.tensor(df["humidity"].to_numpy(dtype=np.float32), device=device),
        "session_progress": torch.tensor(df["session_progress"].to_numpy(dtype=np.float32), device=device),
        "fuel_kg_rel": torch.tensor(df["fuel_kg_rel"].to_numpy(dtype=np.float32), device=device),
        "gap_ahead_s": torch.tensor(df["gap_ahead_s"].to_numpy(dtype=np.float32), device=device),
    }


def make_target(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    return torch.tensor(df["lap_time"].to_numpy(dtype=np.float32), device=device)
