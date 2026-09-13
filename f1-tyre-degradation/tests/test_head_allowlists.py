"""Section 1.1: input restriction is what creates identifiability. This test
fails if any head is (or becomes, after a future edit) wired to a column
outside its permitted allowlist -- in particular that EvoHead never sees a
driver/team/car feature, and that TyreHead and FuelHead never overlap."""

from __future__ import annotations

import pytest

from src.models.heads import CircuitHead, CondHead, EntryHead, EvoHead, FuelHead, TrafficHead, TyreHead
from src.models.nam import HEAD_REGISTRY, NAM, NAMConfig


def _make_cfg() -> NAMConfig:
    return NAMConfig(
        n_circuits=5, n_entries=10, n_compounds=3, max_age=45, fuel_time_per_kg_s=0.03,
    )


def test_nam_constructs_with_correct_allowlists():
    model = NAM(_make_cfg())
    assert set(model.heads.keys()) == set(HEAD_REGISTRY.keys())


def test_evo_head_never_receives_driver_or_team_or_car():
    for forbidden in ("driver_id", "team_id", "entry_id", "car_id"):
        assert forbidden not in EvoHead.ALLOWED_COLUMNS
    assert EvoHead.ALLOWED_COLUMNS == frozenset({"session_progress", "circuit_id"})


def test_tyre_head_is_the_only_head_with_tyre_age():
    all_heads = [CircuitHead, EntryHead, FuelHead, TyreHead, EvoHead, CondHead, TrafficHead]
    holders = [h for h in all_heads if "tyre_age" in h.ALLOWED_COLUMNS]
    assert holders == [TyreHead]


def test_fuel_and_tyre_heads_never_share_a_column():
    # "do NOT give any head both lap_in_stint and fuel_kg" -- the concrete form
    # here is that FuelHead's fuel_kg_rel and TyreHead's tyre_age/lap_in_stint
    # inputs must be disjoint sets.
    assert FuelHead.ALLOWED_COLUMNS.isdisjoint(TyreHead.ALLOWED_COLUMNS)


def test_registry_matches_every_head_exactly():
    declared = {
        "circuit": CircuitHead.ALLOWED_COLUMNS,
        "entry": EntryHead.ALLOWED_COLUMNS,
        "fuel": FuelHead.ALLOWED_COLUMNS,
        "tyre": TyreHead.ALLOWED_COLUMNS,
        "evo": EvoHead.ALLOWED_COLUMNS,
        "cond": CondHead.ALLOWED_COLUMNS,
        "traffic": TrafficHead.ALLOWED_COLUMNS,
    }
    assert declared == dict(HEAD_REGISTRY)


def test_nam_refuses_a_head_wired_outside_its_allowlist(monkeypatch):
    # Simulate a future edit that widens EvoHead's allowlist to include a
    # driver-level feature -- NAM must refuse to construct, not silently train
    # an unidentifiable model.
    monkeypatch.setitem(HEAD_REGISTRY, "evo", frozenset({"session_progress"}))  # narrower than actual
    with pytest.raises(ValueError, match="outside its permitted allowlist"):
        NAM(_make_cfg())


def test_forward_produces_a_decomposition_and_a_sum():
    import torch

    model = NAM(_make_cfg())
    batch_size = 8
    batch = {
        "circuit_id": torch.zeros(batch_size, dtype=torch.long),
        "entry_id": torch.zeros(batch_size, dtype=torch.long),
        "compound": torch.zeros(batch_size, dtype=torch.long),
        "tyre_age": torch.arange(batch_size, dtype=torch.float32),
        "track_temp": torch.full((batch_size,), 35.0),
        "air_temp": torch.full((batch_size,), 25.0),
        "wind_speed": torch.zeros(batch_size),
        "humidity": torch.full((batch_size,), 50.0),
        "session_progress": torch.linspace(0, 1, batch_size),
        "fuel_kg_rel": torch.zeros(batch_size),
        "gap_ahead_s": torch.full((batch_size,), 5.0),
    }
    out = model(batch)
    expected_keys = set(HEAD_REGISTRY.keys()) | {"lap_time_hat"}
    assert set(out.keys()) == expected_keys
    total = sum(out[name] for name in HEAD_REGISTRY)
    assert torch.allclose(total, out["lap_time_hat"], atol=1e-5)


def test_tyre_curve_anchored_and_monotone_at_init():
    import torch

    model = NAM(_make_cfg())
    curve = model.degradation_curve(compound_id=0, circuit_id=0, track_temp_c=35.0,
                                      device=torch.device("cpu"))
    assert curve[0].item() == pytest.approx(0.0, abs=1e-6)
    diffs = curve[1:] - curve[:-1]
    assert (diffs >= -1e-6).all()  # monotone non-decreasing, even before training


def test_evo_head_anchored_at_zero():
    import torch

    model = NAM(_make_cfg())
    batch_size = 4
    out = model.heads["evo"](
        session_progress=torch.zeros(batch_size), circuit_id=torch.zeros(batch_size, dtype=torch.long)
    )
    assert torch.allclose(out, torch.zeros(batch_size), atol=1e-6)
