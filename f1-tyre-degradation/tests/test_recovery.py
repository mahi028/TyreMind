"""Phase 3 — the scientific backbone. Fits the full model on synthetic data
with a KNOWN hidden degradation curve and asserts it is recovered within
tolerance. Do NOT proceed to Phase 5 on real data until this passes, and do
NOT relax these thresholds to make it pass -- fix the model instead (Section 5
of the build spec).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.data.features import split_by_event
from src.evaluate import detect_cliff, recovered_slope
from src.models.dataset import Vocabs
from src.models.nam import NAM, NAMConfig
from src.sim.simulator import SimulatorConfig, generate_synthetic_dataset
from src.train import TrainConfig, fit, pick_device


@pytest.fixture(scope="module")
def raw_config() -> dict:
    return yaml.safe_load(open("config/default.yaml"))


@pytest.fixture(scope="module")
def sim_cfg(raw_config) -> SimulatorConfig:
    return SimulatorConfig.from_yaml()


def _fit_on(df: pd.DataFrame, sim_cfg, raw_config, *, seed: int = 42):
    """Split, build vocabs, construct and fit a NAM. Returns (model, vocabs, val_df).

    Seeds torch BEFORE constructing the model -- `NAM(nam_cfg)` initialises every
    head's weights using whatever the global torch RNG state happens to be at
    that moment, and `fit()`'s own `torch.manual_seed(cfg.seed)` call happens too
    late to affect that. Without this, "the same config" silently produces a
    different random initialisation every run, which was confirmed directly:
    two CPU runs with identical seed/config/data still gave different results
    end to end (e.g. a detected cliff of 31 vs 33 laps).
    """
    torch.manual_seed(seed)
    train_df, val_df = split_by_event(df, val_fraction=0.15, seed=seed)
    vocabs = Vocabs.fit(train_df)
    nam_cfg = NAMConfig(
        n_circuits=vocabs.circuit.size, n_entries=vocabs.entry.size,
        n_compounds=vocabs.compound.size, max_age=sim_cfg.max_age,
        fuel_time_per_kg_s=sim_cfg.fuel_time_per_kg_s,
    )
    model = NAM(nam_cfg)
    train_cfg = TrainConfig.from_yaml_dict(raw_config)
    history = fit(model, train_df, val_df, vocabs, train_cfg, device=pick_device(), verbose=False)
    return model, vocabs, val_df, history


@pytest.fixture(scope="module")
def fitted(sim_cfg, raw_config):
    df, truth = generate_synthetic_dataset(sim_cfg, seed=42, return_truth=True)
    model, vocabs, val_df, history = _fit_on(df, sim_cfg, raw_config)
    return model, vocabs, truth, history


def test_recovers_known_slope_per_compound(fitted, sim_cfg, raw_config):
    model, vocabs, truth, history = fitted
    acceptance = raw_config["acceptance"]
    device = pick_device()

    max_diff = 0.0
    for compound in sim_cfg.compounds:
        compound_id = vocabs.compound.mapping[compound]
        circuit_id = next(iter(vocabs.circuit.mapping.values()))
        curve = model.degradation_curve(
            compound_id=compound_id, circuit_id=circuit_id, track_temp_c=35.0, device=device
        ).cpu().numpy()

        rec_slope = recovered_slope(curve)
        true_slope = sim_cfg.true_linear_slope_s_per_lap[compound]
        diff = abs(rec_slope - true_slope)
        max_diff = max(max_diff, diff)
        print(f"{compound}: recovered={rec_slope:.4f} true={true_slope:.4f} diff={diff:.4f}")

    assert max_diff <= acceptance["slope_tolerance_s_per_lap"], (
        f"worst-case slope error {max_diff:.4f} s/lap exceeds "
        f"{acceptance['slope_tolerance_s_per_lap']} s/lap tolerance"
    )


def test_recovers_cliff_age_per_compound(fitted, sim_cfg, raw_config):
    model, vocabs, truth, history = fitted
    acceptance = raw_config["acceptance"]
    device = pick_device()

    for compound in sim_cfg.compounds:
        compound_id = vocabs.compound.mapping[compound]
        circuit_id = next(iter(vocabs.circuit.mapping.values()))
        curve = model.degradation_curve(
            compound_id=compound_id, circuit_id=circuit_id, track_temp_c=35.0, device=device
        ).cpu().numpy()

        detected = detect_cliff(curve)
        true_cliff = sim_cfg.true_cliff_age[compound]
        assert detected is not None, f"{compound}: no cliff detected (true cliff at {true_cliff})"
        diff = abs(detected - true_cliff)
        print(f"{compound}: detected_cliff={detected} true_cliff={true_cliff} diff={diff}")
        assert diff <= acceptance["cliff_tolerance_laps"], (
            f"{compound}: detected cliff at {detected}, true is {true_cliff} "
            f"(tolerance {acceptance['cliff_tolerance_laps']} laps)"
        )


def test_recovers_track_evolution_shape(fitted, raw_config):
    model, vocabs, truth, history = fitted
    acceptance = raw_config["acceptance"]
    device = pick_device()

    grid = np.linspace(0.0, 1.0, 50)
    correlations = []
    with torch.no_grad():
        for raw_circuit_id in list(vocabs.circuit.mapping.keys())[:4]:
            dense = vocabs.circuit.mapping[raw_circuit_id]
            session_progress = torch.tensor(grid, dtype=torch.float32, device=device)
            circuit_ids = torch.full((50,), dense, dtype=torch.long, device=device)
            recovered = model.heads["evo"](
                session_progress=session_progress, circuit_id=circuit_ids
            ).cpu().numpy()

            amplitude = truth["circuit_evo_amplitude"][raw_circuit_id]
            tau = truth["circuit_evo_tau"][raw_circuit_id]
            true_evo = -amplitude * (1 - np.exp(-grid / tau))

            corr = float(np.corrcoef(recovered, true_evo)[0, 1])
            correlations.append(corr)
            print(f"circuit {raw_circuit_id}: evo correlation = {corr:.4f}")

    assert min(correlations) >= acceptance["evolution_correlation_min"], (
        f"worst-case evolution correlation {min(correlations):.3f} below "
        f"{acceptance['evolution_correlation_min']}"
    )


def test_negative_control_zero_degradation_recovers_flat_curve(sim_cfg, raw_config):
    """A model that finds degradation where there is none is worse than useless."""
    acceptance = raw_config["acceptance"]
    df = generate_synthetic_dataset(sim_cfg, seed=7, zero_degradation=True)
    model, vocabs, val_df, history = _fit_on(df, sim_cfg, raw_config, seed=7)
    device = pick_device()

    max_slope = 0.0
    for compound in sim_cfg.compounds:
        compound_id = vocabs.compound.mapping[compound]
        circuit_id = next(iter(vocabs.circuit.mapping.values()))
        curve = model.degradation_curve(
            compound_id=compound_id, circuit_id=circuit_id, track_temp_c=35.0, device=device
        ).cpu().numpy()
        slope = recovered_slope(curve)
        max_slope = max(max_slope, slope)
        print(f"{compound} (zero-degradation truth): recovered slope = {slope:.4f}")

    assert max_slope <= acceptance["negative_control_slope_max_s_per_lap"], (
        f"model found {max_slope:.4f} s/lap of degradation where the truth is zero "
        f"(limit {acceptance['negative_control_slope_max_s_per_lap']})"
    )
