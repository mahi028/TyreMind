"""Phase 5 — assembly of the Neural Additive Model.

Sums the seven heads' scalar outputs. No shared trunk: `NAM.__init__` validates
every head against its declared `ALLOWED_COLUMNS` and refuses to construct if a
head is wired to a column outside that set -- this is the "assert these
restrictions in code" requirement from Section 1.1, enforced structurally
rather than by convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from src.models.heads import CircuitHead, CondHead, EntryHead, EvoHead, FuelHead, TyreHead, TrafficHead

# The single source of truth for which columns are allowed to reach which head.
# `NAM.__init__` checks every head instance's own `ALLOWED_COLUMNS` against this
# registry too, so a head subclass cannot silently widen its own allowlist
# without this registry being updated in lockstep (checked by
# `tests/test_head_allowlists.py`).
HEAD_REGISTRY: dict[str, frozenset[str]] = {
    "circuit": frozenset({"circuit_id"}),
    "entry": frozenset({"entry_id"}),
    "fuel": frozenset({"fuel_kg_rel"}),
    "tyre": frozenset({"tyre_age", "compound", "circuit_id", "track_temp"}),
    "evo": frozenset({"session_progress", "circuit_id"}),
    "cond": frozenset({"track_temp", "air_temp", "wind_speed", "humidity"}),
    "traffic": frozenset({"gap_ahead_s"}),
}


@dataclass
class NAMConfig:
    n_circuits: int
    n_entries: int
    n_compounds: int
    max_age: int
    fuel_time_per_kg_s: float
    hidden_dim: int = 64
    n_layers: int = 2
    activation: str = "gelu"
    dropout: float = 0.1
    compound_dim: int = 4
    circuit_dim: int = 8
    driver_dim: int = 8  # unused directly (entry embedding is 1-d bias), kept for config parity
    team_dim: int = 4


class NAM(nn.Module):
    """The full additive model. `forward` returns every head's contribution
    plus the summed prediction, so the decomposition is always available --
    it IS the deliverable, not a debugging side effect.
    """

    def __init__(self, cfg: NAMConfig):
        super().__init__()
        self.cfg = cfg

        self.heads = nn.ModuleDict(
            {
                "circuit": CircuitHead(cfg.n_circuits),
                "entry": EntryHead(cfg.n_entries),
                "fuel": FuelHead(cfg.fuel_time_per_kg_s),
                "tyre": TyreHead(
                    cfg.n_compounds, cfg.n_circuits, cfg.max_age,
                    compound_dim=cfg.compound_dim, circuit_dim=cfg.circuit_dim,
                    hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers,
                    activation=cfg.activation, dropout=cfg.dropout,
                ),
                "evo": EvoHead(cfg.n_circuits),
                "cond": CondHead(
                    hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers,
                    activation=cfg.activation, dropout=cfg.dropout,
                ),
                "traffic": TrafficHead(
                    hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers,
                    activation=cfg.activation, dropout=cfg.dropout,
                ),
            }
        )
        self._validate_allowlists()

    def _validate_allowlists(self) -> None:
        """Refuses to construct if any head's declared inputs exceed what the
        identifiability contract in Section 1.1 permits for that head name."""
        for name, head in self.heads.items():
            declared = getattr(head, "ALLOWED_COLUMNS", None)
            if declared is None:
                raise ValueError(f"head {name!r} does not declare ALLOWED_COLUMNS")
            permitted = HEAD_REGISTRY.get(name)
            if permitted is None:
                raise ValueError(f"head {name!r} is not in HEAD_REGISTRY -- add it there first")
            if not declared <= permitted:
                extra = declared - permitted
                raise ValueError(
                    f"head {name!r} declares columns {sorted(extra)} outside its permitted "
                    f"allowlist {sorted(permitted)}. This would break tyre/fuel/evolution "
                    f"identifiability -- see Section 1.1 of the build spec."
                )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        contributions: dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            allowed = HEAD_REGISTRY[name]
            kwargs = {col: batch[col] for col in allowed}
            contributions[name] = head(**kwargs)

        total = torch.zeros_like(next(iter(contributions.values())))
        for value in contributions.values():
            total = total + value
        contributions["lap_time_hat"] = total
        return contributions

    @torch.no_grad()
    def warm_start_circuit_bias(self, mean_lap_time_by_circuit_idx: dict[int, float],
                                  global_mean_lap_time: float) -> None:
        """Initialise `CircuitHead`'s bias from data instead of zero.

        `b_circuit` and `b_entry` sum to roughly a lap time in seconds (75-110s
        for a real circuit). Adam's per-step update is roughly LR-sized
        regardless of gradient magnitude, so climbing from a zero init to ~100
        would take on the order of 100/lr steps -- tens of thousands of them at
        lr=1e-3. Starting from each circuit's observed mean lap time removes
        that climb entirely; training then only has to learn the much smaller
        residual structure (tyre, fuel, evolution, conditions, traffic), which
        is the point of the decomposition in the first place.
        """
        weight = self.heads["circuit"].bias.weight
        weight.fill_(global_mean_lap_time)
        for idx, mean_lap_time in mean_lap_time_by_circuit_idx.items():
            weight[idx] = mean_lap_time

    @torch.no_grad()
    def degradation_curve(self, compound_id: int, circuit_id: int, track_temp_c: float,
                            device: torch.device | None = None) -> torch.Tensor:
        """`(max_age + 1,)` cumulative seconds lost vs. a fresh tyre.

        This is the deliverable API named in Section 7:
        `model.degradation_curve(compound='SOFT', circuit='Silverstone', track_temp=38.0)`
        (string names are resolved to ids by the caller via the fitted vocab).
        """
        device = device or next(self.parameters()).device
        tyre_head: TyreHead = self.heads["tyre"]  # type: ignore[assignment]
        return tyre_head.full_curve(compound_id, circuit_id, track_temp_c, device)

    def tyre_smoothness_penalty(self, compound: torch.Tensor, circuit_id: torch.Tensor,
                                 track_temp: torch.Tensor) -> torch.Tensor:
        """Penalises the second difference of the tyre head's raw per-lap
        increments across the age dimension -- see `TyreHead.increments_for`'s
        docstring for why this is needed: nothing else stops adjacent ages
        from being wildly different, since the conditioning vector carries no
        age information at all."""
        tyre_head: TyreHead = self.heads["tyre"]  # type: ignore[assignment]
        increments = tyre_head.increments_for(compound, circuit_id, track_temp)
        second_diff = increments[:, 2:] - 2 * increments[:, 1:-1] + increments[:, :-2]
        return (second_diff**2).mean()

    def tyre_l1_penalty(self, compound: torch.Tensor, circuit_id: torch.Tensor,
                          track_temp: torch.Tensor) -> torch.Tensor:
        """L1 penalty on the tyre head's raw per-lap increments (already >= 0
        via softplus, so the L1 norm is just their mean) -- the direct
        countermeasure for the monotone parameterisation's one-sided bias under
        limited data: every increment is >= 0 by construction, so noise can
        only ever accumulate upward, never cancel. The smoothness penalty
        (`tyre_smoothness_penalty`) stops adjacent increments from swinging
        wildly relative to each other, but does nothing to stop the whole
        curve sitting uniformly too high -- that is what this penalises."""
        tyre_head: TyreHead = self.heads["tyre"]  # type: ignore[assignment]
        increments = tyre_head.increments_for(compound, circuit_id, track_temp)
        return increments.mean()

    def traffic_zero_penalty(self, gap_ahead_s: torch.Tensor, zero_penalty_gap_s: float) -> torch.Tensor:
        """Penalty pulling `g_traffic -> 0` for laps run in clean air. Added to
        the training loss, not baked into the head's forward pass."""
        g = self.heads["traffic"](gap_ahead_s)
        mask = (gap_ahead_s > zero_penalty_gap_s).float()
        if mask.sum() == 0:
            return torch.zeros((), device=gap_ahead_s.device)
        return ((g**2) * mask).sum() / mask.sum().clamp_min(1.0)
