"""Phase 5 — the additive heads.

Each head is a small MLP (or, for fuel, a fixed non-learned linear map) emitting
a scalar **in seconds**. `nam.py` sums them. There is no shared trunk: every
head declares `ALLOWED_COLUMNS`, the complete set of feature names it is
permitted to see, and `nam.py` enforces that at construction time (see
`tests/test_head_allowlists.py`).

Identifying restrictions, restated here because they are the reason this file
looks the way it does:
  - `TyreHead` is the ONLY head that receives `tyre_age`.
  - `EvoHead` receives `session_progress` and `circuit_id` ONLY -- never driver,
    team or car. Track evolution is shared by the whole field at a given
    wall-clock moment; tyre age is car-specific. That asymmetry is the whole
    identification strategy.
  - No head receives both `lap_in_stint`-derived and `fuel_kg`-derived features.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

_ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU}


class MLP(nn.Module):
    """A small 2-hidden-layer MLP. Width 64 is plenty -- this is an additive
    model, each head sees a handful of scalars, not a deep feature extractor."""

    def __init__(self, in_dim: int, out_dim: int, *, hidden_dim: int = 64,
                 n_layers: int = 2, activation: str = "gelu", dropout: float = 0.1):
        super().__init__()
        act_cls = _ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden_dim), act_cls(), nn.Dropout(dropout)]
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CircuitHead(nn.Module):
    """`b_circuit[circuit]` -- a learned scalar per circuit, absorbing track length."""

    ALLOWED_COLUMNS = frozenset({"circuit_id"})

    def __init__(self, n_circuits: int):
        super().__init__()
        # +1 reserved row for an unseen-circuit <UNK> embedding at test time.
        self.bias = nn.Embedding(n_circuits + 1, 1)
        nn.init.zeros_(self.bias.weight)

    def forward(self, circuit_id: torch.Tensor) -> torch.Tensor:
        return self.bias(circuit_id).squeeze(-1)


class EntryHead(nn.Module):
    """`b_entry[event, driver, stint]` -- car/driver base pace for one stint.

    `entry_id` is a single categorical precomputed from the (event_id,
    driver_id, stint_id) triple by `src/models/dataset.py::_entry_key`; this
    head never sees those columns separately, only the joint id. Keyed per
    STINT rather than per event -- see `_entry_key`'s docstring for why a
    single per-event constant is not enough once a driver has multiple stints.
    """

    ALLOWED_COLUMNS = frozenset({"entry_id"})

    def __init__(self, n_entries: int):
        super().__init__()
        self.bias = nn.Embedding(n_entries + 1, 1)
        nn.init.zeros_(self.bias.weight)

    def forward(self, entry_id: torch.Tensor) -> torch.Tensor:
        return self.bias(entry_id).squeeze(-1)


class FuelHead(nn.Module):
    """`g_fuel(fuel_kg) = FUEL_TIME_PER_KG * fuel_kg` -- FIXED, no trainable
    parameters. See config `fuel.time_per_kg_s`. Section 1.2: a free fuel
    coefficient and a free tyre curve are near-perfectly collinear within a
    stint, so this coefficient must be pinned, not learned.

    Input is `fuel_kg_rel = -FUEL_BURN_PER_LAP * lap_in_stint`, already relative
    (see `features.py`); the unknown absolute start-of-stint fuel is a constant
    that `EntryHead` (and, for multi-stint drivers, residual noise) absorbs.
    """

    ALLOWED_COLUMNS = frozenset({"fuel_kg_rel"})

    def __init__(self, time_per_kg_s: float):
        super().__init__()
        self.register_buffer("coef", torch.tensor(float(time_per_kg_s)))

    def forward(self, fuel_kg_rel: torch.Tensor) -> torch.Tensor:
        return self.coef * fuel_kg_rel


class TyreHead(nn.Module):
    """`g_tyre(tyre_age | compound, circuit, track_temp)` -- cumulative-increment
    parameterisation (Section 1.3). NEVER write `MLP(tyre_age)` directly; an
    unconstrained curve will happily learn a non-monotone shape that absorbs
    noise instead of reporting wear.

    ``increments = softplus(MLP(c)) >= 0``
    ``curve = cumsum(increments)``, prepended with 0 so ``curve[0] == 0``
    ``g_tyre = curve[tyre_age]``

    This guarantees monotonicity (tyres never get faster with age) and anchors
    the head at zero for a fresh tyre, so it cannot steal a constant that
    belongs to `EntryHead` or `CircuitHead`.
    """

    ALLOWED_COLUMNS = frozenset({"tyre_age", "compound", "circuit_id", "track_temp"})

    def __init__(self, n_compounds: int, n_circuits: int, max_age: int, *,
                 compound_dim: int = 4, circuit_dim: int = 8, hidden_dim: int = 64,
                 n_layers: int = 2, activation: str = "gelu", dropout: float = 0.1,
                 track_temp_ref_c: float = 35.0, track_temp_scale_c: float = 10.0):
        super().__init__()
        self.max_age = max_age
        self.track_temp_ref_c = track_temp_ref_c
        self.track_temp_scale_c = track_temp_scale_c
        self.compound_emb = nn.Embedding(n_compounds + 1, compound_dim)
        self.circuit_emb = nn.Embedding(n_circuits + 1, circuit_dim)
        self.mlp = MLP(
            compound_dim + circuit_dim + 1, max_age,
            hidden_dim=hidden_dim, n_layers=n_layers, activation=activation, dropout=dropout,
        )
        # softplus(0) = ln(2) ~= 0.69, not 0 -- with the MLP's ordinary small-random
        # init, every one of the max_age increments starts around 0.69s regardless,
        # giving a substantial, wrong ~0.69*max_age curve before any training at
        # all. Biasing the final layer sharply negative makes softplus(pre-activation)
        # ~= 0 at init instead, so the curve starts flat and has to be pushed up by
        # evidence rather than the reverse. Diagnosed as a likely cause of ~25% of
        # random inits failing to train at all (best_epoch=1, next to zero movement
        # in other heads) at n~13,000 rows.
        nn.init.constant_(self.mlp.net[-1].bias, -5.0)

    def increments_for(self, compound: torch.Tensor, circuit_id: torch.Tensor,
                        track_temp: torch.Tensor) -> torch.Tensor:
        """The raw `(batch, max_age)` per-lap increments, before `cumsum`.

        Exposed separately from `forward` so training can add a smoothness
        penalty across the age dimension (see `train.py`) -- without one,
        nothing in Section 1.3's architecture stops adjacent ages from being
        wildly different: the conditioning vector `c` has no age input at all,
        so the final linear layer's `max_age` output columns are only as
        smooth as an unconstrained MLP happens to make them. Verified
        empirically: without this penalty the fitted curve's per-lap
        increments swing between ~0 and ~5x the true value lap to lap, and
        only the *average* over several laps lands near the true slope --
        which passes a slope-tolerance check while the curve itself is
        nonsense (and fools cliff detection, which needs the real shape).
        """
        temp_norm = ((track_temp - self.track_temp_ref_c) / self.track_temp_scale_c).unsqueeze(-1)
        c = torch.cat([self.compound_emb(compound), self.circuit_emb(circuit_id), temp_norm], dim=-1)
        return F.softplus(self.mlp(c))  # (batch, max_age), all >= 0

    def forward(self, tyre_age: torch.Tensor, compound: torch.Tensor,
                circuit_id: torch.Tensor, track_temp: torch.Tensor) -> torch.Tensor:
        increments = self.increments_for(compound, circuit_id, track_temp)
        curve = torch.cumsum(increments, dim=-1)
        curve = F.pad(curve, (1, 0))  # curve[:, 0] == 0
        age_idx = tyre_age.long().clamp(0, self.max_age).unsqueeze(-1)
        return curve.gather(-1, age_idx).squeeze(-1)

    def full_curve(self, compound_id: int, circuit_id: int, track_temp: float,
                    device: torch.device) -> torch.Tensor:
        """The whole `(max_age + 1,)` curve for one (compound, circuit, temp) —
        the API `model.degradation_curve(...)` in `nam.py` calls this."""
        temp_norm = torch.tensor(
            [(track_temp - self.track_temp_ref_c) / self.track_temp_scale_c], device=device
        ).unsqueeze(-1)
        c_emb = self.compound_emb(torch.tensor([compound_id], device=device))
        ci_emb = self.circuit_emb(torch.tensor([circuit_id], device=device))
        c = torch.cat([c_emb, ci_emb, temp_norm], dim=-1)
        increments = F.softplus(self.mlp(c))
        curve = torch.cumsum(increments, dim=-1)
        curve = F.pad(curve, (1, 0))
        return curve.squeeze(0)


class EvoHead(nn.Module):
    """`g_evo(session_progress | circuit)` -- NEVER given driver, team or car.

    Track evolution/rubber-in is shared by the whole field at a given wall-clock
    moment; that is the identifying asymmetry against car-specific tyre age.

    Parameterised as a **saturating exponential with 2 learned scalars per
    circuit** -- ``g_evo(t) = -softplus(amplitude) * (1 - exp(-t / softplus(tau)))``
    -- rather than a free MLP. This matches the true generative shape (see
    `src/sim/simulator.py`) and, more importantly, matches the fix the sibling
    TyreMind project converged on for the identical collinearity: within one
    stint, `session_progress` and `tyre_age` move together almost linearly, so
    an unconstrained MLP has enough capacity to "explain" part of the true
    tyre-degradation trend as spurious evolution (verified empirically: a free
    2-layer MLP here recovered an evolution amplitude 3-5x too large and
    contaminated the recovered tyre slope by a roughly constant offset across
    every compound).

    Even with only 2 parameters per circuit, the recovered amplitude still
    settles ~2-3x the true value, stably -- not shrinking with more training
    data (tested at 40, 80 and 100 sessions), which is the signature of a real
    structural bias rather than noise. Two counter-measures were tried and
    both made every metric worse, not better, so neither is in this code:
    an L2 shrinkage penalty toward zero (overshoots past the true nonzero
    amplitude in the wrong direction), and a hard sigmoid ceiling on amplitude
    (fixed the bias but broke cliff detection for the HARD compound outright).
    This remains an open item -- see `my_works/PROGRESS.md`.

    Exactly anchored at `g_evo(0) == 0` for any parameter values (not just
    post-hoc), since `1 - exp(0) == 0`.
    """

    ALLOWED_COLUMNS = frozenset({"session_progress", "circuit_id"})

    def __init__(self, n_circuits: int, *, tau_floor: float = 0.02, **_ignored):
        super().__init__()
        # +1 reserved row for an unseen-circuit <UNK>.
        self.raw_amplitude = nn.Embedding(n_circuits + 1, 1)
        self.raw_tau = nn.Embedding(n_circuits + 1, 1)
        self.tau_floor = tau_floor
        nn.init.zeros_(self.raw_amplitude.weight)
        nn.init.zeros_(self.raw_tau.weight)

    def forward(self, session_progress: torch.Tensor, circuit_id: torch.Tensor) -> torch.Tensor:
        amplitude = F.softplus(self.raw_amplitude(circuit_id).squeeze(-1))
        tau = F.softplus(self.raw_tau(circuit_id).squeeze(-1)) + self.tau_floor
        return -amplitude * (1.0 - torch.exp(-session_progress / tau))


class CondHead(nn.Module):
    """`g_cond(track_temp, air_temp, wind_speed, humidity)` -- centered so the
    training-set mean of its output is zero (`calibrate_center`), otherwise it
    can absorb a constant that belongs to `CircuitHead`/`EntryHead`.
    """

    ALLOWED_COLUMNS = frozenset({"track_temp", "air_temp", "wind_speed", "humidity"})

    def __init__(self, *, hidden_dim: int = 64, n_layers: int = 2,
                 activation: str = "gelu", dropout: float = 0.1):
        super().__init__()
        self.mlp = MLP(4, 1, hidden_dim=hidden_dim, n_layers=n_layers,
                        activation=activation, dropout=dropout)
        self.register_buffer("center", torch.zeros(1))
        self.register_buffer("temp_mean", torch.zeros(4))
        self.register_buffer("temp_std", torch.ones(4))

    def _standardise(self, track_temp, air_temp, wind_speed, humidity):
        x = torch.stack([track_temp, air_temp, wind_speed, humidity], dim=-1)
        return (x - self.temp_mean) / self.temp_std

    def forward(self, track_temp: torch.Tensor, air_temp: torch.Tensor,
                wind_speed: torch.Tensor, humidity: torch.Tensor) -> torch.Tensor:
        x = self._standardise(track_temp, air_temp, wind_speed, humidity)
        return self.mlp(x).squeeze(-1) - self.center

    @torch.no_grad()
    def calibrate(self, track_temp: torch.Tensor, air_temp: torch.Tensor,
                  wind_speed: torch.Tensor, humidity: torch.Tensor) -> None:
        """Set the standardisation stats and re-center on a (train-only) batch.

        Runs in eval mode internally regardless of the model's current mode, so
        the center is a stable statistic rather than one draw of dropout noise.
        """
        was_training = self.training
        self.eval()
        x = torch.stack([track_temp, air_temp, wind_speed, humidity], dim=-1)
        self.temp_mean = x.mean(dim=0)
        self.temp_std = x.std(dim=0).clamp_min(1e-6)
        self.center.zero_()
        raw = self.forward(track_temp, air_temp, wind_speed, humidity)
        self.center += raw.mean()
        self.train(was_training)


class TrafficHead(nn.Module):
    """`g_traffic(gap_ahead_s) >= 0`, pulled to 0 for `gap > zero_penalty_gap_s`
    via a penalty term added to the loss (see `train.py`), not enforced in the
    forward pass itself (only non-negativity is architectural, via softplus).
    """

    ALLOWED_COLUMNS = frozenset({"gap_ahead_s"})

    def __init__(self, *, hidden_dim: int = 64, n_layers: int = 2,
                 activation: str = "gelu", dropout: float = 0.1):
        super().__init__()
        self.mlp = MLP(1, 1, hidden_dim=hidden_dim, n_layers=n_layers,
                        activation=activation, dropout=dropout)

    def forward(self, gap_ahead_s: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.mlp(gap_ahead_s.unsqueeze(-1)).squeeze(-1))
