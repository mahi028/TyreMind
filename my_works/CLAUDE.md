# Project: F1 Tyre Degradation Isolation

The full build plan is in `PLAN.md`. Read the section for the current phase before starting it.
Track progress in `PROGRESS.md` — update it after every phase.

## What this project is

Tyre degradation is **never observed**. There is no label for it in any dataset. This is an
effect-isolation problem, not a supervised prediction problem. The model is a Neural Additive
Model whose tyre head, read out in seconds, *is* the deliverable.

## Non-negotiable rules

1. **No shared trunk.** Each head is a separate MLP with a restricted input allowlist. A single
   network seeing all features and emitting a per-feature vector is unidentifiable — output
   dimension *i* would have no reason to correspond to feature *i*. This is the specific failure
   the architecture exists to prevent.

2. **Only `g_tyre` may receive `tyre_age` / `lap_in_stint`.** Enforce with a per-head column
   allowlist plus a test that fails on violation.

3. **`g_evo` never receives driver, team or car inputs.** Track evolution is shared across the
   whole field; that asymmetry is what separates it from tyre age. Removing this restriction
   destroys the identification.

4. **Never learn the fuel coefficient.** It is a fixed config constant
   (`FUEL_TIME_PER_KG = 0.030`). Fuel and tyre age are collinear within a stint; a free fuel
   coefficient lets the optimizer split the effect arbitrarily.

5. **Tyre head uses cumulative softplus increments + cumsum.** Guarantees monotonicity and
   `g_tyre(0) == 0`. Never `MLP(tyre_age)` directly.

6. **Split by EVENT, never by row.** Laps from one stint in both train and val is leakage.
   Fit scalers on train only.

7. **`session.load(laps=True, telemetry=False, weather=True, messages=True)`.** Default args pull
   tens of GB of telemetry this project does not use.

8. **Do not pool 2026 with 2022–2025.** New regulations, new Pirelli tyres. Use an era embedding.

9. **Never relax the Phase 3 synthetic recovery thresholds to make a test pass.** If recovery
   fails, an identifiability constraint is wrong. Fix the model, not the threshold.

10. **Units in every docstring** (seconds, kg, °C). Unit confusion is the most likely silent bug.

## Workflow

- One phase at a time. Stop at each phase's acceptance criteria and report before continuing.
- Use plan mode before any phase that writes more than one file.
- `pytest` must pass before a phase counts as done.
- All constants live in `config/default.yaml`. No magic numbers in functions.
- Phase 3 (synthetic recovery) must pass before the model touches real data.
