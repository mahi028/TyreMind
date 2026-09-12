"""Ambient air density, corrected for the one thing the flat ISA constant misses.

`configs/physics.yaml` sets `environment.air_density_kg_m3` to 1.225 -- sea level,
15 C, unconditionally -- and `VehicleParameters.air_density_kg_m3` in
`physics/dynamics.py` defaults to the same number. That default is silently wrong
for a third of the calendar. Mexico City sits at 2,285 m, where the air is thin
enough that teams run visibly more wing than anywhere else on the calendar for
exactly this reason; Sao Paulo (800 m) and the Red Bull Ring (700 m) are smaller
versions of the same effect. Every downforce number `dynamics.downforce_n`
computes at those circuits -- and everything downstream of it, load transfer,
frictional power, energy exposure -- is computed at the wrong air density unless
something corrects for altitude.

Fixing this needs no new telemetry and no network call: air pressure at altitude
follows the standard barometric formula almost exactly, and every session's own
air temperature is already loaded from FastF1's weather channel. Altitude is the
one input that was missing, and it is a fixed property of the circuit, not
something that needs measuring per session -- so `data/reference/
circuit_elevations.json` is a small, static reference table, not a scraper.

Measured surface pressure (e.g. from a weather reanalysis) would still improve on
the barometric estimate by the few percent that comes from weather systems passing
through on the day -- `measured_pressure_hpa` is accepted here for exactly that
reason -- but altitude is the effect worth an order of magnitude more than that,
and the one that was entirely unmodelled.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ELEVATIONS_PATH = Path("data/reference/circuit_elevations.json")

#: ISA standard atmosphere sea-level pressure, hPa. The reference point the
#: barometric formula and every session's relative pressure are measured from.
SEA_LEVEL_PRESSURE_HPA = 1013.25

#: Specific gas constant for dry air, J/(kg*K).
_R_DRY_AIR = 287.05

#: Specific gas constant for water vapour, J/(kg*K). Humid air is less dense than
#: dry air at the same pressure and temperature -- water molecules are lighter
#: than the N2/O2 they displace -- which is why humidity correction pushes
#: density down, not up.
_R_WATER_VAPOUR = 461.5


@lru_cache(maxsize=1)
def _load_elevations(path: Path | None = None) -> dict[str, float]:
    """Circuit key -> elevation in metres, from the committed reference table.

    Raises:
        FileNotFoundError: If the reference table is missing.
    """
    target = path or ELEVATIONS_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"{target} is missing. It is committed reference data, not generated."
        )
    raw = json.loads(target.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in raw["elevation_m"].items()}


def circuit_elevation_m(circuit: str) -> float:
    """Elevation above sea level for a circuit, in metres.

    Args:
        circuit: Circuit key as used in `data/reference/circuit_elevations.json`
            (and `tyremind.data.f1_loader.CIRCUIT_ALIASES`), e.g. "monza",
            "mexico city". Matched case-insensitively.

    Returns:
        Elevation in metres.

    Raises:
        KeyError: If the circuit is not in the reference table. Silently
            defaulting to sea level would hide exactly the circuits -- Mexico
            City, Sao Paulo -- where getting this wrong matters most.
    """
    table = _load_elevations()
    key = circuit.strip().lower()
    if key not in table:
        raise KeyError(
            f"{circuit!r} has no entry in {ELEVATIONS_PATH}. Add one rather than "
            "assuming sea level -- the whole point of this table is that altitude "
            "is not a safe default."
        )
    return table[key]


def barometric_pressure_hpa(
    elevation_m: float, sea_level_hpa: float = SEA_LEVEL_PRESSURE_HPA
) -> float:
    """Standard-atmosphere pressure at altitude, hPa.

    The barometric formula (ISA, troposphere): a fixed function of elevation
    alone. It will not capture the +/- 1-2% that a passing weather system adds
    or removes on a given day -- `measured_pressure_hpa` in `air_density` is the
    hook for that -- but it recovers the dominant effect, which is altitude, from
    a circuit constant rather than a weather lookup.

    Args:
        elevation_m: Height above mean sea level, metres.
        sea_level_hpa: Reference sea-level pressure.

    Returns:
        Pressure at that elevation, hPa.
    """
    return sea_level_hpa * (1.0 - 2.25577e-5 * elevation_m) ** 5.25588


def air_density(
    elevation_m: float,
    air_temp_c: float,
    *,
    relative_humidity_pct: float | None = None,
    measured_pressure_hpa: float | None = None,
) -> float:
    """Ambient air density, kg/m^3, corrected for altitude and (optionally) humidity.

    Replaces the flat `environment.air_density_kg_m3: 1.225` in
    `configs/physics.yaml`, which is only correct at sea level. At 15 C and sea
    level this function returns ~1.225 kg/m^3 -- the same number -- which is the
    check that it has not changed the physics anywhere the flat constant already
    happened to be right.

    Humidity correction uses virtual temperature: moist air is less dense than
    dry air at the same pressure and temperature, so higher humidity pushes
    density down slightly. The effect is an order of magnitude smaller than
    altitude and temperature, which is why it is optional rather than required.

    Args:
        elevation_m: Circuit elevation above sea level, metres. See
            `circuit_elevation_m`.
        air_temp_c: Ambient air temperature, Celsius. Session-specific, from
            FastF1's weather channel -- see `f1_loader.build_lap_table`'s caller
            or `data/reference/session_conditions.json`.
        relative_humidity_pct: Relative humidity, 0-100. Omit to compute dry-air
            density, which is within a fraction of a percent of the true value
            except in very hot, very humid conditions.
        measured_pressure_hpa: A measured or reanalysis surface pressure for the
            session, if available, used in place of the barometric estimate. The
            barometric formula already captures the dominant altitude effect;
            this only refines the remaining weather-driven variation.

    Returns:
        Air density, kg/m^3.
    """
    pressure_hpa = (
        measured_pressure_hpa
        if measured_pressure_hpa is not None
        else barometric_pressure_hpa(elevation_m)
    )
    pressure_pa = pressure_hpa * 100.0
    temp_k = air_temp_c + 273.15

    if not relative_humidity_pct:
        return pressure_pa / (_R_DRY_AIR * temp_k)

    # Saturation vapour pressure (Tetens' approximation, valid over water for
    # the -20 to 50 C range every F1 session falls within), then the actual
    # vapour pressure from relative humidity.
    saturation_vapour_pa = 610.94 * pow(
        2.718281828, (17.625 * air_temp_c) / (air_temp_c + 243.04)
    )
    vapour_pressure_pa = (relative_humidity_pct / 100.0) * saturation_vapour_pa
    dry_pressure_pa = pressure_pa - vapour_pressure_pa

    return (dry_pressure_pa / (_R_DRY_AIR * temp_k)) + (
        vapour_pressure_pa / (_R_WATER_VAPOUR * temp_k)
    )


def session_air_density(
    circuit: str,
    air_temp_c: float,
    *,
    relative_humidity_pct: float | None = None,
    measured_pressure_hpa: float | None = None,
) -> float:
    """`air_density` looked up by circuit name rather than elevation.

    The convenience wrapper most callers want: `VehicleParameters.air_density_kg_m3`
    only needs a circuit and that session's own air temperature, both of which are
    already on hand by the time a lap table is built.

    Args:
        circuit: Circuit key, as in `circuit_elevation_m`.
        air_temp_c: Session air temperature, Celsius.
        relative_humidity_pct: See `air_density`.
        measured_pressure_hpa: See `air_density`.

    Returns:
        Air density, kg/m^3.
    """
    return air_density(
        circuit_elevation_m(circuit),
        air_temp_c,
        relative_humidity_pct=relative_humidity_pct,
        measured_pressure_hpa=measured_pressure_hpa,
    )
