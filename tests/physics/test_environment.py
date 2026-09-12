"""Tests for altitude-corrected air density.

What must hold: the flat 1.225 kg/m^3 constant in configs/physics.yaml is the
sea-level, 15 C special case of this module, not a different model -- so that
number is the anchor test. Everything else checks the *direction* and rough
*size* of each correction (altitude down, humidity slightly down, temperature
down as it rises), because the coefficients carry no claim beyond that.
"""

from __future__ import annotations

import pytest

from tyremind.physics.environment import (
    air_density,
    barometric_pressure_hpa,
    circuit_elevation_m,
    session_air_density,
)


class TestBarometricPressure:
    def test_sea_level_returns_reference_pressure(self) -> None:
        assert barometric_pressure_hpa(0.0) == pytest.approx(1013.25)

    def test_pressure_falls_with_altitude(self) -> None:
        assert barometric_pressure_hpa(2285.0) < barometric_pressure_hpa(800.0)
        assert barometric_pressure_hpa(800.0) < barometric_pressure_hpa(0.0)

    def test_mexico_city_pressure_matches_known_range(self) -> None:
        # Mexico City's real station pressure runs ~770-780 hPa; the ISA formula
        # at 2285 m should land in that neighbourhood without needing a lookup.
        assert 760.0 < barometric_pressure_hpa(2285.0) < 790.0


class TestAirDensity:
    def test_sea_level_15c_matches_isa_constant(self) -> None:
        # This is the number configs/physics.yaml hardcodes as a flat default.
        # If this drifts from ~1.225, the module has diverged from the constant
        # it is supposed to be replacing, not just refining.
        assert air_density(0.0, 15.0) == pytest.approx(1.225, abs=0.001)

    def test_density_falls_with_altitude(self) -> None:
        sea_level = air_density(0.0, 20.0)
        mexico_city = air_density(2285.0, 20.0)
        # Real-world drop at Mexico City is commonly cited as ~20-23%.
        ratio = mexico_city / sea_level
        assert 0.72 < ratio < 0.85

    def test_density_falls_with_temperature(self) -> None:
        cold = air_density(0.0, 5.0)
        hot = air_density(0.0, 35.0)
        assert hot < cold

    def test_humidity_reduces_density_slightly(self) -> None:
        dry = air_density(0.0, 30.0, relative_humidity_pct=0.0)
        humid = air_density(0.0, 30.0, relative_humidity_pct=90.0)
        assert humid < dry
        # Humidity is a small correction, not a dominant one.
        assert (dry - humid) / dry < 0.02

    def test_measured_pressure_overrides_barometric_estimate(self) -> None:
        via_altitude = air_density(0.0, 20.0)
        via_measurement = air_density(0.0, 20.0, measured_pressure_hpa=990.0)
        assert via_measurement != via_altitude
        assert via_measurement == pytest.approx(
            air_density(0.0, 20.0, measured_pressure_hpa=990.0)
        )


class TestCircuitElevation:
    def test_known_circuit_resolves(self) -> None:
        assert circuit_elevation_m("mexico city") == pytest.approx(2285.0)

    def test_case_insensitive(self) -> None:
        assert circuit_elevation_m("Monza") == circuit_elevation_m("monza")

    def test_unknown_circuit_raises_rather_than_defaulting(self) -> None:
        # Silently assuming sea level would hide exactly the circuits where
        # getting this wrong matters most -- it must fail loudly instead.
        with pytest.raises(KeyError):
            circuit_elevation_m("atlantis")

    def test_high_and_low_circuits_bracket_the_calendar(self) -> None:
        assert circuit_elevation_m("mexico city") > circuit_elevation_m("sao paulo")
        assert circuit_elevation_m("sao paulo") > circuit_elevation_m("monza")


class TestSessionAirDensity:
    def test_matches_manual_lookup_plus_air_density(self) -> None:
        expected = air_density(circuit_elevation_m("silverstone"), 18.0)
        assert session_air_density("silverstone", 18.0) == pytest.approx(expected)

    def test_thin_air_circuit_gives_lower_density_than_sea_level_circuit(self) -> None:
        assert session_air_density("mexico city", 20.0) < session_air_density(
            "zandvoort", 20.0
        )
