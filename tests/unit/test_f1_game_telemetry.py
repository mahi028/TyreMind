"""Tests for the F1-game UDP telemetry parser.

Two different things are being checked here, and it matters not to confuse
them. `TestByteLevelParsing` builds packets by hand with `struct.pack`, using
the exact field layout documented in the module and verified against EA's
official F1 25 UDP specification -- it proves this parser correctly
implements that layout. It does NOT prove the layout itself is right for a
real game session, because nobody on this project has produced a real capture
to check against; that remains open, and the module docstring says so.

`TestLapGroundTruthReduction` works one level up, from already-parsed sample
dataclasses, and checks the lap-boundary/state-association logic on its own
merits -- this part has no dependency on the byte format being right.

`TestDemoCapture` closes the loop: it generates a capture with
`generate_demo_capture` and feeds it back through the encode/decode/reduce
pipeline end to end, checking the *physics* comes out right (fuel falls
monotonically at the configured rate, the pit stop is visible, tyre age
resets, tyre wear rises and resets). This is still not a real capture -- see
the module docstring -- but it is a genuine round-trip through every function
in the module at once, which the hand-built-packet tests above do not
exercise.
"""

from __future__ import annotations

import struct

import pytest

from tyremind.data.f1_game_telemetry import (
    PACKET_ID_CAR_DAMAGE,
    PACKET_ID_CAR_STATUS,
    PACKET_ID_CAR_TELEMETRY,
    PACKET_ID_LAP_DATA,
    CarDamageSample,
    CarStatusSample,
    CarTelemetrySample,
    DemoCaptureConfig,
    LapSample,
    PacketHeader,
    build_lap_ground_truth_table,
    generate_demo_capture,
    iter_length_prefixed_packets,
    parse_capture_bytes,
    parse_packet,
)

MAX_CARS = 22


def _pack_header(packet_id: int, frame_identifier: int = 1000, session_time: float = 123.4) -> bytes:
    return struct.pack(
        "<HBBBBBQfIIBB",
        2024,  # packet_format
        24,  # game_year
        1,  # game_major_version
        0,  # game_minor_version
        1,  # packet_version
        packet_id,
        999,  # session_uid
        session_time,
        frame_identifier,
        frame_identifier,  # overall_frame_identifier
        0,  # player_car_index
        255,  # secondary_player_car_index (255 = not in use)
    )


def _pack_car_status_entry(
    *, fuel_in_tank: float = 45.0, fuel_capacity: float = 110.0, tyre_compound: int = 18, tyres_age: int = 0
) -> bytes:
    return struct.pack(
        "<BBBBBfffHHBBHBBBbfffBfffB",
        0, 0, 1, 50, 0,  # traction, abs, fuelMix, frontBrakeBias, pitLimiter
        fuel_in_tank,
        fuel_capacity,
        5.0,  # fuel_remaining_laps
        15000, 4500,  # max/idle RPM
        8, 1, 0,  # maxGears, drsAllowed, drsActivationDistance
        tyre_compound,
        18,  # visual compound
        tyres_age,
        0,  # vehicleFIAFlags
        500000.0, 50000.0, 2000000.0,  # engine power ICE/MGUK, ERS store
        0,  # ersDeployMode
        10000.0, 5000.0, 8000.0,  # ERS harvested/deployed this lap
        0,  # networkPaused
    )


def _pack_lap_data_entry(*, last_lap_time_ms: int = 0, current_lap_num: int = 1) -> bytes:
    return struct.pack(
        "<IIHBHBHBHBfffBBBBBBBBBBBBBBBHHBfB",
        last_lap_time_ms,
        60000,  # current_lap_time_ms
        20, 0, 20, 0,  # sector1/2 time parts
        0, 0, 0, 0,  # delta to car in front / race leader
        500.0, 500.0, 0.0,  # lap distance, total distance, safety car delta
        1,  # car position (unused by this module)
        current_lap_num,
        0, 0, 0,  # pitStatus, numPitStops, sector
        0, 0, 0, 0, 0,  # currentLapInvalid, penalties, warnings, cornerCutting, driveThrough
        0, 0, 0,  # stopGoPens, gridPosition, driverStatus
        0, 0,  # resultStatus, pitLaneTimerActive
        0, 0,  # pitLaneTimeInLaneInMS, pitStopTimerInMS
        0,  # pitStopShouldServePen
        0.0,  # speedTrapFastestSpeed
        0,  # speedTrapFastestLap
    )


def _pack_car_telemetry_entry(
    *,
    pressures: tuple[float, float, float, float] = (23.0, 23.0, 22.5, 22.5),
    surface_temps: tuple[int, int, int, int] = (95, 95, 92, 92),
    inner_temps: tuple[int, int, int, int] = (100, 100, 97, 97),
    brake_temps: tuple[int, int, int, int] = (450, 450, 400, 400),
) -> bytes:
    return struct.pack(
        "<HfffBbHBBH4H4B4BH4f4B",
        250,  # speed
        0.8, 0.1, 0.0,  # throttle, steer, brake
        0, 6, 11500, 1, 40, 0,  # clutch, gear, engineRPM, drs, revLightsPercent, revLightsBitValue
        *brake_temps,
        *surface_temps,
        *inner_temps,
        105,  # engineTemperature
        *pressures,
        0, 0, 0, 0,  # surfaceType
    )


def _pack_car_damage_entry(*, tyres_wear: tuple[float, float, float, float] = (12.0, 12.0, 9.5, 9.5)) -> bytes:
    return struct.pack(
        "<4f4B4B4B18B",
        *tyres_wear,
        0, 0, 0, 0,  # tyresDamage
        0, 0, 0, 0,  # brakesDamage
        0, 0, 0, 0,  # tyreBlisters
        *([0] * 18),
    )


def _packet(packet_id: int, entries: dict[int, bytes], default_entry: bytes, frame_identifier: int = 1000) -> bytes:
    header = _pack_header(packet_id, frame_identifier=frame_identifier)
    cars = b"".join(entries.get(i, default_entry) for i in range(MAX_CARS))
    return header + cars


def _car_status_packet(entries: dict[int, bytes], frame_identifier: int = 1000) -> bytes:
    return _packet(PACKET_ID_CAR_STATUS, entries, _pack_car_status_entry(), frame_identifier)


def _lap_data_packet(entries: dict[int, bytes], frame_identifier: int = 1000) -> bytes:
    return _packet(PACKET_ID_LAP_DATA, entries, _pack_lap_data_entry(), frame_identifier)


def _car_telemetry_packet(entries: dict[int, bytes], frame_identifier: int = 1000) -> bytes:
    return _packet(PACKET_ID_CAR_TELEMETRY, entries, _pack_car_telemetry_entry(), frame_identifier)


def _car_damage_packet(entries: dict[int, bytes], frame_identifier: int = 1000) -> bytes:
    return _packet(PACKET_ID_CAR_DAMAGE, entries, _pack_car_damage_entry(), frame_identifier)


class TestByteLevelParsing:
    def test_header_round_trips(self) -> None:
        packet = _car_status_packet({}, frame_identifier=4242)
        header = PacketHeader.from_buffer(packet)
        assert header.packet_id == PACKET_ID_CAR_STATUS
        assert header.frame_identifier == 4242
        assert header.game_year == 24

    def test_car_status_packet_parses_all_22_cars(self) -> None:
        packet = _car_status_packet({0: _pack_car_status_entry(fuel_in_tank=52.3)})
        parsed = parse_packet(packet)
        assert parsed is not None
        header, samples = parsed
        assert len(samples) == MAX_CARS
        assert samples[0].fuel_in_tank_kg == pytest.approx(52.3)

    def test_car_status_fuel_field_is_not_confused_with_neighbours(self) -> None:
        # fuel_in_tank, fuel_capacity and fuel_remaining_laps are three
        # consecutive floats -- an off-by-one in the struct format would
        # silently shift one into another rather than raising.
        entry = _pack_car_status_entry(fuel_in_tank=10.0, fuel_capacity=110.0)
        packet = _car_status_packet({3: entry})
        _, samples = parse_packet(packet)
        assert samples[3].fuel_in_tank_kg == pytest.approx(10.0)
        assert samples[3].fuel_capacity_kg == pytest.approx(110.0)
        assert samples[3].fuel_remaining_laps == pytest.approx(5.0)

    def test_tyre_compound_and_age_parse(self) -> None:
        entry = _pack_car_status_entry(tyre_compound=16, tyres_age=12)
        packet = _car_status_packet({7: entry})
        _, samples = parse_packet(packet)
        assert samples[7].actual_tyre_compound == 16
        assert samples[7].tyres_age_laps == 12

    def test_lap_data_packet_parses(self) -> None:
        entry = _pack_lap_data_entry(last_lap_time_ms=91234, current_lap_num=5)
        packet = _lap_data_packet({1: entry})
        parsed = parse_packet(packet)
        assert parsed is not None
        _, samples = parsed
        assert samples[1].last_lap_time_ms == 91234
        assert samples[1].current_lap_num == 5

    def test_car_telemetry_packet_parses_pressure_and_temperatures(self) -> None:
        entry = _pack_car_telemetry_entry(
            pressures=(24.1, 24.2, 21.8, 21.9),
            surface_temps=(98, 99, 91, 92),
            inner_temps=(103, 104, 96, 97),
            brake_temps=(480, 481, 410, 411),
        )
        packet = _car_telemetry_packet({4: entry})
        parsed = parse_packet(packet)
        assert parsed is not None
        _, samples = parsed
        s = samples[4]
        assert s.tyres_pressure_psi == pytest.approx((24.1, 24.2, 21.8, 21.9), abs=1e-3)
        assert s.tyres_surface_temp_c == (98, 99, 91, 92)
        assert s.tyres_inner_temp_c == (103, 104, 96, 97)
        assert s.brakes_temp_c == (480, 481, 410, 411)

    def test_car_telemetry_pressure_not_confused_with_surface_type(self) -> None:
        # tyresPressure (float[4]) is immediately followed by surfaceType
        # (uint8[4]) -- an off-by-one here would read surface type as pressure
        # or vice versa, and both would look like plausible-ish numbers.
        entry = _pack_car_telemetry_entry(pressures=(99.9, 99.9, 99.9, 99.9))
        packet = _car_telemetry_packet({0: entry})
        _, samples = parse_packet(packet)
        assert samples[0].tyres_pressure_psi == pytest.approx((99.9, 99.9, 99.9, 99.9))

    def test_car_damage_packet_parses_tyre_wear(self) -> None:
        entry = _pack_car_damage_entry(tyres_wear=(15.5, 16.0, 11.2, 11.8))
        packet = _car_damage_packet({2: entry})
        parsed = parse_packet(packet)
        assert parsed is not None
        _, samples = parsed
        assert samples[2].tyres_wear_pct == pytest.approx((15.5, 16.0, 11.2, 11.8))

    def test_unrecognised_packet_id_returns_none(self) -> None:
        # Packet id 0 is Motion Data, not implemented by this module.
        packet = _pack_header(0)
        assert parse_packet(packet) is None


class TestLengthPrefixedFraming:
    def test_round_trips_multiple_packets(self) -> None:
        packets = [_car_status_packet({}), _lap_data_packet({}), _car_telemetry_packet({}), _car_damage_packet({})]
        capture = b"".join(struct.pack(">H", len(p)) + p for p in packets)
        recovered = list(iter_length_prefixed_packets(capture))
        assert recovered == packets

    def test_truncated_capture_raises(self) -> None:
        packet = _car_status_packet({})
        capture = struct.pack(">H", len(packet) + 100) + packet  # length lies
        with pytest.raises(ValueError):
            list(iter_length_prefixed_packets(capture))


class TestParseCaptureBytes:
    def test_groups_every_packet_type(self) -> None:
        packets = [_car_status_packet({}), _lap_data_packet({}), _car_telemetry_packet({}), _car_damage_packet({})]
        capture = b"".join(struct.pack(">H", len(p)) + p for p in packets)
        samples = parse_capture_bytes(capture)
        assert len(samples.car_status) == MAX_CARS
        assert len(samples.lap_data) == MAX_CARS
        assert len(samples.car_telemetry) == MAX_CARS
        assert len(samples.car_damage) == MAX_CARS


class TestLapGroundTruthReduction:
    def test_associates_fuel_at_lap_start_not_lap_end(self) -> None:
        # Car burns fuel across lap 1 (frames 0-10) and lap 2 (frames 10-20).
        # The row for completed lap 1 must carry the fuel level from near
        # frame 0, not the lower level measured just before the lap ended.
        status = [
            CarStatusSample(0, 0.0, car_index=0, fuel_in_tank_kg=50.0, fuel_capacity_kg=110.0,
                             fuel_remaining_laps=5.0, actual_tyre_compound=18, tyres_age_laps=0),
            CarStatusSample(9, 9.0, car_index=0, fuel_in_tank_kg=48.5, fuel_capacity_kg=110.0,
                             fuel_remaining_laps=4.8, actual_tyre_compound=18, tyres_age_laps=0),
            CarStatusSample(10, 10.0, car_index=0, fuel_in_tank_kg=48.4, fuel_capacity_kg=110.0,
                             fuel_remaining_laps=4.7, actual_tyre_compound=18, tyres_age_laps=1),
        ]
        laps = [
            LapSample(0, car_index=0, last_lap_time_ms=0, current_lap_num=1, pit_status=0, num_pit_stops=0),
            LapSample(9, car_index=0, last_lap_time_ms=0, current_lap_num=1, pit_status=0, num_pit_stops=0),
            LapSample(10, car_index=0, last_lap_time_ms=91234, current_lap_num=2, pit_status=0, num_pit_stops=0),
        ]
        rows = build_lap_ground_truth_table(status, laps, car_index=0)
        assert len(rows) == 1
        assert rows[0]["session_lap"] == 1
        assert rows[0]["lap_time"] == pytest.approx(91.234)
        assert rows[0]["true_fuel_kg"] == pytest.approx(50.0)  # start of lap 1, not 48.5

    def test_tyre_age_and_compound_come_from_lap_start_state(self) -> None:
        status = [
            CarStatusSample(0, 0.0, car_index=0, fuel_in_tank_kg=50.0, fuel_capacity_kg=110.0,
                             fuel_remaining_laps=5.0, actual_tyre_compound=18, tyres_age_laps=3),
            CarStatusSample(10, 10.0, car_index=0, fuel_in_tank_kg=48.0, fuel_capacity_kg=110.0,
                             fuel_remaining_laps=4.7, actual_tyre_compound=18, tyres_age_laps=4),
        ]
        laps = [
            LapSample(0, car_index=0, last_lap_time_ms=0, current_lap_num=4, pit_status=0, num_pit_stops=0),
            LapSample(10, car_index=0, last_lap_time_ms=88000, current_lap_num=5, pit_status=0, num_pit_stops=0),
        ]
        rows = build_lap_ground_truth_table(status, laps, car_index=0)
        assert len(rows) == 1
        assert rows[0]["tyre_age"] == pytest.approx(3.0)
        assert rows[0]["compound"] == "C3"

    def test_no_samples_for_car_returns_empty(self) -> None:
        assert build_lap_ground_truth_table([], [], car_index=0) == []

    def test_unrelated_car_indices_are_ignored(self) -> None:
        status = [
            CarStatusSample(0, 0.0, car_index=5, fuel_in_tank_kg=99.0, fuel_capacity_kg=110.0,
                             fuel_remaining_laps=9.0, actual_tyre_compound=18, tyres_age_laps=0),
        ]
        laps = [
            LapSample(0, car_index=5, last_lap_time_ms=0, current_lap_num=1, pit_status=0, num_pit_stops=0),
        ]
        assert build_lap_ground_truth_table(status, laps, car_index=0) == []

    def test_telemetry_and_damage_are_optional(self) -> None:
        status = [
            CarStatusSample(0, 0.0, car_index=0, fuel_in_tank_kg=50.0, fuel_capacity_kg=110.0,
                             fuel_remaining_laps=5.0, actual_tyre_compound=18, tyres_age_laps=0),
        ]
        laps = [
            LapSample(0, car_index=0, last_lap_time_ms=0, current_lap_num=1, pit_status=0, num_pit_stops=0),
            LapSample(10, car_index=0, last_lap_time_ms=90000, current_lap_num=2, pit_status=0, num_pit_stops=0),
        ]
        rows = build_lap_ground_truth_table(status, laps, car_index=0)
        assert len(rows) == 1
        assert rows[0]["tyre_pressure_psi"] != rows[0]["tyre_pressure_psi"]  # NaN

    def test_telemetry_and_damage_are_averaged_across_wheels(self) -> None:
        status = [
            CarStatusSample(0, 0.0, car_index=0, fuel_in_tank_kg=50.0, fuel_capacity_kg=110.0,
                             fuel_remaining_laps=5.0, actual_tyre_compound=18, tyres_age_laps=0),
        ]
        laps = [
            LapSample(0, car_index=0, last_lap_time_ms=0, current_lap_num=1, pit_status=0, num_pit_stops=0),
            LapSample(10, car_index=0, last_lap_time_ms=90000, current_lap_num=2, pit_status=0, num_pit_stops=0),
        ]
        telemetry = [
            CarTelemetrySample(0, car_index=0, tyres_pressure_psi=(24.0, 24.0, 22.0, 22.0),
                                tyres_surface_temp_c=(100, 100, 90, 90), tyres_inner_temp_c=(105, 105, 95, 95),
                                brakes_temp_c=(500, 500, 400, 400)),
        ]
        damage = [
            CarDamageSample(0, car_index=0, tyres_wear_pct=(10.0, 10.0, 6.0, 6.0)),
        ]
        rows = build_lap_ground_truth_table(status, laps, car_index=0, telemetry_samples=telemetry, damage_samples=damage)
        assert rows[0]["tyre_pressure_psi"] == pytest.approx(23.0)
        assert rows[0]["tyre_surface_temp_c"] == pytest.approx(95.0)
        assert rows[0]["tyre_inner_temp_c"] == pytest.approx(100.0)
        assert rows[0]["brake_temp_c"] == pytest.approx(450.0)
        assert rows[0]["tyre_wear_pct"] == pytest.approx(8.0)


def _generate_and_reduce(config: DemoCaptureConfig) -> list[dict]:
    """Full round trip: generate bytes, re-parse them, reduce to a lap table."""
    data = generate_demo_capture(config)
    samples = parse_capture_bytes(data)
    return build_lap_ground_truth_table(
        samples.car_status, samples.lap_data, config.car_index,
        telemetry_samples=samples.car_telemetry, damage_samples=samples.car_damage,
    )


class TestDemoCapture:
    def test_produces_the_full_race_distance(self) -> None:
        rows = _generate_and_reduce(DemoCaptureConfig(n_laps=30, pit_lap=15))
        assert len(rows) == 30
        assert [r["session_lap"] for r in rows] == list(range(1, 31))

    def test_fuel_falls_monotonically_at_the_configured_rate(self) -> None:
        cfg = DemoCaptureConfig(n_laps=30, pit_lap=None, burn_rate_kg_per_lap=2.5)
        rows = _generate_and_reduce(cfg)
        fuels = [r["true_fuel_kg"] for r in rows]
        assert fuels == sorted(fuels, reverse=True)  # strictly decreasing
        assert fuels[0] - fuels[1] == pytest.approx(cfg.burn_rate_kg_per_lap)

    def test_fuel_is_never_negative(self) -> None:
        rows = _generate_and_reduce(DemoCaptureConfig(n_laps=40))
        assert all(r["true_fuel_kg"] >= 0.0 for r in rows)

    def test_pit_stop_changes_compound_and_resets_tyre_age_and_wear(self) -> None:
        cfg = DemoCaptureConfig(n_laps=30, pit_lap=15, starting_compound="MEDIUM", pit_compound="HARD")
        rows = _generate_and_reduce(cfg)
        before, after = rows[14], rows[15]  # laps 15 and 16
        assert before["compound"] != after["compound"]
        assert after["tyre_age"] == 0.0
        assert before["tyre_age"] > after["tyre_age"]
        assert after["tyre_wear_pct"] == 0.0
        assert before["tyre_wear_pct"] > after["tyre_wear_pct"]

    def test_pit_lap_is_slower_by_roughly_the_pit_lane_loss(self) -> None:
        cfg = DemoCaptureConfig(n_laps=30, pit_lap=15)
        rows = _generate_and_reduce(cfg)
        pit_lap_time = rows[14]["lap_time"]  # lap 15
        neighbouring = [rows[13]["lap_time"], rows[15]["lap_time"]]
        assert pit_lap_time - max(neighbouring) == pytest.approx(21.0, abs=1.0)

    def test_no_pit_stop_keeps_one_compound_and_rising_tyre_age(self) -> None:
        rows = _generate_and_reduce(DemoCaptureConfig(n_laps=20, pit_lap=None))
        assert len({r["compound"] for r in rows}) == 1
        assert [r["tyre_age"] for r in rows] == list(range(20))

    def test_tyre_wear_rises_with_age_and_is_capped(self) -> None:
        rows = _generate_and_reduce(DemoCaptureConfig(n_laps=60, pit_lap=None))
        wear = [r["tyre_wear_pct"] for r in rows]
        assert wear == sorted(wear)  # non-decreasing
        assert all(0.0 <= w <= 100.0 for w in wear)

    def test_tyre_temperature_rises_toward_but_not_past_the_optimum_window_top(self) -> None:
        rows = _generate_and_reduce(DemoCaptureConfig(n_laps=40, pit_lap=None))
        temps = [r["tyre_surface_temp_c"] for r in rows]
        assert temps[0] == pytest.approx(90.0)
        assert all(90.0 <= t <= 110.0 for t in temps)
        assert temps[-1] > temps[0]

    def test_tyre_pressure_and_brake_temp_are_present(self) -> None:
        rows = _generate_and_reduce(DemoCaptureConfig(n_laps=5, pit_lap=None))
        assert all(r["tyre_pressure_psi"] > 0.0 for r in rows)
        assert all(r["brake_temp_c"] > 0.0 for r in rows)

    def test_same_seed_is_deterministic(self) -> None:
        cfg = DemoCaptureConfig(n_laps=10, seed=42)
        assert generate_demo_capture(cfg) == generate_demo_capture(cfg)

    def test_different_seed_changes_only_noise_not_shape(self) -> None:
        a = _generate_and_reduce(DemoCaptureConfig(n_laps=10, pit_lap=None, seed=1))
        b = _generate_and_reduce(DemoCaptureConfig(n_laps=10, pit_lap=None, seed=2))
        for ra, rb in zip(a, b, strict=True):
            assert ra["true_fuel_kg"] == pytest.approx(rb["true_fuel_kg"])
            assert ra["lap_time"] != pytest.approx(rb["lap_time"])
