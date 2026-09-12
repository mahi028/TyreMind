"""Tests for the F1-game UDP telemetry parser.

Two different things are being checked here, and it matters not to confuse
them. `TestByteLevelParsing` builds packets by hand with `struct.pack`, using
the exact field layout documented in the module and verified against
`pahansen/f1-telemetry`'s independent implementation -- it proves this parser
correctly implements that layout. It does NOT prove the layout itself is
right for a real game session, because nobody on this project has produced a
real capture to check against; that remains open, and the module docstring
says so.

`TestFuelGroundTruthReduction` works one level up, from already-parsed sample
dataclasses, and checks the lap-boundary/fuel-association logic on its own
merits -- this part has no dependency on the byte format being right.
"""

from __future__ import annotations

import struct

import pytest

from tyremind.data.f1_game_telemetry import (
    PACKET_ID_CAR_STATUS,
    PACKET_ID_LAP_DATA,
    CarStatusSample,
    LapSample,
    PacketHeader,
    build_fuel_ground_truth_table,
    iter_length_prefixed_packets,
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


def _car_status_packet(entries: dict[int, bytes], frame_identifier: int = 1000) -> bytes:
    header = _pack_header(PACKET_ID_CAR_STATUS, frame_identifier=frame_identifier)
    default_entry = _pack_car_status_entry()
    cars = b"".join(entries.get(i, default_entry) for i in range(MAX_CARS))
    return header + cars


def _lap_data_packet(entries: dict[int, bytes], frame_identifier: int = 1000) -> bytes:
    header = _pack_header(PACKET_ID_LAP_DATA, frame_identifier=frame_identifier)
    default_entry = _pack_lap_data_entry()
    cars = b"".join(entries.get(i, default_entry) for i in range(MAX_CARS))
    return header + cars


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

    def test_unrecognised_packet_id_returns_none(self) -> None:
        # Packet id 0 is Motion Data, not implemented by this module.
        packet = _pack_header(0)
        assert parse_packet(packet) is None


class TestLengthPrefixedFraming:
    def test_round_trips_multiple_packets(self) -> None:
        packets = [_car_status_packet({}), _lap_data_packet({})]
        capture = b"".join(struct.pack(">H", len(p)) + p for p in packets)
        recovered = list(iter_length_prefixed_packets(capture))
        assert recovered == packets

    def test_truncated_capture_raises(self) -> None:
        packet = _car_status_packet({})
        capture = struct.pack(">H", len(packet) + 100) + packet  # length lies
        with pytest.raises(ValueError):
            list(iter_length_prefixed_packets(capture))


class TestFuelGroundTruthReduction:
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
        rows = build_fuel_ground_truth_table(status, laps, car_index=0)
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
        rows = build_fuel_ground_truth_table(status, laps, car_index=0)
        assert len(rows) == 1
        assert rows[0]["tyre_age"] == pytest.approx(3.0)
        assert rows[0]["compound"] == "C3"

    def test_no_samples_for_car_returns_empty(self) -> None:
        assert build_fuel_ground_truth_table([], [], car_index=0) == []

    def test_unrelated_car_indices_are_ignored(self) -> None:
        status = [
            CarStatusSample(0, 0.0, car_index=5, fuel_in_tank_kg=99.0, fuel_capacity_kg=110.0,
                             fuel_remaining_laps=9.0, actual_tyre_compound=18, tyres_age_laps=0),
        ]
        laps = [
            LapSample(0, car_index=5, last_lap_time_ms=0, current_lap_num=1, pit_status=0, num_pit_stops=0),
        ]
        assert build_fuel_ground_truth_table(status, laps, car_index=0) == []
