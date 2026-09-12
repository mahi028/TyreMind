"""A lap table with genuine fuel ground truth, from F1 game UDP telemetry.

`tyremind.data.synthetic` is the platform's only source of "known truth" for
separating tyre degradation from fuel burn-off: it generates sessions under the
project's own assumptions and then checks whether the estimator recovers them.
That is a real test, but it has one honest limitation -- the generator and the
estimator were written by the same people against the same physical model, so
a shared blind spot in both would not show up as a failure.

The EA/Codemasters F1 game series (F1 22 onward) broadcasts UDP telemetry
during a session, including `CarStatusData.m_fuelInTank` -- the car's exact
fuel mass, every frame, as tracked by a physics engine nobody on this project
wrote. That makes a recorded game session a second, independently-generated
source of ground truth: not real F1 (the aerodynamics and tyre model are the
game's, not reality's), but a genuinely different generating process for
testing whether the fuel/degradation split holds up outside this project's own
synthetic-data assumptions.

Byte layout verified against the current specification and against
`pahansen/f1-telemetry` (github.com/pahansen/f1-telemetry), an actively
maintained independent implementation -- not derived from memory of the spec.
The length-prefixed capture framing in `iter_length_prefixed_packets` matches
that same tool's recorder, which is the most likely source of an actual
capture file for this project to receive.

What this module cannot do, and does not pretend to: nobody on this project
has run the game to produce a real capture, so nothing here has been validated
against genuine session data. `tests/unit/test_f1_game_telemetry.py` checks
this parser against hand-built packets that follow the documented byte layout
exactly -- it proves the parser implements the spec correctly, not that the
spec assumption is correct. Turning a real `.bin` capture (recorded with
`pahansen/f1-telemetry` or equivalent) into a second validation corpus is the
next step this unlocks, not something already done.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

#: PacketHeader, all supported game format versions (2023-2025 share this
#: layout). <HBBBBBQfIIBB: packet format, game year, major/minor version,
#: packet version, packet id, session UID, session time, frame id, overall
#: frame id, player car index, secondary player car index.
_HEADER_STRUCT = struct.Struct("<HBBBBBQfIIBB")

#: CarStatusData, one entry per car (up to 22), immediately after the header
#: in a Car Status packet (m_packetId == 7). Field order and types verified
#: against pahansen/f1-telemetry's car_status_data.py.
_CAR_STATUS_STRUCT = struct.Struct("<BBBBBfffHHBBHBBBbfffBfffB")

#: LapData, one entry per car, immediately after the header in a Lap Data
#: packet (m_packetId == 2). Verified against pahansen/f1-telemetry's
#: lap_data.py.
_LAP_DATA_STRUCT = struct.Struct("<IIHBHBHBHBfffBBBBBBBBBBBBBBBHHBfB")

MAX_CARS = 22

PACKET_ID_LAP_DATA = 2
PACKET_ID_CAR_STATUS = 7

#: Actual tyre compound codes CarStatusData reports for the current
#: generation of slicks. Wet-weather compounds (7, 8) are excluded from the
#: mapping deliberately -- same reasoning as `f1_loader._WET_COMPOUNDS`.
COMPOUND_CODE = {21: "C0", 20: "C1", 19: "C2", 18: "C3", 17: "C4", 16: "C5"}


@dataclass(frozen=True)
class PacketHeader:
    """Common header at the start of every UDP packet."""

    packet_format: int
    game_year: int
    game_major_version: int
    game_minor_version: int
    packet_version: int
    packet_id: int
    session_uid: int
    session_time: float
    frame_identifier: int
    overall_frame_identifier: int
    player_car_index: int
    secondary_player_car_index: int

    @classmethod
    def from_buffer(cls, buffer: bytes, offset: int = 0) -> PacketHeader:
        return cls(*_HEADER_STRUCT.unpack_from(buffer, offset))


@dataclass(frozen=True)
class CarStatusSample:
    """One car's status at one frame. Only the fields this module uses."""

    frame_identifier: int
    session_time: float
    car_index: int
    fuel_in_tank_kg: float
    fuel_capacity_kg: float
    fuel_remaining_laps: float
    actual_tyre_compound: int
    tyres_age_laps: int


@dataclass(frozen=True)
class LapSample:
    """One car's lap state at one frame. Only the fields this module uses."""

    frame_identifier: int
    car_index: int
    last_lap_time_ms: int
    current_lap_num: int
    pit_status: int
    num_pit_stops: int


def _parse_car_status_packet(buffer: bytes, header: PacketHeader) -> list[CarStatusSample]:
    samples = []
    offset = _HEADER_STRUCT.size
    for car_index in range(MAX_CARS):
        fields = _CAR_STATUS_STRUCT.unpack_from(buffer, offset)
        samples.append(
            CarStatusSample(
                frame_identifier=header.frame_identifier,
                session_time=header.session_time,
                car_index=car_index,
                fuel_in_tank_kg=fields[5],
                fuel_capacity_kg=fields[6],
                fuel_remaining_laps=fields[7],
                actual_tyre_compound=fields[13],
                tyres_age_laps=fields[15],
            )
        )
        offset += _CAR_STATUS_STRUCT.size
    return samples


def _parse_lap_data_packet(buffer: bytes, header: PacketHeader) -> list[LapSample]:
    samples = []
    offset = _HEADER_STRUCT.size
    for car_index in range(MAX_CARS):
        fields = _LAP_DATA_STRUCT.unpack_from(buffer, offset)
        samples.append(
            LapSample(
                frame_identifier=header.frame_identifier,
                car_index=car_index,
                last_lap_time_ms=fields[0],
                current_lap_num=fields[14],
                pit_status=fields[15],
                num_pit_stops=fields[16],
            )
        )
        offset += _LAP_DATA_STRUCT.size
    return samples


def parse_packet(buffer: bytes) -> tuple[PacketHeader, list[CarStatusSample] | list[LapSample]] | None:
    """Parse one raw UDP payload if it is a packet type this module reads.

    Args:
        buffer: Exactly one UDP datagram's payload (header + car array).

    Returns:
        `(header, samples)` for a Lap Data or Car Status packet, `None` for
        every other packet id -- motion, session, telemetry, etc. are not
        needed to build the fuel/lap-time table and are skipped rather than
        parsed and discarded, since their layouts are not implemented here.
    """
    header = PacketHeader.from_buffer(buffer)
    if header.packet_id == PACKET_ID_CAR_STATUS:
        return header, _parse_car_status_packet(buffer, header)
    if header.packet_id == PACKET_ID_LAP_DATA:
        return header, _parse_lap_data_packet(buffer, header)
    return None


def iter_length_prefixed_packets(data: bytes):
    """Yield raw UDP payloads from a `pahansen/f1-telemetry`-style capture file.

    That recorder writes each received datagram as a 2-byte big-endian length
    followed by that many bytes -- not a pcap, not raw concatenated UDP frames.
    A capture made a different way needs a different reader; this one matches
    the specific, actively maintained tool most likely to produce a file this
    project would actually receive.

    Args:
        data: The full contents of a `.bin` capture file.

    Yields:
        One UDP payload (bytes) per recorded packet, in capture order.

    Raises:
        ValueError: If a length prefix claims more bytes than remain -- a
            truncated or corrupted capture, which should stop the read rather
            than silently yield a short, misparsed payload.
    """
    pos = 0
    while pos + 2 <= len(data):
        (length,) = struct.unpack_from(">H", data, pos)
        pos += 2
        if pos + length > len(data):
            raise ValueError(
                f"capture truncated: length prefix {length} at byte {pos - 2} "
                f"exceeds the {len(data) - pos} bytes remaining"
            )
        yield data[pos : pos + length]
        pos += length


def read_capture(path: str | Path) -> tuple[list[CarStatusSample], list[LapSample]]:
    """Read a length-prefixed capture file into car-status and lap samples.

    Args:
        path: Path to a `.bin` file recorded by `pahansen/f1-telemetry` (or a
            byte-compatible tool).

    Returns:
        `(car_status_samples, lap_samples)`, each in capture order, from every
        recognised packet in the file. Every other packet type is skipped.
    """
    data = Path(path).read_bytes()
    status_samples: list[CarStatusSample] = []
    lap_samples: list[LapSample] = []

    for payload in iter_length_prefixed_packets(data):
        if len(payload) < _HEADER_STRUCT.size:
            continue
        parsed = parse_packet(payload)
        if parsed is None:
            continue
        header, samples = parsed
        if header.packet_id == PACKET_ID_CAR_STATUS:
            status_samples.extend(samples)  # type: ignore[arg-type]
        else:
            lap_samples.extend(samples)  # type: ignore[arg-type]

    return status_samples, lap_samples


def build_fuel_ground_truth_table(
    status_samples: list[CarStatusSample],
    lap_samples: list[LapSample],
    car_index: int,
) -> list[dict]:
    """Reduce per-frame samples for one car into one row per completed lap.

    This is the bridge to the project's own schema: the output carries the
    same `lap_time`/`compound`/`tyre_age` columns
    `tyremind.data.synthetic.generate_session` produces, plus `true_fuel_kg` --
    the fuel mass the game reports at the moment each lap began, which no
    synthetic generator can claim because nobody has to invent it here.

    Car Status and Lap Data are separate UDP packets sent independently, so
    "the fuel level at the start of lap N" is not one field -- it is the
    earliest Car Status sample recorded while Lap Data still reported the car
    on lap N. The two sample streams are walked together, in frame order, to
    build that association (a single pass, since both are already sorted).

    Args:
        status_samples: From `read_capture`, for the whole session.
        lap_samples: From `read_capture`, for the whole session.
        car_index: Which of the up-to-22 cars to build the table for.

    Returns:
        One dict per completed lap, ordered by lap number, with keys
        `session_lap`, `lap_time`, `compound`, `tyre_age`, `true_fuel_kg`,
        `true_fuel_capacity_kg`. Empty if the car index never appears or
        completed no laps -- callers should treat that as "no data", not
        raise, since a capture legitimately might not include every grid slot.
    """
    car_status = sorted(
        (s for s in status_samples if s.car_index == car_index), key=lambda s: s.frame_identifier
    )
    car_laps = sorted(
        (s for s in lap_samples if s.car_index == car_index), key=lambda s: s.frame_identifier
    )
    if not car_status or not car_laps:
        return []

    # The state (fuel, compound, tyre age) as of the first Car Status sample
    # seen while Lap Data still reported each lap number -- a two-pointer walk
    # over both frame-ordered streams, so each status sample is visited once.
    first_status_for_lap: dict[int, CarStatusSample] = {}
    lap_pointer = 0
    current_lap_num = car_laps[0].current_lap_num
    for status in car_status:
        while lap_pointer < len(car_laps) and car_laps[lap_pointer].frame_identifier <= status.frame_identifier:
            current_lap_num = car_laps[lap_pointer].current_lap_num
            lap_pointer += 1
        first_status_for_lap.setdefault(current_lap_num, status)

    # A lap is "completed" the first frame current_lap_num advances past it,
    # at which point m_lastLapTimeInMS holds that lap's time. Guard against
    # the same completed time being reported on several consecutive frames.
    lap_time_s: dict[int, float] = {}
    previous_lap_num = car_laps[0].current_lap_num
    for sample in car_laps:
        if sample.current_lap_num > previous_lap_num and sample.last_lap_time_ms > 0:
            completed_lap = sample.current_lap_num - 1
            lap_time_s.setdefault(completed_lap, sample.last_lap_time_ms / 1000.0)
        previous_lap_num = sample.current_lap_num

    rows: list[dict] = []
    for completed_lap, lap_time in sorted(lap_time_s.items()):
        state = first_status_for_lap.get(completed_lap)
        rows.append(
            {
                "session_lap": completed_lap,
                "lap_time": lap_time,
                "compound": COMPOUND_CODE.get(state.actual_tyre_compound, "UNKNOWN")
                if state
                else "UNKNOWN",
                "tyre_age": float(state.tyres_age_laps) if state else float("nan"),
                "true_fuel_kg": state.fuel_in_tank_kg if state else float("nan"),
                "true_fuel_capacity_kg": state.fuel_capacity_kg if state else float("nan"),
            }
        )
    return rows
