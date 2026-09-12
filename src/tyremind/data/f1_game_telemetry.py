"""A lap table with genuine tyre and fuel ground truth, from F1 game UDP telemetry.

`tyremind.data.synthetic` is the platform's only source of "known truth" for
separating tyre degradation from fuel burn-off: it generates sessions under the
project's own assumptions and then checks whether the estimator recovers them.
That is a real test, but it has one honest limitation -- the generator and the
estimator were written by the same people against the same physical model, so
a shared blind spot in both would not show up as a failure.

The EA/Codemasters F1 game series (F1 22 onward) broadcasts UDP telemetry
during a session, including several fields real F1 categorically does not
publish -- see `docs/research/03_DATA_AVAILABILITY.md`'s table of what public
F1 telemetry contains. Fuel mass, tyre pressure, tyre surface/inner
temperature and tyre wear are all marked "No" there for real F1; the game
reports all four every frame, from a physics engine nobody on this project
wrote:

    CarStatusData.m_fuelInTank            fuel mass, kg
    CarTelemetryData.m_tyresPressure[4]   PSI, per wheel
    CarTelemetryData.m_tyresSurfaceTemperature[4]   C, per wheel
    CarTelemetryData.m_tyresInnerTemperature[4]     C, per wheel
    CarDamageData.m_tyresWear[4]          %, per wheel

None of this is real F1 physics -- the aerodynamics and tyre model are the
game's, not reality's -- but it is a genuinely different generating process,
useful for testing whether this project's own assumptions hold up against
data nobody here invented. It does NOT include a "traction coefficient" or any
other slip/grip number -- no motorsport telemetry, real or game, publishes
that; the closest the game gets is `CarStatusData.m_tractionControl`, a driver
*aid setting* (off/medium/full), not a measured quantity. Tyre load is not a
direct field either, but `CarMotionData` gives lateral/longitudinal/vertical
g-force per frame -- exactly the quantity `tyremind.physics.dynamics` derives
from real telemetry by differentiating position twice. Neither is wired up
here yet; both are natural next steps, not implemented.

Byte layout verified directly against EA's official specification ("Data
Output from F1 25 v3", published at forums.ea.com) -- every struct size here
was cross-checked against that document's stated packet sizes (header 29
bytes; Car Status 1239; Lap Data 1285; Car Telemetry 1352; Car Damage 1041)
and matches exactly, not derived from memory of the spec. Car Status and Lap
Data also match `pahansen/f1-telemetry` (github.com/pahansen/f1-telemetry),
an independent implementation, which is a second confirmation for those two,
not the primary source. The length-prefixed capture framing in
`iter_length_prefixed_packets` matches that same tool's recorder, which is the
most likely source of an actual capture file for this project to receive.

The UDP *protocol* is public and documented with no NDA -- anyone can read the
spec without owning the game. Recorded *telemetry data* is not: no capture
files are published anywhere (checked Kaggle, GitHub, and the sim-racing
dashboard tools that support this format, none of which ship sample data --
their "replay" features replay a session you already recorded, not a public
one). That distinction is why this module is spec-verified but not yet
reality-verified.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

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

#: CarTelemetryData, one entry per car (m_packetId == 6). Field order and
#: types verified against EA's official F1 25 UDP specification: speed,
#: throttle, steer, brake, clutch, gear, engineRPM, drs, revLightsPercent,
#: revLightsBitValue, brakesTemperature[4], tyresSurfaceTemperature[4],
#: tyresInnerTemperature[4], engineTemperature, tyresPressure[4],
#: surfaceType[4]. Total entry size (60 bytes) plus the 29-byte header and
#: 22 cars plus the 3-byte trailer reproduces the spec's documented 1352-byte
#: packet size exactly.
_CAR_TELEMETRY_STRUCT = struct.Struct("<HfffBbHBBH4H4B4BH4f4B")

#: CarDamageData, one entry per car (m_packetId == 10): tyresWear[4],
#: tyresDamage[4], brakesDamage[4], tyreBlisters[4], then 18 individual
#: uint8 damage/wear fields this module does not use. Entry size (46 bytes)
#: plus the header and 22 cars reproduces the spec's documented 1041-byte
#: packet size exactly, with no trailer.
_CAR_DAMAGE_STRUCT = struct.Struct("<4f4B4B4B18B")

MAX_CARS = 22

PACKET_ID_LAP_DATA = 2
PACKET_ID_CAR_TELEMETRY = 6
PACKET_ID_CAR_STATUS = 7
PACKET_ID_CAR_DAMAGE = 10

#: Actual tyre compound codes CarStatusData reports for the modern F1 slick
#: range (verified against EA's official F1 25 UDP specification, "Data
#: Output from F1 25 v3.pdf"). C6 (code 22) existed for the 2025 season only --
#: same footnote as `compound_hardness_index.C6` in configs/physics.yaml.
#: Wet-weather (7, 8), F1 Classic (9, 10) and F2 (11-15) codes are out of
#: scope, same reasoning as `f1_loader._WET_COMPOUNDS`.
COMPOUND_CODE = {21: "C0", 20: "C1", 19: "C2", 18: "C3", 17: "C4", 16: "C5", 22: "C6"}


def _mean4(values: tuple[float, float, float, float]) -> float:
    """Mean across the four wheels (FL, FR, RL, RR).

    The project's existing lap-table schema (see `tyremind.data.synthetic`)
    is one scalar per lap, not per corner, so per-wheel detail is collapsed
    here to match it. A real capture's per-wheel spread -- e.g. higher front
    load through a corner sequence favouring one side -- is real information
    this throws away; it is available in `CarTelemetrySample`/`CarDamageSample`
    directly for anyone who wants the per-corner picture.
    """
    return sum(values) / 4.0


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


@dataclass(frozen=True)
class CarTelemetrySample:
    """One car's telemetry at one frame. Only the fields this module uses.

    Per-wheel tuples are ordered (front-left, front-right, rear-left,
    rear-right), matching the game's own array order.
    """

    frame_identifier: int
    car_index: int
    tyres_pressure_psi: tuple[float, float, float, float]
    tyres_surface_temp_c: tuple[int, int, int, int]
    tyres_inner_temp_c: tuple[int, int, int, int]
    brakes_temp_c: tuple[int, int, int, int]


@dataclass(frozen=True)
class CarDamageSample:
    """One car's damage/wear state at one frame. Only the fields this module uses."""

    frame_identifier: int
    car_index: int
    tyres_wear_pct: tuple[float, float, float, float]


@dataclass(frozen=True)
class CaptureSamples:
    """Every recognised sample from one capture file, grouped by packet type."""

    car_status: list[CarStatusSample]
    lap_data: list[LapSample]
    car_telemetry: list[CarTelemetrySample]
    car_damage: list[CarDamageSample]


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


def _parse_car_telemetry_packet(buffer: bytes, header: PacketHeader) -> list[CarTelemetrySample]:
    samples = []
    offset = _HEADER_STRUCT.size
    for car_index in range(MAX_CARS):
        f = _CAR_TELEMETRY_STRUCT.unpack_from(buffer, offset)
        # Indices: 0 speed, 1 throttle, 2 steer, 3 brake, 4 clutch, 5 gear,
        # 6 engineRPM, 7 drs, 8 revLightsPercent, 9 revLightsBitValue,
        # 10-13 brakesTemperature, 14-17 tyresSurfaceTemperature,
        # 18-21 tyresInnerTemperature, 22 engineTemperature,
        # 23-26 tyresPressure, 27-30 surfaceType.
        samples.append(
            CarTelemetrySample(
                frame_identifier=header.frame_identifier,
                car_index=car_index,
                tyres_pressure_psi=tuple(f[23:27]),
                tyres_surface_temp_c=tuple(f[14:18]),
                tyres_inner_temp_c=tuple(f[18:22]),
                brakes_temp_c=tuple(f[10:14]),
            )
        )
        offset += _CAR_TELEMETRY_STRUCT.size
    return samples


def _parse_car_damage_packet(buffer: bytes, header: PacketHeader) -> list[CarDamageSample]:
    samples = []
    offset = _HEADER_STRUCT.size
    for car_index in range(MAX_CARS):
        f = _CAR_DAMAGE_STRUCT.unpack_from(buffer, offset)
        samples.append(
            CarDamageSample(
                frame_identifier=header.frame_identifier,
                car_index=car_index,
                tyres_wear_pct=tuple(f[0:4]),
            )
        )
        offset += _CAR_DAMAGE_STRUCT.size
    return samples


def parse_packet(
    buffer: bytes,
) -> tuple[PacketHeader, list] | None:
    """Parse one raw UDP payload if it is a packet type this module reads.

    Args:
        buffer: Exactly one UDP datagram's payload (header + car array).

    Returns:
        `(header, samples)` for a Lap Data, Car Status, Car Telemetry or Car
        Damage packet, `None` for every other packet id -- motion, session,
        etc. are not needed for the tables this module builds and are skipped
        rather than parsed and discarded, since their layouts are not
        implemented here.
    """
    header = PacketHeader.from_buffer(buffer)
    if header.packet_id == PACKET_ID_CAR_STATUS:
        return header, _parse_car_status_packet(buffer, header)
    if header.packet_id == PACKET_ID_LAP_DATA:
        return header, _parse_lap_data_packet(buffer, header)
    if header.packet_id == PACKET_ID_CAR_TELEMETRY:
        return header, _parse_car_telemetry_packet(buffer, header)
    if header.packet_id == PACKET_ID_CAR_DAMAGE:
        return header, _parse_car_damage_packet(buffer, header)
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


def _collect_samples(payloads: Iterable[bytes]) -> CaptureSamples:
    car_status: list[CarStatusSample] = []
    lap_data: list[LapSample] = []
    car_telemetry: list[CarTelemetrySample] = []
    car_damage: list[CarDamageSample] = []

    for payload in payloads:
        if len(payload) < _HEADER_STRUCT.size:
            continue
        parsed = parse_packet(payload)
        if parsed is None:
            continue
        header, samples = parsed
        if header.packet_id == PACKET_ID_CAR_STATUS:
            car_status.extend(samples)
        elif header.packet_id == PACKET_ID_LAP_DATA:
            lap_data.extend(samples)
        elif header.packet_id == PACKET_ID_CAR_TELEMETRY:
            car_telemetry.extend(samples)
        elif header.packet_id == PACKET_ID_CAR_DAMAGE:
            car_damage.extend(samples)

    return CaptureSamples(car_status, lap_data, car_telemetry, car_damage)


def parse_capture_bytes(data: bytes) -> CaptureSamples:
    """`read_capture`, for bytes already in memory rather than a file on disk."""
    return _collect_samples(iter_length_prefixed_packets(data))


def read_capture(path: str | Path) -> CaptureSamples:
    """Read a length-prefixed capture file into samples, grouped by packet type.

    Args:
        path: Path to a `.bin` file recorded by `pahansen/f1-telemetry` (or a
            byte-compatible tool).

    Returns:
        A `CaptureSamples` with every recognised packet's samples, each list
        in capture order. Every other packet type is skipped.
    """
    return parse_capture_bytes(Path(path).read_bytes())


def build_lap_ground_truth_table(
    status_samples: list[CarStatusSample],
    lap_samples: list[LapSample],
    car_index: int,
    *,
    telemetry_samples: list[CarTelemetrySample] | None = None,
    damage_samples: list[CarDamageSample] | None = None,
) -> list[dict]:
    """Reduce per-frame samples for one car into one row per completed lap.

    This is the bridge to the project's own schema: the output carries the
    same `lap_time`/`compound`/`tyre_age` columns
    `tyremind.data.synthetic.generate_session` produces, plus fields no
    synthetic generator can claim because nobody has to invent them here:
    `true_fuel_kg`, and -- when `telemetry_samples`/`damage_samples` are
    supplied -- `tyre_pressure_psi`, `tyre_surface_temp_c`,
    `tyre_inner_temp_c`, `brake_temp_c` and `tyre_wear_pct`, each averaged
    across the four wheels (see `_mean4`).

    Car Status, Lap Data, Car Telemetry and Car Damage are four separate UDP
    packets sent independently, so "the state at the start of lap N" is not
    one field for any of them -- it is the earliest sample of each kind
    recorded while Lap Data still reported the car on lap N. `_first_sample_
    per_lap` builds that association once per sample stream via a two-pointer
    walk against the frame-ordered lap stream.

    Args:
        status_samples: From `read_capture`, for the whole session.
        lap_samples: From `read_capture`, for the whole session.
        car_index: Which of the up-to-22 cars to build the table for.
        telemetry_samples: Optional Car Telemetry samples (tyre pressure and
            temperature, brake temperature). Omit to leave those columns NaN.
        damage_samples: Optional Car Damage samples (tyre wear). Omit to leave
            `tyre_wear_pct` NaN.

    Returns:
        One dict per completed lap, ordered by lap number. Empty if the car
        index never appears or completed no laps -- callers should treat that
        as "no data", not raise, since a capture legitimately might not
        include every grid slot.
    """
    car_status = sorted(
        (s for s in status_samples if s.car_index == car_index), key=lambda s: s.frame_identifier
    )
    car_laps = sorted(
        (s for s in lap_samples if s.car_index == car_index), key=lambda s: s.frame_identifier
    )
    if not car_status or not car_laps:
        return []

    car_telemetry = sorted(
        (s for s in (telemetry_samples or []) if s.car_index == car_index),
        key=lambda s: s.frame_identifier,
    )
    car_damage = sorted(
        (s for s in (damage_samples or []) if s.car_index == car_index),
        key=lambda s: s.frame_identifier,
    )

    first_status_for_lap = _first_sample_per_lap(car_status, car_laps)
    first_telemetry_for_lap = _first_sample_per_lap(car_telemetry, car_laps)
    first_damage_for_lap = _first_sample_per_lap(car_damage, car_laps)

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
        telemetry = first_telemetry_for_lap.get(completed_lap)
        damage = first_damage_for_lap.get(completed_lap)
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
                "tyre_pressure_psi": _mean4(telemetry.tyres_pressure_psi) if telemetry else float("nan"),
                "tyre_surface_temp_c": _mean4(telemetry.tyres_surface_temp_c) if telemetry else float("nan"),
                "tyre_inner_temp_c": _mean4(telemetry.tyres_inner_temp_c) if telemetry else float("nan"),
                "brake_temp_c": _mean4(telemetry.brakes_temp_c) if telemetry else float("nan"),
                "tyre_wear_pct": _mean4(damage.tyres_wear_pct) if damage else float("nan"),
            }
        )
    return rows


def _first_sample_per_lap(samples: list, car_laps: list[LapSample]) -> dict:
    """Earliest `samples` entry recorded while Lap Data still reported each lap.

    Generic over the sample type: works identically for Car Status, Car
    Telemetry or Car Damage samples, since all that's needed is a
    `frame_identifier`. `samples` and `car_laps` must already be filtered to
    one car and sorted by frame.
    """
    result: dict[int, object] = {}
    if not car_laps:
        return result
    lap_pointer = 0
    current_lap_num = car_laps[0].current_lap_num
    for sample in samples:
        while lap_pointer < len(car_laps) and car_laps[lap_pointer].frame_identifier <= sample.frame_identifier:
            current_lap_num = car_laps[lap_pointer].current_lap_num
            lap_pointer += 1
        result.setdefault(current_lap_num, sample)
    return result


# =============================================================================
# Demo capture generator
# =============================================================================
#
# Not a substitute for a real recording -- see the module docstring. This
# exists so the byte format, the reduction logic, and the *shape* of what a
# real capture will eventually produce can all be seen and tested today,
# without waiting on one. Every coefficient below is pulled from
# configs/physics.yaml or tyremind.data.synthetic rather than invented, so the
# numbers a demo run prints are the project's own current best estimates, not
# arbitrary ones: 0.030 s/kg is `fuel.seconds_per_kg`, 2.70 kg/lap is
# `fuel.burn_rate_kg_per_lap`, 21.0 s is `strategy.pit_lane_loss_s`, 90-110 C
# is `tyre.thermal.optimum_window_C`, and the per-compound degradation rates
# match `synthetic.DEFAULT_COMPOUND_RATES`.
#
# Tyre pressure, temperature, wear and brake temperature have no equivalent
# prior anywhere in this project yet -- real F1 telemetry has never carried
# them, so there was nothing to calibrate against before now. The values
# below (flat pressure with a small rise as tyres heat up, temperature
# climbing toward the top of the optimum window, wear rising linearly with
# tyre age) are directionally realistic, not sourced the way the fuel and
# degradation numbers are, and every wheel gets an identical value -- a real
# capture would show real front/rear and left/right asymmetry this cannot.
#
# A real capture will not look like this in the details -- no ERS/thermal
# transients, no traffic, no driver error, tidy 1 Hz sampling -- but the
# fields, their units, and the fuel/lap-time relationship are the real ones.

#: From configs/physics.yaml fuel.seconds_per_kg.
_DEMO_SECONDS_PER_KG = 0.030
#: From configs/physics.yaml fuel.burn_rate_kg_per_lap (circuit-average default).
_DEMO_BURN_RATE_KG_PER_LAP = 2.70
#: From configs/physics.yaml strategy.pit_lane_loss_s.
_DEMO_PIT_LANE_LOSS_S = 21.0
#: From tyremind.data.synthetic.DEFAULT_COMPOUND_RATES. Keyed by the weekend-
#: relative label, same as synthetic.py, for familiarity -- but note that
#: `CarStatusData.m_actualTyreCompound` (see module docstring on COMPOUND_CODE)
#: never actually reports "SOFT"/"MEDIUM"/"HARD"; only the C-number does. The
#: mapping below to a specific C-number is this generator's own arbitrary
#: choice of "a typical weekend's three compounds", exactly the kind of
#: label->compound_id resolution tyremind.data.compounds exists to make
#: explicit rather than assumed.
_DEMO_COMPOUND_DEGRADATION_S_PER_LAP = {"SOFT": 0.115, "MEDIUM": 0.072, "HARD": 0.041}
_DEMO_LABEL_TO_GAME_CODE = {"SOFT": 17, "MEDIUM": 18, "HARD": 20}  # C4, C3, C1

#: From configs/physics.yaml tyre.thermal.optimum_window_C.
_DEMO_TYRE_TEMP_START_C = 90.0
_DEMO_TYRE_TEMP_MAX_C = 110.0
_DEMO_TYRE_TEMP_RISE_PER_LAP_C = 0.8
#: Illustrative only -- no prior exists for these anywhere in the project.
_DEMO_TYRE_PRESSURE_START_PSI = 23.0
_DEMO_TYRE_PRESSURE_RISE_PER_LAP_PSI = 0.01
_DEMO_BRAKE_TEMP_C = 450.0
_DEMO_WEAR_PCT_PER_LAP = 2.2

#: One sample per lap is realistic-ish for the purpose here (the real game
#: sends far more often), spaced far enough apart in frame number that frame
#: ordering in the capture is unambiguous.
_DEMO_FRAMES_PER_LAP = 5_400  # ~90s at 60Hz


@dataclass(frozen=True)
class DemoCaptureConfig:
    """Controls for a generated demo capture.

    Defaults describe a plausible one-stop 50-lap Grand Prix distance on a
    circuit with a ~92s reference lap -- the same reference lap time
    `tyremind.data.synthetic.SessionConfig` uses, so the two illustrative
    datasets are at least talking about the same kind of lap.

    Attributes:
        n_laps: Race distance.
        base_lap_time_s: Reference lap time with a full tank and fresh tyres.
        fuel_capacity_kg: Starting fuel load. `None` derives it as
            `burn_rate_kg_per_lap * n_laps` -- exactly enough for the race
            distance, so fuel never goes negative regardless of `n_laps`.
            Passing an explicit value (e.g. the regulation's 110.0) is fine
            too, but then a lap count that doesn't match will run the tank
            dry before the flag -- realistic, but worth choosing on purpose.
        burn_rate_kg_per_lap: Constant burn rate (real burn is not perfectly
            constant, but neither is the project's own current prior).
        starting_compound: Compound for laps 1..pit_lap.
        pit_lap: Lap after which the car pits. `None` means no stop.
        pit_compound: Compound fitted at the stop.
        car_index: Which grid slot (of 22) this car occupies in the capture.
        seed: Deterministic noise seed -- same seed, same capture, always.
    """

    n_laps: int = 50
    base_lap_time_s: float = 92.0
    fuel_capacity_kg: float | None = None
    burn_rate_kg_per_lap: float = _DEMO_BURN_RATE_KG_PER_LAP
    starting_compound: str = "MEDIUM"
    pit_lap: int | None = 24
    pit_compound: str = "HARD"
    car_index: int = 0
    seed: int = 20260913
    _noise_sd_s: float = field(default=0.05, repr=False)

    @property
    def resolved_fuel_capacity_kg(self) -> float:
        return (
            self.fuel_capacity_kg
            if self.fuel_capacity_kg is not None
            else self.burn_rate_kg_per_lap * self.n_laps
        )


def _pack_header_bytes(packet_id: int, frame_identifier: int, session_time: float) -> bytes:
    return _HEADER_STRUCT.pack(
        2025, 25, 1, 0, 1, packet_id, 0xF1F1F1F1, session_time, frame_identifier,
        frame_identifier, 0, 255,
    )


def _pack_car_status_entry_bytes(fuel_in_tank_kg: float, fuel_capacity_kg: float, compound_code: int, tyres_age: int) -> bytes:
    return _CAR_STATUS_STRUCT.pack(
        0, 0, 1, 50, 0,  # traction, abs, fuelMix, frontBrakeBias, pitLimiter
        fuel_in_tank_kg, fuel_capacity_kg, fuel_in_tank_kg / max(_DEMO_BURN_RATE_KG_PER_LAP, 0.01),
        15000, 4500, 8, 1, 0,  # maxRPM, idleRPM, maxGears, drsAllowed, drsActivationDistance
        compound_code, compound_code, tyres_age,
        0,  # vehicleFiaFlags
        500_000.0, 50_000.0, 2_000_000.0, 0,  # engine power ICE/MGUK, ERS store, deploy mode
        10_000.0, 5_000.0, 8_000.0, 0,  # ERS harvested/deployed this lap, networkPaused
    )


def _pack_lap_data_entry_bytes(last_lap_time_ms: int, current_lap_num: int) -> bytes:
    return _LAP_DATA_STRUCT.pack(
        last_lap_time_ms, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # times, sectors, deltas
        0.0, 0.0, 0.0,  # lapDistance, totalDistance, safetyCarDelta
        1, current_lap_num, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2,  # position..driverStatus/resultStatus
        0, 0, 0, 0, 0.0, 0,  # pitLaneTimerActive..speedTrapFastestLap
    )


def _pack_car_telemetry_entry_bytes(
    tyre_pressure_psi: float, tyre_surface_temp_c: float, tyre_inner_temp_c: float, brake_temp_c: float
) -> bytes:
    return _CAR_TELEMETRY_STRUCT.pack(
        0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0,  # speed..revLightsBitValue (unused here)
        *([int(brake_temp_c)] * 4),
        *([int(tyre_surface_temp_c)] * 4),
        *([int(tyre_inner_temp_c)] * 4),
        0,  # engineTemperature
        *([float(tyre_pressure_psi)] * 4),
        0, 0, 0, 0,  # surfaceType
    )


def _pack_car_damage_entry_bytes(tyre_wear_pct: float) -> bytes:
    return _CAR_DAMAGE_STRUCT.pack(
        *([float(tyre_wear_pct)] * 4),
        0, 0, 0, 0,  # tyresDamage
        0, 0, 0, 0,  # brakesDamage
        0, 0, 0, 0,  # tyreBlisters
        *([0] * 18),  # the 18 individual damage/wear fields this module ignores
    )


def _pack_packet(packet_id: int, frame_identifier: int, entry_for_car_index: bytes, car_index: int, filler_entry: bytes) -> bytes:
    header = _pack_header_bytes(packet_id, frame_identifier, frame_identifier / 60.0)
    cars = b"".join(entry_for_car_index if i == car_index else filler_entry for i in range(MAX_CARS))
    return header + cars


def generate_demo_capture(config: DemoCaptureConfig | None = None) -> bytes:
    """Generate a byte-for-byte valid, clearly synthetic capture file.

    The output is real UDP packet bytes (length-prefixed exactly as
    `iter_length_prefixed_packets` expects) for one car across a full race
    distance, covering all four packet types `build_lap_ground_truth_table`
    reads -- Lap Data, Car Status, Car Telemetry and Car Damage. Every
    coefficient is pulled from configs/physics.yaml or tyremind.data.synthetic
    where a prior already exists; see the module-level comment above this
    section for exactly which fields do and do not have one. Round-tripping
    the output through `parse_capture_bytes` and `build_lap_ground_truth_table`
    is the fastest way to see the shape of what a real capture will eventually
    produce.

    Args:
        config: Race parameters. Defaults to a 50-lap one-stop race.

    Returns:
        Capture file bytes, ready to write to disk or feed to
        `parse_capture_bytes` directly.
    """
    import random

    cfg = config or DemoCaptureConfig()
    rng = random.Random(cfg.seed)
    filler_status = _pack_car_status_entry_bytes(0.0, 0.0, 0, 0)
    filler_lap = _pack_lap_data_entry_bytes(0, 0)
    filler_telemetry = _pack_car_telemetry_entry_bytes(0.0, 0.0, 0.0, 0.0)
    filler_damage = _pack_car_damage_entry_bytes(0.0)

    packets: list[bytes] = []

    def emit(payload: bytes) -> None:
        packets.append(struct.pack(">H", len(payload)) + payload)

    # Frame 0: establish the car is on lap 1 before anything has completed.
    emit(
        _pack_packet(
            PACKET_ID_LAP_DATA, 0, _pack_lap_data_entry_bytes(0, 1), cfg.car_index, filler_lap
        )
    )

    for lap in range(1, cfg.n_laps + 1):
        pitted = cfg.pit_lap is not None and lap > cfg.pit_lap
        compound = cfg.pit_compound if pitted else cfg.starting_compound
        tyre_age = (lap - 1 - cfg.pit_lap) if pitted else (lap - 1)
        fuel_capacity = cfg.resolved_fuel_capacity_kg
        fuel_at_lap_start = fuel_capacity - cfg.burn_rate_kg_per_lap * (lap - 1)

        lap_start_frame = (lap - 1) * _DEMO_FRAMES_PER_LAP
        emit(
            _pack_packet(
                PACKET_ID_CAR_STATUS,
                lap_start_frame,
                _pack_car_status_entry_bytes(
                    fuel_at_lap_start, fuel_capacity,
                    _DEMO_LABEL_TO_GAME_CODE[compound], tyre_age,
                ),
                cfg.car_index,
                filler_status,
            )
        )

        tyre_temp = min(_DEMO_TYRE_TEMP_START_C + _DEMO_TYRE_TEMP_RISE_PER_LAP_C * tyre_age, _DEMO_TYRE_TEMP_MAX_C)
        tyre_pressure = _DEMO_TYRE_PRESSURE_START_PSI + _DEMO_TYRE_PRESSURE_RISE_PER_LAP_PSI * tyre_age
        tyre_wear = min(_DEMO_WEAR_PCT_PER_LAP * tyre_age, 100.0)
        emit(
            _pack_packet(
                PACKET_ID_CAR_TELEMETRY,
                lap_start_frame,
                _pack_car_telemetry_entry_bytes(tyre_pressure, tyre_temp, tyre_temp + 8.0, _DEMO_BRAKE_TEMP_C),
                cfg.car_index,
                filler_telemetry,
            )
        )
        emit(
            _pack_packet(
                PACKET_ID_CAR_DAMAGE, lap_start_frame,
                _pack_car_damage_entry_bytes(tyre_wear), cfg.car_index, filler_damage,
            )
        )

        fuel_burned_before_this_lap = cfg.burn_rate_kg_per_lap * (lap - 1)
        fuel_effect_s = -_DEMO_SECONDS_PER_KG * fuel_burned_before_this_lap
        degradation_s = _DEMO_COMPOUND_DEGRADATION_S_PER_LAP[compound] * tyre_age
        pit_penalty_s = _DEMO_PIT_LANE_LOSS_S if (cfg.pit_lap is not None and lap == cfg.pit_lap) else 0.0
        noise_s = rng.gauss(0.0, cfg._noise_sd_s)
        lap_time_s = cfg.base_lap_time_s + fuel_effect_s + degradation_s + pit_penalty_s + noise_s

        completion_frame = lap * _DEMO_FRAMES_PER_LAP
        emit(
            _pack_packet(
                PACKET_ID_LAP_DATA,
                completion_frame,
                _pack_lap_data_entry_bytes(round(lap_time_s * 1000), lap + 1),
                cfg.car_index,
                filler_lap,
            )
        )

    return b"".join(packets)
