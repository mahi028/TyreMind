"""Tests for the FIA pit-stop-summary parser.

`MONZA_2025_TEXT` is `pypdf`'s real extracted text from FIA's 2025 Italian
Grand Prix "Race Pit Stop Summary" PDF (downloaded and extracted with pypdf
during development from
https://www.fia.com/sites/default/files/2025_16_ita_f1_r0_timing_racepitstopsummary_v01.pdf),
not a hand-written approximation of the format. That matters here more than
usual: `pypdf` puts one table cell per line rather than one row per line, and
its handling of the trailing copyright notice splits digits like the "1" in
"F1" onto their own lines -- exactly the kind of thing a guessed fixture would
not reproduce, and exactly what `parse_pitstop_text`'s field-type validation
has to survive.
"""

from __future__ import annotations

import pytest

from tyremind.data.pit_stops import (
    parse_pitstop_text,
    slugify_event_name,
    summarize,
)

MONZA_2025_TEXT = """\
NO
DRIVER
ENTRANT
LAP
TIME OF DAY
STOP
DURATION
TOTAL TIME
30
Liam LAWSON
Visa Cash App Racing Bulls F1 Team
9
15:16:28
1
24.156
24.156
87
Oliver BEARMAN
MoneyGram Haas F1 Team
18
15:29:07
1
25.372
25.372
22
Yuki TSUNODA
Oracle Red Bull Racing
19
15:30:30
1
24.372
24.372
5
Gabriel BORTOLETO
Stake F1 Team Kick Sauber
20
15:31:51
1
26.143
26.143
14
Fernando ALONSO
Aston Martin Aramco F1 Team
20
15:31:51
1
25.160
25.160
63
George RUSSELL
Mercedes-AMG PETRONAS F1 Team
27
15:41:27
1
25.021
25.021
12
Kimi ANTONELLI
Mercedes-AMG PETRONAS F1 Team
28
15:43:02
1
24.916
24.916
55
Carlos SAINZ
Atlassian Williams Racing
30
15:45:54
1
24.605
24.605
6
Isack HADJAR
Visa Cash App Racing Bulls F1 Team
32
15:48:45
1
24.326
24.326
16
Charles LECLERC
Scuderia Ferrari HP
33
15:49:40
1
24.235
24.235
43
Franco COLAPINTO
BWT Alpine F1 Team
33
15:50:20
1
24.971
24.971
1
Max VERSTAPPEN
Oracle Red Bull Racing
37
15:54:52
1
24.545
24.545
44
Lewis HAMILTON
Scuderia Ferrari HP
38
15:56:43
1
23.971
23.971
23
Alexander ALBON
Atlassian Williams Racing
41
16:01:01
1
24.717
24.717
81
Oscar PIASTRI
McLaren Formula 1 Team
45
16:06:02
1
23.602
23.602
4
Lando NORRIS
McLaren Formula 1 Team
46
16:07:22
1
27.498
27.498
18
Lance STROLL
Aston Martin Aramco F1 Team
49
16:12:23
1
38.418
38.418
10
Pierre GASLY
BWT Alpine F1 Team
49
16:12:32
1
25.012
25.012
31
Esteban OCON
MoneyGram Haas F1 Team
51
16:15:06
1
30.754
30.754
The F
1
 FORMULA
1
 logo, F
1
 logo, FORMULA
1
, FORMULA ONE, F
1
, FIA FORMULA ONE WORLD CHAMPIONSHIP, GRAND PRIX and related marks are trade marks of Formula One Licensing BV, a
Formula
1
 company. The FIA logo is a trade mark of the Federation Internationale de l'Automobile. All rights reserved.
No part of these results/data may be reproduced, stored in a retrieval system or transmitted in any form or by any means electronic, mechanical, photocopying, recording, broadcasting or otherwise without
prior permission of the copyright holder except for reproduction in local/national/international daily press and regular printed publications on sale to the public within
90
 days of the event to which the
results/data relate and provided that the copyright symbol and name of copyright owner appears.
(c) 2025 Formula One World Championship Limited
FORMULA 1 PIRELLI GRAN PREMIO D'ITALIA 2025 - Monza
Race Pit Stop Summary
"""


class TestParsePitstopText:
    def test_parses_every_data_row_and_ignores_boilerplate(self) -> None:
        rows = parse_pitstop_text(MONZA_2025_TEXT)
        assert len(rows) == 19

    def test_header_and_copyright_notice_produce_no_rows(self) -> None:
        rows = parse_pitstop_text(MONZA_2025_TEXT)
        assert all(r.driver_number > 0 for r in rows)

    def test_short_entrant_name_parses(self) -> None:
        rows = {r.driver_number: r for r in parse_pitstop_text(MONZA_2025_TEXT)}
        leclerc = rows[16]
        assert leclerc.driver_name == "Charles LECLERC"
        assert leclerc.entrant == "Scuderia Ferrari HP"

    def test_long_entrant_name_parses(self) -> None:
        rows = {r.driver_number: r for r in parse_pitstop_text(MONZA_2025_TEXT)}
        lawson = rows[30]
        assert lawson.driver_name == "Liam LAWSON"
        assert lawson.entrant == "Visa Cash App Racing Bulls F1 Team"

    def test_entrant_name_containing_a_digit_parses(self) -> None:
        # "Stake F1 Team Kick Sauber" contains "F1" -- a token with a digit in
        # it that must not be mistaken for the LAP field.
        rows = {r.driver_number: r for r in parse_pitstop_text(MONZA_2025_TEXT)}
        bortoleto = rows[5]
        assert bortoleto.driver_name == "Gabriel BORTOLETO"
        assert bortoleto.entrant == "Stake F1 Team Kick Sauber"
        assert bortoleto.lap == 20

    def test_numeric_fields_parse_correctly(self) -> None:
        rows = {r.driver_number: r for r in parse_pitstop_text(MONZA_2025_TEXT)}
        verstappen = rows[1]
        assert verstappen.lap == 37
        assert verstappen.time_of_day == "15:54:52"
        assert verstappen.stop_number == 1
        assert verstappen.duration_s == pytest.approx(24.545)
        assert verstappen.total_time_s == pytest.approx(24.545)

    def test_no_rows_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_pitstop_text("nothing here matches the format\n")


class TestSummarize:
    def test_counts_and_median(self) -> None:
        rows = parse_pitstop_text(MONZA_2025_TEXT)
        summary = summarize(rows)
        assert summary["n_stops"] == 19
        assert summary["min_s"] == pytest.approx(23.602)

    def test_outlier_stop_excluded_from_clean_median(self) -> None:
        # Stroll's 38.418s stop is a clear outlier against a ~24-25s field and
        # must not be allowed to drag the "clean" pit-lane-loss estimate up.
        rows = parse_pitstop_text(MONZA_2025_TEXT)
        summary = summarize(rows)
        assert summary["clean_n_stops"] == summary["n_stops"] - 1
        assert summary["clean_median_s"] < summary["median_s"]


class TestSlugifyEventName:
    @pytest.mark.parametrize(
        ("event_name", "expected"),
        [
            ("Italian Grand Prix", "italian-grand-prix"),
            ("Sao Paulo Grand Prix", "sao-paulo-grand-prix"),
            ("United States Grand Prix", "united-states-grand-prix"),
            ("Emilia Romagna Grand Prix", "emilia-romagna-grand-prix"),
        ],
    )
    def test_known_events(self, event_name: str, expected: str) -> None:
        assert slugify_event_name(event_name) == expected
