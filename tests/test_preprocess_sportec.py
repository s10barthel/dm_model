from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from datatools.ball_carries import _parse_raw_control_records
from scripts import preprocess_sportec as preprocessing


def _touch(path: Path, text: str = "fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _match(match_id: str, season: str) -> preprocessing.MatchFiles:
    placeholder = Path(match_id)
    return preprocessing.MatchFiles(
        match_id=match_id,
        season=season,
        meta_path=placeholder,
        event_path=placeholder,
        tracking_path=placeholder,
    )


def test_discovers_all_three_layouts_and_ignores_appledouble_files(monkeypatch, tmp_path) -> None:
    roots = {season: tmp_path / season for season in ("22_23", "23_24", "24_25")}
    match_ids = {"22_23": "DFL-MAT-22", "23_24": "DFL-MAT-23", "24_25": "DFL-MAT-24"}

    match_id = match_ids["22_23"]
    for relative in ("starting_players", "event_data", "tracking_data", "events_advanced"):
        _touch(roots["22_23"] / relative / match_id)
        _touch(roots["22_23"] / relative / f"._{match_id}")
    _touch(roots["22_23"] / "master" / "matchplan")

    match_id = match_ids["23_24"]
    for relative in ("match_information", "event_data", "tracking_data"):
        _touch(roots["23_24"] / relative / f"{match_id}.xml")
    _touch(roots["23_24"] / "KPI_Merged" / f"KPI_MGD_{match_id}.csv")

    match_id = match_ids["24_25"]
    for relative in ("match_information/starting_players", "event_data", "tracking_data", "KPI_Merged"):
        _touch(roots["24_25"] / relative / match_id)
    _touch(roots["24_25"] / "match_information" / "master" / "matchplan")

    monkeypatch.setattr(preprocessing, "RAW_SEASON_ROOTS", roots)
    matches = {match.season: match for match in preprocessing.discover_match_files()}

    assert set(matches) == set(roots)
    assert matches["22_23"].meta_path == roots["22_23"] / "starting_players" / match_ids["22_23"]
    assert matches["22_23"].kpi_path == roots["22_23"] / "events_advanced" / match_ids["22_23"]
    assert matches["22_23"].kpi_format == preprocessing.KpiFormat.ADVANCED_EVENTS_XML
    assert matches["23_24"].kpi_format == preprocessing.KpiFormat.CSV
    assert matches["24_25"].kpi_format == preprocessing.KpiFormat.ADVANCED_EVENTS_XML


def test_advanced_events_xml_dispatch_preserves_frame_and_reception(tmp_path) -> None:
    kpi_path = _touch(
        tmp_path / "advanced",
        """<PutDataRequest>
        <Event><Play EventId="event-1" SyncedFrameId="101" SyncedEventTime="2022-08-05T18:30:01Z"
          SyncSuccessful="true" ReceiverId="player-2" ReceptionId="reception-1" /></Event>
        <Event><Reception EventId="reception-1" PlayId="event-1" PlayerId="player-2"
          SyncedFrameId="125" SyncedEventTime="2022-08-05T18:30:02Z" /></Event>
        </PutDataRequest>""",
    )
    match = preprocessing.MatchFiles(
        "DFL-MAT-22",
        "22_23",
        tmp_path / "meta",
        tmp_path / "events",
        tmp_path / "tracking",
        kpi_path=kpi_path,
        kpi_format=preprocessing.KpiFormat.ADVANCED_EVENTS_XML,
    )

    kpi = preprocessing.load_kpi_merged_table(match)

    assert kpi["EVENT_ID"].tolist() == ["event-1"]
    assert kpi["FRAME_NUMBER"].tolist() == [101]
    assert kpi["RECFRM"].tolist() == [125]
    assert kpi["PUID2"].tolist() == ["player-2"]


def test_csv_kpi_dispatch_and_unsupported_format(tmp_path) -> None:
    csv_path = _touch(
        tmp_path / "kpi.csv",
        "EVENT_ID;PUID2;GDCP_EVENT_TIME;TRACKING_TIME;FRAME_NUMBER;RECFRM;NORECEIVER\n"
        "event-1;player-2;2023-08-18 20:00:01;2023-08-18 20:00:01;101;125;FALSE\n",
    )
    match = preprocessing.MatchFiles(
        "DFL-MAT-23",
        "23_24",
        tmp_path / "meta",
        tmp_path / "events",
        tmp_path / "tracking",
        kpi_path=csv_path,
        kpi_format=preprocessing.KpiFormat.CSV,
    )
    assert preprocessing.load_kpi_merged_table(match)["FRAME_NUMBER"].tolist() == [101]

    unsupported = preprocessing.MatchFiles(
        **{**match.__dict__, "kpi_format": None},
    )
    with pytest.raises(ValueError, match="Unsupported KPI format"):
        preprocessing.load_kpi_merged_table(unsupported)


def test_missing_kpi_file_has_feed_neutral_error(tmp_path) -> None:
    match = preprocessing.MatchFiles(
        "DFL-MAT-22",
        "22_23",
        tmp_path / "meta",
        tmp_path / "events",
        tmp_path / "tracking",
        kpi_path=tmp_path / "missing",
        kpi_format=preprocessing.KpiFormat.ADVANCED_EVENTS_XML,
    )
    with pytest.raises(FileNotFoundError, match="KPI/AdvancedEvents"):
        preprocessing.load_kpi_merged_table(match)


def test_season_then_match_id_then_limit_filtering() -> None:
    matches = [
        _match("22-a", "22_23"),
        _match("22-b", "22_23"),
        _match("23-a", "23_24"),
        _match("24-a", "24_25"),
    ]
    selected = preprocessing.filter_matches(
        matches,
        requested_seasons=["22_23", "24_25"],
        requested_ids=["22-b", "24-a"],
        limit=1,
    )
    assert [match.match_id for match in selected] == ["22-b"]
    assert preprocessing.filter_matches(matches, None, None, None) == matches


def test_season_selection_is_subset_mode() -> None:
    assert preprocessing.is_subset_mode(Namespace(season=["22_23"], match_id=None, limit=None))
    assert not preprocessing.is_subset_mode(Namespace(season=None, match_id=None, limit=None))


def test_carry_cli_modes_and_defaults() -> None:
    normal = preprocessing.parse_args(["--season", "22_23", "--skip-carry-artifacts"])
    assert normal.season == ["22_23"]
    assert normal.skip_carry_artifacts
    assert not normal.skip_sync
    assert not normal.carry_artifacts_only

    carry_only = preprocessing.parse_args(["--carry-artifacts-only"])
    assert carry_only.season is None
    assert carry_only.carry_artifacts_only

    with pytest.raises(SystemExit):
        preprocessing.parse_args(["--carry-artifacts-only", "--skip-carry-artifacts"])


def test_subset_event_aggregate_preserves_unselected_matches(monkeypatch, tmp_path) -> None:
    event_path = tmp_path / "event.parquet"
    pd.DataFrame(
        {"stats_perform_match_id": ["old-match"], "action_id": [1]}
    ).to_parquet(event_path, index=False)
    monkeypatch.setattr(preprocessing, "EVENT_PATH", event_path)

    merged = preprocessing.merge_unsynced_event_aggregate(
        {"new-match": pd.DataFrame({"stats_perform_match_id": ["new-match"], "action_id": [2]})}
    )

    assert set(merged["stats_perform_match_id"]) == {"old-match", "new-match"}


def test_control_parser_tolerates_period_zero_and_incomplete_tackle(tmp_path) -> None:
    event_path = _touch(
        tmp_path / "events.xml",
        """<PutDataRequest>
        <Event EventId="kick-1" EventTime="2022-08-05T18:30:00Z"><KickOff GameSection="firstHalf" /></Event>
        <Event EventId="tackle-1" EventTime="2022-08-05T18:31:00Z"><TacklingGame Winner="p1" Loser="" /></Event>
        <Event EventId="end-1" EventTime="2022-08-05T19:15:00Z"><FinalWhistle GameSection="firstHalf" /></Event>
        <Event EventId="outside-1" EventTime="2022-08-05T19:16:00Z"><Play Team="t1" Player="p1"><Pass /></Play></Event>
        </PutDataRequest>""",
    )

    records = _parse_raw_control_records(event_path)

    tackle = next(record for record in records if record["event_id"] == "tackle-1")
    outside = next(record for record in records if record["event_id"] == "outside-1")
    assert tackle["loser_player_id"] == ""
    assert tackle["winner_role"] is None
    assert outside["period_id"] == 0
