"""Runtime Sportec season membership from raw match-information filenames."""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from collections.abc import Iterable

from project_config import RAW_SEASON_ROOTS

_METADATA_LAYOUTS = {
    "22_23": ("starting_players", "*"),
    "23_24": ("match_information", "*.xml"),
    "24_25": ("match_information/starting_players", "*"),
}


def add_season_argument(parser: argparse.ArgumentParser, *, help_text=None) -> None:
    parser.add_argument(
        "--season", action="append", choices=sorted(RAW_SEASON_ROOTS),
        help=help_text or "Restrict Sportec to season(s); repeat for multiple seasons. Requires raw match-information files. In inference, combines with --split (use --split all for the whole eligible season).",
    )


def select_season_match_ids(
    candidate_ids: Iterable[str], requested_seasons: Iterable[str] | None,
) -> list[str]:
    candidates = [str(value) for value in candidate_ids]
    if not requested_seasons:
        return candidates
    seasons = set(requested_seasons)
    unknown = seasons - RAW_SEASON_ROOTS.keys()
    if unknown:
        raise ValueError(f"Unknown Sportec seasons: {', '.join(sorted(unknown))}")
    membership: dict[str, str] = {}
    inspected: list[str] = []
    for season, root in RAW_SEASON_ROOTS.items():
        relative, pattern = _METADATA_LAYOUTS[season]
        directory = Path(root) / relative
        inspected.append(f"{season}: {directory}")
        if not directory.is_dir():
            if season in seasons:
                raise FileNotFoundError(f"Sportec season {season} match-information directory unavailable: {directory}")
            continue
        for path in sorted(directory.glob(pattern)):
            if not path.is_file() or not path.name.startswith("DFL-MAT-"):
                continue
            match_id = path.stem
            previous = membership.get(match_id)
            if previous is not None and previous != season:
                raise ValueError(f"Conflicting season membership for {match_id}: {previous} and {season}")
            membership[match_id] = season
    candidates = list(dict.fromkeys(candidates))
    unresolved = [match_id for match_id in candidates if match_id not in membership]
    if unresolved:
        raise ValueError(
            f"Unresolved Sportec season membership for: {', '.join(unresolved)}. "
            f"Inspected match-information locations: {'; '.join(inspected)}"
        )
    selected = [match_id for match_id in candidates if membership[match_id] in seasons]
    if not selected:
        raise ValueError("No Sportec matches selected after season filtering; check match IDs, split and available artifacts.")
    return selected


@lru_cache(maxsize=8)
def _matchplan_entries(root):
    entries = {}
    for directory in (Path(root) / "master/matchplan", Path(root) / "match_information/master/matchplan"):
        paths = [directory] if directory.is_file() else sorted(directory.glob("*")) if directory.is_dir() else []
        for path in paths:
            if path.is_file():
                for element in ET.parse(path).getroot().iter():
                    if element.get("MatchId") and element.get("MatchDay"):
                        entries[element.get("MatchId")] = element.attrib
    return entries


def heldout_season_match_ids(seasons, half=None):
    """Resolve the requested population before inspecting model/artifact coverage."""
    membership = {}
    for season in dict.fromkeys(seasons):
        if season not in RAW_SEASON_ROOTS:
            raise ValueError(f"Unknown Sportec season: {season}.")
        root = Path(RAW_SEASON_ROOTS[season])
        relative, pattern = _METADATA_LAYOUTS[season]
        directory = root / relative
        if not directory.is_dir():
            raise FileNotFoundError(f"Sportec season {season} match-information directory unavailable: {directory}")
        catalog = {match_id: dict(attributes) for match_id, attributes in _matchplan_entries(str(root)).items()}
        for path in sorted(directory.glob(pattern)):
            if not path.is_file() or not path.name.startswith("DFL-MAT-"):
                continue
            general = ET.parse(path).getroot().find(".//General")
            attributes = general.attrib if general is not None else {}
            match_id = attributes.get("MatchId", path.stem)
            catalog[match_id] = {**catalog.get(match_id, {}), **{key: value for key, value in attributes.items() if value}}
        season_count = 0
        for match_id, attributes in catalog.items():
            if half:
                try:
                    matchday = int(attributes["MatchDay"])
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"Missing or invalid MatchDay for {match_id} in season {season}.") from exc
                if not 1 <= matchday <= 34:
                    raise ValueError(f"Invalid Bundesliga MatchDay={matchday} for {match_id}.")
                if (half == "first") != (matchday <= 17):
                    continue
            previous = membership.get(match_id)
            if previous is not None and previous != season:
                raise ValueError(f"Conflicting season membership for {match_id}.")
            membership[match_id] = season
            season_count += 1
        if not season_count:
            raise ValueError(f"No matches selected for season {season}, half={half or 'all'}.")
    return sorted(membership)
