"""Runtime Sportec season membership from raw match-information filenames."""
from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Iterable

from project_config import RAW_SEASON_ROOTS

_METADATA_LAYOUTS = {
    "22_23": ("starting_players", "*"),
    "23_24": ("match_information", "*.xml"),
    "24_25": ("match_information/starting_players", "*"),
}


def add_season_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--season", action="append", choices=sorted(RAW_SEASON_ROOTS),
        help="Restrict Sportec to season(s); repeat for multiple seasons. Requires raw match-information files. In inference, combines with --split (use --split all for the whole eligible season).",
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
