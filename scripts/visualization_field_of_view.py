"""Resolve strict recipient and outfield rosters for the visualization-only overlay."""

from collections import Counter
import warnings

import numpy as np
import pandas as pd

from datatools import config
from datatools.field_of_view import CoveringSector, covering_sector


def add_field_of_view_argument(parser, *, unavailable=None):
    description = "Draw the minimum covering sector from the designated recipient to all other outfield players."
    if unavailable:
        description += f" Unavailable: {unavailable} does not provide a designated recipient."
    parser.add_argument("--field-of-view", action="store_true", help=description + " Default: off.")


def report_unsupported_field_of_view(args, source):
    if getattr(args, "field_of_view", False):
        warnings.warn(
            f"--field-of-view: {source} does not provide a designated recipient; no cones will be drawn.",
            stacklevel=2,
        )


def _confirmed(value):
    return pd.notna(value) and value in (True, "True", "true")


def _sector_from_display(frame, recipient, targets):
    players = [recipient, *targets]
    columns = [f"{player}_{axis}" for player in players for axis in ("x", "y")]
    xy = frame.reindex(columns).to_numpy(dtype=float, na_value=np.nan).reshape(-1, 2)
    return covering_sector(xy[0], xy[1:])


def _active_players(frame):
    by_team = {"home": [], "away": []}
    for column in frame.index:
        if column.startswith(("home_", "away_")) and column.endswith("_x") and "_goal_" not in column:
            player = column[:-2]
            if pd.notna(frame[column]) or pd.notna(frame.get(f"{player}_y")):
                by_team[player[:4]].append(player)
    return by_team


def _outfield_sector(frame, recipient, by_team, keepers):
    outfield = []
    for players in by_team.values():
        targets = [player for player in players if player not in keepers]
        if len(targets) != 10:
            return None, "incomplete outfield roster"
        outfield.extend(player for player in targets if player != recipient)
    sector = _sector_from_display(frame, recipient, outfield)
    return sector, None if sector is not None else "missing player coordinates or undefined bearing"


_HAWKEYE_FRAME_KEY = ["game_id", "half", "abs_time"]
_HAWKEYE_ROSTER_COLUMNS = ["uefa_player_id", "role", "team", "centroid_x", "centroid_y"]


def prepare_hawkeye_visualization_tracking(raw, clean_tracking, args):
    # Keep a separate view of raw identities and roles only when the flag is on.
    # Cleaning for inference drops officials and can hide malformed roster rows.
    source = None
    if getattr(args, "field_of_view", False):
        columns = ["id", "PlayerID", *_HAWKEYE_FRAME_KEY, *_HAWKEYE_ROSTER_COLUMNS]
        source = raw[columns].groupby(raw["id"].astype(str), sort=False)
    return clean_tracking(raw), source


def _hawkeye_roster(rows, recipient_id, recipient_role, team_map):
    rows = rows[_HAWKEYE_ROSTER_COLUMNS].drop_duplicates()
    if not rows["role"].isin([1, 2, 3, 4, 5]).all():
        return None, (), "unknown player role"
    recipient = rows["uefa_player_id"].eq(recipient_id).fillna(False)
    if recipient.sum() != 1 or rows.loc[recipient, "role"].iloc[0] != recipient_role:
        return None, (), "missing or ambiguous recipient"
    outfield = rows["role"].eq(1)
    counts = rows.loc[outfield, "team"].value_counts(dropna=False)
    if set(counts.index) != set(team_map) or not counts.eq(10).all() or any(not team.strip() for team in team_map):
        return None, (), "incomplete outfield roster"
    required = rows.loc[outfield | recipient]
    ids = pd.to_numeric(required["uefa_player_id"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(ids).all() or (ids < 0).any() or (ids % 1 != 0).any() or len(set(ids)) != len(ids):
        return None, (), "missing or ambiguous player ID"
    if not required["team"].isin(team_map).all():
        return None, (), "unknown team"
    if not np.isfinite(required[["centroid_x", "centroid_y"]].to_numpy(dtype=float, na_value=np.nan)).all():
        return None, (), "missing player coordinates"
    players = {int(pid): f"{team_map[team]}_{int(pid)}" for pid, team in zip(ids, required["team"])}
    return players[recipient_id], [player for pid, player in players.items() if pid != recipient_id], None


def hawkeye_field_of_view(source, situation, frame_ids) -> dict[int, CoveringSector]:
    """QC the raw roster, but draw using the frame's displayed (possibly adjusted) coordinates."""
    if source is None:
        return {}
    frame_ids = list(dict.fromkeys(frame_ids))
    if not frame_ids:
        return {}
    rows = source.get_group(str(situation.situation_id)).copy()
    for column in [*_HAWKEYE_FRAME_KEY, "PlayerID", "uefa_player_id", "role", "centroid_x", "centroid_y"]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows["team"] = rows["team"].astype("string")
    recipient_ids = rows["PlayerID"].unique()
    recipient_id = recipient_ids[0] if len(recipient_ids) == 1 else None
    roles = rows.loc[rows["uefa_player_id"].eq(recipient_id), "role"].unique()
    if pd.isna(recipient_id) or len(roles) != 1 or roles[0] not in (1, 2):
        warnings.warn(
            f"--field-of-view: {situation.situation_id}: no unambiguous clip recipient and role; "
            f"omitted the cone on {len(frame_ids)} frame(s).",
            stacklevel=2,
        )
        return {}
    groups = rows.groupby(_HAWKEYE_FRAME_KEY, sort=False)
    omitted = Counter()
    sectors = {}
    for frame_id in frame_ids:
        key = tuple(situation.frame_meta.loc[frame_id, _HAWKEYE_FRAME_KEY])
        if key not in groups.indices:
            omitted["missing source frame"] += 1
            continue
        origin, targets, reason = _hawkeye_roster(groups.get_group(key), recipient_id, roles[0], situation.team_map)
        if reason is None:
            sector = _sector_from_display(situation.tracking.loc[frame_id], origin, targets)
            if sector is not None:
                sectors[frame_id] = sector
                continue
            reason = "nonfinite coordinates or undefined bearing in displayed frame"
        omitted[reason] += 1
    if omitted:
        reasons = ", ".join(f"{reason}: {count}" for reason, count in sorted(omitted.items()))
        warnings.warn(
            f"--field-of-view: {situation.situation_id}: omitted the cone on "
            f"{sum(omitted.values())}/{len(frame_ids)} frame(s) ({reasons}).",
            stacklevel=2,
        )
    return sectors


def sportec_field_of_view(match, action_index, enabled) -> CoveringSector | None:
    if not enabled:
        return None

    def omit(reason):
        warnings.warn(f"--field-of-view: Sportec action {action_index}: no cone ({reason}).", stacklevel=3)

    action = match.actions.loc[action_index]
    recipient = action.get("receiver_id")
    sender = action.get("object_id")
    spadl_type = action.get("spadl_type")
    if (not isinstance(spadl_type, str) or spadl_type not in config.PASS
            or not _confirmed(action.get("success"))
            or not isinstance(sender, str) or not isinstance(recipient, str) or recipient == sender
            or not recipient.startswith(("home_", "away_")) or recipient[:4] != sender[:4]):
        return omit("no confirmed teammate recipient of a completed pass or cross")
    if "source_has_receiver_identifier" in action and not _confirmed(action["source_has_receiver_identifier"]):
        return omit("no source recipient identifier")

    frame = match.tracking.loc[int(action["frame_id"])]
    by_team = _active_players(frame)
    lineup = getattr(match, "lineup", None)
    if lineup is None or not {"object_id", "advanced_position"} <= set(lineup.columns):
        return omit("no lineup roles")
    active = [*by_team["home"], *by_team["away"]]
    roles = lineup[["object_id", "advanced_position"]].drop_duplicates()
    roles = roles.loc[roles["object_id"].isin(active)].set_index("object_id")["advanced_position"]
    if (recipient not in active or not roles.index.is_unique or not set(active) <= set(roles.index)
            or roles.isna().any() or roles.astype(str).str.strip().str.lower().isin(["", "unknown"]).any()):
        return omit("missing recipient or ambiguous player roles")
    sector, reason = _outfield_sector(frame, recipient, by_team, set(roles.index[roles.eq("goal_keeper")]))
    return sector if reason is None else omit(reason)


def skillcorner_field_of_view(context, possession, frame_ids, enabled) -> dict[int, CoveringSector]:
    """Use only a recorded, matched recipient of the possession's outgoing pass."""
    if not enabled:
        return {}
    frame_ids = list(dict.fromkeys(frame_ids))
    scene = f"SkillCorner {possession.match_id}/{possession.event_index}"
    rows = context["events"].loc[context["events"]["index"].eq(possession.event_index)]
    event = rows.iloc[0] if len(rows) == 1 else pd.Series(dtype=object)
    target = pd.to_numeric(event.get("player_targeted_id"), errors="coerce")
    end_type, outcome = event.get("end_type"), event.get("pass_outcome")
    if (pd.isna(target) or not np.isfinite(target) or target % 1 != 0
            or not isinstance(end_type, str) or end_type != "pass"
            or not isinstance(outcome, str) or outcome != "successful"
            or not _confirmed(event.get("is_pass_reception_matched"))):
        warnings.warn(f"--field-of-view: {scene}: no designated recipient of a matched completed pass; "
                      "no cones will be drawn.", stacklevel=2)
        return {}

    player_meta = context["player_meta"]
    sender = int(event["player_id"])
    team_id = player_meta.loc[player_meta["player_id"].eq(sender), "team_id"].iloc[0]
    recorded_roles = set()
    for player in context["match_data"].get("players", []):
        role = player.get("player_role") or {}
        if role.get("acronym") or role.get("name"):
            recorded_roles.add(int(player["id"]))
    roles = {}
    for player in player_meta.itertuples(index=False):
        if player.player_id in recorded_roles:
            prefix = "home" if player.team_id == team_id else "away"
            roles[f"{prefix}_{player.player_id}"] = player.is_goalkeeper
    recipient = f"home_{int(target)}"
    if target == sender or recipient not in roles:
        warnings.warn(f"--field-of-view: {scene}: missing or ambiguous teammate recipient role; "
                      "no cones will be drawn.", stacklevel=2)
        return {}
    keepers = {player for player, keeper in roles.items() if keeper}
    omitted = Counter()
    sectors = {}
    for frame_id in frame_ids:
        frame = possession.tracking.loc[frame_id]
        by_team = _active_players(frame)
        if recipient not in by_team["home"] or not set(by_team["home"] + by_team["away"]) <= roles.keys():
            omitted["missing recipient or unknown player role"] += 1
            continue
        sector, reason = _outfield_sector(frame, recipient, by_team, keepers)
        if reason is None:
            sectors[frame_id] = sector
        else:
            omitted[reason] += 1
    if omitted:
        reasons = ", ".join(f"{reason}: {count}" for reason, count in sorted(omitted.items()))
        warnings.warn(f"--field-of-view: {scene}: omitted the cone on "
                      f"{sum(omitted.values())}/{len(frame_ids)} frame(s) ({reasons}).", stacklevel=2)
    return sectors
