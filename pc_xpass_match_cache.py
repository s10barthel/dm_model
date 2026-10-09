"""Match checkpoints for possession-state pc-xPass caches.

The numerical worker still uses possession partitions internally. This parent-side
transaction translates them to explicit identities and publishes once per match.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datatools.possession_frames import STATE_CONTRACT

FORMAT = "match_possession_frames_v1"
DATASETS = {"sportec", "skillcorner"}
IDENTITY = ["match_id", "possession_id", "state_frame_id"]
REQUIRED = IDENTITY + ["possessor_id", "possession_provenance_id", "action_index", "physical_state_hash", "frame_scope"]
_ACTIVE = ContextVar("pc_xpass_match_transaction", default=None)


def read_json(path, *, memo=None):
    path = Path(path)
    if not path.exists():
        return {}
    stat = path.stat()
    key, signature = ("json", str(path.resolve())), (stat.st_mtime_ns, stat.st_size)
    if memo is not None and key in memo and memo[key][0] == signature:
        return memo[key][1]
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if memo is not None:
        memo[key] = (signature, value)
    return value


def atomic_json(path, value):
    from pc_xpass_versions import atomic_json as write
    write(Path(path), value)


def require_format(root, *, allow_new=False, memo=None):
    root = Path(root)
    metadata = read_json(root / "metadata.json", memo=memo)
    if allow_new and not metadata and not any((root / "matches").glob("*.parquet")):
        return
    if metadata.get("cache_format") != FORMAT:
        raise ValueError(f"Unsupported {root.name} pc-xPass cache format. Regenerate this dataset into a new "
                         "pc-xPass cache version; expected match-level possession/frame schema.")


def initialize(root, *, dry_run=False):
    require_format(root, allow_new=True)
    if not dry_run:
        metadata = read_json(Path(root) / "metadata.json")
        if metadata.get("cache_format") != FORMAT:
            metadata["cache_format"] = FORMAT
            atomic_json(Path(root) / "metadata.json", metadata)


def active(root):
    value = _ACTIVE.get()
    return value if value is not None and value.root == Path(root).resolve() else None


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_match_id(match_id):
    if not str(match_id) or str(match_id) in {".", ".."} or any(char in str(match_id) for char in '<>:"/\\|?*'):
        raise ValueError("Expected an actual match ID without path separators")


def validate_rows(frame, match_id, *, lane=False):
    missing = set(REQUIRED + (["receiver_id", "opponent_id"] if lane else [])) - set(frame.columns)
    if missing:
        raise ValueError(f"Invalid possession/frame cache: missing columns {sorted(missing)}")
    if frame[REQUIRED].isna().any().any():
        raise ValueError("Null possession/frame cache identity")
    for column in ["possession_id", "state_frame_id", "action_index"]:
        numeric = pd.to_numeric(frame[column], errors="raise")
        if not numeric.eq(numeric.round()).all():
            raise ValueError(f"Non-integral {column}")
    if not frame.match_id.astype(str).eq(str(match_id)).all():
        raise ValueError("Match filename and row identity differ")
    if not frame.action_index.eq(frame.state_frame_id).all():
        raise ValueError("action_index must equal state_frame_id")
    keys = IDENTITY + (["receiver_id", "opponent_id"] if lane else [])
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate possession/frame cache identities")
    if not frame.empty and (frame.groupby("possession_id").possession_provenance_id.nunique() > 1).any():
        raise ValueError("Multiple provenance variants for a possession")


def _read_parquet(path):
    table = pq.read_table(path)
    metadata = json.loads((table.schema.metadata or {}).get(b"pc_xpass", b"{}"))
    if metadata.get("cache_format") != FORMAT:
        raise ValueError("Missing embedded possession/frame metadata; regenerate the cache")
    return table.to_pandas(), metadata


def load_match(root, match_id, *, memo=None, include_lane=True):
    validate_match_id(match_id)
    root = Path(root).resolve()
    require_format(root, memo=memo)
    record_path = root / "completion" / f"{match_id}.json"
    record = read_json(record_path, memo=memo)
    if record.get("status") != "completed":
        raise ValueError(f"pc-xPass match {match_id} is missing or has a pending commit; resume generation")
    signatures = []
    for name in record.get("outputs", {}):
        if name not in {"matches", "lane_control"}:
            raise ValueError("Unsupported match output in completion record")
        path = root / name / f"{match_id}.parquet"
        if not path.exists():
            raise ValueError(f"Inconsistent pc-xPass match commit: missing {path}; resume generation")
        stat = path.stat()
        signatures.append((name, stat.st_mtime_ns, stat.st_size))
    key = ("match", str(root), str(match_id), include_lane)
    signature = (record_path.stat().st_mtime_ns, tuple(signatures))
    if memo is not None and key in memo and memo[key][0] == signature:
        return memo[key][1]
    frames = {}
    embedded = {}
    for name, info in record.get("outputs", {}).items():
        path = root / name / f"{match_id}.parquet"
        if not path.exists() or checksum(path) != info["sha256"]:
            raise ValueError(f"Inconsistent pc-xPass match commit: {path}; resume generation")
        if name == "lane_control" and not include_lane:
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows != info["rows"]:
                raise ValueError("Lane-control completion row count differs")
            continue
        frames[name], details = _read_parquet(path)
        validate_rows(frames[name], match_id, lane=name == "lane_control")
        if len(frames[name]) != info["rows"]:
            raise ValueError("Match completion row count differs")
        if embedded and embedded != details:
            raise ValueError("Main and lane-control provenance differ")
        embedded = details
    if "matches" not in frames:
        raise ValueError("Completed match has no main output")
    if (embedded.get("descriptors") != record.get("descriptors")
            or embedded.get("provenance", {}) != record.get("provenance", {})):
        raise ValueError("Embedded match provenance differs from completion record")
    result = frames, embedded, record
    if memo is not None:
        memo[key] = (signature, result)
    return result


@dataclass
class Selection:
    possession_id: int
    provenance_id: str
    memo: dict | None = None


def selection_for(state):
    identity = getattr(state, "pc_cache_match_id", None)
    if not identity:
        return None
    memo = getattr(state, "_pc_match_cache", None)
    if memo is None:
        memo = state._pc_match_cache = {}
    return Selection(int(state.event_index), str(identity), memo)


def load_possession(root, match_id, selection, *, lane=False, requested_frames=None):
    frames, _, record = load_match(root, match_id, memo=selection.memo, include_lane=lane)
    descriptor = record["descriptors"].get(selection.provenance_id)
    if descriptor is None or int(descriptor["possession_id"]) != selection.possession_id:
        raise ValueError("pc-xPass possession provenance differs; regenerate into a new cache version")
    name = "lane_control" if lane else "matches"
    if name not in frames:
        raise ValueError("Missing lane-control output; resume generation with lane-control export")
    result = frames[name].loc[frames[name].possession_id.eq(selection.possession_id)].copy()
    if not result.possession_provenance_id.eq(selection.provenance_id).all():
        raise ValueError("pc-xPass row provenance differs")
    if not result.possessor_id.astype(str).eq(str(descriptor["possessor_id"])).all():
        raise ValueError("pc-xPass row possessor differs from its provenance")
    if lane:
        covered = {int(row["state_frame_id"]): row for row in record.get("lane_states", [])
                   if row["possession_provenance_id"] == selection.provenance_id}
        counts = result.groupby("state_frame_id").size().to_dict()
        main = frames["matches"].loc[frames["matches"].possession_id.eq(selection.possession_id)]
        wanted = set(map(int, requested_frames)) if requested_frames is not None else set(main.state_frame_id)
        for row in main.to_dict("records"):
            frame_id = int(row["state_frame_id"])
            if frame_id in wanted:
                coverage = covered.get(frame_id, {})
                if (coverage.get("row_count") != counts.get(frame_id, 0)
                        or coverage.get("physical_state_hash") != row["physical_state_hash"]):
                    raise ValueError("Missing or inconsistent lane-control coverage; resume generation")
    if requested_frames is not None:
        available = set(result.state_frame_id.astype(int))
        if lane:
            available = {int(row["state_frame_id"]) for row in record.get("lane_states", [])
                         if row["possession_provenance_id"] == selection.provenance_id}
        if set(map(int, requested_frames)) - available:
            raise ValueError("pc-xPass cache is missing requested possession frames; resume generation")
    return result


class MatchAccumulator:
    def __init__(self, root, match_id, *, dry_run=False, request=None):
        validate_match_id(match_id)
        self.root, self.match_id = Path(root).resolve(), str(match_id)
        self.dry_run, self.request = dry_run, request or {}
        require_format(self.root, allow_new=dry_run)
        self.descriptors, self.parts, self.lanes, self.lane_states = {}, {}, {}, {}
        self.expected = {}
        self.record_coverage = True
        self.selection_report = None
        self.dirty = False
        self.lane_dirty = False
        self.failed = False
        self.old_record = read_json(self.root / "completion" / f"{self.match_id}.json")
        self.provenance = self.old_record.get("provenance", {})
        if self.old_record.get("status") == "completed":
            frames, details, _ = load_match(self.root, self.match_id)
            self.descriptors = details["descriptors"]
            for key, frame in frames["matches"].groupby("possession_provenance_id"):
                self.parts[key] = self._internal(frame, key)
            if "lane_control" in frames:
                for key, frame in frames["lane_control"].groupby("possession_provenance_id"):
                    self.lanes[key] = self._internal(frame, key)
            for row in self.old_record.get("lane_states", []):
                key = row["possession_provenance_id"]
                internal = dict(row, match_id=key)
                self.lane_states.setdefault(key, {})[int(row["state_frame_id"])] = internal
        elif self.old_record:
            self.dirty = True
        # Also clean staging abandoned before a pending marker was published.
        if not dry_run:
            for name in ("matches", "lane_control"):
                (self.root / name / f".{self.match_id}.pending.parquet").unlink(missing_ok=True)

    @staticmethod
    def _internal(frame, key):
        frame = frame.drop(columns=["possession_id", "possessor_id", "possession_provenance_id"]).copy()
        frame["match_id"] = key
        frame.attrs = {}
        return frame

    def __enter__(self):
        self.token = _ACTIVE.set(self)
        return self

    def __exit__(self, *exc):
        _ACTIVE.reset(self.token)

    def register(self, key, descriptor):
        descriptor = dict(descriptor)
        if str(descriptor["match_id"]) != self.match_id:
            raise ValueError("Accumulator received another match")
        saved = {**self.old_record.get("descriptors", {}), **self.descriptors}
        for old_key, old in saved.items():
            if int(old["possession_id"]) == int(descriptor["possession_id"]) and old_key != key:
                raise ValueError("Changed possession provenance; create a new pc-xPass version")
            if "spell_artifact_sha256" in old:
                for field in ("feature_run_id", "spell_artifact_sha256", "carry_definition"):
                    if old.get(field) != descriptor.get(field):
                        raise ValueError("Changed match provenance; create a new pc-xPass version")
        previous = saved.get(key, {})
        for field in ("possessor_id", "feature_run_id", "spell_artifact_sha256", "carry_definition",
                      "original_start_frame", "original_end_frame", "period_id"):
            if field in previous and previous[field] != descriptor.get(field):
                raise ValueError("Changed possession provenance; create a new pc-xPass version")
        if previous != descriptor:
            self.dirty = True
        self.descriptors[key] = descriptor

    def set_provenance(self, report):
        provenance = {key: report[key] for key in ("feature_run_id", "carry_definition", "spell_artifact_sha256")
                      if key in report}
        if self.provenance and self.provenance != provenance:
            raise ValueError("Changed match provenance; create a new pc-xPass version")
        self.provenance = provenance

    def expect(self, items):
        seen = set()
        for item in items:
            key = str(item["match_id"])
            for frame in map(int, item["state_frame_id"]):
                if (key, frame) in seen:
                    raise ValueError("Duplicate requested possession/frame identities")
                seen.add((key, frame))
                self.expected.setdefault(key, set()).add(frame)

    def read(self, key):
        if key not in self.parts:
            raise FileNotFoundError(f"No cached possession {key}")
        return self.parts[key].copy()

    def write(self, key, rows):
        if rows.action_index.duplicated().any():
            raise ValueError("Duplicate possession/frame cache identities")
        old = self.parts.get(key)
        if old is not None:
            if set(rows.columns).issubset(old.columns) and rows.action_index.isin(old.action_index).all():
                before = old.set_index("action_index").reindex(rows.action_index).reindex(columns=rows.columns)
                before["action_index"] = before.index
                before = before.reset_index(drop=True)
                after = rows.reset_index(drop=True)
                equal = before.eq(after) | (before.isna() & after.isna())
                if equal.fillna(False).to_numpy().all():
                    return
            rows = pd.concat([old.loc[~old.action_index.isin(rows.action_index)], rows], ignore_index=True)
        self.parts[key] = rows.copy()
        self.dirty = True

    def write_lane(self, key, records, states):
        from physical_pass_model import LANE_CONTROL_COLUMNS, _lane_control_identity
        frame = pd.DataFrame(records, columns=LANE_CONTROL_COLUMNS)
        if frame.duplicated(["action_index", "receiver_id", "opponent_id"]).any():
            raise ValueError("Duplicate lane-control identities")
        replaced = {int(row["action_index"]) for row in states}
        old = self.lanes.get(key)
        if old is not None:
            frame = pd.concat([old.loc[~old.action_index.isin(replaced)], frame], ignore_index=True)
        self.lanes[key] = frame
        self.lane_dirty = True
        for row in states:
            self.lane_states.setdefault(key, {})[int(row["action_index"])] = {
                **_lane_control_identity(row), "row_count": sum(int(r["action_index"]) == int(row["action_index"]) for r in records)}
        self.dirty = True

    def completed_lane(self, key):
        from physical_pass_model import _lane_control_state_key
        frame = self.lanes.get(key, pd.DataFrame(columns=["action_index"]))
        return {_lane_control_state_key(row) for row in self.lane_states.get(key, {}).values()
                if int(frame.action_index.eq(row["action_index"]).sum()) == row["row_count"]}

    def _external(self, frame, key):
        descriptor = self.descriptors[key]
        frame = frame.copy()
        frame["match_id"] = self.match_id
        frame["possession_id"] = int(descriptor["possession_id"])
        frame["possessor_id"] = str(descriptor["possessor_id"])
        frame["possession_provenance_id"] = key
        return frame

    def commit(self, *, export_lane=False):
        if self.dry_run:
            return
        if self.failed:
            raise ValueError("Match computation failed; unsaved results discarded")
        for key, expected in self.expected.items():
            if expected - set(self.parts.get(key, pd.DataFrame(columns=["action_index"])).action_index):
                raise ValueError("Incomplete match: missing requested states")
            if export_lane:
                from physical_pass_model import _lane_control_state_key
                completed = self.completed_lane(key)
                for row in self.parts[key].to_dict("records"):
                    if int(row["action_index"]) in expected and _lane_control_state_key(row) not in completed:
                        raise ValueError("Incomplete match: missing lane-control coverage")
        outputs = dict(self.old_record.get("outputs", {}))
        details = {"cache_format": FORMAT, "state_contract": STATE_CONTRACT,
                   "dataset": self.root.name, "descriptors": self.descriptors, "provenance": self.provenance}
        staged = []
        write = self.dirty or not self.old_record or self.provenance != self.old_record.get("provenance", {})
        if write:
            collections = {"matches": self.parts}
            if (self.lane_dirty or (export_lane and "lane_control" not in outputs)
                    or ("lane_control" in outputs and (self.descriptors != self.old_record.get("descriptors") or self.provenance != self.old_record.get("provenance", {})))):
                collections["lane_control"] = self.lanes
            for name, parts in collections.items():
                frames = [self._external(frame, key) for key, frame in parts.items()]
                from physical_pass_model import LANE_CONTROL_COLUMNS
                columns = list(dict.fromkeys(REQUIRED + (LANE_CONTROL_COLUMNS if name == "lane_control" else [])))
                frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
                validate_rows(frame, self.match_id, lane=name == "lane_control")
                for column in ("possession_id", "state_frame_id", "action_index"):
                    frame[column] = frame[column].astype("int64")
                for column in ("match_id", "possessor_id", "possession_provenance_id", "physical_state_hash", "frame_scope"):
                    frame[column] = frame[column].astype("string")
                sort = IDENTITY + (["receiver_id", "opponent_id"] if name == "lane_control" else [])
                frame = frame.sort_values(sort).reset_index(drop=True)
                path = self.root / name / f"{self.match_id}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_name(f".{self.match_id}.pending.parquet")
                table = pa.Table.from_pandas(frame, preserve_index=False)
                table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"pc_xpass": json.dumps(details).encode()})
                pq.write_table(table, temp)
                outputs[name] = {"sha256": checksum(temp), "rows": len(frame)}
                staged.append((temp, path))
        history = list(self.old_record.get("coverage", []))
        coverage = {**self.request, "states": {key: sorted(values) for key, values in self.expected.items()}}
        if self.record_coverage and coverage not in history:
            history.append(coverage)
        lane_states = [self._external(pd.DataFrame([row]), key).iloc[0].to_dict()
                       for key, states in self.lane_states.items() for row in states.values()]
        record = {"status": "completed", "cache_format": FORMAT, "descriptors": self.descriptors, "provenance": self.provenance,
                  "outputs": outputs, "coverage": history, "lane_states": lane_states,
                  "selection_report": self.selection_report if self.record_coverage else self.old_record.get("selection_report")}
        record_path = self.root / "completion" / f"{self.match_id}.json"
        if staged:
            atomic_json(record_path, {**record, "status": "pending"})
            for temp, path in staged:
                temp.replace(path)
        if record != self.old_record:
            atomic_json(record_path, record)


def begin(args, root, match_id):
    if not getattr(args, "pc_xpass", False):
        return None
    initialize(root, dry_run=args.dry_run)
    transaction = MatchAccumulator(root, match_id, dry_run=args.dry_run,
                                   request={"scope": args.scope, "frames": args.frames})
    transaction.record_coverage = getattr(args, "_pc_refresh", None) is None
    transaction.__enter__()
    return transaction


def finish(transaction, args):
    if transaction is not None:
        transaction.commit(export_lane=bool(getattr(args, "export_lane_control", False)) and getattr(args, "_pc_refresh", None) is None)
        if not transaction.dry_run:
            metadata_path = transaction.root.parent / "metadata.json"
            metadata = read_json(metadata_path)
            if metadata:
                metadata.setdefault("reconstruction", {}).setdefault(transaction.root.name, {})[transaction.match_id] = {
                    "match_id": transaction.match_id, "possessions": transaction.descriptors,
                    **{key: value for descriptor in transaction.descriptors.values() for key, value in descriptor.items()
                       if key in {"feature_run_id", "skillcorner_input_dir"}}}
                atomic_json(metadata_path, metadata)
            metadata = read_json(transaction.root / "metadata.json")
            if getattr(args, "pass_height_model_id", None):
                from models.pass_height import model_definition
                metadata.update(pass_height_model_id=args.pass_height_model_id,
                                pass_height_definition=model_definition(args._pass_height_model.args))
                atomic_json(transaction.root / "metadata.json", metadata)


def end(transaction):
    if transaction is not None:
        transaction.__exit__(None, None, None)
