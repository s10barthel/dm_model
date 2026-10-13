"""Failure diagnostics for offline feature extensions; recovery stays manual."""
from __future__ import annotations

import copy
import json
import sys
import traceback
import warnings
from pathlib import Path

import torch

from project_config import atomic_write_json


def exception_chain(error):
    chain, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append({"type": type(error).__name__, "message": str(error)})
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return list(reversed(chain))


class ExtensionAttempt:
    def __init__(self, root, operation, mode, target_settings, metadata_factory=None, writer=None):
        self.root = Path(root)
        self.mode = mode
        self.metadata_factory, self.writer = metadata_factory, writer
        self.changed = False
        self.uncertain = False
        self.complete = False
        self.published = False
        self.data = dict(operation=operation, phase="staging", artifact_state="unchanged",
                         target_settings=copy.deepcopy(target_settings), current_command=None,
                         current_file=None, completed_replacements=0, primary_error_chain=[],
                         secondary_errors=[], leftover_temporary_paths=[])

    @property
    def artifact_state(self):
        if self.complete:
            return "complete"
        if self.uncertain:
            return "unknown"
        return "partial" if self.changed else "unchanged"

    def snapshot(self, *, hazardous=False):
        result = copy.deepcopy(self.data)
        result["artifact_state"] = "unknown" if hazardous else self.artifact_state
        if hazardous:
            # An abrupt stop can occur after a replacement and before its counter changes.
            result["completed_replacements"] = None
        return result

    def secondary(self, operation, error, path=None):
        message = f"{operation}: {type(error).__name__}: {error}"
        if path is not None:
            message += f" ({path})"
        self.data["secondary_errors"].append(message)
        print(message, file=sys.stderr)

    def persist(self, status="in_progress", error=None, *, final=False, hazardous=False):
        if self.metadata_factory is None:
            return
        payload = self.metadata_factory(status, error, final)
        payload["extension_attempt"] = self.snapshot(hazardous=hazardous)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            try:
                self.writer(self.root, payload)
            except BaseException:
                if final:
                    # The metadata rename can succeed before an interruption in
                    # publication bookkeeping or the separate summary refresh.
                    try:
                        self.published = json.loads((self.root / "metadata.json").read_text(encoding="utf-8-sig")) == payload
                    except (OSError, ValueError):
                        pass
                raise
        if final:
            self.published = True
        for item in caught:
            self.secondary("Metadata summary warning", item.message)
        if caught:
            # Include summary warnings without triggering another summary refresh.
            payload["extension_attempt"] = self.snapshot(hazardous=hazardous)
            try:
                atomic_write_json(self.root / "metadata.json", payload)
            except BaseException as secondary:
                if isinstance(secondary, (KeyboardInterrupt, SystemExit)):
                    raise
                self.secondary("Recording summary warnings failed", secondary)

    def enter(self, phase, *, current_file=None, current_command=None, hazardous=False):
        self.data.update(phase=phase, current_file=str(current_file) if current_file is not None else None,
                         current_command=current_command)
        self.persist(hazardous=hazardous)

    def begin_mutation(self, phase, *, current_file=None, current_command=None):
        self.enter(phase, current_file=current_file, current_command=current_command, hazardous=True)
        self.uncertain = True

    def mutation_finished(self):
        self.changed = True
        self.uncertain = False

    def finish(self):
        self.complete = True
        self.enter("publishing_metadata")
        self.persist("completed", final=True)

    def fail(self, error):
        self.data["primary_error_chain"] = exception_chain(error)
        self.data["primary_traceback"] = "".join(traceback.format_exception(error))
        status = "completed" if self.published or (self.mode == "in_place" and self.artifact_state == "unchanged") else "failed"
        self.data["artifact_state"] = self.artifact_state
        print(f"Feature extension failed during {self.data['phase']}; artifacts={self.artifact_state}; "
              f"metadata={self.root / 'metadata.json'}; manual inspection may be required.", file=sys.stderr)
        try:
            self.persist(status, str(error), final=self.published)
        except BaseException as secondary:
            self.secondary("Writing failure metadata failed", secondary, self.root / "metadata.json")
            error.add_note(self.data["secondary_errors"][-1])
        if self.data["leftover_temporary_paths"]:
            print("Leftover temporary files: " + ", ".join(self.data["leftover_temporary_paths"]), file=sys.stderr)


def save_staged_graphs(graphs, path, attempt):
    """Ensure our Python-owned handle closes even if PyTorch finalization raises."""
    stream = Path(path).open("wb")
    primary = None
    try:
        torch.save(graphs, stream)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            stream.close()
        except BaseException as secondary:
            if primary is None:
                raise
            attempt.secondary("Closing temporary file failed", secondary, path)
            primary.add_note(attempt.data["secondary_errors"][-1])


def cleanup_temporary_files(temporary, attempt, primary=None):
    for _, path in temporary:
        try:
            path.unlink(missing_ok=True)
        except Exception as secondary:
            attempt.secondary("Temporary-file cleanup failed", secondary, path)
            attempt.data["leftover_temporary_paths"].append(str(path))
            if primary is not None:
                primary.add_note(attempt.data["secondary_errors"][-1])


def extend_graph_files(files, transform, suffix, attempt):
    """Stage all files first, then replace them; never resume or reuse old staging."""
    temporary = []
    primary = None
    try:
        first = files[0]
        attempt.enter("staging", current_file=first)
        for path in files:
            attempt.data["current_file"] = str(path)
            graphs = torch.load(path, weights_only=False)
            if not isinstance(graphs, list):
                raise TypeError(f"Graph artifact {path} is not a list.")
            upgraded = [transform(graph) for graph in graphs]
            temp = path.with_name(path.name + suffix)
            temporary.append((path, temp))  # Register before opening or writing.
            save_staged_graphs(upgraded, temp, attempt)
        attempt.enter("replacing", current_file=first, hazardous=True)
        for path, temp in temporary:
            attempt.data["current_file"] = str(path)
            try:
                temp.replace(path)
            except BaseException as primary:
                # If rename completed just before an interruption, its return/count
                # may be missing. Treat disappearance of staging as uncertain.
                try:
                    if not temp.exists():
                        attempt.uncertain = True
                except OSError as secondary:
                    attempt.uncertain = True
                    attempt.secondary("Checking replacement outcome failed", secondary, temp)
                    primary.add_note(attempt.data["secondary_errors"][-1])
                raise
            attempt.changed = True
            attempt.data["completed_replacements"] += 1
    except BaseException as error:
        primary = error
        raise
    finally:
        cleanup_temporary_files(temporary, attempt, primary)
    return files
