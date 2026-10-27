"""Input context and serializable diagnostics for offline feature workers."""
from __future__ import annotations

import json
from pathlib import Path
import traceback


def _annotate_input(error, path, input_format):
    if not getattr(error, "input_path", None):
        error.input_path = str(Path(path).absolute())
        error.input_format = input_format
        error.add_note(f"Input file ({input_format}): {error.input_path}")


def read_json(path, *, encoding="utf-8-sig"):
    try:
        return json.loads(Path(path).read_text(encoding=encoding))
    except Exception as error:
        _annotate_input(error, path, "json")
        raise


def read_parquet(path, *args, **kwargs):
    import pandas as pd

    try:
        return pd.read_parquet(path, *args, **kwargs)
    except Exception as error:
        _annotate_input(error, path, "parquet")
        raise


class WorkerFailure(Exception):
    """Preserve diagnostics across processes, including JSONDecodeError context."""

    def __init__(self, details):
        self.details = details
        super().__init__(details)

    def __str__(self):
        return self.details["error"]


def failure_report(error, match_id):
    if isinstance(error, WorkerFailure):
        return dict(error.details, match_id=str(match_id))
    report = dict(match_id=str(match_id), error=f"{type(error).__name__}: {error}",
                  exception_type=type(error).__name__,
                  traceback="".join(traceback.format_exception(type(error), error, error.__traceback__)))
    current, seen = error, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        path = getattr(current, "input_path", None) or getattr(current, "filename", None)
        if path and "input_path" not in report:
            report["input_path"] = str(Path(path).absolute())
            report["input_format"] = getattr(current, "input_format", Path(path).suffix.lstrip("."))
        if isinstance(current, json.JSONDecodeError) and "json_error" not in report:
            report["json_error"] = dict(line=current.lineno, column=current.colno, position=current.pos)
        current = current.__cause__ or current.__context__
    return report


def failure_summary(report):
    summary = " ".join(report["error"].splitlines())
    path = report.get("input_path")
    if path and path not in summary:
        summary += f" [input: {path}]"
    return summary
