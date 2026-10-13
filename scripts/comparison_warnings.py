"""Explicit, recorded acceptance of limitations in a comparison."""
from __future__ import annotations

import sys


class ComparisonCancelled(RuntimeError):
    pass


def add_warning_argument(parser):
    parser.add_argument("--accept-comparison-warnings", action="store_true",
                        help="Explicitly accept comparison limitations without an interactive prompt.")


class ComparisonWarnings:
    def __init__(self, accept=False):
        self.accept = accept
        self.messages = []
        self.acceptances = []
        self.accepted = set()
        self.keys = {}
        self.cancelled = False

    def add(self, message, *, key=None):
        if message not in self.messages:
            self.messages.append(message)
        self.keys[message] = key or message

    def confirm(self):
        pending = [item for item in self.messages if self.keys[item] not in self.accepted]
        if not pending:
            return
        print("Comparison warnings:", file=sys.stderr, flush=True)
        for message in pending:
            print(f"- {message}", file=sys.stderr, flush=True)
        method = "command_line" if self.accept else "interactive"
        if not self.accept:
            if not sys.stdin.isatty():
                self.cancelled = True
                raise ComparisonCancelled("Comparison warnings require explicit acceptance. Run interactively or pass --accept-comparison-warnings.")
            while True:
                try:
                    answer = input("Proceed despite these warnings? [yes/no] ").strip().lower()
                except EOFError:
                    self.cancelled = True
                    raise ComparisonCancelled("Comparison cancelled: no explicit answer received.") from None
                if answer in {"y", "yes"}:
                    break
                if answer in {"n", "no"}:
                    self.cancelled = True
                    raise ComparisonCancelled("Comparison cancelled by user.")
                print("Please answer yes or no; an empty answer does not accept the warnings.", flush=True)
        self.accepted.update(self.keys[item] for item in pending)
        self.acceptances.append({"method": method, "warnings": pending})

    def metadata(self):
        return {"warnings": self.messages, "warning_acceptances": self.acceptances,
                "comparison_cancelled": self.cancelled,
                "warning_keys": self.keys}


def metadata_differences(records, keys, *, prefix="", labels=None):
    """Describe known differences without treating training choices as invalid."""
    messages = []
    labels = labels or [str(item.get("model_id", index)) for index, item in enumerate(records)]
    for key in keys:
        values = [item.get(key, (item.get("metadata") or {}).get(key, (item.get("args") or {}).get(key))) for item in records]
        if key == "pass_height_definition":
            values = [{k: v for k, v in value.items() if k != "resolution_source"}
                      if isinstance(value, dict) else value for value in values]
        known = [value for value in values if value is not None]
        if known and any(value != known[0] for value in known[1:]):
            details = "; ".join(f"{label}={value!r}" for label, value in zip(labels, values))
            messages.append(f"{prefix}{key} differs: {details}. Results use each variant's own definition unless a common override is specified.")
    return messages
