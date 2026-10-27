"""Required pc-xPass inference inputs and failed-attempt reporting."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
from pathlib import Path
import sys


class PCXPassBlendingError(RuntimeError):
    """A requested pc-xPass blend cannot be evaluated; callers must not skip it."""

    def __init__(self, message, *, stage, context=None):
        self.stage = stage
        self.context = {k: str(v) for k, v in (context or {}).items() if v is not None}
        details = ", ".join(f"{k}={v}" for k, v in self.context.items())
        super().__init__(f"Required pc-xPass blending failed ({stage}): {message}"
                         + (f" [{details}]" if details else ""))

    def as_dict(self):
        return dict(type=type(self).__name__, message=str(self), stage=self.stage, context=self.context)


def pc_blending_required(model_or_args):
    from physical_pass_model import inference_uses_physical_xpass, pc_xpass_enabled
    args = getattr(model_or_args, "args", model_or_args)
    return args is not None and inference_uses_physical_xpass(args) and pc_xpass_enabled(args)


def blending_context(model, match=None, **context):
    args = getattr(model, "args", {})
    return {**dict(cache_id=args.get("pc_xpass_id"), cache_dir=args.get("physical_cache_dir"),
                metric=args.get("x_pass_version", args.get("xpass_version")),
                match_id=getattr(match, "match_id", None),
                partition=getattr(match, "event_index", None)), **context}


@contextmanager
def blending_guard(model, stage, match=None, **context):
    try:
        yield
    except PCXPassBlendingError:
        raise
    except Exception as exc:
        if pc_blending_required(model):
            raise PCXPassBlendingError(str(exc), stage=stage,
                                      context=blending_context(model, match, **context)) from exc
        raise


def blending_stage(stage):
    """Decorate a (match, model, ...) operation that supplies required blend inputs."""
    def decorate(function):
        from inspect import signature
        parameters = signature(function)
        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = parameters.bind(*args, **kwargs).arguments
            match, model = bound.get("match"), bound["model"]
            with blending_guard(model, stage, match):
                return function(*args, **kwargs)
        return wrapped
    return decorate


@contextmanager
def cli_blending_guard(args, *, required=False):
    """Protect cache selection before the pass-success model is configured."""
    from types import SimpleNamespace
    options = {**vars(args), "task": "pass_success",
               "pc_xpass": required or bool(getattr(args, "pc_xpass", False)),
               "inference_use_physical_xpass": required or bool(getattr(args, "use_physical_xpass", False)),
               "physical_cache_dir": getattr(args, "physical_cache_dir", None)
                                     or getattr(args, "pc_xpass_cache_dir", None)}
    with blending_guard(SimpleNamespace(args=options), "startup"):
        yield


def pc_xpass_setup(function):
    @wraps(function)
    def wrapped(args, *positional, **kwargs):
        with cli_blending_guard(args):
            return function(args, *positional, **kwargs)
    return wrapped


def validate_pc_xpass_startup(model, *, memo=None):
    if not pc_blending_required(model):
        return
    state = _attempt.get()
    if state is not None and state["root"] is not None and not state["active"]:
        state["active"] = True
        _write_attempt(state, status="running")
    from physical_pass_model import (physical_xpass_inference_lookup_config,
                                     validate_x_pass_version_available, XPASS_WEIGHT_VERSIONS)
    import math
    with blending_guard(model, "startup"):
        args = model.args
        if args.get("physical_runtime_cache_disabled", False):
            raise ValueError("pc-xPass blending requires an enabled cache")
        directory = args.get("physical_cache_dir")
        if not directory:
            from inference import _physical_xpass_cache_dir_for_inference
            directory = _physical_xpass_cache_dir_for_inference(None, model)
        path = Path(directory) / "metadata.json"
        stat = path.stat()
        signature = ("required_pc_startup", str(path.resolve()), stat.st_mtime_ns, stat.st_size,
                     json.dumps(args, sort_keys=True, default=str))
        if memo is None:
            memo = getattr(model, "_required_pc_startup_memo", None)
            if memo is None:
                memo = model._required_pc_startup_memo = {}
        if signature in memo:
            return
        lookup = physical_xpass_inference_lookup_config(args, cache_dir=directory)
        if lookup["weight_version"] not in XPASS_WEIGHT_VERSIONS:
            raise ValueError(f"Unsupported weighting version {lookup['weight_version']}")
        positive = ("v4_power", "v4_zero") if lookup["weight_version"] == "v4" else (
            ("v5_intent_threshold",) if lookup["weight_version"] == "v5" else ())
        for key in positive:
            if not math.isfinite(float(lookup[key])) or float(lookup[key]) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        validate_x_pass_version_available(directory, source=lookup["source"],
                                          x_pass_version=lookup["x_pass_version"],
                                          metric=lookup["metric"], memo=memo)
        memo[signature] = True


_attempt = ContextVar("pc_xpass_inference_attempt", default=None)


def start_inference_attempt(root, args, *, required=False):
    """Register an allocated output root; leave non-pc inference behavior unchanged."""
    state = _attempt.get()
    if state is None:
        return
    active = required or bool(getattr(args, "pc_xpass", False) and getattr(args, "use_physical_xpass", False))
    state.update(root=Path(root), active=active)
    if active:
        _write_attempt(state, status="running")


def _write_attempt(state, **updates):
    from project_config import write_run_metadata
    root = state["root"]
    path = root / "metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {
        "run_id": root.name, "command": " ".join(sys.argv)}
    if updates.get("status") == "running":
        payload.pop("failure", None)
    payload.update(pc_xpass_blending_policy="required", **updates)
    write_run_metadata(root, payload)


def inference_attempt(function):
    """Keep failures visible even when normal runner metadata is built at the end."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _attempt.set({"root": None, "active": False})
        try:
            result = function(*args, **kwargs)
            state = _attempt.get()
            if state["root"] is not None and state["active"]:
                path = state["root"] / "metadata.json"
                status = json.loads(path.read_text(encoding="utf-8-sig")).get("status", "running")
                _write_attempt(state, status="completed" if status == "running" else status)
            return result
        except Exception as exc:
            state = _attempt.get()
            if state["root"] is not None and (state["active"] or isinstance(exc, PCXPassBlendingError)):
                failure = exc.as_dict() if isinstance(exc, PCXPassBlendingError) else {
                    "type": type(exc).__name__, "message": str(exc), "stage": "startup", "context": {}}
                try:
                    outputs = [str(path.relative_to(state["root"])) for path in state["root"].rglob("*")
                               if path.is_file() and path.suffix in {".parquet", ".csv", ".png", ".mp4"}]
                    _write_attempt(state, status="failed", failure=failure, available_outputs=outputs)
                except Exception as reporting_error:
                    exc.add_note(f"Failed to record inference failure: {reporting_error}")
            raise
        finally:
            _attempt.reset(token)
    return wrapped
