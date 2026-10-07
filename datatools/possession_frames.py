"""Shared sampling contract for inference and pc-xPass (not training carries)."""
from __future__ import annotations

import hashlib
import json
from typing import Callable, TypeVar

STATE_CONTRACT = "possession_states_v1"
T = TypeVar("T")


def add_frame_selection_arguments(parser) -> None:
    parser.add_argument("--scope", choices=("actions", "frames"), default=None,
                        help="Possession endpoints (default: actions), or strided frames.")
    parser.add_argument("--frames", type=int, default=None,
                        help="Frame-ID stride in frames scope, not frames per possession (default: 1).")


def resolve_frame_selection_arguments(parser, args, *, pc_only=False) -> None:
    explicit = args.scope is not None or args.frames is not None
    if pc_only and explicit and not args.pc_xpass:
        parser.error("--scope/--frames require --pc-xpass in this generator.")
    if pc_only and explicit and args.feature_run_id:
        parser.error("--scope/--frames are not supported in legacy feature-sidecar mode.")
    args.scope = args.scope or "actions"
    if args.frames is not None and args.scope != "frames":
        parser.error("--frames is only valid with --scope frames.")
    if args.frames is not None and args.frames < 1:
        parser.error("--frames must be a positive integer.")
    args.frames = args.frames or 1


def select_possession_frames(
    available_frames, start: int, end: int, evaluate: Callable[[int], T | None],
    *, scope: str = "actions", frames: int = 1,
) -> tuple[list[tuple[int, T, str]], dict]:
    """Evaluate endpoints inward, sample the original grid, and never cross boundaries."""
    if scope not in {"actions", "frames"} or isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValueError("Expected scope actions/frames and a positive integer frame stride.")
    if end < start:
        raise ValueError("Possession end precedes start.")
    available = sorted({int(f) for f in available_frames if start <= int(f) <= end})
    results: dict[int, T | None] = {}

    def get(frame):
        if frame not in results:
            results[frame] = evaluate(frame)
        return results[frame]

    first = next((f for f in available if get(f) is not None), None)
    last = next((f for f in reversed(available) if get(f) is not None), None) if first is not None else None
    roles: dict[int, str] = {}
    if first is not None:
        roles[first] = "start_end" if first == last else "start"
        if last != first:
            roles[last] = "end"
        if scope == "frames":
            for frame in available:
                if (frame - start) % frames == 0 and frame not in roles and get(frame) is not None:
                    roles[frame] = "interior"
    report = {
        "original_start_frame": start, "original_end_frame": end,
        "selected_start_frame": first, "selected_end_frame": last,
        "endpoint_substitutions": int(first is not None and first != start) + int(last is not None and last != end),
        "selected_frames": len(roles), "evaluated_frames": len(results),
        "invalid_evaluated_frames": sum(value is None for value in results.values()),
    }
    return [(frame, results[frame], roles[frame]) for frame in sorted(roles)], report


def possession_cache_identity(dataset: str, match_id: str, possession_id, possessor: str, provenance: dict) -> str:
    """Partition by possession/provenance; within a partition action_index is frame ID.

    This preserves the legacy cache row/worker contracts without lossy integer
    encoding, cross-possession collisions, or changing ordinary physical xPass.
    Sampling scope/stride deliberately do not participate in the identity.
    """
    identity = [STATE_CONTRACT, dataset, str(match_id), str(possession_id), str(possessor), provenance]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"{STATE_CONTRACT}_{dataset}_{digest}"
