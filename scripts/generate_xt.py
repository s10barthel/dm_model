from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, UTC
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from datatools.xt import (
    XT_GRID_L,
    XT_GRID_W,
    XT_SOURCE_GRID_FILENAME,
    build_xt_xy_surface,
    annotate_match_xt,
    build_xt_actions,
    fit_xt_surface,
    fit_xt_xy_glm,
    infer_home_team_id,
    rotate_xt_actions,
    save_xt_xy_surface_plot,
    validate_xt_grid_shape,
)
from project_config import EVENT_SYNCED_DIR, XT_DIR, XT_MATCH_DIR, ensure_project_dirs, load_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-split", type=int, default=50, help="Development percentage used to fit the xT surface.")
    parser.add_argument("--match-id", action="append", help="Restrict export generation to one or more match ids.")
    parser.add_argument("--limit", type=int, help="Only process the first N available matches.")
    parser.add_argument("--source-grid-l", type=int, default=None, help="Socceraction source-grid length bins. Default: 12.")
    parser.add_argument("--source-grid-w", type=int, default=None, help="Socceraction source-grid width bins. Default: 8.")
    parser.add_argument("--use-interaction", action="store_true", help="Include normalized_x * normalized_centrality in the xT GLM.")
    parser.add_argument(
        "--use-nonlinear",
        action="append",
        choices=["x", "y"],
        default=None,
        help="Include constrained squared and cubic xT GLM terms for the selected axis. Repeat for both x and y.",
    )
    parser.add_argument(
        "--reuse-source-grid",
        action="store_true",
        help=f"Load {XT_SOURCE_GRID_FILENAME} and skip socceraction source-grid fitting.",
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="Fit xT model artifacts only; do not write xT.csv or per-match xT sidecars.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing xT outputs.")
    args = parser.parse_args()
    if args.reuse_source_grid and (args.source_grid_l is not None or args.source_grid_w is not None):
        parser.error("--source-grid-l/--source-grid-w cannot be combined with --reuse-source-grid; reuse infers dimensions from the CSV.")
    args.source_grid_l = XT_GRID_L if args.source_grid_l is None else args.source_grid_l
    args.source_grid_w = XT_GRID_W if args.source_grid_w is None else args.source_grid_w
    if args.source_grid_l < 1 or args.source_grid_w < 1:
        parser.error("--source-grid-l and --source-grid-w must be positive integers.")
    return args


def load_events(match_id: str) -> pd.DataFrame:
    return pd.read_csv(EVENT_SYNCED_DIR / f"{match_id}.csv", parse_dates=["utc_timestamp"])


def summarize_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def grid_columns(grid_l: int) -> list[str]:
    return [f"X{i}" for i in range(int(grid_l))]


def load_source_grid(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Reusable xT source grid not found at {path}.")
    source_frame = pd.read_csv(path)
    if source_frame.empty:
        raise ValueError(f"Reusable xT source grid at {path} is empty.")
    grid = validate_xt_grid_shape(source_frame.to_numpy(dtype=float), context=f"Reusable xT source grid at {path}")
    return pd.DataFrame(grid, columns=grid_columns(grid.shape[1]))


def resolve_match_ids(requested_match_ids: list[str] | None, limit: int | None) -> list[str]:
    match_ids = sorted(path.stem for path in EVENT_SYNCED_DIR.glob("*.csv"))
    if requested_match_ids:
        requested = set(requested_match_ids)
        match_ids = [match_id for match_id in match_ids if match_id in requested]
    if limit is not None:
        match_ids = match_ids[:limit]
    if not match_ids:
        raise ValueError("No synced event files were selected for xT generation.")
    return match_ids


def fit_actions_for_train_split(train_ids: list[str]) -> tuple[pd.DataFrame, list[str], list[dict[str, str]]]:
    rotated_train_actions = []
    fit_match_ids: list[str] = []
    skipped_matches: list[dict[str, str]] = []

    for match_id in train_ids:
        event_path = EVENT_SYNCED_DIR / f"{match_id}.csv"
        if not event_path.exists():
            skipped_matches.append({"match_id": match_id, "error": "missing_synced_events"})
            continue

        try:
            events = load_events(match_id)
            xt_actions = build_xt_actions(events)
            if xt_actions.empty:
                skipped_matches.append({"match_id": match_id, "error": "no_eligible_actions"})
                continue
            rotated_train_actions.append(rotate_xt_actions(xt_actions, infer_home_team_id(events)))
            fit_match_ids.append(match_id)
        except Exception as exc:
            skipped_matches.append({"match_id": match_id, "error": summarize_exception(exc)})
            continue

    if not rotated_train_actions:
        raise ValueError("No eligible pass/cross/shot actions were found in the training split for xT fitting.")

    return pd.concat(rotated_train_actions, ignore_index=True), fit_match_ids, skipped_matches


def collect_export_matches(match_ids: list[str]) -> tuple[list[pd.DataFrame], list[str], list[dict[str, str]]]:
    all_events: list[pd.DataFrame] = []
    export_match_ids: list[str] = []
    skipped_matches: list[dict[str, str]] = []

    for match_id in match_ids:
        try:
            events = load_events(match_id)
            xt_actions = build_xt_actions(events)
            if not xt_actions.empty:
                infer_home_team_id(events)
            all_events.append(events)
            export_match_ids.append(match_id)
        except Exception as exc:
            skipped_matches.append({"match_id": match_id, "error": summarize_exception(exc)})

    if not all_events:
        raise ValueError("No usable synced event files remained for xT export generation.")

    return all_events, export_match_ids, skipped_matches


def save_export_outputs(
    all_events: list[pd.DataFrame],
    xt_fit,
    output_dir: Path,
    xt_match_dir: Path,
) -> tuple[list[str], list[dict[str, str]], list[pd.DataFrame]]:
    processed_match_ids: list[str] = []
    skipped_matches: list[dict[str, str]] = []
    exported_actions: list[pd.DataFrame] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    xt_match_dir.mkdir(parents=True, exist_ok=True)

    for events in all_events:
        try:
            annotated_events, exported_xt = annotate_match_xt(events, xt_fit)
            match_id = str(
                annotated_events["stats_perform_match_id"].iloc[0]
                if "stats_perform_match_id" in annotated_events.columns
                else annotated_events["game_id"].iloc[0]
            )
            sidecar = annotated_events[["action_id", "xG", "xT", "scores_xT", "concedes_xT"]].copy()
            sidecar.to_csv(xt_match_dir / f"{match_id}.csv", index=False)
            exported_actions.append(exported_xt)
            processed_match_ids.append(match_id)
        except Exception as exc:
            match_id = "<unknown>"
            if "stats_perform_match_id" in events.columns and not events.empty:
                match_id = str(events["stats_perform_match_id"].iloc[0])
            elif "game_id" in events.columns and not events.empty:
                match_id = str(events["game_id"].iloc[0])
            skipped_matches.append({"match_id": match_id, "error": summarize_exception(exc)})

    if not processed_match_ids:
        raise ValueError("No usable synced event files remained for xT export writing.")

    if exported_actions:
        pd.concat(exported_actions, ignore_index=True).to_csv(output_dir / "xT.csv", index=False)
    else:
        pd.DataFrame(columns=["action_id", "xG", "xT", "scores_xT", "concedes_xT"]).to_csv(output_dir / "xT.csv", index=False)

    return processed_match_ids, skipped_matches, exported_actions


def save_model_outputs(xt_fit, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(xt_fit.projection_grid, columns=[f"X{i}" for i in range(XT_GRID_L)]).to_csv(
        output_dir / "xT_grid.csv",
        index=False,
    )
    pd.DataFrame(xt_fit.source_grid, columns=grid_columns(xt_fit.source_grid.shape[1])).to_csv(
        output_dir / XT_SOURCE_GRID_FILENAME,
        index=False,
    )
    surface = build_xt_xy_surface(xt_fit.model)
    surface.to_csv(output_dir / "xT_xy_surface.csv", index=False)
    xt_fit.fit_sample.to_csv(output_dir / "xT_glm_fit_sample.csv", index=False)
    save_xt_xy_surface_plot(surface, output_dir / "xT_xy_surface_3d.png")


def ignored_export_filters(args: argparse.Namespace) -> dict[str, object]:
    ignored: dict[str, object] = {}
    if args.match_id:
        ignored["match_id"] = list(args.match_id)
    if args.limit is not None:
        ignored["limit"] = int(args.limit)
    return ignored


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    manifest = load_split_manifest(args.train_split)

    output_csv = XT_DIR / "xT.csv"
    output_grid = XT_DIR / "xT_grid.csv"
    output_source_grid = XT_DIR / XT_SOURCE_GRID_FILENAME
    output_surface = XT_DIR / "xT_xy_surface.csv"
    output_fit_sample = XT_DIR / "xT_glm_fit_sample.csv"
    output_surface_plot = XT_DIR / "xT_xy_surface_3d.png"
    output_metadata = XT_DIR / "fit_metadata.json"
    model_outputs = [
        output_grid,
        output_source_grid,
        output_surface,
        output_fit_sample,
        output_surface_plot,
        output_metadata,
    ]
    expected_outputs = model_outputs if args.fit_only else [output_csv, *model_outputs]
    if not args.overwrite and all(path.exists() for path in expected_outputs):
        existing_metadata = json.loads(output_metadata.read_text(encoding="utf-8"))
        if existing_metadata.get("split_manifest_id") != manifest.get("manifest_id"):
            raise ValueError(
                "Existing xT artifacts were fitted with a different or legacy split. "
                "Use --overwrite to rebuild them for the requested --train-split."
            )
        print(f"xT outputs already exist in {XT_DIR}. Use --overwrite to rebuild them.")
        return
    train_ids = [match_id for match_id in manifest["train"] if (EVENT_SYNCED_DIR / f"{match_id}.csv").exists()]
    all_events: list[pd.DataFrame] = []
    skipped_export_matches: list[dict[str, str]] = []
    if not args.fit_only:
        match_ids = resolve_match_ids(args.match_id, args.limit)
        all_events, _, skipped_export_matches = collect_export_matches(match_ids)

    rotated_train_actions, fit_match_ids, skipped_fit_matches = fit_actions_for_train_split(train_ids)
    if args.reuse_source_grid:
        source_grid_frame = load_source_grid(output_source_grid)
        source_xt_grid = source_grid_frame.to_numpy(dtype=float)
        source_grid_origin = "reused_source_grid"
    else:
        source_xt_grid = fit_xt_surface(
            rotated_train_actions,
            grid_l=args.source_grid_l,
            grid_w=args.source_grid_w,
        )
        source_grid_origin = "socceraction_fit"
    xt_fit = fit_xt_xy_glm(
        rotated_train_actions,
        source_xt_grid,
        use_interaction=args.use_interaction,
        nonlinear_axes=args.use_nonlinear,
    )
    source_grid_w, source_grid_l = xt_fit.source_grid.shape

    save_model_outputs(xt_fit, XT_DIR)
    processed_export_ids: list[str] = []
    skipped_export_write_matches: list[dict[str, str]] = []
    if not args.fit_only:
        processed_export_ids, skipped_export_write_matches, _ = save_export_outputs(all_events, xt_fit, XT_DIR, XT_MATCH_DIR)

    fit_metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "xt_model_type": "xy_logit_glm",
        "fit_only": bool(args.fit_only),
        "train_split_percent": args.train_split,
        "split_manifest_id": manifest["manifest_id"],
        "split_manifest": manifest["metadata"],
        "match_sidecars_written": not bool(args.fit_only),
        "xT_csv_written": not bool(args.fit_only),
        "ignored_export_filters": ignored_export_filters(args) if args.fit_only else {},
        "formula": "logit(xT) = intercept + " + " + ".join(xt_fit.model.terms),
        "terms": list(xt_fit.model.terms),
        "feature_formulas": {
            "normalized_x": "start_x / 105",
            "normalized_centrality": "sin(pi * start_y / 68)",
        },
        "use_interaction": bool(args.use_interaction),
        "use_nonlinear": list(args.use_nonlinear or []),
        "nonlinear_basis": "quadratic_cubic_polynomial",
        "coefficients": xt_fit.model.to_metadata(),
        "optimizer": xt_fit.diagnostics.get("optimizer", {}),
        "monotonicity_check": xt_fit.diagnostics.get("monotonicity_check", {}),
        "weighting": {
            "method": "equal_aggregate_weight_per_populated_2d_x_centrality_bin",
            "x_bins": 12,
            "centrality_bins": 8,
        },
        "grid_l": XT_GRID_L,
        "grid_w": XT_GRID_W,
        "legacy_projection_grid_l": XT_GRID_L,
        "legacy_projection_grid_w": XT_GRID_W,
        "xT_grid_semantics": "xy_logit_glm_projection_to_legacy_12x8_cell_centers",
        "source_grid_l": int(source_grid_l),
        "source_grid_w": int(source_grid_w),
        "source_grid_path": str((XT_DIR / XT_SOURCE_GRID_FILENAME).relative_to(ROOT)),
        "source_grid_origin": source_grid_origin,
        "reuse_source_grid": bool(args.reuse_source_grid),
        "shot_xT_policy": "max_glm_xT_xG",
        "fit_match_ids": fit_match_ids,
        "export_match_ids": processed_export_ids,
        "eligible_action_types": ["pass", "cross", "shot"],
        "symmetry_pairs": [[1, 8], [2, 7], [3, 6], [4, 5]],
        "source_grid_fit_samples": int(len(rotated_train_actions)),
        "fit_samples": int(len(xt_fit.fit_sample)),
        "output_files": {
            "xT": str((XT_DIR / "xT.csv").relative_to(ROOT)),
            "xT_grid": str((XT_DIR / "xT_grid.csv").relative_to(ROOT)),
            "xT_source_grid": str((XT_DIR / XT_SOURCE_GRID_FILENAME).relative_to(ROOT)),
            "xT_xy_surface": str((XT_DIR / "xT_xy_surface.csv").relative_to(ROOT)),
            "xT_glm_fit_sample": str((XT_DIR / "xT_glm_fit_sample.csv").relative_to(ROOT)),
            "xT_xy_surface_3d": str((XT_DIR / "xT_xy_surface_3d.png").relative_to(ROOT)),
        },
        "diagnostics": xt_fit.diagnostics,
        "skipped_fit_matches": skipped_fit_matches,
        "skipped_export_matches": skipped_export_matches + skipped_export_write_matches,
    }
    (XT_DIR / "fit_metadata.json").write_text(json.dumps(fit_metadata, indent=2), encoding="utf-8")
    if skipped_fit_matches:
        print(f"Skipped {len(skipped_fit_matches)} training matches while fitting xT.")
    total_skipped_export = len(skipped_export_matches) + len(skipped_export_write_matches)
    if total_skipped_export:
        print(f"Skipped {total_skipped_export} export matches while generating xT outputs.")
    print(f"Saved xT outputs to {XT_DIR}")


if __name__ == "__main__":
    main()
