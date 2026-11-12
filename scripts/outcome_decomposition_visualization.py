"""Optional occurrence x severity labels in existing outcome pitch plots."""
import math

from datatools import config

COMPONENTS = {f"outcome_{task}_{branch}" for task in ("scoring", "conceding")
              for branch in ("success", "failure")}


def validate_options(args):
    if not getattr(args, "decomposition", False):
        return
    if getattr(args, "coach_ratings", False) or getattr(args, "selections", False):
        raise ValueError("--decomposition cannot be combined with --coach-ratings or --selections.")
    if getattr(args, "only_tracking", False):
        raise ValueError("--decomposition requires individual outcome components.")
    if getattr(args, "mode", "standard") in {"loc", "freeze"}:
        raise ValueError("--decomposition currently requires standard frames with saved or live head predictions.")


def validate_components(args, names):
    validate_options(args)
    if getattr(args, "decomposition", False) and (not names or not set(names).issubset(COMPONENTS)):
        raise ValueError("--decomposition supports only the four individual outcome scoring/conceding success/failure components; select them explicitly.")


def validate_models(args, names, models):
    from models.outcome_decomposition import is_decomposition
    validate_components(args, names)
    if not getattr(args, "decomposition", False):
        return
    for name in names:
        task = name.rsplit("_", 1)[0]
        model = models.get(task)
        if model is None or not is_decomposition(model.args):
            raise ValueError(f"--decomposition requires a decomposition checkpoint for {task}.")
        model.visualize_outcome_decomposition = True


def annotate(ax, snapshot, attacking_prefix, probs, *, rotate_pitch=False):
    if probs is None or not probs.attrs.get("show_outcome_decomposition"):
        return
    mapping = probs.attrs.get("outcome_decomposition", {})
    # Hawkeye uses frame IDs as action indices. Sportec series retain the model
    # action index even when a comparison table subsequently receives a frame index.
    index = probs.attrs.get("decomposition_action_index", probs.name)
    pairs = mapping.get(index, mapping.get(str(index), {}))
    if not pairs:
        raise ValueError(f"Missing decomposition predictions for displayed frame/action {index}.")
    for candidate in probs.index:
        if not isinstance(candidate, str) or not candidate.startswith(attacking_prefix):
            continue
        if candidate not in pairs:
            continue
        q, m = pairs[candidate]
        xcol, ycol = candidate + "_x", candidate + "_y"
        if xcol not in snapshot or ycol not in snapshot:
            continue
        x, y = snapshot[xcol].iloc[-1], snapshot[ycol].iloc[-1]
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if rotate_pitch:
            x, y = config.FIELD_SIZE[0] - x, config.FIELD_SIZE[1] - y
        text = f"{q:.3f} x {m:.3f}" if math.isfinite(q) and math.isfinite(m) else "- x -"
        ax.annotate(text, xy=(x, y), xytext=(0, -3.5), textcoords="offset points",
                    ha="center", va="top", color="black", fontsize=10, zorder=7)


def attach_saved(tables, rows, scene_id, components=None):
    """Attach saved heads using semantic scene, frame, branch and candidate keys."""
    selected = COMPONENTS & tables.keys()
    if components is not None:
        selected &= set(components)
    for component in selected:
        table = tables[component]
        if table.empty:
            continue
        task, branch = component.rsplit("_", 1)
        mapping = {}
        for row in rows:
            if str(row["scene_id"]) != str(scene_id) or row["task"] != task or int(row["execution_branch"]) != int(branch == "success"):
                continue
            frame, candidate = int(row["frame_id"]), row["candidate_id"]
            if frame not in table.index or candidate not in table.columns:
                continue
            pairs = mapping.setdefault(frame, {})
            if candidate in pairs:
                raise ValueError("Ambiguous saved decomposition predictions; model/mask keys must identify one source.")
            actual = table.at[frame, candidate]
            prediction = row["prediction"]
            if math.isfinite(actual) and not math.isclose(actual, prediction, abs_tol=1e-6, rel_tol=1e-5):
                raise ValueError("Saved decomposition predictions disagree with outcome component values.")
            q, m = float(row["occurrence_probability"]), float(row["conditional_severity"])
            if not row.get("forced", False) and math.isfinite(prediction) and not math.isclose(q*m, prediction, abs_tol=1e-6, rel_tol=1e-5):
                raise ValueError("Saved occurrence x severity does not match its expected value.")
            pairs[candidate] = [q, m]
        for frame in table.index:
            for candidate, value in table.loc[frame].items():
                if math.isfinite(value) and candidate not in mapping.get(frame, {}):
                    raise ValueError(f"Missing saved decomposition for {component}, frame {frame}, candidate {candidate}.")
        table.attrs.update(outcome_decomposition=mapping, show_outcome_decomposition=True)
