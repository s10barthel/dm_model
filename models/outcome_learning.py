"""Versioned scalar and categorical outcome learning in original target units."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

TASKS = {"outcome_scoring", "outcome_conceding"}
DEFAULT_ATOMS = 65


def settings(args):
    return args if isinstance(args, dict) else vars(args)


def is_distributional(args):
    return settings(args).get("learning_mode", "scalar") == "distributional"


def is_signed(args):
    return (settings(args).get("outcome_learning_contract") or {}).get("domain") == "signed_gain"


def signed_contract(args):
    distributional = is_distributional(args)
    return dict(version=1, domain="signed_gain", target_bounds=[-1., 1.],
                output="categorical_mean" if distributional else "identity",
                loss="categorical_ce" if distributional else "weighted_mse",
                checkpoint_monitor="mse_loss")


def configure_outcome_learning(args, metadata, source=None):
    """Resolve from feature metadata, never from the observed label range."""
    from datatools.xt_targets import target_settings, training_target_definition, validate_training_targets
    args = settings(args)
    target = target_settings(metadata if args.get("use_xt") else None)
    signed = bool(args.get("use_xt") and target["representation"] == "gain"
                  and not target["clip_negative_gains"])
    if args.get("use_xt"):
        validate_training_targets(metadata, args.get("return_type"), args=args)
        args["xt_target_definition"] = training_target_definition(metadata)
    contract = signed_contract(args) if signed else None
    if source is not None:
        source = settings(source)
        saved = source.get("outcome_learning_contract", (source.get("args") or {}).get("outcome_learning_contract"))
        if saved != contract:
            raise ValueError("Resume/refit cannot change the saved outcome learning contract.")
        saved_target = source.get("xt_target_definition") or (source.get("args") or {}).get("xt_target_definition")
        if args.get("use_xt") and saved_target is not None and saved_target != args["xt_target_definition"]:
            raise ValueError("Resume/refit cannot change the saved xT surface or target definition.")
    if contract is not None:
        args["outcome_learning_contract"] = contract
        args["checkpoint_monitor"] = "mse_loss"
    elif args.get("outcome_learning_contract") is not None:
        raise ValueError("Outcome learning contract conflicts with feature targets.")
    normalize_learning_settings(args)
    return contract


def mean_mse(logits, targets, args, weight=None):
    validate_targets(targets, -1.)
    losses = (expected_values(logits, args) - targets).square()
    return (losses if weight is None else losses * weight).mean()


def goal_ranking_metrics(target, prediction):
    """Association with binary goals; signed gains are not probabilities."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    target, prediction = np.asarray(target), np.asarray(prediction)
    if not np.isfinite(target).all() or not np.isin(target, [0, 1]).all() or not np.isfinite(prediction).all():
        raise ValueError("Goal ranking requires finite scores and binary diagnostic targets.")
    return dict(roc_auc=float(roc_auc_score(target, prediction)) if np.unique(target).size == 2 else np.nan,
                pr_auc=float(average_precision_score(target, prediction)) if np.any(target > 0) else np.nan)


def add_learning_arguments(parser):
    parser.add_argument("--learning-mode", choices=("scalar", "distributional"), default=None,
                        help="Outcome learning mode; fresh training defaults to scalar.")
    parser.add_argument("--distribution-atoms", type=int, default=None,
                        help="Maximum categorical atom count, including domain endpoints and zero (default: 65).")


def normalize_learning_settings(args, source=None):
    args = settings(args)
    source = settings(source) if source is not None else None
    mode = args.get("learning_mode") or (source or {}).get("learning_mode", "scalar")
    if mode not in ("scalar", "distributional"):
        raise ValueError(f"Unknown learning mode: {mode}")
    count = args.get("distribution_atoms")
    if count is None:
        count = (source or {}).get("distribution_atoms") or DEFAULT_ATOMS
    if type(count) is not int or count < 2:
        raise ValueError("--distribution-atoms must be an integer >= 2.")
    if source is not None:
        saved_contract = source.get("outcome_learning_contract", (source.get("args") or {}).get("outcome_learning_contract"))
        if saved_contract is not None:
            if args.get("outcome_learning_contract") not in (None, saved_contract):
                raise ValueError("Resume/refit outcome learning contract conflicts with its source.")
            args["outcome_learning_contract"] = saved_contract
        if mode != source.get("learning_mode", "scalar"):
            raise ValueError("Resume/refit cannot change the saved learning mode.")
        saved = source.get("outcome_distribution_definition")
        if mode == "distributional":
            validate_definition(saved)
            if count != saved["requested_atoms"]:
                raise ValueError("Resume/refit cannot change saved distribution atoms.")
            if args.get("outcome_distribution_definition") not in (None, saved):
                raise ValueError("Resume/refit distribution definition conflicts with its source.")
            args["outcome_distribution_definition"] = saved
    args["learning_mode"] = mode
    if mode == "distributional":
        if args.get("task") not in TASKS:
            raise ValueError("Distributional learning supports outcome_scoring and outcome_conceding only.")
        if args.get("model", "gat") in {"xgboost", "catboost"}:
            raise ValueError("Distributional outcome learning requires a GNN.")
        if args.get("include_out", False):
            raise ValueError("Distributional outcome learning does not support include_out.")
        if args.get("weight_bce", False):
            raise ValueError("Distributional learning does not support outcome-dependent class balancing.")
        args["distribution_atoms"] = count
        args["checkpoint_monitor"] = "mse_loss" if is_signed(args) else "mean_soft_bce"
        if is_signed(args) and count < 3:
            raise ValueError("Signed distributions require at least 3 atoms (-1, 0, 1).")
        if args.get("outcome_distribution_definition") is not None:
            validate_definition(args["outcome_distribution_definition"])
    if is_signed(args):
        if args["outcome_learning_contract"] != signed_contract(args):
            raise ValueError("Invalid outcome learning contract.")
        args["checkpoint_monitor"] = "mse_loss"
        definition = args.get("outcome_distribution_definition")
        if mode == "distributional" and definition is not None and definition["version"] != 2:
            raise ValueError("Signed outcome learning requires a signed distribution definition.")
    elif mode == "distributional" and (args.get("outcome_distribution_definition") or {}).get("version") == 2:
        raise ValueError("Signed distribution requires an explicit outcome learning contract.")
    return args


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_definition(definition):
    if not isinstance(definition, dict):
        raise ValueError("Distributional checkpoint is missing its distribution definition.")
    payload = {k: v for k, v in definition.items() if k != "definition_sha256"}
    if (payload.get("version") not in (1, 2) or payload.get("encoding") != "linear_two_hot_original_units"
            or payload.get("representation") != "categorical"
            or definition.get("definition_sha256") != _digest(payload)):
        raise ValueError("Invalid or inconsistent outcome distribution definition.")
    atoms = np.asarray(payload.get("atoms", []), dtype=float)
    lower = -1 if payload["version"] == 2 else 0
    if (atoms.ndim != 1 or len(atoms) < 2 or not np.isfinite(atoms).all()
            or atoms[0] != lower or atoms[-1] != 1 or 0 not in atoms or not (np.diff(atoms) > 0).all()
            or payload.get("effective_atoms") != len(atoms)
            or type(payload.get("requested_atoms")) is not int
            or len(atoms) > payload["requested_atoms"]):
        raise ValueError(f"Distribution atoms must be strictly increasing from {lower} to 1 and include zero.")
    return definition


def validate_learning_metadata(args, metadata):
    contract = args.get("outcome_learning_contract")
    if contract is not None and (not args.get("use_xt") or contract != signed_contract(args)):
        raise ValueError("Invalid outcome learning contract or target family.")
    if is_signed(args) and args.get("checkpoint_monitor") not in (None, "mse_loss"):
        raise ValueError("Signed outcome checkpoint must monitor mse_loss.")
    normalize_learning_settings(args)
    if metadata.get("outcome_learning_contract", contract) != contract:
        raise ValueError("Checkpoint args and metadata disagree on outcome learning contract.")
    if is_signed(args) and metadata.get("checkpoint_monitor", "mse_loss") != "mse_loss":
        raise ValueError("Checkpoint metadata has an incompatible outcome monitor.")
    definition = args.get("xt_target_definition") or metadata.get("xt_target_definition") or {}
    target = definition.get("xt_targets") or {}
    signed = target.get("representation") == "gain" and target.get("clip_negative_gains") is False
    if signed != is_signed(args):
        raise ValueError("Checkpoint target definition and outcome learning contract disagree.")
    if signed:
        from datatools.xt_targets import validate_training_targets
        validate_training_targets(definition, args.get("return_type"), args=args)
    if "learning_mode" in metadata and metadata["learning_mode"] != args["learning_mode"]:
        raise ValueError("Checkpoint args and metadata disagree on learning mode.")
    if is_distributional(args):
        definition = validate_definition(args.get("outcome_distribution_definition"))
        if metadata.get("outcome_distribution_definition", definition) != definition:
            raise ValueError("Checkpoint args and metadata disagree on distribution definition.")


def target_column(args):
    from datatools.config import LABEL_INDEX
    args = settings(args)
    family = next((family for family in ("progression", "goal_distance", "xt", "epv", "xg")
                   if args.get("use_" + family)), None)
    if args["task"] == "outcome_scoring":
        name = "scores" + ("_" + family if family else "")
    else:
        name = "concedes" + ("_" + ("goal_distance" if family == "progression" else family) if family else "")
    return LABEL_INDEX[name]


def validate_targets(values, lower=0.):
    if not bool((torch.isfinite(values) & (values >= lower) & (values <= 1)).all()):
        raise ValueError(f"Outcome targets must be finite and in [{lower:g}, 1]; values are not clipped.")


def fit_definition(label_batches, args, provenance):
    """Read retained training labels only, without consuming RNG state."""
    args = settings(args)
    if is_signed(args):
        return fit_signed_definition(label_batches, args, provenance)
    column = target_column(args)
    rows, chunks, digest = 0, [], hashlib.sha256()
    zeros = ones = 0
    for labels in label_batches:
        values = labels[:, column].detach().cpu().to(torch.float64)
        validate_targets(values)
        digest.update(values.numpy().tobytes())
        rows += len(values)
        zeros += int((values == 0).sum())
        ones += int((values == 1).sum())
        chunks.append(values[(values > 0) & (values < 1)])
    if rows == 0:
        raise ValueError("Cannot fit a distribution without retained training labels.")
    interior = torch.cat(chunks)
    count = args.get("distribution_atoms", DEFAULT_ATOMS)
    atoms = torch.tensor([0., 1.], dtype=torch.float64)
    if len(interior) and count > 2:
        quantiles = (torch.tensor([.5], dtype=torch.float64) if count == 3 else
                     torch.linspace(0, 1, count - 2, dtype=torch.float64))
        atoms = torch.unique(torch.cat([atoms, torch.quantile(interior, quantiles)]), sorted=True)
    # Persist atoms representable in the float32 decoder; interpolation gaps stay nonzero.
    atoms = torch.unique(atoms.float(), sorted=True).double()
    occupancy = torch.zeros(len(atoms), dtype=torch.float64)
    occupancy[0], occupancy[-1] = zeros, ones
    for chunk in interior.split(16384):
        occupancy += encode_targets(chunk, atoms).sum(0)
    definition = dict(version=1, representation="categorical", encoding="linear_two_hot_original_units",
                      support_policy="training_interior_quantiles_v1", atoms=atoms.tolist(),
                      requested_atoms=count, effective_atoms=len(atoms), training_label_count=rows,
                      training_labels_sha256=digest.hexdigest(), provenance=provenance,
                      atom_gaps=torch.diff(atoms).tolist(), interior_label_count=len(interior),
                      encoded_training_occupancy=(occupancy / rows).tolist(),
                      zero_label_count=zeros, positive_label_count=rows - zeros)
    definition["definition_sha256"] = _digest(definition)
    return validate_definition(definition)


def fit_signed_definition(label_batches, args, provenance):
    column = target_column(args)
    chunks, digest = [], hashlib.sha256()
    for labels in label_batches:
        values = labels[:, column].detach().cpu().to(torch.float64)
        validate_targets(values, -1.)
        digest.update(values.numpy().tobytes())
        chunks.append(values)
    values = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float64)
    if not len(values):
        raise ValueError("Cannot fit a distribution without retained training labels.")
    count = args.get("distribution_atoms", DEFAULT_ATOMS)
    if count < 3:
        raise ValueError("Signed distributions require at least 3 atoms (-1, 0, 1).")
    interior = values[(values > -1) & (values < 1) & (values != 0)]
    atoms = torch.tensor([-1., 0., 1.], dtype=torch.float64)
    if len(interior) and count > 3:
        quantiles = torch.tensor([.5], dtype=torch.float64) if count == 4 else torch.linspace(0, 1, count - 3, dtype=torch.float64)
        atoms = torch.cat([atoms, torch.quantile(interior, quantiles)])
    atoms = torch.unique(atoms.float(), sorted=True).double()
    occupancy = torch.zeros(len(atoms), dtype=torch.float64)
    for chunk in values.split(16384):
        occupancy += encode_targets(chunk, atoms).sum(0)
    definition = dict(version=2, representation="categorical", encoding="linear_two_hot_original_units",
                      support_policy="signed_training_interior_quantiles_v1", atoms=atoms.tolist(),
                      requested_atoms=count, effective_atoms=len(atoms), training_label_count=len(values),
                      training_labels_sha256=digest.hexdigest(), provenance=provenance,
                      atom_gaps=torch.diff(atoms).tolist(), interior_label_count=len(interior),
                      encoded_training_occupancy=(occupancy / len(values)).tolist(),
                      zero_label_count=int((values == 0).sum()), positive_label_count=int((values > 0).sum()),
                      negative_label_count=int((values < 0).sum()))
    definition["definition_sha256"] = _digest(definition)
    return validate_definition(definition)


def atom_tensor(args, like):
    return torch.as_tensor(settings(args)["outcome_distribution_definition"]["atoms"],
                           dtype=like.dtype, device=like.device)


def encode_targets(values, atoms):
    validate_targets(values, atoms[0])
    upper = torch.searchsorted(atoms, values.contiguous()).clamp(1, len(atoms) - 1)
    lower = upper - 1
    fraction = (values - atoms[lower]) / (atoms[upper] - atoms[lower])
    encoded = values.new_zeros((*values.shape, len(atoms)))
    encoded.scatter_(-1, lower.unsqueeze(-1), (1 - fraction).unsqueeze(-1))
    encoded.scatter_add_(-1, upper.unsqueeze(-1), fraction.unsqueeze(-1))
    return encoded


def categorical_probabilities(logits, args=None):
    # Inference can force an entire candidate/branch to zero or mark it missing.
    zero = torch.isneginf(logits).all(-1)
    missing = torch.isnan(logits).all(-1)
    safe = torch.where((zero | missing).unsqueeze(-1), torch.zeros_like(logits), logits)
    probabilities = safe.softmax(-1)
    point_zero = torch.zeros_like(probabilities)
    zero_index = settings(args)["outcome_distribution_definition"]["atoms"].index(0.) if args is not None else 0
    point_zero[..., zero_index] = 1
    probabilities = torch.where(zero.unsqueeze(-1), point_zero, probabilities)
    return torch.where(missing.unsqueeze(-1), torch.full_like(probabilities, float("nan")), probabilities)


def expected_values(logits, args):
    if not is_distributional(args):
        if is_signed(args):
            return torch.where(torch.isneginf(logits), torch.zeros_like(logits), logits)
        return torch.sigmoid(logits)
    return (categorical_probabilities(logits, args) * atom_tensor(args, logits)).sum(-1)


def distribution_loss(logits, targets, args, weight=None):
    atoms = atom_tensor(args, logits)
    encoded = encode_targets(targets, atoms)
    losses = -(encoded * logits.log_softmax(-1)).sum(-1)
    return (losses if weight is None else losses * weight).mean()


def mean_bce(logits, targets, args, weight=None):
    """Stable BCE of a categorical expectation, even for extreme logits."""
    if is_signed(args):
        raise ValueError("BCE is undefined for signed gains.")
    validate_targets(targets)
    atoms = atom_tensor(args, logits)
    logp = logits.log_softmax(-1)
    logmean = torch.logsumexp(logp + atoms.log(), -1)
    logcomplement = torch.logsumexp(logp + torch.log1p(-atoms), -1)
    # Multiplication in log space avoids underflow of exp for nonzero targets.
    losses = -(torch.where(targets == 0, 0., targets * logmean)
               + torch.where(targets == 1, 0., (1 - targets) * logcomplement))
    return (losses if weight is None else losses * weight).mean()


def crps(logits, targets, args):
    atoms = atom_tensor(args, logits)
    probabilities = categorical_probabilities(logits, args)
    first = (probabilities * (atoms - targets.unsqueeze(-1)).abs()).sum(-1)
    # E|X-X'| / 2 from prefix sums; no K-by-K tensor per sample.
    before = probabilities.cumsum(-1) - probabilities
    weighted_before = (probabilities * atoms).cumsum(-1) - probabilities * atoms
    return first - (probabilities * (atoms * before - weighted_before)).sum(-1)


def distribution_identity(args):
    args = settings(args)
    return {"learning_mode": args.get("learning_mode", "scalar"),
            "outcome_learning_contract": args.get("outcome_learning_contract"),
            "outcome_distribution_definition": args.get("outcome_distribution_definition"),
            "checkpoint_monitor": args.get("checkpoint_monitor", "ce_loss")}


def validate_state_support(state_dict, args):
    if not is_distributional(args):
        return
    expected = torch.tensor(validate_definition(settings(args).get("outcome_distribution_definition"))["atoms"], dtype=torch.float32)
    actual = state_dict.get("decoder.outcome_atoms")
    if actual is None or not torch.equal(actual.detach().cpu().float(), expected):
        raise ValueError("Checkpoint atom buffer disagrees with its distribution definition.")


def configure_distribution_exports(models, enabled):
    for model in models.values():
        if model is not None and enabled and is_distributional(model.args):
            model.outcome_distribution_rows = []


def write_distribution_exports(directory, models):
    from pathlib import Path
    import pandas as pd
    rows = [row for model in models.values() if model is not None
            for row in getattr(model, "outcome_distribution_rows", [])]
    if not rows:
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(directory / "outcome_distributions.parquet", index=False)
    definitions = {model.args.get("model_id", task): model.args["outcome_distribution_definition"]
                   for task, model in models.items() if model is not None and is_distributional(model.args)}
    (directory / "outcome_distribution_definitions.json").write_text(json.dumps(definitions, indent=2), encoding="utf-8")
    return str(directory / "outcome_distributions.parquet")
