"""Versioned categorical outcome learning in the original target units."""
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


def add_learning_arguments(parser):
    parser.add_argument("--learning-mode", choices=("scalar", "distributional"), default=None,
                        help="Outcome learning mode; fresh training defaults to scalar.")
    parser.add_argument("--distribution-atoms", type=int, default=None,
                        help="Maximum categorical atom count, including 0 and 1 (default: 65).")


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
        args["checkpoint_monitor"] = "mean_soft_bce"
        if args.get("outcome_distribution_definition") is not None:
            validate_definition(args["outcome_distribution_definition"])
    return args


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_definition(definition):
    if not isinstance(definition, dict):
        raise ValueError("Distributional checkpoint is missing its distribution definition.")
    payload = {k: v for k, v in definition.items() if k != "definition_sha256"}
    if (payload.get("version") != 1 or payload.get("encoding") != "linear_two_hot_original_units"
            or payload.get("representation") != "categorical"
            or definition.get("definition_sha256") != _digest(payload)):
        raise ValueError("Invalid or inconsistent outcome distribution definition.")
    atoms = np.asarray(payload.get("atoms", []), dtype=float)
    if (atoms.ndim != 1 or len(atoms) < 2 or not np.isfinite(atoms).all()
            or atoms[0] != 0 or atoms[-1] != 1 or not (np.diff(atoms) > 0).all()
            or payload.get("effective_atoms") != len(atoms)
            or type(payload.get("requested_atoms")) is not int
            or len(atoms) > payload["requested_atoms"]):
        raise ValueError("Distribution atoms must be strictly increasing from 0 to 1.")
    return definition


def validate_learning_metadata(args, metadata):
    normalize_learning_settings(args)
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


def validate_targets(values):
    if not bool((torch.isfinite(values) & (values >= 0) & (values <= 1)).all()):
        raise ValueError("Distributional outcome targets must be finite and in [0, 1]; values are not clipped.")


def fit_definition(label_batches, args, provenance):
    """Read retained training labels only, without consuming RNG state."""
    args = settings(args)
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


def atom_tensor(args, like):
    return torch.as_tensor(settings(args)["outcome_distribution_definition"]["atoms"],
                           dtype=like.dtype, device=like.device)


def encode_targets(values, atoms):
    validate_targets(values)
    upper = torch.searchsorted(atoms, values.contiguous()).clamp(1, len(atoms) - 1)
    lower = upper - 1
    fraction = (values - atoms[lower]) / (atoms[upper] - atoms[lower])
    encoded = values.new_zeros((*values.shape, len(atoms)))
    encoded.scatter_(-1, lower.unsqueeze(-1), (1 - fraction).unsqueeze(-1))
    encoded.scatter_add_(-1, upper.unsqueeze(-1), fraction.unsqueeze(-1))
    return encoded


def categorical_probabilities(logits):
    # Inference can force an entire candidate/branch to zero or mark it missing.
    zero = torch.isneginf(logits).all(-1)
    missing = torch.isnan(logits).all(-1)
    safe = torch.where((zero | missing).unsqueeze(-1), torch.zeros_like(logits), logits)
    probabilities = safe.softmax(-1)
    point_zero = torch.zeros_like(probabilities)
    point_zero[..., 0] = 1
    probabilities = torch.where(zero.unsqueeze(-1), point_zero, probabilities)
    return torch.where(missing.unsqueeze(-1), torch.full_like(probabilities, float("nan")), probabilities)


def expected_values(logits, args):
    if not is_distributional(args):
        return torch.sigmoid(logits)
    return (categorical_probabilities(logits) * atom_tensor(args, logits)).sum(-1)


def distribution_loss(logits, targets, args, weight=None):
    atoms = atom_tensor(args, logits)
    encoded = encode_targets(targets, atoms)
    losses = -(encoded * logits.log_softmax(-1)).sum(-1)
    return (losses if weight is None else losses * weight).mean()


def mean_bce(logits, targets, args, weight=None):
    """Stable BCE of a categorical expectation, even for extreme logits."""
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
    probabilities = categorical_probabilities(logits)
    first = (probabilities * (atoms - targets.unsqueeze(-1)).abs()).sum(-1)
    # E|X-X'| / 2 from prefix sums; no K-by-K tensor per sample.
    before = probabilities.cumsum(-1) - probabilities
    weighted_before = (probabilities * atoms).cumsum(-1) - probabilities * atoms
    return first - (probabilities * (atoms * before - weighted_before)).sum(-1)


def distribution_identity(args):
    args = settings(args)
    return {"learning_mode": args.get("learning_mode", "scalar"),
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
