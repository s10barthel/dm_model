"""Explicit positive-target occurrence and conditional severity outcome heads."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F


def is_decomposition(args):
    return (args if isinstance(args, dict) else vars(args)).get("learning_mode") == "decomposition"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def severity_args(args):
    args = args if isinstance(args, dict) else vars(args)
    definition = args["outcome_decomposition_definition"]
    return {"learning_mode": "distributional", "outcome_distribution_definition": definition["severity_distribution"]}


def validate_definition(definition):
    from models.outcome_learning import validate_definition as validate_distribution
    if not isinstance(definition, dict):
        raise ValueError("Missing outcome decomposition definition.")
    payload = {k: v for k, v in definition.items() if k != "definition_sha256"}
    if (definition.get("definition_sha256") != _hash(payload) or definition.get("version") != 1
            or definition.get("occurrence_rule") != "target_strictly_positive"
            or definition.get("severity_mode") not in {"scalar", "distributional"}
            or definition.get("branch_order") != ["failure", "success"]
            or definition.get("target_definition", {}).get("xt_targets", {}).get("representation") != "absolute"
            or definition.get("normalization") != "weighted_joint_mean_over_all_examples"
            or definition.get("checkpoint_monitor") != "mean_soft_bce"
            or definition.get("occurrence_loss") != "binary_ce"
            or definition.get("severity_loss") != ("soft_bce" if definition.get("severity_mode") == "scalar" else "categorical_ce")
            or definition.get("decoder") != "occurrence_probability_times_conditional_severity_mean"):
        raise ValueError("Invalid outcome decomposition definition.")
    if definition["severity_mode"] == "distributional":
        support = validate_distribution(definition.get("severity_distribution"))
        if support["version"] != 1:
            raise ValueError("Decomposition severity must use bounded nonnegative atoms.")
    elif definition.get("severity_distribution") is not None:
        raise ValueError("Scalar severity cannot have categorical support.")
    counts = definition.get("training_branch_counts", {})
    for branch in ("failure", "success"):
        item = counts.get(branch, {})
        if item.get("positive", 0) <= 0 or item.get("total", 0) < item["positive"]:
            raise ValueError(f"Decomposition branch {branch} needs positive training observations.")
    return definition


def normalize(args, source=None):
    from models.outcome_learning import TASKS, DEFAULT_ATOMS
    if args.get("task") not in TASKS:
        raise ValueError("Decomposition supports outcome_scoring and outcome_conceding only.")
    if args.get("model", "gat") in {"xgboost", "catboost"} or args.get("include_out") or args.get("weight_bce"):
        raise ValueError("Decomposition requires a GNN without include_out or outcome class balancing.")
    saved = (source or {}).get("outcome_decomposition_definition")
    severity = args.get("severity_mode") or (source or {}).get("severity_mode") or "scalar"
    if severity not in {"scalar", "distributional"}:
        raise ValueError("Unknown decomposition severity mode.")
    target = args.get("xt_target_definition")
    if target is None and source is not None:
        target = source.get("xt_target_definition")
        if target is not None:
            args["xt_target_definition"] = copy.deepcopy(target)
    # Early CLI normalization can precede feature metadata resolution. Validate again
    # unconditionally in configure_outcome_learning and in checkpoint loading.
    if target is not None and (not args.get("use_xt") or target.get("xt_targets", {}).get("representation") != "absolute"):
        raise ValueError("Decomposition requires absolute xT targets; clipped and signed gain are forbidden.")
    if args.get("target_family") not in (None, "xt"):
        raise ValueError("Decomposition supports absolute xT only.")
    if severity == "scalar" and args.get("distribution_atoms") is not None:
        raise ValueError("--distribution-atoms requires distributional severity in decomposition mode.")
    if args.get("checkpoint_monitor") not in (None, "mean_soft_bce"):
        raise ValueError("Decomposition checkpoint must monitor mean_soft_bce.")
    args["severity_mode"] = severity
    args["checkpoint_monitor"] = "mean_soft_bce"
    if severity == "distributional":
        count = args.get("distribution_atoms")
        if count is None:
            count = (source or {}).get("distribution_atoms") or DEFAULT_ATOMS
        if type(count) is not int or count < 2:
            raise ValueError("--distribution-atoms must be an integer >= 2.")
        args["distribution_atoms"] = count
    if source is not None:
        validate_definition(saved)
        if severity != saved["severity_mode"]:
            raise ValueError("Resume/refit cannot change saved severity mode.")
        if severity == "distributional" and args["distribution_atoms"] != saved["severity_distribution"]["requested_atoms"]:
            raise ValueError("Resume/refit cannot change saved severity atoms.")
        if args.get("outcome_decomposition_definition") not in (None, saved):
            raise ValueError("Resume/refit decomposition definition conflicts with its source.")
        args["outcome_decomposition_definition"] = copy.deepcopy(saved)
    if args.get("outcome_decomposition_definition") is not None:
        definition = validate_definition(args["outcome_decomposition_definition"])
        if severity == "distributional" and args["distribution_atoms"] != definition["severity_distribution"]["requested_atoms"]:
            raise ValueError("Decomposition metadata disagrees on severity atom count.")
        if definition["severity_mode"] != severity or definition["target_definition"] != target:
            raise ValueError("Decomposition metadata disagrees with target or severity mode.")
    return args


def fit_definition(label_batches, args, provenance):
    from datatools.config import LABEL_INDEX
    from models.outcome_learning import target_column, validate_targets, fit_definition as fit_distribution
    column = target_column(args)
    positive_batches, digest = [], hashlib.sha256()
    counts = {name: {"total": 0, "positive": 0, "zero": 0} for name in ("failure", "success")}
    for labels in label_batches:
        values = labels[:, column].detach().cpu()
        branches = labels[:, LABEL_INDEX["success"]].detach().cpu()
        validate_targets(values)
        if not bool(((branches == 0) | (branches == 1)).all()):
            raise ValueError("Decomposition requires valid observed execution branches.")
        digest.update(torch.stack([branches.double(), values.double()], -1).numpy().tobytes())
        for branch, name in enumerate(("failure", "success")):
            selected = values[branches == branch]
            counts[name]["total"] += len(selected)
            counts[name]["positive"] += int((selected > 0).sum())
            counts[name]["zero"] += int((selected == 0).sum())
        if args["severity_mode"] == "distributional":
            positive_batches.append(values[values > 0])
    for branch, count in counts.items():
        if count["total"] == 0 or count["positive"] == 0:
            raise ValueError(f"Cannot fit decomposition: {branch} has {count['total']} observations and {count['positive']} positives.")
    definition = dict(version=1, occurrence_rule="target_strictly_positive", branch_order=["failure", "success"],
        severity_mode=args["severity_mode"], target_definition=args["xt_target_definition"],
        training_branch_counts=counts, training_labels_sha256=digest.hexdigest(), provenance=provenance,
        normalization="weighted_joint_mean_over_all_examples", checkpoint_monitor="mean_soft_bce",
        occurrence_loss="binary_ce", severity_loss="soft_bce" if args["severity_mode"] == "scalar" else "categorical_ce",
        decoder="occurrence_probability_times_conditional_severity_mean", severity_distribution=None)
    if args["severity_mode"] == "distributional":
        def positive_labels():
            for values in positive_batches:
                labels = values.new_zeros((len(values), column+1))
                labels[:, column] = values
                yield labels
        definition["severity_distribution"] = fit_distribution(positive_labels(), {**args, "learning_mode": "distributional"}, provenance)
    definition["definition_sha256"] = _hash(definition)
    return validate_definition(definition)


def decode(logits, args):
    from models.outcome_learning import expected_values
    args = args if isinstance(args, dict) else vars(args)
    forced = torch.isneginf(logits).all(-1)
    missing = torch.isnan(logits).all(-1)
    safe = torch.where((forced | missing).unsqueeze(-1), torch.zeros_like(logits), logits)
    q = safe[..., 0].sigmoid()
    m = (safe[..., 1].sigmoid() if args["severity_mode"] == "scalar"
         else expected_values(safe[..., 1:], severity_args(args)))
    value = q * m
    return dict(occurrence_probability=torch.where(missing, torch.nan, torch.where(forced, 0., q)),
                conditional_severity=torch.where(forced | missing, torch.nan, m),
                prediction=torch.where(missing, torch.nan, torch.where(forced, 0., value)),
                forced=forced, missing=missing, raw_occurrence_probability=torch.where(forced | missing, torch.nan, q),
                raw_conditional_severity=torch.where(forced | missing, torch.nan, m))


def losses(logits, targets, args, weight=None):
    from models.outcome_learning import encode_targets, atom_tensor, validate_targets
    validate_targets(targets)
    positive = targets > 0
    occurrence = F.binary_cross_entropy_with_logits(logits[..., 0], positive.to(logits.dtype), reduction="none")
    severity = torch.zeros_like(occurrence)
    if bool(positive.any()):
        if args.severity_mode == "scalar":
            severity[positive] = F.binary_cross_entropy_with_logits(logits[positive, 1], targets[positive], reduction="none")
        else:
            encoded = encode_targets(targets[positive], atom_tensor(severity_args(args), logits))
            severity[positive] = -(encoded * logits[positive, 1:].log_softmax(-1)).sum(-1)
    weighted = lambda value: (value if weight is None else value * weight).mean()
    return weighted(occurrence + severity), weighted(occurrence), weighted(severity)


def mean_bce(logits, targets, args, weight=None):
    from models.outcome_learning import atom_tensor
    logq, lognotq = F.logsigmoid(logits[..., 0]), F.logsigmoid(-logits[..., 0])
    if args.severity_mode == "scalar":
        logm, lognotm = F.logsigmoid(logits[..., 1]), F.logsigmoid(-logits[..., 1])
    else:
        atoms = atom_tensor(severity_args(args), logits)
        logp = logits[..., 1:].log_softmax(-1)
        logm = torch.logsumexp(logp + atoms.log(), -1)
        lognotm = torch.logsumexp(logp + torch.log1p(-atoms), -1)
    logmean = logq + logm
    logcomplement = torch.logaddexp(lognotq, logq + lognotm)
    loss = -(torch.where(targets == 0, 0., targets * logmean)
             + torch.where(targets == 1, 0., (1 - targets) * logcomplement))
    return (loss if weight is None else loss * weight).mean()


def head_metrics(target, q, m, severity_ce=None, severity_crps=None, outcome_crps=None):
    from sklearn.metrics import roc_auc_score, average_precision_score
    target, q, m = (np.asarray(x, dtype=float) for x in (target, q, m))
    z = target > 0
    count, positives = len(target), int(z.sum())
    clipped = np.clip(q, np.finfo(float).eps, 1 - np.finfo(float).eps)
    metrics = dict(count=count, positives=positives, prevalence=float(z.mean()) if count else None,
        occurrence_log_loss=float(-(z*np.log(clipped)+(~z)*np.log1p(-clipped)).mean()) if count else None,
        occurrence_brier=float(np.mean((q-z)**2)) if count else None,
        occurrence_roc_auc=float(roc_auc_score(z,q)) if 0 < positives < count else None,
        occurrence_average_precision=float(average_precision_score(z,q)) if positives and positives < count else None,
        severity_mae=float(np.mean(np.abs(m[z]-target[z]))) if positives else None,
        severity_rmse=float(np.sqrt(np.mean((m[z]-target[z])**2))) if positives else None,
        severity_bias=float(np.mean(m[z]-target[z])) if positives else None)
    for name, values in (("severity_ce", severity_ce), ("severity_crps", severity_crps)):
        if values is not None:
            metrics[name] = float(np.mean(np.asarray(values)[z])) if positives else None
    if outcome_crps is not None:
        metrics["outcome_crps"] = float(np.mean(outcome_crps)) if count else None
    def bins(pred, observed):
        ids = np.minimum((pred*10).astype(int),9)
        return [dict(bin=i,count=int((ids==i).sum()),prediction=float(pred[ids==i].mean()),
                     observed=float(observed[ids==i].mean())) for i in range(10) if (ids==i).any()]
    conditional_bce = None
    if positives:
        positive_m = np.clip(m[z], np.finfo(float).eps, 1-np.finfo(float).eps)
        conditional_bce = float(-(target[z]*np.log(positive_m)+(1-target[z])*np.log1p(-positive_m)).mean())
    metrics["severity_soft_bce"] = conditional_bce
    if count:
        contribution = (positives/count) * (metrics.get("severity_ce", conditional_bce) or 0.)
        metrics["joint_loss"] = metrics["occurrence_log_loss"] + contribution
    else:
        metrics["joint_loss"] = None
    metrics["occurrence_calibration"] = bins(q,z.astype(float)) if count else []
    metrics["severity_calibration"] = bins(m[z],target[z]) if positives else []
    return metrics


def distribution_diagnostics(logits, targets, args):
    """Conditional and assembled CRPS; zero is only a severity interpolation boundary."""
    from models.outcome_learning import atom_tensor, encode_targets, crps
    support = severity_args(args)
    encoded = encode_targets(targets, atom_tensor(support, logits))
    ce = -(encoded * logits[..., 1:].log_softmax(-1)).sum(-1)
    conditional = crps(logits[..., 1:], targets, support)
    q = logits[..., 0].sigmoid()
    probabilities = q.unsqueeze(-1) * logits[..., 1:].softmax(-1)
    zero = torch.zeros_like(probabilities)
    zero[..., 0] = 1-q
    assembled = probabilities + zero
    # log(0) is safe for CRPS, which decodes with softmax; endpoints can be exact.
    combined = crps(assembled.log(), targets, support)
    return ce, conditional, combined


def configure_exports(models, enabled):
    for model in models.values():
        if model is not None and enabled and is_decomposition(model.args):
            model.outcome_decomposition_rows = []


def write_exports(directory, models):
    from pathlib import Path
    import pandas as pd
    rows = [row for model in models.values() if model is not None
            for row in getattr(model, "outcome_decomposition_rows", [])]
    if not rows:
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(directory / "outcome_decomposition.parquet", index=False)
    definitions = {model.args.get("model_id", task): model.args["outcome_decomposition_definition"]
        for task, model in models.items() if model is not None and is_decomposition(model.args)}
    (directory / "outcome_decomposition_definitions.json").write_text(json.dumps(definitions, indent=2), encoding="utf-8")
    return str(directory / "outcome_decomposition.parquet")
