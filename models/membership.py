"""Read recorded match membership and describe visualization provenance.

Visualization reports are informational. Evaluation uses the same evidence with
strict loading and requires complete exclusion evidence before proceeding.
"""
from __future__ import annotations

from collections.abc import Iterable

from project_config import (
    LEGACY_FEATURE_SPLIT_MANIFESTS,
    load_feature_run_metadata,
    load_recorded_split_manifest,
    split_selector,
)

REPORT_VERSION = 2
NO_VERIFIED_ID = "no_verified_sportec_match_id"
SKIPPED_MESSAGE = "Training membership check skipped: no verified mapping to Sportec match IDs."


class MissingMembershipError(ValueError):
    """Absent historical evidence, distinct from invalid or conflicting records."""

    def __init__(self, message, *, check=None):
        super().__init__(message)
        self.check = check


def comparable_identity(dataset, match_id):
    # Provider-native IDs are never treated as Sportec IDs without a mapping.
    return dataset == "sportec" and match_id is not None and bool(str(match_id).strip())


def _new_report(dataset, identities):
    comparable = [comparable_identity(dataset, match_id) for _, match_id in identities]
    status = ("checked" if comparable and all(comparable) else
              "partially_checked" if any(comparable) else "not_checked")
    result = {"version": REPORT_VERSION, "dataset": dataset, "check_status": status,
              "models": [], "entries": []}
    if status != "checked":
        result["reason"] = NO_VERIFIED_ID
    return result


def _object(value, name):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Invalid recorded {name}: expected an object.")
    return value


def recorded_manifest_id(record, *, feature_loader=None):
    feature_loader = feature_loader or load_feature_run_metadata
    metadata = _object(record.get("metadata"), "metadata")
    arguments = _object(record.get("args"), "arguments")
    manifest_id = record.get("split_manifest_id") or metadata.get("split_manifest_id") or arguments.get("split_manifest_id")
    if manifest_id:
        return manifest_id
    feature_id = record.get("feature_run_id") or metadata.get("feature_run_id") or arguments.get("feature_run_id")
    if feature_id:
        feature = _object(feature_loader(feature_id, required=False), "feature metadata")
        return feature.get("split_manifest_id") or LEGACY_FEATURE_SPLIT_MANIFESTS.get(feature_id)
    return None


def explicit_split_provenance(record):
    """Avoid the legacy 50% default inserted into normalized model records."""
    metadata = _object(record.get("metadata"), "metadata")
    arguments = _object(record.get("args"), "arguments")
    source = next((item for item in (metadata, arguments)
                   if any(item.get(key) is not None for key in ("train_split_percent", "train_split", "train_count"))), None)
    if source is None:
        # Lightweight records from archived runs can carry explicit fields only.
        source = record if "metadata" not in record and "args" not in record else {}
    selector = split_selector(source, default=None)
    return {"train_split_percent": selector["train_split"], "train_count": selector["train_count"],
            "split_manifest_id": record.get("split_manifest_id") or metadata.get("split_manifest_id") or arguments.get("split_manifest_id")}


def resolve_membership_evidence(record, *, strict=False, require_complete=True, manifest_loader=None, feature_loader=None):
    manifest_loader = manifest_loader or load_recorded_split_manifest
    memberships, contributing, sources, errors = {}, {}, {}, []

    def safe_object(value, name):
        try:
            return _object(value, name)
        except ValueError as exc:
            if strict:
                raise
            errors.append(str(exc))
            return {}

    metadata = safe_object(record.get("metadata"), "metadata")
    arguments = safe_object(record.get("args"), "arguments")
    record = dict(record)
    if "metadata" in record:
        record["metadata"] = metadata
    if "args" in record:
        record["args"] = arguments
    datasets = safe_object(metadata.get("split_datasets"), "split datasets")

    def read_ids(value, source):
        if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(f"Invalid recorded match membership: {source}.")
        if len(value) != len(set(value)):
            raise ValueError(f"Duplicate recorded match membership: {source}.")
        return set(value)

    for category, key, dataset_key in (("training", "train_match_ids", "train"),
                                       ("validation", "validation_match_ids", "validation")):
        dataset = safe_object(datasets.get(dataset_key), f"{category} dataset")
        value, source = metadata.get(key), f"metadata.{key}"
        if value is None:
            field = "requested_match_ids" if "requested_match_ids" in dataset else "contributing_match_ids"
            value, source = dataset.get(field), f"split_datasets.{dataset_key}.{field}"
        try:
            if value is not None:
                memberships[category] = read_ids(value, source)
                sources[category] = source
            actual = dataset.get("contributing_match_ids")
            if actual is not None:
                contributing[category] = read_ids(actual, f"split_datasets.{dataset_key}.contributing_match_ids")
                sources[f"{category}_contributing"] = f"split_datasets.{dataset_key}.contributing_match_ids"
                if category in memberships and not contributing[category].issubset(memberships[category]):
                    errors.append(f"Contributing {category} matches conflict with recorded allocation.")
            requested = dataset.get("requested_match_ids")
            if requested is not None and category in memberships and read_ids(
                requested, f"split_datasets.{dataset_key}.requested_match_ids"
            ) != memberships[category]:
                errors.append(f"Requested {category} matches conflict with recorded allocation.")
        except (ValueError, TypeError) as exc:
            if strict:
                raise
            errors.append(str(exc))
    manifest_id = None
    try:
        manifest_id = recorded_manifest_id(record, feature_loader=feature_loader)
        recorded_ids = {item.get("split_manifest_id") for item in (record, metadata, record.get("args") or {})
                        if item.get("split_manifest_id")}
        if len(recorded_ids) > 1:
            errors.append("Conflicting recorded development manifest IDs.")
        if manifest_id:
            manifest = manifest_loader(manifest_id)
            memberships["development"] = set(manifest["train"])
            sources["development"] = f"manifest:{manifest_id}"
            selector = explicit_split_provenance(record)
            if any(selector.get(key) is not None for key in ("train_split_percent", "train_count")) and (
                split_selector(selector, default=None) != split_selector(manifest, default=None)
            ):
                errors.append("Recorded split selector conflicts with the development manifest.")
            for category in ("training", "validation"):
                if not memberships.get(category, set()).issubset(memberships["development"]):
                    errors.append(f"Recorded {category} matches lie outside the development manifest.")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        if strict:
            raise
        errors.append(f"Cannot load development provenance: {exc}")
    if strict and errors:
        raise ValueError(" ".join(errors))
    complete = "development" in memberships or {"training", "validation"}.issubset(memberships)
    if not complete:
        message = f"Cannot verify training/validation exclusion for {record.get('model_id', 'checkpoint')}: missing recorded membership."
        if strict and require_complete:
            raise MissingMembershipError(message)
        errors.append(message)
    return {"memberships": memberships, "contributing": contributing, "sources": sources,
            "split_manifest_id": manifest_id, "complete": complete, "errors": errors}


def describe_match_membership(evidence, dataset, match_id):
    comparable = comparable_identity(dataset, match_id)
    if not comparable:
        return {"canonical_match_id": None, "status": "not_checked", "check_status": "not_checked",
                "reason": NO_VERIFIED_ID,
                "membership": dict.fromkeys(("training", "validation", "development")),
                "contributing": dict.fromkeys(("training", "validation")),
                "evidence_sources": {},
                "reasons": ["No verified mapping from the displayed identity to recorded Sportec match IDs."]}
    flags = {name: (str(match_id) in ids if comparable else None)
             for name, ids in evidence["memberships"].items()}
    flags = {name: flags.get(name) for name in ("training", "validation", "development")}
    contributed = {name: (str(match_id) in evidence["contributing"][name]
                         if comparable and name in evidence["contributing"] else None)
                   for name in ("training", "validation")}
    positive = next((name for name in ("training", "validation", "development") if flags[name]), None)
    status = positive or ("held_out" if comparable and evidence["complete"] and not evidence["errors"] else "unknown")
    reasons = list(evidence["errors"])
    return {"canonical_match_id": str(match_id), "status": status, "check_status": "checked",
            "membership": flags, "contributing": contributed,
            "evidence_sources": evidence["sources"], "reasons": reasons}


def displayed_identities(metadata, dataset):
    """Return (scene, native match) pairs using existing run metadata."""
    if dataset in {"sportec", "skillcorner"} and metadata.get("match_id") is not None:
        return [(str(metadata["match_id"]), str(metadata["match_id"]))]
    if dataset == "skillcorner":
        return [(str(value), str(value)) for value in metadata.get("processed_matches", [])]
    if dataset == "benchmark":
        states = metadata.get("processed_states", [])
        if not states:
            states = [{"modification": item["modification"], "game_state": state}
                      for item in metadata.get("rendered_modifications", []) for state in item.get("game_states", [])]
        return [(f"modification_{item['modification']}_game_state_{item['game_state']}", None) for item in states]
    values = metadata.get("processed_situation_ids") or metadata.get("rendered_situation_ids")
    if values is None:
        values = [item["situation_id"] for item in metadata.get("rendered_situations", [])]
    if not values:
        values = [f"selection_row_{value}" for value in metadata.get("processed_selection_row_ids", [])]
    return [(str(value), None) for value in values]


class MembershipReport:
    """Run-local caches; no global stale checkpoints or manifest assignments."""

    def __init__(self):
        self.records = {}
        self.evidence = {}
        self.manifests = {}
        self.printed = set()
        self.warned_models = set()
        self.printed_skip = False

    def _manifest(self, identifier):
        if identifier not in self.manifests:
            self.manifests[identifier] = load_recorded_split_manifest(identifier)
        return self.manifests[identifier]

    def build(self, dataset, identities: Iterable, source, *, records=None, variant_id=None, roles=None):
        source = source or {}
        embedded = records if records is not None else source.get("model_records") or {}
        ids = next((source[key] for key in ("selected_model_ids", "model_ids", "models", "source_models")
                    if source.get(key)), {})
        ids = {role: (value.get("model_id") if isinstance(value, dict) else value) for role, value in ids.items()}
        if not ids:
            ids = {role: record.get("model_id") for role, record in embedded.items()}
        if roles is not None:
            allowed = set(roles)
            allowed.update({"success_intent"} if "intended_recipient" in allowed else set())
            ids = {role: value for role, value in ids.items() if role in allowed}
        for role, auxiliary in (source.get("xpass_auxiliaries") or {}).items():
            if roles is None or "pass_success" in roles:
                name = "xpass_" + role
                ids[name] = auxiliary.get("model_id")
                embedded = {**embedded, name: auxiliary.get("model_record") or {}}
        snapshot = source.get("model_membership") or {}
        if not isinstance(snapshot, dict) or snapshot.get("version") not in (1, REPORT_VERSION):
            snapshot = {}
        if not ids:
            ids = {item["role"]: item["model_id"] for item in snapshot.get("models", [])
                   if roles is None or item["role"] in roles
                   or ("pass_success" in roles and item["role"].startswith("xpass_"))}
        elif "xpass_auxiliaries" not in source and "pass_success" in ids:
            # Older saved visualizations recorded auxiliary roles only in the
            # membership snapshot. Explicit current auxiliary settings win.
            for item in snapshot.get("models", []):
                if item["role"].startswith("xpass_"):
                    ids.setdefault(item["role"], item["model_id"])
        identities = list(dict.fromkeys(identities))
        result = _new_report(dataset, identities)
        entry_identities = identities if dataset == "sportec" else []
        for role, model_id in ids.items():
            if not isinstance(model_id, str) or not model_id:
                continue
            record = embedded.get(role) or {}
            if not record and isinstance((source.get("models") or {}).get(role), dict):
                record = source["models"][role]
            # Existing snapshots remain useful after their checkpoints are archived.
            saved_model = next((item for item in snapshot.get("models", [])
                                if item.get("role") == role and item.get("model_id") == model_id), None)
            entries = []
            # Native-provider reports retain model provenance but never resolve
            # checkpoint evidence or expand one entry per scene/model pair.
            for scene_id, match_id in entry_identities:
                saved = next((item for item in snapshot.get("entries", [])
                              if item.get("dataset") == dataset and item.get("role") == role
                              and item.get("model_id") == model_id
                              and ((match_id is not None and item.get("match_id") == str(match_id))
                                   or (match_id is None and item.get("scene_id") == str(scene_id)))), None)
                if not comparable_identity(dataset, match_id):
                    entry = {"dataset": dataset, "scene_id": str(scene_id), "match_id": match_id,
                             "variant_id": variant_id, "role": role, "model_id": model_id,
                             **describe_match_membership(None, dataset, match_id)}
                elif saved is not None and saved.get("status") != "not_checked":
                    entry = {**saved, "scene_id": str(scene_id), "variant_id": variant_id,
                             "check_status": "checked"}
                else:
                    if model_id not in self.evidence:
                        # A full embedded checkpoint record is authoritative. Older
                        # summaries need the exact local checkpoint for membership.
                        resolved = record
                        lookup_error = None
                        if not resolved.get("metadata"):
                            try:
                                from models.utils import get_model_record
                                resolved = get_model_record(model_id)
                            except (OSError, ValueError, TypeError, KeyError) as exc:
                                lookup_error = f"Checkpoint unavailable: {exc}"
                        resolved = {**resolved, "model_id": model_id}
                        self.records[model_id] = resolved
                        evidence = resolve_membership_evidence(resolved, manifest_loader=self._manifest)
                        # A summary's valid immutable manifest can still establish
                        # membership when the checkpoint itself is unavailable.
                        if lookup_error and not evidence["complete"]:
                            evidence["errors"].append(lookup_error)
                        self.evidence[model_id] = evidence
                    entry = {"dataset": dataset, "scene_id": str(scene_id),
                             "match_id": str(match_id) if match_id is not None else None,
                             "variant_id": variant_id, "role": role, "model_id": model_id,
                             **describe_match_membership(self.evidence[model_id], dataset, match_id)}
                entries.append(entry)
            if saved_model is not None:
                model = dict(saved_model)
            else:
                resolved = self.records.get(model_id, record)
                try:
                    provenance = explicit_split_provenance(resolved)
                except (ValueError, TypeError) as exc:
                    provenance = {"error": str(exc)}
                model = {"role": role, "model_id": model_id, **provenance}
                if model_id in self.evidence:
                    model["split_manifest_id"] = self.evidence[model_id]["split_manifest_id"]
            result["models"].append(model)
            result["entries"].extend(entries)
        return result

    def print(self, report):
        unchecked = report.get("check_status") in {"not_checked", "partially_checked"}
        missing_identity = any(entry.get("check_status") == "not_checked" for entry in report["entries"])
        if (unchecked and report["models"] and (report.get("dataset") != "sportec" or missing_identity)
                and not self.printed_skip):
            print(SKIPPED_MESSAGE)
            self.printed_skip = True
        for entry in report["entries"]:
            if entry.get("check_status") == "not_checked":
                continue
            identity = entry["match_id"] or entry["scene_id"]
            key = (("unknown", entry["model_id"]) if entry["status"] == "unknown" else
                   (entry["dataset"], identity, entry["role"], entry["model_id"], entry["variant_id"]))
            if key in self.printed:
                continue
            self.printed.add(key)
            actual = entry["contributing"].get("training")
            contributed = "unknown" if actual is None else ("yes" if actual else "no")
            variant = f" | variant={entry['variant_id']}" if entry["variant_id"] is not None else ""
            reasons = entry["reasons"] if entry["model_id"] not in self.warned_models else []
            reason = f" | {'; '.join(reasons)}" if reasons else ""
            if reasons:
                self.warned_models.add(entry["model_id"])
            print(f"Model membership | {entry['dataset']}/{identity} | {entry['role']}={entry['model_id']}"
                  f" | {entry['status']} (recorded membership) | training contributed={contributed}{variant}{reason}")


def attach_membership_report(metadata, dataset, *, source=None, records=None, identities=None, reporter=None):
    reporter = reporter if reporter is not None else MembershipReport()
    source = metadata if source is None else source
    roles = participating_roles(metadata)
    if metadata.get("selected_model_ids"):
        roles = None
    report = reporter.build(dataset, displayed_identities(metadata, dataset) if identities is None else identities,
                            source, records=records, roles=roles)
    metadata["model_membership"] = report
    reporter.print(report)
    return report


def participating_roles(metadata):
    if "rendered_components" not in metadata:
        return None
    from datatools.viz_helpers import DERIVED_COMPONENT_INPUTS
    roles = set()
    for component in metadata["rendered_components"]:
        for name in DERIVED_COMPONENT_INPUTS.get(component, (component,)):
            if name.startswith("outcome_scoring_"):
                name = "outcome_scoring"
            elif name.startswith("outcome_conceding_"):
                name = "outcome_conceding"
            elif name == "model_pass_success":
                name = "pass_success"
            roles.add(name)
    return roles


def attach_comparison_membership(metadata, *, reporter=None):
    dataset = {"visualize_action_components.py": "sportec", "run_and_visualize_hawkeye.py": "hawkeye",
               "visualize_hawkeye.py": "hawkeye", "visualize_benchmark.py": "benchmark",
               "visualize_skillcorner.py": "skillcorner"}.get(metadata.get("script"), "unknown")
    reporter = reporter if reporter is not None else MembershipReport()
    combined = _new_report(dataset, [])
    check_statuses = set()
    arguments = metadata.get("arguments") or {}
    for variant in metadata.get("variants", []):
        provenance = variant.get("provenance") or {}
        source = provenance.get("metadata", provenance)
        scenes = [item for item in metadata.get("scenes", []) if item.get("variant_id") == variant["id"]]
        identities = [(item["scene_id"], str(item.get("match_id") or arguments["match_id"])
                       if dataset in {"sportec", "skillcorner"} and (item.get("match_id") or arguments.get("match_id")) is not None
                       else None) for item in scenes]
        # Native comparison scene IDs can contain possession/action suffixes.
        report = reporter.build(dataset, identities, source, variant_id=variant["id"],
                                roles=participating_roles(metadata))
        check_statuses.add(report["check_status"])
        provenance["model_membership"] = report
        combined["models"].extend({**item, "variant_id": variant["id"]} for item in report["models"])
        combined["entries"].extend(report["entries"])
        reporter.print(report)
    if check_statuses == {"checked"}:
        combined["check_status"] = "checked"
        combined.pop("reason", None)
    elif check_statuses.intersection({"checked", "partially_checked"}):
        combined["check_status"] = "partially_checked"
    metadata["model_membership"] = combined
    return combined
