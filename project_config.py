from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_SEASON_ROOTS = {
    "23_24": PROJECT_ROOT / "Bundesliga_season_23_24",
    "24_25": PROJECT_ROOT / "Bundesliga_season_24_25",
}
TRAIN_SEASONS = ("23_24",)
TEST_SEASONS = ("24_25",)

DATA_ROOT = PROJECT_ROOT / "data" / "ajax"
LINEUP_DIR = DATA_ROOT / "lineup"
LINEUP_PATH = LINEUP_DIR / "line_up.parquet"
EVENT_DIR = DATA_ROOT / "event"
EVENT_PATH = EVENT_DIR / "event.parquet"
TRACKING_DIR = DATA_ROOT / "tracking"
TRACKING_PROCESSED_DIR = DATA_ROOT / "tracking_processed"
EVENT_SYNCED_DIR = DATA_ROOT / "event_synced"
FEATURE_DIR = DATA_ROOT / "features"
ACTION_GRAPH_DIR = FEATURE_DIR / "action_graphs"
ACTION_GRAPH_INTENT_TRAIN_DIR = FEATURE_DIR / "action_graphs_intent_train"
POST_ACTION_GRAPH_DIR = FEATURE_DIR / "post_action_graphs"
XT_DIR = DATA_ROOT / "xT"
XT_MATCH_DIR = XT_DIR / "matches"
SAVED_DIR = PROJECT_ROOT / "saved"
COMPONENT_DIR = DATA_ROOT / "defcon_components"
SPLIT_DIR = DATA_ROOT / "splits"
SPLIT_PATH = SPLIT_DIR / "match_splits.json"

MODEL_TRAIN_FRACTION = 0.8
INTENT_TRAIN_OFFSETS = (12, 25)

INTENDED_RECEIVER_MODE_ORIGINAL = "original"
INTENDED_RECEIVER_MODE_ANGLE_ONLY = "angle_only"
INTENDED_RECEIVER_MODE_MODEL = "model"
INTENDED_RECEIVER_MODES = (
    INTENDED_RECEIVER_MODE_ORIGINAL,
    INTENDED_RECEIVER_MODE_ANGLE_ONLY,
    INTENDED_RECEIVER_MODE_MODEL,
)
DEFAULT_INTENDED_RECEIVER_MODE = INTENDED_RECEIVER_MODE_ANGLE_ONLY
use_intended_receiver_model = False
DEFAULT_INTENDED_RECEIVER_MODEL_ID = "success_intent/00"
SUCCESS_INTENT_GRAPH_DIR = FEATURE_DIR / "action_graphs_success_intent"

RELEVANT_MODEL_IDS = {
    INTENDED_RECEIVER_MODE_ORIGINAL: {
        False: {
            "action_intent": "action_intent/00",
            "pass_intent": "pass_intent/20",
            "pass_success": "pass_success/20",
            "outcome_scoring": "outcome_scoring/20",
            "outcome_conceding": "outcome_conceding/20",
            "failure_receiver": "failure_receiver/21",
            "success_intent": DEFAULT_INTENDED_RECEIVER_MODEL_ID,
        },
        True: {
            "action_intent": "action_intent/00",
            "pass_intent": "pass_intent/20",
            "pass_success": "pass_success/20",
            "outcome_scoring": "outcome_scoring/21",
            "outcome_conceding": "outcome_conceding/21",
            "failure_receiver": "failure_receiver/21",
            "success_intent": DEFAULT_INTENDED_RECEIVER_MODEL_ID,
        },
    },
    INTENDED_RECEIVER_MODE_ANGLE_ONLY: {
        False: {
            "action_intent": "action_intent/30",
            "pass_intent": "pass_intent/30",
            "pass_success": "pass_success/30",
            "outcome_scoring": "outcome_scoring/30",
            "outcome_conceding": "outcome_conceding/30",
            "failure_receiver": "failure_receiver/30",
            "success_intent": DEFAULT_INTENDED_RECEIVER_MODEL_ID,
        },
        True: {
            "action_intent": "action_intent/30",
            "pass_intent": "pass_intent/30",
            "pass_success": "pass_success/30",
            "outcome_scoring": "outcome_scoring/31",
            "outcome_conceding": "outcome_conceding/31",
            "failure_receiver": "failure_receiver/30",
            "success_intent": DEFAULT_INTENDED_RECEIVER_MODEL_ID,
        },
    },
    INTENDED_RECEIVER_MODE_MODEL: {
        False: {
            "action_intent": "action_intent/40",
            "pass_intent": "pass_intent/40",
            "pass_success": "pass_success/40",
            "outcome_scoring": "outcome_scoring/40",
            "outcome_conceding": "outcome_conceding/40",
            "failure_receiver": "failure_receiver/40",
            "success_intent": DEFAULT_INTENDED_RECEIVER_MODEL_ID,
        },
        True: {
            "action_intent": "action_intent/40",
            "pass_intent": "pass_intent/40",
            "pass_success": "pass_success/40",
            "outcome_scoring": "outcome_scoring/41",
            "outcome_conceding": "outcome_conceding/41",
            "failure_receiver": "failure_receiver/40",
            "success_intent": DEFAULT_INTENDED_RECEIVER_MODEL_ID,
        },
    },
}


def ensure_project_dirs() -> None:
    for path in [
        DATA_ROOT,
        LINEUP_DIR,
        EVENT_DIR,
        TRACKING_DIR,
        TRACKING_PROCESSED_DIR,
        EVENT_SYNCED_DIR,
        FEATURE_DIR,
        XT_DIR,
        XT_MATCH_DIR,
        COMPONENT_DIR,
        SUCCESS_INTENT_GRAPH_DIR,
        SAVED_DIR,
        SPLIT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_intended_receiver_mode(
    use_original_intended_receiver: bool = False,
    use_intended_receiver_model: bool | None = None,
) -> str:
    use_model = (
        bool(globals()["use_intended_receiver_model"])
        if use_intended_receiver_model is None
        else bool(use_intended_receiver_model)
    )

    if use_original_intended_receiver and use_model:
        raise ValueError(
            "--use-original-intended-receiver and --use-intended-receiver-model are mutually exclusive."
        )
    if use_model:
        return INTENDED_RECEIVER_MODE_MODEL
    if use_original_intended_receiver:
        return INTENDED_RECEIVER_MODE_ORIGINAL
    return DEFAULT_INTENDED_RECEIVER_MODE


def intended_receiver_suffix(mode: str, include_original: bool = False) -> str:
    if mode not in INTENDED_RECEIVER_MODES:
        raise ValueError(f"Unsupported intended receiver mode: {mode}")
    if mode == INTENDED_RECEIVER_MODE_ORIGINAL and not include_original:
        return ""
    return f"_{mode}"


def get_action_label_dir(return_type: str = "disc_0.9", intended_receiver_mode: str = DEFAULT_INTENDED_RECEIVER_MODE) -> Path:
    return FEATURE_DIR / f"action_labels_{return_type}{intended_receiver_suffix(intended_receiver_mode)}"


def get_intent_train_label_dir(
    return_type: str = "disc_0.9",
    intended_receiver_mode: str = DEFAULT_INTENDED_RECEIVER_MODE,
) -> Path:
    return FEATURE_DIR / f"action_labels_intent_train_{return_type}{intended_receiver_suffix(intended_receiver_mode)}"


def get_resolved_action_dir(intended_receiver_mode: str = DEFAULT_INTENDED_RECEIVER_MODE) -> Path:
    return FEATURE_DIR / f"resolved_actions{intended_receiver_suffix(intended_receiver_mode, include_original=True)}"


def get_resolved_action_path(match_id: str, intended_receiver_mode: str = DEFAULT_INTENDED_RECEIVER_MODE) -> Path:
    return get_resolved_action_dir(intended_receiver_mode) / f"{match_id}.parquet"


def get_augmented_feature_dir(intended_receiver_mode: str = DEFAULT_INTENDED_RECEIVER_MODE) -> Path:
    return FEATURE_DIR / f"augmented_graphs{intended_receiver_suffix(intended_receiver_mode)}"


def get_augmented_label_dir(intended_receiver_mode: str = DEFAULT_INTENDED_RECEIVER_MODE) -> Path:
    return FEATURE_DIR / f"augmented_labels{intended_receiver_suffix(intended_receiver_mode)}"


def get_component_dir(
    use_xt: bool = False,
    intended_receiver_mode: str = DEFAULT_INTENDED_RECEIVER_MODE,
) -> Path:
    base_name = f"{COMPONENT_DIR.name}{intended_receiver_suffix(intended_receiver_mode)}"
    if use_xt:
        base_name = f"{base_name}_xt"
    return COMPONENT_DIR.with_name(base_name)


def get_relevant_model_ids(
    intended_receiver_mode: str = DEFAULT_INTENDED_RECEIVER_MODE,
    use_xt: bool = False,
) -> dict[str, str]:
    try:
        return RELEVANT_MODEL_IDS[intended_receiver_mode][bool(use_xt)].copy()
    except KeyError as exc:
        raise ValueError(
            f"Missing relevant model ids for mode={intended_receiver_mode!r}, use_xt={bool(use_xt)!r}."
        ) from exc


def _unique_in_order(ids: list[str]) -> list[str]:
    return list(dict.fromkeys(str(match_id) for match_id in ids))


def save_split_manifest(train_ids: list[str], test_ids: list[str], metadata: dict[str, Any] | None = None) -> None:
    ensure_project_dirs()
    payload = {
        "train": _unique_in_order(train_ids),
        "test": _unique_in_order(test_ids),
        "metadata": metadata or {},
    }
    SPLIT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_manifest() -> dict[str, Any]:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split manifest not found at {SPLIT_PATH}. Run scripts/preprocess_sportec.py first."
        )

    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


def filter_available_match_ids(match_ids: list[str], feature_dir: str | Path | None = None) -> np.ndarray:
    if feature_dir is None:
        return np.array(_unique_in_order(match_ids))

    feature_dir = Path(feature_dir)
    available_ids = {path.stem for path in feature_dir.glob("*.pt")}
    return np.array([match_id for match_id in _unique_in_order(match_ids) if match_id in available_ids])


def load_base_splits(feature_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    manifest = load_split_manifest()
    train_ids = filter_available_match_ids(manifest["train"], feature_dir)
    test_ids = filter_available_match_ids(manifest["test"], feature_dir)
    return train_ids, test_ids


def derive_model_train_valid(train_pool_ids: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_pool_ids = np.array(_unique_in_order(list(train_pool_ids)))
    if len(train_pool_ids) < 2:
        raise ValueError(f"Expected at least 2 training-pool matches, found {len(train_pool_ids)}.")

    train_size = int(len(train_pool_ids) * MODEL_TRAIN_FRACTION)
    train_size = min(max(train_size, 1), len(train_pool_ids) - 1)
    train_ids = train_pool_ids[:train_size]
    valid_ids = train_pool_ids[train_size:]
    return train_ids, valid_ids


def load_model_splits(feature_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_pool_ids, test_ids = load_base_splits(feature_dir)
    train_ids, valid_ids = derive_model_train_valid(train_pool_ids)
    return train_ids, valid_ids, test_ids
