from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA_ROOT = PROJECT_ROOT / "Bundesliga_season_23_24"
RAW_TRACKING_DIR = RAW_DATA_ROOT / "tracking_data"
RAW_EVENT_DIR = RAW_DATA_ROOT / "event_data"
RAW_META_DIR = RAW_DATA_ROOT / "match_information"

DATA_ROOT = PROJECT_ROOT / "data" / "ajax"
LINEUP_DIR = DATA_ROOT / "lineup"
LINEUP_PATH = LINEUP_DIR / "line_up.parquet"
EVENT_DIR = DATA_ROOT / "event"
EVENT_PATH = EVENT_DIR / "event.parquet"
TRACKING_DIR = DATA_ROOT / "tracking"
TRACKING_PROCESSED_DIR = DATA_ROOT / "tracking_processed"
EVENT_SYNCED_DIR = DATA_ROOT / "event_synced"
FEATURE_DIR = DATA_ROOT / "features"
SAVED_DIR = PROJECT_ROOT / "saved"
COMPONENT_DIR = DATA_ROOT / "defcon_components"
SPLIT_DIR = DATA_ROOT / "splits"
SPLIT_PATH = SPLIT_DIR / "match_splits.json"

TRAIN_POOL_SIZE = 245
MODEL_TRAIN_SIZE = 200
SPLIT_SEED = 100


def ensure_project_dirs() -> None:
    for path in [
        DATA_ROOT,
        LINEUP_DIR,
        EVENT_DIR,
        TRACKING_DIR,
        TRACKING_PROCESSED_DIR,
        EVENT_SYNCED_DIR,
        FEATURE_DIR,
        COMPONENT_DIR,
        SAVED_DIR,
        SPLIT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


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
    train_pool_ids = np.array(sorted(_unique_in_order(list(train_pool_ids))))
    if len(train_pool_ids) < MODEL_TRAIN_SIZE:
        raise ValueError(
            f"Expected at least {MODEL_TRAIN_SIZE} training-pool matches, found {len(train_pool_ids)}."
        )

    rng = np.random.default_rng(SPLIT_SEED)
    sampled = np.sort(rng.choice(train_pool_ids, MODEL_TRAIN_SIZE, replace=False))
    valid = np.array([match_id for match_id in train_pool_ids if match_id not in set(sampled)])
    return sampled, valid


def load_model_splits(feature_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_pool_ids, test_ids = load_base_splits(feature_dir)
    train_ids, valid_ids = derive_model_train_valid(train_pool_ids)
    return train_ids, valid_ids, test_ids
