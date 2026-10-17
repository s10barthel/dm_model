"""Shared conventions for the separately exported checkpoint prediction."""
from pathlib import Path

import pandas as pd


MODEL_PASS_SUCCESS_DEFINITION = (
    "Selected checkpoint probability after normal offside handling and before "
    "inference-time physical xPass blending; checkpoint-internal physical inputs are retained."
)


def require_model_pass_success(components, requested=True):
    """Check structural availability without rejecting legitimately missing values."""
    if requested and "model_pass_success" not in components:
        raise ValueError(
            "This component run does not contain model_pass_success predictions. "
            "Regenerate the source component run with the updated inference scripts "
            "before using --show-model-pass-success."
        )


def model_pass_success_metadata(*, requested=False, available=True, output_root=None):
    metadata = {
        "show_model_pass_success": bool(requested),
        "model_pass_success_definition": MODEL_PASS_SUCCESS_DEFINITION,
        "model_pass_success_available": bool(available),
    }
    if output_root is not None:
        metadata["model_pass_success_output_paths"] = [
            str(path.resolve()) for path in sorted(Path(output_root).rglob("model_pass_success*"))
            if path.is_file()
        ]
    return metadata


def model_pass_success_row(tables, frame_id):
    """Return a row with missing candidates intact, including an unavailable frame."""
    table = tables["model_pass_success"]
    if frame_id in table.index:
        return pd.to_numeric(table.loc[frame_id], errors="raise").astype(float)
    candidates = table.columns
    if not len(candidates):
        candidates = sorted({column for value in tables.values() if isinstance(value, pd.DataFrame)
                             for column in value.columns})
    return pd.Series(float("nan"), index=candidates, dtype=float)
