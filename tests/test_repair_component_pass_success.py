import numpy as np
import pandas as pd
import pytest

from scripts.repair_component_pass_success import rebuild_v5_table


@pytest.fixture
def inputs():
    model = pd.DataFrame({
        "state_id": ["runtime-a", "runtime-b"], "period_id": [1, 2],
        "state_frame_id": [100, 200], "possessor_id": ["home_1", "away_7"],
        "home_1": [0.6, np.nan], "home_2": [0.8, np.nan], "home_3": [0.0, np.nan],
        "away_7": [np.nan, 0.9], "away_8": [np.nan, 0.4],
    })
    height, intent = model.copy(), model.copy()
    height.loc[0, "home_2"], intent.loc[0, "home_2"] = 0.5, 0.02
    height.loc[1, "away_8"], intent.loc[1, "away_8"] = 0.8, 0.005
    physical = model[["period_id", "state_frame_id", "possessor_id"]].copy()
    physical["state_id"] = ["cache-a", "cache-b"]
    for player in ("home_1", "home_2", "home_3", "away_7", "away_8"):
        physical[player + "__top_pass10_xpass"] = np.nan
    physical.loc[0, "home_2__top_pass10_xpass"] = 1.0
    physical.loc[1, "away_8__top_pass10_xpass"] = 0.0
    offside = np.zeros((2, 5), dtype=bool)
    offside[0, 2] = True
    return model, height, intent, physical.iloc[::-1], offside, np.ones((2, 5))


def rebuild(inputs):
    return rebuild_v5_table(*inputs, metric="top_pass10_xpass", eps=1e-4, floor=None,
                            intent_threshold=0.01, discount=True)


def test_repair_aligns_physical_keys_and_preserves_inference_rules(inputs):
    model = inputs[0].copy(deep=True)
    result, stats = rebuild(inputs)

    assert result.at[0, "home_2"] == 0.5 * float(np.float32(0.9999)) + 0.5 * 0.8
    assert result.at[1, "away_8"] == 0.6 * float(np.float32(0.0001)) + 0.4 * 0.4
    assert result.at[0, "home_1"] == 0.6
    assert result.at[1, "away_7"] == 0.9
    assert result.at[0, "home_3"] == 0.0
    assert result.state_id.tolist() == ["runtime-a", "runtime-b"]
    np.testing.assert_array_equal(result.isna(), model.isna())
    pd.testing.assert_frame_equal(inputs[0], model)
    assert stats == {"rows": 2, "finite_values": 5, "blended_pass_candidates": 2,
                     "offside_zeros": 1, "self_values_preserved": 2, "changed_values": 2}


def test_repair_refuses_incomplete_physics(inputs):
    inputs[3].loc[0, "home_2__top_pass10_xpass"] = np.nan
    with pytest.raises(ValueError, match="Nonfinite xpass"):
        rebuild(inputs)


def test_repair_refuses_misaligned_auxiliary(inputs):
    inputs[1].loc[0, "state_id"] = "wrong-state"
    with pytest.raises(ValueError, match="pass_height identifiers"):
        rebuild(inputs)


def test_repair_refuses_missing_physical_state(inputs):
    missing = (*inputs[:3], inputs[3].iloc[:1], *inputs[4:])
    with pytest.raises(ValueError, match="cover every component state"):
        rebuild(missing)


def test_repair_checks_offside_mask_against_saved_model(inputs):
    inputs[4][0, 1] = True
    with pytest.raises(ValueError, match="offside flags disagree"):
        rebuild(inputs)


def test_repair_preserves_valid_zero_probability_as_a_blend_candidate(inputs):
    inputs[0].loc[0, "home_2"] = 0.0
    result, stats = rebuild(inputs)
    assert result.at[0, "home_2"] == 0.5 * float(np.float32(0.9999))
    assert stats["blended_pass_candidates"] == 2
