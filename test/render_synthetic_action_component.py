from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.visualize_action_components import render_component


def _build_tracking(n_frames: int = 25) -> pd.DataFrame:
    frames = np.arange(n_frames)
    data: dict[str, np.ndarray] = {}

    home_base_x = np.array([42, 48, 52, 56, 60, 63, 66, 69, 72, 76, 80], dtype=float)
    home_base_y = np.array([14, 24, 44, 54, 32, 18, 48, 36, 28, 42, 22], dtype=float)
    away_base_x = np.array([34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 30], dtype=float)
    away_base_y = np.array([12, 22, 46, 56, 30, 16, 50, 40, 26, 34, 58], dtype=float)

    for idx in range(11):
        player = idx + 1
        hx = home_base_x[idx] + frames * (0.10 + idx * 0.01)
        hy = home_base_y[idx] + np.sin(frames / 4 + idx * 0.35) * 1.2
        data[f"home_{player}_x"] = hx
        data[f"home_{player}_y"] = hy
        data[f"home_{player}_vx"] = np.full(n_frames, 0.10 + idx * 0.01)
        data[f"home_{player}_vy"] = np.gradient(hy)

        ax = away_base_x[idx] - frames * (0.06 + idx * 0.005)
        ay = away_base_y[idx] + np.cos(frames / 5 + idx * 0.25) * 1.0
        data[f"away_{player}_x"] = ax
        data[f"away_{player}_y"] = ay
        data[f"away_{player}_vx"] = np.full(n_frames, -(0.06 + idx * 0.005))
        data[f"away_{player}_vy"] = np.gradient(ay)

    data["ball_x"] = np.linspace(50.0, 74.0, n_frames)
    data["ball_y"] = np.linspace(34.0, 38.0, n_frames)

    tracking = pd.DataFrame(data, index=frames)
    tracking.index.name = "frame_id"
    return tracking


def _build_match() -> SimpleNamespace:
    tracking = _build_tracking()
    actions = pd.DataFrame(
        [
            {
                "frame_id": int(tracking.index.max()),
                "object_id": "home_8",
                "spadl_type": "pass",
            }
        ],
        index=[0],
    )
    return SimpleNamespace(actions=actions, tracking=tracking)


def _build_probs() -> pd.Series:
    values = {
        "home_1": 0.08,
        "home_2": 0.05,
        "home_3": 0.07,
        "home_4": 0.04,
        "home_5": 0.10,
        "home_6": 0.14,
        "home_7": 0.12,
        "home_8": 0.18,
        "home_9": 0.31,
        "home_10": 0.22,
        "home_11": 0.16,
    }
    return pd.Series(values, dtype=float)


def main() -> None:
    output_path = Path("test") / "synthetic_action_intent.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    match = _build_match()
    probs = _build_probs()

    render_component(
        match=match,
        action_id=0,
        component_name="action_intent",
        probs=probs,
        output_path=output_path,
    )

    print(output_path.resolve())


if __name__ == "__main__":
    main()
