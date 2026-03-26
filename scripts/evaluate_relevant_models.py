from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = [
    "action_intent/00",
    "pass_success/20",
    "outcome_scoring/20",
    "outcome_conceding/20",
]


def main() -> None:
    python = sys.executable
    for model_id in MODEL_IDS:
        command = [python, "test.py", "--model_id", model_id]
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
