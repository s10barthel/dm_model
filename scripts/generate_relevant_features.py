from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_command(args: list[str]) -> None:
    print("Running:", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    python = sys.executable
    run_command([python, "datatools/graph_feature.py", "--action_type", "all", "--split", "train"])
    run_command([python, "datatools/graph_feature.py", "--action_type", "all", "--split", "test"])
    run_command(
        [
            python,
            "datatools/graph_feature.py",
            "--action_type",
            "all",
            "--split",
            "train",
            "--feature_variant",
            "intent_train_augmented",
        ]
    )


if __name__ == "__main__":
    main()
