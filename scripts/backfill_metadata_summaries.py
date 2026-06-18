from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datatools.metadata_summary import refresh_all_summaries


def main() -> None:
    summary_paths = refresh_all_summaries()
    for path in summary_paths:
        print(path)
    print(f"Rebuilt {len(summary_paths)} metadata summary CSV file(s).")


if __name__ == "__main__":
    main()
