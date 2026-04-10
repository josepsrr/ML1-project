from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lemons_project.config import DATA_PATH, DATA_RAW_DIR, DATA_URL


def main() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        print(f"Dataset already exists at {DATA_PATH}")
        return

    print(f"Downloading dataset from {DATA_URL}")
    urlretrieve(DATA_URL, DATA_PATH)
    print(f"Saved dataset to {DATA_PATH}")


if __name__ == "__main__":
    main()
