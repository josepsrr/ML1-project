from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lemons_project.config import DATA_PATH
from lemons_project.data import clean_dataset, load_raw_dataset
from lemons_project.features import add_text_features
from lemons_project.modeling import run_analysis


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected dataset at {DATA_PATH}. Run `python scripts/download_data.py` first."
        )

    raw = load_raw_dataset(DATA_PATH)
    clean = clean_dataset(raw)
    featured = add_text_features(clean)
    summary = run_analysis(featured)

    print("Pipeline finished successfully.")
    print(f"Clean listings: {summary['dataset_rows_after_cleaning']}")
    print(f"Summary file: {PROJECT_ROOT / 'outputs' / 'summary.json'}")


if __name__ == "__main__":
    main()
