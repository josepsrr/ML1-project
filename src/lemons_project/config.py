from __future__ import annotations

from pathlib import Path

RANDOM_STATE = 42
DATA_URL = "https://zenodo.org/records/4252636/files/dataset.csv?download=1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"

DATA_PATH = DATA_RAW_DIR / "dataset.csv"
SUMMARY_PATH = OUTPUTS_DIR / "summary.json"

SAMPLE_SIZE = 60000
INFERENCE_SAMPLE_SIZE = 40000

STRUCTURED_NUMERIC_COLUMNS = [
    "log_km",
    "vehicle_age",
    "power_cv",
]

STRUCTURED_CATEGORICAL_COLUMNS = [
    "advertizer_type_clean",
    "region",
    "brand",
]

LEXICON_COLUMNS = [
    "maintenance_mentions",
    "transparency_mentions",
    "defect_mentions",
    "promotional_mentions",
    "token_count",
    "number_count",
    "disclosure_index_z",
]
