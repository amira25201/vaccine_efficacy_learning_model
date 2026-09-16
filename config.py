"""Paths, column names and model settings for the vaccine recommender.

Nothing here has side effects at import time, so the modules can be imported
by the tests without touching the filesystem.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "vaccine_data_with_all_doses.xlsx"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
SEARCH_ITERATIONS = 30

# Column holding the experiment/animal identifier. Rows sharing a value must
# never be split across training and test, or the model is scored on animals
# it has already seen. Verify this against your spreadsheet: if every row has
# its own value, grouping does nothing and the scores will be optimistic.
GROUP_COLUMN = "id"

# Outcome columns: survival after challenge, one per dose count.
TARGETS = {"Dose 1": "postchallenge_1dose", "Dose 2": "postchallenge_2doses"}

CATEGORICAL_FEATURES = ["vaccine", "strain", "mice_breed", "plague_type"]
NUMERIC_FEATURES = ["cfu", "dosage_micrograms"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

SEARCH_SPACE = {
    "model__n_estimators": [400, 600, 800, 1000],
    "model__max_depth": [None, 8, 12, 16],
    "model__min_samples_leaf": [1, 2, 4, 8],
    "model__max_features": ["sqrt", "log2", 0.6, 0.8, 1.0],
}

# Smaller search for `train.py --quick`, used by the tests and for a fast check.
QUICK_SEARCH_SPACE = {
    "model__n_estimators": [100],
    "model__max_depth": [None, 8],
    "model__min_samples_leaf": [1, 4],
}
QUICK_SEARCH_ITERATIONS = 4


def model_path(target: str) -> Path:
    return MODELS_DIR / f"rf_{target}.joblib"
