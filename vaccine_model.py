"""Training, evaluating and using the per-dose vaccine models.

A trained model is saved as a bundle: the fitted pipeline plus the facts
needed to use it correctly later, including the exact feature columns and
the list of vaccines seen during training. The original script re-derived
that list from the full dataset whenever it loaded a saved model, which
quietly widened the recommendation set; carrying it in the bundle removes
the ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import config
from vaccine_data import parse_quantity


@dataclass
class ModelBundle:
    """A fitted pipeline and everything needed to use it correctly."""

    target: str
    pipeline: Pipeline
    features: list[str]
    categorical: list[str]
    numeric: list[str]
    vaccines: list[str]
    target_scale: str
    metrics: dict = field(default_factory=dict)
    trained_at: str = ""
    group_note: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "ModelBundle":
        bundle = joblib.load(path)
        if not isinstance(bundle, ModelBundle):
            raise TypeError(
                f"{path} holds a bare pipeline from an older version of this project. "
                f"Delete it and run train.py again."
            )
        return bundle


def build_pipeline(categorical: list[str], numeric: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("prep", preprocessor), ("model", RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1))])


def available_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [c for c in config.CATEGORICAL_FEATURES if c in df.columns]
    numeric = [c for c in config.NUMERIC_FEATURES if c in df.columns]
    if not categorical and not numeric:
        raise KeyError(f"None of the expected feature columns are present. Looked for: {config.FEATURES}")
    return categorical, numeric


def train_target(
    df: pd.DataFrame,
    target: str,
    *,
    group_column: str | None = None,
    quick: bool = False,
    group_note: str = "",
) -> ModelBundle:
    """Fit one model for one dose count, scored on held-out groups."""
    group_column = group_column or config.GROUP_COLUMN
    if target not in df.columns:
        raise KeyError(f"Target column {target!r} is not in the spreadsheet.")

    data = df.dropna(subset=[target]).copy()
    if data.empty:
        raise ValueError(f"Every row is missing {target!r}; nothing to train on.")

    categorical, numeric = available_features(data)
    features = categorical + numeric
    # Only the feature columns are passed to the pipeline, so the frame it is
    # fitted on matches the frame the interface sends at prediction time.
    X = data[features]
    y = data[target].astype(float).to_numpy()
    groups = data[group_column].astype(str)

    if y.std() == 0:
        raise ValueError(
            f"{target!r} has the same value in every row, so there is nothing to predict. "
            f"This usually means the column was converted to a flag by mistake."
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups.iloc[train_idx]

    n_splits = min(config.CV_SPLITS, groups_train.nunique())
    if n_splits < 2:
        raise ValueError("The training set has fewer than two groups; cannot run grouped cross-validation.")

    search = RandomizedSearchCV(
        estimator=build_pipeline(categorical, numeric),
        param_distributions=config.QUICK_SEARCH_SPACE if quick else config.SEARCH_SPACE,
        n_iter=config.QUICK_SEARCH_ITERATIONS if quick else config.SEARCH_ITERATIONS,
        scoring="neg_root_mean_squared_error",
        cv=GroupKFold(n_splits=n_splits),
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        error_score="raise",
    )
    search.fit(X_train, y_train, groups=groups_train)
    best = search.best_estimator_

    predicted = best.predict(X_test)
    # The honest comparison: predicting the training mean for everyone.
    baseline = np.full_like(y_test, float(np.mean(y_train)))
    metrics = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_groups_train": int(groups_train.nunique()),
        "n_groups_test": int(groups.iloc[test_idx].nunique()),
        "cv_rmse": float(-search.best_score_),
        "test_rmse": float(root_mean_squared_error(y_test, predicted)),
        "baseline_rmse": float(root_mean_squared_error(y_test, baseline)),
        "test_mae": float(mean_absolute_error(y_test, predicted)),
        "test_r2": float(r2_score(y_test, predicted)),
        "beats_baseline": bool(root_mean_squared_error(y_test, predicted) < root_mean_squared_error(y_test, baseline)),
        "best_params": {k: v for k, v in search.best_params_.items()},
    }

    scales = df.attrs.get("target_scales", {})
    return ModelBundle(
        target=target,
        pipeline=best,
        features=features,
        categorical=categorical,
        numeric=numeric,
        # Recommend only vaccines the model was actually trained on.
        vaccines=sorted(X_train["vaccine"].dropna().astype(str).unique()) if "vaccine" in X_train.columns else [],
        target_scale=scales.get(target, "unknown"),
        metrics=metrics,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        group_note=group_note,
    )


def rank_vaccines(bundle: ModelBundle, profile: dict) -> pd.DataFrame:
    """
    Predict survival for each candidate vaccine under one profile.

    Returns a frame sorted best first. Values are clipped to 0-1 only when
    the target really is a proportion.
    """
    if not bundle.vaccines:
        raise ValueError("This model carries no vaccine list, so nothing can be ranked.")

    row = {}
    for column in bundle.features:
        value = profile.get(column)
        if column in bundle.numeric:
            row[column] = parse_quantity(value)
        elif column != "vaccine":
            row[column] = None if value is None else " ".join(str(value).split()).lower()

    frame = pd.DataFrame([{**row, "vaccine": v} for v in bundle.vaccines])[bundle.features]
    predictions = bundle.pipeline.predict(frame)
    if bundle.target_scale in {"proportion", "percent"}:
        predictions = np.clip(predictions, 0.0, 1.0)

    return (
        pd.DataFrame({"vaccine": bundle.vaccines, "predicted_survival": predictions})
        .sort_values("predicted_survival", ascending=False)
        .reset_index(drop=True)
    )
