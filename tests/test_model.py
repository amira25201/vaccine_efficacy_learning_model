"""Training and recommending, on synthetic data so no real records are needed."""

import numpy as np
import pandas as pd
import pytest

import config
from make_demo_data import write_demo_spreadsheet
from vaccine_data import check_groups, load_dataset
from vaccine_model import ModelBundle, rank_vaccines, train_target


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = write_demo_spreadsheet(tmp_path_factory.mktemp("data") / "demo.xlsx", n_studies=25)
    return load_dataset(path)


@pytest.fixture(scope="module")
def bundle(dataset):
    return train_target(dataset, "postchallenge_1dose", quick=True)


def test_cleaning_parses_messy_cells_and_rescales_targets(dataset):
    assert dataset["cfu"].notna().all(), "every challenge dose should parse"
    assert dataset["cfu"].min() > 1000, "bare powers of ten must not collapse to small numbers"
    assert dataset["dosage_micrograms"].notna().all()
    assert dataset["postchallenge_1dose"].max() <= 1.0, "percentages should be rescaled to proportions"
    assert dataset.attrs["target_scales"]["postchallenge_1dose"] == "percent"
    assert "plague_type" in dataset.columns, "the stray space in the header should be cleaned"


def test_grouping_is_validated(dataset):
    assert "groups" in check_groups(dataset, "id")
    with pytest.raises(KeyError):
        check_groups(dataset, "not_a_column")
    one_per_row = dataset.assign(row_id=range(len(dataset)))
    assert check_groups(one_per_row, "row_id").startswith("WARNING")


def test_the_model_is_fitted_only_on_feature_columns(bundle):
    fitted_on = list(bundle.pipeline.named_steps["prep"].feature_names_in_)
    assert fitted_on == bundle.features
    for leaked in ("id", "reference", "postchallenge_2doses"):
        assert leaked not in fitted_on


def test_test_rows_come_from_unseen_groups(dataset):
    b = train_target(dataset, "postchallenge_1dose", quick=True)
    assert b.metrics["n_groups_test"] >= 1
    assert b.metrics["n_train"] > b.metrics["n_test"]


def test_metrics_include_a_baseline_comparison(bundle):
    m = bundle.metrics
    assert {"test_rmse", "baseline_rmse", "beats_baseline", "test_r2", "cv_rmse"} <= set(m)
    assert m["beats_baseline"] == (m["test_rmse"] < m["baseline_rmse"])


def test_a_constant_target_is_refused(dataset):
    """The MATLAB bug turned the outcome into a flag; on a constant target, refuse to train."""
    flat = dataset.copy()
    flat["postchallenge_1dose"] = 1.0
    with pytest.raises(ValueError, match="nothing to predict"):
        train_target(flat, "postchallenge_1dose", quick=True)


def test_recommendations_cover_training_vaccines_and_are_sorted(bundle):
    ranked = rank_vaccines(bundle, {"mice_breed": "balb/c", "strain": "co92", "plague_type": "bubonic", "cfu": "1e6", "dosage_micrograms": "3 µg"})
    assert set(ranked.vaccine) == set(bundle.vaccines)
    assert ranked.predicted_survival.is_monotonic_decreasing
    assert ranked.predicted_survival.between(0, 1).all()


def test_messy_profile_input_is_parsed_like_the_training_data(bundle):
    profile = {"mice_breed": " BALB/c ", "strain": "CO92", "plague_type": "Bubonic", "cfu": "10⁶", "dosage_micrograms": "3 µg"}
    plain = {"mice_breed": "balb/c", "strain": "co92", "plague_type": "bubonic", "cfu": 1e6, "dosage_micrograms": 3.0}
    pd.testing.assert_frame_equal(rank_vaccines(bundle, profile), rank_vaccines(bundle, plain))


def test_bundle_round_trips_through_disk(bundle, tmp_path):
    path = tmp_path / "b.joblib"
    bundle.save(path)
    loaded = ModelBundle.load(path)
    assert loaded.vaccines == bundle.vaccines
    assert loaded.features == bundle.features
    assert loaded.target_scale == bundle.target_scale


def test_loading_a_bare_pipeline_is_rejected(bundle, tmp_path):
    """Saved files from the old script held a bare pipeline with no vaccine list."""
    import joblib

    path = tmp_path / "old.joblib"
    joblib.dump(bundle.pipeline, path)
    with pytest.raises(TypeError, match="older version"):
        ModelBundle.load(path)
