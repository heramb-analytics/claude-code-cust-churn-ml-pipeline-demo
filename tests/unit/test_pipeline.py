"""Unit tests for the transaction anomaly detection pipeline.

8 tests: model_load, predict_schema, metric_threshold, data_leakage,
latency_under_500ms, invalid_input_raises, output_range, determinism.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")
LOGS_DIR = Path("logs")

MODEL_PATH = MODELS_DIR / "pipeline_model.pkl"
METRICS_PATH = MODELS_DIR / "pipeline_model_metrics.json"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"


@pytest.fixture(scope="module")
def model_bundle():
    """Load the persisted model bundle once for all tests."""
    with MODEL_PATH.open("rb") as fh:
        return pickle.load(fh)


@pytest.fixture(scope="module")
def sample_features(model_bundle):
    """Return a single-row feature DataFrame for inference tests."""
    df = pd.read_parquet(FEATURES_PATH)
    feature_cols = model_bundle["feature_cols"]
    return df[feature_cols].iloc[:1]


@pytest.fixture(scope="module")
def all_features(model_bundle):
    """Return all feature rows."""
    df = pd.read_parquet(FEATURES_PATH)
    feature_cols = model_bundle["feature_cols"]
    return df[feature_cols]


# ── Test 1: model loads without error ────────────────────────────────────────
def test_model_load():
    """Model file exists and deserializes cleanly."""
    assert MODEL_PATH.exists(), f"Model file not found: {MODEL_PATH}"
    with MODEL_PATH.open("rb") as fh:
        bundle = pickle.load(fh)
    assert "model" in bundle, "Bundle missing 'model' key"
    assert "feature_cols" in bundle, "Bundle missing 'feature_cols' key"
    assert len(bundle["feature_cols"]) > 0, "feature_cols is empty"


# ── Test 2: predict returns correct schema ────────────────────────────────────
def test_predict_schema(model_bundle, sample_features):
    """Prediction output has expected shape and dtype."""
    model = model_bundle["model"]
    preds = model.predict(sample_features)
    assert len(preds) == 1, f"Expected 1 prediction, got {len(preds)}"
    assert preds[0] in (0, 1), f"Unexpected prediction value: {preds[0]}"

    probas = model.predict_proba(sample_features)
    assert probas.shape == (1, 2), f"Unexpected proba shape: {probas.shape}"
    assert abs(probas[0].sum() - 1.0) < 1e-5, "Probabilities do not sum to 1"


# ── Test 3: metric threshold ──────────────────────────────────────────────────
def test_metric_threshold():
    """Model metrics file exists and F1 ≥ 0.40 (threshold for small dataset)."""
    assert METRICS_PATH.exists(), f"Metrics file not found: {METRICS_PATH}"
    with METRICS_PATH.open() as fh:
        metrics = json.load(fh)
    f1 = metrics["test_metrics"].get("f1", 0.0)
    assert f1 >= 0.40, f"F1 score {f1:.4f} below threshold 0.40"


# ── Test 4: data leakage check ────────────────────────────────────────────────
def test_data_leakage():
    """Target column is not in the feature columns used for training."""
    with MODEL_PATH.open("rb") as fh:
        bundle = pickle.load(fh)
    feature_cols = bundle["feature_cols"]
    assert "is_anomaly" not in feature_cols, "Target 'is_anomaly' found in feature_cols!"
    assert "transaction_id" not in feature_cols, "ID column in feature_cols!"


# ── Test 5: latency under 500ms ───────────────────────────────────────────────
def test_latency_under_500ms(model_bundle, all_features):
    """Single prediction completes in under 500ms."""
    model = model_bundle["model"]
    row = all_features.iloc[:1]
    start = time.perf_counter()
    model.predict(row)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 500, f"Prediction took {elapsed_ms:.1f}ms (limit: 500ms)"


# ── Test 6: invalid input raises ─────────────────────────────────────────────
def test_invalid_input_raises(model_bundle):
    """Passing wrong column names raises an error."""
    model = model_bundle["model"]
    bad_df = pd.DataFrame({"wrong_col": [999.0], "another_wrong": [1.0]})
    with pytest.raises(Exception):
        model.predict(bad_df)


# ── Test 7: output range ──────────────────────────────────────────────────────
def test_output_range(model_bundle, all_features):
    """All prediction probabilities in [0, 1] and labels in {0, 1}."""
    model = model_bundle["model"]
    probas = model.predict_proba(all_features)
    assert (probas >= 0).all(), "Negative probability found"
    assert (probas <= 1).all(), "Probability > 1 found"
    preds = model.predict(all_features)
    assert set(preds).issubset({0, 1}), f"Unexpected prediction labels: {set(preds)}"


# ── Test 8: determinism ───────────────────────────────────────────────────────
def test_determinism(model_bundle, all_features):
    """Same input produces identical predictions across multiple calls."""
    model = model_bundle["model"]
    preds_1 = model.predict(all_features)
    preds_2 = model.predict(all_features)
    np.testing.assert_array_equal(preds_1, preds_2, err_msg="Predictions not deterministic!")
