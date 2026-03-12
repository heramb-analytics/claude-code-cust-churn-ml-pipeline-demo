"""XGBoost training pipeline for transaction anomaly detection.

Stratified 70/15/15 split with RandomizedSearchCV.
Saves models/pipeline_model.pkl + models/pipeline_model_metrics.json.
"""

from __future__ import annotations

import json
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")

FEATURE_COLS = [
    "amount",
    "log_amount",
    "amount_zscore",
    "amount_vs_merchant_mean",
    "merchant_anomaly_rate",
    "category_anomaly_rate",
    "hour_of_day",
    "day_of_week",
    "minute_of_hour",
    "is_weekend",
    "seconds_since_last_txn",
]
TARGET_COL = "is_anomaly"


def load_features(path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load features and split into X, y.

    Args:
        path: Override path to features.parquet.

    Returns:
        Tuple of (feature matrix, target series).
    """
    if path is None:
        path = PROCESSED_DIR / "features.parquet"
    df = pd.read_parquet(path)
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features]
    y = df[TARGET_COL]
    return X, y


def train(feat_path: Path | None = None) -> dict:
    """Train XGBoost classifier with stratified split and hyperparameter search.

    Args:
        feat_path: Override path to features.parquet.

    Returns:
        Metrics dictionary.
    """
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    X, y = load_features(feat_path)
    n_samples = len(X)

    if n_samples < 10:
        # For tiny datasets train on all, hold-out by index for reporting
        X_train, y_train = X, y
        X_val, y_val = X.iloc[:0], y.iloc[:0]  # empty
        X_test, y_test = X, y  # in-sample evaluation
        print(f"[STAGE 3] Tiny dataset ({n_samples} rows) — using full dataset for train/test")
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
        )
        # Assert zero index overlap
        assert len(set(X_train.index) & set(X_val.index)) == 0, "Train/val overlap!"
        assert len(set(X_train.index) & set(X_test.index)) == 0, "Train/test overlap!"
        assert len(set(X_val.index) & set(X_test.index)) == 0, "Val/test overlap!"

    print(f"[STAGE 3] Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # XGBoost with scale_pos_weight for class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    # RandomizedSearchCV on 3 hyperparameters
    param_dist = {
        "n_estimators": [50, 100, 200],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
    }
    # Use LeaveOneOut for tiny datasets, otherwise StratifiedKFold
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold
    min_class = int(y_train.value_counts().min())
    if min_class < 2 or len(X_train) < 6:
        cv = LeaveOneOut()
        scoring = "accuracy"  # F1 breaks with single-class folds
        n_iter = min(5, len(param_dist["n_estimators"]))
    else:
        cv = StratifiedKFold(n_splits=min(3, min_class))
        scoring = "f1"
        n_iter = 10

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=42,
        n_jobs=-1,
        error_score=0.0,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    # Evaluate
    def evaluate(X_eval: pd.DataFrame, y_eval: pd.Series, split: str) -> dict:
        if len(X_eval) == 0:
            return {}
        y_prob = model.predict_proba(X_eval)[:, 1]
        # Use 0.3 threshold for imbalanced anomaly detection
        threshold = 0.3
        y_pred = (y_prob >= threshold).astype(int)
        metrics = {
            "accuracy": round(float(accuracy_score(y_eval, y_pred)), 4),
            "f1": round(float(f1_score(y_eval, y_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_eval, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_eval, y_pred, zero_division=0)), 4),
        }
        if len(np.unique(y_eval)) > 1:
            metrics["roc_auc"] = round(float(roc_auc_score(y_eval, y_prob)), 4)
        return metrics

    train_metrics = evaluate(X_train, y_train, "train")
    val_metrics = evaluate(X_val, y_val, "val")
    test_metrics = evaluate(X_test, y_test, "test")

    print(f"[STAGE 3] Best params: {search.best_params_}")
    print(f"[STAGE 3] Test metrics: {test_metrics}")

    # Persist model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "pipeline_model.pkl"
    with model_path.open("wb") as fh:
        pickle.dump(
            {
                "model": model,
                "feature_cols": X.columns.tolist(),
                "version": "1.0.0",
            },
            fh,
        )

    metrics = {
        "request_id": request_id,
        "timestamp": timestamp,
        "algorithm": "XGBoostClassifier",
        "version": "1.0.0",
        "best_params": search.best_params_,
        "feature_cols": X.columns.tolist(),
        "split_sizes": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "primary_metric": "f1",
        "primary_metric_value": test_metrics.get("f1", val_metrics.get("f1", 0)),
    }

    metrics_path = MODELS_DIR / "pipeline_model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[STAGE 3] Saved model → {model_path} | metrics → {metrics_path}")
    return metrics


if __name__ == "__main__":
    train()
