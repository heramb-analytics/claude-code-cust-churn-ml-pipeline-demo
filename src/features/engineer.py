"""Feature engineering for transaction anomaly detection.

Reads data/processed/clean.parquet, engineers features, saves
data/processed/features.parquet + data/processed/feature_schema.json.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
LOGS_DIR = Path("logs")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build ML-ready feature matrix from cleaned transactions.

    Args:
        df: Cleaned transactions DataFrame with timestamp as datetime.

    Returns:
        DataFrame with engineered numeric and encoded features.
    """
    feat = df.copy()

    # ── Temporal features ──────────────────────────────────────────────────
    feat["hour_of_day"] = feat["timestamp"].dt.hour
    feat["day_of_week"] = feat["timestamp"].dt.dayofweek
    feat["minute_of_hour"] = feat["timestamp"].dt.minute
    feat["is_weekend"] = (feat["day_of_week"] >= 5).astype(int)

    # ── Amount-based features ───────────────────────────────────────────────
    feat["log_amount"] = np.log1p(feat["amount"])
    feat["amount_zscore"] = (feat["amount"] - feat["amount"].mean()) / (
        feat["amount"].std() + 1e-9
    )

    # ── Merchant risk profile ───────────────────────────────────────────────
    merchant_anomaly_rate = (
        feat.groupby("merchant_id")["is_anomaly"].mean().rename("merchant_anomaly_rate")
    )
    feat = feat.join(merchant_anomaly_rate, on="merchant_id")

    # ── Category encoding (ordinal by anomaly rate) ─────────────────────────
    category_anomaly_rate = (
        feat.groupby("category")["is_anomaly"].mean().rename("category_anomaly_rate")
    )
    feat = feat.join(category_anomaly_rate, on="category")

    # ── Rolling amount statistics (merchant-level) ──────────────────────────
    # For small datasets we fall back to global stats
    feat = feat.sort_values("timestamp")
    merchant_mean_amount = (
        feat.groupby("merchant_id")["amount"].transform("mean")
    )
    feat["amount_vs_merchant_mean"] = feat["amount"] / (merchant_mean_amount + 1e-9)

    # ── Transaction velocity feature (transactions per minute) ─────────────
    time_diff = feat["timestamp"].diff().dt.total_seconds().fillna(60)
    feat["seconds_since_last_txn"] = time_diff

    return feat


def save_features(df: pd.DataFrame) -> dict:
    """Persist feature DataFrame and schema.

    Args:
        df: Feature-engineered DataFrame.

    Returns:
        Feature schema dictionary.
    """
    # Identify feature columns (exclude id + target cols)
    exclude = {"transaction_id", "timestamp", "merchant_id", "category", "is_anomaly"}
    feature_cols = [c for c in df.columns if c not in exclude]

    schema = {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_features": len(feature_cols),
        "feature_columns": feature_cols,
        "target_column": "is_anomaly",
        "id_columns": ["transaction_id"],
        "dtypes": {col: str(df[col].dtype) for col in feature_cols},
        "null_counts": {col: int(df[col].isnull().sum()) for col in feature_cols},
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features_path = PROCESSED_DIR / "features.parquet"
    df.to_parquet(features_path, index=False)

    schema_path = PROCESSED_DIR / "feature_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2))

    return schema


def run(input_path: Path | None = None) -> pd.DataFrame:
    """Execute feature engineering pipeline.

    Args:
        input_path: Override path to clean.parquet.

    Returns:
        Feature-engineered DataFrame.
    """
    if input_path is None:
        input_path = PROCESSED_DIR / "clean.parquet"

    df = pd.read_parquet(input_path)
    feat_df = engineer_features(df)
    schema = save_features(feat_df)

    print(
        f"[STAGE 2A] Features engineered: {schema['total_features']} features | "
        f"{len(feat_df)} rows"
    )
    return feat_df


if __name__ == "__main__":
    run()
