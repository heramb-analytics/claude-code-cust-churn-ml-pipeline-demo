"""Data validation layer: 12 checks on features.parquet.

Saves logs/validation_report.json.
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


class DataQualityError(Exception):
    """Raised when a critical validation check fails."""


def run_validation_checks(feat_path: Path | None = None) -> dict:
    """Execute 12 validation checks on the features DataFrame.

    Args:
        feat_path: Override path to features.parquet.

    Returns:
        Validation report dictionary.

    Raises:
        DataQualityError: On critical failure.
    """
    if feat_path is None:
        feat_path = PROCESSED_DIR / "features.parquet"

    df = pd.read_parquet(feat_path)
    results: list[dict] = []
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    def _chk(name: str, passed: bool, detail: str, critical: bool = False) -> None:
        results.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed and critical:
            raise DataQualityError(f"Critical validation failed — {name}: {detail}")

    # 1. features.parquet exists and is non-empty
    _chk("features_file_exists", feat_path.exists(), str(feat_path), critical=True)
    _chk("features_not_empty", len(df) > 0, f"Rows: {len(df)}", critical=True)

    # 3. Target column present
    _chk("target_column_present", "is_anomaly" in df.columns, "is_anomaly check")

    # 4. log_amount feature present
    _chk("log_amount_feature_present", "log_amount" in df.columns, "log_amount check")

    # 5. No infinite values in numeric columns
    num_df = df.select_dtypes(include=[np.number])
    inf_count = np.isinf(num_df.values).sum()
    _chk("no_infinite_values", inf_count == 0, f"Infinite values: {inf_count}")

    # 6. log_amount >= 0 for all rows (log1p(amount) ≥ 0 when amount ≥ 0)
    if "log_amount" in df.columns:
        neg_log = (df["log_amount"] < 0).sum()
        _chk("log_amount_non_negative", neg_log == 0, f"Negative log_amount rows: {neg_log}")
    else:
        _chk("log_amount_non_negative", False, "Column missing")

    # 7. amount_zscore column present
    _chk("amount_zscore_present", "amount_zscore" in df.columns, "amount_zscore check")

    # 8. merchant_anomaly_rate in [0, 1]
    if "merchant_anomaly_rate" in df.columns:
        out = ((df["merchant_anomaly_rate"] < 0) | (df["merchant_anomaly_rate"] > 1)).sum()
        _chk("merchant_anomaly_rate_valid_range", out == 0, f"Out-of-range: {out}")
    else:
        _chk("merchant_anomaly_rate_valid_range", False, "Column missing")

    # 9. hour_of_day in [0, 23]
    if "hour_of_day" in df.columns:
        bad_hours = ((df["hour_of_day"] < 0) | (df["hour_of_day"] > 23)).sum()
        _chk("hour_of_day_valid_range", bad_hours == 0, f"Invalid hours: {bad_hours}")
    else:
        _chk("hour_of_day_valid_range", False, "Column missing")

    # 10. day_of_week in [0, 6]
    if "day_of_week" in df.columns:
        bad_days = ((df["day_of_week"] < 0) | (df["day_of_week"] > 6)).sum()
        _chk("day_of_week_valid_range", bad_days == 0, f"Invalid days: {bad_days}")
    else:
        _chk("day_of_week_valid_range", False, "Column missing")

    # 11. No null values in numeric feature columns
    num_nulls = num_df.isnull().sum().sum()
    _chk("no_nulls_in_numeric_features", num_nulls == 0, f"Null count: {num_nulls}")

    # 12. At least 2 feature columns beyond target
    feature_cols = [c for c in df.columns if c not in {"transaction_id", "timestamp", "is_anomaly"}]
    _chk(
        "sufficient_feature_columns",
        len(feature_cols) >= 2,
        f"Feature columns found: {len(feature_cols)}",
    )

    passed = sum(1 for r in results if r["passed"])
    report = {
        "request_id": request_id,
        "timestamp": timestamp,
        "checks_total": len(results),
        "checks_passed": passed,
        "checks_failed": len(results) - passed,
        "results": results,
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = LOGS_DIR / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[STAGE 2C] Validation: {passed}/{len(results)} checks passed → {report_path}")
    return report


if __name__ == "__main__":
    run_validation_checks()
