"""Transaction data ingestion and validation pipeline.

Reads raw transaction CSV, runs 10 quality assertions, and outputs
clean.parquet + logs/quality_report.json.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
LOGS_DIR = Path("logs")

REQUIRED_COLUMNS = [
    "transaction_id",
    "timestamp",
    "merchant_id",
    "amount",
    "category",
    "is_anomaly",
]


class DataQualityError(Exception):
    """Raised when a critical data quality assertion fails."""


def _log_event(event: dict) -> None:
    """Append a JSON-Lines entry to logs/audit.jsonl.

    Args:
        event: Dictionary to serialize as one JSON line.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = LOGS_DIR / "audit.jsonl"
    with audit_path.open("a") as fh:
        fh.write(json.dumps(event) + "\n")


def run_quality_assertions(df: pd.DataFrame) -> list[dict]:
    """Execute 10 data quality checks on the raw DataFrame.

    Args:
        df: Raw transactions DataFrame.

    Returns:
        List of assertion result dicts (name, passed, detail).

    Raises:
        DataQualityError: If a critical assertion fails.
    """
    results: list[dict] = []

    def _record(name: str, passed: bool, detail: str, critical: bool = False) -> None:
        results.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed and critical:
            raise DataQualityError(f"Critical check failed — {name}: {detail}")

    # 1. Required columns present
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    _record(
        "required_columns_present",
        not missing,
        f"Missing: {missing}" if missing else "All required columns present",
        critical=True,
    )

    # 2. No empty DataFrame
    _record(
        "dataframe_not_empty",
        len(df) > 0,
        f"Row count: {len(df)}",
        critical=True,
    )

    # 3. No duplicate transaction IDs
    dup_count = df["transaction_id"].duplicated().sum()
    _record(
        "no_duplicate_transaction_ids",
        dup_count == 0,
        f"Duplicates found: {dup_count}",
    )

    # 4. amount is positive
    non_positive = (df["amount"] <= 0).sum()
    _record(
        "amount_positive",
        non_positive == 0,
        f"Non-positive amounts: {non_positive}",
    )

    # 5. is_anomaly is binary (0 or 1)
    invalid_labels = (~df["is_anomaly"].isin([0, 1])).sum()
    _record(
        "is_anomaly_binary",
        invalid_labels == 0,
        f"Invalid label values: {invalid_labels}",
        critical=True,
    )

    # 6. No null values in critical columns
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    total_nulls = null_counts.sum()
    _record(
        "no_nulls_in_required_columns",
        total_nulls == 0,
        f"Null counts: {null_counts.to_dict()}",
    )

    # 7. timestamp is parseable
    try:
        pd.to_datetime(df["timestamp"])
        ts_ok = True
        ts_detail = "All timestamps parseable"
    except Exception as exc:
        ts_ok = False
        ts_detail = str(exc)
    _record("timestamp_parseable", ts_ok, ts_detail)

    # 8. amount within reasonable range (0, 1_000_000)
    out_of_range = ((df["amount"] > 1_000_000) | (df["amount"] < 0)).sum()
    _record(
        "amount_within_range",
        out_of_range == 0,
        f"Out-of-range amounts: {out_of_range}",
    )

    # 9. category non-empty strings
    empty_cats = (df["category"].astype(str).str.strip() == "").sum()
    _record(
        "category_non_empty",
        empty_cats == 0,
        f"Empty category values: {empty_cats}",
    )

    # 10. merchant_id non-empty strings
    empty_merch = (df["merchant_id"].astype(str).str.strip() == "").sum()
    _record(
        "merchant_id_non_empty",
        empty_merch == 0,
        f"Empty merchant_id values: {empty_merch}",
    )

    return results


def ingest(raw_path: Path | None = None) -> pd.DataFrame:
    """Load, validate, clean and persist transaction data.

    Args:
        raw_path: Override path to raw CSV (defaults to data/raw/transactions.csv).

    Returns:
        Cleaned DataFrame persisted to data/processed/clean.parquet.

    Raises:
        DataQualityError: On critical validation failure.
    """
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    if raw_path is None:
        raw_path = RAW_DIR / "transactions.csv"

    logger.info("Ingesting %s", raw_path)
    df = pd.read_csv(raw_path)

    # Normalize types
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["amount"] = df["amount"].astype(float)
    df["is_anomaly"] = df["is_anomaly"].astype(int)

    # Run quality assertions
    assertions = run_quality_assertions(df)
    passed = sum(1 for a in assertions if a["passed"])
    failed = len(assertions) - passed

    # Persist clean parquet
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "clean.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved clean data → %s (%d rows)", out_path, len(df))

    # Quality report
    quality_report = {
        "request_id": request_id,
        "timestamp": timestamp,
        "source_file": str(raw_path),
        "rows_ingested": len(df),
        "checks_passed": passed,
        "checks_failed": failed,
        "assertions": assertions,
    }
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = LOGS_DIR / "quality_report.json"
    report_path.write_text(json.dumps(quality_report, indent=2))

    # Audit log
    _log_event(
        {
            "event": "ingest_complete",
            "request_id": request_id,
            "timestamp": timestamp,
            "rows": len(df),
            "checks_passed": passed,
            "checks_failed": failed,
        }
    )

    print(f"[STAGE 1] Ingested {len(df)} rows | {passed}/{len(assertions)} checks passed")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest()
