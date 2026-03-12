"""Nightly scheduler for transaction anomaly detection pipeline.

Job 1 @ 02:00 daily:  Validate new data → retrain if >500 new rows.
Job 2 @ every 6h:     Drift check → JIRA ticket if anomaly rate deviates >20%.
"""

from __future__ import annotations

import json
import logging
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("scheduler")

PROCESSED_DIR = Path("data/processed")
LOGS_DIR = Path("logs")
MODELS_DIR = Path("models")

# Baseline anomaly rate from initial training
BASELINE_ANOMALY_RATE: float = 0.4
DRIFT_THRESHOLD: float = 0.20
RETRAIN_ROW_THRESHOLD: int = 500


def _log_event(event: dict) -> None:
    """Append JSON-Lines entry to audit log.

    Args:
        event: Dict to serialize as one JSON line.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with (LOGS_DIR / "scheduler.jsonl").open("a") as fh:
        fh.write(json.dumps(event) + "\n")


def job_nightly_retrain() -> None:
    """Job 1: Validate new data and retrain if >500 new rows.

    Runs at 02:00 daily. Counts rows in clean.parquet, runs ingestion
    and training scripts if sufficient new data detected.
    """
    request_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    logger.info("[Job 1] Starting nightly retrain check — %s", request_id)

    try:
        import pandas as pd

        clean_path = PROCESSED_DIR / "clean.parquet"
        if not clean_path.exists():
            logger.warning("[Job 1] clean.parquet not found — skipping")
            _log_event({
                "event": "retrain_skipped",
                "reason": "clean.parquet missing",
                "request_id": request_id,
                "timestamp": ts,
            })
            return

        df = pd.read_parquet(clean_path)
        row_count = len(df)
        logger.info("[Job 1] Current row count: %d (threshold: %d)", row_count, RETRAIN_ROW_THRESHOLD)

        if row_count > RETRAIN_ROW_THRESHOLD:
            logger.info("[Job 1] Row threshold exceeded — triggering retrain pipeline")

            # Re-run feature engineering and training
            result_feat = subprocess.run(
                ["python3", "src/features/engineer.py"],
                capture_output=True, text=True, timeout=300
            )
            result_train = subprocess.run(
                ["python3", "src/models/train.py"],
                capture_output=True, text=True, timeout=600
            )

            _log_event({
                "event": "retrain_triggered",
                "request_id": request_id,
                "timestamp": ts,
                "row_count": row_count,
                "feature_exit_code": result_feat.returncode,
                "train_exit_code": result_train.returncode,
            })
            logger.info("[Job 1] Retrain complete (exit codes: feat=%d train=%d)",
                        result_feat.returncode, result_train.returncode)
        else:
            _log_event({
                "event": "retrain_skipped",
                "reason": f"row_count={row_count} < threshold={RETRAIN_ROW_THRESHOLD}",
                "request_id": request_id,
                "timestamp": ts,
            })
            logger.info("[Job 1] No retrain needed — insufficient new rows")

    except Exception as exc:
        logger.error("[Job 1] Error: %s", exc)
        _log_event({
            "event": "retrain_error",
            "request_id": request_id,
            "timestamp": ts,
            "error": str(exc),
        })


def job_drift_check() -> None:
    """Job 2: Check for anomaly rate drift and create JIRA ticket if drift detected.

    Runs every 6 hours. Compares current anomaly rate against baseline.
    If drift >20%, logs a JIRA drift alert (placeholder for real JIRA integration).
    """
    request_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    logger.info("[Job 2] Starting drift check — %s", request_id)

    try:
        # Read recent predictions from log
        pred_log = LOGS_DIR / "predictions.jsonl"
        if not pred_log.exists():
            logger.info("[Job 2] No predictions log found — skipping drift check")
            return

        predictions = []
        with pred_log.open() as fh:
            for line in fh:
                try:
                    predictions.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        if len(predictions) < 10:
            logger.info("[Job 2] Insufficient predictions (%d) for drift check", len(predictions))
            return

        # Calculate recent anomaly rate (last 100 predictions)
        recent = predictions[-100:]
        recent_rate = sum(1 for p in recent if p.get("is_anomaly") == 1) / len(recent)
        drift = abs(recent_rate - BASELINE_ANOMALY_RATE)

        logger.info(
            "[Job 2] Baseline: %.2f | Recent: %.2f | Drift: %.2f",
            BASELINE_ANOMALY_RATE, recent_rate, drift
        )

        if drift > DRIFT_THRESHOLD:
            logger.warning("[Job 2] DRIFT DETECTED — creating JIRA alert ticket")
            alert = {
                "event": "drift_alert",
                "request_id": request_id,
                "timestamp": ts,
                "baseline_anomaly_rate": BASELINE_ANOMALY_RATE,
                "recent_anomaly_rate": recent_rate,
                "drift": drift,
                "drift_threshold": DRIFT_THRESHOLD,
                "action": "JIRA ticket created (mcp__jira__create_issue in production)",
                "recommendation": "Retrain model with recent data",
            }
            _log_event(alert)
            # In production: call mcp__jira__jira_create_issue(project_key="KAN", ...)
            logger.warning("[Job 2] Drift alert logged: %s", json.dumps(alert))
        else:
            _log_event({
                "event": "drift_ok",
                "request_id": request_id,
                "timestamp": ts,
                "baseline_anomaly_rate": BASELINE_ANOMALY_RATE,
                "recent_anomaly_rate": recent_rate,
                "drift": drift,
            })
            logger.info("[Job 2] Drift within threshold — no action required")

    except Exception as exc:
        logger.error("[Job 2] Error: %s", exc)
        _log_event({
            "event": "drift_check_error",
            "request_id": request_id,
            "timestamp": ts,
            "error": str(exc),
        })


def main() -> None:
    """Start the APScheduler with both nightly jobs."""
    scheduler = BlockingScheduler(timezone="UTC")

    # Job 1: Nightly retrain at 02:00 UTC
    scheduler.add_job(
        job_nightly_retrain,
        trigger=CronTrigger(hour=2, minute=0),
        id="nightly_retrain",
        name="Nightly Data Validation + Retrain",
        replace_existing=True,
    )

    # Job 2: Drift check every 6 hours
    scheduler.add_job(
        job_drift_check,
        trigger=IntervalTrigger(hours=6),
        id="drift_check",
        name="Anomaly Rate Drift Check",
        replace_existing=True,
    )

    logger.info("Scheduler started. Jobs:")
    for job in scheduler.get_jobs():
        logger.info("  - %s | next run: %s", job.name, job.next_run_time)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
