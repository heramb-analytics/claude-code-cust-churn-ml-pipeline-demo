# Transaction Anomaly Detection Pipeline

An end-to-end ML pipeline for detecting anomalous financial transactions, built with XGBoost, FastAPI, and Playwright E2E testing.

## Overview

This pipeline automatically processes raw transaction data, engineers features, trains an XGBoost classifier, serves predictions via a FastAPI REST API, and validates everything with automated tests.

**Problem Type:** Binary Classification (Anomaly Detection)
**Algorithm:** XGBoost Classifier
**Primary Metric:** F1 Score

## Project Structure

```
├── data/
│   ├── raw/                  # Source data (READ ONLY)
│   └── processed/            # Cleaned and feature-engineered parquet files
├── src/
│   ├── data/ingest.py        # Stage 1: Data ingestion + 10 quality checks
│   ├── features/engineer.py  # Stage 2A: Feature engineering (11 features)
│   ├── models/train.py       # Stage 3: XGBoost training + hyperparameter search
│   ├── api/main.py           # Stage 5: FastAPI REST API + UI Dashboard
│   ├── validation/checks.py  # Stage 2C: 12 data validation checks
│   └── scheduler/nightly_job.py  # Stage 10: Nightly retraining scheduler
├── tests/
│   ├── unit/test_pipeline.py # Stage 4: 8 unit tests
│   └── e2e/test_api.py       # Stage 6: 6 Playwright E2E tests
├── models/
│   ├── pipeline_model.pkl    # Trained XGBoost model
│   └── pipeline_model_metrics.json  # Performance metrics
├── reports/
│   ├── figures/              # 5 EDA charts
│   └── screenshots/          # 6 Playwright screenshots
└── logs/
    ├── audit.jsonl           # Operation audit trail
    ├── quality_report.json   # Data quality check results
    └── validation_report.json # Feature validation results
```

## Setup

### Prerequisites

- Python 3.12+
- Homebrew (macOS) for `libomp` (required by XGBoost)

```bash
brew install libomp
pip install -r requirements.txt
python -m playwright install chromium
```

### Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost fastapi uvicorn pytest \
    playwright pytest-playwright APScheduler matplotlib seaborn scipy \
    requests httpx pyarrow
```

## Running the Pipeline

### Step 1: Ingest + Validate Data
```bash
python src/data/ingest.py
```

### Step 2: Feature Engineering + EDA + Validation
```bash
python src/features/engineer.py
python reports/eda_report.py
python src/validation/checks.py
```

### Step 3: Train Model
```bash
python src/models/train.py
```

### Step 4: Run Unit Tests
```bash
pytest tests/unit/ -v
```

### Step 5: Start API Server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 6: Run E2E Tests
```bash
pytest tests/e2e/ -v
```

## API Documentation

### POST /predict

Predict whether a transaction is anomalous.

**Request:**
```json
{
    "amount": 9999.99,
    "merchant_id": "MCC002",
    "category": "electronics",
    "timestamp": "2024-01-01T09:05:00"
}
```

**Response:**
```json
{
    "request_id": "uuid",
    "timestamp": "2024-01-01T09:05:00Z",
    "is_anomaly": 1,
    "anomaly_probability": 0.92,
    "label": "ANOMALY",
    "amount": 9999.99,
    "merchant_id": "MCC002",
    "category": "electronics",
    "inference_ms": 2.3
}
```

### GET /health

```json
{
    "status": "ok",
    "model_loaded": true,
    "uptime_seconds": 120.5,
    "timestamp": "2024-01-01T09:05:00Z",
    "version": "1.0.0"
}
```

### GET /metrics

Returns full model training metrics including F1, precision, recall, ROC-AUC.

### GET /

Interactive web dashboard for real-time transaction analysis.

## Features Engineered

| Feature | Description |
|---------|-------------|
| `amount` | Raw transaction amount |
| `log_amount` | Log-transformed amount (reduces skew) |
| `amount_zscore` | Z-score normalized amount |
| `amount_vs_merchant_mean` | Amount relative to merchant average |
| `merchant_anomaly_rate` | Historical anomaly rate per merchant |
| `category_anomaly_rate` | Historical anomaly rate per category |
| `hour_of_day` | Hour extracted from timestamp |
| `day_of_week` | Day of week (0=Monday) |
| `minute_of_hour` | Minute of hour |
| `is_weekend` | Binary weekend indicator |
| `seconds_since_last_txn` | Time since previous transaction |

## Scheduler

Nightly jobs run automatically:
- **02:00 daily**: Validate new data → retrain if >500 new rows
- **Every 6 hours**: Drift check → creates JIRA ticket if anomaly rate deviates >20%

---

Built with [Claude Code](https://claude.ai/claude-code) | Anthropic
