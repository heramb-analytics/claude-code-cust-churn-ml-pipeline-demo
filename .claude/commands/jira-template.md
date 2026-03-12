# JIRA Ticket Formatting Template
# Used by Stage 8 of the pipeline protocol.
# Claude reads this before creating any JIRA tickets.

## PROJECT CREATION
Project name: "{Problem Type} ML Pipeline" (infer from user prompt)
Project key: uppercase 4-letter abbreviation (e.g., TXAP for Transaction Anomaly Pipeline)
Project type: Scrum
Create before creating any tickets.

## EPIC (create first)
Summary: "{Problem Type} ML Pipeline — End-to-End Automation v1.0"
Description: |
  h2. Overview
  Automated ML pipeline built with Claude Code.
  Covers data ingestion, feature engineering, model training,
  API deployment, Playwright testing, and scheduled monitoring.
  |
  h3. Model Details
  | Attribute | Value |
  | Algorithm | {algorithm} |
  | Primary Metric | {metric}: {value} |
  | Train Size | {n} rows |
  | Features | {n} engineered features |
  h3. Deliverables
  * Trained model: models/pipeline_model.pkl
  * REST API: http://localhost:8000
  * Test coverage: {n} unit + {n} e2e tests
  * GitHub: {repo_url}

## TASK FORMAT (use for each of 6 tasks)

### Task 1 — Data Ingestion & Validation
Summary: [PIPE-{N}] Data Ingestion & Validation — {N} quality checks passed
Description: |
  h3. What was done
  * Loaded {filename} from data/raw/
  * Applied {N} automated quality assertions
  * Output: data/processed/clean.parquet ({N} rows, {N} columns)
  h3. Quality checks passed
  * No nulls in key columns: ✓
  * Amount range valid: ✓
  * Schema matches expected: ✓
  h3. Files created
  * src/data/ingest.py
  * data/processed/clean.parquet
  * logs/quality_report.json
Labels: data-engineering, automated
Story Points: 3
Status: Done

### Task 2 — Feature Engineering
Summary: [PIPE-{N}] Feature Engineering — {N} features created
Description: |
  h3. Features engineered
  | Feature | Type | Description |
  | {name} | {type} | {description} |
  h3. Files created
  * src/features/engineer.py
  * data/processed/features.parquet
  * data/processed/feature_schema.json
Labels: feature-engineering, automated
Story Points: 3
Status: Done

### Task 3 — Model Training
Summary: [PIPE-{N}] Model Training — {algorithm} {metric}: {value}
Description: |
  h3. Model Card
  | Attribute | Value |
  | Algorithm | {algorithm} |
  | Hyperparameters | {params} |
  | Train samples | {N} |
  | Test samples | {N} |
  | {metric_1} | {value_1} |
  | {metric_2} | {value_2} |
  | {metric_3} | {value_3} |
  h3. Files created
  * src/models/pipeline_model.py
  * models/pipeline_model.pkl
  * models/pipeline_model_metrics.json
Labels: ml, model-training, automated
Story Points: 5
Status: Done

### Task 4 — API Deployment with UI
Summary: [PIPE-{N}] FastAPI + Dashboard deployed on port 8000
Description: |
  h3. Endpoints
  | Method | Path | Description |
  | POST | /predict | Run model inference |
  | GET | /health | Service health check |
  | GET | /metrics | Model metrics |
  | GET | / | HTML dashboard |
  h3. Dashboard features
  * Live status indicator (auto-refreshes every 5s)
  * Prediction form with all {N} features
  * Result badge (ANOMALY/NORMAL + confidence)
  * Metrics cards + last-10 predictions table
  h3. URL
  http://localhost:8000
Labels: api, deployment, ui, automated
Story Points: 5
Status: Done

### Task 5 — Playwright E2E Tests + Screenshots
Summary: [PIPE-{N}] Playwright E2E — {N} tests passed, {N} screenshots saved
Description: |
  h3. Test results
  | Test | Status |
  | Dashboard loads | ✓ |
  | Status indicator visible | ✓ |
  | Prediction form submits | ✓ |
  | Result panel appears | ✓ |
  | Swagger docs accessible | ✓ |
  | Health endpoint OK | ✓ |
  h3. Screenshots saved
  * 01_dashboard_home.png
  * 02_form_filled.png
  * 03_prediction_result.png
  * 04_swagger_docs.png
  * 05_metrics_endpoint.png
  * 06_health_endpoint.png
Labels: testing, playwright, automated
Story Points: 3
Status: Done

### Task 6 — Nightly Scheduler
Summary: [PIPE-{N}] Nightly Scheduler — retraining + drift detection configured
Description: |
  h3. Scheduled jobs
  | Job | Schedule | Action |
  | Retrain | 02:00 daily | Validate new data → retrain if >500 rows |
  | Drift check | Every 6h | Compare anomaly rate → alert if >20% deviation |
  h3. Files created
  * src/scheduler/nightly_job.py
Labels: scheduler, monitoring, automated
Story Points: 2
Status: In Progress
