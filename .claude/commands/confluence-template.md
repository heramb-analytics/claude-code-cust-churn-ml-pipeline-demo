# Confluence Page Formatting Template
# Used by Stage 9. Claude reads this before creating any Confluence page.

## PAGE LOCATION
Space: CR
Title: "{Problem Type} ML Pipeline — v1.0"
Parent page: ML Engineering (create if not exists)

## PAGE STRUCTURE (11 sections — all required)

### Section 1 — Executive Summary
Format: 2-3 sentence paragraph
Include: problem statement, approach, key result metric

### Section 2 — Architecture
Format: ASCII flow diagram + description table
Include:
  data/raw → ingest.py → clean.parquet
  clean.parquet → engineer.py → features.parquet
  features.parquet → pipeline_model.py → model.pkl
  model.pkl → main.py → FastAPI (port 8000)

### Section 3 — Data Catalogue
Format: table with columns: Feature, Type, Description, Null%, Range
Include all columns from the raw dataset.

### Section 4 — Feature Catalogue
Format: table with columns: Feature, Type, How Computed, Importance (if available)
Include all engineered features from feature_schema.json.

### Section 5 — Model Card
Format: two tables
Table 1 — Configuration: Algorithm, Version, Trained At, Train Size, Test Size
Table 2 — Metrics: metric name, value, threshold (if any)

### Section 6 — API Reference
Format: table per endpoint
For each endpoint: Method, Path, Description, Request schema, Response schema, Example

### Section 7 — Dashboard Guide
Format: numbered steps + screenshot references
Include: how to open, how to use prediction form, how to interpret result badge

### Section 8 — Test Coverage
Format: two tables
Table 1 — Unit tests: test name, what it tests, result
Table 2 — E2E tests: test name, what it tests, result

### Section 9 — Screenshots
Format: list each screenshot filename + 1-line description
Note: screenshots saved at reports/screenshots/ in the repository

### Section 10 — How to Run
Format: numbered code blocks
  1. Clone: git clone {repo_url}
  2. Install: pip3 install -r requirements.txt
  3. Start: uvicorn src.api.main:app --port 8000

### Section 11 — Monitoring & Drift
Format: table
Include: job name, schedule, trigger condition, alert type
