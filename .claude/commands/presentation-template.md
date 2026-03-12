# Presentation Template — Stage 11
# Claude creates reports/pipeline_presentation.pptx using python-pptx.
# Run: pip3 install python-pptx

## SLIDE STRUCTURE (8 slides)

### Slide 1 — Cover
Title: "{Problem Type} ML Pipeline"
Subtitle: "Built with Claude Code  ·  {date}"
Background: dark navy (#1E3A8A)
Logo area: top right (placeholder)

### Slide 2 — Problem Statement
Title: "The Problem"
Content: 3 bullets from data discovery:
  • {N} transactions with {anomaly_rate}% anomaly rate
  • Manual detection takes {X} hours per review cycle
  • Goal: automated real-time anomaly scoring

### Slide 3 — EDA Highlights
Title: "Data Overview"
Insert chart: reports/figures/01_target_distribution.png (left)
Insert chart: reports/figures/02_feature_correlations.png (right)
Caption: "{N} rows  ·  {N} features  ·  {anomaly_rate}% anomaly rate"

### Slide 4 — Data Engineering
Title: "Data Pipeline"
Insert chart: reports/figures/03_missing_values.png
Table: quality checks — Check Name | Result | Rows Affected

### Slide 5 — Model Results
Title: "Model Performance"
Metric cards (large): {metric_1}: {value_1}, {metric_2}: {value_2}, {metric_3}: {value_3}
Insert chart: reports/figures/04_amount_distribution.png
Caption: "Algorithm: {algorithm}  ·  Trained on {N} samples"

### Slide 6 — App Demo Screenshot
Title: "Live Dashboard"
Insert screenshot: reports/screenshots/01_dashboard_home.png (full width)
Caption: "Accessible at http://localhost:8000"

### Slide 7 — Test Evidence
Title: "Automated Quality Gates"
Insert screenshot: reports/screenshots/03_prediction_result.png (left)
Insert screenshot: reports/screenshots/04_swagger_docs.png (right)
Results row: "8 unit tests passed  ·  6 Playwright E2E tests passed"

### Slide 8 — What Was Built
Title: "PIPELINE COMPLETE"
Content:
  • Model: {algorithm} — {metric}: {value}
  • API: http://localhost:8000
  • GitHub: {repo_url}
  • JIRA: {N} tickets in project {key}
  • Confluence: page in CR space
  • Built by: Claude Code in {N} minutes
Background: dark navy
