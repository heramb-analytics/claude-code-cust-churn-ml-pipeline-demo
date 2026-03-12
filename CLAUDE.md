cat > CLAUDE.md << 'MDEOF'
# AUTONOMOUS PIPELINE AGENT
# Claude reads this on every session start. All protocol below is always active.
 
## IDENTITY
You are an autonomous ML pipeline engineer.
Execute all tasks completely without asking for clarification.
Fix errors automatically in agentic loops.
 
## PROJECT LAYOUT
data/raw/             → READ ONLY — never write here
data/processed/       → cleaned parquet outputs
src/                  → all source code
tests/unit/           → pytest unit tests
tests/e2e/            → Playwright e2e tests
models/               → pkl + metrics.json
logs/                 → audit.jsonl, quality_report.json
reports/figures/      → EDA charts
reports/screenshots/  → Playwright screenshots (auto-saved)
.claude/agents/       → subagent definitions
.claude/commands/     → custom slash commands
.claude/skills/       → coding skill blueprints
 
## CODING STANDARDS
- Type-annotated + Google docstrings on every function
- from pathlib import Path — no hardcoded paths
- DataQualityError for critical validation failures
- JSON Lines logging to logs/*.jsonl for every operation
- model: {name}.pkl + {name}_metrics.json always saved together
- API: every response includes request_id + timestamp
 
## HARD RULES
- NEVER write to data/raw/
- NEVER commit code that fails pytest
- NEVER put credentials in any project file
- NEVER skip the test stage
- ALWAYS save Playwright screenshots to reports/screenshots/
- ALWAYS run setup.sh before Stage 2 if worktrees not already created
 
## GIT WORKFLOW
Branch: feature/{problem-type}-pipeline-v1
Commits: Conventional Commits (feat:, fix:, chore:, docs:, test:)
Never commit to main. Always feature branch.
Always push to GitHub — use gh CLI if remote not set:
  gh repo create claude-code-ml-pipeline-demo --public --source=. --push
 
## GITHUB AUTO-PUSH PROTOCOL
Before Stage 7 git operations:
  1. Check: git remote -v
  2. If no remote: run gh repo create ... --push automatically
  3. Push README.md and all pipeline files
  4. Print the GitHub URL on completion
 
## MCP TOOLS
git, jira, confluence, playwright — registered in ~/.claude.json
JIRA: create project from prompt → create epic → create tasks inside epic
Confluence space: CR
 
## SOUND NOTIFICATIONS
After each stage completes, run: python3 -c "import subprocess; subprocess.run([chr(7)])"
Or on Mac: afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || true
Play different sound for errors: afplay /System/Library/Sounds/Sosumi.aiff 2>/dev/null
 
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PIPELINE PROTOCOL — TRIGGERS ON "create pipeline" or "build pipeline"
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
### STAGE 0 — DATA DISCOVERY
Parse DATA_FOLDER from user message (default: data/raw).
Scan every file → print name, rows, cols, dtypes.
Infer: problem_type, target_col, id_cols.
Announce plan. Proceed immediately.
 
### STAGE 1 — INGEST + VALIDATE
Create src/data/ingest.py with 10 quality assertions.
Self-heal until data/processed/clean.parquet saved.
Output: clean.parquet + logs/quality_report.json
 
### STAGE 2 — FEATURES + EDA + VALIDATION (PARALLEL)
Use worktrees if available (run setup.sh first).
Subagent A: src/features/engineer.py → features.parquet + feature_schema.json
Subagent B: reports/eda_report.py → 5 charts in reports/figures/
Subagent C: src/validation/checks.py → 12 checks → logs/validation_report.json
 
### STAGE 3 — MODEL TRAINING
Auto-select: anomaly→IsolationForest | class→XGBoost | regr→XGBoost
Stratified 70/15/15 split. Assert zero index overlap.
RandomizedSearchCV on 3 hyperparameters.
Output: models/pipeline_model.pkl + models/pipeline_model_metrics.json
 
### STAGE 4 — TEST SUITE (SELF-HEALING)
8 tests: model_load, predict_schema, metric_threshold, data_leakage,
  latency_under_500ms, invalid_input_raises, output_range, determinism
pytest tests/unit/ -v  —  self-heal until all pass.
 
### STAGE 5 — FASTAPI + UI DASHBOARD
Endpoints: POST /predict, GET /health, GET /metrics, GET / (dashboard)
Dashboard: header, live status dot, prediction form, result badge,
  metrics cards, last-10 predictions table, Tailwind CDN styling
Start: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
Wait 4s. Verify /health → model_loaded: true.
 
### STAGE 6 — PLAYWRIGHT + SCREENSHOTS
Verify app running: curl http://localhost:8000/health
Take 6 screenshots: 01_dashboard_home.png, 02_form_filled.png,
  03_prediction_result.png, 04_swagger_docs.png,
  05_metrics_endpoint.png, 06_health_endpoint.png
Write tests/e2e/test_api.py (6 tests). Run. Self-heal until pass.
 
### STAGE 7 — GIT + GITHUB
Check git remote. If none: gh repo create --push automatically.
Create README.md with project description, setup steps, API docs.
Branch: feature/{problem-type}-pipeline-v1
Commit: all src/, tests/, models/, reports/, README.md
Push to GitHub. Print repo URL.
 
### STAGE 8 — JIRA
Create JIRA project named after the pipeline (infer from prompt).
Create Epic: "{Problem Type} ML Pipeline v1.0"
Create 6 tasks inside the epic — see .claude/commands/jira-template.md
Create Sprint 1 with all tasks.
 
### STAGE 9 — CONFLUENCE
Create page in CR space — see .claude/commands/confluence-template.md
Include all 11 sections. Embed screenshot filenames.
 
### STAGE 10 — SCHEDULER
src/scheduler/nightly_job.py:
  Job 1 @ 02:00: validate new data → retrain if >500 new rows
  Job 2 @ every 6h: drift check → JIRA ticket if anomaly rate deviates >20%
 
### STAGE 11 — PRESENTATION
Create a PowerPoint presentation at reports/pipeline_presentation.pptx
Use instructions from .claude/commands/presentation-template.md
 
### STAGE 12 — FINAL SUMMARY
Print exactly:
PIPELINE COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model    : {algorithm} — {metric}: {value}
API      : http://localhost:8000  (running)
UI       : http://localhost:8000  (open in browser to test)
Tests    : {N} unit  |  {N} e2e  — all passed
Screenshots: {N} saved to reports/screenshots/
GitHub   : {repo_url}
JIRA     : Project {key} — {N} tickets in Sprint 1
Confluence: {page_url}
Slides   : reports/pipeline_presentation.pptx
Files    : {total} created
Time     : {elapsed} min
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MDEOF
 
