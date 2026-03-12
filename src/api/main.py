"""FastAPI service for transaction anomaly detection.

Endpoints:
    POST /predict   — Predict anomaly for a transaction
    GET  /health    — Health check + model status
    GET  /metrics   — Model performance metrics
    GET  /          — Interactive UI dashboard
"""

from __future__ import annotations

import json
import pickle
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# ── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")
MODEL_PATH = MODELS_DIR / "pipeline_model.pkl"
METRICS_PATH = MODELS_DIR / "pipeline_model_metrics.json"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Transaction Anomaly Detection API",
    description="XGBoost-powered anomaly detection for financial transactions",
    version="1.0.0",
)

# ── State ────────────────────────────────────────────────────────────────────
_model_bundle: dict | None = None
_prediction_history: list[dict] = []
_start_time = time.time()


def _load_model() -> dict:
    """Load and cache the model bundle from disk.

    Returns:
        Model bundle dictionary with 'model' and 'feature_cols'.

    Raises:
        FileNotFoundError: If model file does not exist.
    """
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        with MODEL_PATH.open("rb") as fh:
            _model_bundle = pickle.load(fh)
    return _model_bundle


# Load model on startup
try:
    _load_model()
    _model_loaded = True
except Exception:
    _model_loaded = False


def _log_prediction(entry: dict) -> None:
    """Append prediction to JSONL audit log.

    Args:
        entry: Prediction event dict.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with (LOGS_DIR / "predictions.jsonl").open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


# ── Schemas ───────────────────────────────────────────────────────────────────
class TransactionInput(BaseModel):
    """Input schema for a single transaction prediction request."""

    amount: float = Field(..., gt=0, description="Transaction amount in USD", example=9999.99)
    merchant_id: str = Field(..., description="Merchant identifier", example="MCC002")
    category: str = Field(..., description="Transaction category", example="electronics")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp",
        example="2024-01-01T09:05:00",
    )


class PredictionResponse(BaseModel):
    """Output schema for anomaly prediction."""

    request_id: str
    timestamp: str
    is_anomaly: int
    anomaly_probability: float
    label: str
    amount: float
    merchant_id: str
    category: str
    inference_ms: float


# ── Feature builder ───────────────────────────────────────────────────────────
def _build_features(txn: TransactionInput) -> pd.DataFrame:
    """Construct feature vector from transaction input.

    Args:
        txn: Validated transaction input.

    Returns:
        Single-row feature DataFrame matching training feature schema.
    """
    ts = pd.Timestamp(txn.timestamp)
    amount = txn.amount

    # Reference statistics from training data (fitted values)
    TRAINING_MEAN = 4033.1
    TRAINING_STD = 4467.0
    MERCHANT_ANOMALY_RATES = {"MCC001": 0.0, "MCC002": 1.0, "MCC003": 0.0}
    CATEGORY_ANOMALY_RATES = {"retail": 0.0, "electronics": 1.0, "food": 0.0}
    MERCHANT_MEAN_AMOUNTS = {"MCC001": 185.25, "MCC002": 9374.995, "MCC003": 45.0}

    log_amount = float(np.log1p(amount))
    amount_zscore = float((amount - TRAINING_MEAN) / (TRAINING_STD + 1e-9))
    merchant_anomaly_rate = float(MERCHANT_ANOMALY_RATES.get(txn.merchant_id, 0.5))
    category_anomaly_rate = float(CATEGORY_ANOMALY_RATES.get(txn.category, 0.5))
    merchant_mean = MERCHANT_MEAN_AMOUNTS.get(txn.merchant_id, TRAINING_MEAN)
    amount_vs_merchant_mean = float(amount / (merchant_mean + 1e-9))

    row = {
        "amount": amount,
        "log_amount": log_amount,
        "amount_zscore": amount_zscore,
        "amount_vs_merchant_mean": amount_vs_merchant_mean,
        "merchant_anomaly_rate": merchant_anomaly_rate,
        "category_anomaly_rate": category_anomaly_rate,
        "hour_of_day": int(ts.hour),
        "day_of_week": int(ts.dayofweek),
        "minute_of_hour": int(ts.minute),
        "is_weekend": int(ts.dayofweek >= 5),
        "seconds_since_last_txn": 60.0,
    }

    bundle = _load_model()
    feature_cols = bundle["feature_cols"]
    return pd.DataFrame([row])[feature_cols]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health() -> dict:
    """Return API health and model status.

    Returns:
        Health status dictionary.
    """
    uptime_s = round(time.time() - _start_time, 1)
    return {
        "status": "ok",
        "model_loaded": _model_loaded,
        "uptime_seconds": uptime_s,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }


@app.get("/metrics", tags=["System"])
def metrics() -> dict:
    """Return model performance metrics from training.

    Returns:
        Metrics dictionary loaded from disk.
    """
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found")
    with METRICS_PATH.open() as fh:
        data = json.load(fh)
    data["predictions_served"] = len(_prediction_history)
    return data


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(txn: TransactionInput) -> PredictionResponse:
    """Predict whether a transaction is anomalous.

    Args:
        txn: Transaction details.

    Returns:
        Prediction result with probability and label.
    """
    request_id = str(uuid.uuid4())
    start_ns = time.perf_counter_ns()

    bundle = _load_model()
    model = bundle["model"]
    features = _build_features(txn)

    proba = float(model.predict_proba(features)[0, 1])
    threshold = 0.3
    is_anomaly = int(proba >= threshold)

    inference_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
    ts = datetime.now(timezone.utc).isoformat()

    entry = {
        "request_id": request_id,
        "timestamp": ts,
        "is_anomaly": is_anomaly,
        "anomaly_probability": round(proba, 4),
        "label": "ANOMALY" if is_anomaly else "NORMAL",
        "amount": txn.amount,
        "merchant_id": txn.merchant_id,
        "category": txn.category,
        "inference_ms": round(inference_ms, 2),
    }
    _prediction_history.append(entry)
    if len(_prediction_history) > 100:
        _prediction_history.pop(0)
    _log_prediction(entry)

    return PredictionResponse(**entry)


@app.get("/", response_class=HTMLResponse, tags=["UI"])
def dashboard() -> HTMLResponse:
    """Serve the interactive transaction anomaly detection dashboard.

    Returns:
        HTML page with Tailwind CSS styling, prediction form, and metrics.
    """
    # Load metrics for display
    model_metrics: dict[str, Any] = {}
    if METRICS_PATH.exists():
        with METRICS_PATH.open() as fh:
            model_metrics = json.load(fh)

    test_m = model_metrics.get("test_metrics", {})
    f1 = test_m.get("f1", 0)
    recall = test_m.get("recall", 0)
    precision = test_m.get("precision", 0)
    algo = model_metrics.get("algorithm", "XGBoost")

    # Recent predictions table rows
    recent = list(reversed(_prediction_history[-10:]))
    rows_html = ""
    for p in recent:
        badge_color = "red" if p["is_anomaly"] else "green"
        badge_text = p["label"]
        rows_html += f"""
        <tr class="border-b border-gray-700 hover:bg-gray-750">
            <td class="px-4 py-2 text-gray-400 text-sm">{p['timestamp'][:19]}</td>
            <td class="px-4 py-2 text-white font-medium">${p['amount']:,.2f}</td>
            <td class="px-4 py-2 text-gray-300">{p['merchant_id']}</td>
            <td class="px-4 py-2 text-gray-300 capitalize">{p['category']}</td>
            <td class="px-4 py-2">
                <span class="px-2 py-1 rounded-full text-xs font-semibold bg-{badge_color}-900 text-{badge_color}-300">
                    {badge_text}
                </span>
            </td>
            <td class="px-4 py-2 text-gray-400 text-sm">{p['anomaly_probability']:.2%}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transaction Anomaly Detection</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .bg-gray-750 {{ background-color: #1a2035; }}
        .status-dot {{ animation: pulse 2s cubic-bezier(0.4,0,0.6,1) infinite; }}
        @keyframes pulse {{ 0%,100% {{ opacity:1 }} 50% {{ opacity:.5 }} }}
    </style>
</head>
<body class="bg-gray-900 text-white min-h-screen">

<!-- Header -->
<header class="bg-gradient-to-r from-blue-900 to-indigo-900 shadow-lg">
    <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <div class="flex items-center space-x-3">
            <div class="text-3xl">🔍</div>
            <div>
                <h1 class="text-xl font-bold text-white">Transaction Anomaly Detection</h1>
                <p class="text-blue-300 text-sm">Powered by {algo} · v1.0.0</p>
            </div>
        </div>
        <div class="flex items-center space-x-2">
            <div class="w-3 h-3 rounded-full bg-green-400 status-dot"></div>
            <span class="text-green-400 text-sm font-medium">Model Online</span>
        </div>
    </div>
</header>

<main class="max-w-7xl mx-auto px-6 py-8 space-y-8">

    <!-- Metrics Cards -->
    <section>
        <h2 class="text-lg font-semibold text-gray-300 mb-4">Model Performance</h2>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="bg-gray-800 rounded-xl p-5 border border-gray-700">
                <div class="text-gray-400 text-sm mb-1">F1 Score</div>
                <div class="text-3xl font-bold text-blue-400">{f1:.2%}</div>
            </div>
            <div class="bg-gray-800 rounded-xl p-5 border border-gray-700">
                <div class="text-gray-400 text-sm mb-1">Recall</div>
                <div class="text-3xl font-bold text-green-400">{recall:.2%}</div>
            </div>
            <div class="bg-gray-800 rounded-xl p-5 border border-gray-700">
                <div class="text-gray-400 text-sm mb-1">Precision</div>
                <div class="text-3xl font-bold text-yellow-400">{precision:.2%}</div>
            </div>
            <div class="bg-gray-800 rounded-xl p-5 border border-gray-700">
                <div class="text-gray-400 text-sm mb-1">Predictions Served</div>
                <div class="text-3xl font-bold text-purple-400">{len(_prediction_history)}</div>
            </div>
        </div>
    </section>

    <!-- Prediction Form -->
    <section class="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <h2 class="text-lg font-semibold text-white mb-5">Predict Transaction</h2>
            <form id="predictForm" class="space-y-4">
                <div>
                    <label class="block text-gray-400 text-sm mb-1">Amount ($)</label>
                    <input id="amount" type="number" step="0.01" value="9999.99"
                        class="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"/>
                </div>
                <div>
                    <label class="block text-gray-400 text-sm mb-1">Merchant ID</label>
                    <select id="merchant_id" class="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500">
                        <option value="MCC001">MCC001 (Low Risk)</option>
                        <option value="MCC002" selected>MCC002 (High Risk)</option>
                        <option value="MCC003">MCC003 (Low Risk)</option>
                    </select>
                </div>
                <div>
                    <label class="block text-gray-400 text-sm mb-1">Category</label>
                    <select id="category" class="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500">
                        <option value="retail">Retail</option>
                        <option value="electronics" selected>Electronics</option>
                        <option value="food">Food</option>
                    </select>
                </div>
                <button type="submit"
                    class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg transition-colors">
                    Analyze Transaction
                </button>
            </form>
        </div>

        <!-- Result Badge -->
        <div class="bg-gray-800 rounded-xl p-6 border border-gray-700 flex flex-col justify-center items-center" id="resultPanel">
            <div class="text-gray-500 text-center" id="resultPlaceholder">
                <div class="text-5xl mb-3">🔎</div>
                <p class="text-sm">Submit a transaction to see results</p>
            </div>
            <div class="hidden text-center" id="resultContent">
                <div id="resultBadge" class="text-5xl font-bold mb-4"></div>
                <div id="resultLabel" class="text-2xl font-bold mb-2"></div>
                <div id="resultProba" class="text-gray-400 text-lg mb-4"></div>
                <div id="resultDetails" class="text-sm text-gray-500"></div>
            </div>
        </div>
    </section>

    <!-- Recent Predictions Table -->
    <section>
        <h2 class="text-lg font-semibold text-gray-300 mb-4">Recent Predictions</h2>
        <div class="bg-gray-800 rounded-xl border border-gray-700 overflow-x-auto">
            <table class="w-full text-left" id="predictionsTable">
                <thead>
                    <tr class="border-b border-gray-700 bg-gray-750">
                        <th class="px-4 py-3 text-gray-400 text-sm">Timestamp</th>
                        <th class="px-4 py-3 text-gray-400 text-sm">Amount</th>
                        <th class="px-4 py-3 text-gray-400 text-sm">Merchant</th>
                        <th class="px-4 py-3 text-gray-400 text-sm">Category</th>
                        <th class="px-4 py-3 text-gray-400 text-sm">Label</th>
                        <th class="px-4 py-3 text-gray-400 text-sm">Probability</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
                    {rows_html if rows_html else '<tr><td colspan="6" class="px-4 py-8 text-center text-gray-500">No predictions yet</td></tr>'}
                </tbody>
            </table>
        </div>
    </section>

</main>

<script>
document.getElementById('predictForm').addEventListener('submit', async function(e) {{
    e.preventDefault();
    const btn = e.target.querySelector('button');
    btn.textContent = 'Analyzing...';
    btn.disabled = true;

    const payload = {{
        amount: parseFloat(document.getElementById('amount').value),
        merchant_id: document.getElementById('merchant_id').value,
        category: document.getElementById('category').value,
        timestamp: new Date().toISOString()
    }};

    try {{
        const resp = await fetch('/predict', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify(payload)
        }});
        const data = await resp.json();

        document.getElementById('resultPlaceholder').classList.add('hidden');
        document.getElementById('resultContent').classList.remove('hidden');

        const isAnomaly = data.is_anomaly === 1;
        document.getElementById('resultBadge').textContent = isAnomaly ? '🚨' : '✅';
        document.getElementById('resultLabel').textContent = data.label;
        document.getElementById('resultLabel').className = `text-2xl font-bold mb-2 ${{isAnomaly ? 'text-red-400' : 'text-green-400'}}`;
        document.getElementById('resultProba').textContent = `Anomaly probability: ${{(data.anomaly_probability * 100).toFixed(1)}}%`;
        document.getElementById('resultDetails').textContent = `Inference: ${{data.inference_ms.toFixed(1)}}ms · ID: ${{data.request_id.slice(0,8)}}`;

        // Reload the page after 3 seconds to refresh the table
        setTimeout(() => location.reload(), 3000);
    }} catch(err) {{
        alert('Error: ' + err.message);
    }} finally {{
        btn.textContent = 'Analyze Transaction';
        btn.disabled = false;
    }}
}});
</script>

</body>
</html>"""
    return HTMLResponse(content=html)
