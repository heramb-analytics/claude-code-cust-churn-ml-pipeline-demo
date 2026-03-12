"""Playwright E2E tests for the Transaction Anomaly Detection API.

6 tests covering dashboard, form, prediction result, swagger docs,
metrics endpoint, and health endpoint — with screenshots.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

BASE_URL = "http://localhost:8000"
SCREENSHOTS_DIR = Path("reports/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session", autouse=True)
def ensure_server_running():
    """Verify the API server is up before running E2E tests."""
    import requests
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    except Exception as exc:
        pytest.skip(f"API server not running at {BASE_URL}: {exc}")


def test_01_dashboard_home(page: Page):
    """Screenshot 01: Dashboard home loads with title and status dot."""
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    expect(page).to_have_title("Transaction Anomaly Detection")
    expect(page.locator("text=Transaction Anomaly Detection").first).to_be_visible()
    expect(page.locator("text=Model Online")).to_be_visible()
    page.screenshot(path=str(SCREENSHOTS_DIR / "01_dashboard_home.png"), full_page=True)


def test_02_form_filled(page: Page):
    """Screenshot 02: Prediction form filled with anomalous transaction data."""
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")

    # Fill form
    page.fill("#amount", "9999.99")
    page.select_option("#merchant_id", "MCC002")
    page.select_option("#category", "electronics")

    page.screenshot(path=str(SCREENSHOTS_DIR / "02_form_filled.png"), full_page=True)

    # Assert form fields have expected values
    assert page.input_value("#amount") == "9999.99"
    assert page.input_value("#merchant_id") == "MCC002"


def test_03_prediction_result(page: Page):
    """Screenshot 03: Submit form and capture prediction result badge."""
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")

    # Fill and submit form
    page.fill("#amount", "9999.99")
    page.select_option("#merchant_id", "MCC002")
    page.select_option("#category", "electronics")
    page.click("button[type='submit']")

    # Wait for result to appear
    page.wait_for_selector("#resultContent:not(.hidden)", timeout=10000)
    result_label = page.locator("#resultLabel").text_content()
    assert result_label in ("ANOMALY", "NORMAL"), f"Unexpected label: {result_label}"

    page.screenshot(path=str(SCREENSHOTS_DIR / "03_prediction_result.png"), full_page=True)


def test_04_swagger_docs(page: Page):
    """Screenshot 04: Swagger UI loads with API documentation."""
    page.goto(f"{BASE_URL}/docs")
    page.wait_for_load_state("networkidle")
    page.wait_for_selector(".swagger-ui", timeout=10000)
    expect(page.locator("text=Transaction Anomaly Detection API")).to_be_visible()
    page.screenshot(path=str(SCREENSHOTS_DIR / "04_swagger_docs.png"), full_page=True)


def test_05_metrics_endpoint(page: Page):
    """Screenshot 05: /metrics endpoint returns JSON with model metrics."""
    page.goto(f"{BASE_URL}/metrics")
    page.wait_for_load_state("networkidle")

    content = page.content()
    assert "algorithm" in content or "f1" in content, "Metrics data not found in response"

    page.screenshot(path=str(SCREENSHOTS_DIR / "05_metrics_endpoint.png"), full_page=True)


def test_06_health_endpoint(page: Page):
    """Screenshot 06: /health endpoint returns model_loaded: true."""
    page.goto(f"{BASE_URL}/health")
    page.wait_for_load_state("networkidle")

    content = page.content()
    assert "model_loaded" in content, "model_loaded not in health response"
    assert "true" in content.lower(), "model_loaded is not true"

    page.screenshot(path=str(SCREENSHOTS_DIR / "06_health_endpoint.png"), full_page=True)
