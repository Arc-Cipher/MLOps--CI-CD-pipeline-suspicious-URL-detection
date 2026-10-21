import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

GOOD_URL = "https://www.google.com/search?q=cybersecurity"
PHISH_URL = "http://login.paypal.com.account-update.xyz/verify.php?token=9283"


@pytest.fixture(scope="session")
def client():
    if not os.path.exists("artifacts/model.joblib"):
        subprocess.check_call([sys.executable, "train.py"])
    from app import app
    app.config["TESTING"] = True
    return app.test_client()


def test_health(client):
    res = client.get("/health").json
    assert res["status"] == "ok"
    assert "model_version" in res


def test_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Phishing &amp; Malicious URL Scanner" in r.data or b"Phishing & Malicious URL Scanner" in r.data


def test_predict_legitimate(client):
    r = client.post("/api/predict", json={"url": GOOD_URL})
    assert r.status_code == 200
    data = r.json
    assert data["prediction"] == "legitimate"
    assert 0 <= data["confidence"] <= 1
    assert "risk_score" in data
    assert data["risk_score"] < 50.0
    assert "entropy" in data["features"]


def test_predict_phishing(client):
    r = client.post("/api/predict", json={"url": PHISH_URL})
    assert r.status_code == 200
    data = r.json
    assert data["prediction"] == "phishing"
    assert data["risk_score"] > 50.0


@pytest.mark.parametrize("bad", [
    {},
    {"url": ""},
    {"url": 12345},
    {"url_length": "invalid_number"}
])
def test_predict_invalid(client, bad):
    assert client.post("/api/predict", json=bad).status_code == 400


def test_feedback_and_metrics(client):
    client.post("/api/predict", json={"url": GOOD_URL})
    r = client.post("/api/feedback",
                    json={"predicted": "legitimate", "actual": "legitimate"})
    assert r.json["correct"] is True

    body = client.get("/metrics").get_data(as_text=True)
    for name in ["model_predictions_total",
                 "model_prediction_latency_seconds_bucket",
                 "model_training_accuracy", "model_feedback_total",
                 "model_live_accuracy"]:
        assert name in body


def test_drift_metrics_emitted(client):
    # Inject 35 requests with long, skewed URLs to populate window >= MIN_DRIFT_SAMPLES
    for i in range(35):
        client.post("/api/predict", json={
            "url": f"http://{'x'*50}-very-long-anomalous-subdomain-{i}.evil-domain.cc/path/payload?token={'a'*40}"
        })
    body = client.get("/metrics").get_data(as_text=True)
    assert "model_drift_ks_statistic" in body
