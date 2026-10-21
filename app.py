"""Flask model-serving app with persistent human feedback storage, active learning retraining, and metrics."""
import csv
import json
import logging
import os
import threading
import time
from collections import deque

import joblib
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)
from scipy.stats import ks_2samp

from features import FEATURES, extract_features

ART_DIR = os.getenv("MODEL_DIR", "artifacts")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
WINDOW = int(os.getenv("DRIFT_WINDOW", "100"))
MIN_DRIFT_SAMPLES = int(os.getenv("MIN_DRIFT_SAMPLES", "30"))
FEEDBACK_FILE = os.getenv("FEEDBACK_FILE", "data/feedback.csv")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mlmon")

# Ensure feedback storage directory exists and safely initialize CSV header
fb_dir = os.path.dirname(FEEDBACK_FILE) or "."
try:
    os.makedirs(fb_dir, exist_ok=True)
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "url", "predicted", "actual", "label", "is_correct"])
except Exception as e:
    log.error(f"Warning: Could not initialize feedback storage at {FEEDBACK_FILE}: {e}")

# ---------- prometheus metrics ----------
HTTP_REQS = Counter("http_requests_total", "HTTP requests",
                    ["endpoint", "method", "status"])
PREDICTIONS = Counter("model_predictions_total", "Predictions served",
                      ["predicted_class"])
LATENCY = Histogram("model_prediction_latency_seconds",
                    "Model inference latency",
                    buckets=(.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1))
CONFIDENCE = Histogram("model_prediction_confidence",
                       "Max class probability of predictions",
                       buckets=(.5, .6, .7, .8, .85, .9, .95, .98, .99, 1.0))
FEEDBACK = Counter("model_feedback_total", "Ground-truth feedback received",
                   ["correct"])
FEEDBACK_STORED = Counter("model_feedback_stored_total", "Total feedback examples saved to disk")
LIVE_ACC = Gauge("model_live_accuracy", "Accuracy over recent feedback window")
DRIFT_KS = Gauge("model_drift_ks_statistic",
                 "KS statistic: live window vs training data", ["feature"])
DRIFT_P = Gauge("model_drift_pvalue", "KS test p-value", ["feature"])
WIN_MEAN = Gauge("model_feature_window_mean",
                 "Mean of feature over live window", ["feature"])
REF_MEAN = Gauge("model_feature_reference_mean",
                 "Mean of feature in training data", ["feature"])
TRAIN_ACC = Gauge("model_training_accuracy", "Hold-out accuracy at training")
TRAIN_F1 = Gauge("model_training_f1_macro", "Hold-out macro F1 at training")
INFO = Gauge("model_info", "Model metadata", ["version", "type"])

# ---------- live state & artifacts loader ----------
_lock = threading.Lock()
_window = deque(maxlen=WINDOW)
_fb_window = deque(maxlen=200)
_last_predictions = deque(maxlen=50)  # cache for matching delayed feedback to URLs
_count = 0

model = None
METRICS = None
REF = None
CLASSES = []


def _load_artifacts():
    global model, METRICS, REF, CLASSES
    model = joblib.load(os.path.join(ART_DIR, "model.joblib"))
    with open(os.path.join(ART_DIR, "metrics.json")) as f:
        METRICS = json.load(f)
    with open(os.path.join(ART_DIR, "reference.json")) as f:
        REF = json.load(f)
    CLASSES = METRICS["classes"]

    TRAIN_ACC.set(METRICS["accuracy"])
    TRAIN_F1.set(METRICS["f1_macro"])
    INFO.labels(METRICS["model_version"], METRICS["model_type"]).set(1)
    for _n in FEATURES:
        REF_MEAN.labels(_n).set(REF["means"][_n])


_load_artifacts()

app = Flask(__name__)


def _parse_input(payload):
    if not isinstance(payload, dict):
        raise ValueError("JSON body required")

    # If URL string is provided, extract features dynamically
    if "url" in payload and isinstance(payload["url"], str) and payload["url"].strip():
        fdict = extract_features(payload["url"])
        vals = [fdict[n] for n in FEATURES]
        return vals, fdict, payload["url"].strip()

    # Otherwise validate explicit feature dict
    vals = []
    fdict = {}
    for n in FEATURES:
        if n not in payload:
            raise ValueError(f"missing feature '{n}' (or provide 'url' field)")
        try:
            v = float(payload[n])
        except (TypeError, ValueError):
            raise ValueError(f"feature '{n}' must be a number")
        vals.append(v)
        fdict[n] = v
    return vals, fdict, None


def _update_drift():
    arr = np.array(_window)
    for i, n in enumerate(FEATURES):
        WIN_MEAN.labels(n).set(float(arr[:, i].mean()))
    if len(arr) >= MIN_DRIFT_SAMPLES:
        for i, n in enumerate(FEATURES):
            stat, p = ks_2samp(arr[:, i], REF["data"][n])
            DRIFT_KS.labels(n).set(float(stat))
            DRIFT_P.labels(n).set(float(p))


@app.after_request
def _count_request(resp):
    ep = request.endpoint or "unknown"
    if ep != "static":
        HTTP_REQS.labels(ep, request.method, str(resp.status_code)).inc()
    return resp


@app.get("/")
def index():
    return render_template("index.html", features=FEATURES, classes=CLASSES,
                           metrics=METRICS, grafana_url=GRAFANA_URL)


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        model_version=METRICS["model_version"],
        model_type=METRICS["model_type"],
        domain=METRICS.get("domain", "Cybersecurity Phishing Detection"),
        training_samples=METRICS.get("total_training_samples", 0),
        feedback_samples=METRICS.get("feedback_samples_count", 0)
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.get("/api/model")
def model_info():
    return jsonify({k: v for k, v in METRICS.items() if k != "report"})


@app.post("/api/predict")
def predict():
    global _count
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="JSON body required"), 400
    try:
        x, fdict, raw_url = _parse_input(payload)
    except ValueError as e:
        return jsonify(error=str(e)), 400

    t0 = time.perf_counter()
    with _lock:
        proba = model.predict_proba([x])[0]
    LATENCY.observe(time.perf_counter() - t0)

    idx = int(np.argmax(proba))
    label = CLASSES[idx]
    conf = float(proba[idx])
    PREDICTIONS.labels(label).inc()
    CONFIDENCE.observe(conf)

    with _lock:
        _window.append(x)
        if raw_url:
            _last_predictions.append({"url": raw_url, "prediction": label, "time": time.time()})
        _count += 1
        _update_drift()

    prob_map = {c: round(float(p), 4) for c, p in zip(CLASSES, proba)}
    phish_risk = round(float(prob_map.get("phishing", 0.0) * 100), 1)

    return jsonify(
        prediction=label,
        confidence=round(conf, 4),
        risk_score=phish_risk,
        probabilities=prob_map,
        url=raw_url,
        features=fdict,
        model_version=METRICS["model_version"]
    )


@app.post("/api/feedback")
def feedback():
    """Receive and persistently store ground-truth label feedback for active learning retraining.

    Payload: {"url": "http://...", "predicted": "phishing", "actual": "phishing"}
    """
    d = request.get_json(silent=True) or {}
    pred = d.get("predicted")
    act = d.get("actual")

    if pred not in CLASSES or act not in CLASSES:
        return jsonify(error=f"'predicted' and 'actual' must be in {CLASSES}"), 400

    url = d.get("url", "").strip()
    if not url and len(_last_predictions) > 0:
        url = _last_predictions[-1]["url"]

    ok = pred == act
    FEEDBACK.labels(str(ok).lower()).inc()
    FEEDBACK_STORED.inc()

    numeric_label = 1 if act == "phishing" else 0
    with _lock:
        _fb_window.append(1 if ok else 0)
        LIVE_ACC.set(sum(_fb_window) / len(_fb_window))

        try:
            with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    url or "N/A",
                    pred,
                    act,
                    numeric_label,
                    1 if ok else 0
                ])
        except Exception as e:
            log.error(f"Error persisting feedback to {FEEDBACK_FILE}: {e}")

    total_stored = 0
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, mode="r", encoding="utf-8") as f:
                total_stored = max(0, sum(1 for _ in f) - 1)
        except Exception:
            pass

    return jsonify(
        status="recorded",
        correct=ok,
        url=url,
        total_feedback_stored=total_stored
    )


@app.get("/api/feedback/stats")
def feedback_stats():
    """Return count and summary of accumulated human feedback."""
    total = 0
    correct = 0
    incorrect = 0
    recent = []

    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    total += 1
                    if r.get("is_correct") == "1":
                        correct += 1
                    else:
                        incorrect += 1
                    recent.append(r)
        except Exception as e:
            log.error(f"Error reading feedback stats from {FEEDBACK_FILE}: {e}")

    return jsonify(
        total_feedback=total,
        correct=correct,
        incorrect=incorrect,
        recent_samples=recent[-10:]
    )


@app.post("/api/retrain")
def retrain():
    """Trigger model retraining incorporating accumulated feedback and reload hot in-memory."""
    try:
        import train
        train.main()
        with _lock:
            _load_artifacts()

        return jsonify(
            status="retrained",
            model_version=METRICS["model_version"],
            accuracy=METRICS["accuracy"],
            total_training_samples=METRICS.get("total_training_samples", 0),
            feedback_incorporated=METRICS.get("feedback_samples_count", 0)
        )
    except Exception as e:
        log.error(f"Retraining failed: {e}")
        return jsonify(error=f"Retraining failed: {str(e)}"), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
    