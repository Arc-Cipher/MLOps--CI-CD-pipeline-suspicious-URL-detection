# Cybersecurity Phishing & Malicious URL Scanner

## Production MLOps CI/CD Pipeline & Active Learning System

An end-to-end, production-ready MLOps platform for **Cybersecurity Phishing & Malicious URL Detection**. The system features real-time threat scoring, continuous **Human-in-the-Loop (HITL) Active Learning**, automated **Covariate Data Drift Detection (KS-Test)**, **Prometheus & Grafana Observability**, and an 8-stage **Jenkins CI/CD Pipeline** with automated quality gate enforcement.

---

## 📌 System Architecture & Workflow

```
                        ┌───────────────────────────────────────────────────────────┐
                        │                  USER INTERACTION LAYER                   │
                        └─────────────────────────────┬─────────────────────────────┘
                                                      │
                                   HTTP Requests / UI Inspection / Feedback
                                                      │
                                                      ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                              FLASK APP SERVICE                                            │
 │                                              (Port 5000 / Gunicorn)                                       │
 │                                                                                                           │
 │  ┌───────────────────────────┐   ┌─────────────────────────────┐   ┌───────────────────────────────────┐  │
 │  │   Lexical Feature Engine  │   │  RandomForest Inference Engine│  │     Active Learning Collector     │  │
 │  │   (12 URL Telemetry Extr.)│──>│  (Artifacts: model, baseline)│   │   (Appends to data/feedback.csv)  │  │
 │  └───────────────────────────┘   └─────────────────────────────┘   └─────────────────┬─────────────────┘  │
 └──────────────────────────────┬───────────────────────────────────────────────────────┼────────────────────┘
                                │                                                       │
                   Exposes /metrics (5s Scrape)                                Persisted via Volume
                                │                                                       │
                                ▼                                                       ▼
 ┌──────────────────────────────────────────────────────────────┐        ┌───────────────────────────────────┐
 │                       OBSERVABILITY                          │        │          PERSISTENT STORE         │
 │                                                              │        │                                   │
 │   ┌──────────────────────┐        ┌──────────────────────┐   │        │   • phishing_urls_1000.csv (Base) │
 │   │      Prometheus      │───────>│       Grafana        │   │        │   • data/feedback.csv (HITL)      │
 │   │     (Port 9090)      │        │     (Port 3000)      │   │        └─────────────────┬─────────────────┘
 │   └──────────────────────┘        └──────────────────────┘   │                          │
 └──────────────────────────────────────────────────────────────┘                          │
                                                                                           │ Git Commit & Push
                                                                                           ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                           JENKINS CI/CD PIPELINE                                          │
 │                                                (Port 8080)                                                │
 │                                                                                                           │
 │  [Checkout] ➔ [Setup Venv] ➔ [Flake8 Lint] ➔ [Train & Quality Gate] ➔ [Pytest] ➔ [Build Docker] ➔ [Deploy]│
 └───────────────────────────────────────────────────────────────────────────────────────────────────────────┘

```

---

## 📑 Executive Summary

Modern cybersecurity threats evolve dynamically; static machine learning models quickly suffer from **concept drift** and **covariate shift** as attackers alter URL patterns. This project addresses the model degradation challenge by implementing a closed-loop **MLOps infrastructure**:

1. **Detection Core**: Extracts 12 lexical/structural features from URLs and predicts threat probability using a `RandomForestClassifier`.
2. **Human-in-the-Loop Active Learning**: Live security analyst feedback is stored dynamically in `data/feedback.csv` and merged with base datasets ($N_{base} + N_{feedback}$) during retraining.
3. **Real-time Observability**: Prometheus metrics track p50/p95/p99 latency, class distributions, moving accuracy, and statistical data drift via two-sample Kolmogorov-Smirnov (KS) tests against baseline distributions (`reference.json`).
4. **Automated Continuous Integration & Deployment**: Jenkins enforces code linting (`flake8`), unit tests (`pytest`), model accuracy Quality Gates ($\ge 85\%$), containerization (`Docker-in-Docker`), and zero-downtime stack redeployment.

---

## 🌐 Services Infrastructure & Port Mapping

| Service | Endpoint | Container Name | Description |
| --- | --- | --- | --- |
| **Phishing Scanner Web UI** | `http://localhost:5000` | `mlmon-app-1` | Dark-mode interactive web dashboard for real-time URL analysis & HITL feedback submission. |
| **Grafana Dashboard** | `http://localhost:3000` | `mlmon-grafana-1` | Pre-provisioned visual dashboard displaying real-time drift, prediction ratios, and latencies. |
| **Prometheus Server** | `http://localhost:9090` | `mlmon-prometheus-1` | Time-series database scraping application metrics every 5 seconds; evaluates alert rules. |
| **Jenkins CI/CD Server** | `http://localhost:8080` | `mlmon-jenkins-1` | Automated pipeline runner handling testing, model training, Quality Gate verification, and deployment. |

---

## 🛠️ Key Technical Features

### 1. Lexical & Structural URL Feature Extraction

The feature engineering engine (`features.py`) converts raw URL strings into 12 numerical feature vectors without requiring external network lookups or DNS queries:

* **Shannon Entropy**: Measures randomness in the URL string ($\text{Entropy} > 4.5$ indicates heavy obfuscation).
* **URL & Domain Length**: Total character count of raw URL, domain, and path components.
* **Subdomain Count**: Identifies domain spoofing tactics (e.g., `paypal.com.security-verify.xyz`).
* **IP Address Presence**: Binary flag checking if the host is a raw IP (`192.168.1.1`).
* **Suspicious Keywords**: Detects security-sensitive terms (`login`, `verify`, `account`, `banking`, `update`).
* **HTTPS Protocol Flag**: Binary indicator for secure transport verification.
* **Syntactic Character Counters**: Counts of digits, dots (`.`), hyphens (`-`), and query parameters (`?`, `=`).

### 2. Active Learning & Human-in-the-Loop (HITL) Retraining

* **In-App Hot Reloading**: Users or analysts can submit corrections via `/api/feedback`. Clicking **"Retrain Model with Feedback"** triggers `/api/retrain`, merging dataset files, re-fitting the `RandomForestClassifier`, saving updated artifacts, and hot-reloading the model in container memory without downtime.
* **Persistent Volume Syncing**: Host-to-container volume mapping (`${DATA_DIR:-./data}:/srv/data`) ensures all collected analyst feedback persists locally in `data/feedback.csv`.

### 3. Model Monitoring & Alerting

* **Covariate Data Drift (KS-Test)**: Computes a two-sample Kolmogorov-Smirnov test over a sliding window of recent URL feature vectors against the reference training distribution (`artifacts/reference.json`).
* **Alerting Rules (`prometheus/alerts.yml`)**:
* `DataDriftDetected`: Triggers when the KS-statistic exceeds $0.4$.
* `LiveAccuracyLow`: Triggers when moving accuracy falls below $80\%$.
* `HighP95Latency`: Triggers when 95th percentile inference latency exceeds $100\text{ms}$.
* `PhishingAttackSpike`: Triggers on anomalous surges in malicious prediction volume.



---

## 🚀 Jenkins CI/CD Pipeline Workflow

The Jenkins pipeline (`Jenkinsfile`) automates the entire software and model lifecycle:

```
[Stage 1: Setup] ──> Configures Python virtual environment (.venv) & dependencies
        │
[Stage 2: Lint]  ──> Enforces PEP 8 compliance via flake8 across all source code
        │
[Stage 3: Train + Evaluate (Quality Gate)]
                 ──> Merges base dataset + feedback.csv
                 ──> Trains RandomForestClassifier
                 ──> Enforces Quality Gate: Accuracy >= 0.85
        │
[Stage 4: Test]  ──> Runs unit & integration tests via pytest (Generates JUnit XML)
        │
[Stage 5: Build Image] ──> Builds versioned Docker image (mlmon-app:<BUILD_NUMBER> & latest)
        │
[Stage 6: Smoke Test Stack]
                 ──> Boots ephemeral container; tests /health, /api/predict, and /metrics
        │
[Stage 7: Deploy] ──> Zero-downtime container update (docker compose up -d app)

```

---

## 🔌 REST API Reference

### 1. Inspect URL Threat Level

```bash
curl -X POST http://localhost:5000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://login.paypal.com.account-update.xyz/verify.php?token=9283"}'

```

**Response:**

```json
{
  "prediction": "phishing",
  "confidence": 0.995,
  "risk_score": 99.5,
  "probabilities": {
    "legitimate": 0.005,
    "phishing": 0.995
  },
  "features": {
    "domain_length": 38.0,
    "entropy": 4.12,
    "has_ip": 0.0,
    "has_suspicious_keywords": 2.0,
    "is_https": 0.0,
    "num_digits": 4.0,
    "num_dots": 4.0,
    "num_hyphens": 1.0,
    "num_query_params": 1.0,
    "num_subdomains": 3.0,
    "path_length": 23.0,
    "url_length": 72.0
  },
  "model_version": "20261021-144157"
}

```

### 2. Submit Analyst Feedback (HITL)

```bash
curl -X POST http://localhost:5000/api/feedback \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://login.paypal.com.account-update.xyz/verify.php?token=9283",
    "predicted": "legitimate",
    "actual": "phishing"
  }'

```

### 3. Trigger In-App Active Learning Retrain

```bash
curl -X POST http://localhost:5000/api/retrain

```

### 4. Health & Observability Endpoints

* `GET /health`: Returns service status, active model version, total dataset samples, and feedback sample count.
* `GET /metrics`: Standard Prometheus metrics export.
* `GET /api/model`: Returns training accuracy, macro F1 score, confusion matrix, and feature importance rankings.

---

## ⚡ Quickstart & Local Setup

### 1. Launch Complete Stack via Docker

```bash
# Clone repository
git clone https://github.com/Arc-Cipher/MLOps--CI-CD-pipeline-suspicious-URL-detection.git
cd MLOps--CI-CD-pipeline-suspicious-URL-detection

# Configure local persistence directory in .env (optional for host-syncing)
echo "DATA_DIR=./data" > .env

# Spin up all containers
docker compose up -d --build

```

### 2. Simulate Synthetic Production Traffic & Data Drift

```bash
# Generate normal traffic (mix of legitimate and phishing URLs)
python scripts/simulate_traffic.py

# Induce artificial covariate data drift (triggers KS-test & Grafana alerts)
python scripts/simulate_traffic.py --drift 2.0

```

### 3. Execute Local Testing & Training

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run static code analysis
flake8 --max-line-length 120 app.py train.py features.py tests scripts

# Run model training locally
python train.py

# Run unit test suite
pytest -v

```

---

## 📊 Mapping Guide for Project Report & Presentation (PPT)

This section maps technical components directly to academic report chapters and presentation slides:

| Chapter / Slide Title | Key Content & Technical Elements |
| --- | --- |
| **1. Problem Statement & Motivation** | • Vulnerability of static ML models to evolving phishing tactics.<br>

<br>• Need for automated MLOps pipelines, continuous feedback, and drift monitoring. |
| **2. Architecture & Design** | • System Architecture Diagram (Flask, Prometheus, Grafana, Jenkins, Docker).<br>

<br>• 12 Lexical URL features extracted in real time without external API overhead. |
| **3. Active Learning & HITL Loop** | • Active Learning flow: Base dataset ($1,000$ samples) + live `feedback.csv` accumulation.<br>

<br>• In-memory model hot-reloading vs. Git-backed pipeline retraining. |
| **4. Observability & Drift Detection** | • Two-sample Kolmogorov-Smirnov (KS-Test) algorithm for feature drift evaluation.<br>

<br>• Prometheus metrics, histogram latency buckets, and pre-configured Grafana alerts. |
| **5. CI/CD Pipeline & Quality Gate** | • 8-stage Jenkins pipeline execution (`flake8`, `pytest`, `docker build`).<br>

<br>• Automated Quality Gate enforcing $\text{Accuracy} \ge 0.85$ to prevent model degradation. |
| **6. Results & Conclusion** | • Evaluation metrics ($100\%$ accuracy on holdout set, $0.9930$ 5-fold CV).<br>

<br>• Zero-downtime deployment capabilities using Docker-in-Docker workflows. |