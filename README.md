# Cybersecurity URL Phishing Scanner · Model Monitoring & CI/CD Pipeline

Production-ready Machine Learning Model Monitoring stack featuring **Phishing & Malicious URL Detection**, automated **Covariate Data Drift Detection (KS-Test)**, **Live Accuracy Tracking**, **Prometheus & Grafana Observability**, and **Jenkins CI/CD Automation**.

```
  Browser UI ─┐                     ┌─ Prometheus ── Grafana (auto-provisioned dashboard)
  curl / API ─┴─> Flask (gunicorn) ─┘   scrapes /metrics every 5s     + alert rules
                   │
    train.py ──> artifacts/ (model, metrics, reference data)
                   │
  Jenkins (port 8080) ──> Automates linting, training quality gate, testing, and deployment
```

---

## Quickstart

Run the complete stack (App, Prometheus, Grafana, and Jenkins):
```bash
docker compose up -d --build
```

### Services Overview
| Service | URL | Default Credentials / Role | Description |
| :--- | :--- | :--- | :--- |
| **Phishing Scanner UI** | `http://localhost:5000` | N/A | Interactive URL inspection, threat scoring, and feedback |
| **Grafana Dashboard** | `http://localhost:3000` | `admin` / `admin` (or anonymous viewer) | Real-time observability dashboard for model drift, latency, accuracy |
| **Prometheus** | `http://localhost:9090` | N/A | Metrics scraper, time-series DB, and alert evaluation engine |
| **Jenkins CI/CD** | `http://localhost:8080` | Initial admin password in logs / container | Automated CI/CD pipeline and scheduled model retraining |

---

## Simulating Traffic & Inducing Data Drift

Generate synthetic production traffic to populate the dashboards:
```bash
# Normal / healthy traffic (legitimate & phishing mix)
python scripts/simulate_traffic.py

# Skewed / adversarial traffic -> triggers Kolmogorov-Smirnov data drift & low accuracy alerts
python scripts/simulate_traffic.py --drift 2.0
```

---

## REST API Reference

### 1. Inspect a URL for Phishing / Threat Risk
```bash
curl -X POST http://localhost:5000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"url": "http://login.paypal.com.account-update.xyz/verify.php?token=9283"}'
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
    "entropy": 4.12,
    "url_length": 62,
    "num_subdomains": 3,
    "has_ip": 0.0,
    "has_suspicious_keywords": 2.0,
    "is_https": 0.0
  },
  "model_version": "20261021-110725"
}
```

### 2. Submit Ground-Truth Label (Human-in-the-Loop Feedback)
```bash
curl -X POST http://localhost:5000/api/feedback \
  -H 'Content-Type: application/json' \
  -d '{"predicted": "phishing", "actual": "phishing"}'
```

### 3. Health & Observability Endpoints
- `GET /health`: Health check and model metadata.
- `GET /metrics`: Prometheus metric exposition format.
- `GET /api/model`: Training metrics, confusion matrix, and feature importances.

---

## What is Monitored

1. **Model Drift (Covariate Shift)**: Two-sample Kolmogorov-Smirnov (KS) hypothesis test computed on a sliding window of the last 100 URL feature vectors against the training baseline (`reference.json`).
2. **Live Accuracy**: Moving accuracy computed from ground-truth feedback labels submitted by analysts or users.
3. **Inference Latency**: p50, p95, and p99 response times via Prometheus histograms.
4. **Classification Distribution**: Real-time ratio of legitimate vs. phishing predictions.
5. **Alerting Rules** (`prometheus/alerts.yml`):
   - `DataDriftDetected`: KS statistic $> 0.4$.
   - `LiveAccuracyLow`: Live accuracy drops below $80\%$.
   - `HighP95Latency`: p95 latency $> 100\text{ms}$.
   - `PhishingAttackSpike`: Elevated burst of malicious URL submissions.

---

## Local Development & Testing

```bash
# 1. Install development dependencies
pip install -r requirements-dev.txt

# 2. Train model & enforce quality gate (MIN_ACCURACY >= 0.85)
python train.py

# 3. Run test suite
pytest -v

# 4. Run application locally
python app.py
```

---

## Jenkins CI/CD Setup

1. Open Jenkins at `http://localhost:8080`.
2. Retrieve the initial admin password:
   ```bash
   docker exec -it mlmon-jenkins-1 cat /var/jenkins_home/secrets/initialAdminPassword
   ```
3. Complete the setup wizard (install suggested plugins).
4. Create a **New Item** $\to$ **Pipeline**.
5. Under **Pipeline Definition**, select **Pipeline script from SCM**, choose **Git**, and provide your repository URL.
6. Jenkins will automatically execute the stages defined in `Jenkinsfile`:
   - Linting with `flake8`
   - Training & Quality Gate enforcement (`MIN_ACCURACY >= 0.85`)
   - Unit testing with `pytest`
   - Docker container build & smoke testing
   - Zero-downtime deployment

---

## Pushing to GitHub

To reflect this project on your GitHub profile:

1. Initialize git and commit:
   ```bash
   git init
   git add .
   git commit -m "feat: Cybersecurity Phishing URL Monitor with Jenkins CI/CD & Grafana"
   ```
2. Create a new repository on [GitHub](https://github.com/new) (e.g., `phishing-url-model-monitoring`).
3. Link and push:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
