"""Generates grafana/dashboards/model-monitoring.json (keeps the JSON maintainable)."""
import json

DS = {"type": "prometheus", "uid": "prometheus"}
_id = [0]


def _base(kind, title, exprs, x, y, w, h, unit=None, extra=None, legends=None):
    _id[0] += 1
    p = {
        "id": _id[0], "type": kind, "title": title, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"refId": chr(65 + i), "datasource": DS, "expr": e,
                     "legendFormat": (legends[i] if legends else "__auto")}
                    for i, e in enumerate(exprs)],
        "fieldConfig": {"defaults": {"unit": unit or "short"}, "overrides": []},
        "options": {},
    }
    if extra:
        for k, v in extra.items():
            if k in ("thresholds", "min", "max", "decimals"):
                p["fieldConfig"]["defaults"][k] = v
            else:
                p[k] = v
    return p


def stat(title, expr, x, y, unit=None, thresholds=None, **kw):
    extra = {"options": {"reduceOptions": {"calcs": ["lastNotNull"]},
                         "colorMode": "value", "graphMode": "area"}}
    if thresholds:
        extra["thresholds"] = thresholds
    extra.update(kw)
    return _base("stat", title, [expr], x, y, 4, 4, unit, extra)


def ts(title, exprs, x, y, w=12, h=8, unit=None, legends=None, **kw):
    return _base("timeseries", title, exprs, x, y, w, h, unit, kw or None, legends)


def th(*steps):
    return {"mode": "absolute",
            "steps": [{"color": c, "value": v} for c, v in steps]}


panels = [
    stat("Training Accuracy", "model_training_accuracy", 0, 0, "percentunit",
         th(("red", None), ("orange", 0.85), ("green", 0.92))),
    stat("Training F1 (macro)", "model_training_f1_macro", 4, 0, "percentunit",
         th(("red", None), ("orange", 0.85), ("green", 0.92))),
    stat("Live Accuracy (1h feedback)",
         'sum(increase(model_feedback_total{correct="true"}[1h])) / sum(increase(model_feedback_total[1h]))',
         8, 0, "percentunit", th(("red", None), ("orange", 0.8), ("green", 0.9))),
    stat("Total Scans", "sum(model_predictions_total)", 12, 0, "none",
         th(("blue", None))),
    stat("Max Drift (KS)", "max(model_drift_ks_statistic)", 16, 0, "none",
         th(("green", None), ("orange", 0.25), ("red", 0.4)), decimals=2),
    stat("API Errors (5xx/s)", 'sum(rate(http_requests_total{status=~"5.."}[5m])) or vector(0)',
         20, 0, "reqps", th(("green", None), ("red", 0.05))),

    ts("Classification Rate: Legitimate vs Phishing",
       ["sum by (predicted_class) (rate(model_predictions_total[1m]))"],
       0, 4, unit="ops", legends=["{{predicted_class}}"]),
    ts("Inference Latency (p50 / p95 / p99)",
       [f"histogram_quantile({q}, sum by (le) (rate(model_prediction_latency_seconds_bucket[1m])))"
        for q in (0.5, 0.95, 0.99)], 12, 4, unit="s", legends=["p50", "p95", "p99"]),

    ts("Prediction Confidence (p10 / p50)",
       [f"histogram_quantile({q}, sum by (le) (rate(model_prediction_confidence_bucket[2m])))"
        for q in (0.1, 0.5)], 0, 12, unit="percentunit", legends=["p10", "p50"], min=0, max=1),
    ts("Live Accuracy Over Time (Feedback Window)", ["model_live_accuracy"], 12, 12,
       unit="percentunit", min=0, max=1, legends=["live accuracy"]),

    ts("Data Drift: Kolmogorov-Smirnov Test per Feature",
       ["model_drift_ks_statistic"], 0, 20, unit="none", legends=["{{feature}}"], min=0, max=1),
    ts("Feature Means: Live Production Window vs Baseline",
       ["model_feature_window_mean", "model_feature_reference_mean"], 12, 20, unit="none",
       legends=["live {{feature}}", "train {{feature}}"]),

    ts("HTTP Requests by Endpoint & Status",
       ["sum by (endpoint, status) (rate(http_requests_total[1m]))"], 0, 28,
       unit="reqps", legends=["{{endpoint}} {{status}}"]),
    ts("Ground Truth Feedback Activity (per minute)",
       ["sum by (correct) (increase(model_feedback_total[1m]))"], 12, 28,
       legends=["correct={{correct}}"]),
]

dash = {
    "uid": "model-monitoring", "title": "Model Monitoring · Phishing & Malicious URL Classifier",
    "tags": ["ml", "monitoring", "cybersecurity"], "timezone": "browser", "schemaVersion": 39,
    "version": 1, "refresh": "5s", "time": {"from": "now-15m", "to": "now"},
    "panels": panels, "templating": {"list": []}, "annotations": {"list": []},
}

with open("grafana/dashboards/model-monitoring.json", "w") as f:
    json.dump(dash, f, indent=2)
print("wrote", len(panels), "panels to grafana/dashboards/model-monitoring.json")
