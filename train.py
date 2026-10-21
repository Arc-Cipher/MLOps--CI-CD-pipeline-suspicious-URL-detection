"""Train, evaluate, and export the Phishing URL Classifier using CSV dataset + human feedback."""
import csv
import json
import os
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import cross_val_score, train_test_split

from features import FEATURES, extract_features

ART_DIR = os.getenv("MODEL_DIR", "artifacts")
MIN_ACCURACY = float(os.getenv("MIN_ACCURACY", "0.85"))  # quality gate for CI
DATASET_PATH = os.getenv("DATASET_PATH", "phishing_urls_1000.csv")
FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", "data/feedback.csv")

np.random.seed(42)


def load_training_data(dataset_path=DATASET_PATH, feedback_path=FEEDBACK_PATH):
    """Load base CSV dataset and merge with accumulated human feedback."""
    dataset = {}  # url -> label (0 or 1)

    # 1. Resolve and load base CSV dataset
    candidates = [
        dataset_path,
        os.path.join(os.path.dirname(__file__), dataset_path),
        "phishing_urls_1000.csv",
        "/srv/phishing_urls_1000.csv",
    ]
    resolved_dataset_path = None
    for c in candidates:
        if os.path.exists(c):
            resolved_dataset_path = c
            break

    if not resolved_dataset_path:
        raise FileNotFoundError(f"Training dataset CSV not found at {dataset_path}")

    base_count = 0
    with open(resolved_dataset_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("url", "").strip()
            if not url:
                continue
            lbl_raw = row.get("label", row.get("class_name", "0")).strip().lower()
            lbl = 1 if lbl_raw in ("1", "phishing", "malicious") else 0
            dataset[url] = lbl
            base_count += 1

    print(f"Loaded {base_count} base URLs from {resolved_dataset_path}")

    # 2. Check and merge accumulated human feedback
    feedback_candidates = [
        feedback_path,
        os.path.join(os.path.dirname(__file__), feedback_path),
        "data/feedback.csv",
        "/srv/data/feedback.csv",
    ]
    resolved_fb_path = None
    for fb in feedback_candidates:
        if os.path.exists(fb):
            resolved_fb_path = fb
            break

    feedback_count = 0
    if resolved_fb_path and os.path.isfile(resolved_fb_path):
        with open(resolved_fb_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get("url", "").strip()
                if not url or url == "N/A":
                    continue
                actual = row.get("actual", row.get("label", "")).strip().lower()
                lbl = 1 if actual in ("1", "phishing", "malicious") else 0
                # Human feedback updates or appends to the dataset
                dataset[url] = lbl
                feedback_count += 1

        print(f"Loaded {feedback_count} feedback samples from {resolved_fb_path}")
    else:
        print("No prior feedback log found. Proceeding with base dataset.")

    urls = list(dataset.keys())
    labels = [dataset[u] for u in urls]
    print(f"Total training dataset size after merging: {len(urls)} samples "
          f"(Legitimate: {labels.count(0)}, Phishing: {labels.count(1)})")

    return urls, labels, base_count, feedback_count


def main():
    os.makedirs(ART_DIR, exist_ok=True)
    classes = ["legitimate", "phishing"]

    urls, labels, base_count, fb_count = load_training_data()

    print("Extracting features from URLs...")
    feature_rows = []
    for u in urls:
        fdict = extract_features(u)
        feature_rows.append([fdict[col] for col in FEATURES])

    X = np.array(feature_rows)
    y = np.array(labels)

    # Stratified split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print(f"Training RandomForestClassifier on {len(X_tr)} samples with {X.shape[1]} features...")
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)
    p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
    cv = cross_val_score(model, X, y, cv=5).mean()

    metrics = {
        "model_version": time.strftime("%Y%m%d-%H%M%S"),
        "model_type": "RandomForestClassifier",
        "domain": "Cybersecurity Phishing & Malicious URL Detection",
        "base_samples_count": base_count,
        "feedback_samples_count": fb_count,
        "total_training_samples": len(urls),
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(p), 4),
        "recall_macro": round(float(r), 4),
        "f1_macro": round(float(f1), 4),
        "cv_accuracy_5fold": round(float(cv), 4),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "classes": classes,
        "feature_names": FEATURES,
        "report": classification_report(
            y_te, pred, target_names=classes, output_dict=True, zero_division=0
        ),
    }

    reference = {
        "feature_names": FEATURES,
        "data": {n: X_tr[:, i].tolist() for i, n in enumerate(FEATURES)},
        "means": {n: float(X_tr[:, i].mean()) for i, n in enumerate(FEATURES)},
        "min": {n: float(X_tr[:, i].min()) for i, n in enumerate(FEATURES)},
        "max": {n: float(X_tr[:, i].max()) for i, n in enumerate(FEATURES)},
    }

    joblib.dump(model, os.path.join(ART_DIR, "model.joblib"))
    with open(os.path.join(ART_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(ART_DIR, "reference.json"), "w") as f:
        json.dump(reference, f)

    print("=== Model Training Summary ===")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | 5-Fold CV: {cv:.4f} | Version: {metrics['model_version']}")
    print(f"Dataset: {base_count} base + {fb_count} feedback = {len(urls)} total samples")
    print(classification_report(y_te, pred, target_names=classes, zero_division=0))

    if acc < MIN_ACCURACY:
        raise SystemExit(f"Quality gate failed: accuracy {acc:.3f} < {MIN_ACCURACY}")
    print(f"Quality gate passed (accuracy {acc:.4f} >= {MIN_ACCURACY})")


if __name__ == "__main__":
    main()
