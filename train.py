"""Train, evaluate, and export the Phishing URL Classifier + artifacts used by the API."""
import json
import os
import random
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

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

BENIGN_DOMAINS = [
    "google.com", "github.com", "wikipedia.org", "amazon.com", "microsoft.com",
    "apple.com", "netflix.com", "stackoverflow.com", "linkedin.com", "nytimes.com",
    "bbc.co.uk", "medium.com", "python.org", "docker.com", "jenkins.io",
    "grafana.com", "reddit.com", "cloudflare.com", "mit.edu", "nih.gov",
    "kaggle.com", "huggingface.co", "fastapi.tiangolo.com", "pytorch.org",
    "mozilla.org", "w3schools.com", "archive.org", "cnn.com", "theguardian.com",
    "dropbox.com", "slack.com", "zoom.us", "spotify.com", "salesforce.com"
]

BENIGN_PATHS = [
    "", "/", "/search?q=machine+learning", "/docs/v2/getting-started",
    "/user/profile/settings", "/wiki/Cybersecurity", "/posts/2026/10/guide.html",
    "/pricing", "/about-us", "/products/item/10293", "/category/technology",
    "/help/articles/123", "/downloads/latest-release", "/developer/api/reference",
    "/community/forum/topic/4921", "/explore/trending", "/terms-and-conditions",
    "/privacy-policy", "/blog/announcements", "/contact-support"
]

PHISHING_BRANDS = [
    "paypal", "chase-bank", "appleid", "netflix-billing", "google-account",
    "wellsfargo", "amazon-security", "binance-verify", "coinbase-auth",
    "microsoft-live", "bankofamerica", "metamask-wallet", "irs-tax-refund",
    "dhl-tracking-delivery", "fedex-parcel-notice", "usps-redelivery"
]

SUSPICIOUS_TLDS = [
    ".xyz", ".top", ".cc", ".ru", ".info", ".work", ".click", ".live",
    ".tk", ".ml", ".gq", ".cf", ".club", ".fit", ".rest", ".link"
]

PHISHING_PATHS = [
    "/login.php?user=verify", "/signin/auth/session=8372", "/account/update-credentials",
    "/confirm-identity", "/password-reset?token=x83k29a", "/secure/wallet/phrase",
    "/webscr?cmd=_login-submit", "/security-checkpoint/verify.html",
    "/banking/authenticate.aspx", "/portal/suspended-account/unlock",
    "/urgent-action/verification", "/billing/update-payment-method.htm",
    "/claim-reward/identity-check", "/re-activate-account/step1"
]


def generate_synthetic_dataset(n_samples=2400):
    """Generate realistic legitimate and phishing URLs."""
    urls = []
    labels = []
    n_each = n_samples // 2

    # 1. Generate Benign URLs
    for _ in range(n_each):
        dom = random.choice(BENIGN_DOMAINS)
        path = random.choice(BENIGN_PATHS)
        proto = "https://" if random.random() < 0.96 else "http://"
        sub = "www." if random.random() < 0.4 else ("blog." if random.random() < 0.1 else "")
        url = f"{proto}{sub}{dom}{path}"
        if random.random() < 0.3 and path:
            url += f"&ref={random.randint(100, 999)}"
        urls.append(url)
        labels.append(0)  # legitimate

    # 2. Generate Phishing URLs
    for _ in range(n_each):
        p_type = random.choice(["spoof_brand", "ip_address", "subdomain_trick", "dga_entropy"])
        proto = "http://" if random.random() < 0.65 else "https://"

        if p_type == "spoof_brand":
            brand = random.choice(PHISHING_BRANDS)
            tld = random.choice(SUSPICIOUS_TLDS)
            prefix = random.choice(["secure-", "login-", "verify-", "account-", "auth-", ""])
            dom = f"{prefix}{brand}{tld}"
            path = random.choice(PHISHING_PATHS)
            url = f"{proto}{dom}{path}"

        elif p_type == "ip_address":
            ip = f"{random.randint(40, 220)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
            path = random.choice(PHISHING_PATHS)
            url = f"http://{ip}{path}"

        elif p_type == "subdomain_trick":
            brand = random.choice(["paypal.com", "apple.com", "chase.com", "google.com", "bankofamerica.com"])
            tld = random.choice(SUSPICIOUS_TLDS)
            dom = f"login.{brand}.account-notice-{random.randint(10, 99)}{tld}"
            path = random.choice(PHISHING_PATHS)
            url = f"{proto}{dom}{path}"

        else:  # dga_entropy
            chars = "abcdefghijklmnopqrstuvwxyz0123456789"
            rand_str = "".join(random.choice(chars) for _ in range(random.randint(12, 22)))
            tld = random.choice(SUSPICIOUS_TLDS)
            dom = f"{rand_str}{tld}"
            path = random.choice(PHISHING_PATHS)
            url = f"{proto}{dom}{path}"

        urls.append(url)
        labels.append(1)  # phishing

    return urls, labels


def main():
    os.makedirs(ART_DIR, exist_ok=True)
    classes = ["legitimate", "phishing"]

    print("Generating synthetic cybersecurity URL dataset...")
    urls, labels = generate_synthetic_dataset(n_samples=2600)

    print("Extracting features from URLs...")
    feature_rows = []
    for u in urls:
        fdict = extract_features(u)
        feature_rows.append([fdict[col] for col in FEATURES])

    X = np.array(feature_rows)
    y = np.array(labels)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    print(f"Training RandomForestClassifier on {len(X_tr)} samples with {X.shape[1]} features...")
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)
    p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="macro")
    cv = cross_val_score(model, X, y, cv=5).mean()

    metrics = {
        "model_version": time.strftime("%Y%m%d-%H%M%S"),
        "model_type": "RandomForestClassifier",
        "domain": "Cybersecurity Phishing & Malicious URL Detection",
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(p), 4),
        "recall_macro": round(float(r), 4),
        "f1_macro": round(float(f1), 4),
        "cv_accuracy_5fold": round(float(cv), 4),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "classes": classes,
        "feature_names": FEATURES,
        "report": classification_report(
            y_te, pred, target_names=classes, output_dict=True
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
    print(classification_report(y_te, pred, target_names=classes))

    if acc < MIN_ACCURACY:
        raise SystemExit(f"Quality gate failed: accuracy {acc:.3f} < {MIN_ACCURACY}")
    print(f"Quality gate passed (accuracy {acc:.4f} >= {MIN_ACCURACY})")


if __name__ == "__main__":
    main()
