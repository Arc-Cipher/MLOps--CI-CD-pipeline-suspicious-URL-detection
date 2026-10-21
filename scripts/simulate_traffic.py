"""Generate traffic + feedback for the Phishing & Malicious URL Classifier.

  python scripts/simulate_traffic.py                 # healthy traffic
  python scripts/simulate_traffic.py --drift 2.0     # high-entropy anomalous URLs -> triggers drift & accuracy alerts
"""
import argparse
import json
import random
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:5000")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--rps", type=float, default=5)
ap.add_argument("--drift", type=float, default=0.0,
                help="injects high-entropy anomalous and distorted URLs")
ap.add_argument("--label-noise", type=float, default=0.05,
                help="fraction of wrong feedback labels")
a = ap.parse_args()

BENIGN_SAMPLES = [
    "https://www.google.com/search?q=machine+learning+observability",
    "https://github.com/torvalds/linux/commits/master",
    "https://en.wikipedia.org/wiki/Information_security",
    "https://www.amazon.com/dp/B08N5WRWNW",
    "https://stackoverflow.com/questions/tagged/python",
    "https://docs.python.org/3/library/urllib.html",
    "https://netflix.com/browse/genre/839338",
    "https://www.nytimes.com/section/technology",
    "https://hub.docker.com/_/python",
    "https://grafana.com/docs/grafana/latest/dashboards/",
    "https://prometheus.io/docs/prometheus/latest/querying/basics/",
    "https://apple.com/macbook-pro/",
]

PHISHING_SAMPLES = [
    "http://login.paypal.com.account-update.xyz/verify.php?token=9283",
    "http://192.168.1.105/bank-security/login.html?user=auth",
    "http://appleid-verify.support-device.cc/signin/auth?session=x83",
    "http://secure-chase-banking.top/portal/update-credentials.htm",
    "http://netflix-billing-alert.info/account/reactivate?id=9281",
    "http://accounts.google.com-recovery.ml/pass-reset?id=9382",
    "http://wellsfargo-security-checkpoint.club/auth/verify.aspx",
    "http://metamask-wallet-phrase.rest/claim/identity-check",
    "http://amazon-security-alert.work/order-confirmation?uid=8271",
    "http://binance-kyc-verify.biz/portal/suspended-account/unlock",
]

CLASSES = ["legitimate", "phishing"]


def post(path, body):
    req = urllib.request.Request(
        a.url + path,
        json.dumps(body).encode(),
        {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


print(f"Starting traffic generator against {a.url} (n={a.n}, rps={a.rps}, drift={a.drift})...")
for i in range(a.n):
    # Choose legitimate or phishing URL
    if random.random() < 0.5:
        target_url = random.choice(BENIGN_SAMPLES)
        actual = "legitimate"
    else:
        target_url = random.choice(PHISHING_SAMPLES)
        actual = "phishing"

    # Add realistic variations or query strings
    if random.random() < 0.4:
        target_url += f"&var={random.randint(100, 999)}"

    # If drift is enabled, distort URLs with abnormal length, excessive subdomains, or high entropy
    if a.drift > 0:
        chars = "!@#$%^&*()_+=-[]{}|;:,.<>?1234567890abcdef"
        noise = "".join(random.choice(chars) for _ in range(int(25 * a.drift)))
        subdomains = ".".join(["anomalous" + str(k) for k in range(int(3 * a.drift))])
        target_url = f"http://{subdomains}.drifted-synthetic-pattern.xyz/deep/path/{noise}?token={noise}"

    try:
        res = post("/api/predict", {"url": target_url})

        # Inject noise into human feedback if drift is high or label-noise applies
        feedback_actual = actual
        if random.random() < a.label_noise + (0.35 if a.drift else 0):
            feedback_actual = "phishing" if actual == "legitimate" else "legitimate"

        post("/api/feedback", {
            "url": target_url,
            "predicted": res["prediction"],
            "actual": feedback_actual
        })

        if (i + 1) % 50 == 0:
            pred_class = res['prediction']
            risk = res.get('risk_score', 0)
            print(f"[{i + 1}/{a.n}] sent: '{target_url[:40]}...' -> {pred_class} (risk: {risk}%)")

    except Exception as e:
        print(f"Request {i+1} failed: {e}")

    time.sleep(1 / a.rps)

print("Traffic simulation completed successfully.")
