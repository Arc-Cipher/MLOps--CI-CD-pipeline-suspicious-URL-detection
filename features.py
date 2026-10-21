"""Feature extraction from URLs for Phishing & Malicious URL detection."""
import math
import re
from collections import Counter
from urllib.parse import urlparse

FEATURES = [
    "url_length",
    "domain_length",
    "path_length",
    "entropy",
    "num_digits",
    "num_dots",
    "num_hyphens",
    "num_subdomains",
    "has_ip",
    "num_query_params",
    "has_suspicious_keywords",
    "is_https",
]

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "update", "secure", "account",
    "banking", "signin", "confirm", "password", "wallet",
    "admin", "support", "security", "authenticate", "billing",
    "recovery", "credential", "suspend", "unlock"
]

IP_PATTERN = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


def calc_entropy(text: str) -> float:
    """Calculate Shannon entropy of character distribution in text."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((cnt / length) * math.log2(cnt / length) for cnt in counts.values())


def extract_features(url: str) -> dict:
    """Extract 12 lexical, structural, and entropy features from a URL string."""
    raw = str(url).strip()
    if not raw.startswith(("http://", "https://")):
        full_url = "http://" + raw
    else:
        full_url = raw

    parsed = urlparse(full_url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    has_ip = 1.0 if IP_PATTERN.match(hostname) else 0.0

    parts = [p for p in hostname.split(".") if p]
    if has_ip:
        subdomains = 0.0
    else:
        subdomains = float(max(0, len(parts) - 2))

    url_lower = raw.lower()
    keyword_hits = sum(1.0 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)
    query_params_count = float(query.count("&") + (1 if query else 0))

    return {
        "url_length": float(len(raw)),
        "domain_length": float(len(hostname)),
        "path_length": float(len(path)),
        "entropy": round(float(calc_entropy(raw)), 4),
        "num_digits": float(sum(c.isdigit() for c in raw)),
        "num_dots": float(raw.count(".")),
        "num_hyphens": float(raw.count("-")),
        "num_subdomains": subdomains,
        "has_ip": has_ip,
        "num_query_params": query_params_count,
        "has_suspicious_keywords": keyword_hits,
        "is_https": 1.0 if raw.lower().startswith("https://") else 0.0,
    }
