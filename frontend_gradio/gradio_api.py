"""
API-client and HTML-rendering helper functions for the Gradio UI:
talking to the Flask backend (/predict, /predict/batch, /health,
/feature-importance) and turning the responses into the HTML/text
blocks the UI displays.
Split out of gradio_app.py to keep that file focused on layout/wiring.
"""

import os
import re
import json
import requests
from typing import Tuple


from .gradio_data import (
    FEATURE_GUIDE, FEATURE_LABELS, FEATURE_CATEGORIES,
    FALLBACK_FEATURE_IMPORTANCE, IMPORTANCE_SHORT_LABELS,
)

FLASK_PORT = os.environ.get('FLASK_API_PORT', '5001')
API_BASE_URL = f"http://localhost:{FLASK_PORT}"

# Path to the metrics file written by train_model.py.
# Resolves relative to this file's location so it works regardless of
# the working directory the user launches Gradio from.
_METRICS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'model_xgboost', 'metrics.json'
)

# Maximum number of URLs allowed in a single batch request.
_BATCH_LIMIT = 20

# ------------------------------------------------------------------
# Input Validation
# ------------------------------------------------------------------

_URL_PATTERN = re.compile(
    r'^(https?://)'            # must start with http:// or https://
    r'([a-zA-Z0-9\-\.]+)'     # domain / IP
    r'(\.[a-zA-Z]{2,}|'       # TLD (letters)
    r'\d{1,3})'                # or numeric (for bare IPs like 192.168.x.x)
    r'(:\d+)?'                 # optional port
    r'(/[^\s]*)?$'             # optional path
)


def _is_valid_url(url: str) -> bool:
    """
    Returns True if the string looks like a single, well-formed URL.
    Must start with http:// or https://.
    Rejects plain text, bare domain names, multi-line input, etc.
    """
    url = url.strip()
    # Reject anything with a newline — user pasted multiple URLs
    if '\n' in url or '\r' in url:
        return False
    return bool(_URL_PATTERN.match(url))


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def _load_metrics() -> dict:
    """
    Read metrics.json from disk and return its contents as a dict.
    Falls back to the values that were current at the time of the last
    known training run if the file is missing or unreadable.
    """
    fallback = {
        "accuracy":  0.9512,
        "precision": 0.9432,
        "recall":    0.9708,
        "f1_score":  0.9568,
    }
    try:
        with open(_METRICS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Validate that the expected keys are present and numeric
        for key in ("accuracy", "precision", "recall", "f1_score"):
            if key not in data or not isinstance(data[key], (int, float)):
                return fallback
        return data
    except Exception:
        return fallback


# ------------------------------------------------------------------
# Feature Importance
# ------------------------------------------------------------------

def fetch_feature_importance():
    """
    Fetches live feature importances from GET /feature-importance.
    Returns (importance_items, is_live) where importance_items is a list of
    (key, value) tuples sorted highest-first, and is_live tells the caller
    whether this came from the API or the hardcoded fallback.
    """
    try:
        r = requests.get(f"{API_BASE_URL}/feature-importance", timeout=10)
        if r.status_code == 200:
            data = r.json().get("feature_importances", {})
            if data:
                items = sorted(data.items(), key=lambda x: x[1], reverse=True)
                return items, True
        return FALLBACK_FEATURE_IMPORTANCE, False
    except Exception:
        return FALLBACK_FEATURE_IMPORTANCE, False


# ------------------------------------------------------------------
# API Health
# ------------------------------------------------------------------

def check_api_health() -> str:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if r.status_code == 200:
            return "✅  API is running and model is loaded" if r.json().get("model_loaded") else "⚠️  API running but model not loaded"
        return f"❌  API returned status {r.status_code}"
    except requests.exceptions.ConnectionError:
        return f"❌  Cannot connect to API at {API_BASE_URL}. Start Flask first."
    except Exception as e:
        return f"❌  Error: {str(e)}"


# ------------------------------------------------------------------
# Single URL Analysis  (Tab 1)
# ------------------------------------------------------------------

def analyze_url(url: str) -> Tuple[str, str]:
    # ── Empty check ────────────────────────────────────────────────
    if not url or not url.strip():
        return "⚠️  Please enter a URL.", ""

    # ── Multi-line check (user pasted several URLs) ────────────────
    lines = [l.strip() for l in url.strip().splitlines() if l.strip()]
    if len(lines) > 1:
        return (
            "⚠️  This tab accepts one URL at a time.\n\n"
            "You entered multiple lines. Please enter a single URL,\n"
            "or use the 📦 Batch Analysis tab for multiple URLs.",
            ""
        )

    # ── Format validation ──────────────────────────────────────────
    if not _is_valid_url(url):
        return (
            "⚠️  Invalid URL format.\n\n"
            "A valid URL must start with http:// or https://\n\n"
            "Examples:\n"
            "  https://example.com\n"
            "  http://suspicious-site.tk/login\n\n"
            "Plain text, bare domain names (e.g. google.com),\n"
            "and IP addresses without a scheme are not accepted.",
            ""
        )

    # ── API call ───────────────────────────────────────────────────
    try:
        r = requests.post(f"{API_BASE_URL}/predict",
                          json={"url": url.strip()}, timeout=30)
        if r.status_code == 503:
            return "❌  Model not loaded. Please train the model first.", ""
        if r.status_code != 200:
            return f"❌  API error (status {r.status_code})", ""

        data = r.json()
        is_phishing = data.get("is_phishing", False)
        confidence = data.get("confidence", 0)
        phishing_prob = data.get("phishing_probability", 0)
        features = data.get("features", {})

        if is_phishing:
            result = (
                f"🔴  PHISHING DETECTED\n\n"
                f"Confidence           :  {confidence:.1%}\n"
                f"Phishing Probability :  {phishing_prob:.1%}\n\n"
                f"⚠️  Do NOT enter any personal information on this site."
            )
        else:
            result = (
                f"🟢  LEGITIMATE WEBSITE\n\n"
                f"Confidence           :  {confidence:.1%}\n"
                f"Phishing Probability :  {phishing_prob:.1%}\n\n"
                f"✅  This URL appears to be legitimate."
            )

        checks = [
            ("SSLfinal_State",              "SSL/HTTPS certificate failed or untrusted"),
            ("having_IP_Address",           "IP address used instead of a domain name"),
            ("is_private_ip",               "Private/local IP address used as domain"),
            ("age_of_domain",               "Domain is less than 6 months old"),
            ("having_At_Symbol",            "@ symbol in URL (hides real destination)"),
            ("Prefix_Suffix",
             "Hyphen in domain name (prefix-suffix trick)"),
            ("Shortining_Service",
             "URL shortener used — real destination hidden"),
            ("DNSRecord",                   "No DNS record found for this domain"),
            ("Domain_registeration_length", "Domain registered for less than 1 year"),
            ("Iframe",                      "Hidden iframes detected on page"),
            ("Submitting_to_email",         "Form submits data to an email address"),
            ("url_entropy",                 "URL contains random-looking characters"),
            ("suspicious_tld",
             "Domain ending associated with phishing (.xyz, .tk…)"),
            ("popUpWidnow",                 "Page opens pop-up windows"),
            ("RightClick",                  "Right-click is disabled on this page"),
            ("on_mouseover",
             "Page manipulates browser status bar on hover"),
        ]
        warnings = [msg for key,
                    msg in checks if features.get(key, 1.0) == -1.0]
        warning_text = "\n".join(
            f"•  {w}" for w in warnings) if warnings else "•  No major warning indicators detected."
        return result, warning_text

    except requests.exceptions.ConnectionError:
        return f"❌  Cannot connect to API at {API_BASE_URL}. Start Flask first.", ""
    except Exception as e:
        return f"❌  Error: {str(e)}", ""


def example_loaded_toast(url: str) -> str:
    safe_url = (url or "").strip().replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<div class="example-loaded-toast">✓ Loaded '
        f'<span class="example-loaded-url">{safe_url}</span></div>'
    )


# ------------------------------------------------------------------
# Batch Analysis  (Tab 2)
# ------------------------------------------------------------------

def batch_analyze(urls_text: str):
    """
    Validates and sends a list of URLs to /predict/batch.
    Rules enforced before the API call:
      - Input must not be empty.
      - Maximum {_BATCH_LIMIT} URLs per request.
      - Every line must be a valid http:// or https:// URL.
    """
    # ── Empty check ────────────────────────────────────────────────
    if not urls_text or not urls_text.strip():
        return [["⚠️  Please enter at least one URL (one per line).", "", "", ""]]

    urls = [u.strip() for u in urls_text.strip().splitlines() if u.strip()]
    if not urls:
        return [["⚠️  Please enter at least one URL (one per line).", "", "", ""]]

    # ── Batch size cap ─────────────────────────────────────────────
    if len(urls) > _BATCH_LIMIT:
        return [[
            f"⚠️  Too many URLs — maximum allowed is {_BATCH_LIMIT}.\n"
            f"You entered {len(urls)}. Please remove {len(urls) - _BATCH_LIMIT} URL(s) and try again.",
            "", "", ""
        ]]

    # ── Per-URL format validation ──────────────────────────────────
    invalid = [(i + 1, u) for i, u in enumerate(urls) if not _is_valid_url(u)]
    if invalid:
        lines = "\n".join(f"  Line {n}: {u}" for n, u in invalid[:5])
        extra = f"\n  … and {len(invalid) - 5} more." if len(invalid) > 5 else ""
        return [[
            f"⚠️  {len(invalid)} invalid URL(s) found. Fix them and retry.\n"
            f"Each URL must start with http:// or https://\n\n"
            f"{lines}{extra}",
            "", "", ""
        ]]

    # ── API call ───────────────────────────────────────────────────
    try:
        r = requests.post(f"{API_BASE_URL}/predict/batch",
                          json={"urls": urls}, timeout=60)
        if r.status_code == 503:
            return [["❌  Model not loaded. Please train the model first.", "", "", ""]]
        if r.status_code != 200:
            return [[f"❌  API error (status {r.status_code})", "", "", ""]]

        data = r.json()
        results = data.get("results", data if isinstance(data, list) else [])

        rows = []
        for i, item in enumerate(results):
            url = item.get("url", urls[i] if i < len(urls) else "")
            is_phishing = item.get("is_phishing", False)
            confidence = item.get("confidence", 0)
            phishing_prob = item.get("phishing_probability", 0)
            verdict = "🔴 PHISHING" if is_phishing else "🟢 SAFE"
            rows.append(
                [url, verdict, f"{confidence:.1%}", f"{phishing_prob:.1%}"])
        return rows if rows else [["No results returned by the API.", "", "", ""]]

    except requests.exceptions.ConnectionError:
        return [[f"❌  Cannot connect to API at {API_BASE_URL}. Start Flask first.", "", "", ""]]
    except Exception as e:
        return [[f"❌  Error: {str(e)}", "", "", ""]]


# ------------------------------------------------------------------
# Feature Extraction  (Tab 3)
# ------------------------------------------------------------------

def extract_features_display(url: str) -> str:
    # ── Empty check ────────────────────────────────────────────────
    if not url or not url.strip():
        return '<p style="color:#96a0b8">Enter a URL above and click Extract to see its 28 features.</p>'

    # ── Multi-line check ───────────────────────────────────────────
    lines = [l.strip() for l in url.strip().splitlines() if l.strip()]
    if len(lines) > 1:
        return (
            '<p style="color:#f59e0b">⚠️  This tab accepts one URL at a time. '
            'You entered multiple lines — please enter a single URL.</p>'
        )

    # ── Format validation ──────────────────────────────────────────
    if not _is_valid_url(url):
        return (
            '<p style="color:#f59e0b">⚠️  Invalid URL format. '
            'A valid URL must start with <code>http://</code> or <code>https://</code><br>'
            'Example: <code>https://example.com</code></p>'
        )

    # ── API call ───────────────────────────────────────────────────
    try:
        r = requests.post(f"{API_BASE_URL}/predict",
                          json={"url": url.strip()}, timeout=30)
        if r.status_code == 503:
            return '<p style="color:#fb5170">❌ Model not loaded. Please train the model first.</p>'
        if r.status_code != 200:
            return f'<p style="color:#fb5170">❌ API error (status {r.status_code})</p>'

        features = r.json().get("features", {})
        if not features:
            return '<p style="color:#96a0b8">No feature data returned for this URL.</p>'

        safe_url = url.strip().replace("<", "&lt;").replace(">", "&gt;")
        blocks = [
            f'<p style="margin:0 0 14px;color:#96a0b8;font-size:0.9rem">Features for '
            f'<span style="color:#22d3ee;font-family:var(--mono)">{safe_url}</span></p>'
        ]
        for category, keys in FEATURE_CATEGORIES.items():
            rows = []
            for key in keys:
                val = features.get(key, None)
                if val is None:
                    continue
                is_flagged = val == -1.0
                label = FEATURE_LABELS.get(key, key)
                pill_class = "feat-val-bad" if is_flagged else "feat-val-good"
                rows.append(
                    f'<div class="feat-row"><span class="feat-row-name">{label}</span>'
                    f'<span class="feat-row-key">{key}</span>'
                    f'<span class="feat-val {pill_class}">{val}</span></div>'
                )
            blocks.append(
                f'<div class="feat-category"><div class="feat-category-title">{category}</div>{"".join(rows)}</div>'
            )
        return f'<div class="feature-extract-grid">{"".join(blocks)}</div>'

    except requests.exceptions.ConnectionError:
        return f'<p style="color:#fb5170">❌ Cannot connect to API at {API_BASE_URL}. Start Flask first.</p>'
    except Exception as e:
        return f'<p style="color:#fb5170">❌ Error: {str(e)}</p>'


# ------------------------------------------------------------------
# Model Insights — metrics are read live from metrics.json each time
# this function is called, so retraining the model automatically
# updates the UI without any manual edits.
# Feature importance is fetched live from the API.
# ------------------------------------------------------------------

def build_importance_bars_html() -> str:
    importance_items, is_live = fetch_feature_importance()
    max_val = max(v for _, v in importance_items)
    rows = []
    for rank, (key, val) in enumerate(importance_items, 1):
        pct = (val / max_val) * 100
        short = IMPORTANCE_SHORT_LABELS.get(key, key)
        display = f'{key} <span class="importance-short">({short})</span>'
        rows.append(f"""
<div class="importance-row">
  <span class="importance-rank">{rank}</span>
  <span class="importance-label">{display}</span>
  <div class="importance-bar-track"><div class="importance-bar-fill" style="width:{pct:.1f}%"></div></div>
  <span class="importance-pct">{val*100:.2f}%</span>
</div>""")

    status = (
        '<p style="color:#34d399;font-size:0.8rem;margin:0 0 12px">🟢 Live from the trained model (/feature-importance)</p>'
        if is_live else
        '<p style="color:#96a0b8;font-size:0.8rem;margin:0 0 12px">⚠️ API unreachable or model not loaded — showing last-known values.</p>'
    )
    return f'{status}<div class="importance-chart">{"".join(rows)}</div>'


def build_model_insights_html() -> str:
    """
    Renders the four metric cards (Accuracy, Precision, Recall, F1).
    Values are read from metrics.json at call time so they stay in sync
    with the trained model. Falls back to the last-known values if the
    file is missing or unreadable.
    """
    metrics = _load_metrics()

    accuracy = metrics.get("accuracy",  0.0)
    precision = metrics.get("precision", 0.0)
    recall = metrics.get("recall",    0.0)
    f1 = metrics.get("f1_score",  0.0)

    metrics_live = os.path.isfile(_METRICS_PATH)
    source_note = (
        '<p style="color:#34d399;font-size:0.8rem;margin:0 0 12px">'
        '🟢 Metrics loaded from model_xgboost/metrics.json</p>'
        if metrics_live else
        '<p style="color:#96a0b8;font-size:0.8rem;margin:0 0 12px">'
        '⚠️ metrics.json not found — showing last-known values.</p>'
    )

    return f"""
{source_note}
<div class="insights-grid">
  <div class="insight-card"><div class="insight-value">{accuracy:.2%}</div><div class="insight-label">Accuracy</div></div>
  <div class="insight-card"><div class="insight-value">{precision:.2%}</div><div class="insight-label">Precision</div></div>
  <div class="insight-card"><div class="insight-value">{recall:.2%}</div><div class="insight-label">Recall</div></div>
  <div class="insight-card"><div class="insight-value">{f1:.2%}</div><div class="insight-label">F1 Score</div></div>
</div>
<p style="color:#96a0b8;font-size:0.9rem;margin:18px 0 4px">
  <strong style="color:#f1f5f9">Model:</strong> XGBoost &nbsp;|&nbsp;
  <strong style="color:#f1f5f9">Dataset:</strong> UCI Phishing Dataset — 11,055 URLs &nbsp;|&nbsp;
  <strong style="color:#f1f5f9">Features:</strong> 28 engineered signals
</p>
"""


def build_feature_accordion_html() -> str:
    """Renders all 28 features as native HTML <details> accordions."""
    items = []
    for i, (label, explanation) in enumerate(FEATURE_GUIDE, 1):
        items.append(f"""
<details>
  <summary>
    <span><span class="feat-num">{i}</span>{label}</span>
    <span class="feat-arrow">▼</span>
  </summary>
  <div class="feat-body">{explanation}</div>
</details>""")
    return f'<div class="feature-accordion">{"".join(items)}</div>'
