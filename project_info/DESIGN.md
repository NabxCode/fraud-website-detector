# Fraud Website Detection System — Design Document

---

## 1. System Overview

A machine learning system that detects phishing websites by analysing URL structure and domain properties. The system provides a Flask REST API and a Gradio web interface for real-time URL analysis.

**Model:** XGBoost trained on 28 engineered features from the UCI Phishing Websites Dataset (11,055 URLs).

---

## 2. System Architecture

```
┌─────────────────────────────────────────────┐
│                  CLIENTS                    │
│   Browser (Gradio UI)  │  API Calls (JSON)  │
└────────────┬───────────┴─────────┬──────────┘
             │                     │
             ▼                     ▼
┌────────────────────────────────────────────┐
│        Flask REST API (port 5001)           │
│         backend_flask/app.py                │
│                                            │
│  GET  /health            POST /predict     │
│  POST /predict/batch     POST /features    │
│  GET  /feature-importance                  │
└────────────────────┬───────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐   ┌─────────────────────┐
│ Override Engine │   │  Feature Extractor  │
│  (5 hard rules) │   │  28 features        │
│  backend_flask/ │   │  backend_flask/     │
│  app.py         │   │  feature_extraction │
└────────┬────────┘   └──────────┬──────────┘
         │                       │
         │              ┌────────▼────────┐
         │              │  XGBoost Model  │
         │              │  model_xgboost/ │
         │              │  phishing_      │
         │              │  detector.pkl   │
         │              └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
             JSON Response
```

---

## 3. Folder Structure

```
FraudWebsiteDetection/
├── backend_flask/          # Flask API + ML core
│   ├── __init__.py
│   ├── app.py              # REST API, override engine, model loading
│   ├── data_loader.py      # UCI ARFF loader, feature dropping
│   ├── feature_extraction.py  # URLFeatureExtractor — 28 live features
│   └── model.py            # PhishingDetector — XGBoost wrapper
├── frontend_gradio/        # Gradio UI
│   ├── gradio_app.py       # UI layout and event wiring
│   ├── gradio_api.py       # HTTP calls to Flask + HTML rendering
│   ├── gradio_data.py      # Static data — examples, labels, guide
│   └── style.css           # All UI styles
├── model_dataset/          # Training data
│   └── Training Dataset.arff
├── model_xgboost/          # Trained model artifacts
│   ├── phishing_detector.pkl
│   └── metrics.json
├── project_info/           # Project documentation
│   ├── README.md
│   ├── DESIGN.md
│   └── requirements.txt
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_model.py
├── .gitignore
└── train_model.py          # Run this to retrain the model
```

---

## 4. Detection Pipeline

Every URL goes through a two-stage pipeline:

### Stage 1 — Hard Override Rules

Five rule-based checks run before the ML model is consulted. If any rule fires, the verdict is returned immediately without calling the model.

| Priority | Rule | Verdict | Confidence |
|----------|------|---------|------------|
| 1 | Private / local IP address | PHISHING | 1.00 |
| 2 | Typosquatting (Levenshtein ≤ 2 from known brand) | PHISHING | 0.99 |
| 3 | URL shortener (bit.ly, tinyurl, etc.) | PHISHING | 0.95 |
| 4 | Suspicious TLD (.xyz, .tk, .top, .ml, etc.) | PHISHING | 0.92 |
| 5 | Trusted domain whitelist (google.com, paypal.com, etc.) | SAFE | 0.99 |

**Why overrides exist:** The XGBoost model was trained on a 2017-era dataset where `SSLfinal_State` dominates at ~37% importance. Signals like URL shorteners and suspicious TLDs are underweighted. Hard rules compensate directly.

**Performance note:** Overrides 3–5 return synthetic feature placeholders instead of triggering a live fetch/WHOIS/SSL check, since the verdict is already decided. This avoids unnecessary 10–12s network calls.

### Stage 2 — XGBoost Model

If no override fires, the trained XGBoost classifier runs on 28 engineered features.

```
Input URL
    │
    ▼
┌──────────────────┐
│   URL Parsing    │  → protocol, domain, path, query string
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────┐
│ Local  │ │ Network      │
│ String │ │ Checks       │
│ Checks │ │              │
│        │ │ • WHOIS      │
│ • UCI  │ │ • SSL/HTTPS  │
│   25   │ │ • DNS        │
│   feat.│ │ • HTTP fetch │
└────┬───┘ └──────┬───────┘
     │             │
     └──────┬──────┘
            ▼
┌───────────────────────┐
│  3 Custom Features    │
│  • is_private_ip      │
│  • url_entropy        │
│  • suspicious_tld     │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  28-feature vector    │
│  → StandardScaler     │
│  → XGBoost predict    │
└───────────┬───────────┘
            ▼
       JSON Response
```

---

## 5. Feature Engineering

### 5.1 UCI Features (25)

5 features from the original 30 were dropped — they depended on discontinued or paid external APIs (Google PageRank, Alexa traffic rankings, backlink APIs).

| Category | Features |
|----------|----------|
| URL Structure | `having_IP_Address`, `URL_Length`, `Shortining_Service`, `having_At_Symbol`, `double_slash_redirecting`, `Prefix_Suffix`, `having_Sub_Domain`, `HTTPS_token` |
| Security | `SSLfinal_State`, `Favicon`, `port` |
| Domain | `Domain_registeration_length`, `age_of_domain`, `DNSRecord`, `Abnormal_URL` |
| Page Content | `Request_URL`, `URL_of_Anchor`, `Links_in_tags`, `SFH`, `Submitting_to_email` |
| Behaviour | `Redirect`, `on_mouseover`, `RightClick`, `popUpWidnow`, `Iframe` |

### 5.2 Custom Engineered Features (3)

| Feature | How Computed (Live) | Training Proxy |
|---------|---------------------|----------------|
| `is_private_ip` | Regex match on domain against RFC1918 ranges | `having_IP_Address == -1.0` |
| `url_entropy` | Shannon entropy of URL character distribution | `URL_Length` value (directional proxy) |
| `suspicious_tld` | Exact TLD match against known phishing TLD list | `having_Sub_Domain` value (rough proxy) |

**Train/serve note:** The UCI dataset contains no raw URL strings, so proxies are used during training. Live inference uses the real computed values. The directional alignment is maintained to avoid train/serve skew.

---

## 6. Model Design

### 6.1 Algorithm

XGBoost — chosen over Random Forest baseline for better performance on this dataset.

| Parameter | Value |
|-----------|-------|
| n_estimators | 100 |
| max_depth | 6 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| eval_metric | logloss |

### 6.2 Training Pipeline

```
UCI ARFF Dataset (11,055 URLs, 30 features)
            │
            ▼
┌───────────────────────┐
│ Drop 5 discontinued   │  → 25 features remain
│ features              │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ Add 3 custom features │  → 28 features total
│ (add_new_features())  │
└──────────┬────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌─────────┐
│  Train  │  │   Val   │
│   80%   │  │   20%   │
└────┬────┘  └────┬────┘
     │             │
     ▼             ▼
┌──────────────────────┐
│  StandardScaler      │
│  → XGBoost.fit()     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Evaluate on val set │
│  + 5-fold CV         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Save to             │
│  model_xgboost/      │
│  phishing_detector   │
│  .pkl + metrics.json │
└──────────────────────┘
```

### 6.3 Performance

| Metric | Score |
|--------|-------|
| Accuracy | 95.12% |
| Precision | 94.32% |
| Recall | 97.08% |
| F1 Score | 95.68% |
| ROC-AUC | 99.18% |
| CV Accuracy | 94.60% ± 0.70% |

### 6.4 Top Feature Importances

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | SSLfinal_State | 37.39% |
| 2 | URL_of_Anchor | 23.20% |
| 3 | Prefix_Suffix | 7.37% |
| 4 | SFH | 3.23% |
| 5 | Links_in_tags | 2.66% |
| 8 | suspicious_tld (custom) | 1.55% |
| 13 | url_entropy (custom) | 1.27% |

---

## 7. API Design

### 7.1 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns `{status, model_loaded}` |
| `/predict` | POST | Single URL — returns verdict + features |
| `/predict/batch` | POST | Multiple URLs — returns list of verdicts |
| `/features` | POST | Feature extraction only, no prediction |
| `/feature-importance` | GET | Live feature importance rankings from model |

### 7.2 Predict Response Format

```json
{
  "url": "https://suspicious-site.com/login",
  "is_phishing": true,
  "confidence": 0.92,
  "phishing_probability": 0.92,
  "override": null,
  "features": {
    "SSLfinal_State": -1.0,
    "having_IP_Address": 1.0,
    "url_entropy": -1.0,
    "suspicious_tld": 1.0,
    "...": "..."
  }
}
```

### 7.3 Override Response Format

When a hard rule fires, `override` is populated and `features` contains synthetic placeholders:

```json
{
  "url": "http://bit.ly/abc123",
  "is_phishing": true,
  "confidence": 0.95,
  "phishing_probability": 0.95,
  "override": "url_shortener",
  "features": { "SSLfinal_State": -1.0, "...": -1.0 }
}
```

---

## 8. Component Overview

| File | Role |
|------|------|
| `backend_flask/app.py` | Flask API — routes, override engine, model loading |
| `backend_flask/feature_extraction.py` | Computes all 28 features live from a URL |
| `backend_flask/model.py` | XGBoost wrapper — train, predict, save, load |
| `backend_flask/data_loader.py` | Loads UCI ARFF, drops discontinued features |
| `frontend_gradio/gradio_app.py` | Gradio UI layout and event wiring |
| `frontend_gradio/gradio_api.py` | HTTP calls to Flask + HTML rendering helpers |
| `frontend_gradio/gradio_data.py` | Static data — feature guide, examples, labels |
| `frontend_gradio/style.css` | All CSS for the Gradio UI |
| `train_model.py` | Full training script — run this to retrain |

---

## 9. Error Handling

| Situation | Behaviour |
|-----------|-----------|
| Invalid or missing URL | 400 Bad Request |
| Model not loaded | 503 Service Unavailable |
| WHOIS / SSL / DNS timeout | Graceful degradation — feature set to neutral, prediction continues |
| Unhandled exception | 500 with error message |

---

## 10. Known Limitations

1. **Brand-name prefix trick** — `apple-id-locked.support-center.info` passes the typosquat check because the brand match is exact (distance 0), which is treated as legitimate. Fixing this requires a brand-to-canonical-domain whitelist.

2. **Fake subdomain trick** — `accounts.google.com.security-check.ru` has `security-check.ru` as the real domain. No current feature detects a brand name buried as an inner subdomain label of an unrelated domain.

3. **2017-era training data** — The UCI dataset reflects phishing patterns from 2017. Modern phishing techniques may not be well-represented.

---

**Document Version:** 2.0.0
**Last Updated:** July 2026
