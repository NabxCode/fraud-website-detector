# Fraud Website Detection System

A machine learning system that detects phishing websites by analysing URL structure and domain properties in real time.

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 95.12% |
| Precision | 94.32% |
| Recall | 97.08% |
| F1 Score | 95.68% |
| ROC-AUC | 99.18% |
| CV Accuracy | 94.60% ± 0.70% |

Trained on the [UCI Phishing Websites Dataset](https://archive.ics.uci.edu/ml/datasets/phishing+websites) — 11,055 URLs, 28 engineered features.

---

## How It Works

Detection runs in two stages:

### Stage 1 — Hard Override Rules (checked first, before the model)

Five rule-based checks run on every URL before the ML model is consulted:

| Priority | Rule | Result | Confidence |
|----------|------|--------|------------|
| 1 | Private or local IP address | PHISHING | 1.00 |
| 2 | Typosquatting (Levenshtein distance ≤ 2 from known brand) | PHISHING | 0.99 |
| 3 | URL shortener (bit.ly, tinyurl, etc.) | PHISHING | 0.95 |
| 4 | Suspicious TLD (.xyz, .tk, .top, etc.) | PHISHING | 0.92 |
| 5 | Trusted domain whitelist (google.com, paypal.com, etc.) | SAFE | 0.99 |

These overrides exist because the XGBoost model was trained on a 2017-era dataset where `SSLfinal_State` dominates at ~37% importance. Signals like URL shorteners and suspicious TLDs are underweighted. Hard rules compensate directly.

### Stage 2 — XGBoost Model

If no override fires, a trained XGBoost classifier predicts using 28 engineered features:

- 25 features from the UCI dataset (SSL state, domain age, subdomain depth, special character counts, WHOIS data, etc.)
- 3 custom-engineered features: `is_private_ip`, `url_entropy`, `suspicious_tld`

---

## Project Structure

```
FraudWebsiteDetection/
├── backend_flask/
│   ├── __init__.py
│   ├── app.py                  # Flask REST API (port 5001)
│   ├── data_loader.py          # UCI ARFF dataset loader
│   ├── feature_extraction.py   # URLFeatureExtractor — 28 features
│   └── model.py                # PhishingDetector class (XGBoost wrapper)
├── frontend_gradio/
│   ├── gradio_app.py           # Gradio UI layout and wiring
│   ├── gradio_api.py           # API calls and HTML rendering helpers
│   ├── gradio_data.py          # Static data for Gradio UI
│   └── style.css               # Gradio UI styles
├── model_dataset/
│   └── Training Dataset.arff   # UCI dataset (11,055 URLs)
├── model_xgboost/
│   ├── phishing_detector.pkl   # Trained XGBoost model
│   └── metrics.json            # Saved training metrics
├── project_info/
│   ├── README.md
│   ├── DESIGN.md
│   └── requirements.txt
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_model.py
├── .gitignore
└── train_model.py              # Training script
```

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/your-username/FraudWebsiteDetection.git
cd FraudWebsiteDetection

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r project_info/requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This loads `model_dataset/Training Dataset.arff`, engineers 3 custom features, trains XGBoost, and saves the model to `model_xgboost/phishing_detector.pkl` and metrics to `model_xgboost/metrics.json`.

### 3. Start the Services

Open two terminals from the project root:

```bash
# Terminal 1 — Flask API
python -m backend_flask.app

# Terminal 2 — Gradio UI
python -m frontend_gradio.gradio_app
```

- Flask API: `http://localhost:5001`
- Gradio UI: `http://localhost:7860`

---

## API Endpoints

### Health Check
```
GET /health
```

### Predict Single URL
```
POST /predict
Content-Type: application/json
{ "url": "https://suspicious-site.com/login" }
```

Response:
```json
{
  "url": "https://suspicious-site.com/login",
  "is_phishing": true,
  "confidence": 0.92,
  "phishing_probability": 0.92,
  "features": { "SSLfinal_State": -1.0, "url_entropy": -1.0, "...": "..." }
}
```

### Batch Prediction
```
POST /predict/batch
Content-Type: application/json
{ "urls": ["https://google.com", "https://suspicious-site.com/login"] }
```

### Extract Features Only
```
POST /features
Content-Type: application/json
{ "url": "https://example.com" }
```

### Feature Importance
```
GET /feature-importance
```

---

## Gradio Interface

The Gradio UI runs on `http://localhost:7860` and has 4 tabs:

| Tab | Description |
|-----|-------------|
| 🔍 Single URL Analysis | Check one URL with full result and warning indicators |
| 📦 Batch Analysis | Process multiple URLs at once |
| 🧬 Feature Extraction | View all 28 extracted features grouped by category |
| 📊 Model Insights | XGBoost performance metrics and feature importance rankings |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Known Limitations

Two URL patterns currently evade detection. These are documented as design gaps, not bugs:

1. **Brand-name prefix trick** — e.g. `apple-id-locked.support-center.info`
   The typosquat detector excludes exact brand-name matches (distance 0) as legitimate. A domain using a brand name as a prefix on an unrelated TLD bypasses this. Resolving it properly requires a curated brand-to-domain whitelist.

2. **Fake subdomain trick** — e.g. `accounts.google.com.security-check.ru`
   The real domain here is `security-check.ru`. No current feature detects a known brand name buried as an inner label of an unrelated domain. This would require a dedicated subdomain-impersonation feature.

Both limitations stem from the training dataset being based on 2017-era URL patterns. They are intentionally documented rather than silently patched.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Model | XGBoost 3.3.0 |
| Feature Engineering | Python, WHOIS, socket (DNS), SSL |
| API | Flask |
| UI | Gradio 6.10.0 |

---

## Dataset

UCI Phishing Websites Dataset — Mohammad, Thabtah, McCluskey (2012) — [Link](https://archive.ics.uci.edu/ml/datasets/phishing+websites)

- 11,055 URLs
- 30 original features (5 deprecated API-dependent features dropped during training)
- 3 custom features added: `is_private_ip`, `url_entropy`, `suspicious_tld`

---

## License

This project is for educational and portfolio purposes.