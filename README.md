# Fraud Website Detection System

A machine learning-based system for detecting phishing websites using URL analysis.

## Features

- **Feature Extraction**: Extracts 20+ features from URLs including:
  - URL length and special character counts
  - HTTPS presence and SSL certificate validation
  - IP address detection
  - Domain age via WHOIS lookup
  - Subdomain analysis
  - Typosquatting detection using Levenshtein distance

- **Machine Learning Models**:
  - Random Forest (baseline - easy to interpret)
  - XGBoost (advanced performance)

- **API**: Flask REST API for real-time URL analysis

## Project Structure

```
FraudWebsiteDetection/
├── src/
│   ├── __init__.py
│   ├── feature_extraction.py  # URL feature extraction
│   ├── model.py                # ML model implementations
│   ├── data_loader.py          # Dataset loading utilities
│   └── app.py                  # Flask API
├── tests/
│   ├── __init__.py
│   ├── test_feature_extraction.py
│   ├── test_model.py
│   └── test_api.py
├── static/
│   └── index.html              # Web UI
├── data/                       # Dataset storage
├── models/                     # Saved models
├── train.py                    # Training script
├── demo.py                     # Interactive demo
├── cli.py                      # Command-line tool
├── gradio_app.py               # Gradio test interface
├── run.sh                      # Quick start script
├── download_model.py           # Get pre-trained model
├── config.py                   # Configuration settings
├── DESIGN.md                   # System design document
├── requirements.txt            # Python dependencies
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a Trained Model

#### Option A: Quick Synthetic Model (30 seconds - for demo)
```bash
python download_model.py --quick
```
Creates a synthetic model instantly for testing. Good for demonstrations.

#### Option B: Train on Sample Data (2-3 minutes - recommended)
```bash
python train.py --sample --n-samples 200
```
Creates model from sample phishing and legitimate URLs.

#### Option C: Use UCI Phishing Dataset (best accuracy)
```bash
# 1. Download from UCI ML Repository:
# https://archive.ics.uci.edu/ml/datasets/phishing+websites
# Download "PhishingData.arff"

# 2. Save to data/PhishingData.arff

# 3. Train (very fast - features are pre-extracted!)
python train.py --dataset data/PhishingData.arff --dataset-type uci
```

#### Option D: Use Your Own Dataset
```bash
python train.py --dataset data/my_urls.csv --dataset-type custom
```
CSV should have columns: `url`, `label` (where label is 'phishing' or 'legitimate')

### 3. Run the API

```bash
python src/app.py
```

The API will be available at `http://localhost:5000`

### 4. Test with Gradio Interface (Optional)

For an interactive UI to test the API:

```bash
python gradio_app.py
```

The Gradio interface will be available at `http://localhost:7860`

### Quick Start with run.sh (Recommended)

Use the provided script to easily start services:

```bash
# Start both Flask API and Gradio (recommended)
./run.sh

# Start only Flask API
./run.sh flask

# Start only Gradio (Flask must be running)
./run.sh gradio

# Show help
./run.sh --help
```

The script will:
- Check Python environment and dependencies
- Train model if not found
- Start services and verify they're ready
- Handle cleanup on exit (Ctrl+C)

## API Endpoints

### Health Check
```bash
GET /health
```

### Predict Single URL
```bash
POST /predict
Content-Type: application/json

{
  "url": "https://suspicious-site.com/login"
}
```

Response:
```json
{
  "url": "https://suspicious-site.com/login",
  "is_phishing": true,
  "confidence": 0.92,
  "phishing_probability": 0.92,
  "features": {...}
}
```

### Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
  "urls": [
    "https://google.com",
    "https://suspicious-site.com/login"
  ]
}
```

### Extract Features
```bash
POST /features
Content-Type: application/json

{
  "url": "https://example.com"
}
```

### Get Feature Importance
```bash
GET /feature-importance
```

## Gradio Interface Features

The Gradio app provides an interactive UI with 4 tabs:

1. **🔗 Single URL Analysis** - Check individual URLs with visual results and warning indicators
2. **📋 Batch Analysis** - Process up to 20 URLs at once
3. **🔧 Feature Extraction** - View all 20+ extracted features grouped by category
4. **📊 Model Insights** - See feature importance rankings from the trained model

Features:
- Real-time API health monitoring
- Example URLs for quick testing
- Warning indicators (no HTTPS, typosquatting, new domain, etc.)
- Copy-to-clipboard for all outputs

## CLI Tool

Command-line interface for quick URL checks:

```bash
# Check a single URL
python cli.py check https://suspicious-site.com

# Check multiple URLs from a file
python cli.py batch urls.txt --output results.json

# Extract features without prediction
python cli.py features https://example.com --output features.json
```

## Model Comparison

To compare Random Forest and XGBoost:

```bash
python train.py --dataset data/uci_phishing.csv --compare
```

## Running Tests

```bash
pytest tests/ -v
```

## Dataset

The system supports:
- **UCI Phishing Websites Dataset**: Pre-extracted features, easy to start
- **Custom URL datasets**: Raw URLs with labels, features are extracted automatically

### Sample Dataset

Create a sample dataset for testing:
```bash
python train.py --sample --n-samples 100
```

## Key Features Explained

1. **URL Length**: Longer URLs often indicate phishing
2. **Special Characters**: `@`, `//`, excessive `-` or `.` are suspicious
3. **HTTPS**: Lack of HTTPS is a phishing indicator
4. **IP in URL**: Direct IP addresses in URLs are suspicious
5. **Domain Age**: New domains are more likely to be phishing
6. **Typosquatting**: Levenshtein distance detects brand impersonation

## License

This project is for educational purposes.
