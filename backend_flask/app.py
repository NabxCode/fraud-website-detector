"""
Flask API for phishing website detection.
"""

from backend_flask.model import PhishingDetector
from backend_flask.feature_extraction import URLFeatureExtractor
import os
import re
import sys
import logging
import traceback
import urllib.parse

from flask import Flask, request, jsonify

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


app = Flask(__name__)

# ------------------------------------------------------------------
# Logging setup — writes full tracebacks to logs/flask.log AND prints
# them to the terminal, so 500 errors are never silent.
# ------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'flask.log')

logger = logging.getLogger('phishing_api')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Global model instance
model = None
feature_extractor = URLFeatureExtractor()

# ------------------------------------------------------------------
# Private IP hard override — checked in BOTH /predict and /predict/batch
# before the ML model runs.
# The ML model only gives ~22% phishing probability to private IPs
# because is_private_ip was trained on a proxy column (having_IP_Address)
# with very low feature importance (0.009). A hard rule is the correct fix.
# ------------------------------------------------------------------
_PRIVATE_IP_PATTERNS = [
    r'^192\.168\.',
    r'^10\.',
    r'^172\.(1[6-9]|2[0-9]|3[01])\.',
    r'^127\.',
    r'^0\.',
    r'^169\.254\.',
    r'^::1$',
    r'^localhost$',
]


def _is_private_ip_url(url: str) -> bool:
    """Returns True if the URL's domain is a private/reserved IP address."""
    if '://' not in url:
        url = 'http://' + url
    domain = (urllib.parse.urlparse(url).hostname or '').lower()
    return any(re.match(p, domain) for p in _PRIVATE_IP_PATTERNS)


# ------------------------------------------------------------------
# Typosquatting hard override
# Catches domains like paypa1-secure.com, amaz0n-security.com,
# faceb00k-login.com, g00gle-verify.com that the ML model misses
# because the UCI training dataset has no typosquatting examples.
# Uses Levenshtein edit distance — no external library needed.
# ------------------------------------------------------------------
def _is_typosquatting_url(url: str) -> bool:
    """Returns True if the URL's domain is a typosquatted brand name."""
    if '://' not in url:
        url = 'http://' + url
    domain = (urllib.parse.urlparse(url).hostname or '').lower()
    return feature_extractor._is_typosquatting(domain)


# ------------------------------------------------------------------
# Trusted domain whitelist — SAFE override
# The ML model incorrectly flags well-known legitimate HTTPS domains
# as phishing because SSLfinal_State = -1.0 when the local machine
# cannot open a raw SSL socket to external servers (firewall/network).
# SSLfinal_State has 0.42 feature importance — almost half the model's
# decision weight — so one bad SSL reading breaks everything.
# Hard-coding these known-safe domains fixes the false positive without
# retraining. Only exact domain matches are whitelisted (no subdomains
# unless explicitly listed), so phishing subdomains still get through.
# ------------------------------------------------------------------
_TRUSTED_DOMAINS = {
    'google.com', 'www.google.com',
    'gmail.com', 'www.gmail.com',
    'youtube.com', 'www.youtube.com',
    'facebook.com', 'www.facebook.com',
    'instagram.com', 'www.instagram.com',
    'whatsapp.com', 'www.whatsapp.com',
    'amazon.com', 'www.amazon.com',
    'microsoft.com', 'www.microsoft.com',
    'outlook.com', 'www.outlook.com',
    'office.com', 'www.office.com',
    'live.com', 'www.live.com',
    'hotmail.com', 'www.hotmail.com',
    'apple.com', 'www.apple.com',
    'icloud.com', 'www.icloud.com',
    'paypal.com', 'www.paypal.com',
    'netflix.com', 'www.netflix.com',
    'twitter.com', 'www.twitter.com',
    'x.com', 'www.x.com',
    'linkedin.com', 'www.linkedin.com',
    'github.com', 'www.github.com',
    'dropbox.com', 'www.dropbox.com',
    'adobe.com', 'www.adobe.com',
    'yahoo.com', 'www.yahoo.com',
    'ebay.com', 'www.ebay.com',
    'walmart.com', 'www.walmart.com',
    'chase.com', 'www.chase.com',
    'spotify.com', 'www.spotify.com',
    'tiktok.com', 'www.tiktok.com',
    'snapchat.com', 'www.snapchat.com',
    'reddit.com', 'www.reddit.com',
    'wikipedia.org', 'www.wikipedia.org',
    'stackoverflow.com', 'www.stackoverflow.com',
    'nytimes.com', 'www.nytimes.com',
    'bbc.com', 'www.bbc.com',
}


def _is_trusted_domain(url: str) -> bool:
    """Returns True if the URL's domain is a well-known legitimate site."""
    if '://' not in url:
        url = 'http://' + url
    domain = (urllib.parse.urlparse(url).hostname or '').lower()
    return domain in _TRUSTED_DOMAINS


# ------------------------------------------------------------------
# URL shortener hard override — always PHISHING
# The model gives Shortining_Service only 1.77% importance, so it
# frequently misses shorteners like bit.ly when other features are
# neutral. URL shorteners hide the real destination — always suspicious
# in a phishing context.
# ------------------------------------------------------------------
_URL_SHORTENERS = {
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
    'buff.ly', 'adf.ly', 'bit.do', 'mcaf.ee', 'su.pr', 'cutt.ly',
    'shorte.st', 'tiny.cc', 'lnkd.in', 'db.tt', 'qr.ae', 'rebrand.ly',
    'shorturl.at', 'clck.ru', 'v.gd', 'urlzs.com', 'snip.ly',
}


def _is_url_shortener(url: str) -> bool:
    """Returns True if the URL uses a known shortening service."""
    if '://' not in url:
        url = 'http://' + url
    domain = (urllib.parse.urlparse(url).hostname or '').lower()
    return domain in _URL_SHORTENERS


# ------------------------------------------------------------------
# Suspicious TLD hard override — always PHISHING
# The model gives suspicious_tld only 1.26% importance, so URLs with
# phishing-heavy TLDs like .xyz, .tk, .click get rated SAFE when other
# features are neutral (no real server = no bad SSL, no page content).
# These TLDs are statistically dominated by phishing — hard override
# is correct here.
# ------------------------------------------------------------------
_SUSPICIOUS_TLDS = {
    '.xyz', '.click', '.club', '.online', '.site', '.lat',
    '.sbs', '.top', '.work', '.loan', '.win', '.gq', '.ml',
    '.cf', '.ga', '.tk', '.pw', '.cc', '.su', '.ws',
}


def _has_suspicious_tld(url: str) -> bool:
    """Returns True if the URL's domain ends with a phishing-heavy TLD."""
    if '://' not in url:
        url = 'http://' + url
    domain = (urllib.parse.urlparse(url).hostname or '').lower()
    return any(domain.endswith(tld) for tld in _SUSPICIOUS_TLDS)


# ------------------------------------------------------------------
# Synthetic feature dict for overrides 3-5 (shortener, suspicious TLD,
# trusted domain).
#
# WHY THIS EXISTS: overrides 1-2 (private IP, typosquatting) get their
# 'features' dict for free and instantly, because feature_extraction.py's
# own extract_all_features() has an internal short-circuit for those two
# cases that returns before any live network call. Overrides 3-5 have no
# such internal short-circuit, so calling extract_all_features() for them
# used to trigger a full live page fetch + WHOIS lookup + SSL handshake
# (up to ~22s combined timeout budget) purely to fill the response's
# 'features' field — even though the verdict was already decided by the
# override itself. This function returns a placeholder 28-key dict
# instantly instead, matching the same "all keys share one directional
# value" convention feature_extraction.py already uses for its own
# override (_phishing_override_features).
#
# These values are placeholders for display purposes only, not real
# per-page measurements — the override already determined the verdict
# without needing them.
# ------------------------------------------------------------------
_OVERRIDE_FEATURE_NAMES = [
    'having_IP_Address', 'URL_Length', 'Shortining_Service',
    'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix',
    'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length',
    'Favicon', 'port', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor',
    'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
    'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe',
    'age_of_domain', 'DNSRecord',
    'is_private_ip', 'url_entropy', 'suspicious_tld',
]


def _synthetic_features(value: float) -> dict:
    """Build an instant 28-key feature dict for fast hard overrides.

    value=-1.0 for phishing-verdict overrides (shortener, suspicious TLD),
    value=1.0 for the safe-verdict override (trusted domain).
    """
    return {name: value for name in _OVERRIDE_FEATURE_NAMES}


def load_model(model_path: str = 'model_xgboost/phishing_detector.pkl'):
    """Load the trained model."""
    global model
    try:
        model = PhishingDetector.load(model_path)
        print(f"Model loaded from {model_path}")
        logger.info(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(
            f"Warning: Model file not found at {model_path}. API will not work until model is trained.")
        logger.warning(
            f"Model file not found at {model_path}. API will not work until model is trained.")
        model = None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if a URL is phishing or legitimate.

    Request body:
        {
            "url": "https://example.com"
        }

    Returns:
        {
            "url": "https://example.com",
            "is_phishing": true/false,
            "confidence": 0.95,
            "features": {...}
        }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    data = request.get_json()

    if not data or 'url' not in data:
        return jsonify({'error': 'Missing required field: url'}), 400

    url = data['url']

    # ------------------------------------------------------------------
    # HARD OVERRIDE 1: private/local IP address is always phishing.
    # feature_extraction.py already sets all features to -1.0 for these,
    # but the ML model still returns is_phishing=False because it was
    # trained on proxy columns with very low feature importance.
    # This is fast: extract_all_features() short-circuits internally
    # before any live network call for private IPs.
    # ------------------------------------------------------------------
    if _is_private_ip_url(url):
        features = feature_extractor.extract_all_features(url)  # all -1.0
        logger.info(f"Private IP override fired for url={url!r}")
        return jsonify({
            'url': url,
            'is_phishing': True,
            'confidence': 1.0,
            'phishing_probability': 1.0,
            'features': features
        })

    # ------------------------------------------------------------------
    # HARD OVERRIDE 2: typosquatting detection via Levenshtein distance.
    # Catches paypa1-secure.com, amaz0n-security.com, faceb00k-login.com
    # etc. that the ML model misses because the UCI training dataset has
    # no typosquatting examples (collected in 2017, before this was common).
    # This is fast: extract_all_features() short-circuits internally
    # before any live network call for typosquatted domains.
    # ------------------------------------------------------------------
    if _is_typosquatting_url(url):
        features = feature_extractor.extract_all_features(url)
        logger.info(f"Typosquatting override fired for url={url!r}")
        return jsonify({
            'url': url,
            'is_phishing': True,
            'confidence': 0.99,
            'phishing_probability': 0.99,
            'features': features
        })

    # ------------------------------------------------------------------
    # HARD OVERRIDE 3: URL shortener — always phishing in this context.
    # Shortining_Service has only 1.77% model importance so the model
    # routinely misses bit.ly, tinyurl.com etc. when other features
    # are neutral (fake domain with no real server to check).
    # Uses synthetic placeholder features instead of a live fetch —
    # the verdict is already decided, so there's no need to spend the
    # ~22s worst-case fetch/WHOIS/SSL timeout budget just to fill the
    # response's 'features' field.
    # ------------------------------------------------------------------
    if _is_url_shortener(url):
        features = _synthetic_features(-1.0)
        logger.info(f"URL shortener override fired for url={url!r}")
        return jsonify({
            'url': url,
            'is_phishing': True,
            'confidence': 0.95,
            'phishing_probability': 0.95,
            'features': features
        })

    # ------------------------------------------------------------------
    # HARD OVERRIDE 4: suspicious TLD — always phishing.
    # suspicious_tld has only 1.26% model importance. Fake phishing
    # domains with no live server return neutral SSL/page features,
    # letting the model incorrectly rate .xyz, .tk, .club etc. as SAFE.
    # Uses synthetic placeholder features instead of a live fetch, same
    # reasoning as override 3 above.
    # ------------------------------------------------------------------
    if _has_suspicious_tld(url):
        features = _synthetic_features(-1.0)
        logger.info(f"Suspicious TLD override fired for url={url!r}")
        return jsonify({
            'url': url,
            'is_phishing': True,
            'confidence': 0.92,
            'phishing_probability': 0.92,
            'features': features
        })

    # ------------------------------------------------------------------
    # SAFE OVERRIDE: well-known legitimate domain.
    # The ML model falsely flags sites like paypal.com and amazon.com
    # as phishing when SSLfinal_State = -1.0 (SSL socket blocked locally).
    # This override runs AFTER the phishing overrides above, so a
    # typosquatted URL like paypa1.com still gets caught correctly.
    # Uses synthetic placeholder features (all 1.0 = safe direction)
    # instead of a live fetch, same reasoning as overrides 3-4 above.
    # ------------------------------------------------------------------
    if _is_trusted_domain(url):
        features = _synthetic_features(1.0)
        logger.info(f"Trusted domain override fired for url={url!r}")
        return jsonify({
            'url': url,
            'is_phishing': False,
            'confidence': 0.99,
            'phishing_probability': 0.01,
            'features': features
        })

    # Extract features
    try:
        features = feature_extractor.extract_all_features(url)
    except Exception as e:
        logger.error(f"Feature extraction failed for url={url!r}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Feature extraction failed: {str(e)}'}), 500

    # Make prediction
    try:
        # Remove non-numeric features for prediction
        prediction_features = {
            k: v for k, v in features.items() if isinstance(v, (int, float))}

        # Ensure features match model's expected feature names and order
        if model.feature_names:
            missing_features = set(model.feature_names) - \
                set(prediction_features.keys())
            if missing_features:
                logger.error(
                    f"Missing features for url={url!r}: {missing_features}")
                return jsonify({'error': f'Missing features: {missing_features}'}), 500
            # Reorder features to match model training order
            prediction_features = {
                k: prediction_features[k] for k in model.feature_names}

        proba = model.predict_proba([prediction_features])[0]
        prediction = model.predict([prediction_features])[0]

        result = {
            'url': url,
            'is_phishing': bool(prediction == 1),
            'confidence': float(max(proba)),
            'phishing_probability': float(proba[1]),
            'features': features
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction failed for url={url!r}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict multiple URLs at once.

    Request body:
        {
            "urls": ["https://example1.com", "https://example2.com"]
        }

    Returns:
        {
            "results": [
                {
                    "url": "https://example1.com",
                    "is_phishing": true/false,
                    "confidence": 0.95
                },
                ...
            ]
        }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    data = request.get_json()

    if not data or 'urls' not in data:
        return jsonify({'error': 'Missing required field: urls'}), 400

    urls = data['urls']

    if not isinstance(urls, list):
        return jsonify({'error': 'urls must be a list'}), 400

    results = []

    for url in urls:
        # Override 1: private IP
        if _is_private_ip_url(url):
            logger.info(f"Batch private IP override fired for url={url!r}")
            results.append({
                'url': url,
                'is_phishing': True,
                'confidence': 1.0,
                'phishing_probability': 1.0
            })
            continue

        # Override 2: typosquatting
        if _is_typosquatting_url(url):
            logger.info(f"Batch typosquatting override fired for url={url!r}")
            results.append({
                'url': url,
                'is_phishing': True,
                'confidence': 0.99,
                'phishing_probability': 0.99
            })
            continue

        # Override 3: URL shortener
        if _is_url_shortener(url):
            logger.info(f"Batch URL shortener override fired for url={url!r}")
            results.append({
                'url': url,
                'is_phishing': True,
                'confidence': 0.95,
                'phishing_probability': 0.95
            })
            continue

        # Override 4: suspicious TLD
        if _has_suspicious_tld(url):
            logger.info(f"Batch suspicious TLD override fired for url={url!r}")
            results.append({
                'url': url,
                'is_phishing': True,
                'confidence': 0.92,
                'phishing_probability': 0.92
            })
            continue

        # Override 5: trusted domain (safe)
        if _is_trusted_domain(url):
            logger.info(f"Batch trusted domain override fired for url={url!r}")
            results.append({
                'url': url,
                'is_phishing': False,
                'confidence': 0.99,
                'phishing_probability': 0.01
            })
            continue

        try:
            features = feature_extractor.extract_all_features(url)
            prediction_features = {
                k: v for k, v in features.items() if isinstance(v, (int, float))}

            # Ensure features match model's expected feature names and order
            if model.feature_names:
                missing_features = set(
                    model.feature_names) - set(prediction_features.keys())
                if missing_features:
                    results.append({
                        'url': url,
                        'error': f'Missing features: {missing_features}'
                    })
                    continue
                # Reorder features to match model training order
                prediction_features = {
                    k: prediction_features[k] for k in model.feature_names}

            proba = model.predict_proba([prediction_features])[0]
            prediction = model.predict([prediction_features])[0]

            results.append({
                'url': url,
                'is_phishing': bool(prediction == 1),
                'confidence': float(max(proba)),
                'phishing_probability': float(proba[1])
            })
        except Exception as e:
            logger.error(f"Batch prediction failed for url={url!r}")
            logger.error(traceback.format_exc())
            results.append({
                'url': url,
                'error': str(e)
            })

    return jsonify({'results': results})


@app.route('/features', methods=['POST'])
def extract_features():
    """
    Extract features from a URL without making a prediction.

    Request body:
        {
            "url": "https://example.com"
        }

    Returns:
        {
            "url": "https://example.com",
            "features": {...}
        }
    """
    data = request.get_json()

    if not data or 'url' not in data:
        return jsonify({'error': 'Missing required field: url'}), 400

    url = data['url']

    try:
        features = feature_extractor.extract_all_features(url)
        return jsonify({
            'url': url,
            'features': features
        })
    except Exception as e:
        logger.error(f"Feature extraction (standalone) failed for url={url!r}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Feature extraction failed: {str(e)}'}), 500


@app.route('/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance from the trained model."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        importances = model.get_feature_importances()
        # Sort by importance; cast to Python float so jsonify can serialize
        # numpy float32 values returned by XGBoost/RandomForest without error.
        sorted_importances = {
            k: float(v)
            for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)
        }
        return jsonify({
            'model_type': model.model_type,
            'feature_importances': sorted_importances
        })
    except Exception as e:
        logger.error("Feature importance retrieval failed")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Try to load model
    load_model()

    # Get port from environment or use default
    port = int(os.environ.get('FLASK_RUN_PORT', 5001))

    # Run Flask app
    print(f"Starting Flask API on port {port}")
    logger.info(f"Starting Flask API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
