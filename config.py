"""
Configuration settings for the fraud detection system.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Model settings
DEFAULT_MODEL_TYPE = 'random_forest'  # or 'xgboost'
MODEL_FILE = os.path.join(MODELS_DIR, 'phishing_detector.pkl')

# Feature extraction
WHOIS_TIMEOUT = 10  # seconds for WHOIS lookup
REQUEST_TIMEOUT = 10  # seconds for HTTP requests

# Common brands for typosquatting detection
COMMON_BRANDS = [
    'google', 'facebook', 'amazon', 'paypal', 'apple', 'microsoft',
    'netflix', 'gmail', 'yahoo', 'linkedin', 'twitter', 'instagram',
    'bankofamerica', 'chase', 'wellsfargo', 'citibank', 'amex',
    'dropbox', 'github', 'spotify', 'uber', 'airbnb', 'wellsfargo',
    'chase', 'citi', 'hsbc', 'barclays', 'santander', 'bbva',
    'americaexpress', 'mastercard', 'visa', 'discover',
]

# Flask API settings
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5001
FLASK_DEBUG = False

# Training settings
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss'
}
