"""Quick training script that skips slow WHOIS lookups."""
import sys
import os
sys.path.insert(0, '/Users/jigarpatel/Desktop/Code/FraudWebsiteDetectionGit')

import pandas as pd
import numpy as np
from src.model import PhishingDetector
from src.feature_extraction import URLFeatureExtractor

# Temporarily disable WHOIS by patching
class FastFeatureExtractor(URLFeatureExtractor):
    def _extract_domain_age_features(self, url):
        # Skip WHOIS - return default values
        return {
            'domain_age_days': -1.0,
            'domain_registration_length': -1.0,
        }

# Sample data
legitimate_urls = [
    'https://www.google.com', 'https://www.facebook.com', 'https://www.amazon.com',
    'https://www.microsoft.com', 'https://www.apple.com', 'https://github.com',
    'https://stackoverflow.com', 'https://www.wikipedia.org', 'https://www.linkedin.com',
    'https://twitter.com', 'https://www.youtube.com', 'https://www.reddit.com',
    'https://www.netflix.com', 'https://www.spotify.com', 'https://www.adobe.com',
    'https://www.nytimes.com', 'https://www.bbc.com', 'https://www.cnn.com',
    'https://www.paypal.com', 'https://www.dropbox.com', 'https://medium.com',
    'https://www.quora.com', 'https://www.instagram.com', 'https://www.pinterest.com',
    'https://www.tumblr.com',
]

phishing_urls = [
    'http://googIe.com.phishing-site.com/login', 'http://paypa1-secure.com/verify',
    'http://amaz0n-security.com/update', 'http://faceb00k-login.com/auth',
    'http://microsoft-verify.com/signin', 'http://apple-id-confirm.com/verify',
    'http://bankofamerica-secure.com/login', 'http://chase-verify.com/authenticate',
    'http://wellsfargo-update.com/confirm', 'http://netflix-billing.com/payment',
    'http://paypal-secure-center.com/login', 'http://google-security-alert.com/verify',
    'http://amazon-account-verify.com/update', 'http://facebook-security.com/confirm',
    'http://microsoft-account-verify.com/auth',
]

# Extract features
extractor = FastFeatureExtractor()
print("Extracting features...")

features_list = []
labels = []

for url in legitimate_urls:
    features = extractor.extract_all_features(url)
    numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
    features_list.append(numeric_features)
    labels.append(0)

for url in phishing_urls:
    features = extractor.extract_all_features(url)
    numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
    features_list.append(numeric_features)
    labels.append(1)

X = pd.DataFrame(features_list).fillna(0)
y = pd.Series(labels)

print(f"Dataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")

# Train model
print("\nTraining Random Forest model...")
detector = PhishingDetector(model_type='random_forest')
metrics = detector.train(X, y, validation_split=0.2)

print("\nValidation Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

# Save model
os.makedirs('models', exist_ok=True)
detector.save('models/phishing_detector.pkl')
print("\nModel saved to models/phishing_detector.pkl")
print("\nSUCCESS! Retrain complete. Restart the Flask server to load the new model.")
