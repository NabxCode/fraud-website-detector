"""Fraud Website Detection System."""

from .feature_extraction import URLFeatureExtractor, extract_features_from_urls
from .model import PhishingDetector, compare_models
from .data_loader import DataLoader

__version__ = '1.0.0'
__all__ = [
    'URLFeatureExtractor',
    'extract_features_from_urls',
    'PhishingDetector',
    'compare_models',
    'DataLoader',
]
