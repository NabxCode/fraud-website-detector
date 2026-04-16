"""
Data loading and preprocessing utilities.
Supports UCI Phishing Dataset and custom URL datasets.
"""

import os
import re
import urllib.parse
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np

from src.feature_extraction import URLFeatureExtractor


class DataLoader:
    """Load and preprocess phishing datasets."""
    
    def __init__(self):
        self.extractor = URLFeatureExtractor()
    
    def load_uci_dataset(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the UCI Phishing Websites Dataset.
        
        The dataset contains 30 features and a target column.
        Features include: having_IP_Address, URL_Length, Shortining_Service,
        having_At_Symbol, double_slash_redirecting, Prefix_Suffix, etc.
        
        Args:
            filepath: Path to the UCI dataset CSV/ARFF file
            
        Returns:
            Tuple of (X, y) where X is feature DataFrame and y is target Series
        """
        # Determine file type
        if filepath.endswith('.arff'):
            df = self._load_arff(filepath)
        else:
            df = pd.read_csv(filepath)
        
        # Separate features and target
        if 'Result' in df.columns:
            y = df['Result']
            X = df.drop('Result', axis=1)
        elif 'class' in df.columns:
            y = df['class']
            X = df.drop('class', axis=1)
        else:
            # Assume last column is target
            y = df.iloc[:, -1]
            X = df.iloc[:, :-1]
        
        # Convert -1/1 labels to 0/1 if needed
        y = y.apply(lambda x: 1 if x in [1, '1', 'phishing', -1, '-1'] else 0)
        
        return X, y
    
    def _load_arff(self, filepath: str) -> pd.DataFrame:
        """Load ARFF file format."""
        from scipy.io import arff
        data, meta = arff.loadarff(filepath)
        df = pd.DataFrame(data)
        
        # Convert byte strings to regular strings
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.decode('utf-8')
        
        return df
    
    def load_custom_dataset(self, filepath: str, 
                            url_column: str = 'url',
                            label_column: str = 'label') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load a custom dataset with URLs and labels.
        
        Args:
            filepath: Path to CSV file
            url_column: Name of column containing URLs
            label_column: Name of column containing labels
            
        Returns:
            Tuple of (X, y) where X is feature DataFrame and y is target Series
        """
        df = pd.read_csv(filepath)
        
        if url_column not in df.columns:
            raise ValueError(f"URL column '{url_column}' not found in dataset")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        
        urls = df[url_column].tolist()
        labels = df[label_column]
        
        # Convert labels to binary
        y = labels.apply(lambda x: 1 if str(x).lower() in ['phishing', '1', 'malicious', 'bad', 'true'] else 0)
        
        # Extract features from URLs
        print(f"Extracting features from {len(urls)} URLs...")
        features_list = []
        for i, url in enumerate(urls):
            if i % 100 == 0:
                print(f"Processed {i}/{len(urls)} URLs...")
            try:
                features = self.extractor.extract_all_features(str(url))
                # Keep only numeric features
                numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
                features_list.append(numeric_features)
            except Exception as e:
                print(f"Error processing URL {url}: {e}")
                # Add empty feature set
                features_list.append({})
        
        X = pd.DataFrame(features_list)
        
        # Fill missing values
        X = X.fillna(0)
        
        return X, y
    
    def create_sample_dataset(self, output_path: str = 'data/sample_urls.csv', 
                              n_samples: int = 100):
        """Create a sample dataset with legitimate and phishing URLs for testing."""
        
        legitimate_urls = [
            'https://www.google.com',
            'https://www.facebook.com',
            'https://www.amazon.com',
            'https://www.microsoft.com',
            'https://www.apple.com',
            'https://github.com',
            'https://stackoverflow.com',
            'https://www.wikipedia.org',
            'https://www.linkedin.com',
            'https://twitter.com',
            'https://www.youtube.com',
            'https://www.reddit.com',
            'https://www.netflix.com',
            'https://www.spotify.com',
            'https://www.adobe.com',
            'https://www.nytimes.com',
            'https://www.bbc.com',
            'https://www.cnn.com',
            'https://www.paypal.com',
            'https://www.dropbox.com',
            'https://medium.com',
            'https://www.quora.com',
            'https://www.instagram.com',
            'https://www.pinterest.com',
            'https://www.tumblr.com',
        ]
        
        # Synthetic phishing URLs (for demonstration)
        phishing_patterns = [
            'http://googIe.com.phishing-site.com/login',
            'http://paypa1-secure.com/verify',
            'http://amaz0n-security.com/update',
            'http://faceb00k-login.com/auth',
            'http://microsoft-verify.com/signin',
            'http://apple-id-confirm.com/verify',
            'http://bankofamerica-secure.com/login',
            'http://chase-verify.com/authenticate',
            'http://wellsfargo-update.com/confirm',
            'http://netflix-billing.com/payment',
            'http://paypal-secure-center.com/login',
            'http://google-security-alert.com/verify',
            'http://amazon-account-verify.com/update',
            'http://facebook-security.com/confirm',
            'http://microsoft-account-verify.com/auth',
        ]
        
        # Generate more samples by combining
        all_legitimate = legitimate_urls * ((n_samples // 2) // len(legitimate_urls) + 1)
        all_phishing = phishing_patterns * ((n_samples // 2) // len(phishing_patterns) + 1)
        
        # Create dataset
        data = []
        for url in all_legitimate[:n_samples//2]:
            data.append({'url': url, 'label': 'legitimate'})
        for url in all_phishing[:n_samples//2]:
            data.append({'url': url, 'label': 'phishing'})
        
        df = pd.DataFrame(data)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"Sample dataset created at {output_path}")
        
        return df
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1,
                   random_state: int = 42) -> Tuple:
        """
        Split data into train/validation/test sets.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation
        val_size = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
