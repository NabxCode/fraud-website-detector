"""
Script to download pre-trained model or use UCI dataset with pre-extracted features.
"""

import os
import sys
import urllib.request
import zipfile
import shutil

# URLs for pre-trained models (these are example URLs - replace with actual hosting)
# For a real project, host models on GitHub Releases, Google Drive, or cloud storage
MODEL_URLS = {
    "uci_rf": {
        "url": "https://github.com/yourusername/fraud-detection/releases/download/v1.0/uci_random_forest.pkl",
        "description": "Random Forest trained on UCI Phishing Dataset (30 pre-extracted features)",
        "features": "pre-extracted",
        "size": "~5 MB"
    },
    "uci_xgb": {
        "url": "https://github.com/yourusername/fraud-detection/releases/download/v1.0/uci_xgboost.pkl",
        "description": "XGBoost trained on UCI Phishing Dataset (30 pre-extracted features)",
        "features": "pre-extracted",
        "size": "~3 MB"
    }
}

# Alternative: Use scikit-learn's built-in datasets to create a quick model
# This doesn't require downloading anything


def create_quick_model_from_uci():
    """
    Download UCI dataset and train a quick model.
    This is faster than training from URLs since UCI has pre-extracted features.
    """
    print("=" * 60)
    print("Quick Model Creation using UCI Dataset")
    print("=" * 60)
    print("\nThis will:")
    print("1. Download UCI Phishing Dataset (if not present)")
    print("2. Train a Random Forest model quickly (30 seconds)")
    print("3. Save the model for immediate use\n")
    
    try:
        # Try to use scikit-learn to fetch a similar dataset
        # or train on synthetic data that mimics phishing patterns
        
        print("Creating synthetic phishing detection model...")
        print("(This mimics patterns from the UCI dataset)\n")
        
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        # Create synthetic dataset based on UCI phishing features
        # UCI has 30 features, we'll create similar synthetic data
        np.random.seed(42)
        n_samples = 1000
        n_features = 30
        
        # Feature names matching UCI dataset
        feature_names = [
            'having_IP_Address', 'URL_Length', 'Shortining_Service',
            'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix',
            'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length',
            'Favicon', 'port', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor',
            'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
            'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe',
            'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank',
            'Google_Index', 'Links_pointing_to_page', 'Statistical_report'
        ]
        
        # Generate synthetic features that mimic phishing patterns
        # Phishing sites typically have certain patterns:
        X = np.random.randn(n_samples, n_features)
        
        # Create labels based on patterns (1 = phishing, -1 = legitimate)
        # Simulate that certain features correlate with phishing
        y = np.zeros(n_samples)
        
        # Pattern 1: IP address in URL (feature 0)
        y[X[:, 0] > 0.5] = 1
        
        # Pattern 2: Long URL (feature 1)
        y[X[:, 1] > 1.0] = 1
        
        # Pattern 3: No HTTPS (feature 7)
        y[X[:, 7] < -0.5] = 1
        
        # Pattern 4: Short domain age (feature 23)
        y[X[:, 23] < -0.5] = 1
        
        # Convert to binary (0 = legitimate, 1 = phishing)
        y = (y > 0).astype(int)
        
        # Train model
        print("Training Random Forest on synthetic data...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        
        # Create scaler (identity for now since data is already normalized)
        scaler = StandardScaler()
        scaler.fit(X)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'model_type': 'random_forest',
            'random_state': 42,
            'is_synthetic': True,
            'note': 'This is a synthetic model for demonstration. Train on real UCI dataset for production use.'
        }
        
        model_path = 'models/phishing_detector.pkl'
        joblib.dump(model_data, model_path)
        
        print(f"✅ Model saved to {model_path}")
        print(f"\nModel info:")
        print(f"  - Type: Random Forest")
        print(f"  - Features: {n_features} (UCI-style pre-extracted)")
        print(f"  - Samples: {n_samples} (synthetic)")
        print(f"  - Accuracy on training: {model.score(X, y):.2%}")
        print(f"\n⚠️  Note: This is a synthetic model for demonstration.")
        print("   For better results, train on the real UCI dataset:")
        print("   python train.py --dataset data/uci_phishing.csv --dataset-type uci")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False


def download_from_url(url: str, output_path: str) -> bool:
    """Download model from URL."""
    try:
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"✅ Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def list_available_models():
    """List available pre-trained models."""
    print("=" * 60)
    print("Available Model Options")
    print("=" * 60)
    
    print("\n1. 🚀 QUICK START (Recommended for Demo)")
    print("   Create synthetic model instantly")
    print("   No download required, works immediately")
    print("   → run: python download_model.py --quick")
    
    print("\n2. 📊 UCI DATASET MODEL (Best for College Project)")
    print("   Download from: https://archive.ics.uci.edu/ml/datasets/phishing+websites")
    print("   Train with: python train.py --dataset data/PhishingData.arff --dataset-type uci")
    
    print("\n3. 🌐 CUSTOM URL DATASET")
    print("   Use your own URLs")
    print("   → run: python train.py --sample --n-samples 200")
    
    print("\n4. 📥 DOWNLOAD PRE-TRAINED (If hosted)")
    print("   Available models:")
    for key, info in MODEL_URLS.items():
        print(f"     • {key}: {info['description']}")
        print(f"       Features: {info['features']}, Size: {info['size']}")
    
    print("\n" + "=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download or create pre-trained phishing detection model'
    )
    parser.add_argument('--quick', action='store_true',
                        help='Create quick synthetic model (recommended)')
    parser.add_argument('--download', type=str, metavar='MODEL_KEY',
                        help='Download specific model (e.g., uci_rf)')
    parser.add_argument('--list', action='store_true',
                        help='List available models')
    parser.add_argument('--uci', action='store_true',
                        help='Show UCI dataset instructions')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    if args.uci:
        print("=" * 60)
        print("UCI Phishing Websites Dataset Instructions")
        print("=" * 60)
        print("\n1. Download from:")
        print("   https://archive.ics.uci.edu/ml/datasets/phishing+websites")
        print("\n2. Save to: data/PhishingData.arff")
        print("\n3. Train model:")
        print("   python train.py --dataset data/PhishingData.arff --dataset-type uci")
        print("\n✅ This dataset has 30 pre-extracted features!")
        print("   No feature extraction needed - model trains in seconds.")
        return
    
    if args.download:
        if args.download not in MODEL_URLS:
            print(f"❌ Unknown model: {args.download}")
            print("Available models:", list(MODEL_URLS.keys()))
            return
        
        model_info = MODEL_URLS[args.download]
        os.makedirs('models', exist_ok=True)
        output_path = f"models/{args.download}.pkl"
        
        if download_from_url(model_info['url'], output_path):
            print(f"\n✅ Model downloaded: {output_path}")
            print(f"\nTo use this model:")
            print(f"  cp {output_path} models/phishing_detector.pkl")
            print(f"  python src/app.py")
        return
    
    # Default: create quick model
    print("\n🚀 Creating quick synthetic model for demonstration...\n")
    if create_quick_model_from_uci():
        print("\n" + "=" * 60)
        print("✅ Ready to use!")
        print("=" * 60)
        print("\nStart the API:")
        print("  python src/app.py")
        print("\nOr use the run script:")
        print("  ./run.sh")
    else:
        print("\n❌ Failed to create model.")
        print("Try training manually:")
        print("  python train.py --sample --n-samples 200")


if __name__ == '__main__':
    main()
