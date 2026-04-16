"""
Demo script showing how to use the fraud detection system.
"""

from src.feature_extraction import URLFeatureExtractor
from src.model import PhishingDetector
from src.data_loader import DataLoader


def demo_feature_extraction():
    """Demonstrate feature extraction."""
    print("=" * 70)
    print("DEMO: Feature Extraction")
    print("=" * 70)
    
    extractor = URLFeatureExtractor()
    
    test_urls = [
        'https://www.google.com/search?q=python',
        'http://paypa1-security.com/login.php',
        'http://192.168.1.1/admin',
        'https://www.amazon.com/gp/product/B08N5WRWNW',
        'http://g00gle-verify.com/signin',
    ]
    
    for url in test_urls:
        print(f"\nAnalyzing: {url}")
        print("-" * 50)
        features = extractor.extract_all_features(url)
        
        # Print key features
        print(f"  URL Length: {features['url_length']}")
        print(f"  Has HTTPS: {features['has_https']}")
        print(f"  Has IP: {features['has_ip_address']}")
        print(f"  Subdomains: {features['num_subdomains']}")
        print(f"  Typosquatting Score: {features['typosquatting_score']}")
        if features.get('closest_brand') and features['closest_brand'] != 'none':
            print(f"  Closest Brand: {features['closest_brand']}")
            print(f"  Levenshtein Distance: {features['levenshtein_distance']}")


def demo_training():
    """Demonstrate model training."""
    print("\n" + "=" * 70)
    print("DEMO: Model Training")
    print("=" * 70)
    
    # Create sample dataset
    print("\nCreating sample dataset...")
    loader = DataLoader()
    loader.create_sample_dataset('data/demo_urls.csv', n_samples=150)
    
    # Load and process
    print("\nLoading dataset and extracting features...")
    X, y = loader.load_custom_dataset('data/demo_urls.csv')
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # Train model
    print("\nTraining Random Forest model...")
    detector = PhishingDetector(model_type='random_forest')
    metrics = detector.train(X, y, validation_split=0.2)
    
    print("\nValidation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Feature importance
    print("\nTop 5 Most Important Features:")
    importances = detector.get_feature_importances()
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_imp[:5]:
        print(f"  {feature}: {importance:.4f}")
    
    # Save model
    detector.save('models/demo_model.pkl')
    print("\nModel saved to models/demo_model.pkl")
    
    return detector


def demo_prediction(detector: PhishingDetector):
    """Demonstrate prediction."""
    print("\n" + "=" * 70)
    print("DEMO: Prediction")
    print("=" * 70)
    
    extractor = URLFeatureExtractor()
    
    test_urls = [
        ('https://www.google.com', 'legitimate'),
        ('https://www.paypal.com', 'legitimate'),
        ('http://paypa1-verify.com/login', 'phishing'),
        ('http://amaz0n-security.com/update', 'phishing'),
        ('https://www.github.com', 'legitimate'),
    ]
    
    print("\nPredictions:")
    print("-" * 70)
    print(f"{'URL':<45} {'Actual':<12} {'Prediction':<12} {'Confidence':<10}")
    print("-" * 70)
    
    for url, actual in test_urls:
        features = extractor.extract_all_features(url)
        prediction_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
        
        proba = detector.predict_proba([prediction_features])[0]
        pred = detector.predict([prediction_features])[0]
        
        pred_label = 'phishing' if pred == 1 else 'legitimate'
        confidence = max(proba)
        
        # Color code (just using text for compatibility)
        match = "✓" if pred_label == actual else "✗"
        
        print(f"{url:<45} {actual:<12} {pred_label:<12} {confidence:.2%} {match}")


def demo_comparison():
    """Demonstrate model comparison."""
    print("\n" + "=" * 70)
    print("DEMO: Model Comparison (Random Forest vs XGBoost)")
    print("=" * 70)
    
    # Load data
    loader = DataLoader()
    X, y = loader.load_custom_dataset('data/demo_urls.csv')
    
    # Compare models
    from src.model import compare_models
    results = compare_models(X, y, validation_split=0.2)
    
    print("\nModel Comparison Results:")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("FRAUD WEBSITE DETECTION - DEMO")
    print("=" * 70)
    
    try:
        # Demo 1: Feature Extraction
        demo_feature_extraction()
        
        # Demo 2: Training
        detector = demo_training()
        
        # Demo 3: Prediction
        demo_prediction(detector)
        
        # Demo 4: Model Comparison
        demo_comparison()
        
        print("\n" + "=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Run the Flask API: python src/app.py")
        print("  2. Train with your own dataset: python train.py --dataset your_data.csv")
        print("  3. Run tests: pytest tests/ -v")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
