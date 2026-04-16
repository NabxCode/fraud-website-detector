"""
Command-line interface for the fraud detection system.
"""

import argparse
import sys
import json

from src.feature_extraction import URLFeatureExtractor
from src.model import PhishingDetector


def check_url(url: str, model_path: str = 'models/phishing_detector.pkl'):
    """Check if a URL is phishing."""
    print(f"\nAnalyzing: {url}")
    print("-" * 60)
    
    # Extract features
    extractor = URLFeatureExtractor()
    features = extractor.extract_all_features(url)
    
    # Print features
    print("\nExtracted Features:")
    numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
    for feature, value in list(numeric_features.items())[:10]:
        print(f"  {feature}: {value}")
    
    # Load model and predict
    try:
        detector = PhishingDetector.load(model_path)
        
        prediction_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
        proba = detector.predict_proba([prediction_features])[0]
        pred = detector.predict([prediction_features])[0]
        
        print("\nPrediction:")
        print(f"  Is Phishing: {'YES' if pred == 1 else 'NO'}")
        print(f"  Confidence: {max(proba):.2%}")
        print(f"  Phishing Probability: {proba[1]:.2%}")
        
        # Warning indicators
        warnings = []
        if not features['has_https']:
            warnings.append("- No HTTPS")
        if features['has_ip_address']:
            warnings.append("- Contains IP address")
        if features['typosquatting_score'] > 0:
            warnings.append(f"- Possible typosquatting of '{features.get('closest_brand', 'unknown')}'")
        if features.get('domain_age_days', -1) >= 0 and features['domain_age_days'] < 30:
            warnings.append("- Domain is less than 30 days old")
        
        if warnings:
            print("\nWarning Indicators:")
            for warning in warnings:
                print(f"  {warning}")
        
    except FileNotFoundError:
        print(f"\nError: Model not found at {model_path}")
        print("Please train the model first: python train.py --sample")
        sys.exit(1)


def batch_check(urls_file: str, model_path: str = 'models/phishing_detector.pkl',
                output_file: str = None):
    """Check multiple URLs from a file."""
    # Read URLs
    with open(urls_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    print(f"\nAnalyzing {len(urls)} URLs...")
    
    extractor = URLFeatureExtractor()
    detector = PhishingDetector.load(model_path)
    
    results = []
    for i, url in enumerate(urls, 1):
        try:
            features = extractor.extract_all_features(url)
            prediction_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
            
            proba = detector.predict_proba([prediction_features])[0]
            pred = detector.predict([prediction_features])[0]
            
            result = {
                'url': url,
                'is_phishing': bool(pred == 1),
                'confidence': float(max(proba)),
                'phishing_probability': float(proba[1])
            }
            results.append(result)
            
            status = "PHISHING" if pred == 1 else "SAFE"
            print(f"  [{i}/{len(urls)}] {status:8} {url[:50]}... ({max(proba):.0%})")
            
        except Exception as e:
            print(f"  [{i}/{len(urls)}] ERROR    {url[:50]}... ({str(e)})")
            results.append({'url': url, 'error': str(e)})
    
    # Summary
    phishing_count = sum(1 for r in results if r.get('is_phishing'))
    safe_count = len(results) - phishing_count - sum(1 for r in results if 'error' in r)
    
    print("\n" + "-" * 60)
    print(f"Summary: {safe_count} safe, {phishing_count} phishing, "
          f"{len(results) - safe_count - phishing_count} errors")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")


def extract_features_only(url: str, output_file: str = None):
    """Extract and display features from a URL."""
    print(f"\nExtracting features from: {url}")
    print("-" * 60)
    
    extractor = URLFeatureExtractor()
    features = extractor.extract_all_features(url)
    
    print("\nAll Features:")
    for feature, value in sorted(features.items()):
        print(f"  {feature}: {value}")
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(features, f, indent=2)
        print(f"\nFeatures saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Fraud Website Detection CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s check https://suspicious-site.com
  %(prog)s batch urls.txt --output results.json
  %(prog)s features https://example.com
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Check single URL
    check_parser = subparsers.add_parser('check', help='Check a single URL')
    check_parser.add_argument('url', help='URL to check')
    check_parser.add_argument('--model', default='models/phishing_detector.pkl',
                              help='Path to trained model')
    
    # Batch check
    batch_parser = subparsers.add_parser('batch', help='Check multiple URLs from file')
    batch_parser.add_argument('file', help='File containing URLs (one per line)')
    batch_parser.add_argument('--model', default='models/phishing_detector.pkl',
                            help='Path to trained model')
    batch_parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    
    # Extract features
    features_parser = subparsers.add_parser('features', help='Extract features from URL')
    features_parser.add_argument('url', help='URL to analyze')
    features_parser.add_argument('--output', '-o', help='Output file for features (JSON)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'check':
        check_url(args.url, args.model)
    elif args.command == 'batch':
        batch_check(args.file, args.model, args.output)
    elif args.command == 'features':
        extract_features_only(args.url, args.output)


if __name__ == '__main__':
    main()
