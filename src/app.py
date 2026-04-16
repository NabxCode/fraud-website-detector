"""
Flask API for phishing website detection.
"""

import os
import sys

from flask import Flask, request, jsonify

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction import URLFeatureExtractor
from src.model import PhishingDetector


app = Flask(__name__)

# Global model instance
model = None
feature_extractor = URLFeatureExtractor()


def load_model(model_path: str = 'models/phishing_detector.pkl'):
    """Load the trained model."""
    global model
    try:
        model = PhishingDetector.load(model_path)
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Warning: Model file not found at {model_path}. API will not work until model is trained.")
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
    
    # Extract features
    try:
        features = feature_extractor.extract_all_features(url)
    except Exception as e:
        return jsonify({'error': f'Feature extraction failed: {str(e)}'}), 500
    
    # Make prediction
    try:
        # Remove non-numeric features for prediction
        prediction_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
        
        # Ensure features match model's expected feature names and order
        if model.feature_names:
            missing_features = set(model.feature_names) - set(prediction_features.keys())
            if missing_features:
                return jsonify({'error': f'Missing features: {missing_features}'}), 500
            # Reorder features to match model training order
            prediction_features = {k: prediction_features[k] for k in model.feature_names}
        
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
        try:
            features = feature_extractor.extract_all_features(url)
            prediction_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
            
            # Ensure features match model's expected feature names and order
            if model.feature_names:
                missing_features = set(model.feature_names) - set(prediction_features.keys())
                if missing_features:
                    results.append({
                        'url': url,
                        'error': f'Missing features: {missing_features}'
                    })
                    continue
                # Reorder features to match model training order
                prediction_features = {k: prediction_features[k] for k in model.feature_names}
            
            proba = model.predict_proba([prediction_features])[0]
            prediction = model.predict([prediction_features])[0]
            
            results.append({
                'url': url,
                'is_phishing': bool(prediction == 1),
                'confidence': float(max(proba)),
                'phishing_probability': float(proba[1])
            })
        except Exception as e:
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
        return jsonify({'error': f'Feature extraction failed: {str(e)}'}), 500


@app.route('/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance from the trained model."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        importances = model.get_feature_importances()
        # Sort by importance
        sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        return jsonify({
            'model_type': model.model_type,
            'feature_importances': sorted_importances
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Try to load model
    load_model()
    
    # Get port from environment or use default
    port = int(os.environ.get('FLASK_RUN_PORT', 5001))
    
    # Run Flask app
    print(f"Starting Flask API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
