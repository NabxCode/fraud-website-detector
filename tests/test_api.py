"""
Unit tests for Flask API endpoints.
"""

import pytest
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import app, load_model


class TestAPI:
    """Test cases for Flask API."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'healthy'
    
    def test_extract_features(self, client):
        """Test feature extraction endpoint."""
        test_url = 'https://www.google.com'
        
        response = client.post('/features',
                               data=json.dumps({'url': test_url}),
                               content_type='application/json')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['url'] == test_url
        assert 'features' in data
        assert 'url_length' in data['features']
        assert 'has_https' in data['features']
    
    def test_extract_features_missing_url(self, client):
        """Test feature extraction with missing URL."""
        response = client.post('/features',
                               data=json.dumps({}),
                               content_type='application/json')
        
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_predict_without_model(self, client):
        """Test prediction without loaded model."""
        # Ensure model is not loaded
        import src.app as app_module
        app_module.model = None
        
        response = client.post('/predict',
                               data=json.dumps({'url': 'https://example.com'}),
                               content_type='application/json')
        
        assert response.status_code == 503
    
    def test_predict_missing_url(self, client):
        """Test prediction with missing URL."""
        response = client.post('/predict',
                               data=json.dumps({}),
                               content_type='application/json')
        
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_batch_predict_missing_urls(self, client):
        """Test batch prediction with missing URLs."""
        response = client.post('/predict/batch',
                               data=json.dumps({}),
                               content_type='application/json')
        
        assert response.status_code == 400
    
    def test_batch_predict_invalid_urls_type(self, client):
        """Test batch prediction with invalid URLs type."""
        response = client.post('/predict/batch',
                               data=json.dumps({'urls': 'not a list'}),
                               content_type='application/json')
        
        assert response.status_code == 400


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
