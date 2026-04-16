"""
Unit tests for the phishing detection model.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from src.model import PhishingDetector


class TestPhishingDetector:
    """Test cases for PhishingDetector."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        return X, y
    
    @pytest.fixture(params=['random_forest', 'xgboost'])
    def detector(self, request):
        return PhishingDetector(model_type=request.param)
    
    def test_model_initialization(self):
        """Test model initialization."""
        rf_detector = PhishingDetector(model_type='random_forest')
        xgb_detector = PhishingDetector(model_type='xgboost')
        
        assert rf_detector.model_type == 'random_forest'
        assert xgb_detector.model_type == 'xgboost'
        assert not rf_detector.is_trained
        assert not xgb_detector.is_trained
    
    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError):
            PhishingDetector(model_type='invalid')
    
    def test_training(self, detector, sample_data):
        """Test model training."""
        X, y = sample_data
        
        metrics = detector.train(X, y, validation_split=0.2)
        
        assert detector.is_trained
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_prediction(self, detector, sample_data):
        """Test making predictions."""
        X, y = sample_data
        
        detector.train(X, y, validation_split=0.2)
        
        # Predict on single sample
        single_pred = detector.predict(X.iloc[[0]])
        assert single_pred.shape == (1,)
        assert single_pred[0] in [0, 1]
        
        # Predict probabilities
        proba = detector.predict_proba(X.iloc[:5])
        assert proba.shape == (5, 2)
        assert np.all((proba >= 0) & (proba <= 1))
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_untrained_prediction(self, detector, sample_data):
        """Test that prediction fails before training."""
        X, y = sample_data
        
        with pytest.raises(RuntimeError):
            detector.predict(X)
    
    def test_feature_importance(self, detector, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        
        detector.train(X, y, validation_split=0.2)
        
        importances = detector.get_feature_importances()
        assert len(importances) == X.shape[1]
        assert all(imp >= 0 for imp in importances.values())
        assert abs(sum(importances.values()) - 1.0) < 1e-6
    
    def test_cross_validation(self, detector, sample_data):
        """Test cross-validation."""
        X, y = sample_data
        
        cv_metrics = detector.cross_validate(X, y, cv=3)
        
        assert 'cv_mean_accuracy' in cv_metrics
        assert 'cv_std_accuracy' in cv_metrics
        assert 0 <= cv_metrics['cv_mean_accuracy'] <= 1
        assert cv_metrics['cv_std_accuracy'] >= 0
    
    def test_save_load(self, detector, sample_data):
        """Test model saving and loading."""
        X, y = sample_data
        
        detector.train(X, y, validation_split=0.2)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            detector.save(temp_path)
            assert os.path.exists(temp_path)
            
            # Load model
            loaded_detector = PhishingDetector.load(temp_path)
            
            assert loaded_detector.is_trained
            assert loaded_detector.model_type == detector.model_type
            assert loaded_detector.feature_names == detector.feature_names
            
            # Check predictions match
            original_pred = detector.predict(X.iloc[:5])
            loaded_pred = loaded_detector.predict(X.iloc[:5])
            assert np.array_equal(original_pred, loaded_pred)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
