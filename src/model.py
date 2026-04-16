"""
Machine learning model for phishing website detection.
Supports Random Forest (baseline) and XGBoost.
"""

import pickle
import joblib
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


class PhishingDetector:
    """Phishing website detection model using Random Forest or XGBoost."""
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize the phishing detector.
        
        Args:
            model_type: 'random_forest' or 'xgboost'
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        
        # Model parameters
        self._init_model()
    
    def _init_model(self):
        """Initialize the ML model based on model_type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray],
              validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target labels (0 for legitimate, 1 for phishing)
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary of training metrics
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate on validation set
        metrics = self._evaluate(X_val_scaled, y_val)
        
        return metrics
    
    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions),
            'recall': recall_score(y, predictions),
            'f1_score': f1_score(y, predictions),
            'roc_auc': roc_auc_score(y, probabilities),
        }
        
        return metrics
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray, List[Dict]]) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix or list of feature dictionaries
            
        Returns:
            Predictions (0 for legitimate, 1 for phishing)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        X = self._prepare_features(X)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray, List[Dict]]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix or list of feature dictionaries
            
        Returns:
            Probabilities for each class [legitimate, phishing]
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        X = self._prepare_features(X)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def _prepare_features(self, X: Union[pd.DataFrame, np.ndarray, List[Dict]]) -> np.ndarray:
        """Prepare features for prediction."""
        if isinstance(X, list):
            # Convert list of dicts to array
            if self.feature_names is None:
                raise ValueError("Feature names not available. Train the model first.")
            X = pd.DataFrame(X)[self.feature_names].values
        elif isinstance(X, pd.DataFrame):
            if self.feature_names:
                X = X[self.feature_names].values
            else:
                X = X.values
        return np.array(X)
    
    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importances from the trained model."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting feature importances")
        
        importances = self.model.feature_importances_
        
        if self.feature_names:
            return dict(zip(self.feature_names, importances))
        else:
            return {f'feature_{i}': imp for i, imp in enumerate(importances)}
    
    def cross_validate(self, X: Union[pd.DataFrame, np.ndarray], 
                       y: Union[pd.Series, np.ndarray],
                       cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
        
        return {
            'cv_mean_accuracy': scores.mean(),
            'cv_std_accuracy': scores.std(),
        }
    
    def save(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'random_state': self.random_state,
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'PhishingDetector':
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        detector = cls(
            model_type=model_data['model_type'],
            random_state=model_data['random_state']
        )
        detector.model = model_data['model']
        detector.scaler = model_data['scaler']
        detector.feature_names = model_data['feature_names']
        detector.is_trained = True
        
        return detector


def compare_models(X: Union[pd.DataFrame, np.ndarray],
                   y: Union[pd.Series, np.ndarray],
                   validation_split: float = 0.2) -> Dict[str, Dict[str, float]]:
    """
    Compare Random Forest and XGBoost models.
    
    Returns:
        Dictionary with metrics for each model
    """
    results = {}
    
    for model_type in ['random_forest', 'xgboost']:
        print(f"\nTraining {model_type}...")
        detector = PhishingDetector(model_type=model_type)
        metrics = detector.train(X, y, validation_split=validation_split)
        cv_metrics = detector.cross_validate(X, y)
        metrics.update(cv_metrics)
        results[model_type] = metrics
    
    return results
