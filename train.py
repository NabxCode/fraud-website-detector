"""
Training script for the phishing detection model.
"""

import os
import sys
import argparse
import json

import pandas as pd
import numpy as np

from src.data_loader import DataLoader
from src.model import PhishingDetector, compare_models
from src.feature_extraction import URLFeatureExtractor


def train_with_uci_dataset(dataset_path: str, model_type: str = 'random_forest',
                           output_dir: str = 'models'):
    """Train model using UCI Phishing Dataset."""
    print("=" * 60)
    print("Training Phishing Detection Model (UCI Dataset)")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading dataset from {dataset_path}...")
    loader = DataLoader()
    X, y = loader.load_uci_dataset(dataset_path)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # Initialize and train model
    print(f"\nTraining {model_type} model...")
    detector = PhishingDetector(model_type=model_type)
    
    # Train with validation
    metrics = detector.train(X, y, validation_split=0.2)
    
    print("\nValidation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_metrics = detector.cross_validate(X, y, cv=5)
    print(f"  CV Accuracy: {cv_metrics['cv_mean_accuracy']:.4f} (+/- {cv_metrics['cv_std_accuracy']:.4f})")
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    importances = detector.get_feature_importances()
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_importances[:10]:
        print(f"  {feature}: {importance:.4f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'phishing_detector.pkl')
    detector.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    all_metrics = {**metrics, **cv_metrics}
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    return detector


def train_with_custom_dataset(dataset_path: str, model_type: str = 'random_forest',
                              output_dir: str = 'models'):
    """Train model using custom URL dataset."""
    print("=" * 60)
    print("Training Phishing Detection Model (Custom Dataset)")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading custom dataset from {dataset_path}...")
    loader = DataLoader()
    X, y = loader.load_custom_dataset(dataset_path)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # Train model
    print(f"\nTraining {model_type} model...")
    detector = PhishingDetector(model_type=model_type)
    metrics = detector.train(X, y, validation_split=0.2)
    
    print("\nValidation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Cross-validation
    cv_metrics = detector.cross_validate(X, y, cv=5)
    print(f"\nCV Accuracy: {cv_metrics['cv_mean_accuracy']:.4f} (+/- {cv_metrics['cv_std_accuracy']:.4f})")
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    importances = detector.get_feature_importances()
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_importances[:10]:
        print(f"  {feature}: {importance:.4f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'phishing_detector.pkl')
    detector.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    return detector


def compare_both_models(dataset_path: str, dataset_type: str = 'uci'):
    """Compare Random Forest and XGBoost models."""
    print("=" * 60)
    print("Comparing Random Forest vs XGBoost")
    print("=" * 60)
    
    loader = DataLoader()
    
    if dataset_type == 'uci':
        X, y = loader.load_uci_dataset(dataset_path)
    else:
        X, y = loader.load_custom_dataset(dataset_path)
    
    print(f"\nDataset shape: {X.shape}")
    
    results = compare_models(X, y, validation_split=0.2)
    
    print("\n" + "=" * 60)
    print("Comparison Results:")
    print("=" * 60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Determine best model based on F1 score
    best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    print(f"\nBest Model (based on F1): {best_model[0]}")
    
    return results


def create_sample_and_train(model_type: str = 'random_forest', 
                            n_samples: int = 100):
    """Create sample dataset and train model."""
    print("Creating sample dataset...")
    
    loader = DataLoader()
    loader.create_sample_dataset('data/sample_urls.csv', n_samples=n_samples)
    
    return train_with_custom_dataset('data/sample_urls.csv', model_type=model_type)


def main():
    parser = argparse.ArgumentParser(description='Train phishing detection model')
    parser.add_argument('--dataset', type=str, help='Path to dataset file')
    parser.add_argument('--dataset-type', type=str, default='uci', 
                        choices=['uci', 'custom'],
                        help='Type of dataset (uci or custom)')
    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['random_forest', 'xgboost'],
                        help='Model type to train')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save model')
    parser.add_argument('--compare', action='store_true',
                        help='Compare Random Forest and XGBoost')
    parser.add_argument('--sample', action='store_true',
                        help='Create sample dataset and train')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of samples for sample dataset')
    
    args = parser.parse_args()
    
    if args.sample:
        create_sample_and_train(args.model_type, args.n_samples)
    elif args.compare:
        if not args.dataset:
            print("Error: --dataset required for comparison")
            sys.exit(1)
        compare_both_models(args.dataset, args.dataset_type)
    elif args.dataset:
        if args.dataset_type == 'uci':
            train_with_uci_dataset(args.dataset, args.model_type, args.output_dir)
        else:
            train_with_custom_dataset(args.dataset, args.model_type, args.output_dir)
    else:
        # Default: create sample dataset
        print("No dataset provided. Creating sample dataset...")
        create_sample_and_train(args.model_type, args.n_samples)


if __name__ == '__main__':
    main()
