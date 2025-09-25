#!/usr/bin/env python3
"""
Hyperparameter tuning script for DVC pipeline.
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import yaml

def load_processed_data(data_dir):
    """Load preprocessed data."""
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv')['WnvPresent'].values

    with open(f'{data_dir}/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]

    return X_train, X_test, y_train, feature_names


def hyperparameter_tuning(X_train, y_train, params):
    """Perform hyperparameter tuning."""
    print("Starting hyperparameter tuning...")

    # Create pipeline
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    # Grid search with cross-validation
    print("Running grid search (this may take several minutes)...")
    grid_search = GridSearchCV(
        pipeline,
        params['param_grid'],
        cv=StratifiedKFold(n_splits=params['cv_folds'], shuffle=True, random_state=42),
        scoring=params['scoring_metric'],
        n_jobs=-1,
        verbose=1
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    print(f"\nBest CV ROC-AUC: {grid_search.best_score_:.4f}")
    print(f"Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_


def save_training_metrics(best_score, best_params, output_dir):
    """Save training metrics."""
    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        'best_cv_score': float(best_score),
        'best_params': best_params
    }

    with open(f'{output_dir}/training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Training metrics saved to {output_dir}/training_metrics.json")


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for WNV model')
    parser.add_argument('--data-dir', default='processed_data', help='Processed data directory')
    parser.add_argument('--params-file', default='params.yaml', help='Parameters file')
    parser.add_argument('--metrics-dir', default='metrics', help='Output directory for metrics')
    parser.add_argument('--models-dir', default='models', help='Output directory for models')

    args = parser.parse_args()

    # Load parameters
    with open(args.params_file, 'r') as f:
        params = yaml.safe_load(f)['train']

    # Load data
    X_train, X_test, y_train, feature_names = load_processed_data(args.data_dir)

    # Hyperparameter tuning
    best_model, best_score, best_params = hyperparameter_tuning(X_train, y_train, params)

    # Save metrics
    save_training_metrics(best_score, best_params, args.metrics_dir)

    # Save best model
    os.makedirs(args.models_dir, exist_ok=True)
    with open(f'{args.models_dir}/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    print("Hyperparameter tuning completed successfully!")


if __name__ == "__main__":
    main()