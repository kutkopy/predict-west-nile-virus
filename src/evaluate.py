#!/usr/bin/env python3
"""
Model evaluation script for DVC pipeline.
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                           roc_auc_score, roc_curve, precision_recall_curve,
                           average_precision_score)
import yaml


def load_processed_data(data_dir):
    """Load preprocessed data."""
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv')['WnvPresent'].values

    with open(f'{data_dir}/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]

    return X_train, X_test, y_train, feature_names


def load_best_model(models_dir):
    """Load the best model."""
    with open(f'{models_dir}/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def evaluate_model(model, X_train, y_train, feature_names):
    """Comprehensive model evaluation."""
    print("Evaluating model...")

    # Cross-validation scores
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1
    )

    print(f"5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Train final model for detailed evaluation
    print("Training final model...")
    model.fit(X_train, y_train)

    # Get predictions on training set
    y_pred_proba = model.predict_proba(X_train)[:, 1]
    y_pred = model.predict(X_train)

    # Calculate metrics
    train_auc = roc_auc_score(y_train, y_pred_proba)
    train_ap = average_precision_score(y_train, y_pred_proba)

    print(f"\nTraining Set Performance:")
    print(f"  ROC-AUC: {train_auc:.4f}")
    print(f"  Average Precision: {train_ap:.4f}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_train, y_pred))

    # Feature importance analysis
    feature_importance = analyze_feature_importance(model, feature_names)

    return {
        'cv_scores': cv_scores,
        'train_auc': train_auc,
        'train_ap': train_ap,
        'feature_importance': feature_importance
    }


def analyze_feature_importance(model, feature_names):
    """Analyze feature importance."""
    print("\nAnalyzing feature importance...")

    # Get feature importance from the classifier
    classifier = model.named_steps['classifier']
    importance = classifier.feature_importances_

    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # Display top features
    print("Top 20 Most Important Features:")
    print("-" * 40)
    for i, (_, row) in enumerate(feature_importance_df.head(20).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<25} {row['importance']:.4f}")

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(20)

    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()  # Don't display, just save

    print("Feature importance plot saved to feature_importance.png")

    return feature_importance_df.to_dict('records')


def save_evaluation_metrics(evaluation_results, output_dir):
    """Save evaluation metrics."""
    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        'cv_mean': float(evaluation_results['cv_scores'].mean()),
        'cv_std': float(evaluation_results['cv_scores'].std()),
        'cv_scores': evaluation_results['cv_scores'].tolist(),
        'train_auc': float(evaluation_results['train_auc']),
        'train_ap': float(evaluation_results['train_ap']),
        'top_10_features': evaluation_results['feature_importance'][:10]
    }

    with open(f'{output_dir}/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation metrics saved to {output_dir}/evaluation_metrics.json")


def main():
    parser = argparse.ArgumentParser(description='Evaluate WNV model')
    parser.add_argument('--data-dir', default='processed_data', help='Processed data directory')
    parser.add_argument('--models-dir', default='models', help='Models directory')
    parser.add_argument('--metrics-dir', default='metrics', help='Output directory for metrics')

    args = parser.parse_args()

    # Load data
    X_train, X_test, y_train, feature_names = load_processed_data(args.data_dir)

    # Load best model
    model = load_best_model(args.models_dir)

    # Evaluate model
    evaluation_results = evaluate_model(model, X_train, y_train, feature_names)

    # Save metrics
    save_evaluation_metrics(evaluation_results, args.metrics_dir)

    print("Model evaluation completed successfully!")


if __name__ == "__main__":
    main()