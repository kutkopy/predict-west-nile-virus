#!/usr/bin/env python3
"""
Enhanced West Nile Virus Predictor

This script trains an improved WNV prediction model using the preprocessed data
with advanced feature engineering from data_preprocessor.py.

Key improvements over the original model:
- Uses 43 engineered features vs basic features
- Better handling of class imbalance with SMOTE
- More comprehensive hyperparameter tuning
- Advanced evaluation metrics and feature importance analysis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           average_precision_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_preprocessor import WestNileDataPreprocessor


class EnhancedWNVPredictor:
    """Enhanced West Nile Virus predictor with advanced preprocessing."""
    
    def __init__(self, use_preprocessed_data=True):
        """Initialize the predictor."""
        self.use_preprocessed_data = use_preprocessed_data
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.cv_scores = None
        
    def load_data(self):
        """Load data (either preprocessed or raw)."""
        if self.use_preprocessed_data:
            print("Loading preprocessed data...")
            try:
                # Try to load preprocessed data
                self.X_train = pd.read_csv('processed_data/X_train.csv')
                self.X_test = pd.read_csv('processed_data/X_test.csv')
                self.y_train = pd.read_csv('processed_data/y_train.csv')['WnvPresent'].values
                
                with open('processed_data/feature_names.txt', 'r') as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
                
                print(f"Loaded preprocessed data: X_train {self.X_train.shape}, X_test {self.X_test.shape}")
                
            except FileNotFoundError:
                print("Preprocessed data not found. Running preprocessing...")
                self._run_preprocessing()
        else:
            print("Running fresh preprocessing...")
            self._run_preprocessing()
            
        return self
    
    def _run_preprocessing(self):
        """Run preprocessing if needed."""
        preprocessor = WestNileDataPreprocessor()
        self.X_train, self.X_test, self.y_train = preprocessor.process_all_data()
        self.feature_names = list(self.X_train.columns)
        
        # Save for future use
        preprocessor.save_processed_data(self.X_train, self.X_test, self.y_train)
        
    def analyze_data(self):
        """Analyze the preprocessed data."""
        print("\n" + "="*60)
        print("DATA ANALYSIS")
        print("="*60)
        
        print(f"Training samples: {len(self.X_train):,}")
        print(f"Test samples: {len(self.X_test):,}")
        print(f"Features: {len(self.feature_names)}")
        
        # Class distribution
        unique, counts = np.unique(self.y_train, return_counts=True)
        print(f"\nClass distribution:")
        for val, count in zip(unique, counts):
            print(f"  Class {val}: {count:,} ({count/len(self.y_train)*100:.2f}%)")
        
        # Feature statistics
        print(f"\nFeature statistics:")
        print(f"  Mean feature value: {self.X_train.mean().mean():.4f}")
        print(f"  Feature std: {self.X_train.std().mean():.4f}")
        print(f"  Missing values: {self.X_train.isnull().sum().sum()}")
        
        return self
    
    def train_baseline_model(self):
        """Train a baseline model for comparison."""
        print("\n" + "="*60)
        print("TRAINING BASELINE MODEL")
        print("="*60)
        
        # Simple Random Forest with SMOTE
        baseline_pipeline = ImbPipeline([
            ('smote', SMOTE(sampling_strategy=0.3, random_state=42)),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # Train baseline
        print("Training baseline model...")
        baseline_pipeline.fit(self.X_train, self.y_train)
        
        # Evaluate baseline
        cv_scores = cross_val_score(
            baseline_pipeline, self.X_train, self.y_train,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        print(f"Baseline CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        self.baseline_model = baseline_pipeline
        self.baseline_cv_scores = cv_scores
        
        return self
    
    def hyperparameter_tuning(self):
        """Perform comprehensive hyperparameter tuning."""
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)
        
        # Define parameter grid
        param_grid = {
            'smote__sampling_strategy': [0.2, 0.3, 0.5],
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [10, 15, 20, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__class_weight': [None, 'balanced']
        }
        
        # Create pipeline
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
        ])
        
        # Grid search with cross-validation
        print("Running grid search (this may take several minutes)...")
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(self.X_train, self.y_train)
        
        # Store results
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_scores = grid_search.best_score_
        
        print(f"\nBest CV ROC-AUC: {self.cv_scores:.4f}")
        print(f"Best parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        return self
    
    def evaluate_model(self):
        """Comprehensive model evaluation."""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        print(f"5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train final model for feature importance
        print("Training final model...")
        self.model.fit(self.X_train, self.y_train)
        
        # Get predictions on training set (for evaluation)
        y_pred_proba = self.model.predict_proba(self.X_train)[:, 1]
        y_pred = self.model.predict(self.X_train)
        
        # Calculate metrics
        train_auc = roc_auc_score(self.y_train, y_pred_proba)
        train_ap = average_precision_score(self.y_train, y_pred_proba)
        
        print(f"\nTraining Set Performance:")
        print(f"  ROC-AUC: {train_auc:.4f}")
        print(f"  Average Precision: {train_ap:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(self.y_train, y_pred))
        
        # Feature importance
        self._analyze_feature_importance()
        
        return self
    
    def _analyze_feature_importance(self):
        """Analyze feature importance."""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Get feature importance from the classifier
        classifier = self.model.named_steps['classifier']
        importance = classifier.feature_importances_
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = feature_importance_df
        
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
        plt.show()
        
        return self
    
    def generate_predictions(self):
        """Generate predictions for test set."""
        print("\n" + "="*60)
        print("GENERATING PREDICTIONS")
        print("="*60)
        
        # Generate predictions
        print("Generating test predictions...")
        test_predictions_proba = self.model.predict_proba(self.X_test)[:, 1]
        test_predictions = self.model.predict(self.X_test)
        
        # Create predictions dataframe
        predictions_df = pd.DataFrame({
            'Id': range(len(test_predictions)),
            'WnvPresent': test_predictions_proba
        })
        
        # Save predictions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'predictions_enhanced_{timestamp}.csv'
        predictions_df.to_csv(filename, index=False)
        
        print(f"Predictions saved to: {filename}")
        print(f"Total test samples: {len(predictions_df):,}")
        print(f"Predicted positive samples: {test_predictions.sum():,} ({test_predictions.mean()*100:.2f}%)")
        print(f"Mean prediction probability: {test_predictions_proba.mean():.4f}")
        print(f"Prediction probability range: {test_predictions_proba.min():.4f} - {test_predictions_proba.max():.4f}")
        
        self.predictions = predictions_df
        
        return self
    
    def compare_with_baseline(self):
        """Compare enhanced model with baseline."""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        print(f"Baseline CV ROC-AUC:  {self.baseline_cv_scores.mean():.4f} ± {self.baseline_cv_scores.std():.4f}")
        print(f"Enhanced CV ROC-AUC:  {self.cv_scores:.4f}")
        
        improvement = self.cv_scores - self.baseline_cv_scores.mean()
        print(f"Improvement:          +{improvement:.4f} ({improvement/self.baseline_cv_scores.mean()*100:+.2f}%)")
        
        return self
    
    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        print("=" * 80)
        print("ENHANCED WEST NILE VIRUS PREDICTOR")
        print("=" * 80)
        
        # Load and analyze data
        self.load_data()
        self.analyze_data()
        
        # Train baseline for comparison
        self.train_baseline_model()
        
        # Hyperparameter tuning
        self.hyperparameter_tuning()
        
        # Evaluate model
        self.evaluate_model()
        
        # Generate predictions
        self.generate_predictions()
        
        # Compare models
        self.compare_with_baseline()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return self


def main():
    """Main function to run the enhanced WNV predictor."""
    try:
        # Initialize and run predictor
        predictor = EnhancedWNVPredictor(use_preprocessed_data=True)
        predictor.run_full_pipeline()
        
        return predictor
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    predictor = main()