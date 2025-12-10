#!/usr/bin/env python3
"""
Hyperparameter tuning script with MLflow tracking.
Compatible with Azure ML MLflow tracking.
"""
import argparse
import json
import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import yaml
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


def setup_mlflow(experiment_name="wnv-prediction", tracking_uri=None):
    """
    Configure MLflow tracking.
    
    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking server URI (None for local tracking)
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Set or create experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                tags={"project": "west-nile-virus", "task": "classification"}
            )
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment: {experiment_name} (ID: {experiment_id})")
    except Exception as e:
        print(f"Warning: Could not set up MLflow experiment: {e}")
        print("Proceeding with default experiment")


def load_processed_data(data_dir):
    """Load preprocessed data."""
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv')['WnvPresent'].values
    
    with open(f'{data_dir}/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    return X_train, X_test, y_train, feature_names


def hyperparameter_tuning(X_train, y_train, params):
    """Perform hyperparameter tuning with MLflow logging."""
    print("Starting hyperparameter tuning...")
    
    # Create pipeline
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    # Log SMOTE parameters
    mlflow.log_param("smote_sampling_strategy", "auto")
    mlflow.log_param("smote_random_state", 42)
    
    # Grid search with cross-validation
    print("Running grid search (this may take several minutes)...")
    grid_search = GridSearchCV(
        pipeline,
        params['param_grid'],
        cv=StratifiedKFold(n_splits=params['cv_folds'], shuffle=True, random_state=42),
        scoring=params['scoring_metric'],
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Log grid search results
    log_grid_search_results(grid_search, params)
    
    print(f"\nBest CV {params['scoring_metric']}: {grid_search.best_score_:.4f}")
    print(f"Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_, grid_search


def log_grid_search_results(grid_search, params):
    """Log detailed grid search results to MLflow."""
    # Log CV configuration
    mlflow.log_param("cv_folds", params['cv_folds'])
    mlflow.log_param("scoring_metric", params['scoring_metric'])
    
    # Log best parameters
    for param_name, param_value in grid_search.best_params_.items():
        mlflow.log_param(f"best_{param_name}", param_value)
    
    # Log best score and std
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    
    # Get the best index to find std
    best_index = grid_search.best_index_
    best_std = grid_search.cv_results_['std_test_score'][best_index]
    mlflow.log_metric("best_cv_score_std", best_std)
    
    # Log mean train score for best params (to detect overfitting)
    if 'mean_train_score' in grid_search.cv_results_:
        best_train_score = grid_search.cv_results_['mean_train_score'][best_index]
        mlflow.log_metric("best_cv_train_score", best_train_score)
        mlflow.log_metric("cv_overfit_gap", best_train_score - grid_search.best_score_)
    
    # Log parameter grid size
    mlflow.log_param("param_grid_size", len(grid_search.cv_results_['params']))
    
    # Create comprehensive CV results
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df = cv_results_df.sort_values('rank_test_score')
    
    # Use timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_results_path = f"cv_results_{timestamp}.csv"
    
    try:
        cv_results_df.to_csv(cv_results_path, index=False)
        mlflow.log_artifact(cv_results_path, artifact_path="results")
        print(f"CV results logged successfully")
    except Exception as e:
        print(f"Warning: Could not log CV results artifact: {e}")
    finally:
        if os.path.exists(cv_results_path):
            os.remove(cv_results_path)
    
    # Log score distribution metrics
    mlflow.log_metric("cv_score_mean", cv_results_df['mean_test_score'].mean())
    mlflow.log_metric("cv_score_std", cv_results_df['mean_test_score'].std())
    mlflow.log_metric("cv_score_min", cv_results_df['mean_test_score'].min())
    mlflow.log_metric("cv_score_max", cv_results_df['mean_test_score'].max())
    
    # Print top 5 parameter combinations
    print("\n" + "="*100)
    print("TOP 5 PARAMETER COMBINATIONS")
    print("="*100)
    
    top_5 = cv_results_df.head(5)
    for idx, row in top_5.iterrows():
        rank = int(row['rank_test_score'])
        score = row['mean_test_score']
        std = row['std_test_score']
        params_dict = row['params']
        
        print(f"\nRank {rank}: Score = {score:.4f} (+/- {std:.4f})")
        for param, value in params_dict.items():
            print(f"  {param}: {value}")
    
    print("\n" + "="*100)


def evaluate_train_set(model, X_train, y_train):
    """Evaluate model on training set and log metrics to MLflow."""
    print("\nEvaluating model on training set...")
    
    # Train predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_proba)
    
    # Log metrics
    mlflow.log_metric("train_roc_auc", train_auc)
    
    # Classification report
    train_report = classification_report(y_train, y_train_pred, output_dict=True)
    mlflow.log_metric("train_precision", train_report['1']['precision'])
    mlflow.log_metric("train_recall", train_report['1']['recall'])
    mlflow.log_metric("train_f1", train_report['1']['f1-score'])
    mlflow.log_metric("train_accuracy", train_report['accuracy'])
    
    # Confusion matrix
    cm = confusion_matrix(y_train, y_train_pred)
    mlflow.log_metric("train_true_negatives", int(cm[0, 0]))
    mlflow.log_metric("train_false_positives", int(cm[0, 1]))
    mlflow.log_metric("train_false_negatives", int(cm[1, 0]))
    mlflow.log_metric("train_true_positives", int(cm[1, 1]))
    
    print(f"Train ROC-AUC: {train_auc:.4f}")
    print(f"Train Precision: {train_report['1']['precision']:.4f}")
    print(f"Train Recall: {train_report['1']['recall']:.4f}")
    
    return {
        'train_auc': train_auc,
        'train_metrics': train_report
    }


def generate_test_predictions(model, X_test, output_dir):
    """Generate predictions for test set and save them."""
    print("\nGenerating predictions for test set...")
    
    # Generate predictions
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'prediction': y_test_pred,
        'probability': y_test_proba
    })
    
    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    predictions_path = f'{output_dir}/test_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    
    # Log as artifact
    try:
        mlflow.log_artifact(predictions_path, artifact_path="predictions")
        print(f"Test predictions logged to MLflow")
    except Exception as e:
        print(f"Warning: Could not log predictions artifact: {e}")
    
    # Log prediction statistics
    mlflow.log_metric("test_predicted_positive_rate", float(y_test_pred.mean()))
    mlflow.log_metric("test_mean_probability", float(y_test_proba.mean()))
    mlflow.log_metric("test_std_probability", float(y_test_proba.std()))
    
    print(f"Test predictions saved to {predictions_path}")
    print(f"Predicted positive rate: {y_test_pred.mean():.4f}")
    print(f"Mean probability: {y_test_proba.mean():.4f}")
    
    return predictions_df


def log_feature_importance(model, feature_names, top_n=20):
    """Log feature importance to MLflow."""
    # Get feature importance from the classifier step
    classifier = model.named_steps['classifier']
    importance = classifier.feature_importances_
    
    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Log top features as metrics
    for idx, row in feature_importance_df.head(top_n).iterrows():
        feature_name_safe = row['feature'].replace('/', '_').replace(':', '_').replace(' ', '_')
        mlflow.log_metric(f"importance_{feature_name_safe}", row['importance'])
    
    # Save and log full feature importance
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    importance_path = f"feature_importance_{timestamp}.csv"
    
    try:
        feature_importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path, artifact_path="feature_importance")
        print(f"Feature importance logged to MLflow")
    except Exception as e:
        print(f"Warning: Could not log feature importance artifact: {e}")
    finally:
        if os.path.exists(importance_path):
            os.remove(importance_path)
    
    print(f"\nTop {top_n} most important features:")
    for idx, row in feature_importance_df.head(top_n).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return feature_importance_df


def save_training_metrics(best_score, best_params, train_results, output_dir):
    """Save training metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = {
        'best_cv_score': float(best_score),
        'best_params': best_params,
        'train_auc': float(train_results['train_auc']),
        'train_metrics': train_results['train_metrics']
    }
    
    metrics_path = f'{output_dir}/training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nTraining metrics saved to {metrics_path}")
    
    # Log metrics file as artifact
    try:
        mlflow.log_artifact(metrics_path, artifact_path="metrics")
        print(f"Metrics file logged to MLflow")
    except Exception as e:
        print(f"Warning: Could not log metrics artifact: {e}")


def log_model_to_mlflow(model, X_train, model_path, register_model=False, model_name=None, run_id=None):
    """
    Log model to MLflow with Azure ML compatibility.
    
    Args:
        model: Trained model to log
        X_train: Training data for signature inference
        model_path: Path to the saved pickle model
        register_model: Whether to register in model registry
        model_name: Name for model registry
        run_id: Current MLflow run ID
    
    Returns:
        bool: True if model was logged successfully
    """
    print("\nLogging model to MLflow...")
    model_logged = False
    
    # Create signature for model
    try:
        signature = infer_signature(X_train, model.predict_proba(X_train))
    except Exception as e:
        print(f"Warning: Could not infer signature: {e}")
        signature = None
    
    # Attempt 1: Use mlflow.sklearn.log_model (preferred)
    try:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            signature=signature,
            input_example=X_train.iloc[:5] if signature else None,
        )
        print("Model logged successfully with mlflow.sklearn.log_model")
        model_logged = True
    except Exception as e:
        print(f"Warning: mlflow.sklearn.log_model failed: {e}")
        
        # Attempt 2: Log pickle file as artifact
        try:
            mlflow.log_artifact(model_path, artifact_path="model")
            print("Model logged as pickle artifact")
            model_logged = True
        except Exception as e2:
            print(f"Warning: Could not log model artifact: {e2}")
    
    # Register model if requested and model was logged
    if register_model and model_name and run_id and model_logged:
        print(f"\nRegistering model as '{model_name}'...")
        try:
            # Try sklearn model path first, fall back to pickle
            model_uri = f"runs:/{run_id}/sklearn-model"
            
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            print(f"Model registered: {model_name} (version {registered_model.version})")
            
            # Add description
            try:
                client = mlflow.tracking.MlflowClient()
                client.update_model_version(
                    name=model_name,
                    version=registered_model.version,
                    description=f"Random Forest with SMOTE - trained {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
            except Exception as e:
                print(f"Warning: Could not update model description: {e}")
                
        except Exception as e:
            print(f"Warning: Could not register model: {e}")
            print("Model is still available in the run artifacts")
    
    return model_logged


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for WNV model')
    parser.add_argument('--data-dir', default='processed_data', help='Processed data directory')
    parser.add_argument('--params-file', default='params.yaml', help='Parameters file')
    parser.add_argument('--metrics-dir', default='metrics', help='Output directory for metrics')
    parser.add_argument('--models-dir', default='models', help='Output directory for models')
    parser.add_argument('--mlflow-tracking-uri', default=None, help='MLflow tracking URI')
    parser.add_argument('--experiment-name', default='wnv-prediction', help='MLflow experiment name')
    parser.add_argument('--run-name', default=None, help='MLflow run name')
    parser.add_argument('--register-model', action='store_true', help='Register model in MLflow Model Registry')
    parser.add_argument('--model-name', default='wnv-classifier', help='Model name for registry')
    
    args = parser.parse_args()
    
    # Load parameters
    with open(args.params_file, 'r') as f:
        params = yaml.safe_load(f)['train']
    
    # Setup MLflow
    setup_mlflow(args.experiment_name, args.mlflow_tracking_uri)
    
    # Load data
    X_train, X_test, y_train, feature_names = load_processed_data(args.data_dir)
    
    # Generate run name if not provided
    run_name = args.run_name or f"wnv_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        try:
            # Log run metadata
            mlflow.set_tags({
                "task": "hyperparameter_tuning",
                "model_type": "RandomForestClassifier",
                "pipeline": "SMOTE + RandomForest",
                "data_dir": args.data_dir,
            })
            
            # Log data characteristics
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("train_positive_rate", float(y_train.sum() / len(y_train)))
            
            # Log parameters file
            try:
                mlflow.log_artifact(args.params_file, artifact_path="config")
                print("Parameters file logged to MLflow")
            except Exception as e:
                print(f"Warning: Could not log params file: {e}")
            
            # Hyperparameter tuning
            best_model, best_score, best_params, grid_search = hyperparameter_tuning(
                X_train, y_train, params
            )
            
            # Evaluate model on training set
            train_results = evaluate_train_set(best_model, X_train, y_train)
            
            # Log feature importance
            feature_importance_df = log_feature_importance(best_model, feature_names, top_n=20)
            
            # Generate test predictions
            predictions_df = generate_test_predictions(best_model, X_test, args.metrics_dir)
            
            # Save metrics
            save_training_metrics(best_score, best_params, train_results, args.metrics_dir)
            
            # Save best model with pickle
            os.makedirs(args.models_dir, exist_ok=True)
            model_path = f'{args.models_dir}/best_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            print(f"Model saved to {model_path}")
            
            # Log the pickle model as artifact (always do this as backup)
            try:
                mlflow.log_artifact(model_path, artifact_path="pickle-model")
                print(f"Pickle model logged to MLflow")
            except Exception as e:
                print(f"Warning: Could not log pickle model: {e}")
            
            # Log model to MLflow with Azure ML compatibility
            log_model_to_mlflow(
                model=best_model,
                X_train=X_train,
                model_path=model_path,
                register_model=args.register_model,
                model_name=args.model_name,
                run_id=run.info.run_id
            )
            
            print(f"\n{'='*100}")
            print(f"✓ Hyperparameter tuning completed successfully!")
            print(f"✓ MLflow run ID: {run.info.run_id}")
            print(f"✓ Best CV Score: {best_score:.4f}")
            print(f"✓ Train AUC: {train_results['train_auc']:.4f}")
            print(f"{'='*100}")
            
        except Exception as e:
            # Log error
            try:
                mlflow.log_param("error", str(e)[:250])
                mlflow.set_tag("status", "failed")
            except:
                pass
            print(f"\nError during training: {e}")
            raise
        else:
            mlflow.set_tag("status", "success")


if __name__ == "__main__":
    main()