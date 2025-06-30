import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           roc_auc_score, precision_score, recall_score, f1_score, roc_curve)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load all datasets"""
    print("Loading datasets...")
    
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    weather_df = pd.read_csv('data/weather.csv')
    spray_df = pd.read_csv('data/spray.csv')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Weather data shape: {weather_df.shape}")
    print(f"Spray data shape: {spray_df.shape}")
    
    return train_df, test_df, weather_df, spray_df

def preprocess_weather_data(weather_df):
    """Preprocess weather data for merging"""
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    
    numeric_cols = ['Tmax', 'Tmin', 'Tavg', 'Depart', 'DewPoint', 'WetBulb', 
                   'Heat', 'Cool', 'PrecipTotal', 'StnPressure', 'SeaLevel', 
                   'ResultSpeed', 'AvgSpeed']
    
    for col in numeric_cols:
        if col in weather_df.columns:
            weather_df[col] = pd.to_numeric(weather_df[col], errors='coerce')
    
    weather_agg = weather_df.groupby('Date')[numeric_cols].mean().reset_index()
    return weather_agg

def add_spray_indicator(df, spray_df):
    """Add spray indicator to dataset"""
    spray_df['Date'] = pd.to_datetime(spray_df['Date'])
    df['Sprayed'] = 0
    
    for idx, row in df.iterrows():
        date = row['Date']
        lat = row['Latitude']
        lon = row['Longitude']
        
        spray_match = spray_df[
            (spray_df['Date'] == date) &
            (abs(spray_df['Latitude'] - lat) < 0.01) &
            (abs(spray_df['Longitude'] - lon) < 0.01)
        ]
        
        if not spray_match.empty:
            df.at[idx, 'Sprayed'] = 1
    
    return df

def merge_datasets(df, weather_df, spray_df):
    """Merge dataset with weather and spray data"""
    print("Merging datasets...")
    
    df['Date'] = pd.to_datetime(df['Date'])
    merged_df = pd.merge(df, weather_df, on='Date', how='left')
    merged_df = add_spray_indicator(merged_df, spray_df)
    
    print(f"Merged dataset shape: {merged_df.shape}")
    return merged_df

def preprocess_features(df, is_train=True):
    """Preprocess features for modeling"""
    print("Preprocessing features...")
    
    feature_columns = [
        'Species', 'Latitude', 'Longitude', 'Tmax', 'Tmin', 'Tavg', 
        'DewPoint', 'WetBulb', 'PrecipTotal', 'StnPressure', 'SeaLevel', 
        'ResultSpeed', 'AvgSpeed', 'Sprayed'
    ]
    
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Week'] = df['Date'].dt.isocalendar().week
    
    feature_columns.extend(['Year', 'Month', 'DayOfYear', 'Week'])
        
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features].copy()
    
    if is_train:
        y = df['WnvPresent']
        return X, y
    else:
        return X

def optimize_model():
    """Main optimization function with grid search and class imbalance handling"""
    print("=== West Nile Virus Predictor Optimization ===")
    
    train_df, test_df, weather_df, spray_df = load_data()
    
    weather_processed = preprocess_weather_data(weather_df)
    
    train_merged = merge_datasets(train_df, weather_processed, spray_df)
    test_merged = merge_datasets(test_df, weather_processed, spray_df)
    
    X_train_full, y_train_full = preprocess_features(train_merged, is_train=True)
    X_test = preprocess_features(test_merged, is_train=False)
    
    train_columns = set(X_train_full.columns)
    test_columns = set(X_test.columns)
    
    for col in train_columns - test_columns:
        X_test[col] = 0
    
    for col in test_columns - train_columns:
        X_train_full[col] = 0
    
    X_train_full = X_train_full.reindex(sorted(X_train_full.columns), axis=1)
    X_test = X_test.reindex(sorted(X_test.columns), axis=1)
    
    le_dict = {}
    for col in ['Species']:
        if col in X_train_full.columns:
            le = LabelEncoder()
            X_train_full[col] = le.fit_transform(X_train_full[col].astype(str))
            le_dict[col] = le
            
            if col in X_test.columns:
                X_test[col] = X_test[col].astype(str)
                unknown_labels = set(X_test[col]) - set(le.classes_)
                if unknown_labels:
                    le.classes_ = np.append(le.classes_, list(unknown_labels))
                X_test[col] = le.transform(X_test[col])
    
    print(f"Target distribution in full training data:")
    print(y_train_full.value_counts())
    print(f"Class imbalance ratio: {y_train_full.value_counts()[0] / y_train_full.value_counts()[1]:.2f}:1")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [10, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__class_weight': ['balanced', None],
        'smote__sampling_strategy': [0.3, 0.5]
    }
    
    print("Starting grid search with ROC-AUC optimization...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best ROC-AUC score: {grid_search.best_score_:.4f}")
    print("Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    best_model = grid_search.best_estimator_
    
    print("\n=== Validation Set Evaluation ===")
    y_val_pred = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    
    val_roc_auc = roc_auc_score(y_val, y_val_proba)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    print(f"Validation ROC-AUC: {val_roc_auc:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    print(f"Validation F1-Score: {val_f1:.4f}")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred))
    
    final_pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42, sampling_strategy=grid_search.best_params_['smote__sampling_strategy'], k_neighbors=3)),
        ('classifier', RandomForestClassifier(
            n_estimators=grid_search.best_params_['classifier__n_estimators'],
            max_depth=grid_search.best_params_['classifier__max_depth'],
            min_samples_split=grid_search.best_params_['classifier__min_samples_split'],
            class_weight=grid_search.best_params_['classifier__class_weight'],
            random_state=42
        ))
    ])
    
    print("\n=== Training Final Model on Full Training Data ===")
    final_pipeline.fit(X_train_full, y_train_full)
    
    print("\n=== Test Set Evaluation ===")
    test_predictions = final_pipeline.predict(X_test)
    test_probabilities = final_pipeline.predict_proba(X_test)[:, 1]
    
    results_df = test_df[['Id']].copy()
    results_df['WnvPresent'] = test_predictions
    results_df['WnvPresent_Probability'] = test_probabilities
    
    results_df.to_csv('predictions_optimized.csv', index=False)
    
    print(f"Predictions saved to predictions_optimized.csv")
    print(f"Predicted positive cases: {sum(test_predictions)}")
    print(f"Total test samples: {len(test_predictions)}")
    print(f"Positive prediction rate: {sum(test_predictions)/len(test_predictions)*100:.2f}%")
    
    if hasattr(final_pipeline.named_steps['classifier'], 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train_full.columns,
            'importance': final_pipeline.named_steps['classifier'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
    
    return final_pipeline, results_df, grid_search

if __name__ == "__main__":
    model, predictions, grid_search = optimize_model()