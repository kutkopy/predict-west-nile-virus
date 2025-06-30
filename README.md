# West Nile Virus Prediction

A machine learning project to predict the presence of West Nile Virus in mosquito traps across Chicago using weather data, spray treatments, and mosquito surveillance data.

## Overview

This project uses scikit-learn to build an optimized Random Forest classifier that predicts whether West Nile Virus is present in mosquito samples. The model addresses class imbalance through SMOTE (Synthetic Minority Oversampling Technique) and uses grid search for hyperparameter optimization.

## Dataset

The project uses four main datasets:
- **train.csv**: Training data with mosquito trap locations, species, and WnvPresent labels
- **test.csv**: Test data for making predictions (no WnvPresent labels)
- **weather.csv**: Weather station data including temperature, humidity, precipitation
- **spray.csv**: Mosquito spray treatment locations and dates

## Features

The model uses the following features:
- **Geographic**: Latitude, Longitude
- **Temporal**: Year, Month, DayOfYear, Week
- **Biological**: Species (encoded)
- **Environmental**: Temperature (Tmax, Tmin, Tavg), DewPoint, WetBulb, PrecipTotal, StnPressure, SeaLevel, ResultSpeed, AvgSpeed
- **Treatment**: Sprayed (binary indicator for spray treatments)

## Model Performance

### Final Results (Validation Set):
- **ROC-AUC**: 0.8268
- **Precision**: 0.1871
- **Recall**: 0.5000
- **F1-Score**: 0.2723

### Grid Search Optimization:
- **Best Cross-Validation ROC-AUC**: 0.8346
- **Optimal Parameters**:
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 5
  - class_weight: None
  - SMOTE sampling_strategy: 0.5

### Class Imbalance Handling:
- **Original Imbalance**: 18:1 ratio (9,955 negative vs 551 positive cases)
- **Solution**: SMOTE oversampling with 50% sampling strategy
- **Result**: 7,425 positive predictions out of 116,293 test samples (6.38%)

## Top Important Features

1. **Species** (13.6%): Type of mosquito species
2. **DayOfYear** (13.5%): Seasonal timing
3. **Longitude** (11.3%): Geographic location (east-west)
4. **Latitude** (9.4%): Geographic location (north-south)
5. **Month** (8.9%): Monthly seasonality

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the optimized predictor:
```bash
python wnv_predictor_optimized.py
```

This will:
1. Load and merge all datasets
2. Preprocess features and handle missing values
3. Perform grid search optimization with ROC-AUC scoring
4. Train the final model on full training data
5. Generate predictions for the test set
6. Save results to `predictions_optimized.csv`

## Output Files

- **predictions_optimized.csv**: Final predictions with probability scores
- **Feature importance rankings**: Displayed in console output
- **Performance metrics**: Validation and cross-validation results

## Technical Details

### Data Preprocessing:
- Date feature engineering (Year, Month, DayOfYear, Week)
- Label encoding for categorical variables (Species)
- Missing value imputation using median strategy
- Standard scaling for numerical features
- Spray treatment mapping based on geographic proximity

### Model Pipeline:
1. **SimpleImputer**: Handle missing values
2. **StandardScaler**: Feature scaling
3. **SMOTE**: Address class imbalance
4. **RandomForestClassifier**: Final prediction model

### Cross-Validation:
- 3-fold stratified cross-validation
- ROC-AUC optimization metric
- Grid search over 32 parameter combinations

## Files Structure

```
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── weather.csv
│   └── spray.csv
├── wnv_predictor_optimized.py
├── predictions_optimized.csv
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.7+
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- imbalanced-learn >= 0.11.0

## Notes

- The model excludes `NumMosquitos` feature as it's not available in the test set
- SMOTE helps address the severe class imbalance (18:1 ratio)
- Grid search optimizes for ROC-AUC to handle imbalanced classification
- Geographic and temporal features are the most important predictors