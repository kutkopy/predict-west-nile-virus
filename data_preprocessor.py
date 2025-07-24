#!/usr/bin/env python3
"""
West Nile Virus Data Preprocessor

This script preprocesses the West Nile Virus dataset based on insights from data exploration.
It handles data loading, cleaning, feature engineering, and merging of all datasets.

Key insights applied:
- Severe class imbalance (18:1 ratio) requires careful handling
- Species is highly predictive with different WNV rates
- Temporal patterns show seasonality (peak in summer months)
- Geographic location is important (lat/lon features)
- Weather features correlate with WNV presence
- NumMosquitos is available in training but not test data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class WestNileDataPreprocessor:
    """Preprocesses West Nile Virus prediction data."""
    
    def __init__(self, data_dir='data'):
        """Initialize preprocessor with data directory."""
        self.data_dir = data_dir
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = []
        
    def load_data(self):
        """Load all datasets."""
        print("Loading datasets...")
        
        # Load main datasets
        self.train_df = pd.read_csv(f'{self.data_dir}/train.csv')
        self.test_df = pd.read_csv(f'{self.data_dir}/test.csv')
        self.weather_df = pd.read_csv(f'{self.data_dir}/weather.csv')
        self.spray_df = pd.read_csv(f'{self.data_dir}/spray.csv')
        
        print(f"Training data: {self.train_df.shape}")
        print(f"Test data: {self.test_df.shape}")
        print(f"Weather data: {self.weather_df.shape}")
        print(f"Spray data: {self.spray_df.shape}")
        
        # Convert date columns
        self.train_df['Date'] = pd.to_datetime(self.train_df['Date'])
        self.test_df['Date'] = pd.to_datetime(self.test_df['Date'])
        self.weather_df['Date'] = pd.to_datetime(self.weather_df['Date'])
        self.spray_df['Date'] = pd.to_datetime(self.spray_df['Date'])
        
        return self
    
    def clean_weather_data(self):
        """Clean and process weather data."""
        print("Cleaning weather data...")
        
        # Define numeric columns in weather data
        numeric_columns = ['Tmax', 'Tmin', 'Tavg', 'DewPoint', 'WetBulb', 
                          'PrecipTotal', 'StnPressure', 'SeaLevel', 
                          'ResultSpeed', 'AvgSpeed']
        
        # Clean each numeric column
        for col in numeric_columns:
            if col in self.weather_df.columns:
                # Handle special cases in PrecipTotal (T = trace amounts)
                if col == 'PrecipTotal':
                    self.weather_df[col] = self.weather_df[col].replace('T', '0.005')
                    self.weather_df[col] = self.weather_df[col].replace('  T', '0.005')
                
                # Convert to numeric
                self.weather_df[col] = pd.to_numeric(self.weather_df[col], errors='coerce')
        
        # Average weather data across stations by date
        weather_cols = ['Date'] + [col for col in numeric_columns if col in self.weather_df.columns]
        self.weather_processed = self.weather_df[weather_cols].groupby('Date').mean().reset_index()
        
        print(f"Weather data processed: {self.weather_processed.shape}")
        return self
    
    def create_temporal_features(self, df):
        """Create temporal features from date."""
        df = df.copy()
        
        # Extract temporal components
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['Week'] = df['Date'].dt.isocalendar().week
        df['WeekOfYear'] = df['Week']  # Alias for consistency
        
        # Cyclical encoding for seasonal patterns
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        
        # Peak season indicator (based on exploration: months 7-9 are peak)
        df['IsPeakSeason'] = ((df['Month'] >= 7) & (df['Month'] <= 9)).astype(int)
        
        return df
    
    def create_spray_features(self, df):
        """Create spray treatment features."""
        print("Creating spray features...")
        
        # Initialize spray features
        df = df.copy()
        df['Sprayed'] = 0
        df['DaysSinceSpray'] = 31  # Default: no recent spray
        
        # For efficiency, create a simplified approach
        spray_buffer = 0.01  # Approximately 1 km buffer
        spray_days_before = 30  # Consider spraying up to 30 days before
        
        print(f"Processing {len(df)} samples against {len(self.spray_df)} spray events...")
        
        # Vectorized approach for better performance
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"  Processed {idx}/{len(df)} samples...")
            
            # Get nearby spray events
            lat_mask = abs(self.spray_df['Latitude'] - row['Latitude']) <= spray_buffer
            lon_mask = abs(self.spray_df['Longitude'] - row['Longitude']) <= spray_buffer
            date_mask = (self.spray_df['Date'] <= row['Date']) & \
                       (self.spray_df['Date'] >= row['Date'] - pd.Timedelta(days=spray_days_before))
            
            spray_nearby = self.spray_df[lat_mask & lon_mask & date_mask]
            
            if len(spray_nearby) > 0:
                df.loc[idx, 'Sprayed'] = 1
                # Days since last spray
                days_since_spray = (row['Date'] - spray_nearby['Date'].max()).days
                df.loc[idx, 'DaysSinceSpray'] = days_since_spray
        
        print(f"Spray features created. {df['Sprayed'].sum()} samples marked as sprayed.")
        return df
    
    def encode_species(self, df, fit=True):
        """Encode species using insights from exploration."""
        df = df.copy()
        
        if fit:
            # Create label encoder for species
            self.label_encoders['Species'] = LabelEncoder()
            df['Species_encoded'] = self.label_encoders['Species'].fit_transform(df['Species'])
            # Store the classes and most common species for handling unseen species
            self.known_species = set(self.label_encoders['Species'].classes_)
            self.most_common_species = df['Species'].value_counts().index[0]
        else:
            # Handle unseen species in test data
            unknown_species_mask = ~df['Species'].isin(self.known_species)
            
            if unknown_species_mask.any():
                print(f"Warning: Found {unknown_species_mask.sum()} samples with unknown species:")
                print(df[unknown_species_mask]['Species'].value_counts().to_dict())
                
                # Replace unknown species with most common species from training
                df.loc[unknown_species_mask, 'Species'] = self.most_common_species
                print(f"Replaced unknown species with: {self.most_common_species}")
            
            # Transform using fitted encoder
            df['Species_encoded'] = self.label_encoders['Species'].transform(df['Species'])
        
        # Create high-risk species indicator based on exploration insights
        # From exploration: CULEX PIPIENS and CULEX RESTUANS have higher WNV rates
        high_risk_species = ['CULEX PIPIENS', 'CULEX RESTUANS', 'CULEX PIPIENS/RESTUANS']
        df['IsHighRiskSpecies'] = df['Species'].isin(high_risk_species).astype(int)
        
        return df
    
    def create_geographic_features(self, df):
        """Create geographic features."""
        df = df.copy()
        
        # Distance from city center (approximate Chicago center)
        chicago_center_lat = 41.8781
        chicago_center_lon = -87.6298
        
        df['DistanceFromCenter'] = np.sqrt(
            (df['Latitude'] - chicago_center_lat)**2 + 
            (df['Longitude'] - chicago_center_lon)**2
        )
        
        # Quadrant features (divide city into quadrants)
        df['NorthSide'] = (df['Latitude'] > chicago_center_lat).astype(int)
        df['WestSide'] = (df['Longitude'] < chicago_center_lon).astype(int)
        
        # Create location clusters based on lat/lon bins
        df['LatBin'] = pd.cut(df['Latitude'], bins=10, labels=False)
        df['LonBin'] = pd.cut(df['Longitude'], bins=10, labels=False)
        df['LocationCluster'] = df['LatBin'] * 10 + df['LonBin']
        
        return df
    
    def create_weather_derived_features(self, df):
        """Create derived weather features."""
        df = df.copy()
        
        # Temperature range and variations
        if all(col in df.columns for col in ['Tmax', 'Tmin']):
            df['TempRange'] = df['Tmax'] - df['Tmin']
            df['TempAvg'] = (df['Tmax'] + df['Tmin']) / 2
        
        # Heat index approximation (when temperature and humidity are high)
        if all(col in df.columns for col in ['Tavg', 'DewPoint']):
            # Relative humidity approximation
            df['RelativeHumidity'] = np.exp(
                17.625 * df['DewPoint'] / (243.04 + df['DewPoint'])
            ) / np.exp(17.625 * df['Tavg'] / (243.04 + df['Tavg']))
            df['RelativeHumidity'] = np.clip(df['RelativeHumidity'], 0, 1)
            
            # Heat stress indicator
            df['HeatStress'] = ((df['Tavg'] > 80) & (df['RelativeHumidity'] > 0.6)).astype(int)
        
        # Precipitation categories
        if 'PrecipTotal' in df.columns:
            df['HasPrecip'] = (df['PrecipTotal'] > 0).astype(int)
            df['HeavyRain'] = (df['PrecipTotal'] > 0.5).astype(int)
        
        # Wind categories
        if 'AvgSpeed' in df.columns:
            df['IsCalm'] = (df['AvgSpeed'] < 5).astype(int)
            df['IsWindy'] = (df['AvgSpeed'] > 15).astype(int)
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features based on domain knowledge."""
        df = df.copy()
        
        # Temperature-humidity interactions (important for mosquito breeding)
        if all(col in df.columns for col in ['Tavg', 'DewPoint']):
            df['TempHumidity'] = df['Tavg'] * df['DewPoint']
        
        # Species-season interactions
        if all(col in df.columns for col in ['Species_encoded', 'Month']):
            df['SpeciesMonth'] = df['Species_encoded'] * df['Month']
        
        # Location-season interactions
        if all(col in df.columns for col in ['LocationCluster', 'IsPeakSeason']):
            df['LocationSeason'] = df['LocationCluster'] * df['IsPeakSeason']
        
        return df
    
    def preprocess_dataset(self, df, is_training=True):
        """Preprocess a dataset (train or test)."""
        print(f"Preprocessing {'training' if is_training else 'test'} dataset...")
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Merge with weather data
        df = df.merge(self.weather_processed, on='Date', how='left')
        
        # Create spray features
        df = self.create_spray_features(df)
        
        # Encode species
        df = self.encode_species(df, fit=is_training)
        
        # Create geographic features
        df = self.create_geographic_features(df)
        
        # Create weather-derived features
        df = self.create_weather_derived_features(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        return df
    
    def select_features(self, df, is_training=True):
        """Select final features for modeling."""
        # Define feature columns (excluding target and metadata)
        feature_cols = [
            # Geographic features
            'Latitude', 'Longitude', 'DistanceFromCenter', 'NorthSide', 'WestSide',
            'LatBin', 'LonBin', 'LocationCluster',
            
            # Temporal features
            'Year', 'Month', 'DayOfYear', 'Week',
            'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos',
            'IsPeakSeason',
            
            # Species features
            'Species_encoded', 'IsHighRiskSpecies',
            
            # Weather features
            'Tmax', 'Tmin', 'Tavg', 'DewPoint', 'WetBulb', 'PrecipTotal',
            'StnPressure', 'SeaLevel', 'ResultSpeed', 'AvgSpeed',
            
            # Derived weather features
            'TempRange', 'TempAvg', 'RelativeHumidity', 'HeatStress',
            'HasPrecip', 'HeavyRain', 'IsCalm', 'IsWindy',
            
            # Spray features
            'Sprayed', 'DaysSinceSpray',
            
            # Interaction features
            'TempHumidity', 'SpeciesMonth', 'LocationSeason'
        ]
        
        # Note: NumMosquitos is excluded as it's not available in test data
        # This ensures consistent feature sets between training and test
        
        # Filter columns that actually exist in the dataframe
        available_features = [col for col in feature_cols if col in df.columns]
        
        if is_training:
            self.feature_names = available_features
            print(f"Selected {len(available_features)} features for training")
        
        return df[available_features]
    
    def scale_features(self, X_train, X_test=None):
        """Scale features using StandardScaler."""
        print("Scaling features...")
        
        # Fit and transform training data
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        if X_test is not None:
            # Transform test data
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def impute_missing_values(self, X_train, X_test=None):
        """Impute missing values using median strategy."""
        print("Imputing missing values...")
        
        # Fit and transform training data
        X_train_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        if X_test is not None:
            # Transform test data
            X_test_imputed = pd.DataFrame(
                self.imputer.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            return X_train_imputed, X_test_imputed
        
        return X_train_imputed
    
    def process_all_data(self):
        """Process all data and return train/test sets."""
        print("=" * 60)
        print("WEST NILE VIRUS DATA PREPROCESSING")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Clean weather data
        self.clean_weather_data()
        
        # Process training data
        train_processed = self.preprocess_dataset(self.train_df, is_training=True)
        
        # Process test data
        test_processed = self.preprocess_dataset(self.test_df, is_training=False)
        
        # Extract target variable
        y_train = train_processed['WnvPresent'].values
        
        # Select features
        X_train = self.select_features(train_processed, is_training=True)
        X_test = self.select_features(test_processed, is_training=False)
        
        # Handle NumMosquitos column - remove from train and don't add to test
        if 'NumMosquitos' in X_train.columns:
            print("Removing NumMosquitos from training data (not available in test data)")
            X_train = X_train.drop(columns=['NumMosquitos'])
            # Update feature names to exclude NumMosquitos
            if 'NumMosquitos' in self.feature_names:
                self.feature_names.remove('NumMosquitos')
        
        # Ensure test data has same columns as training data (excluding NumMosquitos)
        missing_cols = set(X_train.columns) - set(X_test.columns)
        if missing_cols:
            print(f"Warning: Adding missing columns to test data: {missing_cols}")
            for col in missing_cols:
                X_test[col] = 0  # Add missing columns with default value
        
        # Remove extra columns from test data
        extra_cols = set(X_test.columns) - set(X_train.columns)
        if extra_cols:
            print(f"Warning: Removing extra columns from test data: {extra_cols}")
            X_test = X_test.drop(columns=list(extra_cols))
        
        # Reorder columns to match training data
        X_test = X_test[X_train.columns]
        
        # Impute missing values
        X_train, X_test = self.impute_missing_values(X_train, X_test)
        
        # Scale features
        X_train, X_test = self.scale_features(X_train, X_test)
        
        print(f"\nFinal dataset shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}")
        
        print(f"\nClass distribution:")
        unique, counts = np.unique(y_train, return_counts=True)
        for val, count in zip(unique, counts):
            print(f"  Class {val}: {count:,} samples ({count/len(y_train)*100:.2f}%)")
        
        print(f"\nFeatures: {list(X_train.columns)}")
        
        return X_train, X_test, y_train
    
    def save_processed_data(self, X_train, X_test, y_train, output_dir='processed_data'):
        """Save processed data to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving processed data to {output_dir}/...")
        
        # Save as CSV
        X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
        X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
        pd.DataFrame({'WnvPresent': y_train}).to_csv(f'{output_dir}/y_train.csv', index=False)
        
        # Save feature names
        with open(f'{output_dir}/feature_names.txt', 'w') as f:
            f.write('\n'.join(self.feature_names))
        
        print("Data saved successfully!")


def main():
    """Main function to run data preprocessing."""
    try:
        # Initialize preprocessor
        preprocessor = WestNileDataPreprocessor()
        
        # Process all data
        X_train, X_test, y_train = preprocessor.process_all_data()
        
        # Save processed data
        preprocessor.save_processed_data(X_train, X_test, y_train)
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return X_train, X_test, y_train, preprocessor
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    X_train, X_test, y_train, preprocessor = main()