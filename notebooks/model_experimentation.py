"""
ChipFabAI Model Experimentation Script for Vertex AI Workbench
This script can be run in Vertex AI Workbench notebooks for ML experimentation
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from google.cloud import aiplatform
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Initialize Vertex AI
PROJECT_ID = os.getenv("GCP_PROJECT", "chipfab-ai")
REGION = os.getenv("REGION", "europe-west4")

print(f"Vertex AI Model Experimentation for ChipFabAI")
print(f"Project: {PROJECT_ID}, Region: {REGION}")

# Initialize Vertex AI
try:
    aiplatform.init(project=PROJECT_ID, location=REGION)
    print("✓ Vertex AI initialized")
except Exception as e:
    print(f"Warning: Could not initialize Vertex AI: {e}")

# Load training data from Cloud Storage
storage_client = storage.Client(project=PROJECT_ID)
bucket_name = "chipfabai-training-data"

try:
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob("sample_data.csv")
    if blob.exists():
        blob.download_to_filename("/tmp/sample_data.csv")
        df = pd.read_csv("/tmp/sample_data.csv")
        print(f"✓ Loaded {len(df)} samples from Cloud Storage")
    else:
        print("No training data found - using synthetic data for experimentation")
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'temperature': np.random.uniform(150, 250, n_samples),
            'pressure': np.random.uniform(0.5, 2.5, n_samples),
            'etch_time': np.random.uniform(30, 120, n_samples),
            'gas_flow': np.random.uniform(50, 200, n_samples),
            'chamber_pressure': np.random.uniform(1, 10, n_samples),
            'yield': np.random.uniform(70, 99, n_samples)
        })
        print(f"✓ Generated {len(df)} synthetic samples")
except Exception as e:
    print(f"Error loading data: {e}")
    df = None

if df is not None:
    # Feature Engineering
    def engineer_features(df):
        """Create derived features for experimentation"""
        df_features = df.copy()
        df_features['temp_pressure_ratio'] = df_features['temperature'] / (df_features['pressure'] + 1e-6)
        df_features['process_efficiency'] = df_features['yield'] / df_features['etch_time']
        df_features['normalized_gas_flow'] = df_features['gas_flow'] / df_features['chamber_pressure']
        df_features['temp_squared'] = df_features['temperature'] ** 2
        return df_features
    
    df_features = engineer_features(df)
    print(f"✓ Feature engineering complete - {len(df_features.columns)} features")
    
    # Model Training
    feature_cols = ['temperature', 'pressure', 'etch_time', 'gas_flow', 'chamber_pressure',
                    'temp_pressure_ratio', 'process_efficiency', 'normalized_gas_flow', 'temp_squared']
    X = df_features[feature_cols]
    y = df_features['yield']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MAE': mae, 'R2': r2}
        print(f"✓ {name}: MAE={mae:.2f}, R2={r2:.3f}")
    
    print("\n✓ Model experimentation complete!")
    print(f"Best model: {min(results, key=lambda x: results[x]['MAE'])}")

