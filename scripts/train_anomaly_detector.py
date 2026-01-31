#!/usr/bin/env python
"""
Train Anomaly Detector for Site Benchmarks
Uses Isolation Forest to detect anomalous sites based on metrics.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.database.pg_data_service import get_pg_data_service


def main():
    print("Training Anomaly Detector for Site Benchmarks...")
    
    # Load site benchmarks
    service = get_pg_data_service()
    df = service.get_site_benchmarks()
    
    print(f"Loaded {len(df)} sites")
    
    # Select features for anomaly detection
    features = ['query_rate', 'visit_compliance', 'dqi_score', 'enrollment_rate']
    
    X = df[features].fillna(0).values
    print(f"Training on {len(X)} samples with {len(features)} features")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # Expect ~5% anomalies
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)
    
    # Create output directory
    output_dir = PROJECT_ROOT / "data" / "processed" / "ml" / "models" / "anomaly_detector"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and scaler
    model_path = output_dir / "anomaly_model_latest.pkl"
    scaler_path = output_dir / "anomaly_scaler_latest.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Saved model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")
    
    # Quick test
    predictions = model.predict(X_scaled)
    n_anomalies = (predictions == -1).sum()
    print(f"Test: {n_anomalies} anomalies detected out of {len(predictions)} sites ({100*n_anomalies/len(predictions):.1f}%)")
    
    print("Done!")


if __name__ == "__main__":
    main()
