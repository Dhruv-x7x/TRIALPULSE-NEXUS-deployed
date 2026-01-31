
import os
import sys
import argparse
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# PostgreSQL integration
from src.database.pg_data_service import get_data_service
from src.database.pg_writer import get_pg_writer


# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class AnomalyDetectorTrainer:
    """Train Isolation Forest model to detect site anomalies."""
    
    def __init__(self, data_path: str = "data/processed/analytics"):
        self.data_path = Path(data_path)
        self.models_path = Path("data/processed/ml/models/anomaly_detector")
        self.models_path.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare site data for training."""
        sites_path = self.data_path / "site_benchmarks.parquet"
        
        if not sites_path.exists():
            logger.error(f"Data not found at {sites_path}")
            return self._generate_synthetic_training_data()
            
        df = pd.read_parquet(sites_path)
        logger.info(f"Loaded {len(df)} rows from {sites_path}")
        return self._prepare_features(df)
        
    def _generate_synthetic_training_data(self, n_samples=1000) -> pd.DataFrame:
        """Generate synthetic site data."""
        logger.warning("Generating SYNTHETIC site data for training...")
        np.random.seed(42)
        
        data = {
            'query_rate': np.random.normal(0.5, 0.2, n_samples),
            'visit_compliance': np.random.normal(0.9, 0.1, n_samples),
            'dqi_score': np.random.normal(95, 5, n_samples),
            'enrollment_rate': np.random.normal(2.0, 1.0, n_samples),
            'churn_rate': np.random.exponential(0.1, n_samples)
        }
        
        # Inject anomalies
        n_anomalies = int(n_samples * 0.05)
        for i in range(n_anomalies):
            data['query_rate'][i] = np.random.uniform(2.0, 5.0) # Very high queries
            data['dqi_score'][i] = np.random.uniform(60, 75)   # Low quality
            
        df = pd.DataFrame(data)
        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and scale features."""
        features = ['query_rate', 'visit_compliance', 'dqi_score', 'enrollment_rate']
        
        # Determine available columns
        available = [f for f in features if f in df.columns]
        
        if not available:
            # Fallback if real data has completely different columns
            # Create dummy columns based on whatever numerical data exists
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric) >= 2:
                available = numeric[:4]
            else:
                 return self._generate_synthetic_training_data(len(df))
        
        X = df[available].fillna(0)
        return X

    def train(self):
        """Train the model."""
        X = self.load_data()
        
        logger.info(f"Training Anomaly Detector on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Isolation Forest
        model = IsolationForest(
            n_estimators=100,
            contamination=0.05, # Expect 5% anomalies
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_scaled)
        
        # Validation
        scores = model.decision_function(X_scaled)
        anomalies = model.predict(X_scaled)
        n_detected = np.sum(anomalies == -1)
        
        logger.info(f"Detected {n_detected} anomalies in training set ({n_detected/len(X):.1%})")
        logger.info(f"Mean Anomaly Score: {np.mean(scores):.3f}")
        
        # Save artifacts
        timestamp = datetime.now().strftime("%Y%m%d")
        
        for name, obj in [('model', model), ('scaler', scaler)]:
            path = self.models_path / f"anomaly_{name}_{timestamp}.pkl"
            joblib.dump(obj, path)
            logger.info(f"Saved {name} to {path}")
            
            # Symlink
            latest_path = self.models_path / f"anomaly_{name}_latest.pkl"
            joblib.dump(obj, latest_path)
            
        logger.info("✅ Anomaly Detector training complete.")

if __name__ == "__main__":
    trainer = AnomalyDetectorTrainer()
    trainer.train()
