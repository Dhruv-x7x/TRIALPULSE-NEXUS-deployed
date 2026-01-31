
import os
import sys
import argparse
import logging
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

# PostgreSQL integration
from src.database.pg_data_service import get_data_service
from src.database.pg_writer import get_pg_writer


# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class ResolutionTimeTrainer:
    """Train XGBoost model to predict issue resolution time."""
    
    def __init__(self, data_path: str = "data/processed/analytics"):
        self.data_path = Path(data_path)
        self.models_path = Path("data/processed/ml/models/resolution_time")
        self.models_path.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data for training."""
        issues_path = self.data_path / "patient_issues.parquet"
        
        if not issues_path.exists():
            logger.error(f"Data not found at {issues_path}")
            # Generate synthetic data for training if missing
            return self._generate_synthetic_training_data()
            
        df = pd.read_parquet(issues_path)
        logger.info(f"Loaded {len(df)} rows from {issues_path}")
        
        # Check if target exists, if not, generate it based on timestamps or synthetic logic
        if 'days_to_resolution' not in df.columns:
            logger.warning("'days_to_resolution' not found in real data. Generating synthetic target based on properties...")
            df = self._add_synthetic_target(df)
            
        return self._prepare_features(df)

    def _add_synthetic_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add synthetic days_to_resolution column for training purposes."""
        np.random.seed(42)
        n_samples = len(df)
        
        # Prepare required columns for feature generation
        if 'site_performance_score' not in df.columns:
            df['site_performance_score'] = np.random.normal(85, 10, n_samples)
        if 'workload_index' not in df.columns:
            df['workload_index'] = np.random.normal(0.5, 0.2, n_samples)
        if 'issue_type' not in df.columns and 'issue_text' in df.columns:
             df['issue_type'] = df['issue_text'].apply(lambda x: 'unknown' if not x else str(x)[:20])

        base_days_map = {
            'missing_visits': 15, 'sdv_incomplete': 5, 'open_queries': 8,
            'signature_gaps': 3, 'meddra_uncoded': 2, 'lab_issues': 10
        }
        
        resolutions = []
        for idx, row in df.iterrows():
            itype = row.get('issue_type', 'unknown')
            # Extract first match from map keys in issue type string
            base = 5
            for key, val in base_days_map.items():
                if key in str(itype):
                    base = val
                    break
            
            # Apply modifiers
            perf = row.get('site_performance_score', 85)
            load = row.get('workload_index', 0.5)
            
            days = base * (120 / max(1, perf)) * (1 + load)
            days = max(0.1, np.random.normal(days, days * 0.2))
            resolutions.append(days)
            
        df['days_to_resolution'] = resolutions
        return df
        
    def _generate_synthetic_training_data(self, n_samples=5000) -> pd.DataFrame:
        """Generate synthetic data for training."""
        logger.warning("Generating SYNTHETIC training data for bootstrapping...")
        np.random.seed(42)
        
        data = {
            'issue_type': np.random.choice([
                'missing_visits', 'sdv_incomplete', 'open_queries', 
                'signature_gaps', 'meddra_uncoded', 'lab_issues'
            ], n_samples),
            'site_performance_score': np.random.normal(85, 10, n_samples),
            'workload_index': np.random.normal(0.5, 0.2, n_samples),
            'issue_priority': np.random.choice(['High', 'Medium', 'Low'], n_samples),
            'is_auto_generated': np.random.choice([0, 1], n_samples),
            'days_to_resolution': [] # Target
        }
        
        # Logic for target generation
        df = pd.DataFrame(data)
        
        # Base days by type
        base_days = {
            'missing_visits': 15, 'sdv_incomplete': 5, 'open_queries': 8,
            'signature_gaps': 3, 'meddra_uncoded': 2, 'lab_issues': 10
        }
        
        resolution_times = []
        for _, row in df.iterrows():
            days = base_days.get(row['issue_type'], 5)
            
            # Modifiers
            days *= (1.5 if row['issue_priority'] == 'Low' else 0.8) # Lower priority takes longer
            days *= (120 / max(1, row['site_performance_score'])) # Poor site = longer
            days *= (1 + row['workload_index']) # High workload = longer
            
            # Noise
            days = max(0.1, np.random.normal(days, days * 0.2))
            resolution_times.append(days)
            
        df['days_to_resolution'] = resolution_times
        return self._prepare_features(df)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for training."""
        # Clean
        df = df.copy()
        
        # Categorical encoding
        if 'issue_type' in df.columns:
            df['issue_type_cat'] = df['issue_type'].astype('category').cat.codes
            
        if 'issue_priority' in df.columns:
            priority_map = {'Critical': 3, 'High': 2, 'Medium': 1, 'Low': 0}
            df['priority_score'] = df['issue_priority'].map(priority_map).fillna(1)
            
        # Ensure numerical columns exist
        for col in ['site_performance_score', 'workload_index']:
            if col not in df.columns:
                df[col] = np.random.normal(0.5, 0.1, len(df)) # Fallback if missing
                
        return df

    def train(self):
        """Train the model."""
        df = self.load_data()
        
        # DEBUG: Print columns
        logger.info(f"Columns available: {df.columns.tolist()}")
        
        features = ['priority_score', 'site_performance_score', 'workload_index']
        # Add issue_type_cat only if it exists
        if 'issue_type_cat' in df.columns:
            features.insert(0, 'issue_type_cat')
            
        target = 'days_to_resolution'
        
        # Verify all features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return
            
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train quantile regression models (Upper, Median, Lower)
        quantiles = [0.1, 0.5, 0.9]
        models = {}
        
        logger.info("Training Quantile Regression Models...")
        
        for q in quantiles:
            model = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=q,
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            models[str(q)] = model
            
            # Validate
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            logger.info(f"Quantile {q}: MAE = {mae:.2f} days")
            
        # Save models
        timestamp = datetime.now().strftime("%Y%m%d")
        
        for q, model in models.items():
            path = self.models_path / f"resolution_time_q{q}_{timestamp}.pkl"
            joblib.dump(model, path)
            logger.info(f"Saved model to {path}")
            
            # Create symlink/copy to 'latest'
            latest_path = self.models_path / f"resolution_time_q{q}_latest.pkl"
            joblib.dump(model, latest_path)
            
        logger.info("✅ Resolution Time Predictor training complete.")

if __name__ == "__main__":
    trainer = ResolutionTimeTrainer()
    trainer.train()
