import sys
import time
import logging
import pandas as pd
import numpy as np
import uuid
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ml.governance.drift_detector import get_drift_detector, PerformanceDriftResult, DriftSeverity
from src.database.pg_data_service import get_pg_data_service
from src.streaming.streaming_service import get_streaming_service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ML-Watchdog")

def run_drift_check():
    """
    Perform a drift analysis on the current patient population.
    """
    logger.info("Starting Model Drift Analysis...")
    
    try:
        service = get_pg_data_service()
        detector = get_drift_detector()
        streaming = get_streaming_service()
        
        # 1. Fetch current data window
        df = service.get_patients(limit=2000)
        
        if df.empty:
            logger.warning("No data found for drift analysis.")
            return

        # 2. Generate report for the primary models
        target_models = [
            ("risk_classifier", "v9.0"),
            ("issue_detector", "v3.0"),
            ("anomaly_detector", "v2.0"),
            ("resolution_predictor", "v3.0"),
            ("site_ranker", "v2.0")
        ]
        
        # Ensure detectors are initialized with baselines from DB
        detector._load_baselines()
        
        for model_name, version in target_models:
            logger.info(f"Analyzing {model_name} v{version}...")
            
            # Simple feature selection
            features = ['age_at_enrollment', 'dqi_score', 'risk_score', 'open_issues_count']
            features = [f for f in features if f in df.columns]
            
            # Simulate real drift by shifting the mean of some features
            analysis_df = df.copy()
            if not analysis_df.empty:
                # Model-specific drift simulation
                if model_name == "risk_classifier":
                    # Shift risk scores up significantly
                    analysis_df['risk_score'] = analysis_df['risk_score'] * np.random.uniform(1.1, 1.4, len(analysis_df))
                elif model_name == "site_ranker":
                    # Shift DQI scores down
                    analysis_df['dqi_score'] = analysis_df['dqi_score'] * np.random.uniform(0.85, 0.95, len(analysis_df))
                else:
                    # Subtle jitter for others
                    analysis_df['dqi_score'] = analysis_df['dqi_score'] * np.random.uniform(0.98, 1.02, len(analysis_df))
            
            report = detector.generate_drift_report(model_name, version, analysis_df, features=features)
            
            # Add simulated performance metrics
            baseline_acc = 96.0 - (np.random.random() * 2)
            current_acc = baseline_acc - (np.random.random() * 5) # More realistic drop for some
            
            perf_result = PerformanceDriftResult(
                model_name=model_name,
                metric_name="accuracy",
                baseline_value=float(baseline_acc),
                current_value=float(current_acc),
                absolute_change=float(current_acc - baseline_acc),
                relative_change=float(((current_acc - baseline_acc) / baseline_acc) * 100),
                severity=DriftSeverity.MEDIUM if (baseline_acc - current_acc) > 3 else DriftSeverity.NONE,
                window_start=datetime.now(timezone.utc) - timedelta(days=7),
                window_end=datetime.now(timezone.utc),
                sample_size=len(analysis_df)
            )
            report.performance_drifts.append(perf_result)
            
            # 3. Publish to Kafka for UI real-time
            streaming.publish_drift_detected(model_name, report.to_dict())
            
            logger.info(f"Report saved for {model_name} (PSI={report.overall_psi:.4f}, Accuracy={current_acc:.1f}%)")
            
    except Exception as e:
        logger.error(f"Drift monitoring failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

def monitor_loop(interval_seconds=60):
    """
    Run the drift check in a loop for demo.
    """
    logger.info(f"ML Watchdog active. Checking for drift every {interval_seconds} seconds.")
    try:
        while True:
            run_drift_check()
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user.")

if __name__ == "__main__":
    if "--loop" in sys.argv:
        monitor_loop()
    else:
        run_drift_check()
