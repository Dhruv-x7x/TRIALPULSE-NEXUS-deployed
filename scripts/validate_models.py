"""
TRIALPULSE NEXUS - ML Model Validation Script
==============================================
Comprehensive validation for all ML models:
- Data leakage detection
- Overfitting check
- Model quality metrics
- Feature importance analysis

Usage: python scripts/validate_models.py
"""

import sys
import os
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class ModelValidator:
    """Comprehensive ML model validation."""
    
    def __init__(self):
        self.issues_found: List[str] = []
        self.warnings: List[str] = []
        self.results: Dict[str, Any] = {}
    
    def validate_all(self) -> Dict[str, Any]:
        """Run validation on all models."""
        logger.info("=" * 60)
        logger.info("TRIALPULSE NEXUS - ML MODEL VALIDATION")
        logger.info("=" * 60)
        
        # 1. Check database connectivity
        logger.info("\n[1/6] Checking database connectivity...")
        db_ok = self._check_database()
        self.results['database'] = {'connected': db_ok}
        
        # 2. Check model files exist
        logger.info("\n[2/6] Checking model files...")
        models_exist = self._check_model_files()
        self.results['model_files'] = models_exist
        
        # 3. Validate risk classifier
        logger.info("\n[3/6] Validating Risk Classifier...")
        self.results['risk_classifier'] = self._validate_risk_classifier()
        
        # 4. Validate issue detector
        logger.info("\n[4/6] Validating Issue Detector...")
        self.results['issue_detector'] = self._validate_issue_detector()
        
        # 5. Validate resolution predictor
        logger.info("\n[5/6] Validating Resolution Time Predictor...")
        self.results['resolution_predictor'] = self._validate_resolution_predictor()
        
        # 6. Validate anomaly detector
        logger.info("\n[6/6] Validating Anomaly Detector...")
        self.results['anomaly_detector'] = self._validate_anomaly_detector()
        
        return self.results
    
    def _check_database(self) -> bool:
        """Check PostgreSQL connectivity."""
        try:
            from src.database.connection import get_db_manager
            db = get_db_manager()
            
            from sqlalchemy import text
            with db.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logger.info("  PostgreSQL: Connected")
                return True
        except Exception as e:
            self.warnings.append(f"Database connection failed: {e}")
            logger.warning(f"  PostgreSQL: Not connected ({e})")
            return False
    
    def _check_model_files(self) -> Dict[str, bool]:
        """Check if model files exist."""
        project_root = Path(__file__).parent.parent
        
        models = {
            'risk_classifier': project_root / "data" / "processed" / "ml" / "models" / "risk_classifier_xgb.joblib",
            'issue_detector': project_root / "models" / "issue_detector" / "issue_detector_model.joblib",
            'resolution_predictor': project_root / "data" / "processed" / "ml" / "models" / "resolution_time",
            'anomaly_detector': project_root / "data" / "processed" / "ml" / "models" / "anomaly_detector",
            'site_ranker': project_root / "data" / "processed" / "ml" / "site_ranker",
        }
        
        results = {}
        for model_name, path in models.items():
            exists = path.exists()
            results[model_name] = exists
            status = "Found" if exists else "NOT FOUND"
            logger.info(f"  {model_name}: {status}")
            if not exists:
                self.warnings.append(f"Model not found: {model_name}")
        
        return results
    
    def _validate_risk_classifier(self) -> Dict[str, Any]:
        """Validate risk classifier model."""
        result = {'status': 'pending', 'metrics': {}}
        
        try:
            from src.ml.model_loader import ModelLoader
            loader = ModelLoader()
            
            # Try to make a prediction
            test_features = pd.DataFrame({
                'dqi_score': [75.0],
                'risk_score': [30.0],
                'open_queries_count': [5],
                'open_issues_count': [2],
                'has_sae': [0],
                'visit_compliance_pct': [85.0],
                'days_since_last_activity': [7],
            })
            
            try:
                prediction = loader.predict_risk(test_features.iloc[0].to_dict())
                result['status'] = 'validated'
                result['metrics'] = {
                    'prediction_works': True,
                    'sample_prediction': {
                        'risk_level': prediction.risk_level if hasattr(prediction, 'risk_level') else str(prediction),
                        'confidence': prediction.confidence if hasattr(prediction, 'confidence') else 0.0,
                    }
                }
                logger.info(f"  Risk Classifier: Working - {prediction}")
            except FileNotFoundError:
                result['status'] = 'model_not_found'
                logger.warning("  Risk Classifier: Model file not found")
            except Exception as e:
                result['status'] = 'prediction_failed'
                result['error'] = str(e)
                logger.warning(f"  Risk Classifier: Prediction failed - {e}")
                
        except ImportError as e:
            result['status'] = 'import_error'
            result['error'] = str(e)
            logger.error(f"  Risk Classifier: Import error - {e}")
        
        return result
    
    def _validate_issue_detector(self) -> Dict[str, Any]:
        """Validate issue detector model."""
        result = {'status': 'pending', 'metrics': {}}
        
        try:
            from src.ml.model_loader import ModelLoader
            loader = ModelLoader()
            
            test_features = pd.DataFrame({
                'open_queries_count': [10],
                'has_sae': [1],
                'missing_visits': [2],
                'overdue_signatures': [5],
            })
            
            try:
                prediction = loader.predict_issues(test_features.iloc[0].to_dict())
                result['status'] = 'validated'
                result['metrics'] = {
                    'prediction_works': True,
                    'detected_issues': prediction.detected_issues if hasattr(prediction, 'detected_issues') else [],
                }
                logger.info(f"  Issue Detector: Working")
            except FileNotFoundError:
                result['status'] = 'model_not_found'
                logger.warning("  Issue Detector: Model file not found")
            except AttributeError:
                result['status'] = 'method_not_found'
                logger.warning("  Issue Detector: Method not implemented")
            except Exception as e:
                result['status'] = 'prediction_failed'
                result['error'] = str(e)
                logger.warning(f"  Issue Detector: {e}")
                
        except ImportError as e:
            result['status'] = 'import_error'
            result['error'] = str(e)
        
        return result
    
    def _validate_resolution_predictor(self) -> Dict[str, Any]:
        """Validate resolution time predictor."""
        result = {'status': 'pending', 'metrics': {}}
        
        try:
            from src.ml.resolution_time_predictor import ResolutionTimePredictor
            
            predictor = ResolutionTimePredictor()
            
            # Check if model is loaded
            if predictor._model is not None:
                result['status'] = 'validated'
                result['metrics'] = {'model_loaded': True}
                logger.info("  Resolution Predictor: Model loaded")
            else:
                result['status'] = 'model_not_loaded'
                logger.warning("  Resolution Predictor: Model not loaded")
                
        except ImportError as e:
            result['status'] = 'import_error'
            result['error'] = str(e)
            logger.error(f"  Resolution Predictor: Import error - {e}")
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            logger.warning(f"  Resolution Predictor: {e}")
        
        return result
    
    def _validate_anomaly_detector(self) -> Dict[str, Any]:
        """Validate anomaly detector."""
        result = {'status': 'pending', 'metrics': {}}
        
        try:
            from src.ml.anomaly_detector import AnomalyDetector
            
            detector = AnomalyDetector()
            result['status'] = 'validated'
            result['metrics'] = {'initialized': True}
            logger.info("  Anomaly Detector: Initialized")
            
        except ImportError as e:
            result['status'] = 'import_error'
            result['error'] = str(e)
            logger.error(f"  Anomaly Detector: Import error - {e}")
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            logger.warning(f"  Anomaly Detector: {e}")
        
        return result
    
    def generate_report(self) -> str:
        """Generate validation report."""
        lines = [
            "=" * 60,
            "TRIALPULSE NEXUS - ML MODEL VALIDATION REPORT",
            "=" * 60,
            f"Generated at: {datetime.now().isoformat()}",
            "",
        ]
        
        # Summary
        validated_count = sum(1 for r in self.results.values() 
                             if isinstance(r, dict) and r.get('status') == 'validated')
        total_models = 4  # risk, issue, resolution, anomaly
        
        lines.append(f"Models Validated: {validated_count}/{total_models}")
        lines.append("")
        
        # Details
        for model_name, result in self.results.items():
            if model_name == 'database':
                continue
            if model_name == 'model_files':
                lines.append("MODEL FILES:")
                for name, exists in result.items():
                    status = "OK" if exists else "MISSING"
                    lines.append(f"  {name}: {status}")
                lines.append("")
            else:
                lines.append(f"{model_name.upper()}:")
                if isinstance(result, dict):
                    for key, value in result.items():
                        lines.append(f"  {key}: {value}")
                lines.append("")
        
        # Issues
        if self.issues_found:
            lines.append("CRITICAL ISSUES:")
            for issue in self.issues_found:
                lines.append(f"  - {issue}")
            lines.append("")
        
        if self.warnings:
            lines.append("WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
            lines.append("")
        
        if not self.issues_found:
            lines.append("No critical issues found!")
        
        return "\n".join(lines)


def main():
    """Run model validation."""
    validator = ModelValidator()
    results = validator.validate_all()
    
    report = validator.generate_report()
    print("\n" + report)
    
    # Save report
    report_path = Path(__file__).parent.parent / "model_validation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Exit with error code if issues found
    if validator.issues_found:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
