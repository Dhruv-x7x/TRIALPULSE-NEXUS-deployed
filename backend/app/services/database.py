"""
Database Service Bridge
=======================
Bridges FastAPI to the existing PostgreSQL data service.
"""

import sys
import os
import importlib
from typing import Any

# Add src to path for importing existing data service
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Keep a global instance
_service_instance: Any = None

def get_data_service():
    """Get singleton instance of PostgreSQL data service."""
    global _service_instance
    
    # Lazy import to avoid circular dependencies
    import src.database.pg_data_service as pg_mod
    
    # Always force a reload if we want to be sure about latest data/code in dev
    # or just keep as is for production. 
    # For this task, I'll make it more robust.
    if _service_instance is None or not hasattr(_service_instance, 'get_dqi_distribution'):
        importlib.reload(pg_mod)
        _service_instance = pg_mod.PostgreSQLDataService()
    
    return _service_instance


def clear_data_service_cache():
    """Clear lru_cache and internal service cache."""
    global _service_instance
    _service_instance = None
    
    import src.database.pg_data_service as pg_mod
    importlib.reload(pg_mod)
    
    return True


# Export commonly used functions by proxying to the service instance
def get_patients(limit=None, study_id=None, site_id=None):
    return get_data_service().get_patients(limit=limit, study_id=study_id, site_id=site_id)

def get_patient(patient_key: str):
    return get_data_service().get_patient(patient_key)

def search_patients(query: str, limit: int = 20):
    return get_data_service().search_patients(query, limit)

def get_sites():
    return get_data_service().get_sites()

def get_site_benchmarks(study_id=None):
    return get_data_service().get_site_benchmarks(study_id=study_id)

def get_studies(limit=None):
    return get_data_service().get_studies(limit=limit)

def get_issues(status=None, limit=2000, study_id=None, site_id=None):
    return get_data_service().get_issues(status=status, limit=limit, study_id=study_id, site_id=site_id)

def get_patient_issues():
    return get_data_service().get_patient_issues()

def get_queries(status=None):
    return get_data_service().get_queries(status)

def get_portfolio_summary(study_id=None):
    return get_data_service().get_portfolio_summary(study_id=study_id)

def get_patient_dqi(study_id=None):
    return get_data_service().get_patient_dqi(study_id=study_id)

def get_patient_clean_status(study_id=None):
    return get_data_service().get_patient_clean_status(study_id=study_id)

def get_patient_dblock_status(study_id=None):
    return get_data_service().get_patient_dblock_status(study_id=study_id)

def get_regional_metrics(study_id=None):
    return get_data_service().get_regional_metrics(study_id=study_id)

def get_ml_models():
    return get_data_service().get_ml_models()

def get_dqi_distribution(study_id=None):
    return get_data_service().get_dqi_distribution(study_id=study_id)

def get_pattern_alerts(study_id=None):
    return get_data_service().get_pattern_alerts(study_id=study_id)

def get_cascade_analysis(study_id=None):
    return get_data_service().get_cascade_analysis(study_id=study_id)

def health_check():
    return get_data_service().health_check()
