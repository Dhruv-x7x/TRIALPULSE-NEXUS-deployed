"""
Runner script for Resolution Genome Engine v1.2
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import DATA_PROCESSED
from src.ml.resolution_genome import run_resolution_genome, ResolutionGenomeConfig

# PostgreSQL data access
try:
    from src.database.pg_data_service import get_data_service
    from src.database.pg_writer import get_pg_writer
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


if __name__ == '__main__':
    issues_path = DATA_PROCESSED / 'analytics' / 'patient_issues.parquet'
    output_dir = DATA_PROCESSED / 'analytics'
    
    config = ResolutionGenomeConfig(
        embedding_model='all-MiniLM-L6-v2',
        top_k_matches=3,
    )
    
    summary = run_resolution_genome(
        issues_path=issues_path,
        output_dir=output_dir,
        config=config
    )
    
    print("\n✅ Resolution Genome complete!")
    print(f"   Templates: {summary['genome_statistics']['total_templates']}")
    print(f"   Patients with issues: {summary['assignment_results']['patients_with_issues']:,}")
    print(f"   Total recommendations: {summary['assignment_results']['total_recommendations']:,}")
    print(f"   Roles assigned: {summary['assignment_results']['roles_assigned']}")