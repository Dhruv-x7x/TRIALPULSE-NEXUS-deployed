"""
TRIALPULSE NEXUS 10X - UPR Builder Runner
==========================================
Dedicated entry point for UPR building.
"""

import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.upr_builder import UPRBuilder


def run_upr_builder():
    """Run the UPR building pipeline."""
    builder = UPRBuilder()
    upr = builder.build()
    
    if builder.manifest.status == "completed" and not builder.manifest.errors:
        print("\n" + "=" * 70)
        print("[OK] UPR BUILD COMPLETED SUCCESSFULLY!")
        print(f"   Patients: {len(upr):,}")
        print(f"   Columns: {len(upr.columns)}")
        print("=" * 70)

        
        # Refresh analytical views
        try:
            from sqlalchemy import text
            from src.database.connection import get_db_manager
            db = get_db_manager()
            with db.engine.begin() as conn:
                # Refresh patient_issues view
                from scripts.fix_issues import patient_issues_view_sql
                conn.execute(text(patient_issues_view_sql))
                print("\n--- Refreshed analytical views ---")
        except Exception as e:
            print(f"[WARN] Failed to refresh analytical views: {e}")
            
        # Seed knowledge graph

        try:
            from src.knowledge.seed_graph import seed_graph
            print("\n--- Seeding Knowledge Graph ---")
            seed_graph()
        except Exception as e:
            print(f"[WARN] Knowledge Graph seeding skipped/failed: {e}")
            
        return 0
    else:
        print(f"\n[WARN] UPR BUILD COMPLETED WITH ISSUES")
        return 1



if __name__ == "__main__":
    sys.exit(run_upr_builder())