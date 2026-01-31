"""
TRIALPULSE NEXUS 10X - Ingestion Runner
========================================
Dedicated entry point to avoid module import conflicts.
"""

import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingestion import DataIngestionEngine


def run_ingestion():
    """Run the data ingestion pipeline."""
    engine = DataIngestionEngine()
    manifest = engine.run()
    
    if manifest.files_failed == 0 and len(manifest.errors) == 0:
        print("\n" + "=" * 70)
        print("✅ INGESTION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        return 0
    else:
        total_errors = manifest.files_failed + len(manifest.errors)
        print(f"\n⚠️ INGESTION COMPLETED WITH {total_errors} ISSUES")
        return 1


if __name__ == "__main__":
    sys.exit(run_ingestion())