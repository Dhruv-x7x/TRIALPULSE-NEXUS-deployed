"""
TRIALPULSE NEXUS 10X - Cleaning Runner
=======================================
Dedicated entry point for data cleaning.
"""

import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cleaning import DataCleaningEngine


def run_cleaning():
    """Run the data cleaning pipeline."""
    engine = DataCleaningEngine()
    manifest = engine.run()
    
    if manifest.status == "completed" and not manifest.errors:
        print("\n" + "=" * 70)
        print("✅ CLEANING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        return 0
    else:
        print(f"\n⚠️ CLEANING COMPLETED WITH {len(manifest.errors)} ERRORS")
        return 1


if __name__ == "__main__":
    sys.exit(run_cleaning())