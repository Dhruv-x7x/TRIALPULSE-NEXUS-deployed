"""
TRIALPULSE NEXUS 10X - Site Risk Ranker Runner
================================================
Training script for the Site Risk Ranker model.

Usage:
    python src/run_site_risk_ranker.py

Author: TrialPulse Team
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from ml.site_risk_ranker import SiteRiskRankerRunner, main


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 TRIALPULSE NEXUS 10X - SITE RISK RANKER")
    print("   Learning-to-Rank for CRA Site Prioritization")
    print("=" * 70)
    
    runner = main()
    
    if runner is not None and runner.results.get('is_defensible'):
        print("\n✅ SUCCESS - Model is production-ready")
        print(f"   Output: {runner.output_dir}")
    elif runner is not None:
        print("\n⚠️ WARNING - Model has red flags, review required")
    else:
        print("\n❌ FAILED - Could not train model")
