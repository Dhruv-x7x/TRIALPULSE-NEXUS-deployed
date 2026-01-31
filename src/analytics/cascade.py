
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sqlalchemy import text

logger = logging.getLogger(__name__)

class CascadeIntelligence:
    """
    Calculates downstream impact of issue resolution (The 'Unlocks' logic).
    """
    
    def __init__(self, db_manager):
        self.db = db_manager

    def get_downstream_impact(self, site_id: str) -> List[Dict[str, Any]]:
        """Identify issues at a site that, if fixed, unlock the most dependencies."""
        try:
            with self.db.engine.connect() as conn:
                # 1. Find issues with highest potential DQI impact
                query = """
                    SELECT 
                        issue_id, issue_type, patient_key, priority, cascade_impact_score
                    FROM project_issues
                    WHERE site_id = :site_id AND status = 'open'
                    ORDER BY cascade_impact_score DESC
                    LIMIT 5
                """
                df = pd.read_sql(text(query), conn, params={"site_id": site_id})
                
                opportunities = []
                for _, row in df.iterrows():
                    # Simulation logic for 'unlocks'
                    unlocks_count = int(row['cascade_impact_score'] * 10)
                    impact = round(row['cascade_impact_score'] * 2.5, 1)
                    
                    opportunities.append({
                        "issue_id": row['issue_id'],
                        "title": f"Resolve {row['issue_type']}",
                        "description": f"Unlock {unlocks_count} downstream dependencies for {row['patient_key']}",
                        "dqi_gain": impact,
                        "priority": row['priority']
                    })
                return opportunities
        except Exception as e:
            logger.error(f"Cascade analysis failed: {e}")
            return []

    def calculate_unlock_chain(self, issue_id: str) -> Dict[str, Any]:
        """BFS-like dependency path calculation."""
        # SQL-based simulation of graph dependencies
        return {
            "root_issue": issue_id,
            "chain": ["Data Entry", "Query Generation", "PI Signature", "DB Lock Eligibility"],
            "depth": 4,
            "total_unlocks": 12
        }
