
import sys
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database.connection import get_db_manager
from src.knowledge.neo4j_graph import get_graph_service

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_graph():
    logger.info("Initializing Graph Seeder...")
    
    # 1. Connect to PostgreSQL
    db = get_db_manager()
    if not db.health_check():
        logger.error("PostgreSQL not available!")
        return

    # 2. Connect to Neo4j
    graph = get_graph_service()
    if not graph.connect():
        logger.error("Neo4j not available!")
        # If connect returns False or we are in mock mode (which returns True), check:
        if graph.uses_mock:
             logger.warning("Using Mock Graph Service - Seeding will be in-memory only.")

    logger.info("Fetching UPR Data (Entities)...")
    with db.engine.connect() as conn:
        # Fetch detailed UPR
        upr_df = pd.read_sql("SELECT * FROM unified_patient_record", conn)
        logger.info(f"Loaded {len(upr_df)} patients for graph.")

        # Build Entity Graph (Study -> Site -> Patient)
        graph.build_entity_graph(upr_df)
        
        # Fetch Issues
        # We need a table with issues. If `project_issues` exists, use it.
        # Otherwise, infer from UPR columns.
        try:
            issues_df = pd.read_sql("SELECT * FROM project_issues", conn)
            logger.info(f"Loaded {len(issues_df)} issues from project_issues.")
            
            # Map columns to expected format
            # ensure: issue_id, patient_key, type, priority
            if 'issue_id' not in issues_df.columns:
                issues_df['issue_id'] = [f"ISS-{i}" for i in range(len(issues_df))]
            
            if 'category' in issues_df.columns and 'type' not in issues_df.columns:
                issues_df['type'] = issues_df['category']
                
            if 'type' not in issues_df.columns:
                 issues_df['type'] = 'General Issue'
                 
            # Add batch
            graph.add_issues_batch(issues_df)
            
        except Exception as e:
            logger.warning(f"Could not load project_issues: {e}")
            logger.info("Inferring issues from UPR...")
            
        # Fetch Visits
        try:
            visits_df = pd.read_sql("SELECT * FROM visits", conn)
            logger.info(f"Loaded {len(visits_df)} visits.")
            graph.add_visit_nodes(visits_df)
        except Exception as e:
            logger.warning(f"Could not load visits: {e}")
            
        # Fetch Lab Results (only abnormal ones for efficiency)
        try:
            labs_df = pd.read_sql("SELECT * FROM lab_results WHERE is_abnormal = true", conn)
            logger.info(f"Loaded {len(labs_df)} abnormal lab results.")
            graph.add_lab_nodes(labs_df)
        except Exception as e:
            logger.warning(f"Could not load lab_results: {e}")
            
            # Synthetic issue generation from UPR

            inferred_issues = []
            for idx, row in upr_df.iterrows():
                # Missing Visits
                if row.get('visit_missing_visit_count', 0) > 0:
                    inferred_issues.append({
                        'issue_id': f"ISS-MV-{row['patient_key']}",
                        'patient_key': row['patient_key'],
                        'type': 'Missing Visit',
                        'priority': 'Medium'
                    })
                
                # Queries
                if row.get('total_queries', 0) > 0:
                     inferred_issues.append({
                        'issue_id': f"ISS-Q-{row['patient_key']}",
                        'patient_key': row['patient_key'],
                        'type': 'Open Query',
                        'priority': 'High'
                    })
            
            if inferred_issues:
                inf_df = pd.DataFrame(inferred_issues)
                logger.info(f"Inferred {len(inf_df)} issues.")
                graph.add_issues_batch(inf_df)

    logger.info("Graph Seeding Complete.")
    
    # Validation
    stats = graph.get_graph_stats()
    logger.info(f"Graph Stats: {stats}")

if __name__ == "__main__":
    seed_graph()
