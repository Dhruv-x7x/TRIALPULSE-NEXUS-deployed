
import os
import sys
import pandas as pd
import logging

# Add project root to path so 'src' is recognized as a package
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.database.pg_data_service import PostgreSQLDataService
from src.knowledge.neo4j_graph import get_graph_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_graph_from_postgres():
    """Extract data from PostgreSQL and push to Neo4j."""
    logger.info("Starting Neo4j Cascade Seeding from PostgreSQL...")
    
    # 1. Initialize Services
    data_service = PostgreSQLDataService()
    graph_service = get_graph_service()
    
    try:
        if graph_service.uses_mock:
            logger.error("Neo4j is NOT connected. Seeding cannot proceed to the real graph database.")
            logger.info("Please ensure your Neo4j DBMS is STARTED in Neo4j Desktop (Port 7687).")
            return

        # 2. Extract Data from PostgreSQL (Reduced for local stability)
        logger.info("Extracting data (limited for local stability)...")
        upr_df = data_service.get_patients(limit=200, upr=True)
        
        if upr_df.empty:
            logger.warning("No data found in unified_patient_record. Please ensure database is seeded.")
            return

        # 3. Build Entity Graph in small chunks
        logger.info(f"Building entity graph for {len(upr_df)} patients in batches...")
        batch_size = 50
        for i in range(0, len(upr_df), batch_size):
            chunk = upr_df.iloc[i:i+batch_size]
            graph_service.build_entity_graph(chunk)
            logger.info(f"Pushed batch {i//batch_size + 1}")
        
        # 4. Extract and Add Issues (Reduced)
        logger.info("Extracting issue data...")
        issues_df = data_service.get_issues(limit=500)
        
        if not issues_df.empty:
            logger.info(f"Adding {len(issues_df)} issues to graph...")
            graph_service.add_issues_batch(issues_df)
            
            # 5. Create Cascade Relationships (Blocking)
            # We'll create clinical blocking relationships
            logger.info("Generating clinical blocking relationships...")
            with graph_service._driver.session() as session:
                # 1. Core Chain
                rules = [
                    ('missing_visits', 'missing_pages'),
                    ('missing_pages', 'open_queries'),
                    ('open_queries', 'sdv_incomplete'),
                    ('sdv_incomplete', 'signature_gaps'),
                    ('signature_gaps', 'db_lock'),
                    # 2. Safety Case
                    ('sae_dm_pending', 'sae_safety_pending'),
                    ('sae_safety_pending', 'db_lock'),
                    # 3. Medical Coding & Lab
                    ('meddra_uncoded', 'db_lock'),
                    ('whodrug_uncoded', 'db_lock'),
                    ('lab_issues', 'db_lock'),
                    ('edrr_issues', 'db_lock'),
                    ('inactivated_forms', 'db_lock'),
                    # 4. Operational
                    ('high_query_volume', 'open_queries')
                ]
                
                for source, target in rules:
                    session.run("""
                        MATCH (a:Issue {type: $source})
                        MATCH (b:Issue {type: $target})
                        WHERE a.id <> b.id
                        MERGE (a)-[:BLOCKS]->(b)
                    """, source=source, target=target)
                
                # Special rule for db_lock (which is a target node, but might be an issue type in mock)
                session.run("""
                    MERGE (target:Issue {id: 'DB_LOCK_TARGET', type: 'db_lock'})
                    WITH target
                    MATCH (source:Issue)
                    WHERE source.type IN ['signature_gaps', 'sae_safety_pending', 'meddra_uncoded']
                    MERGE (source)-[:BLOCKS]->(target)
                """)
        
        # 6. Verify Stats
        stats = graph_service.get_graph_stats()
        logger.info(f"Seeding complete! Graph stats: {stats}")
        
    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        raise
    finally:
        graph_service.close()

if __name__ == "__main__":
    seed_graph_from_postgres()
