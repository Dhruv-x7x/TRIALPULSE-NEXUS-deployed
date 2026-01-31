"""
TRIALPULSE NEXUS - Knowledge Graph Sync Worker
==============================================
Background worker to synchronize 9 heterogeneous data sources from PostgreSQL 
into the Neo4j Knowledge Graph in real-time.
"""

import sys
import io

# Fix console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import time
import logging
import signal
from datetime import datetime
from pathlib import Path
import pandas as pd
from sqlalchemy import text

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import get_db_manager
from src.knowledge.neo4j_graph import get_graph_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "neo4j_sync.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Neo4jSyncWorker:
    def __init__(self, interval_seconds: int = 300): # Sync every 5 mins
        self.interval = interval_seconds
        self.running = True
        self._db_manager = get_db_manager()
        self.graph_service = get_graph_service()
        
        # Register signals
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        
    def stop(self, *args):
        logger.info("Stopping Neo4j sync worker...")
        self.running = False
        
    def sync(self):
        """Synchronize data from PG to Neo4j."""
        logger.info("Starting Knowledge Graph synchronization...")
        
        if self.graph_service.uses_mock:
            logger.warning("Neo4j service is in mock mode. Real sync skipped.")
            return

        try:
            with self._db_manager.engine.connect() as conn:
                # 1. Sync Studies
                studies = pd.read_sql(text("SELECT * FROM studies"), conn)
                for _, s in studies.iterrows():
                    self.graph_service.upsert_study(str(s['study_id']), str(s['name']))
                
                # 2. Sync Sites
                sites = pd.read_sql(text("SELECT * FROM clinical_sites"), conn)
                for _, s in sites.iterrows():
                    self.graph_service.upsert_site(str(s['site_id']), str(s['name']), str(s['region']))
                
                # 3. Sync Patients
                patients = pd.read_sql(text("SELECT patient_key, site_id, study_id FROM patients"), conn)
                for _, p in patients.iterrows():
                    self.graph_service.upsert_patient(str(p['patient_key']), str(p['site_id']))
                
                # 4. Sync Issues and Dependencies (Cascade Engine Foundation)
                issues = pd.read_sql(text("SELECT * FROM project_issues WHERE status = 'open'"), conn)
                for _, i in issues.iterrows():
                    self.graph_service.upsert_issue(
                        str(i['issue_id']), str(i['patient_key']), str(i['issue_type']), 
                        str(i['priority']), float(i['cascade_impact_score'] or 0.0)
                    )
                
                logger.info(f"✅ Sync complete: {len(studies)} studies, {len(sites)} sites, {len(patients)} patients")
                
        except Exception as e:
            logger.error(f"Sync failed: {e}")

    def run(self):
        logger.info(f"🚀 Knowledge Graph Sync Worker started (Interval: {self.interval}s)")
        
        while self.running:
            try:
                self.sync()
                if self.running:
                    time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    worker = Neo4jSyncWorker()
    worker.run()
