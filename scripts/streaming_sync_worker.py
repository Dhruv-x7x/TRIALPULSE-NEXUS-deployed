"""
TRIALPULSE NEXUS - Real-Time Streaming Synchronization Worker
=============================================================
Consumes events from Kafka and synchronizes them with the PostgreSQL database.
This bridges the gap between raw telemetry and the clinical dashboard.
"""

import sys
import logging
import json
import uuid
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.streaming.kafka_consumer import get_event_consumer
from src.database.connection import get_db_manager
from sqlalchemy import text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("SyncWorker")

def handle_patient_vitals(payload):
    """
    Handle incoming patient vital updates from streaming.
    Updates the DQI and status in PostgreSQL.
    """
    try:
        data = payload.get("payload", {})
        patient_key = payload.get("key")
        
        if not patient_key:
            return

        logger.info(f"Syncing vitals for patient: {patient_key}")
        
        # In production, we'd calculate the new DQI based on these vitals.
        # For this implementation, we simulate the update in the DB.
        db_manager = get_db_manager()
        
        # Example: Update patient DQI based on streamed telemetry
        new_dqi = data.get("dqi_score", 85.0)
        
        db_manager = get_db_manager()
        engine = db_manager.engine
        if engine is None:
            logger.error("Failed to get database engine")
            return
            
        with engine.connect() as conn:
            conn.execute(
                text("UPDATE patients SET dqi_score = :dqi, updated_at = :now WHERE patient_key = :key"),
                {"dqi": new_dqi, "now": datetime.utcnow(), "key": patient_key}
            )
            conn.commit()
            
        logger.info(f"Patient {patient_key} synchronized. New DQI: {new_dqi}")
        
    except Exception as e:
        logger.error(f"Error syncing patient vitals: {e}")

def handle_issue_detected(payload):
    """
    Handle real-time issue detection events.
    """
    try:
        data = payload.get("payload", {})
        patient_key = payload.get("key")
        
        # Guard against None values that cause DB violations
        if patient_key is None:
            logger.debug("Skipping issue sync: patient_key is None")
            return
            
        issue_type = data.get('issue_type') or 'Telemetry Alert'
        logger.warning(f"New Issue Streamed for {patient_key}: {issue_type}")
        
        # Logic to insert into project_issues table
        db_manager = get_db_manager()
        engine = db_manager.engine
        if engine is None:
            logger.error("Failed to get database engine for issue sync")
            return
            
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO project_issues 
                    (issue_id, patient_key, site_id, category, issue_type, description, priority, severity, status, created_at)
                    VALUES 
                    (:id, :p_key, :s_id, :cat, :type, :desc, :pri, :sev, 'open', :now)
                """),
                {
                    "id": f"STRM-{datetime.now().strftime('%M%S')}-{uuid.uuid4().hex[:4]}",
                    "p_key": patient_key,
                    "s_id": data.get("site_id", "UNK") or "UNK",
                    "cat": "Streaming",
                    "type": issue_type,
                    "desc": data.get("description", "Automatic alert from real-time stream"),
                    "pri": data.get("priority", "High") or "High",
                    "sev": data.get("severity", "Minor") or "Minor",
                    "now": datetime.utcnow()
                }
            )
            conn.commit()
            
    except Exception as e:
        logger.error(f"Error processing streamed issue: {e}")

def run_worker():
    import uuid # Ensure uuid is available
    logger.info("Launching TrialPulse Streaming Sync Worker...")
    
    consumer = get_event_consumer()
    
    # Register handlers
    consumer.register_handler("patient_vitals", handle_patient_vitals)
    consumer.register_handler("issue_detected", handle_issue_detected)
    
    try:
        # Start consuming (blocking loop)
        consumer.start(blocking=True)
    except KeyboardInterrupt:
        logger.info("Worker stopped by user.")
    finally:
        consumer.stop()

if __name__ == "__main__":
    run_worker()
