"""
TRIALPULSE NEXUS - Real Data Kafka Feeder
=========================================
Synchronizes local CSV datasets with Kafka topics to simulate real-time production flow.
"""

import sys
import time
import json
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.streaming.kafka_producer import get_event_producer
from src.streaming.stream_config import get_stream_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def run_feeder():
    logger.info("🚀 Starting Real Data Kafka Feeder...")
    
    producer = get_event_producer()
    config = get_stream_config()
    
    # Define sources
    sources = [
        {
            'file': 'data/processed/cpid_edc_metrics.csv',
            'topic_key': 'patient_vitals',
            'entity_id_col': 'unnamed_4', # Subject ID based on head inspection
            'delay': 0.5
        },
        {
            'file': 'data/processed/sae_dashboard_safety.csv',
            'topic_key': 'issue_detected',
            'entity_id_col': 'patient_id',
            'delay': 2.0
        },
        {
            'file': 'data/processed/upr/upr_sample.csv',
            'topic_key': 'patient_updated',
            'entity_id_col': 'patient_key',
            'delay': 1.0
        }
    ]
    
    try:
        while True:
            for source in sources:
                file_path = Path(source['file'])
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                logger.info(f"Processing {file_path.name}...")
                df = pd.read_csv(file_path)
                
                # Stream a sample of records periodically
                sample = df.sample(min(len(df), 5))
                for _, row in sample.iterrows():
                    entity_id = str(row.get(source['entity_id_col'], 'unknown'))
                    payload = row.to_dict()
                    
                    # Convert any timestamp columns to ISO strings
                    for k, v in payload.items():
                        if isinstance(v, (pd.Timestamp, datetime)):
                            payload[k] = v.isoformat()
                    
                    success = producer.publish(
                        event_type=source['topic_key'],
                        key=entity_id,
                        payload=payload,
                        source="real_data_feeder"
                    )
                    
                    if success:
                        logger.info(f"Published {source['topic_key']} for {entity_id}")
                    else:
                        logger.error(f"Failed to publish {source['topic_key']}")
                    
                    time.sleep(source['delay'])
            
            logger.info("Cycle complete. Waiting 10s...")
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("Feeder stopped by user.")
    finally:
        producer.close()

if __name__ == "__main__":
    run_feeder()
