import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database.connection import get_db_manager
from sqlalchemy import text

def deploy_models():
    db = get_db_manager()
    with db.engine.connect() as conn:
        # Check if we have model versions
        res = conn.execute(text("SELECT version_id, model_name FROM ml_model_versions")).fetchall()
        if not res:
            print("No model versions found to deploy.")
            return
        
        # Mark all as deployed for demo purposes, or at least the ones needed
        conn.execute(text("UPDATE ml_model_versions SET status = 'deployed'"))
        conn.commit()
        print(f"Marked {len(res)} model versions as 'deployed'.")

if __name__ == "__main__":
    deploy_models()
