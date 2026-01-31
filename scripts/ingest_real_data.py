
import os
import pandas as pd
import numpy as np
from pathlib import Path
import uuid
from datetime import datetime, timedelta
import sys

# Add project root
sys.path.insert(0, os.getcwd())
from src.database.connection import get_db_manager
from sqlalchemy import text

RAW_DIR = Path(r"D:\trialpulse_nexus\data\raw")

def ingest_real_data():
    db = get_db_manager()
    engine = db.engine
    
    all_issues = []
    
    print("Searching for raw files...")
    
    for study_folder in os.listdir(RAW_DIR):
        folder_path = RAW_DIR / study_folder
        if not folder_path.is_dir():
            continue
            
        print(f"\nProcessing {study_folder}...")
        
        # 1. MedDRA Coding
        for f in os.listdir(folder_path):
            if 'MedDRA' in f and f.endswith('.xlsx'):
                try:
                    df = pd.read_excel(folder_path / f)
                    pending = df[df['Require Coding'] == 'Yes']
                    for _, row in pending.iterrows():
                        all_issues.append({
                            "issue_id": f"EXC-M-{uuid.uuid4().hex[:6]}",
                            "patient_key": f"{row['Study']}_{row['Subject']}",
                            "site_id": f"{row['Study']}_Site_Unknown", # Site info not in this file
                            "category": "coding_required",
                            "issue_type": "meddra_uncoded",
                            "description": f"MedDRA coding required for {row['Field OID']} in {row['Form OID']}",
                            "priority": "Medium",
                            "status": "open",
                            "cascade_impact_score": 5.0,
                            "created_at": datetime.now() - timedelta(days=10)
                        })
                except Exception as e:
                    print(f"Error reading {f}: {e}")

        # 2. WHODrug Coding
        for f in os.listdir(folder_path):
            if ('WHODD' in f or 'WHODrug' in f) and f.endswith('.xlsx'):
                try:
                    df = pd.read_excel(folder_path / f)
                    pending = df[df['Require Coding'] == 'Yes']
                    for _, row in pending.iterrows():
                        all_issues.append({
                            "issue_id": f"EXC-W-{uuid.uuid4().hex[:6]}",
                            "patient_key": f"{row['Study']}_{row['Subject']}",
                            "site_id": f"{row['Study']}_Site_Unknown",
                            "category": "coding_required",
                            "issue_type": "whodrug_uncoded",
                            "description": f"WHODrug coding required for {row['Field OID']} in {row['Form OID']}",
                            "priority": "Medium",
                            "status": "open",
                            "cascade_impact_score": 5.0,
                            "created_at": datetime.now() - timedelta(days=10)
                        })
                except Exception as e:
                    print(f"Error reading {f}: {e}")

        # 3. SAE Discrepancies
        for f in os.listdir(folder_path):
            if 'SAE Dashboard' in f and f.endswith('.xlsx'):
                try:
                    xl = pd.ExcelFile(folder_path / f)
                    for sheet in xl.sheet_names:
                        df = pd.read_excel(folder_path / f, sheet_name=sheet)
                        pending = df[df['Review Status'].astype(str).str.contains('Pending|Open', na=False)]
                        for _, row in pending.iterrows():
                            all_issues.append({
                                "issue_id": f"EXC-S-{row['Discrepancy ID']}",
                                "patient_key": f"{row['Study ID']}_{row['Patient ID']}",
                                "site_id": f"{row['Study ID']}_{row['Site']}".replace(' ', '_'),
                                "category": "safety_issue",
                                "issue_type": "sae_discrepancy",
                                "description": f"SAE discrepancy in {row['Form Name']}: {row['Review Status']}",
                                "priority": "High",
                                "status": "open",
                                "cascade_impact_score": 8.5,
                                "created_at": pd.to_datetime(row['Discrepancy Created Timestamp in Dashboard'])
                            })
                except Exception as e:
                    print(f"Error reading {f}: {e}")

        # 4. Open Queries
        for f in os.listdir(folder_path):
            if 'EDC_Metrics' in f and f.endswith('.xlsx'):
                try:
                    xl = pd.ExcelFile(folder_path / f)
                    if 'Query Report - Cumulative' in xl.sheet_names:
                        df = pd.read_excel(folder_path / f, sheet_name='Query Report - Cumulative')
                        open_q = df[df['Query Status'] == 'Open']
                        for _, row in open_q.iterrows():
                            all_issues.append({
                                "issue_id": f"EXC-Q-{uuid.uuid4().hex[:6]}",
                                "patient_key": f"{row['Study']}_{row['Subject Name']}",
                                "site_id": f"{row['Study']}_{row['Site Number']}".replace(' ', '_'),
                                "category": "open_query",
                                "issue_type": "Open Query",
                                "description": f"Open Query on {row['Form']} {row['Field OID']}: Assigned to {row['Action Owner']}",
                                "priority": "Medium",
                                "status": "open",
                                "cascade_impact_score": 4.0,
                                "created_at": pd.to_datetime(row['Query Open Date'])
                            })
                except Exception as e:
                    print(f"Error reading {f}: {e}")

    if not all_issues:
        print("No real issues found in Excel files.")
        return

    print(f"\nFound {len(all_issues)} real issues in Excel files.")
    
    # Convert to DataFrame
    df_issues = pd.DataFrame(all_issues)
    
    with engine.begin() as conn:
        print("Cleaning up old mock/seeded data (issue_id starting with COD- or EXC-)...")
        conn.execute(text("DELETE FROM project_issues WHERE issue_id LIKE 'COD-%' OR issue_id LIKE 'EXC-%'"))
        
        print("Inserting real data from Excel...")
        df_issues.to_sql('project_issues', engine, if_exists='append', index=False)
        
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_real_data()
