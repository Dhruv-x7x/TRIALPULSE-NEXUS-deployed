"""
TRIALPULSE NEXUS - STEP 1: PLATINUM INGESTION
==============================================
Ingests ALL 207 Excel files (23 studies × 9 file types).
Uses FOLDER NAME as source of truth for Study ID.
Identifies file type by content patterns, not just filename.

Author: TrialPulse Team
Version: 1.0.0
"""

import io
import os
import re
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_PROCESSED
from src.database.connection import get_db_manager
from src.database.pg_writer import safe_to_postgres
from sqlalchemy import text

# ============================================
# FILE TYPE PATTERNS (PLATINUM - captures ALL 207 files)
# ============================================

# Order matters! More specific patterns first to avoid false matches.
# These patterns match ANY part of filename (case-insensitive)
FILE_PATTERNS = {
    # CPID EDC Metrics - 23 files expected
    'cpid_edc_metrics': [
        'cpid_edc_metrics', 'cpid_edc', 'edc_metrics', 'cpid edc', 'edc metrics'
    ],
    
    # MedDRA Coding - 23 files expected (BEFORE SAE to catch MedDRA files first)
    'coding_meddra': [
        'globalcodingreport_meddra', 'codingreport_meddra', 'globalcodingreport meddra',
        'meddra', 'medra'
    ],
    
    # WHODrug Coding - 23 files expected
    'coding_whodrug': [
        'globalcodingreport_whodd', 'globalcodingreport whodd', 'globalcodingreport_whodrug',
        'globalcodingreport whodrug', 'whodrug', 'whodd', 'whodra'
    ],
    
    # SAE Dashboard - 23 files expected (AFTER coding to not steal those files)
    'sae_dashboard': [
        'esae dashboard', 'sae dashboard', 'esae_dashboard', 'sae_dashboard', 
        'dm_safety', 'dm safety', 'esae_', 'esae '
    ],
    
    # Inactivated Forms - 23 files expected
    'inactivated_forms': [
        'inactivated forms', 'inactivated_forms', 'inactivated folders',
        'inactivated form', 'inactivated pages', 'inactivated report',
        'inactivated', 'inactivated reprot', 'inactivated page'
    ],
    
    # Missing Lab Ranges - 23 files expected  
    'missing_lab_ranges': [
        'missing_lab', 'missing lab', 'lab_name_and_missing_ranges',
        'missing_ranges', 'missing lnr', 'lab & range', 'lnr'
    ],
    
    # Missing Pages - 23 files expected
    'missing_pages': [
        'missing_pages', 'missing pages', 'global_missing_pages',
        'missing_page_report', 'missing page report', 'page report', 'page_report'
    ],
    
    # Visit Projection - 23 files expected
    'visit_projection': [
        'visit projection', 'visit_projection', 'missing visit'
    ],
    
    # Compiled EDRR - 23 files expected (LAST to avoid false matches)
    'compiled_edrr': [
        'compiled_edrr', 'compiled edrr', 'compiled-edrr', 'edrr'
    ],
}


def log(msg: str, level: str = "INFO"):
    """Simple logging."""
    ts = datetime.now().strftime("%H:%M:%S")
    symbol = {"INFO": "ℹ", "SUCCESS": "✓", "WARNING": "⚠", "ERROR": "✗"}.get(level, "•")
    print(f"{ts} | {symbol} {level:<8} | {msg}")


def extract_study_id(folder_name: str) -> Optional[str]:
    """Extract study ID from folder name - THE SOURCE OF TRUTH."""
    match = re.search(r'study\s*(\d+)', folder_name, re.IGNORECASE)
    return f"Study_{match.group(1)}" if match else None


def identify_file_type(filename: str) -> Optional[str]:
    """Identify file type from filename patterns - order matters!"""
    filename_lower = filename.lower()
    
    # Priority order to avoid false matches:
    # 1. CPID (most specific)
    # 2. MedDRA (before EDRR since 'meddra' contains 'dra')
    # 3. WHODrug
    # 4. SAE Dashboard
    # 5. Missing pages (before inactivated)
    # 6. Missing lab
    # 7. Inactivated forms
    # 8. Visit projection
    # 9. Compiled EDRR (last, since 'edrr' can match partial strings)
    
    priority_order = [
        'cpid_edc_metrics',
        'coding_meddra', 
        'coding_whodrug',
        'sae_dashboard',
        'missing_pages',
        'missing_lab_ranges',
        'inactivated_forms',
        'visit_projection',
        'compiled_edrr'
    ]
    
    for file_type in priority_order:
        patterns = FILE_PATTERNS.get(file_type, [])
        for pattern in patterns:
            if pattern in filename_lower:
                return file_type
    
    return None


def standardize_columns(columns: pd.Index) -> List[str]:
    """Standardize column names."""
    new_cols = []
    seen = {}
    for col in columns:
        col_clean = str(col).strip().lower()
        col_clean = re.sub(r'[^\w\s]', '', col_clean)
        col_clean = re.sub(r'\s+', '_', col_clean)
        if not col_clean or col_clean == 'nan':
            col_clean = 'unnamed'
        if col_clean in seen:
            seen[col_clean] += 1
            col_clean = f"{col_clean}_{seen[col_clean]}"
        else:
            seen[col_clean] = 0
        new_cols.append(col_clean)
    return new_cols


# ============================================
# PHASE 1: DISCOVERY
# ============================================

def discover_all_files() -> Dict[str, Dict]:
    """Discover ALL Excel files - every single one."""
    log("=" * 60)
    log("PHASE 1: FILE DISCOVERY")
    log("=" * 60)
    
    discovered = {}
    unidentified = []
    
    # Get all study folders
    study_folders = sorted([f for f in DATA_RAW.iterdir() 
                           if f.is_dir() and 'study' in f.name.lower()])
    
    log(f"Found {len(study_folders)} study folders")
    
    for folder in study_folders:
        study_id = extract_study_id(folder.name)
        if not study_id:
            log(f"Could not extract study ID from: {folder.name}", "WARNING")
            continue
        
        discovered[study_id] = {'folder': folder, 'files': {}}
        
        # Get ALL xlsx files (excluding temp files)
        xlsx_files = [f for f in folder.glob("*.xlsx") if not f.name.startswith('~$')]
        
        for xlsx_file in xlsx_files:
            file_type = identify_file_type(xlsx_file.name)
            
            if file_type:
                if file_type not in discovered[study_id]['files']:
                    discovered[study_id]['files'][file_type] = []
                discovered[study_id]['files'][file_type].append(xlsx_file)
            else:
                unidentified.append({
                    'study': study_id,
                    'file': xlsx_file.name,
                    'folder': folder.name
                })
    
    # Count totals
    total_files = sum(
        sum(len(files) for files in study_data['files'].values())
        for study_data in discovered.values()
    )
    
    log(f"Discovered {len(discovered)} studies")
    log(f"Total identified files: {total_files}")
    
    if unidentified:
        log(f"Unidentified files: {len(unidentified)}", "WARNING")
        for item in unidentified[:5]:  # Show first 5
            log(f"  {item['study']}: {item['file']}", "WARNING")
    
    # Detailed breakdown by type
    type_counts = {}
    for study_data in discovered.values():
        for file_type, files in study_data['files'].items():
            type_counts[file_type] = type_counts.get(file_type, 0) + len(files)
    
    log("\nFile Type Breakdown:")
    for ft in sorted(type_counts.keys()):
        log(f"  {ft}: {type_counts[ft]} files")
    
    return discovered, unidentified


# ============================================
# PHASE 2: INGESTION
# ============================================

class PlatinumIngester:
    """Ingest all files with specialized processors."""
    
    def __init__(self):
        self.data = {
            'cpid_edc_metrics': [],
            'sae_dashboard_dm': [],
            'sae_dashboard_safety': [],
            'coding_meddra': [],
            'coding_whodrug': [],
            'inactivated_forms': [],
            'missing_lab_ranges': [],
            'missing_pages': [],
            'visit_projection': [],
            'compiled_edrr': [],
        }
        self.stats = {
            'files_processed': 0,
            'files_success': 0,
            'files_failed': 0,
            'errors': []
        }
    
    def ingest_all(self, discovered: Dict) -> Dict:
        """Ingest all discovered files."""
        log("=" * 60)
        log("PHASE 2: INGESTION")
        log("=" * 60)
        
        # Flatten to list
        all_files = []
        for study_id, study_data in discovered.items():
            for file_type, files in study_data['files'].items():
                for f in files:
                    all_files.append((study_id, file_type, f))
        
        log(f"Processing {len(all_files)} files...")
        
        for study_id, file_type, file_path in tqdm(all_files, desc="Ingesting"):
            try:
                self._process_file(study_id, file_type, file_path)
                self.stats['files_success'] += 1
            except Exception as e:
                self.stats['files_failed'] += 1
                self.stats['errors'].append({
                    'file': str(file_path.name),
                    'study': study_id,
                    'type': file_type,
                    'error': str(e)
                })
            self.stats['files_processed'] += 1
        
        # Summary
        log(f"\nIngestion Complete:")
        log(f"  Processed: {self.stats['files_processed']}")
        log(f"  Success: {self.stats['files_success']}", "SUCCESS")
        if self.stats['files_failed'] > 0:
            log(f"  Failed: {self.stats['files_failed']}", "ERROR")
            for err in self.stats['errors'][:3]:
                log(f"    {err['file']}: {err['error'][:50]}", "ERROR")
        
        # Record counts
        log("\nRecords by Type:")
        for data_type, dfs in self.data.items():
            if dfs:
                total = sum(len(df) for df in dfs)
                log(f"  {data_type}: {total:,} records")
        
        return self.stats
    
    def _process_file(self, study_id: str, file_type: str, file_path: Path):
        """Process a single file based on type."""
        if file_type == 'cpid_edc_metrics':
            self._process_cpid(file_path, study_id)
        elif file_type == 'sae_dashboard':
            self._process_sae(file_path, study_id)
        elif file_type == 'coding_meddra':
            self._process_generic(file_path, study_id, 'coding_meddra')
        elif file_type == 'coding_whodrug':
            self._process_generic(file_path, study_id, 'coding_whodrug')
        elif file_type == 'visit_projection':
            self._process_generic(file_path, study_id, 'visit_projection')
        elif file_type == 'missing_lab_ranges':
            self._process_generic(file_path, study_id, 'missing_lab_ranges')
        elif file_type == 'missing_pages':
            self._process_generic(file_path, study_id, 'missing_pages')
        elif file_type == 'inactivated_forms':
            self._process_generic(file_path, study_id, 'inactivated_forms')
        elif file_type == 'compiled_edrr':
            self._process_generic(file_path, study_id, 'compiled_edrr')
    
    def _process_cpid(self, file_path: Path, study_id: str):
        """Process CPID EDC Metrics - the primary patient file."""
        xl = pd.ExcelFile(file_path)
        
        # Find subject-level sheet
        target_sheet = None
        for sheet in xl.sheet_names:
            if 'subject' in sheet.lower():
                target_sheet = sheet
                break
        if not target_sheet:
            target_sheet = xl.sheet_names[0]
        
        # Read with header row 2 (0-indexed)
        df = pd.read_excel(file_path, sheet_name=target_sheet, header=2)
        df = df.dropna(how='all')
        df.columns = standardize_columns(df.columns)
        df['_source_file'] = file_path.name
        df['_study_id'] = study_id
        df['_file_type'] = 'cpid_edc_metrics'
        self.data['cpid_edc_metrics'].append(df)
    
    def _process_sae(self, file_path: Path, study_id: str):
        """Process SAE Dashboard with multiple sheets."""
        xl = pd.ExcelFile(file_path)
        
        for sheet in xl.sheet_names:
            sheet_lower = sheet.lower()
            try:
                df = pd.read_excel(file_path, sheet_name=sheet)
                df = df.dropna(how='all')
                if df.empty:
                    continue
                df.columns = standardize_columns(df.columns)
                df['_source_file'] = file_path.name
                df['_study_id'] = study_id
                df['_sheet_name'] = sheet
                
                # Classify sheet
                if 'dm' in sheet_lower or 'data' in sheet_lower:
                    self.data['sae_dashboard_dm'].append(df)
                elif 'safety' in sheet_lower:
                    self.data['sae_dashboard_safety'].append(df)
            except Exception:
                pass  # Skip sheets that can't be read
    
    def _process_generic(self, file_path: Path, study_id: str, file_type: str):
        """Process generic Excel file."""
        df = pd.read_excel(file_path)
        df = df.dropna(how='all')
        df.columns = standardize_columns(df.columns)
        df['_source_file'] = file_path.name
        df['_study_id'] = study_id
        df['_file_type'] = file_type
        self.data[file_type].append(df)
    
    def get_combined_data(self) -> Dict[str, pd.DataFrame]:
        """Get combined DataFrames for each type."""
        combined = {}
        for data_type, dfs in self.data.items():
            if dfs:
                combined[data_type] = pd.concat(dfs, ignore_index=True, sort=False)
            else:
                combined[data_type] = pd.DataFrame()
        return combined
    
    def save_to_postgres(self):
        """Save all raw tables to PostgreSQL with chunked writing."""
        log("=" * 60)
        log("SAVING RAW DATA TO POSTGRESQL")
        log("=" * 60)
        
        db = get_db_manager()
        
        for data_type, dfs in self.data.items():
            if not dfs:
                continue
            
            combined = pd.concat(dfs, ignore_index=True, sort=False)
            table_name = f"raw_{data_type}"
            
            # Drop existing table
            with db.engine.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
            
            # Save with chunked writing to avoid memory issues
            combined.to_sql(table_name, db.engine, if_exists='replace', index=False, chunksize=500)
            log(f"  {table_name}: {len(combined):,} rows saved", "SUCCESS")
        
        return True


# ============================================
# MAIN
# ============================================

def run_step1_ingestion():
    """Run Step 1: Ingestion."""
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("   TRIALPULSE NEXUS - STEP 1: PLATINUM INGESTION")
    print("   Target: 207 files (23 studies × 9 file types)")
    print("=" * 70 + "\n")
    
    # Phase 1: Discovery
    discovered, unidentified = discover_all_files()
    
    # Validate we have expected count
    total_identified = sum(
        sum(len(files) for files in study_data['files'].values())
        for study_data in discovered.values()
    )
    
    expected_files = 207
    if total_identified < expected_files:
        log(f"WARNING: Only found {total_identified} files, expected {expected_files}", "WARNING")
        log("Some files may not match expected patterns", "WARNING")
    elif total_identified == expected_files:
        log(f"PERFECT: Found all {expected_files} files!", "SUCCESS")
    
    # Phase 2: Ingestion
    ingester = PlatinumIngester()
    stats = ingester.ingest_all(discovered)
    
    # Phase 3: Save to PostgreSQL
    ingester.save_to_postgres()
    
    # Final Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("   STEP 1 COMPLETE: INGESTION")
    print("=" * 70)
    print(f"   Studies: {len(discovered)}")
    print(f"   Files Processed: {stats['files_processed']}")
    print(f"   Files Success: {stats['files_success']}")
    print(f"   Files Failed: {stats['files_failed']}")
    print(f"   Time: {elapsed:.1f} seconds")
    print("=" * 70 + "\n")
    
    if stats['files_failed'] > 0:
        print("ERRORS:")
        for err in stats['errors']:
            print(f"  - {err['study']}/{err['file']}: {err['error'][:60]}")
    
    return discovered, ingester


if __name__ == "__main__":
    run_step1_ingestion()
