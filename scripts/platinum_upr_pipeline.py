"""
TRIALPULSE NEXUS - PLATINUM UPR PIPELINE (v1.0.0)
==================================================
Complete end-to-end pipeline that:
1. Ingests ALL 207 Excel files (23 studies × 9 file types)
2. Cleans and standardizes all data
3. Builds a perfect Unified Patient Record with 264+ features
4. Saves to PostgreSQL and Parquet

Author: TrialPulse Team
Version: 1.0.0 PLATINUM
"""

import io
import os
import re
import sys
import json
import traceback
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_PROCESSED, LOGS_DIR
from src.database.connection import get_db_manager
from sqlalchemy import text

# ============================================
# CONFIGURATION
# ============================================

# File type patterns for identification
FILE_TYPE_PATTERNS = {
    'cpid_edc_metrics': ['CPID_EDC_Metrics', 'CPID_EDC'],
    'sae_dashboard': ['eSAE Dashboard', 'SAE Dashboard', 'eSAE_Dashboard'],
    'coding_meddra': ['GlobalCodingReport_MedDRA', 'MedDRA'],
    'coding_whodrug': ['GlobalCodingReport_WHODD', 'WHODD', 'WHODrug'],
    'inactivated_forms': ['Inactivated Forms', 'Inactivated_Forms'],
    'missing_lab_ranges': ['Missing_Lab', 'Missing Lab', 'LNR'],
    'missing_pages': ['Missing_Pages', 'Missing Pages'],
    'visit_projection': ['Visit Projection', 'Visit_Projection'],
    'compiled_edrr': ['Compiled_EDRR', 'EDRR'],
}

# Column mapping for CPID EDC (unnamed columns)
CPID_COLUMN_MAP = {
    'unnamed_0': 'project_name',
    'unnamed_1': 'region', 
    'unnamed_2': 'country',
    'unnamed_3': 'site',
    'unnamed_4': 'subject',
    'unnamed_5': 'latest_visit',
    'unnamed_6': 'subject_status',
    'unnamed_7': 'input_files',
    'unnamed_8': 'cpmd',
    'unnamed_9': 'ssm',
    'unnamed_10': 'missing_visits',
    'unnamed_11': 'missing_pages',
    'unnamed_12': 'coded_terms',
    'unnamed_13': 'uncoded_terms',
    'unnamed_14': 'open_issues_lnr',
    'unnamed_15': 'open_issues_edrr',
    'unnamed_16': 'inactivated_forms_folders',
    'unnamed_17': 'sae_review_dm',
    'unnamed_18': 'sae_review_safety',
}

# Numeric columns for conversion
NUMERIC_COLS = [
    'missing_visits', 'missing_pages', 'coded_terms', 'uncoded_terms',
    'open_issues_lnr', 'open_issues_edrr', 'inactivated_forms_folders',
    'sae_review_dm', 'sae_review_safety', 'expected_visits_rave_edc_bo4',
    'pages_entered', 'pages_with_nonconformant_data',
    'total_crfs_with_queries_nonconformant_data',
    'crfs_require_verification_sdv', 'forms_verified',
    'crfs_frozen', 'crfs_not_frozen', 'crfs_locked', 'crfs_unlocked',
    'pds_confirmed', 'pds_proposed', 'crfs_signed', 'crfs_never_signed',
    'broken_signatures', 'dm_queries', 'clinical_queries', 'medical_queries',
    'site_queries', 'field_monitor_queries', 'coding_queries', 'safety_queries',
    'total_queries', 'clean_entered_crf', 'days_outstanding', 'issue_count',
]

# Subject status mapping
STATUS_MAP = {
    'screen failure': 'Screen Failure', 'screening failure': 'Screen Failure',
    'screen fail': 'Screen Failure', 'discontinued': 'Discontinued',
    'withdrawn': 'Discontinued', 'dropout': 'Discontinued',
    'completed': 'Completed', 'complete': 'Completed', 'finished': 'Completed',
    'ongoing': 'Ongoing', 'active': 'Ongoing', 'enrolled': 'Ongoing',
    'in progress': 'Ongoing', 'on study': 'Ongoing', 'screening': 'Screening',
    'survival': 'Ongoing', 'follow-up': 'Ongoing', 'follow up': 'Ongoing',
}


# ============================================
# UTILITY FUNCTIONS
# ============================================

def log(msg: str, level: str = "INFO"):
    """Simple logging."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{ts} | {level:<8} | {msg}")


def extract_study_id(folder_name: str) -> Optional[str]:
    """Extract study ID from folder name."""
    match = re.search(r'study\s*(\d+)', folder_name, re.IGNORECASE)
    return f"Study_{match.group(1)}" if match else None


def identify_file_type(filename: str) -> Optional[str]:
    """Identify file type from filename."""
    for file_type, patterns in FILE_TYPE_PATTERNS.items():
        if any(p.lower() in filename.lower() for p in patterns):
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


def safe_numeric(val, default=0):
    """Safely convert to numeric."""
    try:
        return pd.to_numeric(val, errors='coerce').fillna(default)
    except:
        return default


def standardize_study_id(val: Any) -> str:
    """Standardize study ID."""
    if pd.isna(val):
        return 'Unknown'
    val_str = str(val).strip()
    match = re.search(r'(\d+)', val_str)
    return f"Study_{match.group(1)}" if match else val_str


def standardize_site_id(val: Any) -> Optional[str]:
    """Standardize site ID."""
    if pd.isna(val) or str(val).lower() in ['none', 'nan', '']:
        return None
    val_str = str(val).strip()
    # Remove 'Site' prefix if present
    clean = re.sub(r'^site[\s_-]*', '', val_str, flags=re.IGNORECASE)
    return f"Site_{clean}" if clean else None


def standardize_subject_id(val: Any) -> Optional[str]:
    """Standardize subject ID."""
    if pd.isna(val) or str(val).lower() in ['none', 'nan', '', 'null']:
        return None
    val_str = str(val).strip()
    # Remove 'Subject' prefix if present and re-add in standard format
    clean = re.sub(r'^(subject|subj)[\s_-]*', '', val_str, flags=re.IGNORECASE)
    return f"Subject_{clean}" if clean else None


def standardize_status(val: Any) -> str:
    """Standardize subject status."""
    if pd.isna(val):
        return 'Unknown'
    val_lower = str(val).lower().strip()
    for pattern, canonical in STATUS_MAP.items():
        if pattern in val_lower:
            return canonical
    return 'Unknown'


def create_patient_key(study_id: str, site_id: Optional[str], subject_id: Optional[str]) -> str:
    """Create unique patient key."""
    site = site_id if site_id else 'Unknown'
    subject = subject_id if subject_id else 'Unknown'
    return f"{study_id}|{site}|{subject}"


def get_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Get column name from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
        for c in df.columns:
            if c.lower() == col.lower():
                return c
    return None


# ============================================
# PHASE 1: DISCOVERY
# ============================================

def discover_all_files() -> Dict[str, Dict[str, List[Path]]]:
    """Discover ALL Excel files in data/raw."""
    log("=" * 70)
    log("PHASE 1: FILE DISCOVERY")
    log("=" * 70)
    
    discovered = {}
    total_files = 0
    
    for folder in sorted(DATA_RAW.iterdir()):
        if not folder.is_dir() or 'study' not in folder.name.lower():
            continue
            
        study_id = extract_study_id(folder.name)
        if not study_id:
            continue
            
        discovered[study_id] = {'folder': folder, 'files': {}}
        
        for xlsx_file in folder.glob("*.xlsx"):
            if xlsx_file.name.startswith('~$'):  # Skip temp files
                continue
            file_type = identify_file_type(xlsx_file.name)
            if file_type:
                if file_type not in discovered[study_id]['files']:
                    discovered[study_id]['files'][file_type] = []
                discovered[study_id]['files'][file_type].append(xlsx_file)
                total_files += 1
    
    log(f"Discovered {len(discovered)} studies with {total_files} Excel files")
    
    # Summary by file type
    type_counts = {}
    for study_data in discovered.values():
        for file_type in study_data['files']:
            type_counts[file_type] = type_counts.get(file_type, 0) + len(study_data['files'][file_type])
    
    for ft, count in sorted(type_counts.items()):
        log(f"  {ft}: {count} files")
    
    return discovered


# ============================================
# PHASE 2: INGESTION
# ============================================

@dataclass
class IngestionResult:
    """Track ingestion results."""
    files_processed: int = 0
    files_success: int = 0
    files_failed: int = 0
    records_by_type: Dict[str, int] = field(default_factory=dict)
    errors: List[Dict] = field(default_factory=list)


class PlatinumIngester:
    """Ingest all file types with specialist processors."""
    
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
        self.result = IngestionResult()
    
    def ingest_all(self, discovered: Dict) -> IngestionResult:
        """Ingest all discovered files."""
        log("=" * 70)
        log("PHASE 2: INGESTION")
        log("=" * 70)
        
        all_files = []
        for study_id, study_data in discovered.items():
            for file_type, files in study_data['files'].items():
                for f in files:
                    all_files.append((study_id, file_type, f))
        
        log(f"Processing {len(all_files)} files...")
        
        for study_id, file_type, file_path in tqdm(all_files, desc="Ingesting"):
            try:
                self._process_file(study_id, file_type, file_path)
                self.result.files_success += 1
            except Exception as e:
                self.result.files_failed += 1
                self.result.errors.append({
                    'file': str(file_path),
                    'error': str(e)
                })
            self.result.files_processed += 1
        
        # Report results
        for data_type, dfs in self.data.items():
            if dfs:
                total = sum(len(df) for df in dfs)
                self.result.records_by_type[data_type] = total
                log(f"  {data_type}: {total:,} records")
        
        log(f"Ingestion Complete: {self.result.files_success}/{self.result.files_processed} successful")
        if self.result.files_failed > 0:
            log(f"  Failed: {self.result.files_failed} files", "WARNING")
        
        return self.result
    
    def _process_file(self, study_id: str, file_type: str, file_path: Path):
        """Process a single file."""
        if file_type == 'cpid_edc_metrics':
            self._process_cpid(file_path, study_id)
        elif file_type == 'sae_dashboard':
            self._process_sae(file_path, study_id)
        elif file_type == 'coding_meddra':
            self._process_coding(file_path, study_id, 'meddra')
        elif file_type == 'coding_whodrug':
            self._process_coding(file_path, study_id, 'whodrug')
        elif file_type == 'visit_projection':
            self._process_visit(file_path, study_id)
        elif file_type == 'missing_lab_ranges':
            self._process_lab(file_path, study_id)
        elif file_type == 'missing_pages':
            self._process_pages(file_path, study_id)
        elif file_type == 'inactivated_forms':
            self._process_inactivated(file_path, study_id)
        elif file_type == 'compiled_edrr':
            self._process_edrr(file_path, study_id)
    
    def _process_cpid(self, file_path: Path, study_id: str):
        """Process CPID EDC Metrics file."""
        xl = pd.ExcelFile(file_path)
        # Find the subject-level sheet
        target_sheet = next((s for s in xl.sheet_names if 'subject' in s.lower()), xl.sheet_names[0])
        df = pd.read_excel(file_path, sheet_name=target_sheet, header=2)
        df = df.dropna(how='all')
        df.columns = standardize_columns(df.columns)
        df['_source_file'] = file_path.name
        df['_study_id'] = study_id
        df['_file_type'] = 'cpid_edc_metrics'
        self.data['cpid_edc_metrics'].append(df)
    
    def _process_sae(self, file_path: Path, study_id: str):
        """Process SAE Dashboard file with DM and Safety sheets."""
        xl = pd.ExcelFile(file_path)
        for sheet in xl.sheet_names:
            sheet_lower = sheet.lower()
            if 'dm' in sheet_lower or 'data' in sheet_lower:
                df = pd.read_excel(file_path, sheet_name=sheet)
                df.columns = standardize_columns(df.columns)
                df['_source_file'] = file_path.name
                df['_study_id'] = study_id
                self.data['sae_dashboard_dm'].append(df)
            elif 'safety' in sheet_lower:
                df = pd.read_excel(file_path, sheet_name=sheet)
                df.columns = standardize_columns(df.columns)
                df['_source_file'] = file_path.name
                df['_study_id'] = study_id
                self.data['sae_dashboard_safety'].append(df)
    
    def _process_coding(self, file_path: Path, study_id: str, coding_type: str):
        """Process coding report (MedDRA or WHODrug)."""
        df = pd.read_excel(file_path)
        df = df.dropna(how='all')
        df.columns = standardize_columns(df.columns)
        df['_source_file'] = file_path.name
        df['_study_id'] = study_id
        df['_coding_type'] = coding_type
        self.data[f'coding_{coding_type}'].append(df)
    
    def _process_visit(self, file_path: Path, study_id: str):
        """Process visit projection tracker."""
        df = pd.read_excel(file_path)
        df = df.dropna(how='all')
        df.columns = standardize_columns(df.columns)
        df['_source_file'] = file_path.name
        df['_study_id'] = study_id
        self.data['visit_projection'].append(df)
    
    def _process_lab(self, file_path: Path, study_id: str):
        """Process missing lab ranges file."""
        df = pd.read_excel(file_path)
        df = df.dropna(how='all')
        df.columns = standardize_columns(df.columns)
        df['_source_file'] = file_path.name
        df['_study_id'] = study_id
        self.data['missing_lab_ranges'].append(df)
    
    def _process_pages(self, file_path: Path, study_id: str):
        """Process missing pages report."""
        df = pd.read_excel(file_path)
        df = df.dropna(how='all')
        df.columns = standardize_columns(df.columns)
        df['_source_file'] = file_path.name
        df['_study_id'] = study_id
        self.data['missing_pages'].append(df)
    
    def _process_inactivated(self, file_path: Path, study_id: str):
        """Process inactivated forms report."""
        df = pd.read_excel(file_path)
        df = df.dropna(how='all')
        df.columns = standardize_columns(df.columns)
        df['_source_file'] = file_path.name
        df['_study_id'] = study_id
        self.data['inactivated_forms'].append(df)
    
    def _process_edrr(self, file_path: Path, study_id: str):
        """Process compiled EDRR file."""
        df = pd.read_excel(file_path)
        df = df.dropna(how='all')
        df.columns = standardize_columns(df.columns)
        df['_source_file'] = file_path.name
        df['_study_id'] = study_id
        self.data['compiled_edrr'].append(df)
    
    def get_combined_data(self) -> Dict[str, pd.DataFrame]:
        """Get combined DataFrames for each type."""
        combined = {}
        for data_type, dfs in self.data.items():
            if dfs:
                combined[data_type] = pd.concat(dfs, ignore_index=True, sort=False)
            else:
                combined[data_type] = pd.DataFrame()
        return combined


# ============================================
# PHASE 3: CLEANING & AGGREGATION
# ============================================

class PlatinumCleaner:
    """Clean and aggregate all data types."""
    
    def __init__(self, raw_data: Dict[str, pd.DataFrame]):
        self.raw = raw_data
        self.clean = {}
        self.agg = {}
    
    def clean_all(self):
        """Clean and aggregate all data."""
        log("=" * 70)
        log("PHASE 3: CLEANING & AGGREGATION")
        log("=" * 70)
        
        # 1. Clean CPID (primary patient table)
        self._clean_cpid()
        
        # 2. Aggregate detail tables to patient level
        self._agg_visit_projection()
        self._agg_missing_lab()
        self._agg_missing_pages()
        self._agg_inactivated()
        self._agg_edrr()
        self._agg_sae_dm()
        self._agg_sae_safety()
        self._agg_coding_meddra()
        self._agg_coding_whodrug()
        
        log("Cleaning complete")
        return self
    
    def _clean_cpid(self):
        """Clean CPID EDC Metrics."""
        log("Cleaning CPID EDC Metrics...")
        df = self.raw.get('cpid_edc_metrics')
        if df is None or df.empty:
            log("  No CPID data found!", "WARNING")
            self.clean['cpid'] = pd.DataFrame()
            return
        
        initial = len(df)
        
        # Map unnamed columns
        rename_map = {old: new for old, new in CPID_COLUMN_MAP.items() if old in df.columns}
        df = df.rename(columns=rename_map)
        
        # Remove junk rows
        junk_patterns = ['responsible', 'lf for action', 'site/cra', 'coder', 'safety team', 
                         'investigator', 'cse/cdd', 'cdmd', 'cpmd', 'ssm']
        check_cols = [c for c in ['project_name', 'region', 'country', 'site', 'subject'] if c in df.columns]
        if not check_cols:
            check_cols = list(df.columns)[:5]
        
        def is_junk(row):
            for col in check_cols:
                val = row.get(col)
                if pd.notna(val):
                    val_lower = str(val).lower().strip()
                    if any(p in val_lower for p in junk_patterns):
                        return True
            return False
        
        mask = ~df.apply(is_junk, axis=1)
        df = df[mask].reset_index(drop=True)
        
        # Remove rows without valid subject
        if 'subject' in df.columns:
            valid = (df['subject'].notna() & 
                    (df['subject'].astype(str) != 'None') &
                    (df['subject'].astype(str) != '') &
                    (df['subject'].astype(str) != 'nan'))
            df = df[valid].reset_index(drop=True)
        
        # Standardize IDs
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        df['site_id'] = df['site'].apply(standardize_site_id) if 'site' in df.columns else None
        df['subject_id'] = df['subject'].apply(standardize_subject_id) if 'subject' in df.columns else None
        df['subject_status_clean'] = df['subject_status'].apply(standardize_status) if 'subject_status' in df.columns else 'Unknown'
        df['patient_key'] = df.apply(
            lambda r: create_patient_key(r['study_id'], r.get('site_id'), r.get('subject_id')),
            axis=1
        )
        
        # Convert numeric columns
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = safe_numeric(df[col], 0)
        
        # Drop duplicates
        df = df.drop_duplicates(subset=['patient_key'], keep='first').reset_index(drop=True)
        
        log(f"  CPID: {initial:,} → {len(df):,} rows ({df['patient_key'].nunique():,} unique patients)")
        self.clean['cpid'] = df
    
    def _agg_visit_projection(self):
        """Aggregate visit projection to patient level."""
        df = self.raw.get('visit_projection')
        if df is None or df.empty:
            self.agg['visit'] = pd.DataFrame(columns=['patient_key', 'visit_missing_visit_count', 
                                                       'visit_visits_overdue_max_days', 'visit_visits_overdue_avg_days'])
            return
        
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        subject_col = get_column(df, ['subject', 'subject_id', 'subjectname'])
        site_col = get_column(df, ['site', 'site_id', 'sitenumber'])
        days_col = get_column(df, ['days_outstanding', 'days_outstanding_1'])
        
        df['subject_id'] = df[subject_col].apply(standardize_subject_id) if subject_col else None
        df['site_id'] = df[site_col].apply(standardize_site_id) if site_col else None
        df['days_outstanding'] = safe_numeric(df[days_col], 0) if days_col else 0
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], r.get('site_id'), r['subject_id']), axis=1)
        
        agg = df.groupby('patient_key').agg(
            visit_missing_visit_count=('patient_key', 'count'),
            visit_visits_overdue_max_days=('days_outstanding', 'max'),
            visit_visits_overdue_avg_days=('days_outstanding', 'mean')
        ).reset_index()
        
        log(f"  Visit Projection: {len(agg):,} patients")
        self.agg['visit'] = agg
    
    def _agg_missing_lab(self):
        """Aggregate missing lab ranges to patient level."""
        df = self.raw.get('missing_lab_ranges')
        if df is None or df.empty:
            self.agg['lab'] = pd.DataFrame(columns=['patient_key', 'lab_lab_issue_count'])
            return
        
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        subject_col = get_column(df, ['subject', 'subject_id', 'patient_id'])
        site_col = get_column(df, ['site', 'site_id', 'sitenumber'])
        
        df['subject_id'] = df[subject_col].apply(standardize_subject_id) if subject_col else None
        df['site_id'] = df[site_col].apply(standardize_site_id) if site_col else None
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], r.get('site_id'), r['subject_id']), axis=1)
        
        agg = df.groupby('patient_key').agg(
            lab_lab_issue_count=('patient_key', 'count')
        ).reset_index()
        
        log(f"  Missing Lab Ranges: {len(agg):,} patients")
        self.agg['lab'] = agg
    
    def _agg_missing_pages(self):
        """Aggregate missing pages to patient level."""
        df = self.raw.get('missing_pages')
        if df is None or df.empty:
            self.agg['pages'] = pd.DataFrame(columns=['patient_key', 'pages_missing_page_count', 
                                                       'pages_pages_missing_max_days', 'pages_pages_missing_avg_days'])
            return
        
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        subject_col = get_column(df, ['subjectname', 'subject_name', 'subject', 'subject_id'])
        site_col = get_column(df, ['sitenumber', 'site_number', 'site', 'site_id'])
        days_col = get_column(df, ['no_days_page_missing', 'of_days_missing', 'days_missing'])
        
        df['subject_id'] = df[subject_col].apply(standardize_subject_id) if subject_col else None
        df['site_id'] = df[site_col].apply(standardize_site_id) if site_col else None
        df['days_missing'] = safe_numeric(df[days_col], 0) if days_col else 0
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], r.get('site_id'), r['subject_id']), axis=1)
        
        agg = df.groupby('patient_key').agg(
            pages_missing_page_count=('patient_key', 'count'),
            pages_pages_missing_max_days=('days_missing', 'max'),
            pages_pages_missing_avg_days=('days_missing', 'mean')
        ).reset_index()
        
        log(f"  Missing Pages: {len(agg):,} patients")
        self.agg['pages'] = agg
    
    def _agg_inactivated(self):
        """Aggregate inactivated forms to patient level."""
        df = self.raw.get('inactivated_forms')
        if df is None or df.empty:
            self.agg['inactivated'] = pd.DataFrame(columns=['patient_key', 'inactivated_inactivated_form_count'])
            return
        
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        subject_col = get_column(df, ['subject', 'subject_id', 'patient_id'])
        site_col = get_column(df, ['site', 'site_id', 'study_site_number'])
        
        df['subject_id'] = df[subject_col].apply(standardize_subject_id) if subject_col else None
        df['site_id'] = df[site_col].apply(standardize_site_id) if site_col else None
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], r.get('site_id'), r['subject_id']), axis=1)
        
        agg = df.groupby('patient_key').agg(
            inactivated_inactivated_form_count=('patient_key', 'count')
        ).reset_index()
        
        log(f"  Inactivated Forms: {len(agg):,} patients")
        self.agg['inactivated'] = agg
    
    def _agg_edrr(self):
        """Aggregate EDRR to patient level."""
        df = self.raw.get('compiled_edrr')
        if df is None or df.empty:
            self.agg['edrr'] = pd.DataFrame(columns=['patient_key', 'edrr_edrr_issue_count'])
            return
        
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        subject_col = get_column(df, ['subject', 'subject_id', 'patient_id'])
        
        df['subject_id'] = df[subject_col].apply(standardize_subject_id) if subject_col else None
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], 'Unknown', r['subject_id']), axis=1)
        
        count_col = get_column(df, ['total_open_issue_count_per_subject', 'issue_count'])
        df['issue_count'] = safe_numeric(df[count_col], 1) if count_col else 1
        
        agg = df.groupby('patient_key').agg(
            edrr_edrr_issue_count=('issue_count', 'sum')
        ).reset_index()
        
        log(f"  Compiled EDRR: {len(agg):,} patients")
        self.agg['edrr'] = agg
    
    def _agg_sae_dm(self):
        """Aggregate SAE DM to patient level."""
        df = self.raw.get('sae_dashboard_dm')
        if df is None or df.empty:
            self.agg['sae_dm'] = pd.DataFrame(columns=['patient_key', 'sae_dm_sae_dm_total', 
                                                        'sae_dm_sae_dm_pending', 'sae_dm_sae_dm_completed'])
            return
        
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        patient_col = get_column(df, ['patient_id', 'patientid', 'subject', 'subject_id'])
        site_col = get_column(df, ['site', 'site_id', 'siteid'])
        
        df['subject_id'] = df[patient_col].apply(standardize_subject_id) if patient_col else None
        df['site_id'] = df[site_col].apply(standardize_site_id) if site_col else None
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], r.get('site_id'), r['subject_id']), axis=1)
        
        review_col = get_column(df, ['review_status', 'reviewstatus', 'status'])
        if review_col:
            df['is_pending'] = df[review_col].astype(str).str.contains('Pending', case=False, na=False)
            df['is_completed'] = df[review_col].astype(str).str.contains('Completed|Complete', case=False, na=False)
        else:
            df['is_pending'] = False
            df['is_completed'] = False
        
        agg = df.groupby('patient_key').agg(
            sae_dm_sae_dm_total=('patient_key', 'count'),
            sae_dm_sae_dm_pending=('is_pending', 'sum'),
            sae_dm_sae_dm_completed=('is_completed', 'sum')
        ).reset_index()
        
        log(f"  SAE DM: {len(agg):,} patients")
        self.agg['sae_dm'] = agg
    
    def _agg_sae_safety(self):
        """Aggregate SAE Safety to patient level."""
        df = self.raw.get('sae_dashboard_safety')
        if df is None or df.empty:
            self.agg['sae_safety'] = pd.DataFrame(columns=['patient_key', 'sae_safety_sae_safety_total', 
                                                            'sae_safety_sae_safety_pending', 'sae_safety_sae_safety_completed'])
            return
        
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        patient_col = get_column(df, ['patient_id', 'patientid', 'subject', 'subject_id'])
        site_col = get_column(df, ['site', 'site_id', 'siteid'])
        
        df['subject_id'] = df[patient_col].apply(standardize_subject_id) if patient_col else None
        df['site_id'] = df[site_col].apply(standardize_site_id) if site_col else None
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], r.get('site_id'), r['subject_id']), axis=1)
        
        review_col = get_column(df, ['review_status', 'reviewstatus', 'status'])
        if review_col:
            df['is_pending'] = df[review_col].astype(str).str.contains('Pending', case=False, na=False)
            df['is_completed'] = df[review_col].astype(str).str.contains('Completed|Complete', case=False, na=False)
        else:
            df['is_pending'] = False
            df['is_completed'] = False
        
        agg = df.groupby('patient_key').agg(
            sae_safety_sae_safety_total=('patient_key', 'count'),
            sae_safety_sae_safety_pending=('is_pending', 'sum'),
            sae_safety_sae_safety_completed=('is_completed', 'sum')
        ).reset_index()
        
        log(f"  SAE Safety: {len(agg):,} patients")
        self.agg['sae_safety'] = agg
    
    def _agg_coding_meddra(self):
        """Aggregate MedDRA coding to patient level."""
        df = self.raw.get('coding_meddra')
        if df is None or df.empty:
            self.agg['meddra'] = pd.DataFrame(columns=['patient_key', 'meddra_coding_meddra_total', 
                                                        'meddra_coding_meddra_coded', 'meddra_coding_meddra_uncoded'])
            return
        
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        subject_col = get_column(df, ['subject', 'subject_id', 'patient_id'])
        
        df['subject_id'] = df[subject_col].apply(standardize_subject_id) if subject_col else None
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], 'Unknown', r['subject_id']), axis=1)
        
        status_col = get_column(df, ['coding_status', 'status', 'codingstatus'])
        if status_col:
            status_str = df[status_col].astype(str)
            df['is_coded'] = status_str.str.contains('Coded', case=False, na=False) & \
                            ~status_str.str.contains('UnCoded|Un-Coded|Not Coded', case=False, na=False)
            df['is_uncoded'] = status_str.str.contains('UnCoded|Un-Coded|Not Coded', case=False, na=False)
        else:
            df['is_coded'] = False
            df['is_uncoded'] = True
        
        agg = df.groupby('patient_key').agg(
            meddra_coding_meddra_total=('patient_key', 'count'),
            meddra_coding_meddra_coded=('is_coded', 'sum'),
            meddra_coding_meddra_uncoded=('is_uncoded', 'sum')
        ).reset_index()
        
        log(f"  Coding MedDRA: {len(agg):,} patients")
        self.agg['meddra'] = agg
    
    def _agg_coding_whodrug(self):
        """Aggregate WHODrug coding to patient level."""
        df = self.raw.get('coding_whodrug')
        if df is None or df.empty:
            self.agg['whodrug'] = pd.DataFrame(columns=['patient_key', 'whodrug_coding_whodrug_total', 
                                                         'whodrug_coding_whodrug_coded', 'whodrug_coding_whodrug_uncoded'])
            return
        
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        subject_col = get_column(df, ['subject', 'subject_id', 'patient_id'])
        
        df['subject_id'] = df[subject_col].apply(standardize_subject_id) if subject_col else None
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], 'Unknown', r['subject_id']), axis=1)
        
        status_col = get_column(df, ['coding_status', 'status', 'codingstatus'])
        if status_col:
            status_str = df[status_col].astype(str)
            df['is_coded'] = status_str.str.contains('Coded', case=False, na=False) & \
                            ~status_str.str.contains('UnCoded|Un-Coded|Not Coded', case=False, na=False)
            df['is_uncoded'] = status_str.str.contains('UnCoded|Un-Coded|Not Coded', case=False, na=False)
        else:
            df['is_coded'] = False
            df['is_uncoded'] = True
        
        agg = df.groupby('patient_key').agg(
            whodrug_coding_whodrug_total=('patient_key', 'count'),
            whodrug_coding_whodrug_coded=('is_coded', 'sum'),
            whodrug_coding_whodrug_uncoded=('is_uncoded', 'sum')
        ).reset_index()
        
        log(f"  Coding WHODrug: {len(agg):,} patients")
        self.agg['whodrug'] = agg
    
    def get_clean_cpid(self) -> pd.DataFrame:
        """Return cleaned CPID data."""
        return self.clean.get('cpid', pd.DataFrame())
    
    def get_aggregated_data(self) -> Dict[str, pd.DataFrame]:
        """Return all aggregated data."""
        return self.agg


# ============================================
# PHASE 4: UPR BUILDER
# ============================================

class PlatinumUPRBuilder:
    """Build the Unified Patient Record with 264+ features."""
    
    def __init__(self, cpid: pd.DataFrame, aggregations: Dict[str, pd.DataFrame]):
        self.cpid = cpid
        self.aggs = aggregations
        self.upr = None
    
    def build(self) -> pd.DataFrame:
        """Build the complete UPR."""
        log("=" * 70)
        log("PHASE 4: UPR CONSTRUCTION")
        log("=" * 70)
        
        # Start with CPID as base
        df = self.cpid.copy()
        log(f"Base CPID: {len(df):,} patients")
        
        # Merge all aggregations
        for agg_name, agg_df in self.aggs.items():
            if not agg_df.empty and 'patient_key' in agg_df.columns:
                df = df.merge(agg_df, on='patient_key', how='left')
                log(f"  + {agg_name}: merged")
        
        # Fill NaN with 0 for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        log(f"After merging: {len(df):,} patients, {len(df.columns)} columns")
        
        # Run advanced feature engineering
        df = self._engineer_features(df)
        
        self.upr = df
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply advanced feature engineering."""
        log("Running Advanced Feature Engineering...")
        
        # Import the feature engineer
        from src.data.advanced_feature_engineering import AdvancedFeatureEngineer
        
        engineer = AdvancedFeatureEngineer(df)
        df = engineer.run_all()
        
        log(f"  Final feature count: {len(df.columns)}")
        return df


# ============================================
# PHASE 5: SAVE TO DATABASE
# ============================================

def save_to_database(upr: pd.DataFrame, db):
    """Save UPR to PostgreSQL."""
    log("=" * 70)
    log("PHASE 5: SAVE TO DATABASE")
    log("=" * 70)
    
    # Clean column names for Postgres
    upr.columns = [c.lower().replace(' ', '_').replace('(', '').replace(')', '')
                   .replace('-', '_').replace('.', '_') for c in upr.columns]
    
    # Remove duplicates
    upr = upr.loc[:, ~upr.columns.duplicated()]
    
    # Add metadata
    upr['_upr_built_ts'] = datetime.now().isoformat()
    upr['_upr_version'] = '1.0.0-PLATINUM'
    
    # Drop existing table
    with db.engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS unified_patient_record CASCADE"))
    
    # Save to SQL
    log(f"Saving {len(upr):,} rows to unified_patient_record...")
    upr.to_sql('unified_patient_record', db.engine, if_exists='fail', index=False, chunksize=1000)
    
    # Create views
    with db.engine.begin() as conn:
        conn.execute(text("""
            CREATE OR REPLACE VIEW patient_dqi AS 
            SELECT patient_key, site_id, study_id, data_quality_index_8comp as dqi_score 
            FROM unified_patient_record;
        """))
        conn.execute(text("""
            CREATE OR REPLACE VIEW patient_clean_status AS 
            SELECT patient_key, site_id, study_id, clean_status_tier 
            FROM unified_patient_record;
        """))
        conn.execute(text("""
            CREATE OR REPLACE VIEW patient_dblock_status AS 
            SELECT patient_key, site_id, study_id, is_db_lock_ready 
            FROM unified_patient_record;
        """))
    
    # Verify
    with db.engine.connect() as conn:
        row_count = conn.execute(text("SELECT COUNT(*) FROM unified_patient_record")).scalar()
        col_count = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.columns 
            WHERE table_name = 'unified_patient_record'
        """)).scalar()
    
    log(f"Verified: {row_count:,} rows, {col_count} columns in PostgreSQL")
    
    return row_count, col_count


def save_to_parquet(upr: pd.DataFrame):
    """Save UPR to Parquet for analytics."""
    log("Saving to Parquet...")
    
    upr_dir = DATA_PROCESSED / "upr"
    upr_dir.mkdir(parents=True, exist_ok=True)
    upr.to_parquet(upr_dir / "unified_patient_record.parquet", index=False)
    
    analytics_dir = DATA_PROCESSED / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    
    # Export subsets
    if 'data_quality_index_8comp' in upr.columns:
        dqi_df = upr[['patient_key', 'site_id', 'study_id', 'data_quality_index_8comp']].rename(
            columns={'data_quality_index_8comp': 'dqi_score'})
        dqi_df.to_parquet(analytics_dir / "patient_dqi_enhanced.parquet", index=False)
    
    if 'clean_status_tier' in upr.columns:
        clean_df = upr[['patient_key', 'site_id', 'study_id', 'clean_status_tier']].copy()
        clean_df['tier1_clean'] = clean_df['clean_status_tier'].apply(lambda x: 1 if 'Tier 1' in str(x) else 0)
        clean_df['tier2_clean'] = clean_df['clean_status_tier'].apply(lambda x: 1 if 'Tier 2' in str(x) else 0)
        clean_df.to_parquet(analytics_dir / "patient_clean_status.parquet", index=False)
    
    if 'is_db_lock_ready' in upr.columns:
        dblock_df = upr[['patient_key', 'site_id', 'study_id', 'is_db_lock_ready']].copy()
        dblock_df['dblock_status'] = dblock_df['is_db_lock_ready'].apply(lambda x: 'ready' if x else 'not_ready')
        dblock_df.to_parquet(analytics_dir / "patient_dblock_status.parquet", index=False)
    
    log(f"Saved Parquet files to {upr_dir} and {analytics_dir}")


# ============================================
# MAIN PIPELINE
# ============================================

def run_platinum_pipeline():
    """Run the complete Platinum UPR Pipeline."""
    start_time = datetime.now()
    
    print("\n" + "=" * 80)
    print("   TRIALPULSE NEXUS - PLATINUM UPR PIPELINE v1.0.0")
    print("   Building Perfect Unified Patient Record from 207 Excel Files")
    print("=" * 80 + "\n")
    
    # Phase 1: Discover
    discovered = discover_all_files()
    
    # Phase 2: Ingest
    ingester = PlatinumIngester()
    ingester.ingest_all(discovered)
    raw_data = ingester.get_combined_data()
    
    # Phase 3: Clean & Aggregate
    cleaner = PlatinumCleaner(raw_data)
    cleaner.clean_all()
    cpid = cleaner.get_clean_cpid()
    aggs = cleaner.get_aggregated_data()
    
    # Phase 4: Build UPR
    builder = PlatinumUPRBuilder(cpid, aggs)
    upr = builder.build()
    
    # Phase 5: Save
    db = get_db_manager()
    row_count, col_count = save_to_database(upr, db)
    save_to_parquet(upr)
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("   PLATINUM PIPELINE COMPLETE")
    print("=" * 80)
    print(f"   Studies Processed: {len(discovered)}")
    print(f"   Files Processed: {ingester.result.files_processed}")
    print(f"   Patients in UPR: {row_count:,}")
    print(f"   Features: {col_count}")
    print(f"   Time Elapsed: {elapsed:.1f} seconds")
    print("=" * 80 + "\n")
    
    return upr


if __name__ == "__main__":
    run_platinum_pipeline()
