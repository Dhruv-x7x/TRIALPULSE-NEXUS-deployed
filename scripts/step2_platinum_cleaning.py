"""
TRIALPULSE NEXUS - STEP 2: PLATINUM CLEANING & STANDARDIZATION
================================================================
Industry-grade data cleaning pipeline that:
1. Reads raw ingested data from PostgreSQL
2. Applies comprehensive cleaning rules:
   - ID Normalization (Study, Site, Subject)
   - Subject Status Standardization
   - Numeric Conversion with Validation
   - Date Parsing with Multiple Formats
   - Duplicate Detection & Handling
   - Outlier Detection
3. Aggregates detail tables to patient level
4. Saves cleaned data to PostgreSQL

Author: TrialPulse Team
Version: 1.0.0 PLATINUM
"""

import io
import os
import re
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
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
from sqlalchemy import text


# ============================================
# CONFIGURATION
# ============================================

# Subject Status Mapping (comprehensive)
STATUS_MAP = {
    # Screen Failure
    'screen failure': 'Screen Failure', 'screening failure': 'Screen Failure',
    'screen fail': 'Screen Failure', 'screenfailure': 'Screen Failure',
    'sf': 'Screen Failure',
    
    # Discontinued
    'discontinued': 'Discontinued', 'discontinue': 'Discontinued',
    'withdrawn': 'Discontinued', 'dropout': 'Discontinued',
    'drop out': 'Discontinued', 'early termination': 'Discontinued',
    'lost to follow-up': 'Discontinued', 'ltfu': 'Discontinued',
    
    # Completed
    'completed': 'Completed', 'complete': 'Completed', 
    'finished': 'Completed', 'study completed': 'Completed',
    
    # Ongoing
    'ongoing': 'Ongoing', 'active': 'Ongoing', 'enrolled': 'Ongoing',
    'in progress': 'Ongoing', 'on study': 'Ongoing', 'on treatment': 'Ongoing',
    'survival': 'Ongoing', 'survival follow-up': 'Ongoing',
    'follow-up': 'Ongoing', 'follow up': 'Ongoing', 'treatment': 'Ongoing',
    
    # Screening
    'screening': 'Screening', 'in screening': 'Screening',
}

# Primary numeric columns for CPID
CPID_NUMERIC_COLS = [
    'expected_visits_rave_edc_bo4', 'pages_entered', 'pages_with_nonconformant_data',
    'total_crfs_with_queries_nonconformant_data', 'clean_entered_crf',
    'crfs_require_verification_sdv', 'forms_verified',
    'crfs_frozen', 'crfs_not_frozen', 'crfs_locked', 'crfs_unlocked',
    'pds_confirmed', 'pds_proposed', 'crfs_signed', 'crfs_never_signed',
    'broken_signatures',
    'crfs_overdue_for_signs_within_45_days_of_data_entry',
    'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry',
    'crfs_overdue_for_signs_beyond_90_days_of_data_entry',
    'dm_queries', 'clinical_queries', 'medical_queries', 'safety_queries',
    'coding_queries', 'site_queries', 'field_monitor_queries', 'total_queries',
]

# Date formats to try
DATE_FORMATS = [
    "%Y-%m-%d", "%d-%b-%Y", "%d%b%Y", "%m/%d/%Y", "%d/%m/%Y",
    "%Y-%m-%d %H:%M:%S", "%d-%b-%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
    "%d-%m-%Y", "%Y/%m/%d", "%b %d, %Y"
]

# Junk row patterns to remove
JUNK_PATTERNS = [
    'responsible', 'lf for action', 'site/cra', 'coder', 'safety team',
    'investigator', 'cse/cdd', 'cdmd', 'cpmd', 'ssm'
]


def log(msg: str, level: str = "INFO"):
    """Simple logging with symbols."""
    ts = datetime.now().strftime("%H:%M:%S")
    symbol = {"INFO": "ℹ", "SUCCESS": "✓", "WARNING": "⚠", "ERROR": "✗"}.get(level, "•")
    print(f"{ts} | {symbol} {level:<8} | {msg}")


# ============================================
# UTILITY FUNCTIONS
# ============================================

def safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    """Convert to numeric safely."""
    return pd.to_numeric(series, errors='coerce').fillna(default)


def parse_date(value: Any) -> Optional[str]:
    """Parse date with multiple formats."""
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d")
    
    value_str = str(value).strip()
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(value_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except:
            continue
    
    # Try pandas parser
    try:
        dt = pd.to_datetime(value_str, errors='coerce')
        if pd.notna(dt):
            return dt.strftime("%Y-%m-%d")
    except:
        pass
    
    return None


def standardize_study_id(value: Any) -> str:
    """Standardize study ID to Study_X format."""
    if pd.isna(value) or str(value).lower() in ['none', 'nan', '']:
        return 'Unknown'
    value_str = str(value).strip()
    match = re.search(r'(\d+)', value_str)
    return f"Study_{match.group(1)}" if match else value_str


def standardize_site_id(value: Any) -> Optional[str]:
    """Standardize site ID to Site_X format."""
    if pd.isna(value) or str(value).lower() in ['none', 'nan', '']:
        return None
    value_str = str(value).strip()
    if not value_str:
        return None
    # Remove 'Site' prefix if present
    clean = re.sub(r'^site[\s_-]*', '', value_str, flags=re.IGNORECASE)
    return f"Site_{clean}" if clean else None


def standardize_subject_id(value: Any) -> Optional[str]:
    """Standardize subject ID to Subject_X format."""
    if pd.isna(value):
        return None
    value_str = str(value).strip()
    if value_str.lower() in ['none', 'nan', '', 'null']:
        return None
    # Remove 'Subject' prefix if present
    clean = re.sub(r'^(subject|subj)[\s_-]*', '', value_str, flags=re.IGNORECASE)
    return f"Subject_{clean}" if clean else None


def standardize_status(value: Any) -> str:
    """Standardize subject status."""
    if pd.isna(value):
        return 'Unknown'
    value_lower = str(value).lower().strip()
    
    # Direct match
    if value_lower in STATUS_MAP:
        return STATUS_MAP[value_lower]
    
    # Partial match
    for pattern, canonical in STATUS_MAP.items():
        if pattern in value_lower:
            return canonical
    
    return 'Unknown'


def create_patient_key(study_id: str, site_id: Optional[str], subject_id: Optional[str]) -> str:
    """Create unique patient key."""
    site = site_id if site_id else 'Unknown'
    subject = subject_id if subject_id else 'Unknown'
    return f"{study_id}|{site}|{subject}"


def get_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Get first matching column name from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
        for c in df.columns:
            if c.lower() == col.lower():
                return c
    return None


def detect_outliers_iqr(series: pd.Series, multiplier: float = 3.0) -> pd.Series:
    """Detect outliers using IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return (series < lower) | (series > upper)


# ============================================
# CPID CLEANER (PRIMARY PATIENT TABLE)
# ============================================

class CPIDCleaner:
    """Clean CPID EDC Metrics - the primary patient table."""
    
    def __init__(self):
        self.stats = {
            'input_rows': 0,
            'output_rows': 0,
            'junk_removed': 0,
            'invalid_subjects_removed': 0,
            'duplicates_removed': 0,
            'outliers_detected': 0,
        }
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full cleaning pipeline for CPID."""
        log("=" * 60)
        log("CLEANING CPID EDC METRICS (Primary Patient Table)")
        log("=" * 60)
        
        self.stats['input_rows'] = len(df)
        log(f"Input: {len(df):,} rows, {len(df.columns)} columns")
        
        # Step 0: Map unnamed columns to meaningful names
        df = self._map_columns(df)
        
        # Step 1: Remove junk rows
        df = self._remove_junk_rows(df)
        
        # Step 2: Remove invalid subjects
        df = self._remove_invalid_subjects(df)
        
        # Step 3: Standardize IDs
        df = self._standardize_ids(df)
        
        # Step 4: Standardize subject status
        df = self._standardize_status(df)
        
        # Step 5: Convert numeric columns
        df = self._convert_numeric(df)
        
        # Step 6: Parse dates
        df = self._parse_dates(df)
        
        # Step 7: Create patient key
        df = self._create_patient_key(df)
        
        # Step 8: Detect outliers
        df = self._detect_outliers(df)
        
        # Step 9: Remove duplicates (keep first)
        df = self._remove_duplicates(df)
        
        # Step 10: Add metadata
        df['_cleaned_ts'] = datetime.now().isoformat()
        df['_cleaning_version'] = '2.0.0-PLATINUM'
        
        self.stats['output_rows'] = len(df)
        
        log(f"\nCPID Cleaning Summary:")
        log(f"  Input: {self.stats['input_rows']:,} rows")
        log(f"  Junk removed: {self.stats['junk_removed']:,}")
        log(f"  Invalid subjects removed: {self.stats['invalid_subjects_removed']:,}")
        log(f"  Duplicates removed: {self.stats['duplicates_removed']:,}")
        log(f"  Outliers flagged: {self.stats['outliers_detected']:,}")
        log(f"  Output: {self.stats['output_rows']:,} rows", "SUCCESS")
        log(f"  Unique patients: {df['patient_key'].nunique():,}", "SUCCESS")
        
        return df
    
    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map unnamed columns to meaningful names based on CPID structure."""
        df = df.copy()
        
        # CPID column mapping (unnamed_X -> meaningful names)
        CPID_COLUMN_MAP = {
            'unnamed_0': 'project_name',
            'unnamed_1': 'region', 
            'unnamed_2': 'country',
            'unnamed_3': 'site',           # Site ID is in unnamed_3
            'unnamed_4': 'subject',        # Subject ID is in unnamed_4
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
        
        # Apply mapping
        rename_map = {old: new for old, new in CPID_COLUMN_MAP.items() if old in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)
            log(f"  Mapped {len(rename_map)} unnamed columns to meaningful names")
        
        # Log sample values
        if 'subject' in df.columns:
            sample_subjects = df['subject'].dropna().head(3).tolist()
            log(f"  Subject sample: {sample_subjects}")
        if 'site' in df.columns:
            sample_sites = df['site'].dropna().head(3).tolist()
            log(f"  Site sample: {sample_sites}")
        
        return df
    
    def _remove_junk_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove junk/header rows."""
        initial = len(df)
        
        check_cols = [c for c in df.columns if not c.startswith('_')][:10]
        
        def is_junk(row):
            for col in check_cols:
                val = row.get(col)
                if pd.notna(val):
                    val_lower = str(val).lower().strip()
                    if any(p in val_lower for p in JUNK_PATTERNS):
                        return True
            return False
        
        mask = ~df.apply(is_junk, axis=1)
        df = df[mask].reset_index(drop=True)
        
        self.stats['junk_removed'] = initial - len(df)
        log(f"  Removed {self.stats['junk_removed']:,} junk rows")
        return df
    
    def _remove_invalid_subjects(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows without valid subject ID."""
        initial = len(df)
        
        subject_col = get_column(df, ['subject', 'subject_id', 'subjectname', 'patient_id'])
        
        if subject_col:
            valid = (
                df[subject_col].notna() &
                (df[subject_col].astype(str).str.lower() != 'none') &
                (df[subject_col].astype(str).str.lower() != 'nan') &
                (df[subject_col].astype(str) != '') &
                (df[subject_col].astype(str).str.len() > 1)
            )
            df = df[valid].reset_index(drop=True)
        
        self.stats['invalid_subjects_removed'] = initial - len(df)
        log(f"  Removed {self.stats['invalid_subjects_removed']:,} invalid subjects")
        return df
    
    def _standardize_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize Study, Site, Subject IDs."""
        df = df.copy()
        
        # Study ID (from _study_id metadata)
        if '_study_id' in df.columns:
            df['study_id'] = df['_study_id'].apply(standardize_study_id)
        else:
            df['study_id'] = 'Unknown'
        
        # Site ID
        site_col = get_column(df, ['site', 'site_id', 'sitenumber', 'site_number'])
        if site_col:
            df['site_id'] = df[site_col].apply(standardize_site_id)
        else:
            df['site_id'] = None
        
        # Subject ID
        subject_col = get_column(df, ['subject', 'subject_id', 'subjectname', 'patient_id'])
        if subject_col:
            df['subject_id'] = df[subject_col].apply(standardize_subject_id)
        else:
            df['subject_id'] = None
        
        log(f"  Standardized IDs: {df['study_id'].nunique()} studies, {df['site_id'].nunique()} sites")
        return df
    
    def _standardize_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize subject status."""
        df = df.copy()
        
        status_col = get_column(df, ['subject_status', 'status', 'subjectstatus'])
        if status_col:
            df['subject_status_original'] = df[status_col]
            df['subject_status_clean'] = df[status_col].apply(standardize_status)
            
            dist = df['subject_status_clean'].value_counts()
            log("  Status distribution:")
            for status, count in dist.head(5).items():
                log(f"    {status}: {count:,}")
        else:
            df['subject_status_clean'] = 'Unknown'
        
        return df
    
    def _convert_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert numeric columns with validation."""
        df = df.copy()
        converted = 0
        
        for col in CPID_NUMERIC_COLS:
            if col in df.columns:
                df[col] = safe_numeric(df[col], 0.0)
                converted += 1
        
        # Also convert any column with numeric patterns
        numeric_patterns = ['count', 'total', 'queries', 'pages', 'crfs', 'forms', 
                          'visits', 'verified', 'signed', 'pending']
        for col in df.columns:
            if col not in CPID_NUMERIC_COLS:
                if any(p in col.lower() for p in numeric_patterns):
                    df[col] = safe_numeric(df[col], 0.0)
                    converted += 1
        
        log(f"  Converted {converted} numeric columns")
        return df
    
    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date columns."""
        df = df.copy()
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'visit' in c.lower()]
        
        for col in date_cols:
            if df[col].dtype == 'object':
                df[f'{col}_parsed'] = df[col].apply(parse_date)
        
        log(f"  Parsed {len(date_cols)} date columns")
        return df
    
    def _create_patient_key(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create unique patient key."""
        df = df.copy()
        df['patient_key'] = df.apply(
            lambda r: create_patient_key(
                str(r.get('study_id', 'Unknown')),
                r.get('site_id'),
                r.get('subject_id')
            ), axis=1
        )
        return df
    
    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers in numeric columns."""
        df = df.copy()
        
        # Key columns to check
        outlier_cols = ['total_queries', 'pages_entered', 'expected_visits_rave_edc_bo4']
        outlier_flags = []
        
        for col in outlier_cols:
            if col in df.columns:
                outliers = detect_outliers_iqr(df[col].fillna(0))
                df[f'{col}_is_outlier'] = outliers.astype(int)
                outlier_flags.append(f'{col}_is_outlier')
        
        if outlier_flags:
            df['has_outlier'] = (df[outlier_flags].sum(axis=1) > 0).astype(int)
            self.stats['outliers_detected'] = df['has_outlier'].sum()
        else:
            df['has_outlier'] = 0
        
        log(f"  Detected {self.stats['outliers_detected']:,} outlier rows")
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate patient_keys, keeping first occurrence."""
        initial = len(df)
        df = df.drop_duplicates(subset=['patient_key'], keep='first').reset_index(drop=True)
        self.stats['duplicates_removed'] = initial - len(df)
        log(f"  Removed {self.stats['duplicates_removed']:,} duplicates")
        return df


# ============================================
# DETAIL TABLE AGGREGATORS
# ============================================

class DetailAggregator:
    """Aggregate detail tables to patient level."""
    
    def __init__(self, db):
        self.db = db
    
    def aggregate_visit_projection(self) -> pd.DataFrame:
        """Aggregate visit projection to patient level."""
        log("Aggregating visit_projection...")
        
        try:
            df = pd.read_sql("SELECT * FROM raw_visit_projection", self.db.engine)
        except:
            return pd.DataFrame(columns=['patient_key'])
        
        if df.empty:
            return pd.DataFrame(columns=['patient_key'])
        
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        
        subject_col = get_column(df, ['subject', 'subject_id', 'subjectname'])
        site_col = get_column(df, ['site', 'site_id', 'sitenumber'])
        days_col = get_column(df, ['days_outstanding', 'days_outstanding_1', 'days'])
        
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
        
        log(f"  {len(agg):,} patients with visit issues", "SUCCESS")
        return agg
    
    def aggregate_missing_pages(self) -> pd.DataFrame:
        """Aggregate missing pages to patient level."""
        log("Aggregating missing_pages...")
        
        try:
            df = pd.read_sql("SELECT * FROM raw_missing_pages", self.db.engine)
        except:
            return pd.DataFrame(columns=['patient_key'])
        
        if df.empty:
            return pd.DataFrame(columns=['patient_key'])
        
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        
        subject_col = get_column(df, ['subjectname', 'subject_name', 'subject', 'subject_id'])
        site_col = get_column(df, ['sitenumber', 'site_number', 'site', 'site_id'])
        days_col = get_column(df, ['no_days_page_missing', 'of_days_missing', 'days_missing', 'days'])
        
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
        
        log(f"  {len(agg):,} patients with page issues", "SUCCESS")
        return agg
    
    def aggregate_missing_lab(self) -> pd.DataFrame:
        """Aggregate missing lab ranges to patient level."""
        log("Aggregating missing_lab_ranges...")
        
        try:
            df = pd.read_sql("SELECT * FROM raw_missing_lab_ranges", self.db.engine)
        except:
            return pd.DataFrame(columns=['patient_key'])
        
        if df.empty:
            return pd.DataFrame(columns=['patient_key'])
        
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        
        subject_col = get_column(df, ['subject', 'subject_id', 'patient_id', 'subjectname'])
        site_col = get_column(df, ['site', 'site_id', 'sitenumber'])
        
        df['subject_id'] = df[subject_col].apply(standardize_subject_id) if subject_col else None
        df['site_id'] = df[site_col].apply(standardize_site_id) if site_col else None
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], r.get('site_id'), r['subject_id']), axis=1)
        
        agg = df.groupby('patient_key').agg(
            lab_lab_issue_count=('patient_key', 'count')
        ).reset_index()
        
        log(f"  {len(agg):,} patients with lab issues", "SUCCESS")
        return agg
    
    def aggregate_inactivated(self) -> pd.DataFrame:
        """Aggregate inactivated forms to patient level."""
        log("Aggregating inactivated_forms...")
        
        try:
            df = pd.read_sql("SELECT * FROM raw_inactivated_forms", self.db.engine)
        except:
            return pd.DataFrame(columns=['patient_key'])
        
        if df.empty:
            return pd.DataFrame(columns=['patient_key'])
        
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        
        subject_col = get_column(df, ['subject', 'subject_id', 'patient_id', 'subjectname'])
        site_col = get_column(df, ['site', 'site_id', 'study_site_number', 'sitenumber'])
        
        df['subject_id'] = df[subject_col].apply(standardize_subject_id) if subject_col else None
        df['site_id'] = df[site_col].apply(standardize_site_id) if site_col else None
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], r.get('site_id'), r['subject_id']), axis=1)
        
        agg = df.groupby('patient_key').agg(
            inactivated_inactivated_form_count=('patient_key', 'count')
        ).reset_index()
        
        log(f"  {len(agg):,} patients with inactivated forms", "SUCCESS")
        return agg
    
    def aggregate_edrr(self) -> pd.DataFrame:
        """Aggregate EDRR to patient level."""
        log("Aggregating compiled_edrr...")
        
        try:
            df = pd.read_sql("SELECT * FROM raw_compiled_edrr", self.db.engine)
        except:
            return pd.DataFrame(columns=['patient_key'])
        
        if df.empty:
            return pd.DataFrame(columns=['patient_key'])
        
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        
        subject_col = get_column(df, ['subject', 'subject_id', 'patient_id', 'subjectname'])
        
        df['subject_id'] = df[subject_col].apply(standardize_subject_id) if subject_col else None
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], 'Unknown', r['subject_id']), axis=1)
        
        count_col = get_column(df, ['total_open_issue_count_per_subject', 'issue_count', 'open_issues'])
        df['issue_count'] = safe_numeric(df[count_col], 1) if count_col else 1
        
        agg = df.groupby('patient_key').agg(
            edrr_edrr_issue_count=('issue_count', 'sum')
        ).reset_index()
        
        log(f"  {len(agg):,} patients with EDRR issues", "SUCCESS")
        return agg
    
    def aggregate_sae_dm(self) -> pd.DataFrame:
        """Aggregate SAE DM to patient level."""
        log("Aggregating sae_dashboard_dm...")
        
        try:
            df = pd.read_sql("SELECT * FROM raw_sae_dashboard_dm", self.db.engine)
        except:
            return pd.DataFrame(columns=['patient_key'])
        
        if df.empty:
            return pd.DataFrame(columns=['patient_key'])
        
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        
        patient_col = get_column(df, ['patient_id', 'patientid', 'subject', 'subject_id'])
        site_col = get_column(df, ['site', 'site_id', 'siteid'])
        
        df['subject_id'] = df[patient_col].apply(standardize_subject_id) if patient_col else None
        df['site_id'] = df[site_col].apply(standardize_site_id) if site_col else None
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], r.get('site_id'), r['subject_id']), axis=1)
        
        # Check review status
        review_col = get_column(df, ['review_status', 'reviewstatus', 'status'])
        if review_col:
            df['is_pending'] = df[review_col].astype(str).str.contains('Pending|pending', na=False)
            df['is_completed'] = df[review_col].astype(str).str.contains('Completed|Complete|completed', na=False)
        else:
            df['is_pending'] = False
            df['is_completed'] = False
        
        agg = df.groupby('patient_key').agg(
            sae_dm_sae_dm_total=('patient_key', 'count'),
            sae_dm_sae_dm_pending=('is_pending', 'sum'),
            sae_dm_sae_dm_completed=('is_completed', 'sum')
        ).reset_index()
        
        log(f"  {len(agg):,} patients with SAE DM records", "SUCCESS")
        return agg
    
    def aggregate_sae_safety(self) -> pd.DataFrame:
        """Aggregate SAE Safety to patient level."""
        log("Aggregating sae_dashboard_safety...")
        
        try:
            df = pd.read_sql("SELECT * FROM raw_sae_dashboard_safety", self.db.engine)
        except:
            return pd.DataFrame(columns=['patient_key'])
        
        if df.empty:
            return pd.DataFrame(columns=['patient_key'])
        
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        
        patient_col = get_column(df, ['patient_id', 'patientid', 'subject', 'subject_id'])
        site_col = get_column(df, ['site', 'site_id', 'siteid'])
        
        df['subject_id'] = df[patient_col].apply(standardize_subject_id) if patient_col else None
        df['site_id'] = df[site_col].apply(standardize_site_id) if site_col else None
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], r.get('site_id'), r['subject_id']), axis=1)
        
        # Check review status
        review_col = get_column(df, ['review_status', 'reviewstatus', 'status'])
        if review_col:
            df['is_pending'] = df[review_col].astype(str).str.contains('Pending|pending', na=False)
            df['is_completed'] = df[review_col].astype(str).str.contains('Completed|Complete|completed', na=False)
        else:
            df['is_pending'] = False
            df['is_completed'] = False
        
        agg = df.groupby('patient_key').agg(
            sae_safety_sae_safety_total=('patient_key', 'count'),
            sae_safety_sae_safety_pending=('is_pending', 'sum'),
            sae_safety_sae_safety_completed=('is_completed', 'sum')
        ).reset_index()
        
        log(f"  {len(agg):,} patients with SAE Safety records", "SUCCESS")
        return agg
    
    def aggregate_coding_meddra(self) -> pd.DataFrame:
        """Aggregate MedDRA coding to patient level."""
        log("Aggregating coding_meddra...")
        
        try:
            df = pd.read_sql("SELECT * FROM raw_coding_meddra", self.db.engine)
        except:
            return pd.DataFrame(columns=['patient_key'])
        
        if df.empty:
            return pd.DataFrame(columns=['patient_key'])
        
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        
        subject_col = get_column(df, ['subject', 'subject_id', 'patient_id', 'subjectname'])
        
        df['subject_id'] = df[subject_col].apply(standardize_subject_id) if subject_col else None
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], 'Unknown', r['subject_id']), axis=1)
        
        # Check coding status
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
        
        log(f"  {len(agg):,} patients with MedDRA coding", "SUCCESS")
        return agg
    
    def aggregate_coding_whodrug(self) -> pd.DataFrame:
        """Aggregate WHODrug coding to patient level."""
        log("Aggregating coding_whodrug...")
        
        try:
            df = pd.read_sql("SELECT * FROM raw_coding_whodrug", self.db.engine)
        except:
            return pd.DataFrame(columns=['patient_key'])
        
        if df.empty:
            return pd.DataFrame(columns=['patient_key'])
        
        df['study_id'] = df['_study_id'].apply(standardize_study_id)
        
        subject_col = get_column(df, ['subject', 'subject_id', 'patient_id', 'subjectname'])
        
        df['subject_id'] = df[subject_col].apply(standardize_subject_id) if subject_col else None
        
        df = df[df['subject_id'].notna()].copy()
        df['patient_key'] = df.apply(lambda r: create_patient_key(r['study_id'], 'Unknown', r['subject_id']), axis=1)
        
        # Check coding status
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
        
        log(f"  {len(agg):,} patients with WHODrug coding", "SUCCESS")
        return agg


# ============================================
# MAIN CLEANING PIPELINE
# ============================================

def run_step2_cleaning():
    """Run Step 2: Cleaning & Standardization."""
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("   TRIALPULSE NEXUS - STEP 2: PLATINUM CLEANING")
    print("   Industry-Grade Data Cleaning & Standardization")
    print("=" * 70 + "\n")
    
    db = get_db_manager()
    
    # =====================================
    # Phase 1: Clean CPID (Primary Table)
    # =====================================
    log("Loading raw_cpid_edc_metrics from PostgreSQL...")
    cpid_raw = pd.read_sql("SELECT * FROM raw_cpid_edc_metrics", db.engine)
    log(f"Loaded {len(cpid_raw):,} rows")
    
    cleaner = CPIDCleaner()
    cpid_clean = cleaner.clean(cpid_raw)
    
    # =====================================
    # Phase 2: Aggregate Detail Tables
    # =====================================
    log("\n" + "=" * 60)
    log("AGGREGATING DETAIL TABLES TO PATIENT LEVEL")
    log("=" * 60)
    
    aggregator = DetailAggregator(db)
    
    agg_visit = aggregator.aggregate_visit_projection()
    agg_pages = aggregator.aggregate_missing_pages()
    agg_lab = aggregator.aggregate_missing_lab()
    agg_inactivated = aggregator.aggregate_inactivated()
    agg_edrr = aggregator.aggregate_edrr()
    agg_sae_dm = aggregator.aggregate_sae_dm()
    agg_sae_safety = aggregator.aggregate_sae_safety()
    agg_meddra = aggregator.aggregate_coding_meddra()
    agg_whodrug = aggregator.aggregate_coding_whodrug()
    
    # =====================================
    # Phase 3: Save Cleaned Data
    # =====================================
    log("\n" + "=" * 60)
    log("SAVING CLEANED DATA TO POSTGRESQL")
    log("=" * 60)
    
    # Save CPID
    with db.engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS cleaned_cpid_edc_metrics CASCADE"))
    cpid_clean.to_sql('cleaned_cpid_edc_metrics', db.engine, if_exists='replace', index=False, chunksize=500)
    log(f"  cleaned_cpid_edc_metrics: {len(cpid_clean):,} rows", "SUCCESS")
    
    # Save aggregations
    agg_tables = [
        ('agg_visit_projection', agg_visit),
        ('agg_missing_pages', agg_pages),
        ('agg_missing_lab_ranges', agg_lab),
        ('agg_inactivated_forms', agg_inactivated),
        ('agg_compiled_edrr', agg_edrr),
        ('agg_sae_dashboard_dm', agg_sae_dm),
        ('agg_sae_dashboard_safety', agg_sae_safety),
        ('agg_coding_meddra', agg_meddra),
        ('agg_coding_whodrug', agg_whodrug),
    ]
    
    for table_name, agg_df in agg_tables:
        if not agg_df.empty:
            with db.engine.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
            agg_df.to_sql(table_name, db.engine, if_exists='replace', index=False, chunksize=500)
            log(f"  {table_name}: {len(agg_df):,} rows", "SUCCESS")
    
    # =====================================
    # Summary
    # =====================================
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("   STEP 2 COMPLETE: CLEANING & STANDARDIZATION")
    print("=" * 70)
    print(f"   CPID Cleaned: {len(cpid_clean):,} patients")
    print(f"   Unique Patients: {cpid_clean['patient_key'].nunique():,}")
    print(f"   Unique Studies: {cpid_clean['study_id'].nunique()}")
    print(f"   Unique Sites: {cpid_clean['site_id'].nunique()}")
    print(f"   Time: {elapsed:.1f} seconds")
    print("=" * 70 + "\n")
    
    return cpid_clean, aggregator


if __name__ == "__main__":
    run_step2_cleaning()
