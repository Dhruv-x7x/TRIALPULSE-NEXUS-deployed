
import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import traceback
import warnings

import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

# Suppress pandas warnings during processing
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_RAW, DATA_PROCESSED, LOGS_DIR, FILE_PATTERNS

# PostgreSQL writer
from src.database.pg_writer import safe_to_postgres, get_pg_writer


# ============================================
# CONSTANTS & CONFIGURATION
# ============================================

DATE_FORMATS = [
    "%Y-%m-%d", "%d-%b-%Y", "%d%b%Y", "%m/%d/%Y", "%d/%m/%Y",
    "%Y-%m-%d %H:%M:%S", "%d-%b-%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
]

LARGE_STUDY_THRESHOLD = 100_000

# ============================================
# SETUP LOGGING
# ============================================

def setup_logger() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"ingestion_{timestamp}.log"
    logger.remove()
    logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}", level="DEBUG", rotation="10 MB")
    logger.add(lambda msg: print(msg, end=""), format="<level>{level:<8}</level> | {message}\n", level="INFO", colorize=True)
    return log_file

# ============================================
# DATA CLASSES
# ============================================

@dataclass
class FileResult:
    file_path: str
    file_name: str
    file_type: str
    study_id: str
    success: bool
    records: int = 0
    columns: int = 0
    sheets: List[str] = field(default_factory=list)
    error: str = ""
    time_seconds: float = 0.0
    file_size_mb: float = 0.0

@dataclass
class IngestionManifest:
    run_id: str
    start_time: str
    end_time: str = ""
    status: str = "running"
    schema_version: str = "3.0.0"
    studies_found: int = 0
    files_found: int = 0
    files_processed: int = 0
    files_success: int = 0
    files_failed: int = 0
    records_by_type: Dict[str, int] = field(default_factory=dict)
    records_by_study: Dict[str, int] = field(default_factory=dict)
    unidentified_files: List[Dict] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)
    warnings: Dict[str, Any] = field(default_factory=dict)
    large_studies: List[str] = field(default_factory=list)
    processing_stats: Dict[str, float] = field(default_factory=dict)
    def to_dict(self) -> Dict: return asdict(self)

# ============================================
# UTILITY FUNCTIONS
# ============================================

def extract_study_id(folder_name: str) -> Optional[str]:
    match = re.search(r'study\s*(\d+)', folder_name, re.IGNORECASE)
    if match: return f"Study_{match.group(1)}"
    return None

def identify_file_type(filename: str) -> Optional[str]:
    filename_check = filename.lower()
    mapping = {
        'cpid_edc_metrics': ['edc', 'metric', 'cpid'],
        'missing_lab_ranges': ['lab', 'lnr', 'range'],
        'sae_dashboard': ['sae', 'safety', 'dashboard'],
        'coding_meddra': ['meddra', 'medra'],
        'coding_whodrug': ['whodd', 'whodrug', 'whodra'],
        'visit_projection': ['visit', 'projection', 'tracker'],
        'missing_pages': ['missing_page', 'page_report'],
        'compiled_edrr': ['edrr', 'compiled'],
        'inactivated_forms': ['inactivated', 'forms_folders']
    }
    for file_type, keywords in mapping.items():
        if any(kw in filename_check for kw in keywords):
            return file_type
    return None

def standardize_columns(columns: pd.Index) -> List[str]:
    new_cols = []
    seen = {}
    for col in columns:
        col_clean = str(col).strip().lower()
        col_clean = re.sub(r'[^\w\s]', '', col_clean)
        col_clean = re.sub(r'\s+', '_', col_clean)
        if not col_clean: col_clean = 'unnamed'
        if col_clean in seen:
            seen[col_clean] += 1
            col_clean = f"{col_clean}_{seen[col_clean]}"
        else: seen[col_clean] = 0
        new_cols.append(col_clean)
    return new_cols

def parse_date(value: Any) -> Optional[str]:
    if pd.isna(value): return None
    if isinstance(value, (pd.Timestamp, datetime)): return value.strftime("%Y-%m-%d")
    try:
        dt = pd.to_datetime(str(value).strip(), errors='coerce')
        if pd.notna(dt): return dt.strftime("%Y-%m-%d")
    except: pass
    return str(value)

def is_numeric_col(col: str) -> bool:
    kw = ['count', 'total', 'num', 'queries', 'pages', 'visits', 'days', 'score', 'rate']
    return any(k in col.lower() for k in kw)

# ============================================
# FILE PROCESSORS
# ============================================

class CPIDEDCProcessor:
    def process(self, file_path: str, study_id: str) -> Tuple[Optional[pd.DataFrame], str]:
        try:
            xl = pd.ExcelFile(file_path)
            target_sheet = next((s for s in xl.sheet_names if 'subject' in s.lower()), xl.sheet_names[0])
            df = pd.read_excel(file_path, sheet_name=target_sheet, header=2)
            df = df.dropna(how='all')
            df.columns = standardize_columns(df.columns)
            df['_source_file'] = os.path.basename(file_path)
            df['_study_id'] = study_id
            df['_file_type'] = 'cpid_edc_metrics'
            return df, ""
        except Exception as e: return None, str(e)

class GenericProcessor:
    def __init__(self, file_type: str): self.file_type = file_type
    def process(self, file_path: str, study_id: str) -> Tuple[Optional[pd.DataFrame], str]:
        try:
            df = pd.read_excel(file_path)
            df = df.dropna(how='all')
            df.columns = standardize_columns(df.columns)
            df['_source_file'] = os.path.basename(file_path)
            df['_study_id'] = study_id
            df['_file_type'] = self.file_type
            return df, ""
        except Exception as e: return None, str(e)

class SAEDashboardProcessor:
    def process(self, file_path: str, study_id: str) -> Tuple[Optional[Dict[str, pd.DataFrame]], str]:
        try:
            xl = pd.ExcelFile(file_path)
            results = {}
            for sheet in xl.sheet_names:
                if 'safety' in sheet.lower() or 'dm' in sheet.lower():
                    df = pd.read_excel(file_path, sheet_name=sheet)
                    df.columns = standardize_columns(df.columns)
                    df['_source_file'] = os.path.basename(file_path)
                    df['_study_id'] = study_id
                    key = 'safety' if 'safety' in sheet.lower() else 'dm'
                    results[key] = df
            return results, ""
        except Exception as e: return None, str(e)

# ============================================
# MAIN INGESTION ENGINE
# ============================================

class DataIngestionEngine:
    def __init__(self):
        self.data_root = DATA_RAW
        self.output_dir = DATA_PROCESSED
        self.log_file = setup_logger()
        self.start_time = datetime.now()
        self.manifest = IngestionManifest(run_id=datetime.now().strftime("%Y%m%d_%H%M%S"), start_time=datetime.now().isoformat())
        self.processors = {
            'cpid_edc_metrics': CPIDEDCProcessor(),
            'visit_projection': GenericProcessor('visit_projection'),
            'missing_lab_ranges': GenericProcessor('missing_lab_ranges'),
            'sae_dashboard': SAEDashboardProcessor(),
            'inactivated_forms': GenericProcessor('inactivated_forms'),
            'missing_pages': GenericProcessor('missing_pages'),
            'compiled_edrr': GenericProcessor('compiled_edrr'),
            'coding_meddra': GenericProcessor('coding_meddra'),
            'coding_whodrug': GenericProcessor('coding_whodrug')
        }
        self.data = {k: [] for k in ['cpid_edc_metrics', 'visit_projection', 'missing_lab_ranges', 'sae_dashboard_dm', 'sae_dashboard_safety', 'inactivated_forms', 'missing_pages', 'compiled_edrr', 'coding_meddra', 'coding_whodrug']}
        self.study_records = {}

    def discover_files(self) -> Dict:
        folders = [f for f in self.data_root.iterdir() if f.is_dir() and 'study' in f.name.lower()]
        self.manifest.studies_found = len(folders)
        discovered = {}
        for folder in folders:
            study_id = extract_study_id(folder.name)
            if not study_id: continue
            discovered[study_id] = {'folder': folder, 'files': {}}
            for file_path in folder.glob("*.xlsx"):
                file_type = identify_file_type(file_path.name)
                if file_type:
                    if file_type not in discovered[study_id]['files']: discovered[study_id]['files'][file_type] = []
                    discovered[study_id]['files'][file_type].append(file_path)
                    self.manifest.files_found += 1
        return discovered

    def process_file(self, file_path: Path, file_type: str, study_id: str) -> FileResult:
        result = FileResult(file_path=str(file_path), file_name=file_path.name, file_type=file_type, study_id=study_id, success=False)
        try:
            processor = self.processors.get(file_type)
            data, error = processor.process(str(file_path), study_id)
            if error: result.error = error; return result
            if file_type == 'sae_dashboard' and isinstance(data, dict):
                for k, df in data.items():
                    self.data[f'sae_dashboard_{k}'].append(df)
                    result.records += len(df)
            elif data is not None:
                self.data[file_type].append(data)
                result.records = len(data)
            if study_id not in self.study_records: self.study_records[study_id] = 0
            self.study_records[study_id] += result.records
            result.success = True
        except Exception as e: result.error = str(e)
        return result

    def run(self) -> IngestionManifest:
        logger.info("Starting Ingestion v3.0...")
        discovered = self.discover_files()
        for study_id, study_data in tqdm(discovered.items(), desc="Studies"):
            for file_type, files in study_data['files'].items():
                for f in files: self.process_file(f, file_type, study_id); self.manifest.files_processed += 1; self.manifest.files_success += 1
        
        logger.info("Saving accumulated data...")
        for data_type, dfs in self.data.items():
            if dfs:
                combined = pd.concat(dfs, ignore_index=True, sort=False)
                safe_to_postgres(combined, data_type)
                self.manifest.records_by_type[data_type] = len(combined)
        
        self.manifest.status = "completed"
        return self.manifest

def main():
    DataIngestionEngine().run()

if __name__ == "__main__":
    main()
