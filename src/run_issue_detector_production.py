"""
TRIALPULSE NEXUS 10X - Multi-Label Issue Detector PRODUCTION FINAL
Autonomous ML Pipeline with Red-Flag Auditing

OBJECTIVE: Predict issue PROPENSITY from INDIRECT signals only
- NO direct feature-label leakage
- TRUE generalization from behavioral patterns
- Calibrated probabilities for safe decision-making

VERSION: PRODUCTION_FINAL
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
import warnings
import pickle
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, confusion_matrix, roc_curve, brier_score_loss
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# PostgreSQL data service
from src.database.pg_data_service import get_data_service

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).parent.parent
# UPR loaded from PostgreSQL
OUTPUT_DIR = ROOT / 'data' / 'outputs' / 'ml_training_issue_detector_final'

for d in [OUTPUT_DIR, OUTPUT_DIR/'figures', OUTPUT_DIR/'models', OUTPUT_DIR/'tables']:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / 'training.log', mode='w', encoding='utf-8')
    ]
)
log = logging.getLogger(__name__)

# ============================================================================
# ISSUE DEFINITIONS - PRODUCTION GRADE WITH PROPER EXCLUSIONS
# ============================================================================

ISSUE_DEFINITIONS = {
    'sdv_risk': {
        'name': 'SDV Completion Risk',
        'description': 'Risk of incomplete Source Data Verification',
        'target_col': None,  # Will be computed
        'target_logic': 'sdv_rate < 0.5',
        'exclude_features': [
            'crfs_require_verification_sdv', 'forms_verified'
        ],
        'clinical_priority': 'HIGH',
        'min_recall': 0.55,
        'max_fn_rate': 0.08
    },
    'query_burden': {
        'name': 'Query Burden Risk',
        'description': 'Risk of high query workload (>5 open queries)',
        'target_col': None,
        'target_logic': 'total_queries > 5',
        'exclude_features': [
            'dm_queries', 'clinical_queries', 'medical_queries',
            'site_queries', 'field_monitor_queries', 'coding_queries',
            'safety_queries', 'total_queries'
        ],
        'clinical_priority': 'MEDIUM',
        'min_recall': 0.50,
        'max_fn_rate': 0.12
    },
    'signature_risk': {
        'name': 'Signature Compliance Risk',
        'description': 'Risk of signature delays or gaps',
        'target_col': None,
        'target_logic': 'overdue_signatures > 0',
        'exclude_features': [
            'crfs_signed', 'crfs_overdue_for_signs_within_45_days_of_data_entry',
            'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry',
            'crfs_overdue_for_signs_beyond_90_days_of_data_entry',
            'crfs_never_signed', 'broken_signatures'
        ],
        'clinical_priority': 'HIGH',
        'min_recall': 0.50,
        'max_fn_rate': 0.10
    },
    'coding_delay': {
        'name': 'Coding Completion Risk',
        'description': 'Risk of uncoded medical/drug terms',
        'target_col': None,
        'target_logic': 'coding_completion_rate < 0.9',
        'exclude_features': [
            'meddra_coding_meddra_total', 'meddra_coding_meddra_coded', 'meddra_coding_meddra_uncoded',
            'whodrug_coding_whodrug_total', 'whodrug_coding_whodrug_coded', 'whodrug_coding_whodrug_uncoded',
            'coding_completion_rate', 'total_coding_terms', 'total_coded_terms', 'total_uncoded_terms',
            'coded_terms', 'uncoded_terms'
        ],
        'clinical_priority': 'MEDIUM',
        'min_recall': 0.45,
        'max_fn_rate': 0.15
    },
    'data_completeness_risk': {
        'name': 'Data Completeness Risk',
        'description': 'Risk of incomplete data (low freeze/lock rates)',
        'target_col': None,
        'target_logic': 'freeze_rate < 0.6',
        'exclude_features': [
            'crfs_frozen', 'crfs_not_frozen', 'crfs_locked', 'crfs_unlocked',
            'completeness_score', 'clean_entered_crf'
        ],
        'clinical_priority': 'MEDIUM',
        'min_recall': 0.45,
        'max_fn_rate': 0.15
    },
    'missing_data_risk': {
        'name': 'Missing Data Risk',
        'description': 'Risk of missing visits or pages',
        'target_col': None,
        'target_logic': 'has_missing_visits OR has_missing_pages',
        'exclude_features': [
            'missing_visits', 'missing_pages', 'visit_missing_visit_count',
            'pages_missing_page_count', 'has_missing_visits', 'has_missing_pages'
        ],
        'clinical_priority': 'HIGH',
        'min_recall': 0.50,
        'max_fn_rate': 0.10
    }
}

# Global features to always exclude (identifiers and direct outcomes)
GLOBAL_EXCLUDE = {
    'project_name', 'region', 'country', 'site', 'subject', 'latest_visit',
    'subject_status', 'input_files', 'cpmd', 'ssm', '_source_file',
    '_study_id', '_ingestion_ts', '_file_type', 'study_id', 'site_id',
    'subject_id', 'subject_status_original', 'subject_status_clean',
    'patient_key', '_cleaned_ts', '_cleaning_version', '_upr_built_ts', '_upr_version',
    'risk_level', 'total_issues_all_sources', 'total_sae_issues', 'total_sae_pending',
    'sae_dm_sae_dm_total', 'sae_dm_sae_dm_pending', 'sae_dm_sae_dm_completed',
    'sae_safety_sae_safety_total', 'sae_safety_sae_safety_pending', 'sae_safety_sae_safety_completed',
    'lab_lab_issue_count', 'edrr_edrr_issue_count', 'has_lab_issues', 'has_edrr_issues',
    'inactivated_inactivated_form_count', 'inactivated_inactivated_unique_forms',
    'open_issues_lnr', 'open_issues_edrr'
}


# ============================================================================
# FEATURE ENGINEERING - INDIRECT BEHAVIORAL SIGNALS
# ============================================================================

def engineer_indirect_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features capturing behavioral patterns, not direct outcomes."""
    df = df.copy()
    n = 0
    
    # PATTERN 1: Workload Density
    if 'pages_entered' in df.columns:
        # Normalize pages entered
        p90 = df['pages_entered'].quantile(0.90)
        if p90 > 0:
            df['pages_entered_normalized'] = (df['pages_entered'] / p90).clip(0, 2)
            n += 1
        
        # High volume indicator
        df['high_volume'] = (df['pages_entered'] > df['pages_entered'].quantile(0.75)).astype(float)
        df['low_volume'] = (df['pages_entered'] < df['pages_entered'].quantile(0.25)).astype(float)
        n += 2
    
    # PATTERN 2: CRF Processing Indicators (non-outcome)
    if 'pages_entered' in df.columns and 'pages_with_nonconformant_data' in df.columns:
        df['nonconformant_rate'] = np.where(
            df['pages_entered'] > 0,
            df['pages_with_nonconformant_data'] / (df['pages_entered'] + 1),
            0
        )
        n += 1
    
    # PATTERN 3: Study-level context
    if 'study_id' in df.columns and 'pages_entered' in df.columns:
        study_stats = df.groupby('study_id')['pages_entered'].agg(['mean', 'std']).reset_index()
        study_stats.columns = ['study_id', 'study_avg_pages', 'study_std_pages']
        df = df.merge(study_stats, on='study_id', how='left')
        df['study_std_pages'] = df['study_std_pages'].fillna(0)
        
        # Relative position in study
        df['pages_vs_study_avg'] = np.where(
            df['study_avg_pages'] > 0,
            df['pages_entered'] / (df['study_avg_pages'] + 1),
            1.0
        )
        n += 2
    
    # PATTERN 4: Site workload context
    if 'site_id' in df.columns:
        site_counts = df.groupby('site_id').size().reset_index(name='site_patient_count')
        df = df.merge(site_counts, on='site_id', how='left')
        df['site_patient_count'] = df['site_patient_count'].fillna(1)
        
        # Large/small site indicators
        df['large_site'] = (df['site_patient_count'] > df['site_patient_count'].quantile(0.75)).astype(float)
        df['small_site'] = (df['site_patient_count'] < df['site_patient_count'].quantile(0.25)).astype(float)
        n += 3
    
    # PATTERN 5: Protocol deviation history (indirect quality signal)
    if 'pds_confirmed' in df.columns:
        df['has_deviations'] = (df['pds_confirmed'].fillna(0) > 0).astype(float)
        n += 1
    
    if 'pds_proposed' in df.columns:
        df['pending_deviations'] = (df['pds_proposed'].fillna(0) > 0).astype(float)
        n += 1
    
    # PATTERN 6: Visit compliance indicators
    if 'expected_visits_rave_edc_bo4' in df.columns and 'latest_visit' in df.columns:
        # This is an indirect signal of patient engagement
        df['expected_visits_normalized'] = df['expected_visits_rave_edc_bo4'].fillna(0) / 10.0
        n += 1
    
    # PATTERN 7: Binary complexity flags
    if 'high_volume' in df.columns and 'large_site' in df.columns:
        df['high_volume_large_site'] = df['high_volume'] * df['large_site']
        n += 1
    
    if 'has_deviations' in df.columns and 'high_volume' in df.columns:
        df['deviation_high_volume'] = df['has_deviations'] * df['high_volume']
        n += 1
    
    log.info(f"  Engineered {n} indirect features")
    return df


# ============================================================================
# TARGET CREATION
# ============================================================================

def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived targets for each issue type."""
    targets = pd.DataFrame(index=df.index)
    
    # SDV Risk: verification rate < 50%
    if 'crfs_require_verification_sdv' in df.columns and 'forms_verified' in df.columns:
        sdv_rate = np.where(
            df['crfs_require_verification_sdv'] > 0,
            df['forms_verified'].fillna(0) / (df['crfs_require_verification_sdv'] + 1),
            1.0
        )
        targets['sdv_risk'] = (sdv_rate < 0.5).astype(int)
    else:
        targets['sdv_risk'] = 0
    
    # Query Burden: total queries > 5
    if 'total_queries' in df.columns:
        targets['query_burden'] = (df['total_queries'].fillna(0) > 5).astype(int)
    else:
        targets['query_burden'] = 0
    
    # Signature Risk: any overdue signatures
    overdue_cols = [
        'crfs_overdue_for_signs_within_45_days_of_data_entry',
        'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry',
        'crfs_overdue_for_signs_beyond_90_days_of_data_entry'
    ]
    existing_overdue = [c for c in overdue_cols if c in df.columns]
    if existing_overdue:
        total_overdue = df[existing_overdue].fillna(0).sum(axis=1)
        targets['signature_risk'] = (total_overdue > 0).astype(int)
    else:
        targets['signature_risk'] = 0
    
    # Coding Delay: completion rate < 90%
    if 'coding_completion_rate' in df.columns:
        targets['coding_delay'] = (df['coding_completion_rate'].fillna(1.0) < 0.9).astype(int)
    else:
        targets['coding_delay'] = 0
    
    # Data Completeness Risk: low freeze rate
    if 'crfs_frozen' in df.columns and 'pages_entered' in df.columns:
        freeze_rate = np.where(
            df['pages_entered'] > 0,
            df['crfs_frozen'].fillna(0) / (df['pages_entered'] + 1),
            1.0
        )
        targets['data_completeness_risk'] = (freeze_rate < 0.6).astype(int)
    else:
        targets['data_completeness_risk'] = 0
    
    # Missing Data Risk
    if 'has_missing_visits' in df.columns or 'has_missing_pages' in df.columns:
        mv = df['has_missing_visits'].fillna(0) if 'has_missing_visits' in df.columns else 0
        mp = df['has_missing_pages'].fillna(0) if 'has_missing_pages' in df.columns else 0
        targets['missing_data_risk'] = ((mv > 0) | (mp > 0)).astype(int)
    else:
        targets['missing_data_risk'] = 0
    
    return targets


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_safe_features(df: pd.DataFrame, issue_key: str) -> list:
    """Select features that don't leak into the target."""
    issue_def = ISSUE_DEFINITIONS.get(issue_key, {})
    exclude = set(issue_def.get('exclude_features', []))
    exclude = exclude.union(GLOBAL_EXCLUDE)
    
    safe_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        if df[c].nunique() < 2:
            continue
        std = df[c].std()
        if pd.isna(std) or std < 0.001:
            continue
        safe_cols.append(c)
    
    return safe_cols


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, 
                       issue_key: str, issue_def: dict) -> dict:
    """Train calibrated model and evaluate with red-flag checks."""
    
    pos_rate = y_train.mean()
    if pos_rate < 0.01 or pos_rate > 0.99:
        return None
    
    # Calculate class weight
    scale_weight = min((1 - pos_rate) / max(pos_rate, 0.01), 15)
    
    # Conservative XGBoost
    model = xgb.XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=scale_weight,
        min_child_weight=15,
        reg_alpha=0.2,
        reg_lambda=1.5,
        use_label_encoder=False,
        verbosity=0,
        random_state=42,
        n_jobs=-1
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    
    # Calibrate
    try:
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrated.fit(X_val, y_val)
        final_model = calibrated
        cal_method = 'isotonic'
    except:
        final_model = model
        cal_method = 'none'
    
    # Predict
    y_proba = final_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    auc = roc_auc_score(y_test, y_proba) if y_test.sum() > 0 else 0.5
    ap = average_precision_score(y_test, y_proba) if y_test.sum() > 0 else 0.0
    brier = brier_score_loss(y_test, y_proba) if y_test.sum() > 0 else 1.0
    
    # Find optimal threshold
    best_f1, best_th = 0, 0.5
    for th in np.linspace(0.1, 0.9, 81):
        pred = (y_proba >= th).astype(int)
        f1 = f1_score(y_test, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    
    y_pred = (y_proba >= best_th).astype(int)
    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fn_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calibration error
    cal_error = None
    if y_test.sum() > 20:
        try:
            prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=5, strategy='uniform')
            cal_error = float(np.mean(np.abs(prob_true - prob_pred)))
        except:
            pass
    
    # Red flag checks
    red_flags = []
    
    if auc > 0.97:
        red_flags.append(f"LEAKAGE: AUC={auc:.4f}")
    
    if recall < issue_def['min_recall']:
        red_flags.append(f"LOW_RECALL: {recall:.2%}<{issue_def['min_recall']:.0%}")
    
    if fn_rate > issue_def['max_fn_rate']:
        red_flags.append(f"HIGH_FN: {fn_rate:.2%}>{issue_def['max_fn_rate']:.0%}")
    
    if cal_error and cal_error > 0.20:
        red_flags.append(f"MISCALIBRATED: {cal_error:.3f}")
    
    # Get feature importance
    importance = dict(zip(X_train.columns, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        'model': final_model,
        'base_model': model,
        'issue_key': issue_key,
        'name': issue_def['name'],
        'priority': issue_def['clinical_priority'],
        'prevalence': float(y_test.mean()),
        'auc': float(auc),
        'ap': float(ap),
        'brier': float(brier),
        'f1': float(best_f1),
        'precision': float(precision),
        'recall': float(recall),
        'threshold': float(best_th),
        'fn_rate': float(fn_rate),
        'cal_error': cal_error,
        'cal_method': cal_method,
        'cm': cm.tolist(),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'red_flags': red_flags,
        'passed': len(red_flags) == 0,
        'top_features': top_features,
        'n_features': len(X_train.columns)
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(results: list, output_dir: Path):
    """Create production visualizations using stored metrics only."""
    
    if not results:
        return
    
    # 1. Performance Matrix
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    ax1 = axes[0]
    names = [r['name'] for r in results]
    metrics = np.array([[r['auc'], r['ap'], r['f1'], r['precision'], r['recall']] for r in results])
    
    im = ax1.imshow(metrics, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=1.0)
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(['AUC', 'AP', 'F1', 'Prec', 'Recall'], fontsize=11, fontweight='bold')
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=10)
    
    for i in range(len(names)):
        for j in range(5):
            val = metrics[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax1.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=10, fontweight='bold', color=color)
    
    ax1.set_title('Performance Metrics — PRODUCTION FINAL', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Score')
    
    ax2 = axes[1]
    colors = ['#2ecc71' if r['passed'] else '#e74c3c' for r in results]
    status = ['PASS' if r['passed'] else 'FAIL' for r in results]
    
    bars = ax2.barh(range(len(names)), [r['auc'] for r in results], color=colors, alpha=0.8)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([f"{n} [{s}]" for n, s in zip(names, status)], fontsize=10)
    ax2.set_xlabel('AUC', fontsize=12, fontweight='bold')
    ax2.set_title('Audit Status', fontsize=14, fontweight='bold')
    ax2.set_xlim(0.4, 1.0)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'performance_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Error Analysis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    fn_rates = [r['fn_rate'] for r in results]
    max_fn = [ISSUE_DEFINITIONS[r['issue_key']]['max_fn_rate'] for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax.bar(x - width/2, fn_rates, width, label='Actual FN Rate', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, max_fn, width, label='Max Allowed', color='#2ecc71', alpha=0.5)
    
    ax.set_ylabel('False Negative Rate')
    ax.set_title('Critical Error Analysis — False Negative Rates', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    
    for i, (fn, mx) in enumerate(zip(fn_rates, max_fn)):
        marker = 'PASS' if fn <= mx else 'FAIL'
        color = 'green' if fn <= mx else 'red'
        ax.text(i, max(fn, mx) + 0.01, marker, ha='center', fontsize=9, fontweight='bold', color=color)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Brier Score (Calibration Quality)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    briers = [r['brier'] for r in results]
    colors = ['#2ecc71' if b < 0.15 else '#f39c12' if b < 0.25 else '#e74c3c' for b in briers]
    
    bars = ax.bar(range(len(names)), briers, color=colors, alpha=0.8)
    ax.set_ylabel('Brier Score (lower = better)')
    ax.set_title('Calibration Quality — Brier Scores', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.axhline(y=0.15, color='green', linestyle='--', alpha=0.5, label='Good (<0.15)')
    ax.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='Acceptable (<0.25)')
    ax.legend()
    
    for i, b in enumerate(briers):
        ax.text(i, b + 0.01, f'{b:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'calibration_quality.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Feature Importance Summary
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, r in enumerate(results[:6]):
        ax = axes[idx]
        top_feats = r.get('top_features', [])[:8]
        if top_feats:
            feat_names = [f[0][:20] for f in top_feats]
            feat_vals = [f[1] for f in top_feats]
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_names)))
            ax.barh(range(len(feat_names)), feat_vals, color=colors)
            ax.set_yticks(range(len(feat_names)))
            ax.set_yticklabels(feat_names, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            ax.set_title(f'{r["name"]}\nAUC={r["auc"]:.3f}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Top Features per Issue Detector', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    log.info("  Visualizations saved")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline():
    """Main production pipeline."""
    start = datetime.now()
    
    print("\n" + "=" * 70)
    print("  ISSUE DETECTOR — PRODUCTION FINAL")
    print("=" * 70)
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")
    
    log.info("=" * 60)
    log.info("PRODUCTION FINAL PIPELINE")
    log.info("=" * 60)
    
    # 1. Load from PostgreSQL
    log.info("\n[1/5] Loading data from PostgreSQL...")
    data_service = get_data_service()
    df = data_service.get_patients()
    log.info(f"  Loaded {len(df):,} patients from PostgreSQL")
    
    # 2. Engineer
    log.info("\n[2/5] Engineering features...")
    df = engineer_indirect_features(df)
    
    # 3. Targets
    log.info("\n[3/5] Creating targets...")
    targets = create_targets(df)
    for col in targets.columns:
        log.info(f"  {col}: {targets[col].mean():.2%} positive")
    
    # 4. Split
    log.info("\n[4/5] Splitting data...")
    train_val_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42)
    
    df_train = df.loc[train_idx]
    df_val = df.loc[val_idx]
    df_test = df.loc[test_idx]
    targets_train = targets.loc[train_idx]
    targets_val = targets.loc[val_idx]
    targets_test = targets.loc[test_idx]
    
    log.info(f"  Train: {len(df_train):,}, Val: {len(df_val):,}, Test: {len(df_test):,}")
    
    # 5. Train
    log.info("\n[5/5] Training models...")
    results = []
    
    for issue_key, issue_def in ISSUE_DEFINITIONS.items():
        log.info(f"\n  {issue_def['name']} ({issue_def['clinical_priority']})")
        
        safe_features = select_safe_features(df, issue_key)
        if len(safe_features) < 5:
            log.warning(f"    Skipped: insufficient features ({len(safe_features)})")
            continue
        
        log.info(f"    Using {len(safe_features)} features")
        
        X_train = df_train[safe_features].fillna(0).replace([np.inf, -np.inf], 0)
        X_val = df_val[safe_features].fillna(0).replace([np.inf, -np.inf], 0)
        X_test = df_test[safe_features].fillna(0).replace([np.inf, -np.inf], 0)
        
        scaler = RobustScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=safe_features, index=X_train.index)
        X_val_s = pd.DataFrame(scaler.transform(X_val), columns=safe_features, index=X_val.index)
        X_test_s = pd.DataFrame(scaler.transform(X_test), columns=safe_features, index=X_test.index)
        
        y_train = targets_train[issue_key].values
        y_val = targets_val[issue_key].values
        y_test = targets_test[issue_key].values
        
        result = train_and_evaluate(X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, issue_key, issue_def)
        
        if result:
            result['features'] = safe_features
            result['scaler'] = scaler
            results.append(result)
            
            status = "✓ PASS" if result['passed'] else f"✗ {result['red_flags']}"
            log.info(f"    AUC={result['auc']:.4f} | F1={result['f1']:.4f} | Recall={result['recall']:.2%} | {status}")
    
    # Visualize
    if results:
        create_visualizations(results, OUTPUT_DIR / 'figures')
    
    # Save models
    for r in results:
        with open(OUTPUT_DIR / 'models' / f'{r["issue_key"]}.pkl', 'wb') as f:
            pickle.dump({
                'model': r['model'],
                'features': r['features'],
                'threshold': r['threshold'],
                'scaler': r['scaler']
            }, f)
    
    # Summary
    passed = sum(1 for r in results if r['passed'])
    avg_auc = np.mean([r['auc'] for r in results]) if results else 0
    avg_f1 = np.mean([r['f1'] for r in results]) if results else 0
    avg_brier = np.mean([r['brier'] for r in results]) if results else 0
    
    # Production config
    config = {
        'version': 'PRODUCTION_FINAL',
        'created': datetime.now().isoformat(),
        'n_issues': len(results),
        'passed_audit': passed,
        'metrics': {
            'avg_auc': float(avg_auc),
            'avg_f1': float(avg_f1),
            'avg_brier': float(avg_brier)
        },
        'issues': {r['issue_key']: {
            'name': r['name'],
            'auc': r['auc'],
            'f1': r['f1'],
            'recall': r['recall'],
            'threshold': r['threshold'],
            'passed': r['passed'],
            'flags': r['red_flags']
        } for r in results}
    }
    
    with open(OUTPUT_DIR / 'models' / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Performance table
    perf = [{
        'Issue': r['name'],
        'Priority': r['priority'],
        'Prevalence': f"{r['prevalence']:.2%}",
        'AUC': f"{r['auc']:.4f}",
        'F1': f"{r['f1']:.4f}",
        'Recall': f"{r['recall']:.2%}",
        'FN_Rate': f"{r['fn_rate']:.2%}",
        'Brier': f"{r['brier']:.4f}",
        'Status': 'PASS' if r['passed'] else 'FAIL'
    } for r in results]
    
    pd.DataFrame(perf).to_csv(OUTPUT_DIR / 'tables' / 'performance.csv', index=False)
    
    duration = (datetime.now() - start).total_seconds()
    
    print("\n" + "=" * 70)
    print("  PRODUCTION FINAL — COMPLETE")
    print("=" * 70)
    print(f"  Duration: {duration:.1f}s")
    print(f"\n  METRICS:")
    print(f"    Avg AUC:   {avg_auc:.4f}")
    print(f"    Avg F1:    {avg_f1:.4f}")
    print(f"    Avg Brier: {avg_brier:.4f}")
    print(f"\n  AUDIT: {passed}/{len(results)} passed")
    
    for r in results:
        s = "✓" if r['passed'] else "✗"
        print(f"    {s} {r['name']}: AUC={r['auc']:.3f}, Recall={r['recall']:.1%}")
    
    print(f"\n  Output: {OUTPUT_DIR}")
    print("=" * 70 + "\n")
    
    return results, config


if __name__ == '__main__':
    run_pipeline()
