"""
TRIALPULSE NEXUS 10X — RESOLUTION TIME PREDICTOR PRODUCTION v3
5-Star Quality — Enhanced Long-Tail Handling — Real-World Usable

IMPROVEMENTS OVER v2:
1. More robust long-tail handling with ensemble approach
2. Better uncertainty quantification
3. Comprehensive 8-visualization suite
4. Honest confidence intervals
5. Production deployment guidance

VERSION: PRODUCTION_v3
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
import os
import gc
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
except ImportError:
    print("XGBoost required")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / 'data' / 'raw'
OUTPUT_DIR = ROOT / 'data' / 'outputs' / 'resolution_time_PRODUCTION_v3'

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

# Quantiles for prediction intervals
QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

# Duration thresholds
SHORT_THRESHOLD = 7
MEDIUM_THRESHOLD = 30


# ============================================================================
# DATA LOADING
# ============================================================================

def load_query_data() -> pd.DataFrame:
    """Load query data from all studies."""
    all_queries = []
    
    for study_folder in os.listdir(RAW_DIR):
        folder_path = RAW_DIR / study_folder
        if not folder_path.is_dir():
            continue
        
        for f in os.listdir(folder_path):
            if 'EDC_Metrics' in f and f.endswith('.xlsx'):
                try:
                    path = folder_path / f
                    df = pd.read_excel(path, sheet_name='Query Report - Cumulative')
                    df['_source_study'] = study_folder
                    all_queries.append(df)
                except:
                    pass
    
    if not all_queries:
        return pd.DataFrame()
    
    return pd.concat(all_queries, ignore_index=True)


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare resolved and open issue datasets."""
    df['Query Open Date'] = pd.to_datetime(df['Query Open Date'], errors='coerce')
    df['Query Response Date'] = pd.to_datetime(df['Query Response Date'], errors='coerce')
    df['resolution_days'] = (df['Query Response Date'] - df['Query Open Date']).dt.days
    
    resolved_mask = (
        (df['Query Status'] == 'Answered') & 
        (df['resolution_days'].notna()) & 
        (df['resolution_days'] >= 0) &
        (df['resolution_days'] <= 365)  # Cap at 1 year for outliers
    )
    resolved = df[resolved_mask].copy()
    open_issues = df[df['Query Status'] == 'Open'].copy()
    
    return resolved, open_issues


# ============================================================================
# ENHANCED FEATURE ENGINEERING (35+ features)
# ============================================================================

def engineer_features_v3(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer comprehensive features for resolution time prediction."""
    features = pd.DataFrame(index=df.index)
    
    # === OWNER FEATURES ===
    if 'Action Owner' in df.columns:
        owner_str = df['Action Owner'].astype(str).str.lower()
        features['owner_dm'] = owner_str.str.contains('dm|data', na=False).astype(int)
        features['owner_site'] = owner_str.str.contains('site', na=False).astype(int)
        features['owner_cra'] = owner_str.str.contains('cra', na=False).astype(int)
        features['owner_safety'] = owner_str.str.contains('safety', na=False).astype(int)
        features['owner_medical'] = owner_str.str.contains('medical', na=False).astype(int)
        features['owner_sponsor'] = owner_str.str.contains('sponsor', na=False).astype(int)
    
    # === MARKING GROUP FEATURES ===
    if 'Marking Group Name' in df.columns:
        marking_str = df['Marking Group Name'].astype(str).str.lower()
        features['marking_dm'] = marking_str.str.contains('dm|data', na=False).astype(int)
        features['marking_clinical'] = marking_str.str.contains('clinical', na=False).astype(int)
        features['marking_safety'] = marking_str.str.contains('safety', na=False).astype(int)
        features['marking_medical'] = marking_str.str.contains('medical', na=False).astype(int)
    
    # === FORM COMPLEXITY FEATURES ===
    if 'Form' in df.columns:
        form_str = df['Form'].astype(str).str.lower()
        features['form_ae'] = form_str.str.contains('ae|adverse', na=False).astype(int)
        features['form_lab'] = form_str.str.contains('lab', na=False).astype(int)
        features['form_vital'] = form_str.str.contains('vital', na=False).astype(int)
        features['form_med'] = form_str.str.contains('med|drug|conmed', na=False).astype(int)
        features['form_ecg'] = form_str.str.contains('ecg', na=False).astype(int)
        features['form_efficacy'] = form_str.str.contains('efficacy|endpoint', na=False).astype(int)
        features['form_demo'] = form_str.str.contains('demo|demographics', na=False).astype(int)
        
        # Complexity indicator
        features['complex_form'] = (
            features.get('form_ae', 0) | 
            features.get('form_efficacy', 0) |
            features.get('owner_safety', 0) |
            features.get('owner_medical', 0)
        ).astype(int)
    
    # === TIME CONTEXT FEATURES ===
    if 'Query Open Date' in df.columns:
        open_date = pd.to_datetime(df['Query Open Date'], errors='coerce')
        features['open_day_of_week'] = open_date.dt.dayofweek.fillna(2)
        features['open_hour'] = open_date.dt.hour.fillna(12)
        features['open_month'] = open_date.dt.month.fillna(6)
        features['open_is_weekend'] = (features['open_day_of_week'] >= 5).astype(int)
        features['open_is_monday'] = (features['open_day_of_week'] == 0).astype(int)
        features['open_is_friday'] = (features['open_day_of_week'] == 4).astype(int)
        features['open_is_end_of_month'] = (open_date.dt.day >= 25).astype(int).fillna(0)
        features['open_is_start_of_month'] = (open_date.dt.day <= 5).astype(int).fillna(0)
        features['open_is_holiday_season'] = open_date.dt.month.isin([11, 12, 1]).astype(int).fillna(0)
        features['open_is_summer'] = open_date.dt.month.isin([6, 7, 8]).astype(int).fillna(0)
        features['open_is_q4'] = open_date.dt.quarter.eq(4).astype(int).fillna(0)
        features['open_is_q1'] = open_date.dt.quarter.eq(1).astype(int).fillna(0)
    
    # === SITE CONTEXT ===
    if 'Site Number' in df.columns:
        site_counts = df.groupby('Site Number').size().reset_index(name='site_query_count')
        df_site = df[['Site Number']].merge(site_counts, on='Site Number', how='left')
        features['site_query_count'] = df_site['site_query_count'].values
        
        p75 = features['site_query_count'].quantile(0.75)
        p25 = features['site_query_count'].quantile(0.25)
        p90 = features['site_query_count'].quantile(0.90)
        
        features['high_volume_site'] = (features['site_query_count'] > p75).astype(int)
        features['low_volume_site'] = (features['site_query_count'] < p25).astype(int)
        features['very_high_volume_site'] = (features['site_query_count'] > p90).astype(int)
        features['site_volume_norm'] = (features['site_query_count'] / max(p75, 1)).clip(0, 3)
    
    # === STUDY CONTEXT ===
    if 'Study' in df.columns:
        study_counts = df.groupby('Study').size().reset_index(name='study_query_count')
        df_study = df[['Study']].merge(study_counts, on='Study', how='left')
        features['study_query_count'] = df_study['study_query_count'].values
        features['large_study'] = (features['study_query_count'] > features['study_query_count'].median()).astype(int)
    
    # === COMBINED COMPLEXITY SCORE ===
    features['complexity_score'] = (
        features.get('complex_form', 0) * 2 +
        features.get('owner_safety', 0) * 3 +
        features.get('owner_medical', 0) * 2 +
        features.get('marking_safety', 0) * 2 +
        features.get('high_volume_site', 0) +
        features.get('open_is_holiday_season', 0) * 2 +
        features.get('open_is_q4', 0) +
        features.get('open_is_weekend', 0)
    )
    
    features['is_likely_slow'] = (features['complexity_score'] >= 4).astype(int)
    features['is_very_likely_slow'] = (features['complexity_score'] >= 6).astype(int)
    
    # Clean up
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
    
    return features


# ============================================================================
# STRATIFIED MODEL TRAINING
# ============================================================================

def classify_duration(days: np.ndarray) -> np.ndarray:
    """Classify resolution duration into categories."""
    categories = np.zeros(len(days), dtype=int)
    categories[days <= SHORT_THRESHOLD] = 0
    categories[(days > SHORT_THRESHOLD) & (days <= MEDIUM_THRESHOLD)] = 1
    categories[days > MEDIUM_THRESHOLD] = 2
    return categories


def train_quantile_model(X, y, quantile: float, is_specialist: bool = False):
    """Train quantile regression model."""
    params = {
        'n_estimators': 120 if is_specialist else 100,
        'max_depth': 5 if is_specialist else 4,
        'learning_rate': 0.03 if is_specialist else 0.05,
        'subsample': 0.6 if is_specialist else 0.7,
        'colsample_bytree': 0.6 if is_specialist else 0.7,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
        'objective': 'reg:quantileerror',
        'quantile_alpha': quantile,
        'verbosity': 0,
        'random_state': 42,
        'n_jobs': 4
    }
    
    model = xgb.XGBRegressor(**params)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    
    return model


def train_ensemble_models(X_train, y_train, X_val, y_val):
    """Train ensemble of models for robust prediction."""
    
    categories = classify_duration(y_train)
    models = {}
    
    # =========================================================================
    # MODEL GROUP 1: Standard Quantile Models (all data)
    # =========================================================================
    log.info("\n  Training standard quantile models...")
    models['standard'] = {}
    for q in QUANTILES:
        log.info(f"    Quantile {q:.2f}")
        models['standard'][q] = train_quantile_model(X_train, y_train, q)
    
    # =========================================================================
    # MODEL GROUP 2: Short-case Specialist (<=7 days)
    # =========================================================================
    short_mask = categories == 0
    n_short = short_mask.sum()
    
    if n_short >= 100:
        log.info(f"\n  Training SHORT specialist ({n_short} cases)...")
        X_short = X_train[short_mask]
        y_short = y_train[short_mask]
        
        models['short'] = {}
        for q in [0.1, 0.5, 0.9]:
            models['short'][q] = train_quantile_model(X_short, y_short, q, is_specialist=True)
    else:
        models['short'] = None
    
    # =========================================================================
    # MODEL GROUP 3: Long-case Specialist (>30 days)
    # =========================================================================
    long_mask = categories == 2
    n_long = long_mask.sum()
    
    if n_long >= 30:
        log.info(f"\n  Training LONG-TAIL specialist ({n_long} cases)...")
        X_long = X_train[long_mask]
        y_long = y_train[long_mask]
        
        models['long'] = {}
        for q in QUANTILES:
            models['long'][q] = train_quantile_model(X_long, y_long, q, is_specialist=True)
    else:
        log.warning(f"  Not enough long-tail cases ({n_long}) for specialist")
        models['long'] = None
    
    # =========================================================================
    # MODEL GROUP 4: Duration Classifier
    # =========================================================================
    log.info("\n  Training duration classifier...")
    
    # Slow-case classifier
    y_is_slow = (y_train > MEDIUM_THRESHOLD).astype(int)
    slow_rate = y_is_slow.mean()
    
    slow_classifier = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=max(1, (1 - slow_rate) / max(slow_rate, 0.01)),
        verbosity=0,
        random_state=42,
        n_jobs=4
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slow_classifier.fit(X_train, y_is_slow)
    
    models['slow_classifier'] = slow_classifier
    
    # Medium-case classifier (7-30 days)
    y_is_medium = ((y_train > SHORT_THRESHOLD) & (y_train <= MEDIUM_THRESHOLD)).astype(int)
    
    medium_classifier = xgb.XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        verbosity=0,
        random_state=42,
        n_jobs=4
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        medium_classifier.fit(X_train, y_is_medium)
    
    models['medium_classifier'] = medium_classifier
    
    return models


def predict_with_ensemble(X, models) -> Dict:
    """Make predictions using ensemble with uncertainty."""
    n = len(X)
    
    # Get slow-case probability
    slow_prob = models['slow_classifier'].predict_proba(X)[:, 1]
    medium_prob = models['medium_classifier'].predict_proba(X)[:, 1]
    
    # Standard predictions
    standard_preds = {q: models['standard'][q].predict(X) for q in QUANTILES}
    
    # Specialist predictions (if available)
    long_preds = {q: models['long'][q].predict(X) for q in QUANTILES} if models['long'] else standard_preds
    short_preds = {q: models['short'][q].predict(X) for q in [0.1, 0.5, 0.9]} if models['short'] else {q: standard_preds[q] for q in [0.1, 0.5, 0.9]}
    
    # Blend predictions based on probability
    final_preds = {}
    
    for q in QUANTILES:
        # Base prediction
        base = standard_preds[q]
        
        # For high slow probability, blend toward long specialist
        if q in long_preds:
            long_blend = slow_prob * long_preds[q] + (1 - slow_prob) * base
        else:
            long_blend = base
        
        # For low slow probability, blend toward short specialist
        if q in short_preds:
            short_weight = np.maximum(0, 1 - slow_prob - medium_prob)
            short_blend = short_weight * short_preds[q] + (1 - short_weight) * long_blend
        else:
            short_blend = long_blend
        
        final_preds[q] = short_blend
    
    # Apply minimum interval width based on uncertainty
    uncertainty = slow_prob * (1 - slow_prob)  # Highest at 0.5
    
    # Minimum interval: 8 days for short, 15 for medium, 25 for long
    min_width = 8 + slow_prob * 12 + uncertainty * 10
    
    actual_width = final_preds[0.9] - final_preds[0.1]
    needs_expansion = actual_width < min_width
    
    expansion = np.maximum(0, min_width - actual_width)
    final_preds[0.1] = np.where(needs_expansion, final_preds[0.1] - expansion * 0.3, final_preds[0.1])
    final_preds[0.9] = np.where(needs_expansion, final_preds[0.9] + expansion * 0.7, final_preds[0.9])
    
    # Ensure non-negative
    for q in final_preds:
        final_preds[q] = np.maximum(0, final_preds[q])
    
    # Confidence level
    confidence = np.where(
        slow_prob < 0.2, 'HIGH',
        np.where(slow_prob < 0.5, 'MEDIUM', 'LOW')
    )
    
    return {
        'predictions': final_preds,
        'slow_probability': slow_prob,
        'medium_probability': medium_prob,
        'is_likely_slow': (slow_prob > 0.5).astype(int),
        'confidence': confidence,
        'interval_80': (final_preds[0.1], final_preds[0.9])
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_comprehensive(y_true, results: Dict, categories: np.ndarray) -> Dict:
    """Comprehensive evaluation with stratification."""
    
    preds = results['predictions']
    median_pred = preds[0.5]
    lower_10 = preds[0.1]
    upper_90 = preds[0.9]
    lower_05 = preds[0.05]
    upper_95 = preds[0.95]
    
    # Overall metrics
    mae = mean_absolute_error(y_true, median_pred)
    rmse = np.sqrt(mean_squared_error(y_true, median_pred))
    
    in_80_interval = (y_true >= lower_10) & (y_true <= upper_90)
    in_90_interval = (y_true >= lower_05) & (y_true <= upper_95)
    
    coverage_80 = in_80_interval.mean()
    coverage_90 = in_90_interval.mean()
    
    interval_width_80 = (upper_90 - lower_10).mean()
    interval_width_90 = (upper_95 - lower_05).mean()
    
    # Slow-case detection accuracy
    slow_actual = (y_true > MEDIUM_THRESHOLD)
    slow_predicted = results['is_likely_slow'].astype(bool)
    slow_accuracy = (slow_actual == slow_predicted).mean()
    slow_recall = slow_actual[slow_predicted].mean() if slow_predicted.any() else 0
    
    metrics = {
        'overall': {
            'mae': float(mae),
            'rmse': float(rmse),
            'coverage_80': float(coverage_80),
            'coverage_90': float(coverage_90),
            'interval_width_80': float(interval_width_80),
            'interval_width_90': float(interval_width_90),
            'slow_detection_accuracy': float(slow_accuracy),
            'n': len(y_true)
        }
    }
    
    # Per-category metrics
    category_names = ['SHORT (≤7d)', 'MEDIUM (7-30d)', 'LONG (>30d)']
    for cat_idx, cat_name in enumerate(category_names):
        mask = categories == cat_idx
        n_cat = mask.sum()
        if n_cat > 0:
            metrics[cat_name] = {
                'mae': float(mean_absolute_error(y_true[mask], median_pred[mask])),
                'rmse': float(np.sqrt(mean_squared_error(y_true[mask], median_pred[mask]))),
                'coverage_80': float(in_80_interval[mask].mean()),
                'coverage_90': float(in_90_interval[mask].mean()),
                'interval_width_80': float((upper_90[mask] - lower_10[mask]).mean()),
                'n': int(n_cat),
                'pct': float(n_cat / len(y_true))
            }
    
    return metrics


def run_red_flags(metrics: Dict) -> List[str]:
    """Check for red flags indicating model issues."""
    flags = []
    
    overall = metrics['overall']
    
    # Suspiciously good
    if overall['mae'] < 0.5:
        flags.append('🔴 SUSPICIOUS: MAE < 0.5 days (too good)')
    
    if overall['coverage_80'] > 0.95:
        flags.append('⚠️ WARNING: Coverage > 95% (intervals too wide)')
    
    if overall['coverage_80'] < 0.65:
        flags.append('⚠️ WARNING: Coverage < 65% (intervals too narrow)')
    
    # Long-case issues
    if 'LONG (>30d)' in metrics:
        long_m = metrics['LONG (>30d)']
        if long_m['mae'] > 40:
            flags.append(f"⚠️ WARNING: Long-case MAE = {long_m['mae']:.1f} days")
        if long_m['coverage_80'] < 0.50:
            flags.append(f"⚠️ WARNING: Long-case coverage = {long_m['coverage_80']:.0%}")
        if long_m['n'] < 20:
            flags.append(f"⚠️ WARNING: Long-case sample small (n={long_m['n']})")
    
    return flags


# ============================================================================
# COMPREHENSIVE VISUALIZATIONS (8 charts)
# ============================================================================

def create_comprehensive_viz(y_test, results: Dict, categories: np.ndarray,
                              metrics: Dict, output_dir: Path):
    """Create 8 comprehensive visualizations."""
    
    preds = results['predictions']
    median_pred = preds[0.5]
    lower_10 = preds[0.1]
    upper_90 = preds[0.9]
    slow_prob = results['slow_probability']
    
    # =========================================================================
    # 1. PREDICTED VS ACTUAL (by category)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['green', 'orange', 'red']
    labels = ['Short (≤7d)', 'Medium (7-30d)', 'Long (>30d)']
    
    for cat_idx, (color, label) in enumerate(zip(colors, labels)):
        mask = categories == cat_idx
        if mask.any():
            ax.scatter(y_test[mask], median_pred[mask], alpha=0.4, s=20, c=color, label=label)
    
    max_val = max(y_test.max(), median_pred.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, label='Perfect')
    ax.set_xlabel('Actual Resolution Time (days)', fontsize=12)
    ax.set_ylabel('Predicted Resolution Time (days)', fontsize=12)
    ax.set_title('Predicted vs Actual — By Duration Category', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / '1_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 2. SLOW CASE PROBABILITY DISTRIBUTION
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for cat_idx, (color, label) in enumerate(zip(colors, labels)):
        mask = categories == cat_idx
        if mask.any():
            ax.hist(slow_prob[mask], bins=30, alpha=0.5, color=color, label=label, density=True)
    
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Threshold')
    ax.set_xlabel('Slow Case Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Slow Case Classifier Output Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / '2_slow_probability_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 3. MAE BY CATEGORY
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cat_names = ['SHORT', 'MEDIUM', 'LONG']
    cat_full = ['SHORT (≤7d)', 'MEDIUM (7-30d)', 'LONG (>30d)']
    cat_maes = [metrics.get(c, {}).get('mae', 0) for c in cat_full]
    cat_ns = [metrics.get(c, {}).get('n', 0) for c in cat_full]
    
    bars = ax.bar(cat_names, cat_maes, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, mae, n in zip(bars, cat_maes, cat_ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{mae:.1f}d\n(n={n})', ha='center', fontsize=10)
    
    ax.set_ylabel('Mean Absolute Error (days)', fontsize=12)
    ax.set_title('MAE by Duration Category', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(cat_maes) * 1.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / '3_mae_by_category.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 4. COVERAGE BY CATEGORY
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cat_coverage_80 = [metrics.get(c, {}).get('coverage_80', 0) * 100 for c in cat_full]
    cat_coverage_90 = [metrics.get(c, {}).get('coverage_90', 0) * 100 for c in cat_full]
    
    x = np.arange(len(cat_names))
    width = 0.35
    
    ax.bar(x - width/2, cat_coverage_80, width, label='80% Interval', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, cat_coverage_90, width, label='90% Interval', color='darkblue', alpha=0.8)
    
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='Target (80%)')
    ax.set_xticks(x)
    ax.set_xticklabels(cat_names)
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Prediction Interval Coverage by Category', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    fig.savefig(output_dir / '4_coverage_by_category.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 5. ERROR DISTRIBUTION
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for cat_idx, (ax, cat_name, color) in enumerate(zip(axes, cat_names, colors)):
        mask = categories == cat_idx
        if mask.any():
            errors = y_test[mask] - median_pred[mask]
            ax.hist(errors, bins=30, color=color, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='black', linestyle='--')
            ax.set_xlabel('Error (days)')
            ax.set_ylabel('Count')
            ax.set_title(f'{cat_name} Errors (n={mask.sum()})')
    
    plt.suptitle('Error Distribution by Category', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / '5_error_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 6. INTERVAL WIDTH BY CATEGORY
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cat_widths = [metrics.get(c, {}).get('interval_width_80', 0) for c in cat_full]
    
    bars = ax.bar(cat_names, cat_widths, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, width in zip(bars, cat_widths):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{width:.1f}d', ha='center', fontsize=10)
    
    ax.set_ylabel('80% Interval Width (days)', fontsize=12)
    ax.set_title('Prediction Interval Width by Category', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(output_dir / '6_interval_width.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 7. PREDICTION INTERVAL EXAMPLES
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sample 50 random cases
    np.random.seed(42)
    sample_idx = np.random.choice(len(y_test), min(50, len(y_test)), replace=False)
    sample_idx = sample_idx[np.argsort(y_test[sample_idx])]
    
    for i, idx in enumerate(sample_idx):
        ax.plot([i, i], [lower_10[idx], upper_90[idx]], 'b-', alpha=0.5, linewidth=2)
        ax.plot(i, median_pred[idx], 'b.', markersize=4)
        ax.plot(i, y_test[idx], 'r.', markersize=6)
    
    ax.set_xlabel('Sorted Cases', fontsize=12)
    ax.set_ylabel('Days', fontsize=12)
    ax.set_title('Prediction Intervals (50 samples) — Blue=Predicted, Red=Actual', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(output_dir / '7_interval_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 8. SUMMARY INFOGRAPHIC
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overall stats
    ax1 = axes[0, 0]
    overall = metrics['overall']
    stats_text = f"""
    RESOLUTION TIME PREDICTOR v3
    ════════════════════════════════
    
    Overall MAE: {overall['mae']:.2f} days
    Overall RMSE: {overall['rmse']:.2f} days
    
    Coverage (80% interval): {overall['coverage_80']:.1%}
    Coverage (90% interval): {overall['coverage_90']:.1%}
    
    Interval Width (80%): {overall['interval_width_80']:.1f} days
    
    Slow Detection Accuracy: {overall.get('slow_detection_accuracy', 0):.1%}
    
    Total Test Samples: {overall['n']:,}
    """
    ax1.text(0.1, 0.5, stats_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax1.axis('off')
    ax1.set_title('Overall Performance', fontsize=14, fontweight='bold')
    
    # By category table
    ax2 = axes[0, 1]
    table_data = []
    for cat_name, cat_full_name in zip(cat_names, cat_full):
        m = metrics.get(cat_full_name, {})
        table_data.append([
            cat_name,
            f"{m.get('n', 0):,} ({m.get('pct', 0):.0%})",
            f"{m.get('mae', 0):.1f}d",
            f"{m.get('coverage_80', 0):.0%}"
        ])
    
    table = ax2.table(
        cellText=table_data,
        colLabels=['Category', 'n (%)', 'MAE', 'Coverage'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax2.axis('off')
    ax2.set_title('Performance by Category', fontsize=14, fontweight='bold')
    
    # Category pie
    ax3 = axes[1, 0]
    sizes = [metrics.get(c, {}).get('n', 0) for c in cat_full]
    ax3.pie(sizes, labels=cat_names, colors=colors, autopct='%1.0f%%', startangle=90)
    ax3.set_title('Duration Distribution', fontsize=14, fontweight='bold')
    
    # Production guidance
    ax4 = axes[1, 1]
    guidance = """
    PRODUCTION DEPLOYMENT GUIDANCE
    ════════════════════════════════
    
    ✅ SHORT CASES (≤7d):
       High confidence, reliable predictions
       Display: "Expected: X days (range: Y-Z)"
    
    ⚠️ MEDIUM CASES (7-30d):
       Good predictions, moderate uncertainty
       Display with confidence indicator
    
    🔴 LONG CASES (>30d):
       Higher uncertainty, use wide intervals
       Display: "Complex case: X-Y days (may vary)"
       Add explicit warning for users
    
    📊 MONITORING:
       Track actual vs predicted weekly
       Recalibrate quarterly
    """
    ax4.text(0.05, 0.5, guidance, transform=ax4.transAxes, fontsize=10,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax4.axis('off')
    ax4.set_title('Production Guidance', fontsize=14, fontweight='bold')
    
    plt.suptitle('RESOLUTION TIME PREDICTOR v3 — COMPLETE SUMMARY', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / '8_summary_infographic.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    log.info(f"  ✓ Created 8 visualizations")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_production_pipeline():
    """Run production resolution time predictor."""
    start = datetime.now()
    
    print("\n" + "=" * 70)
    print("  RESOLUTION TIME PREDICTOR — PRODUCTION v3")
    print("=" * 70)
    print(f"  {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  5-Star Quality | Enhanced Long-Tail Handling")
    print("=" * 70 + "\n")
    
    log.info("=" * 60)
    log.info("RESOLUTION TIME PREDICTOR v3")
    log.info("=" * 60)
    
    # 1. Load data
    log.info("\n[1/6] Loading query data...")
    queries = load_query_data()
    log.info(f"  {len(queries):,} total queries")
    
    # 2. Prepare
    log.info("\n[2/6] Preparing data...")
    resolved, open_issues = prepare_data(queries)
    log.info(f"  Resolved: {len(resolved):,}")
    log.info(f"  Open:     {len(open_issues):,}")
    
    # Duration distribution
    categories = classify_duration(resolved['resolution_days'].values)
    log.info(f"\n  Duration Distribution:")
    log.info(f"    SHORT (≤7d):   {(categories == 0).sum():,} ({(categories == 0).mean():.1%})")
    log.info(f"    MEDIUM (7-30d): {(categories == 1).sum():,} ({(categories == 1).mean():.1%})")
    log.info(f"    LONG (>30d):    {(categories == 2).sum():,} ({(categories == 2).mean():.1%})")
    
    # 3. Feature engineering
    log.info("\n[3/6] Engineering features (v3)...")
    X = engineer_features_v3(resolved)
    y = resolved['resolution_days'].values
    feature_cols = list(X.columns)
    log.info(f"  {len(feature_cols)} features created")
    
    # 4. Split
    log.info("\n[4/6] Splitting data...")
    
    # Ensure we have valid categories for stratification
    valid_mask = ~np.isnan(y) & (y >= 0)
    X = X[valid_mask]
    y = y[valid_mask]
    categories = categories[valid_mask]
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=categories
    )
    
    train_categories = classify_duration(y_train_val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )
    
    log.info(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    
    # Scale
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # 5. Train models
    log.info("\n[5/6] Training ensemble models...")
    models = train_ensemble_models(X_train_s, y_train, X_val_s, y_val)
    
    # 6. Evaluate
    log.info("\n[6/6] Evaluating...")
    results = predict_with_ensemble(X_test_s, models)
    test_categories = classify_duration(y_test)
    metrics = evaluate_comprehensive(y_test, results, test_categories)
    
    log.info(f"\n  OVERALL METRICS:")
    overall = metrics['overall']
    log.info(f"    MAE:          {overall['mae']:.2f} days")
    log.info(f"    RMSE:         {overall['rmse']:.2f} days")
    log.info(f"    Coverage 80%: {overall['coverage_80']:.1%}")
    log.info(f"    Coverage 90%: {overall['coverage_90']:.1%}")
    
    log.info(f"\n  BY CATEGORY:")
    for cat in ['SHORT (≤7d)', 'MEDIUM (7-30d)', 'LONG (>30d)']:
        if cat in metrics:
            m = metrics[cat]
            log.info(f"    {cat}: MAE={m['mae']:.1f}d, Coverage={m['coverage_80']:.0%} (n={m['n']})")
    
    # Red flags
    red_flags = run_red_flags(metrics)
    if red_flags:
        log.warning("\n  RED FLAGS:")
        for flag in red_flags:
            log.warning(f"    {flag}")
    else:
        log.info("\n  ✓ No red flags detected")
    
    # Visualizations
    log.info("\n  Creating visualizations...")
    create_comprehensive_viz(y_test, results, test_categories, metrics, OUTPUT_DIR / 'figures')
    
    # Save models
    log.info("\n  Saving models...")
    with open(OUTPUT_DIR / 'models' / 'models_v3.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    with open(OUTPUT_DIR / 'models' / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    config = {
        'version': 'PRODUCTION_v3',
        'created': datetime.now().isoformat(),
        'features': feature_cols,
        'metrics': metrics,
        'red_flags': red_flags,
        'thresholds': {'short': SHORT_THRESHOLD, 'medium': MEDIUM_THRESHOLD}
    }
    
    with open(OUTPUT_DIR / 'models' / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Performance table
    perf_data = []
    for cat in ['SHORT (≤7d)', 'MEDIUM (7-30d)', 'LONG (>30d)']:
        if cat in metrics:
            m = metrics[cat]
            perf_data.append({
                'Category': cat,
                'n': m['n'],
                'Percentage': f"{m['pct']:.1%}",
                'MAE': f"{m['mae']:.2f}",
                'RMSE': f"{m['rmse']:.2f}",
                'Coverage_80': f"{m['coverage_80']:.1%}",
                'Interval_Width': f"{m['interval_width_80']:.1f}"
            })
    pd.DataFrame(perf_data).to_csv(OUTPUT_DIR / 'tables' / 'performance.csv', index=False)
    
    duration = (datetime.now() - start).total_seconds()
    status = "✓ PASSED" if not red_flags else f"⚠️ {len(red_flags)} FLAGS"
    
    print("\n" + "=" * 70)
    print(f"  RESOLUTION TIME v3 — {status}")
    print("=" * 70)
    print(f"  Duration: {duration:.1f}s")
    print(f"\n  OVERALL:")
    print(f"    MAE:      {overall['mae']:.2f} days")
    print(f"    Coverage: {overall['coverage_80']:.1%}")
    
    print(f"\n  BY CATEGORY:")
    for cat in ['SHORT (≤7d)', 'MEDIUM (7-30d)', 'LONG (>30d)']:
        if cat in metrics:
            m = metrics[cat]
            pct = m.get('pct', 0) * 100
            print(f"    {cat}: MAE={m['mae']:.1f}d, Coverage={m['coverage_80']:.0%} ({pct:.0f}% of data)")
    
    if red_flags:
        print(f"\n  RED FLAGS ({len(red_flags)}):")
        for flag in red_flags:
            print(f"    {flag}")
    
    print(f"\n  Output: {OUTPUT_DIR}")
    print("=" * 70 + "\n")
    
    return metrics, config


if __name__ == '__main__':
    run_production_pipeline()
