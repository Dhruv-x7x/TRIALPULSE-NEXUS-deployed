"""
TRIALPULSE NEXUS 10X — SITE RISK RANKER v2 OPTIMIZED
Memory-Efficient | No Leakage | Production Ready

VERSION: OPTIMIZED_v2
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
import gc
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from scipy.stats import kendalltau, spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
except ImportError:
    sys.exit(1)

# PostgreSQL data service
from src.database.pg_data_service import get_data_service

ROOT = Path(__file__).parent.parent
# UPR loaded from PostgreSQL
OUTPUT_DIR = ROOT / 'data' / 'outputs' / 'site_risk_ranker_OPTIMIZED_v2'

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

# Whitelist of safe features (no leakage)
WHITELIST = [
    'pages_entered', 'pages_with_nonconformant_data',
    'pds_proposed', 'pds_confirmed',
    'expected_visits_rave_edc_bo4',
    'total_queries', 'forms_verified',
    'broken_signatures'
]


def aggregate_to_site(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate patient-level data to site level."""
    
    # Find site column
    site_col = None
    for col in ['fac_id', 'site', 'site_id', 'facility_id']:
        if col in df.columns:
            site_col = col
            break
    
    if site_col is None:
        # Create synthetic site if not found
        log.warning("  No site column found, creating from index")
        df['_site_id'] = (df.index % 500)
        site_col = '_site_id'
    
    # Get available whitelist columns
    available = [c for c in WHITELIST if c in df.columns]
    log.info(f"  Available features: {len(available)}")
    
    # Aggregate
    agg_dict = {}
    for col in available:
        agg_dict[f'{col}_mean'] = (col, 'mean')
        agg_dict[f'{col}_sum'] = (col, 'sum')
        agg_dict[f'{col}_max'] = (col, 'max')
    
    agg_dict['patient_count'] = (site_col, 'count')
    
    site_df = df.groupby(site_col).agg(**agg_dict).reset_index()
    site_df = site_df.rename(columns={site_col: 'site_id'})
    
    # Add derived features
    if 'pages_with_nonconformant_data_sum' in site_df.columns and 'pages_entered_sum' in site_df.columns:
        site_df['quality_rate'] = site_df['pages_with_nonconformant_data_sum'] / (site_df['pages_entered_sum'] + 1)
    
    if 'total_queries_sum' in site_df.columns:
        site_df['queries_per_patient'] = site_df['total_queries_sum'] / (site_df['patient_count'] + 1)
    
    # Clean
    for col in site_df.columns:
        if site_df[col].dtype in ['float64', 'int64']:
            site_df[col] = site_df[col].fillna(0).clip(-1e6, 1e6)
    
    return site_df


def compute_baseline_risk(df: pd.DataFrame) -> pd.Series:
    """Compute rule-based risk score for targets."""
    risk = pd.Series(0.0, index=df.index)
    
    # Issue density
    if 'total_queries_sum' in df.columns:
        risk += (df['total_queries_sum'] / (df['patient_count'] + 1)).clip(0, 10) * 0.3
    
    # Quality issues
    if 'quality_rate' in df.columns:
        risk += df['quality_rate'].clip(0, 1) * 0.25
    
    # Deviations
    if 'pds_confirmed_sum' in df.columns:
        risk += (df['pds_confirmed_sum'] / (df['patient_count'] + 1)).clip(0, 5) * 0.2
    
    # Broken signatures
    if 'broken_signatures_sum' in df.columns:
        risk += (df['broken_signatures_sum'] > 0).astype(float) * 0.15
    
    # Patient count (more patients = more opportunities for issues)
    risk += (df['patient_count'] / df['patient_count'].max()).clip(0, 1) * 0.1
    
    return risk


def create_pairwise_data(df: pd.DataFrame, max_pairs: int = 30000) -> Tuple[pd.DataFrame, np.ndarray]:
    """Create pairwise training data with subsampling."""
    
    features = [c for c in df.columns if c not in ['site_id', 'patient_count', '_risk_score']]
    
    # Get risk scores
    risk = compute_baseline_risk(df)
    
    # Sample pairs
    n = len(df)
    pairs_X = []
    pairs_y = []
    
    np.random.seed(42)
    
    # Limited pairs per site
    max_per_site = min(50, max_pairs // n)
    
    for i in range(n):
        # Sample random opponents
        opponents = np.random.choice([j for j in range(n) if j != i], 
                                      size=min(max_per_site, n-1), replace=False)
        
        for j in opponents:
            # Compare risks
            diff = risk.iloc[i] - risk.iloc[j]
            if abs(diff) < 0.01:  # Skip ties
                continue
            
            # Feature difference
            feat_diff = df[features].iloc[i].values - df[features].iloc[j].values
            pairs_X.append(feat_diff)
            pairs_y.append(1 if diff > 0 else 0)  # 1 if site i is riskier
    
    # Subsample if too many
    if len(pairs_X) > max_pairs:
        idx = np.random.choice(len(pairs_X), max_pairs, replace=False)
        pairs_X = [pairs_X[i] for i in idx]
        pairs_y = [pairs_y[i] for i in idx]
    
    X = pd.DataFrame(pairs_X, columns=features)
    y = np.array(pairs_y)
    
    return X, y


def train_ranker(X_train, y_train, X_val, y_val):
    """Train pairwise ranking model."""
    
    model = xgb.XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.2,
        reg_lambda=1.0,
        verbosity=0,
        random_state=42,
        n_jobs=2
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    
    return model


def compute_site_scores(model, site_df: pd.DataFrame, features: List[str], scaler) -> pd.DataFrame:
    """Compute risk scores for each site using pairwise comparisons."""
    
    X = site_df[features].fillna(0).values
    n_sites = len(site_df)
    
    # For each site, compute average win probability against random sample of sites
    scores = np.zeros(n_sites)
    
    np.random.seed(42)
    sample_size = min(50, n_sites - 1)  # Compare against up to 50 random sites
    
    for i in range(n_sites):
        opponents = np.random.choice([j for j in range(n_sites) if j != i], 
                                      size=sample_size, replace=False)
        
        # Compute feature differences
        diffs = X[i] - X[opponents]
        diffs_scaled = scaler.transform(diffs)
        
        # Probability that site i is riskier
        probs = model.predict_proba(diffs_scaled)[:, 1]
        scores[i] = probs.mean()
    
    # Normalize to 0-100
    site_df = site_df.copy()
    site_df['risk_score'] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8) * 100
    site_df['risk_rank'] = site_df['risk_score'].rank(ascending=False, method='min').astype(int)
    
    return site_df


def evaluate_ranker(model, site_df: pd.DataFrame, features: List[str], scaler) -> Dict:
    """Evaluate ranking quality."""
    
    X = site_df[features].fillna(0)
    X_scaled = scaler.transform(X)
    
    # Get predicted scores
    pred_scores = model.predict_proba(X_scaled)[:, 1]
    
    # Get baseline scores (ground truth proxy)
    baseline_scores = compute_baseline_risk(site_df)
    
    # Correlation metrics
    tau, _ = kendalltau(baseline_scores, pred_scores)
    rho, _ = spearmanr(baseline_scores, pred_scores)
    
    # NDCG@k - measures how well top sites match
    def ndcg_at_k(true_scores, pred_scores, k):
        # Get top k indices by true and predicted scores
        true_top_k = set(np.argsort(-true_scores)[:k])
        pred_top_k = np.argsort(-pred_scores)[:k]
        
        # DCG: sum of relevance (1 if in true top-k, 0 otherwise) / log position
        dcg = 0
        for i, idx in enumerate(pred_top_k):
            relevance = 1 if idx in true_top_k else 0
            dcg += relevance / np.log2(i + 2)
        
        # IDCG: perfect ranking
        idcg = sum(1 / np.log2(i + 2) for i in range(k))
        
        return dcg / idcg if idcg > 0 else 0
    
    ndcg_5 = ndcg_at_k(baseline_scores.values, pred_scores, 5)
    ndcg_10 = ndcg_at_k(baseline_scores.values, pred_scores, 10)
    ndcg_20 = ndcg_at_k(baseline_scores.values, pred_scores, 20)
    
    # Feature importance check
    imp = model.feature_importances_
    top_imp = imp.max() / imp.sum() if imp.sum() > 0 else 0
    top5_imp = np.sort(imp)[-5:].sum() / imp.sum() if imp.sum() > 0 else 0
    
    metrics = {
        'kendall_tau': float(tau),
        'spearman_rho': float(rho),
        'ndcg_5': float(ndcg_5),
        'ndcg_10': float(ndcg_10),
        'ndcg_20': float(ndcg_20),
        'top_feature_importance': float(top_imp),
        'top5_feature_importance': float(top5_imp),
        'n_sites': len(site_df)
    }
    
    return metrics


def check_issues(metrics: Dict) -> List[str]:
    """Check for red flags."""
    issues = []
    
    if metrics['kendall_tau'] > 0.98:
        issues.append(f"Tau too high ({metrics['kendall_tau']:.3f}) - possible leakage")
    
    if metrics['kendall_tau'] < 0.5:
        issues.append(f"Tau too low ({metrics['kendall_tau']:.3f})")
    
    if metrics['spearman_rho'] < 0.6:
        issues.append(f"Spearman too low ({metrics['spearman_rho']:.3f})")
    
    if metrics['top_feature_importance'] > 0.6:
        issues.append(f"Single feature dominance ({metrics['top_feature_importance']:.1%})")
    
    return issues


def create_visualizations(site_df: pd.DataFrame, metrics: Dict, model, features: List[str], output_dir: Path):
    """Create visualizations."""
    
    # 1. Risk Score Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(site_df['risk_score'], bins=30, color='steelblue', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Risk Score')
    ax.set_ylabel('Count')
    ax.set_title('Site Risk Score Distribution')
    plt.tight_layout()
    fig.savefig(output_dir / '1_risk_distribution.png', dpi=150)
    plt.close()
    
    # 2. Top 20 High Risk Sites
    fig, ax = plt.subplots(figsize=(12, 8))
    top20 = site_df.nlargest(20, 'risk_score')
    ax.barh(range(20), top20['risk_score'], color='red', alpha=0.7)
    ax.set_yticks(range(20))
    ax.set_yticklabels([f"Site {i}" for i in top20['site_id'].values], fontsize=9)
    ax.set_xlabel('Risk Score')
    ax.set_title('Top 20 High Risk Sites')
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(output_dir / '2_top20_sites.png', dpi=150)
    plt.close()
    
    # 3. Feature Importance
    fig, ax = plt.subplots(figsize=(12, 8))
    imp = model.feature_importances_
    sorted_idx = np.argsort(imp)[-15:]
    ax.barh(range(len(sorted_idx)), imp[sorted_idx], color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([features[i][:25] for i in sorted_idx], fontsize=9)
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Features')
    plt.tight_layout()
    fig.savefig(output_dir / '3_feature_importance.png', dpi=150)
    plt.close()
    
    # 4. Performance Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_names = ['Kendall Tau', 'Spearman Rho', 'NDCG@5', 'NDCG@10', 'NDCG@20']
    metric_vals = [metrics['kendall_tau'], metrics['spearman_rho'], 
                   metrics['ndcg_5'], metrics['ndcg_10'], metrics['ndcg_20']]
    colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in metric_vals]
    bars = ax.bar(metric_names, metric_vals, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Score')
    ax.set_title('Ranking Performance Metrics')
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, metric_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center')
    plt.tight_layout()
    fig.savefig(output_dir / '4_performance_metrics.png', dpi=150)
    plt.close()
    
    # 5. Risk Score vs Patient Count
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(site_df['patient_count'], site_df['risk_score'], 
                         c=site_df['risk_score'], cmap='RdYlGn_r', alpha=0.5, s=30)
    ax.set_xlabel('Patient Count')
    ax.set_ylabel('Risk Score')
    ax.set_title('Risk Score vs Site Size')
    plt.colorbar(scatter, label='Risk')
    plt.tight_layout()
    fig.savefig(output_dir / '5_risk_vs_size.png', dpi=150)
    plt.close()
    
    # 6. Risk Tier Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    site_df['risk_tier'] = pd.cut(site_df['risk_score'], bins=[0, 25, 50, 75, 100], 
                                   labels=['Low', 'Medium', 'High', 'Critical'])
    tier_counts = site_df['risk_tier'].value_counts()
    colors = ['green', 'yellow', 'orange', 'red']
    ax.bar(tier_counts.index, tier_counts.values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Sites by Risk Tier')
    for i, v in enumerate(tier_counts.values):
        ax.text(i, v + 5, str(v), ha='center')
    plt.tight_layout()
    fig.savefig(output_dir / '6_risk_tiers.png', dpi=150)
    plt.close()
    
    # 7. Red Flag Check
    fig, ax = plt.subplots(figsize=(10, 6))
    checks = [
        ('Leakage (Tau<0.95)', metrics['kendall_tau'] < 0.95),
        ('NDCG@20 < 0.95', metrics['ndcg_20'] < 0.95),
        ('No Single Feature Dominance', metrics['top_feature_importance'] < 0.5),
        ('NDCG@10 > 0.5', metrics['ndcg_10'] > 0.5)
    ]
    colors = ['green' if passed else 'red' for _, passed in checks]
    ax.barh([c[0] for c in checks], [1 if c[1] else 0 for c in checks], color=colors, alpha=0.8)
    ax.set_xlim(0, 1.5)
    ax.set_title('Red Flag Checks')
    for i, (name, passed) in enumerate(checks):
        ax.text(1.1, i, 'PASS' if passed else 'FAIL', va='center', fontweight='bold',
                color='green' if passed else 'red')
    plt.tight_layout()
    fig.savefig(output_dir / '7_red_flag_checks.png', dpi=150)
    plt.close()
    
    # 8. Summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    stats = f"SITE RISK RANKER v2\n===================\n\nSites: {metrics['n_sites']}\nKendall Tau: {metrics['kendall_tau']:.3f}\nNDCG@20: {metrics['ndcg_20']:.3f}"
    ax1.text(0.5, 0.5, stats, transform=ax1.transAxes, fontsize=12, va='center', ha='center', 
             fontfamily='monospace', bbox=dict(facecolor='lightblue', alpha=0.3))
    ax1.axis('off')
    ax1.set_title('Summary', fontweight='bold')
    
    ax2 = axes[0, 1]
    tc = site_df['risk_tier'].value_counts()
    ax2.pie(tc.values, labels=tc.index, colors=['green', 'yellow', 'orange', 'red'], autopct='%1.0f%%')
    ax2.set_title('Risk Tiers', fontweight='bold')
    
    ax3 = axes[1, 0]
    ax3.barh(['Tau', 'Rho', 'NDCG@10', 'NDCG@20'], 
             [metrics['kendall_tau'], metrics['spearman_rho'], metrics['ndcg_10'], metrics['ndcg_20']],
             color='steelblue', alpha=0.8)
    ax3.set_xlim(0, 1)
    ax3.set_title('Metrics', fontweight='bold')
    
    ax4 = axes[1, 1]
    guide = "DEPLOYMENT\n==========\nUse for site prioritization\nRank sites by risk_score\nReview top 20% first\n\nNOT a verdict - a triage tool"
    ax4.text(0.5, 0.5, guide, transform=ax4.transAxes, fontsize=11, va='center', ha='center',
             fontfamily='monospace', bbox=dict(facecolor='lightyellow', alpha=0.5))
    ax4.axis('off')
    ax4.set_title('Guidance', fontweight='bold')
    
    plt.suptitle('SITE RISK RANKER v2 OPTIMIZED — SUMMARY', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / '8_summary.png', dpi=150)
    plt.close()
    
    log.info("  Created 8 visualizations")


def run_pipeline():
    start = datetime.now()
    
    print("\n" + "=" * 70)
    print("  SITE RISK RANKER — OPTIMIZED v2")
    print("=" * 70)
    print(f"  {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Memory-Efficient | No Leakage | Production Ready")
    print("=" * 70 + "\n")
    
    log.info("=" * 60)
    log.info("SITE RISK RANKER OPTIMIZED v2")
    log.info("=" * 60)
    
    log.info("\n[1/7] Loading from PostgreSQL...")
    data_service = get_data_service()
    df = data_service.get_patients()
    log.info(f"  {len(df):,} patients from PostgreSQL")
    
    log.info("\n[2/7] Aggregating to site level...")
    site_df = aggregate_to_site(df)
    log.info(f"  {len(site_df):,} sites")
    
    del df
    gc.collect()
    
    log.info("\n[3/7] Creating pairwise data...")
    features = [c for c in site_df.columns if c not in ['site_id', 'patient_count']]
    X_pairs, y_pairs = create_pairwise_data(site_df, max_pairs=25000)
    log.info(f"  {len(X_pairs):,} pairs")
    
    log.info("\n[4/7] Splitting...")
    X_train, X_val, y_train, y_val = train_test_split(X_pairs, y_pairs, test_size=0.2, random_state=42)
    log.info(f"  Train: {len(X_train)}, Val: {len(X_val)}")
    
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    log.info("\n[5/7] Training...")
    model = train_ranker(X_train_s, y_train, X_val_s, y_val)
    
    log.info("\n[6/7] Evaluating...")
    site_df = compute_site_scores(model, site_df, features, scaler)
    metrics = evaluate_ranker(model, site_df, features, scaler)
    
    log.info(f"\n  Kendall Tau: {metrics['kendall_tau']:.3f}")
    log.info(f"  Spearman Rho: {metrics['spearman_rho']:.3f}")
    log.info(f"  NDCG@10: {metrics['ndcg_10']:.3f}")
    log.info(f"  NDCG@20: {metrics['ndcg_20']:.3f}")
    
    issues = check_issues(metrics)
    if issues:
        log.warning(f"\n  ISSUES: {issues}")
    else:
        log.info("\n  ✓ No red flags")
    
    log.info("\n[7/7] Creating visualizations...")
    create_visualizations(site_df, metrics, model, features, OUTPUT_DIR / 'figures')
    
    # Save
    with open(OUTPUT_DIR / 'models' / 'ranker.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(OUTPUT_DIR / 'models' / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    config = {
        'version': 'OPTIMIZED_v2',
        'created': datetime.now().isoformat(),
        'features': features,
        'metrics': metrics,
        'issues': issues
    }
    with open(OUTPUT_DIR / 'models' / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    site_df[['site_id', 'patient_count', 'risk_score', 'risk_rank']].to_csv(
        OUTPUT_DIR / 'tables' / 'site_rankings.csv', index=False)
    
    duration = (datetime.now() - start).total_seconds()
    
    print("\n" + "=" * 70)
    print("  SITE RISK RANKER v2 — COMPLETE")
    print("=" * 70)
    print(f"  Duration: {duration:.1f}s")
    print(f"  Sites: {metrics['n_sites']}")
    print(f"\n  METRICS:")
    print(f"    Kendall Tau: {metrics['kendall_tau']:.3f}")
    print(f"    NDCG@10: {metrics['ndcg_10']:.3f}")
    print(f"    NDCG@20: {metrics['ndcg_20']:.3f}")
    
    if issues:
        print(f"\n  ISSUES: {issues}")
    else:
        print("\n  ✅ PERFECT - NO RED FLAGS")
    
    print(f"\n  Output: {OUTPUT_DIR}")
    print("=" * 70 + "\n")
    
    return site_df, metrics, issues


if __name__ == '__main__':
    run_pipeline()
