"""
TRIALPULSE NEXUS 10X — ANOMALY DETECTOR v2 OPTIMIZED
Memory-Efficient | Ensemble Approach | Production Ready

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
from typing import Dict, List

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# PostgreSQL data service
from src.database.pg_data_service import get_data_service

ROOT = Path(__file__).parent.parent
# UPR loaded from PostgreSQL
OUTPUT_DIR = ROOT / 'data' / 'outputs' / 'anomaly_detector_OPTIMIZED_v2'

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

# Features for anomaly detection
ANOMALY_FEATURES = [
    'pages_entered', 'pages_with_nonconformant_data',
    'pds_proposed', 'pds_confirmed',
    'total_queries', 'forms_verified', 'broken_signatures'
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for anomaly detection."""
    
    # Get available features
    available = [c for c in ANOMALY_FEATURES if c in df.columns]
    log.info(f"  Available features: {len(available)}")
    
    features = df[available].fillna(0).copy()
    
    # Add derived features
    if 'pages_entered' in features.columns and 'pages_with_nonconformant_data' in features.columns:
        features['quality_rate'] = features['pages_with_nonconformant_data'] / (features['pages_entered'] + 1)
    
    if 'total_queries' in features.columns and 'pages_entered' in features.columns:
        features['queries_per_page'] = features['total_queries'] / (features['pages_entered'] + 1)
    
    # Clean
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
        # Clip outliers for stability
        p99 = features[col].quantile(0.99)
        features[col] = features[col].clip(0, p99 * 2)
    
    return features


def train_isolation_forest(X_scaled: np.ndarray, contamination: float = 0.05) -> IsolationForest:
    """Train Isolation Forest model."""
    
    model = IsolationForest(
        n_estimators=100,
        max_samples=min(1000, len(X_scaled)),
        contamination=contamination,
        random_state=42,
        n_jobs=2
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_scaled)
    
    return model


def compute_anomaly_scores(model: IsolationForest, X_scaled: np.ndarray) -> np.ndarray:
    """Compute anomaly scores (higher = more anomalous)."""
    # decision_function returns negative values for anomalies
    raw_scores = -model.decision_function(X_scaled)
    # Normalize to 0-100
    scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8) * 100
    return scores


def detect_clusters(X_scaled: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    """Use DBSCAN to find outlier clusters."""
    
    # Subsample for speed
    if len(X_scaled) > 10000:
        np.random.seed(42)
        idx = np.random.choice(len(X_scaled), 10000, replace=False)
        X_sample = X_scaled[idx]
    else:
        X_sample = X_scaled
        idx = np.arange(len(X_scaled))
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=2)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels_sample = dbscan.fit_predict(X_sample)
    
    # Map back to full dataset
    labels = np.zeros(len(X_scaled), dtype=int) - 1
    labels[idx] = labels_sample
    
    return labels


def evaluate_detector(anomaly_scores: np.ndarray, labels: np.ndarray) -> Dict:
    """Evaluate anomaly detection quality."""
    
    # Basic stats
    n_anomalies = (anomaly_scores > 50).sum()
    n_extreme = (anomaly_scores > 90).sum()
    
    # DBSCAN stats
    n_outliers = (labels == -1).sum()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Score distribution
    score_mean = anomaly_scores.mean()
    score_std = anomaly_scores.std()
    score_p50 = np.percentile(anomaly_scores, 50)
    score_p90 = np.percentile(anomaly_scores, 90)
    score_p99 = np.percentile(anomaly_scores, 99)
    
    metrics = {
        'n_total': len(anomaly_scores),
        'n_anomalies_50': int(n_anomalies),
        'n_extreme_90': int(n_extreme),
        'pct_anomalies': float(n_anomalies / len(anomaly_scores)),
        'n_dbscan_outliers': int(n_outliers),
        'n_clusters': int(n_clusters),
        'score_mean': float(score_mean),
        'score_std': float(score_std),
        'score_p50': float(score_p50),
        'score_p90': float(score_p90),
        'score_p99': float(score_p99)
    }
    
    return metrics


def check_issues(metrics: Dict) -> List[str]:
    """Check for red flags."""
    issues = []
    
    # Too many or too few anomalies
    if metrics['pct_anomalies'] > 0.20:
        issues.append(f"Too many anomalies ({metrics['pct_anomalies']:.1%})")
    
    if metrics['pct_anomalies'] < 0.01:
        issues.append(f"Too few anomalies ({metrics['pct_anomalies']:.1%})")
    
    # Score distribution issues
    if metrics['score_std'] < 5:
        issues.append(f"Low score variance (std={metrics['score_std']:.1f})")
    
    if metrics['n_extreme_90'] == 0:
        issues.append("No extreme anomalies detected")
    
    return issues


def create_visualizations(df: pd.DataFrame, anomaly_scores: np.ndarray, 
                          labels: np.ndarray, metrics: Dict, 
                          features: List[str], output_dir: Path):
    """Create 8 comprehensive visualizations."""
    
    # 1. Anomaly Score Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(anomaly_scores, bins=50, color='steelblue', alpha=0.8, edgecolor='black')
    ax.axvline(50, color='orange', linestyle='--', label='Anomaly threshold (50)')
    ax.axvline(90, color='red', linestyle='--', label='Extreme threshold (90)')
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Count')
    ax.set_title('Anomaly Score Distribution')
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / '1_score_distribution.png', dpi=150)
    plt.close()
    
    # 2. Anomaly Rate by Percentile
    fig, ax = plt.subplots(figsize=(10, 6))
    percentiles = np.arange(0, 101, 5)
    thresholds = np.percentile(anomaly_scores, percentiles)
    ax.plot(percentiles, thresholds, 'b-', linewidth=2)
    ax.fill_between(percentiles, thresholds, alpha=0.3)
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Anomaly Score')
    ax.set_title('Score by Percentile')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / '2_score_percentiles.png', dpi=150)
    plt.close()
    
    # 3. Top Anomalies Feature Profile
    fig, ax = plt.subplots(figsize=(12, 6))
    top_idx = np.argsort(anomaly_scores)[-20:]
    normal_idx = np.argsort(anomaly_scores)[:100]
    
    feature_subset = features[:min(6, len(features))]
    x = np.arange(len(feature_subset))
    width = 0.35
    
    if len(df.columns) > 0:
        top_means = [df.iloc[top_idx][f].mean() if f in df.columns else 0 for f in feature_subset]
        normal_means = [df.iloc[normal_idx][f].mean() if f in df.columns else 0 for f in feature_subset]
        
        ax.bar(x - width/2, normal_means, width, label='Normal', color='green', alpha=0.7)
        ax.bar(x + width/2, top_means, width, label='Anomalous', color='red', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f[:12] for f in feature_subset], rotation=45, ha='right')
        ax.set_ylabel('Mean Value')
        ax.set_title('Feature Comparison: Normal vs Anomalous')
        ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / '3_feature_comparison.png', dpi=150)
    plt.close()
    
    # 4. DBSCAN Cluster Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    unique_labels = np.unique(labels)
    counts = [np.sum(labels == l) for l in unique_labels]
    colors = ['red' if l == -1 else 'steelblue' for l in unique_labels]
    labels_str = ['Outliers' if l == -1 else f'Cluster {l}' for l in unique_labels]
    ax.bar(labels_str[:10], counts[:10], color=colors[:10], alpha=0.8, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('DBSCAN Cluster Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(output_dir / '4_cluster_distribution.png', dpi=150)
    plt.close()
    
    # 5. Anomaly Score by Tier
    fig, ax = plt.subplots(figsize=(10, 6))
    tiers = pd.cut(anomaly_scores, bins=[0, 25, 50, 75, 100], labels=['Normal', 'Low', 'Medium', 'High'])
    tier_counts = pd.Series(tiers).value_counts()
    colors = ['green', 'yellow', 'orange', 'red']
    ax.bar(tier_counts.index.astype(str), tier_counts.values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Patients by Anomaly Tier')
    for i, v in enumerate(tier_counts.values):
        ax.text(i, v + 100, str(v), ha='center')
    plt.tight_layout()
    fig.savefig(output_dir / '5_anomaly_tiers.png', dpi=150)
    plt.close()
    
    # 6. Score vs Primary Features
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, col in enumerate(features[:3]):
        if col in df.columns:
            axes[idx].scatter(df[col], anomaly_scores, alpha=0.3, s=10, c=anomaly_scores, cmap='RdYlGn_r')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Anomaly Score')
            axes[idx].set_title(f'Score vs {col}')
    plt.tight_layout()
    fig.savefig(output_dir / '6_score_vs_features.png', dpi=150)
    plt.close()
    
    # 7. Red Flag Check
    fig, ax = plt.subplots(figsize=(10, 6))
    checks = [
        ('Anomaly Rate 1-20%', 0.01 <= metrics['pct_anomalies'] <= 0.20),
        ('Score Variance > 5', metrics['score_std'] > 5),
        ('Extreme Anomalies > 0', metrics['n_extreme_90'] > 0),
        ('Clusters Detected', metrics['n_clusters'] > 0)
    ]
    colors = ['green' if passed else 'red' for _, passed in checks]
    ax.barh([c[0] for c in checks], [1 if c[1] else 0 for c in checks], color=colors, alpha=0.8)
    ax.set_xlim(0, 1.5)
    ax.set_title('Quality Checks')
    for i, (name, passed) in enumerate(checks):
        ax.text(1.1, i, 'PASS' if passed else 'FAIL', va='center', fontweight='bold', color=colors[i])
    plt.tight_layout()
    fig.savefig(output_dir / '7_quality_checks.png', dpi=150)
    plt.close()
    
    # 8. Summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    stats = f"ANOMALY DETECTOR v2\n===================\n\nPatients: {metrics['n_total']:,}\nAnomalies (>50): {metrics['n_anomalies_50']:,}\nExtreme (>90): {metrics['n_extreme_90']:,}\n\nAnomaly Rate: {metrics['pct_anomalies']:.1%}\nScore Std: {metrics['score_std']:.1f}"
    ax1.text(0.5, 0.5, stats, transform=ax1.transAxes, fontsize=12, va='center', ha='center',
             fontfamily='monospace', bbox=dict(facecolor='lightblue', alpha=0.3))
    ax1.axis('off')
    ax1.set_title('Summary', fontweight='bold')
    
    ax2 = axes[0, 1]
    tiers = pd.cut(anomaly_scores, bins=[0, 25, 50, 75, 100], labels=['Normal', 'Low', 'Medium', 'High'])
    tier_counts = pd.Series(tiers).value_counts()
    ax2.pie(tier_counts.values, labels=tier_counts.index.astype(str), colors=['green', 'yellow', 'orange', 'red'], autopct='%1.0f%%')
    ax2.set_title('Tier Distribution', fontweight='bold')
    
    ax3 = axes[1, 0]
    ax3.hist(anomaly_scores, bins=30, color='steelblue', alpha=0.8, edgecolor='black')
    ax3.axvline(50, color='red', linestyle='--')
    ax3.set_xlabel('Score')
    ax3.set_title('Score Distribution', fontweight='bold')
    
    ax4 = axes[1, 1]
    guide = "DEPLOYMENT\n==========\n\nScore > 90: CRITICAL - Review immediately\nScore 75-90: HIGH - Review within 24h\nScore 50-75: MEDIUM - Monitor\nScore < 50: NORMAL - No action\n\nUse for triage, not verdict"
    ax4.text(0.5, 0.5, guide, transform=ax4.transAxes, fontsize=10, va='center', ha='center',
             fontfamily='monospace', bbox=dict(facecolor='lightyellow', alpha=0.5))
    ax4.axis('off')
    ax4.set_title('Guidance', fontweight='bold')
    
    plt.suptitle('ANOMALY DETECTOR v2 — SUMMARY', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / '8_summary.png', dpi=150)
    plt.close()
    
    log.info("  Created 8 visualizations")


def run_pipeline():
    start = datetime.now()
    
    print("\n" + "=" * 70)
    print("  ANOMALY DETECTOR — OPTIMIZED v2")
    print("=" * 70)
    print(f"  {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Memory-Efficient | Ensemble Approach | Production Ready")
    print("=" * 70 + "\n")
    
    log.info("=" * 60)
    log.info("ANOMALY DETECTOR OPTIMIZED v2")
    log.info("=" * 60)
    
    log.info("\n[1/6] Loading from PostgreSQL...")
    data_service = get_data_service()
    df = data_service.get_patients()
    log.info(f"  {len(df):,} patients from PostgreSQL")
    
    log.info("\n[2/6] Preparing features...")
    features_df = prepare_features(df)
    feature_cols = list(features_df.columns)
    log.info(f"  {len(feature_cols)} features: {feature_cols}")
    
    log.info("\n[3/6] Scaling...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(features_df)
    
    gc.collect()
    
    log.info("\n[4/6] Training Isolation Forest...")
    model = train_isolation_forest(X_scaled, contamination=0.05)
    anomaly_scores = compute_anomaly_scores(model, X_scaled)
    log.info(f"  Mean score: {anomaly_scores.mean():.1f}")
    log.info(f"  Anomalies (>50): {(anomaly_scores > 50).sum():,}")
    
    log.info("\n[5/6] Running DBSCAN clustering...")
    labels = detect_clusters(X_scaled, eps=0.8, min_samples=10)
    log.info(f"  Clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
    log.info(f"  Outliers: {(labels == -1).sum():,}")
    
    log.info("\n[6/6] Evaluating...")
    metrics = evaluate_detector(anomaly_scores, labels)
    
    log.info(f"\n  Total patients: {metrics['n_total']:,}")
    log.info(f"  Anomalies (>50): {metrics['n_anomalies_50']:,} ({metrics['pct_anomalies']:.1%})")
    log.info(f"  Extreme (>90): {metrics['n_extreme_90']:,}")
    
    issues = check_issues(metrics)
    if issues:
        log.warning(f"\n  ISSUES: {issues}")
    else:
        log.info("\n  ✓ No red flags")
    
    log.info("\n  Creating visualizations...")
    create_visualizations(features_df, anomaly_scores, labels, metrics, feature_cols, OUTPUT_DIR / 'figures')
    
    # Save
    df['anomaly_score'] = anomaly_scores
    df['is_anomaly'] = anomaly_scores > 50
    df['cluster_id'] = labels
    
    with open(OUTPUT_DIR / 'models' / 'isolation_forest.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(OUTPUT_DIR / 'models' / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    config = {
        'version': 'OPTIMIZED_v2',
        'created': datetime.now().isoformat(),
        'features': feature_cols,
        'metrics': metrics,
        'issues': issues,
        'thresholds': {'anomaly': 50, 'extreme': 90}
    }
    with open(OUTPUT_DIR / 'models' / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save anomaly list
    anomalies_df = df[df['is_anomaly']][['anomaly_score', 'cluster_id'] + feature_cols[:5]]
    anomalies_df = anomalies_df.sort_values('anomaly_score', ascending=False).head(1000)
    anomalies_df.to_csv(OUTPUT_DIR / 'tables' / 'top_anomalies.csv', index=True)
    
    duration = (datetime.now() - start).total_seconds()
    
    print("\n" + "=" * 70)
    print("  ANOMALY DETECTOR v2 — COMPLETE")
    print("=" * 70)
    print(f"  Duration: {duration:.1f}s")
    print(f"  Patients: {metrics['n_total']:,}")
    print(f"  Anomalies: {metrics['n_anomalies_50']:,} ({metrics['pct_anomalies']:.1%})")
    print(f"  Extreme: {metrics['n_extreme_90']:,}")
    
    if issues:
        print(f"\n  ISSUES: {issues}")
    else:
        print("\n  ✅ PERFECT - NO RED FLAGS")
    
    print(f"\n  Output: {OUTPUT_DIR}")
    print("=" * 70 + "\n")
    
    return metrics, issues


if __name__ == '__main__':
    run_pipeline()
