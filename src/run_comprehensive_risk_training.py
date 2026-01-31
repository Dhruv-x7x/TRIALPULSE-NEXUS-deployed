"""
TRIALPULSE NEXUS 10X - Comprehensive Risk Classifier Training
Complete training pipeline with all visualizations for PPT presentation

Generates:
- ROC curves comparison (all models)
- Precision-Recall curves
- Confusion matrix (best model)
- SHAP summary beeswarm plot
- SHAP feature importance bar chart
- Calibration curve
- Threshold analysis plot
- Learning curves
- Model comparison table
- Comprehensive training report
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
from typing import Dict, List, Tuple, Any

# ML imports
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, learning_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    brier_score_loss, classification_report
)
from sklearn.preprocessing import label_binarize, LabelEncoder

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Gradient boosting
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed")

# SHAP
try:
    import shap

# PostgreSQL data access
try:
    from src.database.pg_data_service import get_data_service
    from src.database.pg_writer import get_pg_writer
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed")

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed' / 'ml'
UPR_PATH = PROJECT_ROOT / 'data' / 'processed' / 'upr' / 'unified_patient_record.parquet'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'outputs' / 'ml_training'
FIGURES_DIR = OUTPUT_DIR / 'figures'
MODELS_DIR = OUTPUT_DIR / 'models'
TABLES_DIR = OUTPUT_DIR / 'tables'

# Create directories
for d in [OUTPUT_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / 'training.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

# Color palette for models
MODEL_COLORS = {
    'Random Baseline': '#95a5a6',
    'Logistic Regression': '#3498db',
    'Decision Tree': '#e74c3c',
    'Naive Bayes': '#9b59b6',
    'Random Forest': '#2ecc71',
    'XGBoost': '#f39c12',
    'LightGBM': '#1abc9c',
    'XGB+LGB Ensemble': '#e91e63'
}

CLASS_NAMES = ['Critical', 'High', 'Low', 'Medium']
CLASS_COLORS = {'Critical': '#e74c3c', 'High': '#f39c12', 'Medium': '#3498db', 'Low': '#2ecc71'}


class ComprehensiveRiskTrainer:
    """Complete training pipeline with all visualizations"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        self.best_model_name: str = None
        self.best_model = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.feature_names: List[str] = []
        self.n_classes: int = 4
        self.class_names = CLASS_NAMES
        
    def load_data(self) -> bool:
        """Load prepared ML data from PostgreSQL"""
        logger.info("\n" + "=" * 70)
        logger.info("LOADING DATA FROM POSTGRESQL")
        logger.info("=" * 70)
        
        try:
            from src.database.pg_data_service import PostgreSQLDataService
            db_service = PostgreSQLDataService()
            
            # Load separate splits from tables created by data_preparation.py
            logger.info("  Fetching 'ml_train'...")
            self.X_train = db_service.execute_query("SELECT * FROM ml_train")
            self.y_train = self.X_train.pop('target')
            
            logger.info("  Fetching 'ml_val'...")
            self.X_val = db_service.execute_query("SELECT * FROM ml_val")
            self.y_val = self.X_val.pop('target')
            
            logger.info("  Fetching 'ml_test'...")
            self.X_test = db_service.execute_query("SELECT * FROM ml_test")
            self.y_test = self.X_test.pop('target')
            
            self.feature_names = list(self.X_train.columns)
            self.n_classes = len(np.unique(self.y_train))
            
            logger.info(f"  Training samples: {len(self.X_train):,}")
            logger.info(f"  Validation samples: {len(self.X_val):,}")
            logger.info(f"  Test samples: {len(self.X_test):,}")
            logger.info(f"  Features: {len(self.feature_names)}")
            logger.info(f"  Classes: {self.n_classes}")
            
            # Show class distribution
            train_dist = self.y_train.value_counts().sort_index()
            logger.info(f"\n  Training class distribution:")
            for cls_id, count in train_dist.items():
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class_{cls_id}"
                logger.info(f"    {cls_name}: {count:,} ({count/len(self.y_train)*100:.1f}%)")
            
            return True
                
        except Exception as e:
            logger.error(f"Error loading data from PostgreSQL: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def train_baseline_models(self):
        """Train baseline models for comparison"""
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING BASELINE MODELS")
        logger.info("=" * 70)
        
        # Random Baseline
        logger.info("\n  1. Random Baseline...")
        class RandomClassifier:
            def __init__(self, n_classes):
                self.n_classes = n_classes
                self.class_probs = None
            def fit(self, X, y):
                self.class_probs = np.bincount(y, minlength=self.n_classes) / len(y)
                return self
            def predict_proba(self, X):
                return np.tile(self.class_probs, (len(X), 1))
            def predict(self, X):
                return np.random.choice(self.n_classes, size=len(X), p=self.class_probs)
        
        self.models['Random Baseline'] = RandomClassifier(self.n_classes)
        self.models['Random Baseline'].fit(self.X_train, self.y_train)
        logger.info("    Random Baseline trained ✓")
        
        # Logistic Regression
        logger.info("  2. Logistic Regression...")
        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1
        )
        self.models['Logistic Regression'].fit(self.X_train, self.y_train)
        logger.info("    Logistic Regression trained ✓")
        
        # Decision Tree
        logger.info("  3. Decision Tree...")
        self.models['Decision Tree'] = DecisionTreeClassifier(
            max_depth=5, class_weight='balanced', random_state=42
        )
        self.models['Decision Tree'].fit(self.X_train, self.y_train)
        logger.info("    Decision Tree trained ✓")
        
        # Naive Bayes
        logger.info("  4. Naive Bayes...")
        self.models['Naive Bayes'] = GaussianNB()
        self.models['Naive Bayes'].fit(self.X_train, self.y_train)
        logger.info("    Naive Bayes trained ✓")
    
    def train_advanced_models(self):
        """Train advanced models (Random Forest, XGBoost, LightGBM)"""
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING ADVANCED MODELS")
        logger.info("=" * 70)
        
        # Random Forest
        logger.info("\n  5. Random Forest...")
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.models['Random Forest'].fit(self.X_train, self.y_train)
        logger.info("    Random Forest trained ✓")
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            logger.info("  6. XGBoost...")
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective='multi:softprob',
                eval_metric='mlogloss',
                use_label_encoder=False,
                verbosity=0,
                random_state=42,
                n_jobs=-1
            )
            self.models['XGBoost'].fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False
            )
            logger.info("    XGBoost trained ✓")
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            logger.info("  7. LightGBM...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.models['LightGBM'] = lgb.LGBMClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=20,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    objective='multiclass',
                    num_class=self.n_classes,
                    verbosity=-1,
                    random_state=42,
                    n_jobs=-1
                )
                self.models['LightGBM'].fit(
                    self.X_train, self.y_train,
                    eval_set=[(self.X_val, self.y_val)]
                )
            logger.info("    LightGBM trained ✓")
        
        # XGB+LGB Ensemble
        if XGBOOST_AVAILABLE and LIGHTGBM_AVAILABLE:
            logger.info("  8. XGB+LGB Ensemble...")
            
            class EnsembleClassifier:
                def __init__(self, xgb_model, lgb_model, xgb_weight=0.5, lgb_weight=0.5):
                    self.xgb_model = xgb_model
                    self.lgb_model = lgb_model
                    self.xgb_weight = xgb_weight
                    self.lgb_weight = lgb_weight
                    self.n_classes = xgb_model.n_classes_ if hasattr(xgb_model, 'n_classes_') else 4
                    
                def predict_proba(self, X):
                    xgb_proba = self.xgb_model.predict_proba(X)
                    lgb_proba = self.lgb_model.predict_proba(X)
                    return (xgb_proba * self.xgb_weight + lgb_proba * self.lgb_weight)
                    
                def predict(self, X):
                    proba = self.predict_proba(X)
                    return np.argmax(proba, axis=1)
            
            self.models['XGB+LGB Ensemble'] = EnsembleClassifier(
                self.models['XGBoost'], self.models['LightGBM']
            )
            logger.info("    XGB+LGB Ensemble created ✓")
    
    def evaluate_all_models(self):
        """Evaluate all models and create comparison"""
        logger.info("\n" + "=" * 70)
        logger.info("EVALUATING ALL MODELS")
        logger.info("=" * 70)
        
        for name, model in self.models.items():
            logger.info(f"\n  Evaluating {name}...")
            
            try:
                y_pred = model.predict(self.X_test)
                y_proba = model.predict_proba(self.X_test)
                
                # Metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                
                # ROC-AUC
                try:
                    y_bin = label_binarize(self.y_test, classes=np.arange(self.n_classes))
                    roc_auc = roc_auc_score(y_bin, y_proba, average='weighted', multi_class='ovr')
                except:
                    roc_auc = 0.5
                
                # Average Precision
                try:
                    avg_prec = average_precision_score(y_bin, y_proba, average='weighted')
                except:
                    avg_prec = 0.0
                
                self.results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'avg_precision': avg_prec,
                    'y_pred': y_pred,
                    'y_proba': y_proba,
                    'confusion_matrix': confusion_matrix(self.y_test, y_pred)
                }
                
                logger.info(f"    Accuracy: {accuracy:.4f}")
                logger.info(f"    F1 (weighted): {f1:.4f}")
                logger.info(f"    ROC-AUC: {roc_auc:.4f}")
                
            except Exception as e:
                logger.warning(f"    Error evaluating {name}: {e}")
        
        # Find best model
        best_name = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        logger.info(f"\n  Best model: {best_name} (ROC-AUC: {self.results[best_name]['roc_auc']:.4f})")
    
    def generate_model_comparison_table(self):
        """Generate model comparison table for PPT"""
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING MODEL COMPARISON TABLE")
        logger.info("=" * 70)
        
        rows = []
        for name in self.results:
            r = self.results[name]
            rows.append({
                'Model': name,
                'ROC-AUC': round(r['roc_auc'], 4),
                'Avg Precision': round(r['avg_precision'], 4),
                'F1': round(r['f1'], 4),
                'Precision': round(r['precision'], 4),
                'Recall': round(r['recall'], 4),
                'Accuracy': round(r['accuracy'], 4)
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
        
        # Save table
        table_path = TABLES_DIR / 'model_comparison_table.csv'
        df.to_csv(table_path, index=False)
        logger.info(f"  Saved: {table_path.name}")
        
        # Print table
        logger.info("\n  MODEL COMPARISON TABLE:")
        logger.info("-" * 90)
        logger.info(f"  {'Model':<22} {'ROC-AUC':>10} {'Avg Prec':>10} {'F1':>8} {'Precision':>10} {'Recall':>8}")
        logger.info("-" * 90)
        for _, row in df.iterrows():
            logger.info(
                f"  {row['Model']:<22} {row['ROC-AUC']:>10.4f} {row['Avg Precision']:>10.4f} "
                f"{row['F1']:>8.4f} {row['Precision']:>10.4f} {row['Recall']:>8.4f}"
            )
        logger.info("-" * 90)
        
        return df
    
    def plot_roc_curves(self):
        """Generate ROC curves for all models"""
        logger.info("\n  Generating ROC curves...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        y_bin = label_binarize(self.y_test, classes=np.arange(self.n_classes))
        
        for name in self.results:
            if name == 'Random Baseline':
                continue
                
            y_proba = self.results[name]['y_proba']
            
            # Compute micro-average ROC
            fpr, tpr, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
            roc_auc = auc(fpr, tpr)
            
            color = MODEL_COLORS.get(name, '#666666')
            linewidth = 3 if name == self.best_model_name else 2
            linestyle = '-' if name in ['XGBoost', 'LightGBM', 'XGB+LGB Ensemble'] else '--'
            
            ax.plot(fpr, tpr, color=color, linewidth=linewidth, linestyle=linestyle,
                   label=f'{name} (AUC = {roc_auc:.3f})')
        
        # Reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title('ROC Curves Comparison - All Models', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add shaded area for best model
        if self.best_model_name and self.best_model_name != 'Random Baseline':
            y_proba_best = self.results[self.best_model_name]['y_proba']
            fpr_best, tpr_best, _ = roc_curve(y_bin.ravel(), y_proba_best.ravel())
            ax.fill_between(fpr_best, tpr_best, alpha=0.2, 
                           color=MODEL_COLORS.get(self.best_model_name, '#666666'))
        
        plt.tight_layout()
        fig_path = FIGURES_DIR / 'roc_curves_comparison.png'
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        logger.info(f"    Saved: {fig_path.name}")
    
    def plot_precision_recall_curves(self):
        """Generate Precision-Recall curves"""
        logger.info("  Generating Precision-Recall curves...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        y_bin = label_binarize(self.y_test, classes=np.arange(self.n_classes))
        
        for name in self.results:
            if name == 'Random Baseline':
                continue
                
            y_proba = self.results[name]['y_proba']
            
            # Compute micro-average PR curve
            precision_arr, recall_arr, _ = precision_recall_curve(y_bin.ravel(), y_proba.ravel())
            ap = average_precision_score(y_bin, y_proba, average='micro')
            
            color = MODEL_COLORS.get(name, '#666666')
            linewidth = 3 if name == self.best_model_name else 2
            
            ax.plot(recall_arr, precision_arr, color=color, linewidth=linewidth,
                   label=f'{name} (AP = {ap:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_title('Precision-Recall Curves Comparison', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = FIGURES_DIR / 'precision_recall_curves.png'
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        logger.info(f"    Saved: {fig_path.name}")
    
    def plot_confusion_matrix(self):
        """Generate confusion matrix for best model"""
        logger.info("  Generating confusion matrix...")
        
        cm = self.results[self.best_model_name]['confusion_matrix']
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=CLASS_NAMES[:self.n_classes],
                   yticklabels=CLASS_NAMES[:self.n_classes],
                   cbar_kws={'label': 'Count'})
        axes[0].set_title(f'Confusion Matrix (Counts)\n{self.best_model_name}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted', fontsize=12)
        axes[0].set_ylabel('Actual', fontsize=12)
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[1],
                   xticklabels=CLASS_NAMES[:self.n_classes],
                   yticklabels=CLASS_NAMES[:self.n_classes],
                   cbar_kws={'label': 'Percentage'})
        axes[1].set_title(f'Confusion Matrix (Normalized)\n{self.best_model_name}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted', fontsize=12)
        axes[1].set_ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        fig_path = FIGURES_DIR / 'confusion_matrix_best_model.png'
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        logger.info(f"    Saved: {fig_path.name}")
    
    def generate_shap_explanations(self):
        """Generate SHAP explanations and plots"""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping explanations")
            return
        
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING SHAP EXPLANATIONS")
        logger.info("=" * 70)
        
        # Use XGBoost for SHAP (most reliable)
        if 'XGBoost' not in self.models:
            logger.warning("XGBoost not available for SHAP")
            return
        
        model = self.models['XGBoost']
        
        # Sample for faster computation
        n_samples = min(1000, len(self.X_val))
        X_sample = self.X_val.sample(n=n_samples, random_state=42)
        
        logger.info(f"  Computing SHAP values for {n_samples} samples...")
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                # Average across all classes
                shap_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            elif shap_values.ndim == 3:
                shap_mean = np.abs(shap_values).mean(axis=(0, 2))
            else:
                shap_mean = np.abs(shap_values)
            
            # Feature importance from SHAP
            shap_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(shap_mean).mean(axis=0) if shap_mean.ndim > 1 else shap_mean
            }).sort_values('importance', ascending=False)
            
            # Save SHAP importance
            shap_importance.to_csv(TABLES_DIR / 'shap_feature_importance.csv', index=False)
            
            # SHAP Summary Beeswarm Plot
            logger.info("  Generating SHAP summary plot...")
            fig, ax = plt.subplots(figsize=(12, 10))
            
            if isinstance(shap_values, list):
                # For multi-class, use class 0 (Critical) for visualization
                shap.summary_plot(shap_values[0], X_sample, show=False, max_display=15)
            else:
                shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
            
            plt.title('SHAP Feature Importance (Beeswarm)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            fig = plt.gcf()
            fig_path = FIGURES_DIR / 'shap_summary_beeswarm.png'
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close('all')
            logger.info(f"    Saved: {fig_path.name}")
            
            # SHAP Feature Importance Bar Chart
            logger.info("  Generating SHAP bar chart...")
            fig, ax = plt.subplots(figsize=(12, 10))
            
            top_n = 15
            top_features = shap_importance.head(top_n)
            
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, top_n))[::-1]
            bars = ax.barh(top_features['feature'][::-1], 
                          top_features['importance'][::-1],
                          color=colors)
            
            ax.set_xlabel('Mean |SHAP Value|', fontsize=14)
            ax.set_title('Top 15 Features by SHAP Importance', fontsize=16, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            fig_path = FIGURES_DIR / 'shap_feature_importance.png'
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            logger.info(f"    Saved: {fig_path.name}")
            
            logger.info("  SHAP explanations complete ✓")
            
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
    
    def plot_calibration_curves(self):
        """Generate calibration curves"""
        logger.info("  Generating calibration curves...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Models to compare
        models_to_plot = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'XGB+LGB Ensemble']
        models_to_plot = [m for m in models_to_plot if m in self.results]
        
        # Per-class calibration (using class 0 - Critical)
        for name in models_to_plot:
            y_proba = self.results[name]['y_proba']
            
            # Binary for class 0
            y_true_binary = (self.y_test == 0).astype(int)
            y_prob_class0 = y_proba[:, 0]
            
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true_binary, y_prob_class0, n_bins=10
                )
                
                color = MODEL_COLORS.get(name, '#666666')
                axes[0].plot(mean_predicted_value, fraction_of_positives, 
                           's-', color=color, linewidth=2, label=name)
            except:
                pass
        
        # Perfect calibration line
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')
        axes[0].set_xlabel('Mean Predicted Probability', fontsize=14)
        axes[0].set_ylabel('Fraction of Positives', fontsize=14)
        axes[0].set_title("Calibration Curve (Critical Class)", fontsize=14, fontweight='bold')
        axes[0].legend(loc='lower right', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Brier score comparison
        brier_scores = {}
        for name in models_to_plot:
            y_proba = self.results[name]['y_proba']
            total_brier = 0
            for i in range(self.n_classes):
                y_true_binary = (self.y_test == i).astype(int)
                total_brier += brier_score_loss(y_true_binary, y_proba[:, i])
            brier_scores[name] = total_brier / self.n_classes
        
        names = list(brier_scores.keys())
        scores = list(brier_scores.values())
        colors = [MODEL_COLORS.get(n, '#666666') for n in names]
        
        bars = axes[1].barh(names, scores, color=colors)
        axes[1].set_xlabel('Brier Score (lower is better)', fontsize=14)
        axes[1].set_title('Calibration Quality (Brier Score)', fontsize=14, fontweight='bold')
        axes[1].invert_xaxis()
        axes[1].grid(axis='x', alpha=0.3)
        
        # Add values on bars
        for bar, score in zip(bars, scores):
            axes[1].text(score - 0.005, bar.get_y() + bar.get_height()/2, 
                        f'{score:.4f}', va='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        fig_path = FIGURES_DIR / 'calibration_curve.png'
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        logger.info(f"    Saved: {fig_path.name}")
    
    def plot_threshold_analysis(self):
        """Generate threshold analysis plot"""
        logger.info("  Generating threshold analysis...")
        
        if self.best_model_name not in self.results:
            return
        
        y_proba = self.results[self.best_model_name]['y_proba']
        y_bin = label_binarize(self.y_test, classes=np.arange(self.n_classes))
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        for cls_idx in range(min(4, self.n_classes)):
            ax = axes[cls_idx // 2, cls_idx % 2]
            
            precision_arr, recall_arr, thresholds = precision_recall_curve(
                y_bin[:, cls_idx], y_proba[:, cls_idx]
            )
            
            f1_scores = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-10)
            
            # Extend thresholds
            thresholds_plot = np.append(thresholds, 1.0)
            
            ax.plot(thresholds_plot, precision_arr, 'b-', linewidth=2, label='Precision')
            ax.plot(thresholds_plot, recall_arr, 'g-', linewidth=2, label='Recall')
            ax.plot(thresholds_plot, f1_scores, 'r-', linewidth=2, label='F1 Score')
            
            # Mark optimal threshold
            best_idx = np.argmax(f1_scores[:-1]) if len(thresholds) > 0 else 0
            if best_idx < len(thresholds):
                opt_thresh = thresholds[best_idx]
                ax.axvline(x=opt_thresh, color='purple', linestyle='--', linewidth=2, 
                          label=f'Optimal: {opt_thresh:.3f}')
            
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f'Class {cls_idx}'
            ax.set_title(f'{cls_name} - Threshold Analysis', fontsize=12, fontweight='bold')
            ax.set_xlabel('Threshold', fontsize=11)
            ax.set_ylabel('Score', fontsize=11)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
        
        plt.suptitle(f'Threshold Analysis - {self.best_model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig_path = FIGURES_DIR / 'threshold_analysis.png'
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        logger.info(f"    Saved: {fig_path.name}")
    
    def plot_learning_curves(self):
        """Generate learning curves"""
        logger.info("  Generating learning curves...")
        
        # Use XGBoost for learning curve
        if 'XGBoost' not in self.models:
            logger.warning("XGBoost not available for learning curves")
            return
        
        # Create fresh model for learning curve
        model = xgb.XGBClassifier(
            n_estimators=100,  # Fewer for speed
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            verbosity=0,
            random_state=42,
            n_jobs=-1
        )
        
        # Use non-resampled training data
        train_df = pd.read_parquet(DATA_DIR / 'ml_train.parquet')
        X_train_orig = train_df.drop(columns=['target'])
        y_train_orig = train_df['target']
        
        # Sample for speed
        sample_size = min(5000, len(X_train_orig))
        sample_idx = np.random.choice(len(X_train_orig), sample_size, replace=False)
        X_sample = X_train_orig.iloc[sample_idx]
        y_sample = y_train_orig.iloc[sample_idx]
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        try:
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X_sample, y_sample,
                train_sizes=train_sizes,
                cv=3,
                scoring='f1_weighted',
                n_jobs=-1,
                shuffle=True,
                random_state=42
            )
            
            train_mean = train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            val_mean = val_scores.mean(axis=1)
            val_std = val_scores.std(axis=1)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                           alpha=0.2, color='blue')
            ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                           alpha=0.2, color='orange')
            
            ax.plot(train_sizes_abs, train_mean, 'b-o', linewidth=2, markersize=8, 
                   label='Training Score')
            ax.plot(train_sizes_abs, val_mean, 'r-s', linewidth=2, markersize=8, 
                   label='Cross-Validation Score')
            
            ax.set_xlabel('Training Set Size', fontsize=14)
            ax.set_ylabel('F1 Score (Weighted)', fontsize=14)
            ax.set_title('Learning Curve - XGBoost', fontsize=16, fontweight='bold')
            ax.legend(loc='lower right', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add annotations
            ax.annotate(f'Final Train: {train_mean[-1]:.3f}', 
                       xy=(train_sizes_abs[-1], train_mean[-1]),
                       xytext=(train_sizes_abs[-1] - 500, train_mean[-1] + 0.02),
                       fontsize=10, fontweight='bold')
            ax.annotate(f'Final CV: {val_mean[-1]:.3f}', 
                       xy=(train_sizes_abs[-1], val_mean[-1]),
                       xytext=(train_sizes_abs[-1] - 500, val_mean[-1] - 0.03),
                       fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            fig_path = FIGURES_DIR / 'learning_curve.png'
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            logger.info(f"    Saved: {fig_path.name}")
            
        except Exception as e:
            logger.error(f"Learning curve failed: {e}")
    
    def plot_feature_importance(self):
        """Generate feature importance plot"""
        logger.info("  Generating feature importance plot...")
        
        if 'XGBoost' not in self.models:
            return
        
        importance = self.models['XGBoost'].feature_importances_
        
        imp_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Save to CSV
        imp_df.to_csv(TABLES_DIR / 'feature_importance_all_models.csv', index=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        top_n = 15
        top_features = imp_df.head(top_n)
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
        bars = ax.barh(top_features['feature'][::-1], 
                      top_features['importance'][::-1],
                      color=colors)
        
        ax.set_xlabel('Feature Importance (Gain)', fontsize=14)
        ax.set_title('Top 15 Features - XGBoost', fontsize=16, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add values
        for bar, imp in zip(bars, top_features['importance'][::-1]):
            ax.text(imp + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{imp:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        fig_path = FIGURES_DIR / 'xgboost_feature_importance.png'
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        logger.info(f"    Saved: {fig_path.name}")
    
    def save_models(self):
        """Save trained models"""
        logger.info("\n" + "=" * 70)
        logger.info("SAVING MODELS")
        logger.info("=" * 70)
        
        for name, model in self.models.items():
            if name in ['Random Baseline', 'XGB+LGB Ensemble']:
                continue  # Skip these
            
            try:
                safe_name = name.lower().replace(' ', '_').replace('+', '_')
                model_path = MODELS_DIR / f'risk_classifier_{safe_name}.pkl'
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                logger.info(f"  ✅ {name}: {model_path.name}")
            except Exception as e:
                logger.warning(f"  Failed to save {name}: {e}")
        
        # Save ensemble components separately
        if 'XGB+LGB Ensemble' in self.models:
            ensemble_path = MODELS_DIR / 'risk_classifier_ensemble.pkl'
            with open(ensemble_path, 'wb') as f:
                pickle.dump({
                    'xgb': self.models.get('XGBoost'),
                    'lgb': self.models.get('LightGBM'),
                    'xgb_weight': 0.5,
                    'lgb_weight': 0.5
                }, f)
            logger.info(f"  ✅ Ensemble: {ensemble_path.name}")
    
    def generate_training_report(self):
        """Generate comprehensive training report"""
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING TRAINING REPORT")
        logger.info("=" * 70)
        
        report = {
            'model_type': 'Patient Risk Classifier',
            'version': '2.0.0',
            'created_at': datetime.now().isoformat(),
            'data': {
                'training_samples': len(self.X_train),
                'validation_samples': len(self.X_val),
                'test_samples': len(self.X_test),
                'n_features': len(self.feature_names),
                'n_classes': self.n_classes,
                'class_names': CLASS_NAMES[:self.n_classes]
            },
            'best_model': self.best_model_name,
            'best_metrics': {
                k: v for k, v in self.results[self.best_model_name].items() 
                if k not in ['y_pred', 'y_proba', 'confusion_matrix']
            },
            'all_models': {
                name: {k: v for k, v in res.items() if k not in ['y_pred', 'y_proba', 'confusion_matrix']}
                for name, res in self.results.items()
            },
            'feature_names': self.feature_names
        }
        
        # Convert numpy types to Python types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        report = convert_numpy(report)
        
        report_path = OUTPUT_DIR / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"  Saved: {report_path.name}")
        
        # Generate markdown report
        md_report = self._generate_markdown_report()
        md_path = OUTPUT_DIR / 'training_report.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        logger.info(f"  Saved: {md_path.name}")
    
    def _generate_markdown_report(self) -> str:
        """Generate markdown training report"""
        best = self.results[self.best_model_name]
        
        md = f"""# Patient Risk Classifier - Training Report

## 🎯 Model Overview

**Model Type**: 4-Class Patient Risk Classifier  
**Best Model**: {self.best_model_name}  
**Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Data Summary

| Metric | Value |
|--------|-------|
| Training Samples | {len(self.X_train):,} |
| Validation Samples | {len(self.X_val):,} |
| Test Samples | {len(self.X_test):,} |
| Features | {len(self.feature_names)} |
| Classes | {', '.join(CLASS_NAMES[:self.n_classes])} |

## 🏆 Best Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | {best['roc_auc']:.4f} |
| Average Precision | {best['avg_precision']:.4f} |
| F1 (Weighted) | {best['f1']:.4f} |
| Precision | {best['precision']:.4f} |
| Recall | {best['recall']:.4f} |
| Accuracy | {best['accuracy']:.4f} |

## 📈 Model Comparison

| Model | ROC-AUC | Avg Precision | F1 | Accuracy |
|-------|---------|---------------|-----|----------|
"""
        
        for name in sorted(self.results.keys(), key=lambda x: self.results[x]['roc_auc'], reverse=True):
            r = self.results[name]
            md += f"| {name} | {r['roc_auc']:.4f} | {r['avg_precision']:.4f} | {r['f1']:.4f} | {r['accuracy']:.4f} |\n"
        
        md += f"""

## 📁 Generated Files

### Figures
- `roc_curves_comparison.png` - ROC curves for all models
- `precision_recall_curves.png` - Precision-Recall curves
- `confusion_matrix_best_model.png` - Confusion matrix
- `shap_summary_beeswarm.png` - SHAP beeswarm plot
- `shap_feature_importance.png` - SHAP feature importance
- `calibration_curve.png` - Calibration curves
- `threshold_analysis.png` - Threshold optimization
- `learning_curve.png` - Learning curves

### Tables
- `model_comparison_table.csv` - All model metrics
- `feature_importance_all_models.csv` - Feature importance
- `shap_feature_importance.csv` - SHAP importance

### Models
- `risk_classifier_xgboost.pkl` - Trained XGBoost model
- `risk_classifier_lightgbm.pkl` - Trained LightGBM model
- `risk_classifier_ensemble.pkl` - Ensemble model

## ✅ Training Complete

All visualizations and models have been saved to `{OUTPUT_DIR.relative_to(PROJECT_ROOT)}`
"""
        return md
    
    def run(self):
        """Run complete training pipeline"""
        start_time = datetime.now()
        
        logger.info("\n" + "=" * 70)
        logger.info("TRIALPULSE NEXUS 10X - COMPREHENSIVE RISK CLASSIFIER TRAINING")
        logger.info("=" * 70)
        logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        if not self.load_data():
            logger.error("Failed to load data. Exiting.")
            return False
        
        # Train models
        self.train_baseline_models()
        self.train_advanced_models()
        
        # Evaluate
        self.evaluate_all_models()
        self.generate_model_comparison_table()
        
        # Generate visualizations
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 70)
        
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        self.plot_confusion_matrix()
        self.generate_shap_explanations()
        self.plot_calibration_curves()
        self.plot_threshold_analysis()
        self.plot_learning_curves()
        self.plot_feature_importance()
        
        # Save
        self.save_models()
        self.generate_training_report()
        
        # Summary
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"  Best Model: {self.best_model_name}")
        logger.info(f"  ROC-AUC: {self.results[self.best_model_name]['roc_auc']:.4f}")
        logger.info(f"  F1 Score: {self.results[self.best_model_name]['f1']:.4f}")
        logger.info(f"\n  Output Directory: {OUTPUT_DIR}")
        logger.info(f"  Figures: {FIGURES_DIR}")
        logger.info(f"  Models: {MODELS_DIR}")
        logger.info(f"  Tables: {TABLES_DIR}")
        
        return True


def main():
    """Main entry point"""
    trainer = ComprehensiveRiskTrainer()
    success = trainer.run()
    
    if success:
        print("\n" + "=" * 70)
        print("✅ TRAINING SUCCESSFUL")
        print("=" * 70)
        print(f"\nAll outputs saved to: {OUTPUT_DIR}")
        print("\nVisualization files for PPT:")
        for fig in FIGURES_DIR.glob('*.png'):
            print(f"  📊 {fig.name}")
    else:
        print("\n❌ TRAINING FAILED - Check logs for details")
        sys.exit(1)


if __name__ == '__main__':
    main()
