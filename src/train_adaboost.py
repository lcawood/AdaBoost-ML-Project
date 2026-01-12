import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, 
    accuracy_score, precision_score, recall_score, f1_score
)
import shap
from utils.logger_config import setup_logger

logger = setup_logger(__name__)

def load_config() -> dict:
    """Loads project configuration from YAML file."""
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_and_visualize(data_path: Optional[str] = None, plots_dir: Optional[str] = None) -> None:
    """
    Main pipeline to load data, train an AdaBoost model, and generate evaluation plots.

    Args:
        data_path: Path to the input CSV data.
        plots_dir: Directory where evaluation plots will be saved.
    """
    config = load_config()
    project_root = Path(__file__).parent.parent
    
    if data_path is None:
        data_path = project_root / config['data_generation']['output_file']
    else:
        data_path = Path(data_path)
        
    if plots_dir is None:
        plots_dir = project_root / config['paths']['plots_dir']
    else:
        plots_dir = Path(plots_dir)
    
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}. Please run data_generator.py first.")
        return

    # Load Data
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    X = df.drop('booking_conversion', axis=1)
    y = df['booking_conversion']
    feature_names = X.columns.tolist()

    # 1. Hold-out Split for Final Evaluation
    logger.info("Executing train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data_generation']['test_size'], 
        random_state=config['project']['random_state'],
        stratify=y
    )

    # 2. Model Initialization
    logger.info("Training final AdaBoost model on training set...")
    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=config['model']['max_depth']),
        n_estimators=config['model']['n_estimators'],
        learning_rate=config['model']['learning_rate'],
        random_state=config['project']['random_state']
    )
    clf.fit(X_train, y_train)

    # 3. Final Model Evaluation on Test Set
    y_pred = clf.predict(X_test)
    logger.info(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"Test Set F1 Score: {f1_score(y_test, y_pred):.4f}")

    # 4. Generate Visualizations
    logger.info("Generating plots...")
    plot_feature_importances(clf, feature_names, plots_dir)
    perform_cross_validation(clf, X, y, plots_dir, config)
    plot_learning_curve_func(clf, X, y, plots_dir)
    generate_shap_analysis(clf, X_test, feature_names, plots_dir, config)

def plot_feature_importances(clf: AdaBoostClassifier, feature_names: List[str], plots_dir: Path) -> None:
    """Generates and saves a feature importance plot."""
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(range(len(importances)), importances[indices], align="center", color='#1f77b4')
    plt.title("AdaBoost Feature Importances", fontsize=14, pad=20)
    plt.ylabel("Importance Score")
    plt.figtext(0.5, 0.90, "Top features contributing to the model's decision-making process", 
                ha='center', fontsize=10, style='italic')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importances.png')
    plt.close()

def perform_cross_validation(clf: AdaBoostClassifier, X: pd.DataFrame, y: pd.Series, plots_dir: Path, config: dict) -> None:
    """Performs K-fold cross-validation and saves ROC and Confusion Matrix plots."""
    logger.info("Starting K-Fold Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['project']['random_state'])
    
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    y_true_all, y_pred_all = [], []

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = clf.predict(X.iloc[test_idx])
        y_proba = clf.predict_proba(X.iloc[test_idx])[:, 1]
        
        y_true_all.extend(y.iloc[test_idx])
        y_pred_all.extend(y_pred)

        fpr, tpr, _ = roc_curve(y.iloc[test_idx], y_proba)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {i} (AUC = {roc_auc:.2f})')

    # Plot Mean ROC
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {np.std(aucs):.2f})', lw=2, alpha=.8)
    
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='Receiver Operating Characteristic (ROC)')
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plots_dir / 'cv_roc_curve.png')
    plt.close()

    # Aggregate Confusion Matrix
    cm = confusion_matrix(y_true_all, y_pred_all)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Booking", "Booked"])
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
    plt.title("Aggregate Cross-Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(plots_dir / 'cv_confusion_matrix.png')
    plt.close()

def plot_learning_curve_func(clf: AdaBoostClassifier, X: pd.DataFrame, y: pd.Series, plots_dir: Path) -> None:
    """Generates and saves the learning curve for the model."""
    logger.info("Generating Learning Curve...")
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring="accuracy"
    )

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training score")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g", label="Cross-validation score")
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.savefig(plots_dir / 'cv_learning_curve.png')
    plt.close()

def generate_shap_analysis(clf: AdaBoostClassifier, X_test: pd.DataFrame, feature_names: List[str], plots_dir: Path, config: dict) -> None:
    """Performs SHAP analysis using a general Explainer for AdaBoost."""
    logger.info("Generating SHAP analysis...")
    
    # AdaBoost requires a more general explainer or specifically masker-based
    def model_predict(data):
        return clf.predict_proba(data)[:, 1]

    # Sample data for background and explanation
    n_shap = min(config['shap']['n_samples'], len(X_test))
    X_explain = X_test.sample(n_shap, random_state=config['project']['random_state'])
    
    background = shap.maskers.Independent(X_explain, max_samples=config['shap']['max_background_samples'])
    explainer = shap.Explainer(model_predict, background)
    shap_values = explainer(X_explain)

    # SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values.values, X_explain, show=False)
    plt.title("SHAP Summary Plot", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(plots_dir / 'shap_summary.png')
    plt.close()

    # SHAP Feature Importance (Bar)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values.values, X_explain, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Mean |SHAP Value|)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(plots_dir / 'shap_feature_importance.png')
    plt.close()

    analyze_probability_impact(shap_values.values, X_explain, feature_names, plots_dir)

def analyze_probability_impact(shap_values: np.ndarray, X_explain: pd.DataFrame, feature_names: List[str], plots_dir: Path) -> None:
    """Translates SHAP values into individual probability lifts."""
    logger.info("Performing quantitative probability impact analysis...")
    target_feature = 'promotional_interest'
    if target_feature not in feature_names: return
    
    idx = feature_names.index(target_feature)
    
    # For tree-based models, shap_values are often in the margin space
    # We estimate probability lift by looking at the specific SHAP value for the feature
    # and comparing it to the overall distribution.
    
    prob_lift = shap_values[:, idx]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_explain[target_feature], prob_lift * 100, alpha=0.5, color='purple')
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.title(f"Quantitative Impact: {target_feature}", fontsize=14)
    plt.xlabel(f"Raw {target_feature} Score", fontsize=11)
    plt.ylabel("Booking Probability Lift (Estimated % Points)", fontsize=11)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(plots_dir / 'shap_probability_lift.png')
    plt.close()

if __name__ == "__main__":
    train_and_visualize()
