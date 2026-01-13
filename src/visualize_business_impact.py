import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from typing import Optional
from utils.logger_config import setup_logger

logger = setup_logger(__name__)

def load_config() -> dict:
    """Loads project configuration from YAML file."""
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_business_plots(data_path: Optional[str] = None, plots_dir: Optional[str] = None) -> None:
    """
    Generates high-level business impact plots based on raw data and model predictions.

    Args:
        data_path: Path to the input CSV data.
        plots_dir: Directory where business impact plots will be saved.
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

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading data for business plots from {data_path}...")
    df = pd.read_csv(data_path)
    
    # --- 1. Empirical Conversion Rate by Decile ---
    logger.info("Generating Conversion Rate by Decile plot...")
    df['interest_decile'] = pd.qcut(df['promotional_interest'], 10, labels=False, duplicates='drop')
    decile_stats = df.groupby('interest_decile')['booking_conversion'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(decile_stats['interest_decile'], decile_stats['booking_conversion'], color='#2ca02c', alpha=0.8)
    
    plt.suptitle('Validation: Real-World Conversion Rate by Interest Tier', fontsize=14)
    plt.title("Actual booking rates observed within each decile of the model's promotional interest score", 
              fontsize=10, style='italic', pad=10)
    plt.xlabel('Promotional Interest Score (Decile 0=Low, 9=High)', fontsize=11)
    plt.ylabel('Conversion Rate (Actual %)', fontsize=11)
    plt.xticks(decile_stats['interest_decile'])
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.1%}', ha='center', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plots_dir / 'business_conversion_by_interest.png', dpi=300)
    plt.close()

    # --- 2. Distribution Plot ---
    logger.info("Generating Engagement Score Distribution plot...")
    plt.figure(figsize=(10, 6))
    kwargs = dict(alpha=0.5, bins=25, density=True)
    plt.hist(df[df['booking_conversion']==0]['promotional_interest'], label='Did NOT Book', color='gray', **kwargs)
    plt.hist(df[df['booking_conversion']==1]['promotional_interest'], label='DID Book', color='blue', **kwargs)
    
    plt.suptitle('Customer Concentration by Engagement Score', fontsize=14)
    plt.title("Density distribution showing how promotional interest scores differ between customers who booked vs. those who didn't", 
              fontsize=10, style='italic', pad=10)
    plt.xlabel('Promotional Interest Score (0.0 to 1.0)', fontsize=11)
    plt.ylabel('Density (Relative Concentration)', fontsize=11)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plots_dir / 'business_interest_distribution.png', dpi=300)
    plt.close()

    logger.info(f"Business plots saved to {plots_dir}")

if __name__ == "__main__":
    generate_business_plots()
