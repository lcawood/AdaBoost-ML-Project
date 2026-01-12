import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import os
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

def generate_car_service_data(output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generates a synthetic dataset for a Car Service Booking scenario.

    Args:
        output_path: The path where the generated CSV will be saved. 
            Defaults to the path specified in config.yaml.

    Returns:
        The generated dataframe.
    """
    config = load_config()
    gen_config = config['data_generation']
    
    logger.info(f"Generating {gen_config['n_samples']} samples...")

    # 1. Generate base synthetic data using make_classification
    X, y = make_classification(
        n_samples=gen_config['n_samples'],
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        weights=[0.7, 0.3],
        flip_y=0.05,
        random_state=config['project']['random_state']
    )

    # 2. Convert to DataFrame
    feature_names = [
        'vehicle_age', 'annual_mileage', 'previous_bookings',
        'last_service_months', 'vehicle_value', 'daily_commute_miles',
        'customer_income', 'marketing_engagement_score',
        'competitor_distance_km', 'promotional_interest'
    ]
    
    df = pd.DataFrame(X, columns=feature_names)
    
    def rescale(series: pd.Series, min_val: float, max_val: float) -> pd.Series:
        """Helper to rescale a column to a range [min, max]."""
        s_min, s_max = series.min(), series.max()
        if s_max == s_min:
            return pd.Series(np.full_like(series, (min_val + max_val) / 2))
        normalized = (series - s_min) / (s_max - s_min)
        return normalized * (max_val - min_val) + min_val

    # 3. Apply semantic mapping
    df['vehicle_age'] = rescale(df['vehicle_age'], 0, 20).astype(int)
    df['annual_mileage'] = rescale(df['annual_mileage'], 5000, 50000).astype(int)
    df['previous_bookings'] = rescale(df['previous_bookings'], 0, 10).astype(int)
    df['last_service_months'] = rescale(df['last_service_months'], 0, 48).astype(int)
    df['vehicle_value'] = rescale(df['vehicle_value'], 5000, 100000).astype(int)
    df['daily_commute_miles'] = rescale(df['daily_commute_miles'], 0, 100).astype(int)
    df['customer_income'] = rescale(df['customer_income'], 20000, 200000).astype(int)
    df['marketing_engagement_score'] = rescale(df['marketing_engagement_score'], 0, 10).round(1)
    df['competitor_distance_km'] = rescale(df['competitor_distance_km'], 0, 50).round(1)
    df['promotional_interest'] = rescale(df['promotional_interest'], 0, 1).round(2)

    df['booking_conversion'] = y

    # 4. Save to CSV
    if output_path is None:
        project_root = Path(__file__).parent.parent
        output_path = project_root / gen_config['output_file']
    else:
        output_path = Path(output_path)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
        raise

    return df

if __name__ == "__main__":
    generate_car_service_data()
