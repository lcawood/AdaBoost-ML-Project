
import pandas as pd
import numpy as np
import joblib
import yaml
from pathlib import Path
from typing import Tuple
from utils.logger_config import setup_logger

# Bridge the gap between different versions of the math library (NumPy).
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

if not hasattr(np, 'in1d'):
    np.in1d = np.isin

logger = setup_logger(__name__)

def load_config() -> dict:
    """Loads project configuration from YAML file."""
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_business_recommendation(feature_name: str, impact_direction: str) -> str:
    """Returns a business recommendation based on the feature and its impact."""
    
    recommendations = {
        'vehicle_age': {
            'positive': "Target owners of vehicles 5+ years old with 'aging vehicle' maintenance packages.",
            'negative': "Highlight warranty-safe servicing for newer vehicles to capture early lifecycle customers."
        },
        'annual_mileage': {
            'positive': "Offer interval-based tier discounts for high-mileage drivers.",
            'negative': "Promote time-based 'annual checkups' for low-mileage drivers to prevent stagnation issues."
        },
        'previous_bookings': {
            'positive': "Implement a VIP priority lane for customers with 3+ previous bookings.",
            'negative': "Offer 'First Time Booking' discount code to convert new leads."
        },
        'last_service_months': {
            'positive': "Urgent: Send automated re-engagement emails for service gaps > 12 months.",
            'negative': "Reinforce regular maintenance habits with 'on-time' loyalty points."
        },
        'vehicle_value': {
            'positive': "Upsell 'detail and protect' add-ons to high-value vehicle owners.",
            'negative': "Focus on 'essential maintenance' packages for budget-conscious owners."
        },
        'daily_commute_miles': {
            'positive': "Partner with local employers for commuter fleet discounts.",
            'negative': "Target weekend drivers with 'leisure check' specials."
        },
        'customer_income': {
            'positive': "Feature premium convenience services (pickup/drop-off) for higher income segments.",
            'negative': "Emphasize value-for-money and flexible payment plans for budget-sensitive segments."
        },
        'marketing_engagement_score': {
            'positive': "Immediate call-to-action for high scorers; they are ready to buy.",
            'negative': "Nurture low scorers with educational content to build trust."
        },
        'competitor_distance_km': {
            'positive': "Geofence ads within 5km of competitors with 'We are worth the extra drive' incentives.",
            'negative': "Focus on 'local convenience' messaging for nearby customers."
        },
        'promotional_interest': {
            'positive': "Invest in high-interest segments with time-limited VIP offers.",
            'negative': "Re-engage low-interest users with brand awareness campaigns."
        }
    }
    
    feature_recs = recommendations.get(feature_name, {})
    return feature_recs.get(impact_direction, "Monitor this feature for emerging trends.")

def get_cached_shap_data(project_root: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Loads cached SHAP values and data."""
    cache_path = project_root / 'data' / 'shap_values_data.pkl'
    if not cache_path.exists():
        raise FileNotFoundError(f"Cached SHAP data not found at {cache_path}. Run train_adaboost.py first.")
    
    data = joblib.load(cache_path)
    return data['shap_values'], data['data']

def interpret_shap_analysis():
    """
    Loads cached SHAP values and generates a business insight report.
    """
    project_root = Path(__file__).parent.parent
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading cached SHAP values...")
    try:
        shap_values_obj, X_explain = get_cached_shap_data(project_root)
        # shap_values_obj can be an Explanation object or array. 
        # If it's an Explanation object, .values gives the array.
        if hasattr(shap_values_obj, 'values'):
            shap_values = shap_values_obj.values
            base_values = shap_values_obj.base_values
        else:
            shap_values = shap_values_obj
            # If we don't have base values from array, we must estimate or assumes it's handled
            # But the Explanation object is best. train_adaboost saves the result of explainer(X) which is an Explanation object.
            base_values = None

    except FileNotFoundError as e:
        logger.error(e)
        return

    feature_names = X_explain.columns.tolist()
    
    # Analyze Results
    logger.info("Generating Business Insight Report...")
    
    report_lines = []
    report_lines.append("SHAP Business Insight Report")
    report_lines.append("============================")
    report_lines.append("")
    
    # Calculate Base Rate (Typical Customer Probability)
    # The base value in SHAP (for TreeExplainer/Exact) is usually the mean prediction of the background dataset.
    # Since we are in probability space (predict_proba), this is the average booking probability.
    
    if base_values is not None:
        # base_values is often a single number for single-output models, or array for multi-output
        if isinstance(base_values, (np.ndarray, list)):
             base_rate = np.mean(base_values)
        else:
             base_rate = base_values
    else:
        # Fallback if likely not needed given the robust artifact saving
        base_rate = 0.3 # default or estimated from print
    
    report_lines.append(f"Typical Customer Booking Probability (Base Rate): {base_rate*100:.1f}%")
    report_lines.append("")

    # Sort by importance
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    
    for idx in sorted_indices:
        feature = feature_names[idx]
        shap_vals = shap_values[:, idx]
        feature_vals = X_explain[feature].values
        
        # 1. Average Impact
        avg_impact = np.mean(np.abs(shap_vals))
        
        # 2. Probability Shift Calculation
        # "For a typical customer (base probability ~30%), this feature shifts the booking likelihood to 44% (+14%)."
        # We use the average positive impact to demonstrate the potential "lift"
        
        # Filter for instances where the feature increased probability
        positive_impacts = shap_vals[shap_vals > 0]
        if len(positive_impacts) > 0:
            avg_positive_lift = np.mean(positive_impacts)
            new_prob = base_rate + avg_positive_lift
            # Ensure 0-1 bounds
            new_prob = min(max(new_prob, 0), 1)
            
            lift_desc = f"For a typical customer (base probability {base_rate*100:.0f}%), when this feature is favorable, it shifts the booking likelihood to **{new_prob*100:.0f}%** (+{avg_positive_lift*100:.1f}%)."
        else:
             lift_desc = "This feature predominantly decreases booking probability across the sample."

        # 3. Directionality
        correlation = np.corrcoef(feature_vals, shap_vals)[0, 1]
        
        if correlation > 0.1:
            direction_desc = "Higher values increase booking probability."
            impact_direction = "positive"
        elif correlation < -0.1:
            direction_desc = "Higher values decrease booking probability."
            impact_direction = "negative"
        else:
            direction_desc = "Complex relationship."
            impact_direction = "neutral"

        recommendation = get_business_recommendation(feature, impact_direction)

        report_lines.append(f"Feature: {feature}")
        report_lines.append(f"  - Impact Analysis: {lift_desc}")
        report_lines.append(f"  - Trend: {direction_desc}")
        report_lines.append(f"  - Business Recommendation: {recommendation}")
        report_lines.append("")

    output_file = logs_dir / 'shap_business_insight.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
        
    logger.info(f"Report saved to {output_file}")
    print(f"Analysis complete. Report saved to: {output_file}")

if __name__ == "__main__":
    interpret_shap_analysis()
