# Car Service Booking Analysis: AdaBoost & SHAP

This project demonstrates a production-ready machine learning pipeline using **AdaBoost** and **SHAP** (Shapley additive explanations). It predicts car service bookings based on customer engagement and vehicle data, providing both high accuracy and clear business explainability.

---

## 🚀 Project Overview
The objective is to quantify the impact of specific features—such as `promotional_interest`—on booking probability. We translate complex model outputs into actionable business insights (e.g., "Probability Lift").

### 📁 Repository Structure
```text
├── config/
│   └── config.yaml                  # Centralized project configuration
├── data/                            # Local data storage (ignored by git)
├── logs/                            # Centralized execution logs
├── plots/                           # Output visualizations (ignored by git)
├── src/                             # Source code
│   ├── utils/
│   │   └── logger_config.py         # Standardized logging utility
│   ├── data_generator.py            # Synthetic data generation logic
│   ├── train_adaboost.py            # Training, validation, and SHAP analysis
│   ├── interpret_shap.py            # SHAP probability interpretation & business insights
│   └── visualize_business_impact.py # Business-centric validation plots
├── run_pipeline.bat                 # Windows automation script
├── run_pipeline.sh                  # Linux/Mac automation script
├── requirements.txt                 # Pinned dependencies
└── LICENSE                          # MIT License
```

## 📊 Business Metrics Summary
| Metric | Value* | Business Meaning |
| :--- | :--- | :--- |
| **Model Accuracy** | ~0.94 AUC | How well the model separates "likely bookers" from "unlikely bookers." |
| **Primary Driver** | `promotional_interest` | The single most important factor in predicting if someone will book. |
| **Probability Lift** | Up to +40% | The increase in booking chance for customers with high engagement. |
| **High-Value Segment** | Top 20% | The portion of customers responsible for the majority of conversions. |

*\*Sample values based on the default synthetic dataset. These figures are automatically calculated by the pipeline during execution.*

### 💡 Where do these numbers come from?
- **The +40% Lift:** This is calculated by our "Explainability" tool (SHAP). It looks at thousands of customers and measures how much a high interest score "boosts" their individual probability of booking compared to the average.
- **The Top 20% Segment:** Our analysis groups customers into 10 groups (deciles) based on their interest. We consistently find that the top two groups contain the highest concentration of actual bookings, making them the primary targets for marketing.

## 🛠️ Installation & Usage

### 1. Setup Environment
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 2. Run the Pipeline
You can run the entire process (data → training → plots) with a single command:

**Windows:**
```powershell
./run_pipeline.bat
```

**Linux/Mac:**
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### 3. Generate Business Insights
After training the model, run the interpretation script to generate a plain-English report on feature impacts:
```bash
python src/interpret_shap.py
```
**Key Features:**
*   **Efficiency**: Loads cached values from `train_adaboost.py` to generate insights instantly.
*   **Human-Readable Insights**: Explains probability impact in terms of a "typical customer" (e.g., "shifts probability from 47% to 61%").
*   **Report**: Creates `logs/shap_business_insight.txt` with detailed probability analysis and strategic recommendations.

**Sample Output:**
```text
Feature: promotional_interest
  - Impact Analysis: For a typical customer (base probability 47%), when this feature is favorable, it shifts the booking likelihood to **61%** (+14.3%).
  - Trend: Higher values increase booking probability.
  - Business Recommendation: A/B test aggressive discounts vs. value-add offers for high-interest segments.
```

---

## 🔍 Key Findings & Visual Analysis

All visualizations are generated during the pipeline execution and saved to the `plots/` directory. These files are excluded from the repository to ensure you are always viewing results from your specific environment and data version.

### 1. Model Reliability (`src/train_adaboost.py`)
- **ROC Curve (`cv_roc_curve.png`)**: Demonstrates the model's ability to distinguish between classes. A high AUC (~0.94) indicates excellent predictive power across different thresholds.
- **Learning Curve (`cv_learning_curve.png`)**: Compares training and cross-validation accuracy as more data is added. This confirms that the model generalizes well and benefits from the dataset size without overfitting.
- **Confusion Matrix (`cv_confusion_matrix.png`)**: Provides a detailed breakdown of true positives/negatives and false alarms, aggregated across all cross-validation folds.

### 2. Explainability (`src/train_adaboost.py`)
- **SHAP Summary (`shap_summary.png`)**: A beeswarm plot showing how high or low values of each feature (like `promotional_interest`) push the model's prediction toward or away from a booking.
- **Probability Lift (`shap_probability_lift.png`)**: Quantifies the business impact by showing exactly how many percentage points of booking probability are added (or removed) based on a customer's engagement score.
- **Feature Importance (`feature_importances.png`)**: A global view of which variables the AdaBoost algorithm found most useful for its decision trees.
- **Business Insight Report (`logs/shap_business_insight.txt`)**: A generated text report that translates technical SHAP values into "typical customer" probability shifts and actionable business recommendations.

### 3. Empirical Validation (`src/visualize_business_impact.py`)
- **Conversion by Tier (`business_conversion_by_interest.png`)**: A bar chart grouping customers into deciles based on interest. It validates the model by showing that real-world conversion rates scale linearly with the predicted interest scores.
- **Interest Distribution (`business_interest_distribution.png`)**: Highlights the separation between customers who booked and those who didn't, showing where the highest density of conversions occurs.

---

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author
**Louie Cawood**
