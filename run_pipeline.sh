#!/bin/bash
set -e

echo "Starting AdaBoost Car Service Analysis Pipeline..."

echo "[1/4] Generating Synthetic Data..."
python3 src/data_generator.py

echo "[2/4] Training Model and Running SHAP Analysis..."
python3 src/train_adaboost.py

echo "[3/4] Generating Business Impact Visualizations..."
python3 src/visualize_business_impact.py

echo "[4/4] Generating Business Insights Report..."
python3 src/interpret_shap.py

echo "Pipeline completed successfully!"
