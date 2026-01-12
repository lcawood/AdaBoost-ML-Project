#!/bin/bash
set -e

echo "Starting AdaBoost Car Service Analysis Pipeline..."

echo "[1/3] Generating Synthetic Data..."
python3 src/data_generator.py

echo "[2/3] Training Model and Running SHAP Analysis..."
python3 src/train_adaboost.py

echo "[3/3] Generating Business Impact Visualizations..."
python3 src/visualize_business_impact.py

echo "Pipeline completed successfully!"
