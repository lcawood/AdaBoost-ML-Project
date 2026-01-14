@echo off
echo Starting AdaBoost Car Service Analysis Pipeline...

echo [1/4] Generating Synthetic Data...
venv\Scripts\python "src/data_generator.py"
if %ERRORLEVEL% NEQ 0 (
    echo Error during data generation. Exiting.
    exit /b %ERRORLEVEL%
)

echo [2/4] Training Model and Running SHAP Analysis...
venv\Scripts\python "src/train_adaboost.py"
if %ERRORLEVEL% NEQ 0 (
    echo Error during model training. Exiting.
    exit /b %ERRORLEVEL%
)

echo [3/4] Generating Business Impact Visualizations...
venv\Scripts\python "src/visualize_business_impact.py"
if %ERRORLEVEL% NEQ 0 (
    exit /b %ERRORLEVEL%
)

echo [4/4] Generating Business Insights Report...
venv\Scripts\python "src/interpret_shap.py"
if %ERRORLEVEL% NEQ 0 (
    echo Error during business insight generation. Exiting.
    exit /b %ERRORLEVEL%
)

echo Pipeline completed successfully!
pause
