# FinTech Risk Tier Classification & 30-Day Net Revenue Prediction

This project simulates a real-world FinTech analytics case using a synthetic dataset of 5,000 customers. It demonstrates an end-to-end machine learning workflow with missing data handling, pipeline-based preprocessing, model training, evaluation, and business interpretation.

## Business Context
A FinTech platform wants to:
- Prioritize compliance reviews by predicting customer **risk tier** (Low / Medium / High)
- Support growth and retention by predicting **next 30-day net revenue (EUR)**

## Dataset
- 5,000 customer records
- Realistic FinTech features: transaction behavior, failure rates, disputes/chargebacks, support tickets, compliance flags, product usage
- Missing values included to demonstrate imputation strategies

Targets:
- `risk_tier` (categorical): Low / Medium / High
- `next_30d_net_revenue_eur` (numerical): net revenue proxy over the next 30 days

## Approach
- Leakage-safe train/test splits
- Preprocessing with `ColumnTransformer`:
  - Numeric: median imputation + scaling
  - Categorical: most frequent imputation + one-hot encoding
- Models:
  - Classification: Logistic Regression vs Random Forest
  - Regression: Ridge vs Random Forest Regressor

## Key Results
Risk tier classification (Random Forest):
- Macro F1: ~0.72
- High-risk precision: ~0.86 (conservative, operations-friendly behavior)

Revenue regression (Random Forest Regressor):
- MAE: ~€7.16
- RMSE: ~€9.97
- R²: ~0.93

## Notebooks
- `01_eda_and_data_quality.ipynb`: data loading, missingness analysis, target validation
- `02_preprocessing_pipeline.ipynb`: preprocessing pipeline setup (imputation + encoding)
- `03_classification_risk_tier.ipynb`: risk tier modeling, evaluation, and business insights
- `04_regression_revenue.ipynb`: revenue modeling, evaluation, and business insights

## How to Run
```bash
pip install -r requirements.txt
jupyter lab
