# FinTech Risk Tier Classification & 30-Day Net Revenue Prediction

This project simulates a real world FinTech analytics use case using a synthetic dataset of 5,000 customers. It demonstrates an end to end machine learning workflow with missing data handling, pipeline based preprocessing, model training, evaluation, and business focused interpretation.

The project is designed to reflect how decisions are made in FinTech environments with the help of data, balancing compliance risk management with revenue optimization.

# Business Context
A FinTech platform aims to:
- Prioritize compliance and fraud reviews by predicting customer risk tier (Low / Medium / High)
- Support growth and retention strategies by predicting next 30 day net revenue (EUR) at the customer level

These predictions help optimize operational effort, reduce financial risk, and focus commercial actions on high impact users.

# Dataset
- 5,000 customer records
- Realistic FinTech features including transaction behavior, failure rates, disputes and chargebacks, support interactions, compliance signals, and product usage
- Missing values intentionally included to demonstrate real world data quality challenges and imputation strategies

Targets:
- `risk_tier` (categorical): Low / Medium / High
- `next_30d_net_revenue_eur` (numerical): proxy for customer level net revenue over the next 30 days

# Approach
- Leakage safe train/test splits for both classification and regression tasks
- Preprocessing implemented using `ColumnTransformer`:
  - Numerical features: median imputation and scaling
  - Categorical features: most frequent imputation and one hot encoding
- Model comparison:
  - Classification: Logistic Regression vs Random Forest
  - Regression: Ridge Regression vs Random Forest Regressor 
          (Ridge Regression was selected as the linear baseline for revenue prediction instead of standard Linear Regression due to the presence of multicollinearity among behavioral and transactional features. Ridge regularization stabilizes coefficient estimates, reduces overfitting, and provides more robust and generalizable revenue predictions in FinTech contexts where correlated signals are common.)
- Pipeline based modeling to ensure reproducibility and consistency across experiments

# Key Results
Risk tier classification (Random Forest):
- Macro F1 score: ~0.72
- High-risk precision: ~0.86, indicating conservative and operations friendly compliance behavior.

Revenue regression (Random Forest Regressor):
- MAE: ~€7.16
- RMSE: ~€9.97
- R²: ~0.93

## Business Outputs
- Compliance queue prioritization with reduced false positives
- Conservative high risk flagging aligned with risk based compliance practices
- Revenue based customer segmentation to support retention and targeting decisions
- Clear identification of behavioral and transactional drivers impacting risk and revenue

## Notebooks
- `01_eda_and_data_quality.ipynb`: data loading, missingness analysis, and target validation
- `02_preprocessing_pipeline.ipynb`: preprocessing pipeline design and train/test splitting
- `03_classification_risk_tier.ipynb`: risk tier modeling, evaluation, and compliance focused insights
- `04_regression_revenue.ipynb`: revenue modeling, evaluation, and growth oriented insights

## How to Run
```bash
pip install -r requirements.txt
jupyter lab
