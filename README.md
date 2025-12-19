# Credit Card Fraud Detection (Machine Learning)

## Overview
This project builds an end-to-end machine learning pipeline to detect fraudulent credit card transactions using imbalanced classification techniques. The focus is on realistic evaluation, business cost awareness, and model robustness rather than raw accuracy.

## Dataset
- Source: Kaggle
- Size: ~10,000 transactions
- Fraud rate: ~1.5%
- Target variable: `is_fraud`

## Features
- Transaction amount
- Transaction hour
- Merchant category (one-hot encoded)
- Device trust score
- Transaction velocity
- Location mismatch
- Foreign transaction flag
- Cardholder age

## Methodology
1. Exploratory Data Analysis (EDA)
2. Handling class imbalance
3. Feature encoding
4. Train-test and time-aware splits
5. Baseline Logistic Regression
6. Random Forest
7. Gradient Boosting (XGBoost)
8. Threshold tuning using Precision-Recall tradeoffs
9. Business cost evaluation
10. Stress testing for data leakage and robustness

## Models Used
- Logistic Regression
- Random Forest
- XGBoost

## Evaluation Metrics
- Precision
- Recall
- F1-score
- Precision-Recall Curve
- Confusion Matrix

Accuracy was intentionally deprioritized due to class imbalance.

## Key Findings
- Behavioral features (device trust score, transaction timing, velocity) are stronger fraud indicators than transaction amount.
- Gradient boosting achieved near-perfect performance due to highly separable patterns in the dataset.
- Stress testing revealed potential realism limitations, highlighting the importance of leakage prevention and temporal validation.

## Business Perspective
A cost-based evaluation framework was used to balance:
- False positives (customer friction)
- False negatives (financial loss)

This approach reflects real-world fraud detection systems.

## Limitations
- Dataset exhibits highly deterministic fraud patterns
- Limited noise compared to real-world fraud
- No adversarial adaptation over time

## Future Work
- Cost-sensitive learning
- Time-series validation
- SHAP explainability
- Deployment simulation
- Testing on noisier, real-world datasets

## Technologies
- Python
- pandas, NumPy
- scikit-learn
- XGBoost
- Matplotlib

## Author
Aymane Ait Belarbi
