# Heart Disease Prediction System

A Machine Learning web app that predicts heart disease risk using patient clinical data.

## Live Demo
🚀 [Click here to try the app](#) *(coming soon)*

## What it does
Enter patient data (age, blood pressure, cholesterol, ECG, etc.) and get instant risk assessment:
- **LOW** — Keep maintaining healthy lifestyle
- **MODERATE** — Consult a doctor
- **HIGH** — Urgent cardiac evaluation needed

## Models Trained & Results

| Model | Test Accuracy | F1 Score |
|-------|--------------|----------|
| Logistic Regression | 82.6% | 84.6% |
| Random Forest | 83.7% | 85.6% |
| SVM | 84.2% | 86.6% |
| **XGBoost** *(Best)* | **85.3%** | **87.3%** |

## Tech Stack
- **Python** — Core language
- **XGBoost / Scikit-learn** — ML models
- **Streamlit** — Web UI
- **Plotly** — Interactive charts
- **Pandas / NumPy** — Data processing

## Features
- 4 ML models trained and compared
- 5-Fold Cross Validation
- Feature engineering (5 new interaction features)
- EDA: correlation heatmap, ROC curves, confusion matrices
- Dual scoring: ML probability + clinical risk score
- Actionable health recommendations

## How to Run Locally

```bash
pip install -r requirements.txt
python heart_disease_training.py
streamlit run heart_disease_project.py
```

## Dataset
UCI Heart Disease Dataset — 920 patients, 13 clinical features.

## Author
**Danish Ilyas** — [GitHub](https://github.com/danish614)
