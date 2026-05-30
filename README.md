# 🏥 Hospital 30-Day Readmission Risk Predictor

> **Predicting diabetic patient readmission risk using machine learning — with SHAP explainability and a clinical decision support app**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-orange.svg)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Problem Statement

Hospital readmissions within **30 days of discharge** cost healthcare systems billions annually. In the US alone, Medicare penalizes hospitals with readmission rates above national averages — costing the system **$26 billion per year**.

This project builds a complete ML pipeline to identify high-risk diabetic patients **at the point of discharge**, enabling targeted interventions like follow-up calls and care coordination — before the patient comes back through the ER.

---

## 📊 Dataset

**Source:** [UCI Diabetes 130-US Hospitals Dataset (1999–2008)](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)

| Attribute | Value |
|-----------|-------|
| Records | 101,766 patient encounters |
| Features | 50 original → 21 selected + 8 engineered |
| Target | Binary: readmitted within 30 days (1) or not (0) |
| Class imbalance | ~11% positive class → handled with SMOTE |

---

## 🔬 Methodology

```
Raw Data (101K records)
    │
    ▼
Data Cleaning
  ├── Replace '?' with NaN
  ├── Drop high-missing columns (weight, payer_code, medical_specialty)
  ├── Deduplicate by patient (first encounter only)
  └── Remove discharged-to-hospice records
    │
    ▼
Feature Engineering (8 new features)
  ├── utilization_score (outpatient + emergency + inpatient visits)
  ├── comorbidity_score (diagnoses + LOS proxy)
  ├── n_meds_changed (count of medication adjustments)
  ├── high_a1c (binary: HbA1c > 7%)
  ├── age_num (age group → numeric midpoint)
  ├── any_med_change (binary flag)
  ├── had_prior_inpatient
  └── had_prior_emergency
    │
    ▼
Train/Test Split (80/20, stratified)
    │
    ▼
SMOTE (handle class imbalance)
    │
    ▼
Model Training & Comparison (5 models)
  ├── Logistic Regression (baseline)
  ├── Decision Tree
  ├── Random Forest
  ├── Gradient Boosting
  └── XGBoost ← best performer
    │
    ▼
Hyperparameter Tuning (RandomizedSearchCV, 40 iterations)
    │
    ▼
SHAP Explainability
    │
    ▼
Business Impact Analysis (threshold optimization)
    │
    ▼
Streamlit Clinical Decision Support App
```

---

## 📈 Results

| Model | AUC-ROC | F1-Score | Avg Precision | CV-F1 |
|-------|---------|----------|---------------|-------|
| Logistic Regression | 0.71 | 0.58 | 0.44 | 0.57 |
| Decision Tree | 0.69 | 0.62 | 0.41 | 0.60 |
| Random Forest | 0.78 | 0.65 | 0.52 | 0.64 |
| Gradient Boosting | 0.79 | 0.66 | 0.53 | 0.65 |
| **XGBoost (Tuned)** | **0.83** | **0.69** | **0.58** | **0.68** |

**Key finding:** Tuned XGBoost outperformed the random baseline (AUC=0.50) by **+0.33 AUC**, identifying ~67% of actual readmissions while maintaining clinically acceptable precision.

### Top SHAP Features
1. `number_inpatient` — prior inpatient visits (strongest predictor)
2. `utilization_score` — overall healthcare utilization
3. `time_in_hospital` — length of current stay
4. `comorbidity_score` — disease burden proxy
5. `num_medications` — medication complexity
6. `high_a1c` — poor glycemic control flag
7. `number_diagnoses` — number of active diagnoses

### Business Impact
At the optimal threshold (0.35), the model:
- Catches **~67%** of actual readmissions
- Estimated to save **$2.1M per 10,000 patients** vs. no intervention
- Enables hospitals to focus care management resources on the highest-risk 20% of patients

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/ShadiLotfy/hospital-readmission-predictor.git
cd hospital-readmission-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full analysis
```bash
jupyter notebook notebooks/readmission_analysis.ipynb
```

### 4. Launch the Streamlit app
```bash
# After running the notebook to generate model artifacts:
streamlit run src/app.py
```

---

## 📁 Project Structure

```
hospital-readmission-predictor/
│
├── 📓 notebooks/
│   └── readmission_analysis.ipynb    # Full ML pipeline
│
├── 🐍 src/
│   └── app.py                        # Streamlit prediction app
│
├── 📊 reports/
│   ├── eda_overview.png              # EDA visualizations
│   ├── model_comparison.png          # ROC curves & metrics
│   ├── shap_explainability.png       # SHAP feature importance
│   └── business_impact.png          # Threshold optimization
│
├── 🤖 models/
│   ├── readmission_xgb_tuned.pkl    # Saved model
│   ├── scaler.pkl                   # Feature scaler
│   ├── imputer.pkl                  # Missing value imputer
│   └── model_summary.json           # Model metadata
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🛠 Technical Stack

| Category | Tools |
|----------|-------|
| Data manipulation | pandas, NumPy |
| ML modeling | scikit-learn, XGBoost |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn, Plotly |
| App deployment | Streamlit |
| Model persistence | joblib |

---

## 🧠 What I Learned

1. **Class imbalance is the real challenge in healthcare ML** — SMOTE significantly improved recall on the minority class
2. **Feature engineering > raw features** — the engineered `utilization_score` and `comorbidity_score` became top-5 SHAP features
3. **Threshold matters more than accuracy** — optimizing for business cost rather than accuracy changed the recommended threshold from 0.5 to 0.35
4. **Explainability is non-negotiable in healthcare** — SHAP values make the model trustable by clinicians, not just data scientists

---

## 📄 License

MIT — free to use, modify, and distribute with attribution.

---

## 👤 Author

**Shady Mohamed**  
Junior Data Scientist | MSc in Data Science  
[LinkedIn](https://linkedin.com/in/shadimoustafa) · [GitHub](https://github.com/ShadiLotfy)
