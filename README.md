# Diabetes Prediction & Clinical EDA

**Tools:** Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter Notebooks  
**Domain:** Clinical Healthcare Analytics  
**Timeline:** Fall 2024 (Business Programming, University of North Texas)

---

## Problem statement

Diabetes affects hundreds of millions of people globally, yet many cases go undetected until complications arise. This project explores clinical patient data to understand what factors predict diabetes onset, and builds machine learning classifiers to identify high-risk individuals.

---

## Dataset

- **File:** `Healthcare-Diabetes.csv`
- **Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target:** `Outcome` (1 = Diabetes, 0 = No Diabetes)
- **Challenge:** Several columns contained biologically implausible zero values requiring careful imputation

---

## Methodology

### 1. Data cleaning
Identified that zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` were not valid — a person cannot have zero blood pressure or zero glucose. These were treated as missing values and imputed using **median imputation** (more robust to outliers than mean).

```python
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
imputer = SimpleImputer(strategy='median')
df[cols_to_fix] = imputer.fit_transform(df[cols_to_fix])
```

### 2. Exploratory Data Analysis (EDA)

**Key questions explored:**

| Question | Finding |
|---|---|
| Age distribution by diabetes status | Diabetic patients tend to be older; peak risk in 40–60 range |
| Average glucose by outcome | Diabetic: ~141 mg/dL vs Non-diabetic: ~110 mg/dL — significant gap |
| Pregnancies vs. diabetes rate | Higher pregnancy count correlates with increased diabetes likelihood |
| BMI vs. diabetes | Clear positive correlation (r ≈ 0.29); diabetic patients have higher median BMI |
| Blood pressure distribution | Overlapping distributions — less predictive on its own |

### 3. Feature preparation
- Standardized features using `StandardScaler` before model training
- Split data: 80% training / 20% testing using `train_test_split` with `random_state=42`

### 4. Model training & comparison

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | ~77% | ~0.83 |
| Random Forest | ~76% | ~0.82 |

Both models evaluated using:
- Classification report (Precision, Recall, F1-score)
- Confusion matrix
- AUC-ROC score

### 5. Key insights
- **Glucose** is the single strongest predictor of diabetes outcome
- **BMI** and **Age** are the next most important features
- **Insulin** and **SkinThickness** have high missing-value rates, limiting their individual predictive power
- Logistic Regression and Random Forest perform comparably — Logistic Regression preferred for interpretability in clinical settings

---

## Files in this repository

| File | Description |
|---|---|
| `diabetes_prediction.ipynb` | Full Jupyter Notebook: cleaning, EDA, modeling, evaluation |
| `README.md` | Project documentation |

---

## How to run

```bash
# Clone the repository
git clone https://github.com/Hima-03/diabetes-prediction-eda

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# Open the notebook
jupyter notebook diabetes_prediction.ipynb
```

---

## Skills demonstrated

`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn` `Logistic Regression` `Random Forest` `EDA` `Data Cleaning` `Feature Engineering` `AUC-ROC` `Jupyter Notebooks` `Healthcare Analytics` `Predictive Modeling`
# diabetes-prediction-eda
