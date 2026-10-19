# 🎓 Gradewatch — Academic Risk Prediction & Early Intervention Platform

> **A Production-Grade Python Machine Learning Platform for Student Performance Forecasting, Vulnerability Classification, and Actionable Educational Interventions.**

---

## 1. Project Title & Overview

**Student Performance Prediction Using Machine Learning (Gradewatch v2.5)**  
An end-to-end Machine Learning web application designed to forecast secondary/higher-education student final performance (grade $G_3$, scale 0–20), classify academic risk tiers (**High Risk**, **Watchlist**, **Low Risk**), isolate individual factor drivers via marginal impact analysis, and generate targeted evidence-informed educational support strategies.

---

## 2. Problem Statement

Identifying academically vulnerable students before final report cards or semester examinations are released is a critical challenge in educational institutions. Traditional grading systems are reactive, identifying failure only after final exams. 

Early-warning systems face two key data science challenges:
1. **Data Leakage**: Relying on mid-term or period assessment grades ($G_1, G_2$) when performing week-one early triage yields misleading accuracy in early stages.
2. **Actionability Gap**: Standard regression models predict numbers without explaining *which factors drove that grade* or *what concrete interventions* teachers, academic advisors, or parents should apply.

---

## 3. Objectives

- **Develop Genuine ML Pipelines**: Implement and compare **Random Forest Regressor**, **Decision Tree Regressor**, and **Linear Regression Baseline** using `scikit-learn`.
- **Zero-Data-Leakage Architecture**: Provide two distinct pipelines:
  - **Mid-Year Mode**: High-precision forecasting utilizing Period 1 ($G_1$) and Period 2 ($G_2$) assessment results ($R^2 \approx 0.80$, $\text{MAE} \approx 1.2$ pts).
  - **Early-Warning Mode**: Week-one triage strictly excluding $G_1$ and $G_2$, relying solely on demographic, socio-economic, and study habit features ($R^2 \approx 0.25$, $\text{MAE} \approx 3.1$ pts).
- **Explainability & What-If Simulation**: Isolate marginal factor drivers and simulate the projected outcome of interventions (e.g., reducing absences or increasing study hours) in real time.
- **Evidence-Informed Academic Interventions**: Rule-based recommendation engine suggesting specific educational interventions based on individual student risk signals.
- **Cohort Intelligence & CSV Upload**: Enable batch evaluation and scoring of entire classrooms with exportable CSV reports.

---

## 4. Dataset

The platform uses the benchmark **UCI Student Performance Dataset** (*Cortez and Silva, 2008*), sourced from secondary schools in Portugal:
- `data/student-mat.csv`: 395 student records for Mathematics course.
- `data/student-por.csv`: 649 student records for Portuguese language course.

---

## 5. Feature Schema

| Group | Feature Key | Type | Description |
|---|---|---|---|
| **Demographics** | `school` | Categorical | Student's school (`GP`: Gabriel Pereira, `MS`: Mousinho da Silveira) |
| | `sex` | Categorical | Sex (`F`: Female, `M`: Male) |
| | `age` | Numerical | Student's age (15 to 22) |
| | `address` | Categorical | Home address type (`U`: Urban, `R`: Rural) |
| | `famsize` | Categorical | Family size (`LE3`: $\le 3$, `GT3`: $> 3$) |
| | `Pstatus` | Categorical | Parents' cohabitation status (`T`: Together, `A`: Apart) |
| **Family** | `Medu`, `Fedu` | Numerical | Mother's / Father's education level (0: None to 4: Higher) |
| | `Mjob`, `Fjob` | Categorical | Mother's / Father's job (`teacher`, `health`, `services`, `at_home`, `other`) |
| | `guardian` | Categorical | Student's legal guardian (`mother`, `father`, `other`) |
| | `famrel` | Numerical | Family relationship quality (1: Very bad to 5: Excellent) |
| | `famsup` | Categorical | Family educational support (`yes`, `no`) |
| **Academic** | `reason` | Categorical | Reason for school choice (`home`, `reputation`, `course`, `other`) |
| | `traveltime` | Numerical | Travel time to school (1: $<15$ min to 4: $>60$ min) |
| | `studytime` | Numerical | Weekly study time (1: $<2$ hrs, 2: 2–5 hrs, 3: 5–10 hrs, 4: $>10$ hrs) |
| | `failures` | Numerical | Number of past class failures (0 to 3) |
| | `schoolsup` | Categorical | Extra educational school support (`yes`, `no`) |
| | `paid` | Categorical | Extra paid classes within the course subject (`yes`, `no`) |
| | `higher` | Categorical | Desires to pursue higher education (`yes`, `no`) |
| | `internet` | Categorical | Internet access at home (`yes`, `no`) |
| | `G1` | Numerical | First period grade (0 to 20) — *Mid-Year mode only* |
| | `G2` | Numerical | Second period grade (0 to 20) — *Mid-Year mode only* |
| **Lifestyle** | `activities` | Categorical | Extracurricular activities (`yes`, `no`) |
| | `nursery` | Categorical | Attended nursery school (`yes`, `no`) |
| | `romantic` | Categorical | In a romantic relationship (`yes`, `no`) |
| | `freetime` | Numerical | Free time after school (1: Very low to 5: Very high) |
| | `goout` | Numerical | Going out with friends (1: Very low to 5: Very high) |
| | `Dalc`, `Walc` | Numerical | Workday / Weekend alcohol consumption (1 to 5) |
| | `health` | Numerical | Current health status (1: Very bad to 5: Very good) |
| | `absences` | Numerical | Number of school absences (0 to 75) |
| **Target** | `G3` | Numerical | Final student grade (0 to 20 scale) |

---

## 6. Machine Learning Architecture

```
                       ┌───────────────────────────────┐
                       │   UCI Student Performance     │
                       │          Dataset              │
                       └──────────────┬────────────────┘
                                      │
                      ┌───────────────┴───────────────┐
                      │    Data Preprocessing &       │
                      │  Leak-Free Train/Test Split   │
                      └───────┬───────────────┬───────┘
                              │               │
               [Mid-Year Mode]│               │[Early-Warning Mode]
               (With G1 + G2) │               │(Strictly No G1/G2)
                              ▼               ▼
           ┌─────────────────────┐         ┌─────────────────────┐
           │ scikit-learn        │         │ scikit-learn        │
           │ ColumnTransformer   │         │ ColumnTransformer   │
           │ (OHE + Scaler)      │         │ (OHE + Scaler)      │
           └──────────┬──────────┘         └──────────┬──────────┘
                      │                               │
        ┌─────────────┼─────────────┐   ┌─────────────┼─────────────┐
        ▼             ▼             ▼   ▼             ▼             ▼
  ┌───────────┐ ┌───────────┐ ┌─────────┐ ┌───────────┐ ┌───────────┐ ┌─────────┐
  │  Random   │ │ Decision  │ │ Linear  │ │  Random   │ │ Decision  │ │ Linear  │
  │  Forest   │ │   Tree    │ │ Regress │ │  Forest   │ │   Tree    │ │ Regress │
  └─────┬─────┘ └─────┬─────┘ └────┬────┘ └─────┬─────┘ └─────┬─────┘ └────┬────┘
        │             │            │            │             │            │
        └─────────────┼────────────┘            └─────────────┼────────────┘
                      ▼                                       ▼
        ┌───────────────────────────┐           ┌───────────────────────────┐
        │  Serialized Pipeline (.joblib) │      │  Serialized Pipeline (.joblib) │
        └─────────────┬─────────────┘           └─────────────┬─────────────┘
                      │                                       │
                      └───────────────────┬───────────────────┘
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    Streamlit Web Application & UI     │
                      │  - Single Prediction & Uncertainty    │
                      │  - What-If Scenario Sandbox           │
                      │  - Cohort Ledger & CSV Upload         │
                      │  - Model Diagnostics & Viva Guide     │
                      └───────────────────────────────────────┘
```

---

## 7. Model Evaluation & Comparison

All models are trained with fixed random states (`random_state=42`) and evaluated on held-out test sets (80/20 split) as well as 5-Fold Cross-Validation:

### Mode A: Mid-Year Prediction (With $G_1$ and $G_2$)
| Model | Test MAE | Test RMSE | Test $R^2$ | 5-Fold CV $R^2$ (Mean $\pm$ Std) |
|---|---|---|---|---|
| **Random Forest Regressor** | **1.20** | **2.03** | **0.80** | **0.890 $\pm$ 0.052** |
| **Linear Regression** | 1.60 | 2.35 | 0.73 | 0.804 $\pm$ 0.081 |
| **Decision Tree Regressor** | 1.43 | 2.53 | 0.69 | 0.865 $\pm$ 0.063 |

### Mode B: Early-Warning Prediction (Zero-Leakage, No $G_1$ or $G_2$)
| Model | Test MAE | Test RMSE | Test $R^2$ | 5-Fold CV $R^2$ (Mean $\pm$ Std) |
|---|---|---|---|---|
| **Random Forest Regressor** | **3.15** | **3.92** | **0.25** | **0.205 $\pm$ 0.114** |
| **Linear Regression** | 3.40 | 4.20 | 0.14 | -0.092 $\pm$ 0.150 |
| **Decision Tree Regressor** | 3.50 | 4.64 | -0.05 | -0.090 $\pm$ 0.120 |

---

## 8. Academic Risk Classification & Thresholds

Final grades are mapped into three actionable risk categories:
- 🚨 **High Risk** ($G_3 < 10.0$): Student is in danger of failing the course (passing grade in Portuguese educational system is 10/20). Immediate intervention required.
- ⚠️ **Watchlist** ($10.0 \le G_3 < 12.5$): Vulnerable academic standing. Recommended for structured study monitoring and tutoring.
- ✅ **Low Risk** ($G_3 \ge 12.5$): Solid academic standing on track for completion.

---

## 9. Project Directory Structure

```
website project/
│
├── app.py                          # Streamlit Main Hub & Executive Overview
├── student.py                      # CLI & Launch Entrypoint
├── requirements.txt                # Python Dependencies
├── README.md                       # Documentation & College Viva Guide
│
├── data/
│   ├── student-mat.csv             # UCI Math Course Dataset (395 records)
│   └── student-por.csv             # UCI Portuguese Course Dataset (649 records)
│
├── models/
│   ├── rf_midyear_model.joblib     # Serialized Random Forest (Mid-Year)
│   ├── rf_earlywarning_model.joblib# Serialized Random Forest (Early-Warning)
│   ├── dt_midyear_model.joblib     # Serialized Decision Tree
│   ├── lr_midyear_model.joblib     # Serialized Linear Regression
│   └── metrics_summary.joblib      # Evaluation metrics cache
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # ColumnTransformer, OHE, Scaler & loaders
│   ├── train_model.py              # ML training, cross-validation & serialization
│   ├── prediction.py               # Inference engine, uncertainty & risk tiers
│   ├── evaluation.py               # MAE, MSE, RMSE, R² and residual calculators
│   ├── feature_importance.py       # RF MDI and Marginal Impact analysis
│   └── recommendations.py          # Rule-based Suggested Academic Interventions
│
├── pages/
│   ├── 1_🎯_Student_Prediction.py   # Single student evaluator & driver chart
│   ├── 2_🎛️_What_If_Simulator.py   # Live policy intervention sandbox
│   ├── 3_📊_Class_Dashboard.py     # Class roster, CSV upload, export & inspector
│   └── 4_📈_Model_Diagnostics.py   # Benchmark comparison & viva defense guide
│
├── scripts/
│   └── download_data.py            # Dataset downloader & verification utility
│
└── assets/
    └── style.css                   # Custom dark chalkboard & gold theme CSS
```

---

## 10. Installation & VS Code Setup

### Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, or 3.13 installed.
- VS Code.

### Step 1: Open Project in VS Code
Open VS Code and choose **File > Open Folder...**, then select the project folder.

### Step 2: Open Terminal & Create Virtual Environment
In the VS Code terminal (PowerShell or Bash):

```bash
# Windows
python -m venv venv

# Activate on Windows PowerShell:
.\venv\Scripts\Activate.ps1

# (Or Windows Command Prompt):
.\venv\Scripts\activate.bat

# (Or macOS / Linux):
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train Machine Learning Models (Optional — models are pre-trained)
```bash
python -m src.train_model
```

### Step 5: Launch the Streamlit Web Application
```bash
streamlit run app.py
```

The browser will automatically open to `http://localhost:8501`.

---

## 11. Viva Voce & College Presentation Defense Guide

### Q1: Why did you choose Random Forest over a Single Decision Tree or Linear Regression?
> **Answer:** Student performance data exhibits non-linear relationships and high-order interactions (e.g., low study time compounded by high absenteeism and lack of home support). Linear Regression assumes strictly linear relationships and is sensitive to outliers. A single Decision Tree suffers from high variance and easily overfits small cohorts. Random Forest aggregates an ensemble of 120 randomized decision trees (bagging with feature subspace sampling), effectively reducing variance without increasing bias.

### Q2: What is Data Leakage, and how did your project prevent it?
> **Answer:** Data leakage occurs when variables that would not be available at prediction time are included in model training, producing deceptively high accuracy. Period 1 ($G_1$) and Period 2 ($G_2$) grades are strong indicators of final grade ($G_3$), but they do not exist during the first week of class. Our system addresses this by building two independent pipelines: **Mid-Year Mode** (using $G_1, G_2$ for mid-term forecasting) and **Early-Warning Mode** (strictly excluding $G_1, G_2$ for week-one triage).

### Q3: What is the difference between Regression and Classification in this project?
> **Answer:** Our core ML engine operates as a **Regressor**, estimating continuous final grade $G_3 \in [0, 20]$. This preserves fine-grained numerical differences between students. After regression, we apply a domain-specific threshold mapping to categorize continuous scores into actionable **Risk Tiers** (High Risk: $<10$, Watchlist: $10-12.5$, Low Risk: $\ge 12.5$).

### Q4: What do MAE, RMSE, and $R^2$ represent?
> - **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual grades on the 0–20 scale. An MAE of $1.2$ means the model's prediction is, on average, within $1.2$ points.
> - **RMSE (Root Mean Squared Error):** Penalizes larger mistakes quadratically, useful because missing a failing student is more harmful than a small point error.
> - **$R^2$ Score:** Measures the proportion of variance in final grades explained by the features ($1.0$ is perfect; $0.0$ is equivalent to always predicting the class mean).

### Q5: Does Feature Importance prove causality?
> **Answer:** No. Feature Importance measures the relative contribution of a feature within the Random Forest model (reduction in impurity). High importance for absences indicates strong correlation, not necessarily direct causation, as absences may proxy for external socioeconomic or personal factors.

---

## 12. Limitations & Ethical Considerations

1. **Advisory Tool**: Gradewatch is designed as a decision-support system to guide pastoral and academic support, not an automated punitive mechanism.
2. **Dataset Specificity**: The model is trained on Portuguese secondary education benchmarks; local institutional calibration is recommended before live deployment.

---

## 13. License & Acknowledgments

- **Dataset Credit**: P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance*. EUROSIS-ETI, 2008.
- Built for academic demonstration, college capstone, and viva evaluation.
