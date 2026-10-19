"""
Student Performance Prediction Using Machine Learning
Model: Random Forest Regressor | Dataset: UCI Student Performance (Mathematics)
Target: Final Academic Grade (G3: 0-20 scale)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Dataset Discovery & Loading
def load_student_data():
    candidate_paths = [
        os.path.join("data", "student-mat.csv"),
        os.path.join("website project", "data", "student-mat.csv"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "student-mat.csv"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "website project", "data", "student-mat.csv"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return pd.read_csv(p, sep=None, engine="python")
    raise FileNotFoundError("Could not locate student-mat.csv dataset.")


# 2. Main Training, Evaluation, and Prediction Routine
def main():
    print("=" * 60)
    print(" STUDENT PERFORMANCE PREDICTION SYSTEM (ML ENGINE)")
    print("=" * 60)

    # Step 1: Load authentic dataset
    df = load_student_data()
    print(f"[+] Loaded dataset successfully: {df.shape[0]} students, {df.shape[1]} raw attributes.")

    # Step 2: Separate features and target
    target_col = "G3"
    metadata_cols = ["name", "roll_no", "school_name"]
    drop_cols = [c for c in [target_col] + metadata_cols if c in df.columns]

    X_raw = df.drop(columns=drop_cols)
    y = df[target_col].astype(float)

    # Step 3: One-Hot Encoding for categorical features
    X = pd.get_dummies(X_raw, drop_first=True)
    feature_names = X.columns.tolist()

    # Step 4: Train / Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"[+] Partitioned data: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # Step 5: Model Training - Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    print("[+] Random Forest model training complete (100 Decision Trees).")

    # Step 6: Model Evaluation on Held-Out Test Set
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--- MODEL PERFORMANCE METRICS ---")
    print(f"  * Coefficient of Determination (R^2): {r2:.3f}")
    print(f"  * Mean Absolute Error (MAE)        : {mae:.2f} points (0-20 scale)")
    print(f"  * Root Mean Squared Error (RMSE)   : {rmse:.2f} points")

    # Step 7: Top Important Factors
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\n--- TOP 5 PREDICTIVE FACTORS (FEATURE IMPORTANCE) ---")
    for feat, imp in importances.head(5).items():
        print(f"  - {feat:<22} : {imp*100:5.2f}%")

    # Step 8: Interactive Student Inference
    print("\n" + "=" * 60)
    print(" INFERENCE MODULE: PREDICT STUDENT FINAL GRADE")
    print("=" * 60)
    print("(Press Enter to accept sample profile values or type custom input)")

    def get_input(prompt_text, default_val, cast_type):
        try:
            raw = input(f" {prompt_text} [Default={default_val}]: ").strip()
            return cast_type(raw) if raw else default_val
        except (EOFError, KeyboardInterrupt):
            return default_val

    # Construct single student baseline from class median
    student_profile = X.median().to_dict()

    # Prompt key academic and behavioral predictors
    try:
        g2 = get_input("Enter Period 2 Exam Marks (G2: 0-20)", 11.0, float)
        g1 = get_input("Enter Period 1 Exam Marks (G1: 0-20)", 11.0, float)
        study = get_input("Weekly Study Hours (1:<2h, 2:2-5h, 3:5-10h, 4:>10h)", 2.0, float)
        failures = get_input("Past Class Failures (0 to 3)", 0.0, float)
        absences = get_input("Total Days Absent (0 to 75)", 4.0, float)
    except (ValueError, KeyboardInterrupt):
        print("\n[!] Input skipped or interrupted; using default baseline profile.")
        g2, g1, study, failures, absences = 11.0, 11.0, 2.0, 0.0, 4.0

    student_profile["G2"] = g2
    student_profile["G1"] = g1
    student_profile["studytime"] = study
    student_profile["failures"] = failures
    student_profile["absences"] = absences

    # Format vector into DataFrame matching training columns
    student_vector = pd.DataFrame([student_profile], columns=feature_names)
    predicted_g3 = float(model.predict(student_vector)[0])
    predicted_g3 = max(0.0, min(20.0, predicted_g3))
    percentage = (predicted_g3 / 20.0) * 100.0

    # Determine Academic Risk Category & Grade Letter
    if predicted_g3 >= 16.0:
        grade_letter, risk_tier = "A+ (Distinction)", "Low Risk (Safe)"
    elif predicted_g3 >= 14.0:
        grade_letter, risk_tier = "A (First Class)", "Low Risk (Safe)"
    elif predicted_g3 >= 12.0:
        grade_letter, risk_tier = "B (Second Class)", "Low Risk (Safe)"
    elif predicted_g3 >= 10.0:
        grade_letter, risk_tier = "C (Pass / Average)", "Watchlist (Vulnerable)"
    elif predicted_g3 >= 8.0:
        grade_letter, risk_tier = "D (Borderline)", "High Risk (Critical)"
    else:
        grade_letter, risk_tier = "F (Failing)", "High Risk (Critical)"

    print("\n" + "-" * 60)
    print(" PREDICTION RESULT SUMMARY")
    print("-" * 60)
    print(f" Predicted Final Grade (G3) : {predicted_g3:.2f} / 20.0")
    print(f" Equivalent Percentage Score : {percentage:.1f}%")
    print(f" Academic Grade Letter       : {grade_letter}")
    print(f" Academic Risk Standing      : [{risk_tier}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
