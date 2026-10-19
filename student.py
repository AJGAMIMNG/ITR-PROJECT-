"""
Gradewatch - Student Performance Prediction CLI & Launch Entrypoint.
Can be executed directly from terminal or VS Code.
"""

import os
import sys
import subprocess
import argparse
import webbrowser
import time
import socket

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.train_model import train_and_save_all_models
from src.prediction import predict_single_student, get_dataset_defaults


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch_streamlit_app():
    port = find_free_port()
    app_url = f"http://127.0.0.1:{port}"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless",
        "true",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
    ]
    subprocess.Popen(cmd, cwd=BASE_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    time.sleep(3)
    webbrowser.open(app_url)
    print(f"\nOpening Gradewatch in your browser: {app_url}\n")


def main():
    parser = argparse.ArgumentParser(description="Gradewatch Student Performance Prediction Platform")
    parser.add_argument("--train", action="store_true", help="Train all Machine Learning pipelines")
    parser.add_argument("--predict", action="store_true", help="Run sample student inference")
    parser.add_argument("--app", action="store_true", help="Launch Streamlit web application")
    args = parser.parse_args()

    if args.train:
        print("Executing model training...")
        train_and_save_all_models()
    elif args.predict:
        defaults = get_dataset_defaults()
        print("\n" + "=" * 60)
        print(" STUDENT PERFORMANCE PREDICTION (INDIAN STUDENT SAMPLE)")
        print("=" * 60)
        res = predict_single_student(defaults, mode="midyear", model_type="rf")
        print(f" Student Name             : Aarav Sharma")
        print(f" Expected Marks           : {res['percentage']:.0f}% ({res['predicted_grade']} / 20)")
        print(f" Academic Grade Letter    : Grade {res['grade_letter']}")
        print(f" Academic Risk Status     : [{res['risk_tier']}]")
        print(f" Passing Probability      : {res['pass_probability']}% Chance of Passing")
        print("\nWhy this prediction? (Top Factors):")
        for r in res["simple_reasons"]:
            clean_r = r.replace("✅", "[+]").replace("⚠️", "[-]")
            print(f"  {clean_r}")
        print("\nSuggested Action Steps:")
        for itm in res["interventions"][:3]:
            print(f"  - [{itm['priority']}] {itm['title']}")
        print("=" * 60)
    else:
        defaults = get_dataset_defaults()
        print("=" * 65)
        print(" GRADWATCH: STUDENT PERFORMANCE PREDICTION ENGINE (PYTHON ML)")
        print("=" * 65)
        print(" Sample Prediction for Indian Student Profile (Aarav Sharma):")
        res = predict_single_student(defaults, mode="midyear", model_type="rf")
        print(f"  -> Expected Marks        : {res['percentage']:.0f}% ({res['predicted_grade']} / 20)")
        print(f"  -> Academic Grade Letter : Grade {res['grade_letter']} ({res['grade_remarks']})")
        print(f"  -> Risk Category         : [{res['risk_tier']}]")
        print(f"  -> Pass Probability      : {res['pass_probability']}% Chance of Passing")
        print("\n Top Key Factors:")
        for r in res["simple_reasons"][:3]:
            clean_r = r.replace("✅", "[+]").replace("⚠️", "[-]")
            print(f"   {clean_r}")

        print("\n" + "=" * 65)
        print(" Launching Streamlit Web Dashboard at http://localhost:8501 ...")
        print("=" * 65)
        launch_streamlit_app()

if __name__ == "__main__":
    main()
