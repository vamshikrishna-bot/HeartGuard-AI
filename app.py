from pathlib import Path
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT))

model = joblib.load(ROOT / "KNN_heart.pkl")
scaler = joblib.load(ROOT / "scaler_heart.pkl")
expected_columns = joblib.load(ROOT / "columns_heart.pkl")


@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(ROOT, filename)


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        sex = data["sex"]
        chest_pain = data["chest_pain"]
        resting_ecg = data["resting_ecg"]
        exercise_angina = data["exercise_angina"]
        st_slope = data["st_slope"]
        values = {
            "Age": float(data["age"]),
            "RestingBP": float(data["resting_bp"]),
            "Cholesterol": float(data["cholesterol"]),
            "FastingBS": int(data["fasting_bs"]),
            "MaxHR": float(data["max_heart_rate"]),
            "Oldpeak": float(data["oldpeak"]),
            "Sex_M": int(sex == "M"),
            "ChestPainType_ATA": int(chest_pain == "ATA"),
            "ChestPainType_NAP": int(chest_pain == "NAP"),
            "ChestPainType_TA": int(chest_pain == "TA"),
            "RestingECG_Normal": int(resting_ecg == "Normal"),
            "RestingECG_ST": int(resting_ecg == "ST"),
            "ExerciseAngina_Y": int(exercise_angina == "Y"),
            "ST_Slope_Flat": int(st_slope == "Flat"),
            "ST_Slope_Up": int(st_slope == "Up"),
        }
        input_data = pd.DataFrame([values], columns=expected_columns)
        prediction = int(model.predict(scaler.transform(input_data))[0])
    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"error": f"Invalid assessment data: {error}"}), 400

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8501)), debug=False)
