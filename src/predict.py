import argparse
from pathlib import Path

import joblib
import pandas as pd

from config import (
    LINEAR_REGRESSION_MODEL_PATH,
    DECISION_TREE_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    GRADIENT_BOOSTING_MODEL_PATH,
    FEATURE_COLUMNS
)


MODEL_PATHS = {
    "linear_regression": LINEAR_REGRESSION_MODEL_PATH,
    "decision_tree": DECISION_TREE_MODEL_PATH,
    "random_forest": RANDOM_FOREST_MODEL_PATH,
    "gradient_boosting": GRADIENT_BOOSTING_MODEL_PATH
}


def default_sample():
    return {
        "Crop": "Rice",
        "SoilType": "Clay",
        "Rainfall": 1180.0,
        "Temperature": 27.5,
        "Humidity": 76.0,
        "Soil_pH": 6.5,
        "Nitrogen": 85.0,
        "Phosphorus": 45.0,
        "Potassium": 52.0,
        "Fertilizer": 125.0,
        "Irrigation": 680.0,
        "SunshineHours": 6.8
    }


def predict_sample(model_name):
    model_path = MODEL_PATHS[model_name]

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train the models first.")

    model = joblib.load(model_path)
    sample = default_sample()

    input_df = pd.DataFrame([sample], columns=FEATURE_COLUMNS)
    predicted_yield = float(model.predict(input_df)[0])

    print("Crop Yield Prediction Result")
    print("----------------------------")
    print(f"Selected Model  : {model_name}")
    print(f"Predicted Yield : {predicted_yield:.3f} tons/hectare")

    print("\nInput Sample:")
    for key, value in sample.items():
        print(f"{key}: {value}")

    print("\nNote: This result is for educational purposes. Actual yield depends on many local field conditions.")


def main():
    parser = argparse.ArgumentParser(description="Predict crop yield for one demo farm record")
    parser.add_argument(
        "--model",
        choices=list(MODEL_PATHS.keys()),
        default="random_forest",
        help="Model to use"
    )
    args = parser.parse_args()

    predict_sample(args.model)


if __name__ == "__main__":
    main()
