import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import (
    DEMO_DATASET_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    LINEAR_REGRESSION_MODEL_PATH,
    DECISION_TREE_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    GRADIENT_BOOSTING_MODEL_PATH,
    MODEL_DIR
)


MODEL_PATHS = {
    "Linear Regression": LINEAR_REGRESSION_MODEL_PATH,
    "Decision Tree": DECISION_TREE_MODEL_PATH,
    "Random Forest": RANDOM_FOREST_MODEL_PATH,
    "Gradient Boosting": GRADIENT_BOOSTING_MODEL_PATH
}


def load_dataset(data_path):
    df = pd.read_csv(data_path)

    missing_columns = [
        column for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(float)

    return X, y


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(mse ** 0.5), 4),
        "r2_score": round(float(r2_score(y_true, y_pred)), 4)
    }


def evaluate_model(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)

    print("\n" + "=" * 70)
    print(f"Evaluation Result: {model_name}")
    print("=" * 70)
    print(metrics)

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.65)
    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")
    plt.title(f"Evaluation Actual vs Predicted - {model_name}")

    min_value = min(min(y_test), min(y_pred))
    max_value = max(max(y_test), max(y_pred))
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    filename = model_name.lower().replace(" ", "_") + "_evaluation_actual_vs_predicted.png"
    output_path = MODEL_DIR / filename
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained crop yield prediction models")
    parser.add_argument("--data", default=str(DEMO_DATASET_PATH), help="Path to crop yield CSV dataset")
    args = parser.parse_args()

    X, y = load_dataset(args.data)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    results = {}

    for model_name, model_path in MODEL_PATHS.items():
        if not Path(model_path).exists():
            print(f"Skipping {model_name}: model file not found at {model_path}")
            continue

        model = joblib.load(model_path)
        results[model_name] = evaluate_model(model_name, model, X_test, y_test)

    output_path = MODEL_DIR / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nEvaluation results saved to: {output_path}")


if __name__ == "__main__":
    main()
