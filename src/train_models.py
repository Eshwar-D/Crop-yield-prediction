import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from config import (
    MODEL_DIR,
    DEMO_DATASET_PATH,
    LINEAR_REGRESSION_MODEL_PATH,
    DECISION_TREE_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    GRADIENT_BOOSTING_MODEL_PATH,
    FEATURE_SCHEMA_PATH,
    METRICS_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES
)


def load_dataset(data_path):
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Run create_demo_dataset.py first."
        )

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


def build_preprocessor():
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ("categorical", categorical_transformer, CATEGORICAL_FEATURES)
    ])

    return preprocessor


def build_models():
    preprocessor = build_preprocessor()

    models = {
        "Linear Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ]),

        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", DecisionTreeRegressor(
                max_depth=8,
                min_samples_split=10,
                random_state=RANDOM_STATE
            ))
        ]),

        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_split=4,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", GradientBoostingRegressor(
                n_estimators=220,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE
            ))
        ])
    }

    return models


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5

    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "r2_score": round(float(r2_score(y_true, y_pred)), 4)
    }


def save_actual_vs_predicted_plot(model_name, y_true, y_pred):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.65)
    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")
    plt.title(f"Actual vs Predicted Yield - {model_name}")

    min_value = min(min(y_true), min(y_pred))
    max_value = max(max(y_true), max(y_pred))
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    filename = model_name.lower().replace(" ", "_") + "_actual_vs_predicted.png"
    output_path = MODEL_DIR / filename
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return str(output_path)


def save_residual_plot(model_name, y_true, y_pred):
    residuals = y_true - y_pred

    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.65)
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Predicted Yield")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot - {model_name}")

    filename = model_name.lower().replace(" ", "_") + "_residual_plot.png"
    output_path = MODEL_DIR / filename
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return str(output_path)


def save_feature_importance(model_name, model, top_n=15):
    try:
        regressor = model.named_steps["regressor"]
        if not hasattr(regressor, "feature_importances_"):
            return None

        preprocessor = model.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
        importances = regressor.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(top_n)

        csv_path = MODEL_DIR / f"{model_name.lower().replace(' ', '_')}_feature_importance.csv"
        importance_df.to_csv(csv_path, index=False)

        plt.figure(figsize=(8, 6))
        plt.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"Top Feature Importance - {model_name}")
        plt.tight_layout()

        plot_path = MODEL_DIR / f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

        return str(csv_path)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Train crop yield prediction regression models")
    parser.add_argument("--data", default=str(DEMO_DATASET_PATH), help="Path to crop yield CSV dataset")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    models = build_models()

    model_paths = {
        "Linear Regression": LINEAR_REGRESSION_MODEL_PATH,
        "Decision Tree": DECISION_TREE_MODEL_PATH,
        "Random Forest": RANDOM_FOREST_MODEL_PATH,
        "Gradient Boosting": GRADIENT_BOOSTING_MODEL_PATH
    }

    all_metrics = {}

    for model_name, model in models.items():
        print("\n" + "=" * 70)
        print(f"Training model: {model_name}")
        print("=" * 70)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = calculate_metrics(y_test, y_pred)
        actual_plot = save_actual_vs_predicted_plot(model_name, y_test.values, y_pred)
        residual_plot = save_residual_plot(model_name, y_test.values, y_pred)
        importance_path = save_feature_importance(model_name, model)

        metrics["actual_vs_predicted_plot"] = actual_plot
        metrics["residual_plot"] = residual_plot
        metrics["feature_importance_csv"] = importance_path

        joblib.dump(model, model_paths[model_name])
        all_metrics[model_name] = metrics

        print(f"MAE: {metrics['mae']}")
        print(f"RMSE: {metrics['rmse']}")
        print(f"R2 Score: {metrics['r2_score']}")
        print(f"Saved model to: {model_paths[model_name]}")

    feature_schema = {
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target_column": TARGET_COLUMN
    }

    with open(FEATURE_SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_schema, f, indent=4)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)

    print("\nTraining completed successfully.")
    print(f"Feature schema saved to: {FEATURE_SCHEMA_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")

    print("\nModel Performance Summary")
    print("-" * 70)
    for model_name, values in all_metrics.items():
        print(
            f"{model_name}: "
            f"MAE={values['mae']}, "
            f"RMSE={values['rmse']}, "
            f"R2={values['r2_score']}"
        )


if __name__ == "__main__":
    main()
