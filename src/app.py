import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from config import (
    LINEAR_REGRESSION_MODEL_PATH,
    DECISION_TREE_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    GRADIENT_BOOSTING_MODEL_PATH,
    METRICS_PATH,
    FEATURE_COLUMNS
)


MODEL_OPTIONS = {
    "Linear Regression": LINEAR_REGRESSION_MODEL_PATH,
    "Decision Tree": DECISION_TREE_MODEL_PATH,
    "Random Forest": RANDOM_FOREST_MODEL_PATH,
    "Gradient Boosting": GRADIENT_BOOSTING_MODEL_PATH
}

CROP_OPTIONS = ["Rice", "Wheat", "Maize", "Sugarcane", "Cotton", "Groundnut"]
SOIL_OPTIONS = ["Clay", "Loamy", "Sandy", "Black", "Red", "Alluvial"]


def load_metrics():
    if not METRICS_PATH.exists():
        return {}

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


def predict_single(model, input_values):
    input_df = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)
    predicted_yield = float(model.predict(input_df)[0])
    return predicted_yield


def predict_csv(model, input_df):
    missing_columns = [column for column in FEATURE_COLUMNS if column not in input_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = input_df[FEATURE_COLUMNS]
    predictions = model.predict(X)

    result_df = input_df.copy()
    result_df["Predicted_Yield"] = [round(float(value), 3) for value in predictions]

    return result_df


def main():
    st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾")

    st.title("🌾 Crop Yield Prediction using Weather and Soil Data")
    st.write(
        "Predict crop yield using rainfall, temperature, soil pH, soil nutrients, fertilizer, and irrigation data."
    )

    available_models = {
        model_name: model_path
        for model_name, model_path in MODEL_OPTIONS.items()
        if Path(model_path).exists()
    }

    if not available_models:
        st.error("No trained models found. Please train the models first.")
        st.code("python src/create_demo_dataset.py\npython src/train_models.py --data data/crop_yield_demo_dataset.csv")
        return

    metrics = load_metrics()

    selected_model_name = st.sidebar.selectbox("Select Regression Model", list(available_models.keys()))
    model = load_model(available_models[selected_model_name])

    st.sidebar.subheader("Model Performance")
    if selected_model_name in metrics:
        for key, value in metrics[selected_model_name].items():
            if "plot" not in key and "csv" not in key:
                st.sidebar.write(f"{key}: {value}")
    else:
        st.sidebar.write("Metrics not available.")

    tab1, tab2 = st.tabs(["Single Farm Prediction", "CSV Batch Prediction"])

    with tab1:
        st.subheader("Enter Crop, Weather, Soil, and Fertilizer Details")

        col1, col2 = st.columns(2)

        crop = col1.selectbox("Crop", CROP_OPTIONS)
        soil_type = col1.selectbox("Soil Type", SOIL_OPTIONS)
        rainfall = col1.number_input("Rainfall (mm)", min_value=0.0, max_value=3000.0, value=1180.0, step=10.0)
        temperature = col1.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=27.5, step=0.1)
        humidity = col1.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=76.0, step=0.1)
        sunshine_hours = col1.number_input("Sunshine Hours per Day", min_value=0.0, max_value=15.0, value=6.8, step=0.1)

        soil_ph = col2.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        nitrogen = col2.number_input("Nitrogen (kg/ha)", min_value=0.0, max_value=250.0, value=85.0, step=1.0)
        phosphorus = col2.number_input("Phosphorus (kg/ha)", min_value=0.0, max_value=150.0, value=45.0, step=1.0)
        potassium = col2.number_input("Potassium (kg/ha)", min_value=0.0, max_value=200.0, value=52.0, step=1.0)
        fertilizer = col2.number_input("Fertilizer Used (kg/ha)", min_value=0.0, max_value=400.0, value=125.0, step=1.0)
        irrigation = col2.number_input("Irrigation (mm)", min_value=0.0, max_value=2000.0, value=680.0, step=10.0)

        input_values = {
            "Crop": crop,
            "SoilType": soil_type,
            "Rainfall": rainfall,
            "Temperature": temperature,
            "Humidity": humidity,
            "Soil_pH": soil_ph,
            "Nitrogen": nitrogen,
            "Phosphorus": phosphorus,
            "Potassium": potassium,
            "Fertilizer": fertilizer,
            "Irrigation": irrigation,
            "SunshineHours": sunshine_hours
        }

        if st.button("Predict Crop Yield"):
            predicted_yield = predict_single(model, input_values)

            st.subheader("Prediction Result")
            st.success(f"Predicted Yield: {predicted_yield:.3f} tons/hectare")
            st.info("Actual yield may vary due to pest attack, seed variety, farm practices, and local climate.")

    with tab2:
        uploaded_file = st.file_uploader(
            "Upload CSV file with crop, weather, soil, and fertilizer columns",
            type=["csv"]
        )

        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview")
            st.dataframe(input_df.head())

            if st.button("Predict CSV Crop Yield"):
                try:
                    result_df = predict_csv(model, input_df)
                    st.subheader("Batch Prediction Results")
                    st.dataframe(result_df)

                    csv_data = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Prediction Results",
                        data=csv_data,
                        file_name="crop_yield_predictions.csv",
                        mime="text/csv"
                    )
                except Exception as error:
                    st.error(str(error))

    st.warning(
        "This system is for educational demonstration only. Field-level validation is required for real agricultural decision-making."
    )


if __name__ == "__main__":
    main()
