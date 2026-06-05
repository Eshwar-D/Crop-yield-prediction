# Crop Yield Prediction using Weather and Soil Data

## Project Title
**Crop Yield Prediction using Regression and Random Forest**

## Objective
The objective of this project is to predict crop yield based on weather, soil, irrigation, and fertilizer parameters.

The system predicts expected crop yield using features such as:

- Rainfall
- Temperature
- Humidity
- Soil pH
- Nitrogen
- Phosphorus
- Potassium
- Fertilizer amount
- Irrigation level
- Crop type
- Soil type

## Algorithms Used
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

## Target Variable

```text
Yield
```

The yield is represented in:

```text
tons_per_hectare
```

## Dataset Format

Expected CSV format:

```text
FarmID,Crop,SoilType,Rainfall,Temperature,Humidity,Soil_pH,Nitrogen,Phosphorus,Potassium,Fertilizer,Irrigation,SunshineHours,Yield
F001,Rice,Clay,1200,27.5,78,6.4,85,45,50,120,650,6.2,4.8
F002,Wheat,Loamy,650,22.1,62,7.1,70,35,40,95,420,7.1,3.9
```

Required target column:

```text
Yield
```

## Demo Dataset

If you do not have a real crop dataset, generate a demo dataset:

```bash
python src/create_demo_dataset.py
```

This creates:

```text
data/crop_yield_demo_dataset.csv
```

## Project Structure

```text
crop_yield_prediction_project/
│
├── data/
│   └── crop_yield_demo_dataset.csv
│
├── models/
│   ├── linear_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── feature_schema.json
│   └── metrics.json
│
├── reports/
│   └── PROJECT_REPORT.md
│
├── sample_inputs/
│   └── sample_crop_data.csv
│
├── src/
│   ├── config.py
│   ├── create_demo_dataset.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── predict.py
│   └── app.py
│
├── requirements.txt
└── README.md
```

## Installation

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Step 1: Create Demo Dataset

```bash
python src/create_demo_dataset.py
```

## Step 2: Train Models

```bash
python src/train_models.py --data data/crop_yield_demo_dataset.csv
```

This trains:
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting

## Step 3: Evaluate Models

```bash
python src/evaluate_models.py --data data/crop_yield_demo_dataset.csv
```

## Step 4: Predict One Sample

```bash
python src/predict.py
```

## Step 5: Run Web Application

```bash
streamlit run src/app.py
```

## Expected Output

```text
Selected Model  : Random Forest
Predicted Yield : 4.82 tons/hectare
```

## Using a Real Dataset

Place a real crop yield CSV file inside the `data/` folder and train:

```bash
python src/train_models.py --data data/your_crop_dataset.csv
```

Required columns:

```text
Crop, SoilType, Rainfall, Temperature, Humidity, Soil_pH, Nitrogen, Phosphorus, Potassium, Fertilizer, Irrigation, SunshineHours, Yield
```

## Important Note
This project is for academic and educational purposes. Real crop yield depends on many additional factors such as pest attacks, seed quality, farm practices, soil organic carbon, variety, and local climate.
