from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DEMO_DATASET_PATH = DATA_DIR / "crop_yield_demo_dataset.csv"

LINEAR_REGRESSION_MODEL_PATH = MODEL_DIR / "linear_regression_model.pkl"
DECISION_TREE_MODEL_PATH = MODEL_DIR / "decision_tree_model.pkl"
RANDOM_FOREST_MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"
GRADIENT_BOOSTING_MODEL_PATH = MODEL_DIR / "gradient_boosting_model.pkl"

FEATURE_SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COLUMN = "Yield"
FARM_ID_COLUMN = "FarmID"

CATEGORICAL_FEATURES = [
    "Crop",
    "SoilType"
]

NUMERIC_FEATURES = [
    "Rainfall",
    "Temperature",
    "Humidity",
    "Soil_pH",
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "Fertilizer",
    "Irrigation",
    "SunshineHours"
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
