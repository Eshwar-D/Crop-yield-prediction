import random
import numpy as np
import pandas as pd

from config import DATA_DIR, DEMO_DATASET_PATH


CROPS = ["Rice", "Wheat", "Maize", "Sugarcane", "Cotton", "Groundnut"]
SOIL_TYPES = ["Clay", "Loamy", "Sandy", "Black", "Red", "Alluvial"]

CROP_BASE_YIELD = {
    "Rice": 4.2,
    "Wheat": 3.6,
    "Maize": 4.8,
    "Sugarcane": 70.0,
    "Cotton": 2.4,
    "Groundnut": 2.1
}

CROP_IDEAL_CONDITIONS = {
    "Rice": {"rainfall": 1200, "temperature": 27, "ph": 6.4, "fertilizer": 120, "irrigation": 700},
    "Wheat": {"rainfall": 650, "temperature": 22, "ph": 7.0, "fertilizer": 95, "irrigation": 420},
    "Maize": {"rainfall": 750, "temperature": 25, "ph": 6.5, "fertilizer": 110, "irrigation": 500},
    "Sugarcane": {"rainfall": 1500, "temperature": 28, "ph": 6.8, "fertilizer": 180, "irrigation": 1100},
    "Cotton": {"rainfall": 700, "temperature": 30, "ph": 7.2, "fertilizer": 90, "irrigation": 450},
    "Groundnut": {"rainfall": 600, "temperature": 28, "ph": 6.6, "fertilizer": 70, "irrigation": 350}
}

SOIL_EFFECT = {
    "Clay": 0.05,
    "Loamy": 0.12,
    "Sandy": -0.10,
    "Black": 0.10,
    "Red": -0.03,
    "Alluvial": 0.08
}


def suitability_score(value, ideal, tolerance):
    diff = abs(value - ideal)
    score = 1.0 - (diff / tolerance)
    return max(0.0, min(1.0, score))


def create_demo_dataset(n_samples=1500, random_state=42):
    random.seed(random_state)
    np.random.seed(random_state)

    records = []

    for i in range(n_samples):
        crop = random.choice(CROPS)
        soil_type = random.choice(SOIL_TYPES)
        ideal = CROP_IDEAL_CONDITIONS[crop]

        rainfall = np.random.normal(ideal["rainfall"], ideal["rainfall"] * 0.22)
        rainfall = float(np.clip(rainfall, 250, 2200))

        temperature = np.random.normal(ideal["temperature"], 4.0)
        temperature = float(np.clip(temperature, 12, 42))

        humidity = np.random.normal(68, 12)
        humidity = float(np.clip(humidity, 30, 95))

        soil_ph = np.random.normal(ideal["ph"], 0.7)
        soil_ph = float(np.clip(soil_ph, 4.5, 9.0))

        nitrogen = np.random.normal(75, 25)
        phosphorus = np.random.normal(42, 15)
        potassium = np.random.normal(48, 18)

        nitrogen = float(np.clip(nitrogen, 10, 160))
        phosphorus = float(np.clip(phosphorus, 5, 100))
        potassium = float(np.clip(potassium, 5, 130))

        fertilizer = np.random.normal(ideal["fertilizer"], ideal["fertilizer"] * 0.28)
        fertilizer = float(np.clip(fertilizer, 20, 260))

        irrigation = np.random.normal(ideal["irrigation"], ideal["irrigation"] * 0.25)
        irrigation = float(np.clip(irrigation, 100, 1600))

        sunshine_hours = np.random.normal(7.0, 1.4)
        sunshine_hours = float(np.clip(sunshine_hours, 3.0, 10.5))

        rainfall_score = suitability_score(rainfall, ideal["rainfall"], ideal["rainfall"] * 0.65)
        temp_score = suitability_score(temperature, ideal["temperature"], 12)
        ph_score = suitability_score(soil_ph, ideal["ph"], 2.2)
        fertilizer_score = suitability_score(fertilizer, ideal["fertilizer"], ideal["fertilizer"] * 0.85)
        irrigation_score = suitability_score(irrigation, ideal["irrigation"], ideal["irrigation"] * 0.75)

        nutrient_score = (
            suitability_score(nitrogen, 80, 90)
            + suitability_score(phosphorus, 45, 55)
            + suitability_score(potassium, 50, 65)
        ) / 3.0

        climate_score = (rainfall_score + temp_score + ph_score + irrigation_score) / 4.0
        management_score = (fertilizer_score + nutrient_score) / 2.0

        base_yield = CROP_BASE_YIELD[crop]
        soil_adjustment = SOIL_EFFECT[soil_type]

        yield_value = base_yield * (
            0.45
            + 0.33 * climate_score
            + 0.17 * management_score
            + soil_adjustment
            + 0.03 * (sunshine_hours - 6.5)
        )

        noise = np.random.normal(0, base_yield * 0.07)
        yield_value = yield_value + noise

        if crop == "Sugarcane":
            yield_value = float(np.clip(yield_value, 30, 120))
        else:
            yield_value = float(np.clip(yield_value, 0.4, 9.0))

        records.append({
            "FarmID": f"FARM{i + 1:05d}",
            "Crop": crop,
            "SoilType": soil_type,
            "Rainfall": round(rainfall, 2),
            "Temperature": round(temperature, 2),
            "Humidity": round(humidity, 2),
            "Soil_pH": round(soil_ph, 2),
            "Nitrogen": round(nitrogen, 2),
            "Phosphorus": round(phosphorus, 2),
            "Potassium": round(potassium, 2),
            "Fertilizer": round(fertilizer, 2),
            "Irrigation": round(irrigation, 2),
            "SunshineHours": round(sunshine_hours, 2),
            "Yield": round(yield_value, 3)
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = create_demo_dataset()
    df.to_csv(DEMO_DATASET_PATH, index=False)

    print("Demo crop yield dataset created successfully.")
    print(f"Dataset saved at: {DEMO_DATASET_PATH}")
    print(f"Number of records: {df.shape[0]}")
    print("\nCrop distribution:")
    print(df["Crop"].value_counts())
    print("\nDataset preview:")
    print(df.head())


if __name__ == "__main__":
    main()
