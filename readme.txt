Created the complete Python project for **Crop Yield Prediction using Weather and Soil Data**.

[Download Crop Yield Prediction Python Project](sandbox:/mnt/data/crop_yield_prediction_project.zip)

It includes:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* Demo crop yield dataset generator
* Weather, soil, fertilizer, and irrigation features
* Single sample prediction script
* CSV batch prediction support
* Streamlit web app
* Project report
* README with execution steps

Run the project:

```bash
pip install -r requirements.txt
python src/create_demo_dataset.py
python src/train_models.py --data data/crop_yield_demo_dataset.csv
streamlit run src/app.py
```

For real dataset usage, place a CSV file inside `data/` with columns such as:

```text
Crop, SoilType, Rainfall, Temperature, Humidity, Soil_pH, Nitrogen, Phosphorus, Potassium, Fertilizer, Irrigation, SunshineHours, Yield
```
