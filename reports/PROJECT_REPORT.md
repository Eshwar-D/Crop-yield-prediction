# Project Report: Crop Yield Prediction using Weather and Soil Data

## 1. Introduction
Crop yield prediction is an important application of machine learning in agriculture. Farmers, researchers, and policymakers can use yield prediction models to estimate production, plan fertilizer usage, manage irrigation, and improve food supply planning.

Crop yield depends on several factors, including rainfall, temperature, humidity, soil pH, nutrient levels, fertilizer usage, irrigation, soil type, and crop type. Machine learning models can learn relationships between these factors and crop yield.

## 2. Problem Statement
To develop a Python-based regression system that predicts crop yield from weather, soil, irrigation, and fertilizer data using regression and Random Forest models.

## 3. Objectives
- To prepare a crop yield dataset.
- To include weather, soil, fertilizer, irrigation, and crop-related features.
- To train multiple regression models.
- To evaluate model performance using regression metrics.
- To predict crop yield for new farm records.
- To develop a Streamlit web application for user-friendly prediction.

## 4. Dataset Description
The dataset contains crop and field-level information.

Expected columns:

```text
FarmID, Crop, SoilType, Rainfall, Temperature, Humidity, Soil_pH, Nitrogen, Phosphorus, Potassium, Fertilizer, Irrigation, SunshineHours, Yield
```

Target variable:

```text
Yield
```

The yield is represented in tons per hectare.

## 5. Features Used

### 5.1 Categorical Features
- Crop
- SoilType

### 5.2 Weather Features
- Rainfall
- Temperature
- Humidity
- SunshineHours

### 5.3 Soil and Nutrient Features
- Soil_pH
- Nitrogen
- Phosphorus
- Potassium

### 5.4 Farm Management Features
- Fertilizer
- Irrigation

## 6. Algorithms Used

### 6.1 Linear Regression
Linear Regression predicts yield by fitting a linear relationship between input features and crop yield.

### 6.2 Decision Tree Regressor
Decision Tree Regressor predicts yield using rule-based splits on feature values.

### 6.3 Random Forest Regressor
Random Forest Regressor builds multiple decision trees and averages their predictions. It is useful for handling nonlinear relationships in agricultural data.

### 6.4 Gradient Boosting Regressor
Gradient Boosting builds weak learners sequentially and improves prediction by correcting previous errors.

## 7. Methodology

1. Load the crop yield dataset.
2. Check required columns.
3. Separate input features and target variable.
4. Apply preprocessing:
   - Median imputation for numeric features
   - Most-frequent imputation for categorical features
   - Standard scaling for numeric features
   - One-hot encoding for categorical features
5. Split the dataset into training and testing sets.
6. Train Linear Regression, Decision Tree, Random Forest, and Gradient Boosting models.
7. Evaluate models using MAE, MSE, RMSE, and R² score.
8. Save trained models.
9. Predict crop yield for new field data.
10. Display predicted yield in tons per hectare.

## 8. System Architecture

```text
Crop + Weather + Soil + Fertilizer Data
                  ↓
            Preprocessing
                  ↓
          Regression Model
                  ↓
         Predicted Crop Yield
```

## 9. Evaluation Metrics

### Mean Absolute Error
MAE measures the average absolute difference between actual and predicted yield.

### Mean Squared Error
MSE measures the average squared difference between actual and predicted yield.

### Root Mean Squared Error
RMSE is the square root of MSE and is easier to interpret in the same unit as the target variable.

### R² Score
R² score indicates how well the model explains variation in crop yield. A value closer to 1 indicates better performance.

## 10. Expected Output

Example:

```text
Predicted Yield: 4.82 tons/hectare
```

## 11. Applications
- Crop yield forecasting
- Agricultural planning
- Fertilizer recommendation support
- Irrigation management support
- Food production estimation
- Smart farming and precision agriculture projects
- Academic machine learning demonstration

## 12. Limitations
- The demo dataset is synthetic and intended only for education.
- Real crop yield depends on many additional factors.
- Pest attack, crop variety, sowing date, soil organic carbon, and farm practices are not fully modeled.
- Weather data should ideally be location-specific and season-specific.
- Field validation is required before real agricultural decision-making.

## 13. Future Enhancements
- Add real government or agricultural datasets.
- Add satellite vegetation indices such as NDVI.
- Add district-wise and season-wise prediction.
- Add crop recommendation module.
- Add fertilizer recommendation module.
- Add time-series weather forecasting.
- Add explainable AI for feature importance.
- Deploy the system as a web or mobile application.

## 14. Conclusion
This project demonstrates how regression and Random Forest models can be used to predict crop yield using weather, soil, fertilizer, and irrigation data. It includes demo dataset generation, model training, evaluation, prediction, and a Streamlit interface for educational demonstration.
