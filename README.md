# Vehicle Price Prediction

A supervised learning project to predict vehicle selling prices using linear regression.

## Overview

This project loads, cleans, and analyzes a dataset of used car sales to build a predictive model for vehicle prices. It includes exploratory data analysis (EDA), feature engineering, model training, evaluation, and visualization.

## Files

- `car_prices.csv`: Dataset containing vehicle sales data.
- `Vehicle_Price_Prediction.ipynb`: Jupyter notebook version of the analysis.
- `Vehicle_Price_Prediction.py`: Python script for the full pipeline.
- `vehicle_price_model.pkl`: Trained linear regression model (generated after running the script).
- `label_encoders.pkl`: Saved label encoders for categorical features (generated after running the script).

## Installation

1. Ensure Python 3.13+ is installed.
2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```
   pip install numpy pandas matplotlib seaborn scikit-learn joblib
   ```

## Usage

Run the Python script to execute the full pipeline:

```
python Vehicle_Price_Prediction.py
```

This will:
- Load and clean the data.
- Perform EDA (plots may require a GUI backend).
- Train the model.
- Evaluate performance (R² ≈ 0.9590).
- Save the model and encoders.

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- joblib

## Results

- **R² Score**: 0.9590
- **MAE**: $1,017.08
- **RMSE**: $1,552.78

The model uses features like year, condition, odometer, MMR, vehicle age, mileage per year, and encoded categorical variables (make, body, transmission, state, color).

