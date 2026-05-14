# ============================================================
# 🚗 Vehicle Price Prediction — Supervised Learning Project
#     Lean Pipeline (Data Processing & Training)
# ============================================================

import numpy as np
import pandas as pd
import datetime
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load the CSV dataset and print basic info."""
    df = pd.read_csv(filepath)
    print(f"Data loaded — Shape: {df.shape}")
    return df


# ────────────────────────────────────────────────────────────
# 2. DATA CLEANING
# ────────────────────────────────────────────────────────────
def extract_sale_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse saledate column into sale_year and sale_month."""
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['sale_year'] = pd.to_numeric(
        df['saledate'].astype(str).str.split().str[3], errors='coerce'
    )
    df['sale_month'] = (
        df['saledate'].astype(str).str.split().str[1]
        .str.lower().map(month_map)
    )
    return df


def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove outliers from a column using the IQR method."""
    Q1, Q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df = df[(df[column] >= lower) & (df[column] <= upper)]
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline."""
    df = extract_sale_date_features(df)
    df.drop(columns=['vin', 'seller', 'saledate'], inplace=True, errors='ignore')
    df.drop_duplicates(inplace=True)

    # Fill missing values
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Standardise text
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.lower().str.strip()

    # Remove outliers
    df = remove_outliers_iqr(df, 'sellingprice')
    df = remove_outliers_iqr(df, 'odometer')

    print(f"Cleaned shape : {df.shape}")
    return df


# ────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add vehicle_age and mileage_per_year features."""
    current_year = datetime.datetime.now().year
    df['vehicle_age'] = current_year - df['year']
    df['mileage_per_year'] = df['odometer'] / df['vehicle_age'].replace(0, 1)
    return df


# ────────────────────────────────────────────────────────────
# 4. MODEL BUILDING
# ────────────────────────────────────────────────────────────
FEATURES = [
    'year', 'condition', 'odometer',
    'vehicle_age', 'mileage_per_year', 'sale_year', 'sale_month',
    'make_enc', 'body_enc', 'transmission_enc', 'state_enc', 'color_enc'
]
CAT_COLS = ['make', 'body', 'transmission', 'state', 'color']


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns and return updated df + encoder dict."""
    le_dict = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    return df, le_dict


def prepare_data(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """Encode, split, and return X_train, X_test, y_train, y_test."""
    df, le_dict = encode_features(df)

    X = df[FEATURES].copy().replace([np.inf, -np.inf], np.nan)
    y = df['sellingprice'].copy()

    valid = X.notnull().all(axis=1) & y.notnull()
    X, y = X[valid], y[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test, le_dict


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> LinearRegression:
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# ────────────────────────────────────────────────────────────
# 5. MODEL EVALUATION
# ────────────────────────────────────────────────────────────
def evaluate_model(
    model: LinearRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> None:
    """Print model performance metrics."""
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\nLinear Regression Metrics:")
    print(f"  R² Score : {r2:.4f}")
    print(f"  MAE      : ${mae:,.2f}")
    print(f"  RMSE     : ${rmse:,.2f}")


# ────────────────────────────────────────────────────────────
# 6. SAVE MODEL
# ────────────────────────────────────────────────────────────
def save_model(
    model: LinearRegression,
    le_dict: dict,
    model_path: str = 'vehicle_price_model.pkl',
    encoder_path: str = 'label_encoders.pkl'
) -> None:
    """Save trained model and label encoders to disk."""
    joblib.dump(model, model_path)
    joblib.dump(le_dict, encoder_path)
    print(f"\nModel saved   - {model_path}")
    print(f"Encoders saved - {encoder_path}")


# ────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ────────────────────────────────────────────────────────────
def run_pipeline(filepath: str = 'car_prices.csv') -> None:
    """
    Execute the full Vehicle Price Prediction pipeline:
      load → clean → feature engineering →
      prepare → train → evaluate → save
    """
    df = load_data(filepath)
    df = clean_data(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, le_dict = prepare_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, le_dict)
    print("\nPipeline complete!")


# ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_pipeline('car_prices.csv')