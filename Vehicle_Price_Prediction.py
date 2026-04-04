# ============================================================
# 🚗 Vehicle Price Prediction — Supervised Learning Project
#     Refactored into modular functions
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')


# ────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load the CSV dataset and print basic info."""
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded — Shape: {df.shape}")
    print(df.head())
    print(df.info())
    print(df.describe())
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
    before = len(df)
    df = df[(df[column] >= lower) & (df[column] <= upper)]
    print(f"  {column}: removed {before - len(df):,} rows  "
          f"(bounds: {lower:,.0f} to {upper:,.0f})")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
      - extract sale date features
      - drop unused columns & duplicates
      - fill missing values
      - standardise text
      - remove outliers
    """
    print("\n🔧 Cleaning data...")
    print("Missing values before cleaning:")
    print(df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False))

    df = extract_sale_date_features(df)
    df.drop(columns=['vin', 'seller', 'saledate'], inplace=True)
    df.drop_duplicates(inplace=True)

    # Fill missing values
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Standardise text
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.lower().str.strip()

    # Remove outliers
    print("\nOutlier removal (IQR method):")
    df = remove_outliers_iqr(df, 'sellingprice')
    df = remove_outliers_iqr(df, 'odometer')

    print(f"\nCleaned shape : {df.shape}")
    print(f"Remaining nulls: {df.isnull().sum().sum()}")
    return df


# ────────────────────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ────────────────────────────────────────────────────────────
def plot_price_distribution(df: pd.DataFrame) -> None:
    """Histogram of selling price with median line."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(df['sellingprice'], bins=80, color='#2196F3', edgecolor='white')
    ax.axvline(
        df['sellingprice'].median(), color='red', linestyle='--',
        label=f"Median: ${df['sellingprice'].median():,.0f}"
    )
    ax.set_title('Distribution of Selling Price', fontweight='bold')
    ax.set_xlabel('Selling Price ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_top_makes(df: pd.DataFrame, top_n: int = 15) -> None:
    """Horizontal bar chart of top N makes by average price."""
    avg_price = (
        df.groupby('make')['sellingprice']
        .mean().sort_values(ascending=False).head(top_n)
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    avg_price.plot(
        kind='barh',
        color=plt.cm.viridis(np.linspace(0.2, 0.9, top_n)),
        ax=ax
    )
    ax.set_title(f'Top {top_n} Makes by Average Selling Price', fontweight='bold')
    ax.set_xlabel('Average Price ($)')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_price_vs_odometer(df: pd.DataFrame, sample_size: int = 5000) -> None:
    """Scatter plot of selling price vs odometer, coloured by year."""
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(12, 6))
    sc = ax.scatter(
        sample['odometer'], sample['sellingprice'],
        c=sample['year'], cmap='plasma', alpha=0.5, s=12
    )
    ax.set_title('Selling Price vs Odometer', fontweight='bold')
    ax.set_xlabel('Odometer (miles)')
    ax.set_ylabel('Selling Price ($)')
    plt.colorbar(sc, label='Year')
    plt.tight_layout()
    plt.show()


def plot_avg_price_by_year(df: pd.DataFrame) -> None:
    """Line chart of average selling price by year."""
    year_avg = df.groupby('year')['sellingprice'].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(year_avg.index, year_avg.values, marker='o', color='#E91E63', linewidth=2)
    ax.fill_between(year_avg.index, year_avg.values, alpha=0.15, color='#E91E63')
    ax.set_title('Average Selling Price by Year', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Price ($)')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of correlations among numeric features."""
    num_cols = ['year', 'condition', 'odometer', 'mmr',
                'sellingprice', 'sale_year', 'sale_month']
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df[num_cols].corr(), annot=True, fmt='.2f',
        cmap='RdYlBu_r', square=True, ax=ax
    )
    ax.set_title('Correlation Heatmap', fontweight='bold')
    plt.tight_layout()
    plt.show()


def run_eda(df: pd.DataFrame) -> None:
    """Run all EDA plots."""
    print("\n📊 Running EDA...")
    plot_price_distribution(df)
    plot_top_makes(df)
    plot_price_vs_odometer(df)
    plot_avg_price_by_year(df)
    plot_correlation_heatmap(df)


# ────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add vehicle_age and mileage_per_year features."""
    current_year = datetime.datetime.now().year
    df['vehicle_age'] = current_year - df['year']
    df['mileage_per_year'] = df['odometer'] / df['vehicle_age'].replace(0, 1)
    print("\n✅ New features added: vehicle_age, mileage_per_year")
    print(df[['year', 'vehicle_age', 'odometer', 'mileage_per_year']].head())
    return df


# ────────────────────────────────────────────────────────────
# 5. MODEL BUILDING
# ────────────────────────────────────────────────────────────
FEATURES = [
    'year', 'condition', 'odometer', 'mmr',
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Encode, split, and return X_train, X_test, y_train, y_test."""
    df, le_dict = encode_features(df)

    X = df[FEATURES].copy().replace([np.inf, -np.inf], np.nan)
    y = df['sellingprice'].copy()

    valid = X.notnull().all(axis=1) & y.notnull()
    X, y = X[valid], y[valid]
    print(f"\nRows after removing NaN/inf: {len(X):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test, le_dict


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> LinearRegression:
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\n✅ Model trained.")
    return model


# ────────────────────────────────────────────────────────────
# 6. MODEL EVALUATION
# ────────────────────────────────────────────────────────────
def evaluate_model(
    model: LinearRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> np.ndarray:
    """Print metrics and return predictions."""
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n📈 Linear Regression Results:")
    print(f"  R² Score : {r2:.4f}")
    print(f"  MAE      : ${mae:,.2f}")
    print(f"  RMSE     : ${rmse:,.2f}")
    return y_pred


def plot_actual_vs_predicted(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """Scatter plot of actual vs predicted prices."""
    idx = np.random.choice(len(y_test), min(3000, len(y_test)), replace=False)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test.values[idx], y_pred[idx], alpha=0.3, s=10, color='#2196F3')
    lim = max(y_test.max(), y_pred.max()) * 0.9
    ax.plot([0, lim], [0, lim], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_title('Actual vs Predicted — Linear Regression', fontweight='bold')
    ax.set_xlabel('Actual Price ($)')
    ax.set_ylabel('Predicted Price ($)')
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def plot_feature_coefficients(model: LinearRegression) -> None:
    """Horizontal bar chart of feature coefficients."""
    coef = pd.Series(model.coef_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    coef.plot(
        kind='barh',
        color=plt.cm.viridis(np.linspace(0.2, 0.9, len(coef))),
        ax=ax
    )
    ax.set_title('Feature Coefficients — Linear Regression', fontweight='bold')
    ax.set_xlabel('Coefficient Value')
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────
# 7. SAVE MODEL
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
    print(f"\n💾 Model saved   → {model_path}")
    print(f"💾 Encoders saved → {encoder_path}")


# ────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ────────────────────────────────────────────────────────────
def run_pipeline(filepath: str = 'car_prices.csv') -> None:
    """
    Execute the full Vehicle Price Prediction pipeline:
      load → clean → EDA → feature engineering →
      prepare → train → evaluate → save
    """
    df = load_data(filepath)
    df = clean_data(df)
    run_eda(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, le_dict = prepare_data(df)
    model = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    plot_actual_vs_predicted(y_test, y_pred)
    plot_feature_coefficients(model)
    save_model(model, le_dict)
    print("\n🎉 Pipeline complete!")


# ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_pipeline('car_prices.csv')