"""
Cab Fare Prediction System
Single-file end-to-end example using: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
Followed steps from CRISP-DM: business understanding, data understanding, preparation, modeling, evaluation, deployment (model export).

How to use:
1. Place your dataset as 'cab_fare_data.csv' or 'cab_fare_data.xlsx' in the same folder.
   Expected columns (recommended): ['fare_amount','pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']
   If your column names differ, adjust `DATA_COLS` mapping below.
2. Install dependencies: pip install pandas numpy scikit-learn matplotlib seaborn joblib openpyxl
3. Run: python cab_fare_prediction.py

This script will:
- Load and clean data
- Create spatial & time-based features (haversine distance, bearing, manhattan approximations, time features)
- Train baseline Linear Regression and stronger models (RandomForest, GradientBoosting)
- Evaluate with RMSE/MAE/R2, plot diagnostics
- Save the best model to 'final_cab_fare_model.joblib'

Notes:
- This is an educational example. Tune and adapt for production datasets.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------------------
# Configuration / Constants
# ---------------------------
DATA_CSV = 'cab_fare_data.csv'
DATA_XLSX = 'cab_fare_data.xlsx'
MODEL_OUTPUT = 'final_cab_fare_model.joblib'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Column mapping — change if your dataset uses different names
DATA_COLS = {
    'fare': 'fare_amount',
    'pickup_datetime': 'pickup_datetime',
    'pu_long': 'pickup_longitude',
    'pu_lat': 'pickup_latitude',
    'do_long': 'dropoff_longitude',
    'do_lat': 'dropoff_latitude',
    'passengers': 'passenger_count'
}

# ---------------------------
# Utility functions
# ---------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on earth (kilometers).
    Vectorized implementation that accepts numpy arrays or pandas Series.
    """
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def bearing_between_points(lat1, lon1, lat2, lon2):
    """Compute bearing in degrees between two points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    initial_bearing = np.degrees(np.arctan2(x, y))
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing


def manhattan_distance(lat1, lon1, lat2, lon2):
    """Approximate Manhattan distance using haversine for lat and lon separately."""
    a = haversine_distance(lat1, lon1, lat1, lon2)
    b = haversine_distance(lat1, lon1, lat2, lon1)
    return a + b

# ---------------------------
# Data loading and initial cleaning
# ---------------------------

def load_data():
    if os.path.exists(DATA_CSV):
        print(f"Loading CSV data from {DATA_CSV}")
        df = pd.read_csv(DATA_CSV)
    elif os.path.exists(DATA_XLSX):
        print(f"Loading Excel data from {DATA_XLSX}")
        df = pd.read_excel(DATA_XLSX)
    else:
        raise FileNotFoundError(f"Please provide '{DATA_CSV}' or '{DATA_XLSX}' in working directory.")
    return df


def initial_cleaning(df):
    # rename columns if necessary
    df = df.rename(columns={
        DATA_COLS['fare']: DATA_COLS['fare'],
        DATA_COLS['pickup_datetime']: DATA_COLS['pickup_datetime'],
        DATA_COLS['pu_long']: DATA_COLS['pu_long'],
        DATA_COLS['pu_lat']: DATA_COLS['pu_lat'],
        DATA_COLS['do_long']: DATA_COLS['do_long'],
        DATA_COLS['do_lat']: DATA_COLS['do_lat'],
        DATA_COLS['passengers']: DATA_COLS['passengers']
    })

    # Keep only necessary columns (if present)
    cols_to_keep = [v for v in DATA_COLS.values() if v in df.columns]
    df = df[cols_to_keep].copy()

    # Parse datetime
    if df[DATA_COLS['pickup_datetime']].dtype == object:
        df[DATA_COLS['pickup_datetime']] = pd.to_datetime(df[DATA_COLS['pickup_datetime']], errors='coerce')

    # Drop rows with missing essential data
    df = df.dropna(subset=[DATA_COLS['fare'], DATA_COLS['pickup_datetime'], DATA_COLS['pu_long'], DATA_COLS['pu_lat'], DATA_COLS['do_long'], DATA_COLS['do_lat']])

    # Basic types
    df[DATA_COLS['passengers']] = pd.to_numeric(df[DATA_COLS['passengers']], errors='coerce').fillna(1).astype(int)
    df[DATA_COLS['fare']] = pd.to_numeric(df[DATA_COLS['fare']], errors='coerce')

    return df

# ---------------------------
# Feature engineering
# ---------------------------

def feature_engineering(df):
    pu_lat = df[DATA_COLS['pu_lat']]
    pu_lon = df[DATA_COLS['pu_long']]
    do_lat = df[DATA_COLS['do_lat']]
    do_lon = df[DATA_COLS['do_long']]

    df['haversine_km'] = haversine_distance(pu_lat, pu_lon, do_lat, do_lon)
    df['manhattan_km'] = manhattan_distance(pu_lat, pu_lon, do_lat, do_lon)
    df['bearing'] = bearing_between_points(pu_lat, pu_lon, do_lat, do_lon)

    # Time features
    dt = df[DATA_COLS['pickup_datetime']]
    df['pickup_hour'] = dt.dt.hour
    df['pickup_minute'] = dt.dt.minute
    df['pickup_day'] = dt.dt.day
    df['pickup_weekday'] = dt.dt.weekday
    df['pickup_month'] = dt.dt.month

    # Simple speed feature (fare per km) - careful with zero distance
    df['fare_per_km'] = df[DATA_COLS['fare']] / (df['haversine_km'].replace(0, np.nan))

    # Log transforms to reduce skew
    df['log_fare'] = np.log1p(df[DATA_COLS['fare']].clip(lower=0))
    df['log_distance'] = np.log1p(df['haversine_km'].clip(lower=0))

    # Flag short trips
    df['is_short'] = (df['haversine_km'] < 1.0).astype(int)

    return df

# ---------------------------
# Cleaning / Outlier handling
# ---------------------------

def remove_outliers(df):
    # Remove impossible coordinates (latitude must be between -90 and 90, longitude -180 to 180)
    mask_valid_coords = (
        df[DATA_COLS['pu_lat']].between(-90, 90) &
        df[DATA_COLS['do_lat']].between(-90, 90) &
        df[DATA_COLS['pu_long']].between(-180, 180) &
        df[DATA_COLS['do_long']].between(-180, 180)
    )
    df = df[mask_valid_coords].copy()

    # Fare reasonable: remove negative and extremely high fares
    df = df[df[DATA_COLS['fare']].between(0, 1000)]

    # Passenger count reasonable
    df = df[df[DATA_COLS['passengers']].between(1, 6)]

    # Distance not absurd
    df = df[df['haversine_km'] <= 500]

    # Remove rows where distance = 0 and fare != 0 (could be pickups in same location but fare>0; keep some)
    # We'll keep zero-distance trips but cap influence via modeling

    # Drop NA created by divisions
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['haversine_km', 'log_fare', 'log_distance'])
    return df

# ---------------------------
# Modeling helpers
# ---------------------------

def prepare_features(df):
    # Feature list
    numeric_features = ['haversine_km', 'manhattan_km', 'bearing', 'pickup_hour', 'pickup_minute', 'pickup_weekday', 'pickup_month', 'passenger_count', 'log_distance', 'is_short']
    for f in numeric_features:
        if f not in df.columns:
            raise KeyError(f"Expected feature '{f}' not in dataframe. Current columns: {df.columns.tolist()}")

    X = df[numeric_features].copy()
    y = df['log_fare']  # predict log_fare for stability
    return X, y, numeric_features

# ---------------------------
# Train & evaluate models
# ---------------------------

def evaluate_model(model, X_train, X_val, y_train, y_val, label='Model'):
    model.fit(X_train, y_train)
    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)
    y_true = np.expm1(y_val)

    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)

    print(f"{label} performance on validation set:")
    print(f" RMSE: {rmse:.4f}")
    print(f" MAE:  {mae:.4f}")
    print(f" R2:   {r2:.4f}\n")

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'preds': preds, 'y_true': y_true}

# ---------------------------
# Plot diagnostics
# ---------------------------

def plot_diagnostics(y_true, preds, title_suffix=''):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.scatterplot(x=y_true, y=preds, alpha=0.3)
    plt.xlabel('Actual Fare')
    plt.ylabel('Predicted Fare')
    plt.title('Predicted vs Actual ' + title_suffix)
    lims = [0, max(y_true.max(), preds.max())]
    plt.plot(lims, lims, '--')

    plt.subplot(1,2,2)
    residuals = y_true - preds
    sns.histplot(residuals, bins=50, kde=True)
    plt.title('Residuals ' + title_suffix)
    plt.xlabel('Actual - Predicted')
    plt.tight_layout()
    plt.show()

# ---------------------------
# Feature importance plot for tree models
# ---------------------------

def plot_feature_importances(model, feature_names, top_n=15):
    try:
        importances = model.feature_importances_
    except Exception:
        print('Model does not expose feature_importances_')
        return
    idx = np.argsort(importances)[-top_n:][::-1]
    plt.figure(figsize=(8,6))
    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx])
    plt.title('Feature importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

# ---------------------------
# Main pipeline
# ---------------------------

def main():
    df = load_data()
    print('Initial rows:', len(df))

    df = initial_cleaning(df)
    df = feature_engineering(df)
    df = remove_outliers(df)

    print('Rows after cleaning:', len(df))
    print('Example rows:')
    print(df.head())

    X, y, numeric_features = prepare_features(df)

    # Train test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Simple baseline: Linear Regression
    baseline_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])

    print('\nTraining baseline Linear Regression...')
    baseline_results = evaluate_model(baseline_pipeline, X_train, X_val, y_train, y_val, label='LinearRegression (baseline)')
    plot_diagnostics(baseline_results['y_true'], baseline_results['preds'], title_suffix='(Linear Regression)')

    # Random Forest
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
    ])

    print('Training Random Forest (with basic hyperparameter search)...')
    rf_param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10, 20, None],
        'rf__min_samples_split': [2, 5]
    }
    rf_search = GridSearchCV(rf_pipeline, rf_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    rf_search.fit(X_train, y_train)
    print('Best RF params:', rf_search.best_params_)
    rf_best = rf_search.best_estimator_
    rf_results = evaluate_model(rf_best, X_train, X_val, y_train, y_val, label='RandomForest')
    plot_feature_importances(rf_best.named_steps['rf'], numeric_features)
    plot_diagnostics(rf_results['y_true'], rf_results['preds'], title_suffix='(Random Forest)')

    # Gradient Boosting (scikit-learn GB)
    gb_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gb', GradientBoostingRegressor(random_state=RANDOM_STATE))
    ])
    print('Training Gradient Boosting (with small grid)...')
    gb_param_grid = {
        'gb__n_estimators': [100, 200],
        'gb__learning_rate': [0.05, 0.1],
        'gb__max_depth': [3, 5]
    }
    gb_search = GridSearchCV(gb_pipeline, gb_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    gb_search.fit(X_train, y_train)
    print('Best GB params:', gb_search.best_params_)
    gb_best = gb_search.best_estimator_
    gb_results = evaluate_model(gb_best, X_train, X_val, y_train, y_val, label='GradientBoosting')
    plot_feature_importances(gb_best.named_steps['gb'], numeric_features)
    plot_diagnostics(gb_results['y_true'], gb_results['preds'], title_suffix='(Gradient Boosting)')

    # Compare models by RMSE
    scores = {
        'LinearRegression': baseline_results['rmse'],
        'RandomForest': rf_results['rmse'],
        'GradientBoosting': gb_results['rmse']
    }
    print('\nModel RMSE comparison:')
    for k, v in scores.items():
        print(f" {k}: RMSE={v:.4f}")

    # Choose best model (lowest RMSE)
    best_model_name = min(scores, key=scores.get)
    print('\nSelected model:', best_model_name)
    if best_model_name == 'RandomForest':
        final_model = rf_best
    elif best_model_name == 'GradientBoosting':
        final_model = gb_best
    else:
        final_model = baseline_pipeline

    # Train final model on full data
    print('\nRetraining selected model on full dataset...')
    final_model.fit(X, y)

    # Save artifacts
    joblib.dump({'model': final_model, 'feature_names': numeric_features}, MODEL_OUTPUT)
    print(f"Saved final model to {MODEL_OUTPUT}")

    # Quick feature importance and sample predictions
    try:
        plot_feature_importances(final_model.named_steps[list(final_model.named_steps.keys())[-1]], numeric_features)
    except Exception:
        pass

    # Show sample predictions
    sample = X.sample(10, random_state=RANDOM_STATE)
    preds_log = final_model.predict(sample)
    preds = np.expm1(preds_log)
    print('\nSample predictions:')
    out = sample.copy()
    out['predicted_fare'] = preds
    print(out.head())


if __name__ == '__main__':
    main()
