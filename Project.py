# Core
import numpy as np
import pandas as pd
import re
from datetime import datetime
from urllib.parse import urlparse

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

import os

# model + split
from sklearn.model_selection import train_test_split

# preprocessing
from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures,
)

# models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import joblib

# ── Preprocessing via shared module ──────────────────────────────────────────
from preprocess import build_training_df, INFERRABLE_FEATURES

# ── Load dataset ──────────────────────────────────────────────────────────────
df_raw = pd.read_csv("Data/train_data.csv")

print(f"Dataset shape: {df_raw.shape}")
print(list(df_raw.columns))

# ── Train/test split BEFORE preprocessing (prevents leakage) ──────────────────
df_train_raw, df_test_raw = train_test_split(df_raw, test_size=0.2, random_state=42)

# ── Preprocess training set ───────────────────────────────────────────────────
df_train, y_train_full = build_training_df(df_train_raw)
# Preprocess test set separately (fit nothing on test)
df_test, y_test_full = build_training_df(df_test_raw)

# ── Keep only inferrable features ────────────────────────────────────────────
# These 51 features can all be reconstructed from UI inputs at inference time.
inferrable_cols = [c for c in INFERRABLE_FEATURES if c in df_train.columns]
print(f"\nTraining on {len(inferrable_cols)} inferrable features.")
print(inferrable_cols)

df_model_train = df_train[inferrable_cols].copy()
df_model_test  = df_test[inferrable_cols].copy()

# Fill any test-set nulls using train medians
train_medians = df_model_train.median(numeric_only=True).to_dict()
df_model_train = df_model_train.fillna(train_medians)
df_model_test  = df_model_test.fillna(train_medians)

# Align y to df index after preprocessing (drop_duplicates may have removed rows)
y_train = y_train_full.loc[df_model_train.index]
y_test  = y_test_full.loc[df_model_test.index]

# ── Save feature metadata for App.py inference ───────────────────────────────
os.makedirs("Models", exist_ok=True)
joblib.dump(inferrable_cols, "Models/feature_columns.pkl")
joblib.dump(train_medians,   "Models/feature_medians.pkl")
print(f"Saved {len(inferrable_cols)} feature columns and medians to Models/")

X_train = df_model_train.values
X_test  = df_model_test.values


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate(y_true, y_pred):
    return {
        "MSE":  mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "R2":   r2_score(y_true, y_pred),
    }


# ── Model training functions ──────────────────────────────────────────────────

def train_linear(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, {
        "train": evaluate(y_train, model.predict(X_train)),
        "test":  evaluate(y_test,  model.predict(X_test)),
    }


def train_polynomial(X_train, X_test, y_train, y_test, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_tr_p = poly.fit_transform(X_train)
    X_te_p = poly.transform(X_test)
    model = Ridge(alpha=1.0)
    model.fit(X_tr_p, y_train)
    return model, poly, {
        "train": evaluate(y_train, model.predict(X_tr_p)),
        "test":  evaluate(y_test,  model.predict(X_te_p)),
    }


def train_ridge(X_train, X_test, y_train, y_test, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model, {
        "train": evaluate(y_train, model.predict(X_train)),
        "test":  evaluate(y_test,  model.predict(X_test)),
    }


def train_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(
        n_estimators=300, max_depth=None,
        min_samples_split=5, min_samples_leaf=2,
        max_features="sqrt", bootstrap=True, random_state=42,
    )
    model.fit(X_train, y_train)
    return model, {
        "train": evaluate(y_train, model.predict(X_train)),
        "test":  evaluate(y_test,  model.predict(X_test)),
    }


def train_gradient_boosting(X_train, X_test, y_train, y_test):
    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=3, subsample=0.8,
        min_samples_leaf=3, random_state=42,
    )
    model.fit(X_train, y_train)
    return model, {
        "train": evaluate(y_train, model.predict(X_train)),
        "test":  evaluate(y_test,  model.predict(X_test)),
    }


def train_xgboost(X_train, X_test, y_train, y_test):
    model = xgb.XGBRegressor(
        n_estimators=800, learning_rate=0.03,
        max_depth=4, subsample=0.7, colsample_bytree=0.7,
        min_child_weight=5, gamma=0.2,
        reg_alpha=0.3, reg_lambda=2.0, random_state=42,
    )
    model.fit(X_train, y_train)
    return model, {
        "train": evaluate(y_train, model.predict(X_train)),
        "test":  evaluate(y_test,  model.predict(X_test)),
    }


# ── Run all models ────────────────────────────────────────────────────────────

def run_models(X_train, X_test, y_train, y_test, save_dir="Models"):
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    # Linear
    lr_model, lr_res = train_linear(X_train, X_test, y_train, y_test)
    results["Linear Regression Train"] = lr_res["train"]
    results["Linear Regression Test"]  = lr_res["test"]
    joblib.dump(lr_model, f"{save_dir}/linear.pkl")
    print("Saved linear.pkl")

    # Polynomial
    poly_model, poly_transformer, poly_res = train_polynomial(X_train, X_test, y_train, y_test)
    results["Polynomial Regression Train"] = poly_res["train"]
    results["Polynomial Regression Test"]  = poly_res["test"]
    joblib.dump(poly_model,       f"{save_dir}/polynomial.pkl")
    joblib.dump(poly_transformer, f"{save_dir}/polynomial_transformer.pkl")
    print("Saved polynomial.pkl + polynomial_transformer.pkl")

    # Ridge
    ridge_model, ridge_res = train_ridge(X_train, X_test, y_train, y_test)
    results["Ridge Train"] = ridge_res["train"]
    results["Ridge Test"]  = ridge_res["test"]
    joblib.dump(ridge_model, f"{save_dir}/ridge.pkl")
    print("Saved ridge.pkl")

    # Random Forest
    rf_model, rf_res = train_random_forest(X_train, X_test, y_train, y_test)
    results["Random Forest Train"] = rf_res["train"]
    results["Random Forest Test"]  = rf_res["test"]
    joblib.dump(rf_model, f"{save_dir}/random_forest.pkl")
    print("Saved random_forest.pkl")

    # Gradient Boosting
    gb_model, gb_res = train_gradient_boosting(X_train, X_test, y_train, y_test)
    results["Gradient Boosting Train"] = gb_res["train"]
    results["Gradient Boosting Test"]  = gb_res["test"]
    joblib.dump(gb_model, f"{save_dir}/gradient_boosting.pkl")
    print("Saved gradient_boosting.pkl")

    # XGBoost
    xgb_model, xgb_res = train_xgboost(X_train, X_test, y_train, y_test)
    results["XGBoost Train"] = xgb_res["train"]
    results["XGBoost Test"]  = xgb_res["test"]
    joblib.dump(xgb_model, f"{save_dir}/xgboost.pkl")
    print("Saved xgboost.pkl")

    # Save test metrics for App.py metric cards
    model_metrics = {
        "Linear Regression (Project)": lr_res["test"],
        "Polynomial Regression":        poly_res["test"],
        "Ridge":                        ridge_res["test"],
        "Random Forest":                rf_res["test"],
        "Gradient Boosting":            gb_res["test"],
        "XGBoost":                      xgb_res["test"],
    }
    joblib.dump(model_metrics, f"{save_dir}/model_metrics.pkl")
    print("Saved model_metrics.pkl")

    results_df = pd.DataFrame(results).T
    print("\nResults (Train vs Test):")
    print(results_df)

    test_rows = results_df.loc[results_df.index.str.contains("Test")]
    best_model = test_rows["R2"].idxmax()
    print(f"\nBest Model (Test R2): {best_model}")

    return results_df


run_models(X_train, X_test, y_train, y_test, save_dir="Models")
