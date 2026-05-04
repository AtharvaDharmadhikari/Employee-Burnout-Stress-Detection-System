"""
Burnout Model Trainer
Trains a GradientBoostingRegressor on the Kaggle "Are Your Employees Burning Out?" dataset.

Steps:
  1. Download train.csv from Kaggle:
     https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out
  2. Place it at: data/train.csv
  3. Run: python train_burnout_model.py
  4. Model saved to: models/burnout_model.pkl

Feature mapping (Kaggle → our app):
  Mental Fatigue Score (0-10) → avg_stress collected from mood logs
  Resource Allocation  (1-10) → avg workload from task logs (default 5 if none)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

CSV_PATH   = os.path.join("data", "train.csv")
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "burnout_model.pkl")
META_PATH  = os.path.join(MODEL_DIR, "burnout_model_meta.pkl")

FEATURES = ["Mental Fatigue Score", "Resource Allocation"]
TARGET   = "Burn Rate"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.columns.tolist()}")

    # Drop rows where target is missing
    before = len(df)
    df = df.dropna(subset=[TARGET])
    print(f"Dropped {before - len(df)} rows with missing Burn Rate → {len(df):,} remaining")

    # Keep only the features we can replicate at runtime
    df = df[FEATURES + [TARGET]].copy()

    # Fill missing feature values with median
    for col in FEATURES:
        median = df[col].median()
        n_missing = df[col].isna().sum()
        if n_missing:
            print(f"  Imputing {n_missing} missing values in '{col}' with median={median:.2f}")
        df[col] = df[col].fillna(median)

    return df


def train(df: pd.DataFrame):
    X = df[FEATURES].values
    y = df[TARGET].values       # 0.0 – 1.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    print("\n── Model Performance ──────────────────────────────────────────")
    print(f"  Test MAE  : {mae:.4f}  (burn rate units, 0-1 scale)")
    print(f"  Test R²   : {r2:.4f}")
    print(f"  5-fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Feature importances:")
    for feat, imp in zip(FEATURES, model.feature_importances_):
        print(f"    {feat:<30} {imp:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "features":      FEATURES,
        "target":        TARGET,
        "mae":           round(mae, 4),
        "r2":            round(r2, 4),
        "cv_r2_mean":    round(cv_scores.mean(), 4),
        "cv_r2_std":     round(cv_scores.std(), 4),
        "importances":   dict(zip(FEATURES, model.feature_importances_.tolist())),
        "training_rows": len(df),
    }
    joblib.dump(meta, META_PATH)

    print(f"\n✅ Model saved to  : {MODEL_PATH}")
    print(f"✅ Metadata saved  : {META_PATH}")
    return model, meta


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"❌ Dataset not found at '{CSV_PATH}'")
        print("   Download 'train.csv' from:")
        print("   https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out")
        print(f"   and place it at: {CSV_PATH}")
        sys.exit(1)

    df    = load_and_clean(CSV_PATH)
    train(df)