
# for data handling
import pandas as pd
import numpy as np
import os

# models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# pipeline
from sklearn.pipeline import Pipeline

# model selection
from sklearn.model_selection import RandomizedSearchCV

# evaluation
from sklearn.metrics import mean_squared_error, r2_score

# saving model
import joblib

# hugging face upload
from huggingface_hub import HfApi

# -----------------------------
# Load data
# -----------------------------
Xtrain = pd.read_csv("SuperKart/data/Xtrain.csv")
Xtest = pd.read_csv("SuperKart/data/Xtest.csv")
ytrain = pd.read_csv("SuperKart/data/ytrain.csv").values.ravel()
ytest = pd.read_csv("SuperKart/data/ytest.csv").values.ravel()

print("Data loaded successfully")

# -----------------------------
# Define Pipelines
# -----------------------------

rf_pipeline = Pipeline([
    ("model", RandomForestRegressor(random_state=42))
])

gb_pipeline = Pipeline([
    ("model", GradientBoostingRegressor(random_state=42))
])

# -----------------------------
# Hyperparameter tuning
# -----------------------------

rf_params = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [5, 10, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

gb_params = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [3, 5, 7]
}

print("Tuning Random Forest...")
rf_search = RandomizedSearchCV(
    rf_pipeline,
    rf_params,
    n_iter=10,
    cv=3,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=-1
)

rf_search.fit(Xtrain, ytrain)

print("Tuning Gradient Boosting...")
gb_search = RandomizedSearchCV(
    gb_pipeline,
    gb_params,
    n_iter=10,
    cv=3,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=-1
)

gb_search.fit(Xtrain, ytrain)

# -----------------------------
# Select best model
# -----------------------------

rf_best = rf_search.best_estimator_
gb_best = gb_search.best_estimator_

# Predictions (log scale)
rf_pred = rf_best.predict(Xtest)
gb_pred = gb_best.predict(Xtest)

# Convert back to original scale
rf_pred_actual = np.expm1(rf_pred)
gb_pred_actual = np.expm1(gb_pred)
ytest_actual = np.expm1(ytest)

# Evaluate
rf_rmse = np.sqrt(mean_squared_error(ytest_actual, rf_pred_actual))
gb_rmse = np.sqrt(mean_squared_error(ytest_actual, gb_pred_actual))

rf_r2 = r2_score(ytest_actual, rf_pred_actual)
gb_r2 = r2_score(ytest_actual, gb_pred_actual)

print("\nModel Performance:")
print(f"Random Forest → RMSE: {rf_rmse:.2f}, R2: {rf_r2:.4f}")
print(f"Gradient Boosting → RMSE: {gb_rmse:.2f}, R2: {gb_r2:.4f}")

# -----------------------------
# Choose best model
# -----------------------------

if rf_rmse < gb_rmse:
    best_model = rf_best
    model_name = "RandomForest"
else:
    best_model = gb_best
    model_name = "GradientBoosting"

print(f"\nBest Model: {model_name}")

# -----------------------------
# Save model
# -----------------------------

os.makedirs("SuperKart/models", exist_ok=True)

model_path = f"SuperKart/models/{model_name}.pkl"
joblib.dump(best_model, model_path)

print(f"Model saved at {model_path}")

# -----------------------------
# Upload model to Hugging Face
# -----------------------------

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_name + ".pkl",
    repo_id="PratzPrathibha/Super-kart-sales-prediction",
    repo_type="dataset",
)

print("Model uploaded to Hugging Face")
