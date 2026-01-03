
# Imports
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error

import xgboost as xgb
import joblib
import os

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load data from Hugging Face
Xtrain_path = "hf://datasets/karthick2613/superkart_mlops/Xtrain.csv"
Xtest_path  = "hf://datasets/karthick2613/superkart_mlops/Xtest.csv"
ytrain_path = "hf://datasets/karthick2613/superkart_mlops/ytrain.csv"
ytest_path  = "hf://datasets/karthick2613/superkart_mlops/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest  = pd.read_csv(ytest_path).squeeze()

# Feature definitions
categorical_features = [
    'Product_Sugar_Content',
    'Product_Type',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type'
]

numeric_features = [
    'Product_Weight',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Establishment_Year'
]

# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# XGBoost Regressor
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

# Hyperparameter grid
param_grid = {
    'xgbregressor__n_estimators': [100, 200],
    'xgbregressor__max_depth': [3, 5, 7],
    'xgbregressor__learning_rate': [0.03, 0.05, 0.1],
    'xgbregressor__subsample': [0.7, 0.9],
    'xgbregressor__colsample_bytree': [0.7, 0.9],
    'xgbregressor__reg_lambda': [0.5, 1.0, 1.5]
}

# Pipeline
model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)

# Adjusted R² scorer
def adjusted_r2(estimator, X, y):
    y_pred = estimator.predict(X)
    r2 = r2_score(y, y_pred)
    n = X.shape[0]
    p = estimator.named_steps['xgbregressor'].n_features_in_
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))

# Grid Search - Hyperparameter tuning
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring=adjusted_r2,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(Xtrain, ytrain)

# Best model
best_model = grid_search.best_estimator_
print("Best Parameters:\n", grid_search.best_params_)

# Evaluation
y_train_pred = best_model.predict(Xtrain)
y_test_pred  = best_model.predict(Xtest)

train_r2 = r2_score(ytrain, y_train_pred)
test_r2  = r2_score(ytest, y_test_pred)

train_rmse = root_mean_squared_error(ytrain, y_train_pred)
test_rmse  = root_mean_squared_error(ytest, y_test_pred)

print("\nTrain R²:", train_r2)
print("Test R²:", test_r2)
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# Save model
model_path = "best_product_sales_xgb_regressor_v1.joblib"
joblib.dump(best_model, model_path)

# Upload model to Hugging Face
repo_id = "karthick2613/product_sales_xgb_regressor"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Created model repo '{repo_id}'.")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type=repo_type
)
