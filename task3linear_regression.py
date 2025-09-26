# linear_regression.py
# Corrected Linear Regression example with feature scaling

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1) CONFIG - edit DATA_PATH if your file has another name
DATA_PATH = os.path.join("data", "housing.csv")
TARGET_CANDIDATES = ["SalePrice", "price", "Price", "target", "median_house_value"]

# 2) LOAD
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Put your CSV in the data/ folder.")
df = pd.read_csv(DATA_PATH)
print("Loaded dataset:", DATA_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# 3) FIND TARGET
target = None
for t in TARGET_CANDIDATES:
    if t in df.columns:
        target = t
        break
if target is None:
    raise SystemExit("Could not auto-detect target column. Edit TARGET_CANDIDATES or set TARGET manually.")

y = df[target]

# 4) SELECT NUMERIC FEATURES
X = df.select_dtypes(include=[np.number]).copy()
if target in X.columns:
    X = X.drop(columns=[target])

print("Using numeric features (first 10):", X.columns.tolist()[:10])

# 5) HANDLE MISSING VALUES
X = X.fillna(X.median())
y = y.fillna(y.median())

# 6) SPLIT
if len(df) > 5:
    # Use train/test split if dataset is not tiny
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
else:
    # For tiny dataset, use all data for training/testing
    X_train, X_test, y_train, y_test = X, X, y, y

# 7) SCALE FEATURES
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8) FIT MODEL
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 9) PREDICT & EVALUATE
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nEvaluation on test set:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# 10) COEFFICIENTS
coeffs = pd.Series(model.coef_, index=X.columns)
coeffs = coeffs.sort_values(key=lambda x: x.abs(), ascending=False)
print("\nTop coefficients (abs sorted):")
print(coeffs.head(15))

# 11) PLOTS - save to results/
os.makedirs("results", exist_ok=True)

if X.shape[1] == 1:
    feat = X.columns[0]
    plt.figure(figsize=(8,6))
    plt.scatter(X_test[feat], y_test, label="Actual", alpha=0.6)
    plt.scatter(X_test[feat], y_pred, label="Predicted", alpha=0.6)
    plt.xlabel(feat)
    plt.ylabel(target)
    plt.legend()
    plt.title("Actual vs Predicted")
    plt.savefig("results/simple_regression.png", bbox_inches="tight")
else:
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
    plt.xlabel("Actual " + target)
    plt.ylabel("Predicted " + target)
    plt.title("Actual vs Predicted")
    plt.savefig("results/pred_vs_actual.png", bbox_inches="tight")

print("\nPlots saved to results/ (simple_regression.png or pred_vs_actual.png)")
