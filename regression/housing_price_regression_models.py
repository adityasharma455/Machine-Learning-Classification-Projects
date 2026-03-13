import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

# Load California housing dataset
housing = fetch_california_housing()

data = pd.DataFrame(housing.data, columns=housing.feature_names)
target = pd.DataFrame(housing.target, columns=["Price"])
df = pd.concat([data, target], axis=1)

print("Dataset Preview:")
print(df.head())

# Split features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
print(f"R² Score (Linear Regression): {lr_r2:.4f}")

# Ridge Regression
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_r2 = r2_score(y_test, ridge_pred)
print(f"R² Score (Ridge Regression): {ridge_r2:.4f}")

# Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)
lasso_r2 = r2_score(y_test, lasso_pred)
print(f"R² Score (Lasso Regression): {lasso_r2:.4f}")

# Identify features removed by Lasso
zero_coef_idx = np.where(lasso_model.coef_ == 0)[0]
removed_features = list(X.columns[zero_coef_idx])
print("Features removed by Lasso:", removed_features)

# Remove these features
X_train_filtered = X_train.drop(columns=removed_features)
X_test_filtered = X_test.drop(columns=removed_features)

# Train again after removing features
filtered_lr = LinearRegression()
filtered_lr.fit(X_train_filtered, y_train)
filtered_pred = filtered_lr.predict(X_test_filtered)

print(
    f"R² Score (Linear Regression after feature removal): "
    f"{r2_score(y_test, filtered_pred):.4f}"
)

models = ["Linear Regression", "Ridge Regression", "Lasso Regression"]
scores = [lr_r2, ridge_r2, lasso_r2]
plt.figure()

plt.bar(models, scores)

plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.title("Model Performance Comparison")

plt.savefig("housing_model_comparison.png")

plt.close()