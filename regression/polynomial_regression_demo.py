import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate synthetic nonlinear dataset
X = 6 * np.random.rand(200, 1) - 3
y = 0.8 * X**2 + 0.9 * X + 2 + np.random.rand(200, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3
)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

print(f"R² Score (Linear Regression): {r2_score(y_test, linear_pred):.4f}")

# Polynomial Regression
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
poly_pred = poly_model.predict(X_test_poly)

print(f"R² Score (Polynomial Regression): {r2_score(y_test, poly_pred):.4f}")

# Visualization
X_plot = np.linspace(-3, 3, 200).reshape(200, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = poly_model.predict(X_plot_poly)

plt.figure()

plt.plot(X_plot, y_plot, label="Polynomial Prediction")
plt.scatter(X_train, y_train, label="Training Data")
plt.scatter(X_test, y_test, label="Test Data")

plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Polynomial Regression Fit")

plt.legend()

plt.savefig("polynomial_regression_plot.png")
plt.close()