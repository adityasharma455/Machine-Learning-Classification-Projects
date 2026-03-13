import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("breast_cancer.csv")

# Encode target variable
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Feature & target split
X = df.iloc[:, 2:21]
y = df["diagnosis"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Base Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Manual Hyperparameter Tuning: n_estimators
print("\nTuning n_estimators:")
for n in [1, 2, 10, 50, 100, 200]:
    model = RandomForestClassifier(n_estimators=n)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"n_estimators={n} → Accuracy: {accuracy_score(y_test, preds):.4f}")

# Manual Hyperparameter Tuning: min_samples_leaf
print("\nTuning min_samples_leaf:")
for leaf in [1, 2, 3, 5, 10]:
    model = RandomForestClassifier(n_estimators=50, min_samples_leaf=leaf)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"leaf={leaf} → Accuracy: {accuracy_score(y_test, preds):.4f}")
