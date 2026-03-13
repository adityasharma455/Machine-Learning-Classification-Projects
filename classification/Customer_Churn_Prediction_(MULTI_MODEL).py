import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric and drop null values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Feature & target separation
X = df.drop(["customerID", "Churn"], axis=1)
y = df["Churn"].values

# One-hot encoding for categorical features
X = pd.get_dummies(
    X,
    columns=[
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ],
    drop_first=True,
    dtype=int
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models dictionary
models = {
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "Naive Bayes": BernoulliNB(),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

# Create accuracy comparison graph
model_names = list(results.keys())
accuracies = list(results.values())

plt.figure()
plt.bar(model_names, accuracies)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison - Customer Churn Prediction")
plt.xticks(rotation=45)

plt.tight_layout()

# Save graph as image
plt.savefig("model_accuracy_comparison.png")

plt.close()
