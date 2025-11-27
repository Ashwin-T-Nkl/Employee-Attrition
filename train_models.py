import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import joblib

INPUT_FILE = "clean_employee.csv"

# Simple common feature list for both models
feature_cols = [
    "Age",
    "MonthlyIncome",
    "JobSatisfaction",
    "YearsAtCompany",
    "Education",
    "JobInvolvement",
    "JobLevel",
    "YearsInCurrentRole",
]

print("Step 3: Training models...")

try:
    df = pd.read_csv(INPUT_FILE)
    print("File loaded. Rows:", len(df))
except FileNotFoundError:
    print("clean_employee.csv not found.")
    exit()

# Keep only rows that have all needed columns and targets
needed_cols = feature_cols + ["Attrition", "PerformanceRating"]
for col in needed_cols:
    if col not in df.columns:
        print(f"Missing column in data: {col}")
        exit()

df = df.dropna(subset=needed_cols)

# =========================
# 1) Attrition Model (Logistic Regression)
# =========================

x = df[feature_cols]
y = df["Attrition"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=4
)

clf = LogisticRegression(max_iter=2000)
clf.fit(x_train, y_train)

yp = clf.predict(x_test)

print("\n--- Attrition Model (Logistic Regression) ---")
print("Accuracy (clf.score):", clf.score(x_test, y_test))
print("Accuracy (accuracy_score):", accuracy_score(y_test, yp))

joblib.dump(clf, "attrition_model.pkl")
print("Saved: attrition_model.pkl")

# =========================
# 2) Performance Rating Model (Linear Regression)
# =========================

x2 = df[feature_cols]
y2 = df["PerformanceRating"]

x_train, x_test, y_train, y_test = train_test_split(
    x2, y2, test_size=0.2, random_state=4
)

reg = LinearRegression()
reg.fit(x_train, y_train)

yp2 = reg.predict(x_test)

print("\n--- Performance Rating Model (Linear Regression) ---")
print("R2 (reg.score):", reg.score(x_test, y_test))
print("R2 (r2_score):", r2_score(y_test, yp2))
print("MSE:", mean_squared_error(y_test, yp2))

joblib.dump(reg, "performance_model.pkl")
print("Saved: performance_model.pkl")

print("\n✅ Model training completed.")
