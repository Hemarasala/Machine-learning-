import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ================================
# STEP 1: Load Dataset
# ================================

df = pd.read_csv("smoking.csv")

# Remove hidden spaces in column names
df.columns = df.columns.str.strip()
print("Dataset Shape:", df.shape)
print("\nColumns in Dataset:")
print(df.columns)
print("\nFirst 5 Rows:")
print(df.head())

# ================================
# STEP 2: Handle Missing Values
# ================================

print("\nMissing Values:")
print(df.isnull().sum())

df = df.fillna(df.mean(numeric_only=True))

# ================================
# STEP 3: Define Target Column
# ================================

# ✅ IMPORTANT: Change this if your actual column name is different
target_column = "Smoking"   # <-- CHANGE if needed

if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset!")

# Separate features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Keep only numeric features
X = X.select_dtypes(include=[np.number])

print("\nNumber of Features Used:", X.shape[1])

# ================================
# STEP 4: Train-Test Split
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# STEP 5: Feature Scaling
# ================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# STEP 6A: Logistic Regression
# ================================

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\n===== Logistic Regression Results =====")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))

# ================================
# STEP 6B: Random Forest
# ================================

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n===== Random Forest Results =====")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

# ================================
# STEP 7: Test with New Sample
# ================================

print("\n===== Predicting New Sample =====")

# Create a random sample automatically with correct feature size
new_sample = np.random.rand(1, X.shape[1])

new_sample_scaled = scaler.transform(new_sample)
prediction = rf_model.predict(new_sample_scaled)

if prediction[0] == 1:
    print("Prediction: Smoking Person 🚬")
else:
    print("Prediction: Non-Smoking Person 🙂")
