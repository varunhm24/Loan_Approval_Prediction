
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==================================
# 1. LOAD DATASET
# ==================================

print("\n========== LOAN APPROVAL ML ==========")

df = pd.read_excel(
    "dataset/loan_prediction_dataset.xlsx"
)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

# ==================================
# 2. DATA INSPECTION
# ==================================

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# ==================================
# 3. SEPARATE FEATURES AND TARGET
# ==================================

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"].map({"Y": 1, "N": 0})

# ==================================
# 4. IDENTIFY COLUMN TYPES
# ==================================

numerical_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

categorical_features = [
    "Education",
    "Self_Employed"
]

# ==================================
# 5. PREPROCESSING
# ==================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# ==================================
# 6. TRAIN TEST SPLIT
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ==================================
# 7. DEFINE MODELS
# ==================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
}

# ==================================
# 8. TRAIN AND EVALUATE
# ==================================

results = []
trained_models = {}

for name, classifier in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[name] = pipeline

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# ==================================
# 9. MODEL COMPARISON
# ==================================

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# ==================================
# 10. SELECT BEST MODEL
# ==================================

# Select using F1-score rather than accuracy alone
best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]

best_model = trained_models[best_model_name]

print("\nBest model:", best_model_name)

# ==================================
# 11. SAVE MODEL
# ==================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/loan_approval_model.pkl"
)

# Save comparison results
results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\nModel saved successfully!")
print("Location: models/loan_approval_model.pkl")

print("\n========== TRAINING COMPLETED ==========")
