
import pandas as pd
import numpy as np
import os

# Reproducible dataset
np.random.seed(42)

# Number of records
n = 1000

# Generate applicant information
data = {
    "ApplicantIncome": np.random.randint(1500, 30000, n),

    "CoapplicantIncome": np.random.randint(0, 15000, n),

    "LoanAmount": np.random.randint(20, 600, n),

    "Loan_Amount_Term": np.random.choice(
        [120, 180, 240, 300, 360, 480],
        n
    ),

    "Credit_History": np.random.choice(
        [0, 1],
        n,
        p=[0.25, 0.75]
    ),

    "Education": np.random.choice(
        ["Graduate", "Not Graduate"],
        n,
        p=[0.75, 0.25]
    ),

    "Self_Employed": np.random.choice(
        ["Yes", "No"],
        n,
        p=[0.15, 0.85]
    )
}

df = pd.DataFrame(data)

# Generate loan approval labels
def determine_loan_status(row):

    score = 0

    # Credit history
    if row["Credit_History"] == 1:
        score += 4

    # Applicant income
    if row["ApplicantIncome"] >= 10000:
        score += 2
    elif row["ApplicantIncome"] >= 5000:
        score += 1

    # Combined income
    if (
        row["ApplicantIncome"] +
        row["CoapplicantIncome"]
    ) >= 15000:
        score += 2

    # Education
    if row["Education"] == "Graduate":
        score += 1

    # Loan amount
    if row["LoanAmount"] <= 200:
        score += 1

    return "Y" if score >= 5 else "N"


df["Loan_Status"] = df.apply(
    determine_loan_status,
    axis=1
)

# Save Excel dataset
output_path = os.path.join(
    "dataset",
    "loan_prediction_dataset.xlsx"
)

df.to_excel(output_path, index=False)

# Display dataset information
print("\n========== DATASET GENERATED ==========")

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nLoan Status Distribution:")
print(df["Loan_Status"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset saved successfully at:")
print(output_path)

print("\n========== COMPLETED ==========")
