
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_excel(
    "dataset/loan_prediction_dataset.xlsx"
)

# Create charts directory
os.makedirs("static/charts", exist_ok=True)

sns.set_theme(style="whitegrid")

# ==================================
# 1. LOAN APPROVAL DISTRIBUTION
# ==================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Loan_Status"
)

plt.title("Loan Approval Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applications")

plt.savefig(
    "static/charts/loan_distribution.png",
    bbox_inches="tight"
)

plt.close()


# ==================================
# 2. APPLICANT INCOME DISTRIBUTION
# ==================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="ApplicantIncome",
    bins=30,
    kde=True
)

plt.title("Applicant Income Distribution")
plt.xlabel("Applicant Income")
plt.ylabel("Frequency")

plt.savefig(
    "static/charts/income_distribution.png",
    bbox_inches="tight"
)

plt.close()


# ==================================
# 3. CREDIT HISTORY ANALYSIS
# ==================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Credit_History",
    hue="Loan_Status"
)

plt.title("Credit History vs Loan Status")
plt.xlabel("Credit History")
plt.ylabel("Number of Applicants")

plt.savefig(
    "static/charts/credit_history.png",
    bbox_inches="tight"
)

plt.close()


# ==================================
# 4. EDUCATION ANALYSIS
# ==================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Education",
    hue="Loan_Status"
)

plt.title("Education vs Loan Status")
plt.xlabel("Education")
plt.ylabel("Number of Applicants")

plt.savefig(
    "static/charts/education_analysis.png",
    bbox_inches="tight"
)

plt.close()


# ==================================
# 5. MODEL COMPARISON
# ==================================

results = pd.read_csv(
    "models/model_comparison.csv"
)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results,
    x="Model",
    y="Accuracy"
)

plt.title("Machine Learning Model Accuracy")
plt.xlabel("ML Models")
plt.ylabel("Accuracy")

plt.ylim(0, 1.05)
plt.xticks(rotation=15)

plt.savefig(
    "static/charts/model_comparison.png",
    bbox_inches="tight"
)

plt.close()


print("\n========== VISUALIZATION COMPLETED ==========")

print("Generated charts:")

for file in os.listdir("static/charts"):
    print(file)

print("\nCharts saved inside static/charts/")

print("\n========== SUCCESS ==========")
