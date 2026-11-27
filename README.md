# Loan Approval Prediction System

live demo:https://loan-approval-prediction-eto9.onrender.com/

## Project Overview

The **Loan Approval Prediction System** is a Machine Learning-based application designed to predict whether a loan application is likely to be approved or rejected based on applicant details and financial information.

The system analyzes various factors such as applicant income, loan amount, credit history, employment status, education, and other relevant attributes to generate predictions using trained Machine Learning models.

The main objective is to automate the loan eligibility prediction process and support faster, data-driven lending decisions.

## Objectives

- Predict loan approval status using Machine Learning.
- Analyze financial and personal details of loan applicants.
- Identify important factors influencing loan approval.
- Compare different Machine Learning algorithms.
- Develop an easy-to-use prediction interface.
- Support efficient and consistent loan application assessment.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data preprocessing and analysis |
| NumPy | Numerical computation |
| Scikit-learn | Machine Learning algorithms |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Flask | Web application development |
| HTML | Frontend structure |
| CSS | Frontend styling |
| Joblib | Model serialization |

## Machine Learning Workflow

1. Dataset Collection
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Handling Missing Values
5. Categorical Feature Encoding
6. Feature Selection
7. Model Training
8. Model Evaluation
9. Best Model Selection
10. Model Deployment Using Flask

## Features

- Loan approval prediction.
- Applicant financial information analysis.
- Data preprocessing and feature handling.
- Machine Learning model integration.
- User-friendly web interface.
- Fast prediction results.
- Model performance evaluation.
- Automated approval/rejection classification.

## Input Features

The system can consider the following applicant details, depending on the dataset:

- Gender
- Marital Status
- Dependents
- Education
- Self Employment Status
- Applicant Income
- Co-applicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area

## Project Structure

```text
Loan_Approval_Prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
│
├── dataset/
│   └── loan_data.csv
│
├── models/
│   └── best_model.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── css/
│       └── style.css
│
├── charts/
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── other_visualizations.png
│
└── README.md
```

