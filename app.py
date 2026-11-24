
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import joblib
import sqlite3
import os
import math
from datetime import datetime

app = Flask(__name__)
app.secret_key = "loan_prediction_project_key"


# =====================================
# PATH CONFIGURATION
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "loan_approval_model.pkl"
)

DATABASE_PATH = os.path.join(
    BASE_DIR, "database", "loan_history.db"
)

COMPARISON_PATH = os.path.join(
    BASE_DIR, "models", "model_comparison.csv"
)

# Load trained machine learning model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Trained model not found. Run train_model.py first."
    )

model = joblib.load(MODEL_PATH)


# =====================================
# DATABASE CONFIGURATION
# =====================================

def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():

    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_income REAL,
            coapplicant_income REAL,
            loan_amount REAL,
            loan_term INTEGER,
            credit_history INTEGER,
            education TEXT,
            self_employed TEXT,
            prediction TEXT,
            approval_probability REAL,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


initialize_database()


# =====================================
# DASHBOARD
# =====================================

@app.route("/")
def index():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE prediction = 'Approved'
    """)
    approved_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE prediction = 'Rejected'
    """)
    rejected_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC LIMIT 5
    """)
    recent_predictions = cursor.fetchall()

    conn.close()

    model_results = []

    if os.path.exists(COMPARISON_PATH):
        try:
            df = pd.read_csv(COMPARISON_PATH)
            model_results = df.to_dict(orient="records")
        except Exception:
            app.logger.exception("Could not read model comparison file.")

    return render_template(
        "index.html",
        total_predictions=total_predictions,
        approved_count=approved_count,
        rejected_count=rejected_count,
        recent_predictions=recent_predictions,
        model_results=model_results
    )


# =====================================
# CUSTOMER LOAN PREDICTION
# =====================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Read customer details
        applicant_income = float(
            request.form["ApplicantIncome"]
        )

        coapplicant_income = float(
            request.form["CoapplicantIncome"]
        )

        loan_amount = float(
            request.form["LoanAmount"]
        )

        loan_term = int(
            request.form["Loan_Amount_Term"]
        )

        credit_history = int(
            request.form["Credit_History"]
        )

        education = request.form["Education"]

        self_employed = request.form["Self_Employed"]

        # Validate numeric values
        numeric_values = [
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term
        ]

        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("Please enter valid numeric values.")

        if applicant_income <= 0:
            raise ValueError("Applicant income must be greater than zero.")

        if coapplicant_income < 0:
            raise ValueError("Co-applicant income cannot be negative.")

        if loan_amount <= 0:
            raise ValueError("Loan amount must be greater than zero.")

        if loan_term <= 0:
            raise ValueError("Loan term must be greater than zero.")

        if credit_history not in [0, 1]:
            raise ValueError("Invalid credit history selection.")

        if education not in ["Graduate", "Not Graduate"]:
            raise ValueError("Invalid education selection.")

        if self_employed not in ["Yes", "No"]:
            raise ValueError("Invalid employment selection.")

        # Prepare input for ML model
        input_data = pd.DataFrame([{
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Education": education,
            "Self_Employed": self_employed
        }])

        # Generate prediction
        prediction_value = int(model.predict(input_data)[0])

        # Find probability belonging to class 1
        class_index = list(model.classes_).index(1)

        probability = float(
            model.predict_proba(input_data)[0][class_index]
        ) * 100

        if prediction_value == 1:
            result = "Approved"
        else:
            result = "Rejected"

        # Save prediction to SQLite database
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO predictions (
                applicant_income,
                coapplicant_income,
                loan_amount,
                loan_term,
                credit_history,
                education,
                self_employed,
                prediction,
                approval_probability,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history,
            education,
            self_employed,
            result,
            round(probability, 2),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()
        conn.close()

        # Display result page
        return render_template(
            "result.html",
            result=result,
            probability=round(probability, 2),
            applicant_income=applicant_income,
            coapplicant_income=coapplicant_income,
            loan_amount=loan_amount,
            loan_term=loan_term,
            credit_history=credit_history,
            education=education,
            self_employed=self_employed
        )

    except (ValueError, TypeError, KeyError) as error:

        flash(str(error), "error")
        return redirect(url_for("index"))

    except Exception:

        app.logger.exception("Loan prediction failed.")

        flash(
            "Unable to process prediction. Please check the application logs.",
            "error"
        )

        return redirect(url_for("index"))


# =====================================
# PREDICTION HISTORY
# =====================================

@app.route("/history")
def history():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
    """)

    records = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        records=records
    )


# =====================================
# CLEAR PREDICTION HISTORY
# =====================================

@app.route("/clear-history", methods=["POST"])
def clear_history():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM predictions")

    conn.commit()
    conn.close()

    flash(
        "Prediction history cleared successfully.",
        "success"
    )

    return redirect(url_for("history"))


# =====================================
# MODEL PERFORMANCE
# =====================================

@app.route("/model-performance")
def model_performance():

    model_results = []
    columns = []

    if os.path.exists(COMPARISON_PATH):

        try:
            df = pd.read_csv(COMPARISON_PATH)

            columns = df.columns.tolist()
            model_results = df.to_dict(orient="records")

        except Exception:
            app.logger.exception("Could not load model performance data.")

            flash(
                "Unable to load model performance data.",
                "error"
            )

    else:
        flash(
            "Model comparison file not found. Run train_model.py first.",
            "error"
        )

    return render_template(
        "model_performance.html",
        model_results=model_results,
        columns=columns
    )


# =====================================
# APPLICATION START
# =====================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
