"""
Customer Churn Prediction - Streamlit demo app.

Loads the trained pipeline from outputs/churn_model.joblib and scores a
single customer entered through the form. The trained model is bundled:

    python -m streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "outputs" / "churn_model.joblib"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉",
                   layout="centered")
st.title("📉 Customer Churn Predictor")
st.write(
    "Enter a customer's details to estimate their probability of churning. "
    "The model is a Logistic Regression classifier trained on the IBM Telco "
    "Customer Churn dataset (7,032 customers, ROC-AUC 0.836 on an untouched "
    "test set)."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=1.0)
    total = st.number_input("Total charges ($)", 0.0, 10000.0,
                            round(tenure * monthly, 2), step=10.0)
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    phone = st.selectbox("Phone service", ["Yes", "No"])
    multiple = st.selectbox("Multiple lines",
                            ["No", "Yes", "No phone service"])
with col2:
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online security",
                            ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
    device = st.selectbox("Device protection",
                          ["No", "Yes", "No internet service"])
    tech = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming movies",
                          ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract",
                            ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])

threshold = st.slider("Decision threshold (flag as churn risk if probability "
                      "is at least)", 0.20, 0.80, 0.50, step=0.05)

customer = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    "OnlineSecurity": security,
    "OnlineBackup": backup,
    "DeviceProtection": device,
    "TechSupport": tech,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
}])

prob = float(model.predict_proba(customer)[0, 1])
st.subheader(f"Churn probability: {prob:.1%}")
st.progress(min(max(prob, 0.0), 1.0))
if prob >= threshold:
    st.error(f"⚠️ At the {threshold:.2f} threshold this customer is flagged "
             "as a churn risk.")
else:
    st.success(f"✅ At the {threshold:.2f} threshold this customer is "
               "likely to stay.")
st.caption("Prediction is a statistical association learned from "
           "historical data, not a certainty about any individual.")
