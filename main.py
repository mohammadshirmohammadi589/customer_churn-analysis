import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Customer Churn", layout="centered")

# =============================
# 🔐 LOGIN SYSTEM
# =============================

CREDENTIALS = {
    "Datavisionary": "Datacode2024"
}

USERS_FILE = "users_usage.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def check_usage(username):
    users = load_users()
    return users.get(username, 0)

def increment_usage(username):
    users = load_users()
    users[username] = users.get(username, 0) + 1
    save_users(users)

# Session
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

if not st.session_state.authenticated:
    st.title("🔐 Customer Churn Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in CREDENTIALS and CREDENTIALS[username] == password:
            if check_usage(username) >= 3:
                st.error("❌ Prediction limit reached.")
                st.stop()

            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("✅ Login successful")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

    st.stop()

# =============================
# 🤖 LOAD MODEL (Pipeline Safe)
# =============================

@st.cache_resource
def load_model():
    return joblib.load("voting_classifier_final_model.pkl")

model = load_model()

# =============================
# 📋 SIDEBAR
# =============================

st.sidebar.title("Menu")
menu = st.sidebar.radio("Select Page", ["Forecast", "About"])

# =============================
# 🔮 FORECAST
# =============================

if menu == "Forecast":

    st.title("📊 Customer Churn Prediction")

    usage = check_usage(st.session_state.username)
    remaining = 3 - usage
    st.info(f"Remaining Predictions: {remaining}")

    if remaining <= 0:
        st.error("❌ Limit exceeded")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["No", "Yes"])
        Dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure", 0, 72, 12)
        MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
        TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    with col2:
        PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
        MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
        OnlineBackup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
        DeviceProtection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
        TechSupport = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
        StreamingTV = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
    PaymentMethod = st.selectbox("Payment Method",
        ["Bank transfer (automatic)", "Credit card (automatic)",
         "Electronic check", "Mailed check"])

    if st.button("Predict"):

        input_df = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")
        st.metric("Churn Probability", f"{probability*100:.2f}%")

        if prediction == 1:
            st.error("🚨 Customer likely to churn")
        else:
            st.success("✅ Customer likely to stay")

        increment_usage(st.session_state.username)

# =============================
# ℹ️ ABOUT
# =============================

elif menu == "About":
    st.title("About")
    st.write("Customer Churn Prediction using ML Pipeline with SMOTE + CatBoost.")
