import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import time
import warnings
warnings.filterwarnings('ignore')  # برای جلوگیری از warningهای sklearn

# 🔐 Credentials
CREDENTIALS = {
    'Datavisionary': 'Datacode2024'
}
USERS_FILE = 'users_usage.json'

# تابع ذخیره/لود users
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=4, ensure_ascii=False)

def check_usage(username):
    users = load_users()
    return users.get(username, {'usage_count': 0})['usage_count']

def increment_usage(username):
    users = load_users()
    if username not in users:
        users[username] = {'usage_count': 0}
    users[username]['usage_count'] += 1
    save_users(users)

# Authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

if not st.session_state.authenticated:
    st.title("🔐 Customer churn prediction login")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")
    
    if st.button("Login"):
        if username in CREDENTIALS and CREDENTIALS[username] == password:
            usage_count = check_usage(username)
            if usage_count >= 3:
                st.error("❌ You have made more than 3 predictions. Access is blocked.")
                st.stop()
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("✅ Successful login!")
            st.rerun()
        else:
            st.error("❌ The login information is incorrect.")
    st.stop()

# Demo Limitations
DEMO_VERSION = "true"
if DEMO_VERSION == "true":
    st.sidebar.warning("🔒 Demo version: Only 3 predictions allowed per user.")

# Model & Encoders Loading (فقط current_dir برای Cloud)
def load_artifacts():
    model_filename = 'voting_classifier_final_model.pkl'
    encoders_filename = 'label_encoders.pkl'
    
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, model_filename)
    encoders_path = os.path.join(current_dir, encoders_filename)
    
    # دیباگ
    st.write(f"Debug paths - Model: {model_path}, Encoders: {encoders_path}")  # در UI نشون بده برای تست
    
    if os.path.exists(model_path) and os.path.exists(encoders_path):
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        st.success("✅ The model and encoders were loaded.")
        return model, encoders
    else:
        st.error(f"❌ Files not found: Model={os.path.exists(model_path)}, encoders={os.path.exists(encoders_path)}")
        st.info("Download and upload .pkl files from Notebook.")
        return None, None

model, encoders = load_artifacts()

if model is None:
    st.stop()

# Sidebar
st.sidebar.title("📋 Menu")
menu = st.sidebar.radio("Choice:", ["Forecast", "About", "Analyses", "Upload"])
if st.sidebar.button("Exit"):
    st.session_state.clear()
    st.rerun()

# Predict Function (robust شده)
def predict(input_df, model, encoders):
    try:
        df = input_df.copy()

        # ✅ اگر ستون customerID وجود نداشت، مقدار ساختگی بساز
        if 'customerID' not in df.columns:
            df['customerID'] = '0000'

        # ✅ تبدیل عددی برای ستون‌های عددی
        numeric_cols = ['TotalCharges', 'MonthlyCharges', 'tenure', 'SeniorCitizen']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # ✅ کپی برای encode
        encoded_df = df.copy()
        categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                               'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                               'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                               'PaperlessBilling', 'PaymentMethod']

        # ✅ تبدیل LabelEncoderها
        for col in categorical_columns:
            if col in encoded_df.columns:
                if col in encoders:
                    classes = encoders[col].classes_
                    encoded_df[col] = encoded_df[col].apply(
                        lambda x: encoders[col].transform([str(x)])[0] if str(x) in classes else 0
                    )
                else:
                    encoded_df[col] = 0  # اگر encoder نبود، مقدار 0 بگذار

        # ✅ مرتب‌سازی نهایی ستون‌ها برای مدل
        encoded_df = encoded_df[['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                                 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                                 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                                 'MonthlyCharges', 'TotalCharges']]

        # ✅ چاپ برای دیباگ (در صفحه‌ی Streamlit نشان می‌دهد)
        st.write("✅ Encoded shape:", encoded_df.shape)
        st.write("✅ Columns:", encoded_df.columns.tolist())

        # ✅ پیش‌بینی
        encoded_df['customerID'] = 0
        pred = model.predict(encoded_df)
        proba = model.predict_proba(encoded_df)[:, 1]
        return pred, proba

    except Exception as e:
        st.error(f"❌ Error in prediction: {e}")
        st.write(f"Debug: Input columns: {input_df.columns.tolist()}")
        return None, None

# Prediction Section
if menu == "Forecast":
    st.title("🔍Customer churn prediction")
    usage_count = check_usage(st.session_state.username)
    remaining = 3 - usage_count
    st.info(f"Number of remaining predictions: {remaining}")
    if remaining <= 0:
        st.error("❌ Limit exceeded. Contact admin.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Elderly؟", [0, 1])
        Partner = st.selectbox("Partner?", ["No", "Yes"])
        Dependents = st.selectbox("Dependents?", ["No", "Yes"])
        tenure = st.slider("Subscription period (months)", 0, 72, 1)
        MonthlyCharges = st.number_input("Monthly fee", min_value=0.0, value=20.0)
        TotalCharges = st.number_input("Total cost", min_value=0.0, value=20.0)

    with col2:
        PhoneService = st.selectbox("Telephone services", ["No", "Yes"])
        MultipleLines = st.selectbox("How many lines?", ["No phone service", "No", "Yes"])
        InternetService = st.selectbox("Internet services", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online security", ["No internet service", "No", "Yes"])
        OnlineBackup = st.selectbox("Online backup", ["No internet service", "No", "Yes"])
        DeviceProtection = st.selectbox("Device protection", ["No internet service", "No", "Yes"])
        TechSupport = st.selectbox("Technical support", ["No internet service", "No", "Yes"])
        StreamingTV = st.selectbox("TV streaming", ["No internet service", "No", "Yes"])
        StreamingMovies = st.selectbox("Movie streaming", ["No internet service", "No", "Yes"])

    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless billing", ["No", "Yes"])
    PaymentMethod = st.selectbox("Payment method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])

    if st.button("Forecast"):
        input_data = {
            "customerID": [f"ID-{np.random.randint(1000,9999)}"],
            "gender": [gender], "SeniorCitizen": [SeniorCitizen], "Partner": [Partner],
            "Dependents": [Dependents], "tenure": [tenure], "PhoneService": [PhoneService],
            "MultipleLines": [MultipleLines], "InternetService": [InternetService],
            "OnlineSecurity": [OnlineSecurity], "OnlineBackup": [OnlineBackup],
            "DeviceProtection": [DeviceProtection], "TechSupport": [TechSupport],
            "StreamingTV": [StreamingTV], "StreamingMovies": [StreamingMovies],
            "Contract": [Contract], "PaperlessBilling": [PaperlessBilling],
            "PaymentMethod": [PaymentMethod], "MonthlyCharges": [MonthlyCharges],
            "TotalCharges": [TotalCharges]
        }
        input_df = pd.DataFrame(input_data)
        
        pred, proba = predict(input_df, model, encoders)
        if pred is not None:
            st.subheader("Prediction result")
            churn_prob = proba[0] * 100
            st.metric("Possibility of falling", f"{churn_prob:.1f}%")
            if pred[0] == 1:
                st.error("🚨 There is customer churn.!")
            else:
                st.success("✅ has no customer churn!")
            
            # Feature Importance (اصلاح‌شده)
            try:
                importances = None
                tree_names = ['gbc', 'abc']  # فقط tree-based
                if hasattr(model, 'named_estimators_'):
                    for name in tree_names:
                        est = model.named_estimators_.get(name)
                        if est and hasattr(est, 'feature_importances_'):
                            importances = est.feature_importances_
                            st.write(f"از estimator {name} Used.")
                            break
                
                if importances is not None:
                    feature_names = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                 'MonthlyCharges', 'TotalCharges']

                    if len(importances) == len(feature_names):
                        feat_imp = pd.DataFrame({
                            "Feature": feature_names,
                            "Importance": importances
                        }).sort_values("Importance", ascending=False).head(10)
                        fig = px.bar(feat_imp, x="Importance", y="Feature", title="Important features")
                        st.plotly_chart(fig)
                    else:
                        st.warning("⚠️Feature mismatch.")
                else:
                    st.info("ℹ️ Importance of features is not available..")
            except Exception as e:
                st.info(f"ℹ️ Error in significance: {e}")
            
            # Increment usage
            increment_usage(st.session_state.username)
            new_remaining = 3 - check_usage(st.session_state.username)
            st.info(f"Remainder: {new_remaining}")
            if new_remaining <= 0:
                st.warning("It was the last prediction. Exit...")
                if st.button("Exit"):
                    st.session_state.clear()
                    st.rerun()

# Other Sections
elif menu == "About":
    st.title("About")
    st.markdown("Forecasting the fall with Voting Classifier (GBC + LR + ABC). Accuracy: ~80% Recall Weighted.")

elif menu == "Analyses":
    st.title("Analyses")
    st.info("Sample charts – add later.")

elif menu == "Upload":
    st.title("Upload")
    uploaded_file = st.file_uploader("CSV Upload it.", type="csv")
    if uploaded_file:
        df_up = pd.read_csv(uploaded_file)
        st.write("Data:", df_up.head())

        # predict batch اگر خواستی اضافه کن




