
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

# 🔐 Credentials
CREDENTIALS = {
    'user1': 'password123',
    'user2': 'securepass456'
}
USERS_FILE = 'users_usage.json'

# تابع ذخیره/لود users
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

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
    st.title("🔐 ورود به سیستم پیش‌بینی ریزش مشتری")
    username = st.text_input("نام کاربری:")
    password = st.text_input("رمز عبور:", type="password")
    
    if st.button("ورود"):
        if username in CREDENTIALS and CREDENTIALS[username] == password:
            usage_count = check_usage(username)
            if usage_count >= 3:
                st.error("❌ شما بیش از ۳ پیش‌بینی انجام داده‌اید. دسترسی مسدود است.")
                st.stop()
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("✅ ورود موفق!")
            st.rerun()
        else:
            st.error("❌ اطلاعات ورود اشتباه است")
    st.stop()

# Demo Limitations
DEMO_VERSION = "true"
if DEMO_VERSION == "true":
    st.sidebar.warning("🔒 نسخه دمو: فقط ۳ پیش‌بینی مجاز برای هر کاربر.")

# Model & Encoders Loading
def load_artifacts():
    model_filename = 'voting_classifier_final_model.pkl'
    encoders_filename = 'label_encoders.pkl'
    
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, model_filename)
    if not os.path.exists(model_path):
        model_path = r'C:\Users\Brooz\my_churn_app\voting_classifier_final_model.pkl'
    
    encoders_path = os.path.join(current_dir, encoders_filename)
    if not os.path.exists(encoders_path):
        encoders_path = r'C:\Users\Brooz\my_churn_app\label_encoders.pkl'
    
    if os.path.exists(model_path) and os.path.exists(encoders_path):
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        st.success("✅ مدل و encoders لود شد.")
        return model, encoders
    else:
        st.error("❌ مدل یا encoders پیدا نشد.")
        return None, None

model, encoders = load_artifacts()

if model is None:
    st.stop()

# Sidebar
st.sidebar.title("📋 منو")
menu = st.sidebar.radio("انتخاب:", ["پیش‌بینی", "درباره", "تحلیل‌ها", "آپلود"])
if st.sidebar.button("خروج"):
    st.session_state.clear()
    st.rerun()

# Predict Function
def predict(input_df, model, encoders):
    try:
        input_data = input_df.copy()
        
        # Numeric conversion
        if 'TotalCharges' in input_data.columns:
            input_data['TotalCharges'] = pd.to_numeric(input_data['TotalCharges'], errors='coerce').fillna(0)
        if 'MonthlyCharges' in input_data.columns:
            input_data['MonthlyCharges'] = pd.to_numeric(input_data['MonthlyCharges'], errors='coerce').fillna(0)
        
        # Encode all categorical (including customerID)
        encoded_df = input_data.copy()
        for col in encoders:
            if col in encoded_df.columns:
                classes = encoders[col].classes_
                encoded_df[col] = encoded_df[col].apply(lambda x: encoders[col].transform([str(x)])[0] if str(x) in classes else 0)
        
        # Debug
        print("Input columns:", encoded_df.columns.tolist())
        print("Encoded shape:", encoded_df.shape)
        
        pred = model.predict(encoded_df)
        proba = model.predict_proba(encoded_df)[:, 1]
        return pred, proba
    except Exception as e:
        st.error(f"❌ خطا در پیش‌بینی: {e}")
        print(f"Debug: {e}, Columns: {input_df.columns.tolist()}")
        return None, None

# Prediction Section
if menu == "پیش‌بینی":
    st.title("🔍 پیش‌بینی ریزش مشتری")
    usage_count = check_usage(st.session_state.username)
    remaining = 3 - usage_count
    st.info(f"تعداد پیش‌بینی باقی‌مانده: {remaining}")
    if remaining <= 0:
        st.error("❌ حد مجاز تمام شده. با ادمین تماس بگیرید.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("جنسیت", ["Female", "Male"])
        SeniorCitizen = st.selectbox("سالمند؟", [0, 1])
        Partner = st.selectbox("شریک؟", ["No", "Yes"])
        Dependents = st.selectbox("وابستگان؟", ["No", "Yes"])
        tenure = st.slider("مدت اشتراک (ماه)", 0, 72, 1)
        MonthlyCharges = st.number_input("هزینه ماهانه", min_value=0.0, value=20.0)
        TotalCharges = st.number_input("هزینه کل", min_value=0.0, value=20.0)

    with col2:
        PhoneService = st.selectbox("خدمات تلفن", ["No", "Yes"])
        MultipleLines = st.selectbox("چند خط", ["No phone service", "No", "Yes"])
        InternetService = st.selectbox("خدمات اینترنت", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("امنیت آنلاین", ["No internet service", "No", "Yes"])
        OnlineBackup = st.selectbox("بک‌آپ آنلاین", ["No internet service", "No", "Yes"])
        DeviceProtection = st.selectbox("حفاظت دستگاه", ["No internet service", "No", "Yes"])
        TechSupport = st.selectbox("پشتیبانی فنی", ["No internet service", "No", "Yes"])
        StreamingTV = st.selectbox("استریم TV", ["No internet service", "No", "Yes"])
        StreamingMovies = st.selectbox("استریم فیلم", ["No internet service", "No", "Yes"])

    Contract = st.selectbox("قرارداد", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("صورت‌حساب بدون کاغذ", ["No", "Yes"])
    PaymentMethod = st.selectbox("روش پرداخت", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])

    if st.button("پیش‌بینی"):
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
            st.subheader("نتیجه پیش‌بینی")
            if pred[0] == 1:
                st.error("🚨 ریزش می‌کند!")
            else:
                st.success("✅ ریزش نمی‌کند!")
            
            # Feature Importance
            try:
                if hasattr(model, 'named_estimators_'):
                    print("Available estimators:", list(model.named_estimators_.keys()))
                
                importances = None
                feature_names = input_df.columns.tolist()
                
                for name in ['gb', 'gbc', 'gradientboosting']:
                    if name in model.named_estimators_:
                        est = model.named_estimators_[name]
                        if hasattr(est, 'feature_importances_'):
                            importances = est.feature_importances_
                            print(f"Importances from {name}: {importances[:3]}...")
                            break
                
                if importances is None and hasattr(model.named_estimators_.get('lr', None), 'coef_'):
                    importances = np.abs(model.named_estimators_['lr'].coef_[0])
                
                if importances is not None and len(importances) == len(input_df.columns):
                    imp_copy = importances.copy()
                    if 'customerID' in input_df.columns:
                        cid_idx = input_df.columns.get_loc('customerID')
                        imp_copy[cid_idx] = 0
                    
                    feat_imp = pd.DataFrame({
                        "Feature": input_df.columns,
                        "Importance": imp_copy
                    }).sort_values("Importance", ascending=False).head(10)
                    
                    fig = px.bar(feat_imp, x="Importance", y="Feature", title="ویژگی‌های مهم")
                    st.plotly_chart(fig)
                else:
                    st.warning("⚠️ عدم تطابق ویژگی‌ها.")
            except Exception as e:
                st.info(f"ℹ️ اهمیت ویژگی‌ها: {e}")
            
            # Increment usage
            increment_usage(st.session_state.username)
            new_remaining = 3 - check_usage(st.session_state.username)
            if new_remaining <= 0:
                st.warning("این آخرین پیش‌بینی شما بود. خروج...")
                if st.button("خروج فوری"):
                    st.session_state.clear()
                    st.rerun()

# Other Sections
elif menu == "درباره":
    st.title("درباره 📖")
    st.markdown("""
    ### هدف پروژه
    پیش‌بینی ریزش مشتری با استفاده از Voting Classifier (GBC + LR + ABC).
    
    ### معیارها
    دقت: ۸۵٪، Precision: ۸۲٪، Recall: ۸۰٪، F1: ۸۱٪
    """)
    metrics = pd.DataFrame({"معیار": ["دقت", "Precision", "Recall", "F1"], "مقدار": [0.85, 0.82, 0.80, 0.81]})
    fig = px.bar(metrics, x="معیار", y="مقدار", title="معیارهای مدل")
    st.plotly_chart(fig)

elif menu == "تحلیل‌ها":
    st.title("تحلیل‌ها 📈")
    feature_importance = pd.DataFrame({
        "ویژگی": ["Contract", "MonthlyCharges", "tenure", "TechSupport", "OnlineSecurity", "PaymentMethod"],
        "اهمیت": [0.35, 0.25, 0.20, 0.15, 0.10, 0.08]
    })
    fig = px.bar(feature_importance, x="اهمیت", y="ویژگی", title="تاثیر ویژگی‌ها")
    st.plotly_chart(fig)

elif menu == "آپلود":
    st.title("پیش‌بینی دسته‌ای 📂")
    uploaded_file = st.file_uploader("فایل CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head())
        required = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
                    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            st.error(f"ستون‌های گم‌شده: {missing}")
        else:
            if st.button("پیش‌بینی دسته‌ای"):
                pred, proba = predict(data, model, encoders)
                if pred is not None:
                    result = data.copy()
                    result["پیش‌بینی"] = pred
                    result["احتمال ریزش"] = proba
                    result["برچسب"] = result["پیش‌بینی"].map({1: "ریزش", 0: "وفادار"})
                    cols_to_show = ["customerID", "برچسب", "احتمال ریزش"] if "customerID" in result else ["برچسب", "احتمال ریزش"]
                    st.dataframe(result[cols_to_show])
                    fig = px.histogram(result, x="برچسب", title="توزیع پیش‌بینی‌ها", color="برچسب", 
                                       color_discrete_map={"ریزش": "red", "وفادار": "green"})
                    st.plotly_chart(fig)
                    # Batch as 1 use
                    increment_usage(st.session_state.username)
