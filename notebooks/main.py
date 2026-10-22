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
CREDENTIALS = st.secrets.get("credentials", {})
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
        st.success("✅ مدل و encoders لود شد.")
        return model, encoders
    else:
        st.error(f"❌ فایل‌ها پیدا نشد: مدل={os.path.exists(model_path)}, encoders={os.path.exists(encoders_path)}")
        st.info("فایل‌های .pkl رو از نوت‌بوک دانلود و آپلود کن.")
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
        st.error(f"❌ خطا در پیش‌بینی: {e}")
        st.write(f"Debug: Input columns: {input_df.columns.tolist()}")
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
            churn_prob = proba[0] * 100
            st.metric("احتمال ریزش", f"{churn_prob:.1f}%")
            if pred[0] == 1:
                st.error("🚨 ریزش می‌کند!")
            else:
                st.success("✅ ریزش نمی‌کند!")
            
            # Feature Importance (اصلاح‌شده)
            try:
                importances = None
                tree_names = ['gbc', 'abc']  # فقط tree-based
                if hasattr(model, 'named_estimators_'):
                    for name in tree_names:
                        est = model.named_estimators_.get(name)
                        if est and hasattr(est, 'feature_importances_'):
                            importances = est.feature_importances_
                            st.write(f"از estimator {name} استفاده شد.")
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
                        fig = px.bar(feat_imp, x="Importance", y="Feature", title="ویژگی‌های مهم")
                        st.plotly_chart(fig)
                    else:
                        st.warning("⚠️ عدم تطابق ویژگی‌ها.")
                else:
                    st.info("ℹ️ اهمیت ویژگی‌ها در دسترس نیست.")
            except Exception as e:
                st.info(f"ℹ️ خطا در اهمیت: {e}")
            
            # Increment usage
            increment_usage(st.session_state.username)
            new_remaining = 3 - check_usage(st.session_state.username)
            st.info(f"باقی‌مانده: {new_remaining}")
            if new_remaining <= 0:
                st.warning("آخرین پیش‌بینی بود. خروج...")
                if st.button("خروج"):
                    st.session_state.clear()
                    st.rerun()

# Other Sections
elif menu == "درباره":
    st.title("درباره")
    st.markdown("پیش‌بینی ریزش با Voting Classifier (GBC + LR + ABC). دقت: ~80% Recall Weighted.")

elif menu == "تحلیل‌ها":
    st.title("تحلیل‌ها")
    st.info("نمودارهای نمونه – بعداً اضافه کن.")

elif menu == "آپلود":
    st.title("آپلود")
    uploaded_file = st.file_uploader("CSV آپلود کن", type="csv")
    if uploaded_file:
        df_up = pd.read_csv(uploaded_file)
        st.write("داده‌ها:", df_up.head())
        # predict batch اگر خواستی اضافه کن