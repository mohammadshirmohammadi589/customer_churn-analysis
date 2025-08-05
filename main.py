
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# این تابع تلاش می‌کند فایل را در مسیرهای مختلف پیدا کند
def find_and_load_model(filename):
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, filename)
    if os.path.exists(file_path):
        st.success(f"فایل مدل در مسیر {file_path} پیدا شد.")
        return joblib.load(file_path)
    
    possible_paths = [
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(current_dir, 'my_churn_app', filename),
        r'C:\Users\Brooz\my_churn_app\voting_classifier_final_model.pkl'
    ]

    for path in possible_paths:
        if os.path.exists(path):
            st.success(f"فایل مدل در مسیر {path} پیدا شد.")
            return joblib.load(path)
    
    st.error(f"خطا: فایل '{filename}' در هیچ مسیری پیدا نشد.")
    return None

# بارگذاری مدل
model_filename = 'voting_classifier_final_model.pkl'
model = find_and_load_model(model_filename)

if model is not None:
    st.title('پیش‌بینی ریزش مشتری')
    st.write('لطفاً اطلاعات مشتری را برای پیش‌بینی احتمال ریزش وارد کنید.')
    # ... (بقیه کدهای ورودی کاربر)
    st.subheader('اطلاعات کلی مشتری')
    gender = st.selectbox('جنسیت', ['Female', 'Male'])
    SeniorCitizen = st.selectbox('آیا سالمند است؟', [0, 1])
    Partner = st.selectbox('آیا شریک دارد؟', ['No', 'Yes'])
    Dependents = st.selectbox('آیا وابسته دارد؟', ['No', 'Yes'])
    
    st.subheader('خدمات استفاده شده')
    PhoneService = st.selectbox('سرویس تلفن', ['No', 'Yes'])
    MultipleLines = st.selectbox('چند خطی', ['No phone service', 'No', 'Yes'])
    InternetService = st.selectbox('سرویس اینترنت', ['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.selectbox('امنیت آنلاین', ['No internet service', 'No', 'Yes'])
    OnlineBackup = st.selectbox('پشتیبان‌گیری آنلاین', ['No internet service', 'No', 'Yes'])
    DeviceProtection = st.selectbox('محافظت از دستگاه', ['No internet service', 'No', 'Yes'])
    TechSupport = st.selectbox('پشتیبانی فنی', ['No internet service', 'No', 'Yes'])
    StreamingTV = st.selectbox('تلویزیون آنلاین', ['No internet service', 'No', 'Yes'])
    StreamingMovies = st.selectbox('فیلم آنلاین', ['No internet service', 'No', 'Yes'])

    st.subheader('قرارداد و پرداخت')
    Contract = st.selectbox('نوع قرارداد', ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.selectbox('صورتحساب بدون کاغذ', ['No', 'Yes'])
    PaymentMethod = st.selectbox('روش پرداخت', ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'])

    st.subheader('اطلاعات مالی')
    tenure = st.slider('مدت مشتری بودن (ماه)', 0, 72, 1)
    MonthlyCharges = st.number_input('هزینه ماهانه', min_value=0.0, value=20.0)
    TotalCharges = st.number_input('مجموع هزینه‌ها', min_value=0.0, value=20.0)

    # ---
    if st.button('پیش‌بینی ریزش مشتری'):
        
        # ساخت یک DataFrame کامل از ورودی‌های کاربر
        input_data = {
            'customerID': [str(np.random.randint(1000, 9999)) + '-ABCD'],
            'gender': [gender], 'SeniorCitizen': [SeniorCitizen], 'Partner': [Partner],
            'Dependents': [Dependents], 'tenure': [tenure], 'PhoneService': [PhoneService],
            'MultipleLines': [MultipleLines], 'InternetService': [InternetService],
            'OnlineSecurity': [OnlineSecurity], 'OnlineBackup': [OnlineBackup],
            'DeviceProtection': [DeviceProtection], 'TechSupport': [TechSupport],
            'StreamingTV': [StreamingTV], 'StreamingMovies': [StreamingMovies],
            'Contract': [Contract], 'PaperlessBilling': [PaperlessBilling],
            'PaymentMethod': [PaymentMethod], 'MonthlyCharges': [MonthlyCharges],
            'TotalCharges': [TotalCharges]
        }
        input_df = pd.DataFrame(input_data)
        
        # لیست ستون‌های دسته‌بندی‌شده برای Label Encoding
        categorical_features = input_df.select_dtypes(include=['object']).columns

        # اعمال LabelEncoder بر روی هر ستون دسته‌بندی‌شده
        le = LabelEncoder()
        for col in categorical_features:
            input_df[col] = le.fit_transform(input_df[col])

        # انجام پیش‌بینی
        prediction = model.predict(input_df)
        
        if prediction[0] == 1:
            st.error('**احتمال ریزش این مشتری بالا است!**')
            st.write('برای جلوگیری از ریزش، راهکارهای بازاریابی را بررسی کنید.')
        else:
            st.success('**این مشتری احتمالاً وفادار خواهد ماند.**')
            st.write('نیازی به اقدام فوری نیست.')
else:
    st.warning('برنامه نمی‌تواند بدون مدل اجرا شود+. لطفاً از وجود فایل مدل مطمئن شوید+. ')


