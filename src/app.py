import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import shap
import io
import time
import matplotlib.pyplot as plt

# تنظیمات صفحه
st.set_page_config(page_title="پیش‌بینی ریزش مشتری", layout="wide", page_icon="📊")

# تابع برای پیدا کردن و بارگذاری مدل
def find_and_load_model(filename):
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, filename)
    if os.path.exists(file_path):
        st.success(f"✅ فایل مدل در مسیر {file_path} پیدا شد.")
        return joblib.load(file_path)
    
    possible_paths = [
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(current_dir, 'my_churn_app', filename),
        r'C:\Users\Brooz\my_churn_app\voting_classifier_final_model.pkl'
    ]

    for path in possible_paths:
        if os.path.exists(path):
            st.success(f"✅ فایل مدل در مسیر {path} پیدا شد.")
            return joblib.load(path)
    
    st.error(f"❌ خطا: فایل '{filename}' در هیچ مسیری پیدا نشد.")
    return None

# بارگذاری مدل
model_filename = 'voting_classifier_final_model.pkl'
model = find_and_load_model(model_filename)

# نوار کناری
st.sidebar.title("📋 منوی برنامه")
menu = st.sidebar.radio("انتخاب بخش:", ["پیش‌بینی", "درباره پروژه", "تحلیل و بینش", "آپلود داده"])

# تابع برای پیش‌بینی و تحلیل SHAP
def predict_and_explain(input_df, model):
    try:
        # آماده‌سازی داده‌ها
        categorical_features = input_df.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        encoded_df = input_df.copy()
        for col in categorical_features:
            encoded_df[col] = le.fit_transform(encoded_df[col])
        
        # پیش‌بینی
        prediction = model.predict(encoded_df)
        
        # استفاده از KernelExplainer به جای TreeExplainer
        # نمونه کوچک از داده‌ها برای سرعت بخشیدن به محاسبات SHAP
        background_data = encoded_df.sample(min(100, len(encoded_df)), random_state=42) if len(encoded_df) > 100 else encoded_df
        explainer = shap.KernelExplainer(model.predict_proba, background_data)
        shap_values = explainer.shap_values(encoded_df)
        
        return prediction, shap_values, explainer
    except Exception as e:
        st.error(f"❌ خطا در تحلیل SHAP: {str(e)}")
        return prediction, None, None

# بخش پیش‌بینی
if menu == "پیش‌بینی" and model is not None:
    st.title("پیش‌بینی ریزش مشتری 📊")
    st.markdown("لطفاً اطلاعات مشتری را وارد کنید تا احتمال ریزش او پیش‌بینی شود.")

    # تقسیم‌بندی صفحه به دو ستون
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("اطلاعات کلی مشتری")
        gender = st.selectbox("جنسیت مشتری", ["Female", "Male"], key="gender", help="جنسیت مشتری را انتخاب کنید.")
        SeniorCitizen = st.selectbox("آیا مشتری سالمند است؟", [0, 1], key="senior", help="0 برای خیر، 1 برای بله.")
        Partner = st.selectbox("آیا مشتری شریک دارد؟", ["No", "Yes"], key="partner", help="وضعیت تأهل یا شریک.")
        Dependents = st.selectbox("آیا مشتری وابسته دارد؟", ["No", "Yes"], key="dependents", help="وابستگان مانند فرزندان.")

        st.subheader("اطلاعات مالی")
        tenure = st.slider("مدت مشتری بودن (ماه)", 0, 72, 1, key="tenure", help="تعداد ماه‌هایی که مشتری از خدمات استفاده کرده.")
        MonthlyCharges = st.number_input("هزینه ماهانه (تومان)", min_value=0.0, value=20.0, key="monthly", help="هزینه ماهانه خدمات.")
        TotalCharges = st.number_input("مجموع هزینه‌ها (تومان)", min_value=0.0, value=20.0, key="total", help="مجموع هزینه‌های پرداختی.")

    with col2:
        st.subheader("خدمات استفاده‌شده")
        PhoneService = st.selectbox("سرویس تلفن", ["No", "Yes"], key="phone", help="آیا مشتری از سرویس تلفن استفاده می‌کند؟")
        MultipleLines = st.selectbox("چند خطی", ["No phone service", "No", "Yes"], key="multiple", help="وضعیت خطوط تلفن.")
        InternetService = st.selectbox("سرویس اینترنت", ["DSL", "Fiber optic", "No"], key="internet", help="نوع سرویس اینترنت.")
        OnlineSecurity = st.selectbox("امنیت آنلاین", ["No internet service", "No", "Yes"], key="security", help="آیا امنیت آنلاین فعال است؟")
        OnlineBackup = st.selectbox("پشتیبان‌گیری آنلاین", ["No internet service", "No", "Yes"], key="backup", help="آیا پشتیبان‌گیری آنلاین فعال است؟")
        DeviceProtection = st.selectbox("محافظت از دستگاه", ["No internet service", "No", "Yes"], key="protection", help="آیا محافظت از دستگاه فعال است؟")
        TechSupport = st.selectbox("پشتیبانی فنی", ["No internet service", "No", "Yes"], key="support", help="آیا پشتیبانی فنی فعال است؟")
        StreamingTV = st.selectbox("تلویزیون آنلاین", ["No internet service", "No", "Yes"], key="tv", help="آیا تلویزیون آنلاین فعال است؟")
        StreamingMovies = st.selectbox("فیلم آنلاین", ["No internet service", "No", "Yes"], key="movies", help="آیا فیلم آنلاین فعال است؟")

    st.subheader("قرارداد و پرداخت")
    Contract = st.selectbox("نوع قرارداد", ["Month-to-month", "One year", "Two year"], key="contract", help="مدت قرارداد مشتری.")
    PaperlessBilling = st.selectbox("صورتحساب بدون کاغذ", ["No", "Yes"], key="paperless", help="آیا صورتحساب الکترونیکی است؟")
    PaymentMethod = st.selectbox("روش پرداخت", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"], key="payment", help="روش پرداخت مشتری.")

    if st.button("🔍 پیش‌بینی ریزش مشتری", key="predict"):
        with st.spinner("⏳ در حال پردازش پیش‌بینی..."):
            time.sleep(1)  # شبیه‌سازی زمان پردازش
            input_data = {
                "customerID": [str(np.random.randint(1000, 9999)) + "-ABCD"],
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
            
            prediction, shap_values, explainer = predict_and_explain(input_df, model)
            
            # نمایش نتیجه در باکس برجسته
            st.subheader("نتیجه پیش‌بینی")
            if prediction[0] == 1:
                st.markdown("""
                <div style='background-color: #ffcccc; padding: 15px; border-radius: 10px;'>
                    <h3 style='color: #d32f2f;'>🚨 احتمال ریزش این مشتری بالا است!</h3>
                    <p><b>توصیه:</b> برای جلوگیری از ریزش، پیشنهاد قرارداد بلندمدت یا تخفیف ویژه ارائه دهید.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background-color: #ccffcc; padding: 15px; border-radius: 10px;'>
                    <h3 style='color: #2e7d32;'>✅ این مشتری احتمالاً وفادار خواهد ماند.</h3>
                    <p><b>توصیه:</b> نیازی به اقدام فوری نیست، اما ارتباط با مشتری را حفظ کنید.</p>
                </div>
                """, unsafe_allow_html=True)
            # نمایش نمودار SHAP
            if shap_values is not None:
                st.subheader("تحلیل ویژگی‌های تأثیرگذار")

                fig, ax = plt.subplots(figsize=(10, 6))  # ایجاد figure
                if isinstance(shap_values, list):  # اگر چندکلاسه باشه
                    if len(shap_values) > 1:
                        shap.summary_plot(shap_values[1], input_df, show=False, plot_type="bar")
                    else:
                        shap.summary_plot(shap_values[0], input_df, show=False, plot_type="bar")
                else:  # تک‌کلاسه
                    shap.summary_plot(shap_values, input_df, show=False, plot_type="bar")

                st.pyplot(fig)  # ارسال figure به استریملت
            else:
                st.warning("⚠️ تحلیل SHAP به دلیل خطا نمایش داده نمی‌شود.")



# بخش درباره پروژه
elif menu == "درباره پروژه":
    st.title("درباره پروژه 📖")
    st.markdown("""
    ### هدف پروژه
    این پروژه با هدف پیش‌بینی ریزش مشتریان (Customer Churn) طراحی شده است. با استفاده از مدل‌های یادگیری ماشین، می‌توانیم مشتریانی که احتمال ترک خدمات دارند را شناسایی کرده و اقدامات پیشگیرانه انجام دهیم.

    ### مدل‌های استفاده‌شده
    از یک مدل Voting Classifier استفاده شده که ترکیبی از چندین مدل یادگیری ماشین (مانند Random Forest، XGBoost و Logistic Regression) است. این مدل با داده‌های مشتریان آموزش دیده و دقت بالایی در پیش‌بینی دارد.

    ### معیارهای ارزیابی
    معیارهای زیر برای ارزیابی مدل استفاده شده‌اند:
    - **دقت (Accuracy):** 85%
    - **دقت (Precision):** 82%
    - **فراخوان (Recall):** 80%
    - **امتیاز F1:** 81%
    """)
    
    # نمودار معیارهای ارزیابی
    metrics = pd.DataFrame({
        "معیار": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "مقدار": [0.85, 0.82, 0.80, 0.81]
    })
    fig = px.bar(metrics, x="معیار", y="مقدار", title="معیارهای ارزیابی مدل", color="معیار")
    st.plotly_chart(fig)

# بخش تحلیل و بینش
elif menu == "تحلیل و بینش":
    st.title("تحلیل و بینش 📈")
    st.markdown("""
    ### تحلیل ویژگی‌ها
    در این بخش، تأثیر ویژگی‌های مختلف بر پیش‌بینی ریزش مشتری بررسی می‌شود. با استفاده از SHAP، می‌توانیم ببینیم کدام ویژگی‌ها بیشترین تأثیر را دارند.
    
    ### بینش‌های عملی
    - **قراردادهای کوتاه‌مدت:** مشتریانی که قرارداد ماه‌به‌ماه دارند، احتمال ریزش بیشتری دارند.
    - **هزینه‌های بالا:** هزینه‌های ماهانه بالا می‌تواند باعث نارضایتی و ریزش شود.
    - **عدم استفاده از خدمات پشتیبانی:** مشتریانی که از پشتیبانی فنی یا امنیت آنلاین استفاده نمی‌کنند، احتمالاً کمتر به خدمات وفادار هستند.
    """)
    
    # نمودار اهمیت ویژگی‌ها
    feature_importance = pd.DataFrame({
        "ویژگی": ["Contract", "MonthlyCharges", "Tenure", "TechSupport"],
        "اهمیت": [0.35, 0.25, 0.20, 0.15]
    })
    fig = px.bar(feature_importance, x="ویژگی", y="اهمیت", title="تأثیر ویژگی‌ها بر پیش‌بینی", color="ویژگی")
    st.plotly_chart(fig)

# بخش آپلود داده
elif menu == "آپلود داده":
    st.title("آپلود داده و پیش‌بینی دسته‌ای 📂")
    st.markdown("لطفاً فایل CSV حاوی اطلاعات مشتریان را آپلود کنید.")
    
    uploaded_file = st.file_uploader("انتخاب فایل CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            # خواندن فایل CSV
            data = pd.read_csv(uploaded_file)
            st.success("✅ فایل با موفقیت آپلود شد!")
            st.subheader("نمایش داده‌های آپلود شده")
            st.dataframe(data.head())

            # اعتبارسنجی ستون‌ها
            required_columns = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                               "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                               "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                               "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
                               "MonthlyCharges", "TotalCharges"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                st.error(f"❌ خطا: ستون‌های زیر در فایل وجود ندارند: {missing_columns}")
            else:
                if st.button("🔍 پیش‌بینی برای داده‌های آپلود شده"):
                    with st.spinner("⏳ در حال پردازش پیش‌بینی..."):
                        time.sleep(1)  # شبیه‌سازی زمان پردازش
                        predictions, shap_values, explainer = predict_and_explain(data, model)
                        data["Prediction"] = predictions
                        data["Prediction"] = data["Prediction"].map({1: "ریزش", 0: "وفادار"})
                        st.subheader("نتایج پیش‌بینی")
                        st.dataframe(data[["customerID", "Prediction"]])

                        # نمایش نمودار توزیع پیش‌بینی‌ها
                        fig = px.histogram(data, x="Prediction", title="توزیع پیش‌بینی‌ها", color="Prediction")
                        st.plotly_chart(fig)

        except Exception as e:
            st.error(f"❌ خطا در پردازش فایل: {str(e)}")

if model is None:
    st.warning("⚠️")
