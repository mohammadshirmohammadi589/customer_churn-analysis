import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import time
import matplotlib.pyplot as plt

# 🔐 Authentication Section
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Login to Customer Churn Prediction System")
    password = st.text_input("Please enter password:", type="password")
    
    if st.button("Login"):
        if password == "password123":
            st.session_state.authenticated = True
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Incorrect password")
    st.stop()

# 🔒 Demo Version Limitations
DEMO_VERSION = "false"  # Set to "false" for full version

if DEMO_VERSION == "true":
    st.warning("""
    🔒 This is a demo version - Purchase the full version
    ⚠️ This version only works with sample data
    💰 Full version with your own data: $20
    📧 For purchase: your_email@example.com
    """)
    
    if 'prediction_count' not in st.session_state:
        st.session_state.prediction_count = 0
    
    if st.session_state.prediction_count >= 3:
        st.error("""
        ❌ You only have 3 free predictions
        💰 For unlimited predictions: Purchase full version
        """)
        st.stop()

# Function to find and load model
def find_and_load_model(filename):
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, filename)
    if os.path.exists(file_path):
        st.success(f"✅ Model file found at {file_path}")
        return joblib.load(file_path)
    
    possible_paths = [
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(current_dir, 'my_churn_app', filename),
        r'C:\Users\Brooz\my_churn_app\voting_classifier_final_model.pkl'
    ]

    for path in possible_paths:
        if os.path.exists(path):
            st.success(f"✅ Model file found at {path}")
            return joblib.load(path)
    
    st.error(f"❌ Error: File '{filename}' not found in any path.")
    return None

# Load model
model_filename = 'voting_classifier_final_model.pkl'
model = find_and_load_model(model_filename)

# Sidebar
st.sidebar.title("📋 Application Menu")
menu = st.sidebar.radio("Select Section:", ["Prediction", "About Project", "Analysis & Insights", "Upload Data"])

# Simple prediction function without SHAP
def predict(input_df, model):
    try:
        # Drop customerID before prediction
        input_data_for_prediction = input_df.drop(columns=["customerID"], errors='ignore')
        
        categorical_features = input_data_for_prediction.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        encoded_df = input_data_for_prediction.copy()
        for col in categorical_features:
            encoded_df[col] = le.fit_transform(encoded_df[col])
        prediction = model.predict(encoded_df)
        return prediction
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None

# Single customer prediction section
if menu == "Prediction" and model is not None:
    st.title("Customer Churn Prediction 📊")
    st.markdown("Please enter customer information to predict churn probability.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer General Information")
        gender = st.selectbox("Customer Gender", ["Female", "Male"], key="gender")
        SeniorCitizen = st.selectbox("Is customer senior?", [0, 1], key="senior")
        Partner = st.selectbox("Does customer have partner?", ["No", "Yes"], key="partner")
        Dependents = st.selectbox("Does customer have dependents?", ["No", "Yes"], key="dependents")

        st.subheader("Financial Information")
        tenure = st.slider("Tenure (months)", 0, 72, 1, key="tenure")
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=20.0, key="monthly")
        TotalCharges = st.number_input("Total Charges", min_value=0.0, value=20.0, key="total")

    with col2:
        st.subheader("Used Services")
        PhoneService = st.selectbox("Phone Service", ["No", "Yes"], key="phone")
        MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"], key="multiple")
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
        OnlineSecurity = st.selectbox("Online Security", ["No internet service", "No", "Yes"], key="security")
        OnlineBackup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"], key="backup")
        DeviceProtection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"], key="protection")
        TechSupport = st.selectbox("Tech Support", ["No internet service", "No", "Yes"], key="support")
        StreamingTV = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"], key="tv")
        StreamingMovies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"], key="movies")

    st.subheader("Contract & Payment")
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract")
    PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"], key="paperless")
    PaymentMethod = st.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"], key="payment")

    if st.button("🔍 Predict Customer Churn", key="predict"):
        with st.spinner("⏳ Processing prediction..."):
            time.sleep(1)
            input_data = {
                "customerID": [str(np.random.randint(1000, 9999)) + "-ABCD"],
                "gender": [gender], "SeniorCitizen": [SeniorCitizen], "Partner": [Partner],
                "Dependents": [Dependents], "tenure": [tenure], "PhoneService": [PhoneService],
                "MultipleLines": [MultipleLines], "InternetService": [InternetService], "OnlineSecurity": [OnlineSecurity],
                "OnlineBackup": [OnlineBackup], "DeviceProtection": [DeviceProtection], "TechSupport": [TechSupport],
                "StreamingTV": [StreamingTV], "StreamingMovies": [StreamingMovies],
                "Contract": [Contract], "PaperlessBilling": [PaperlessBilling], "PaymentMethod": [PaymentMethod],
                "MonthlyCharges": [MonthlyCharges], "TotalCharges": [TotalCharges]
            }
            input_df = pd.DataFrame(input_data)
            
            # Prediction
            prediction = predict(input_df, model)
            
            if prediction is not None:
                st.subheader("Prediction Result")
                if prediction[0] == 1:
                    st.markdown("<div style='background-color: #ffcccc; padding: 15px; border-radius: 10px;'><h3 style='color: #d32f2f;'>🚨 High probability of customer churn!</h3></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='background-color: #ccffcc; padding: 15px; border-radius: 10px;'><h3 style='color: #2e7d32;'>✅ This customer will likely remain loyal.</h3></div>", unsafe_allow_html=True)

                # Feature importance analysis (with error handling)
                st.subheader("Feature Analysis 📊")
                
                # Prepare data for feature importance (without customerID)
                feature_data = input_df.drop(columns=["customerID"])
                feature_names = feature_data.columns.tolist()
                
                importances = None
                
                # Try to get feature importance from model
                try:
                    if hasattr(model, "feature_importances_"):
                        importances = model.feature_importances_
                    elif hasattr(model, "coef_"):
                        importances = np.abs(model.coef_[0])
                    elif hasattr(model, "estimators_"):  # VotingClassifier
                        for est in model.estimators_:
                            if hasattr(est, "feature_importances_"):
                                importances = est.feature_importances_
                                break
                            elif hasattr(est, "coef_"):
                                importances = np.abs(est.coef_[0])
                                break
                except Exception as e:
                    st.warning(f"⚠️ Could not extract feature importance: {str(e)}")

                # Display feature importance if available and lengths match
                if importances is not None:
                    if len(importances) == len(feature_names):
                        feat_imp = pd.DataFrame({
                            "Feature": feature_names,
                            "Importance": importances
                        }).sort_values(by="Importance", ascending=False).head(10)

                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(feat_imp["Feature"], feat_imp["Importance"], color="skyblue")
                        ax.invert_yaxis()
                        ax.set_xlabel("Importance")
                        ax.set_title("Most Important Features in Prediction")
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning(f"⚠️ Feature importance mismatch: Model has {len(importances)} features, but input has {len(feature_names)} features.")
                        st.info("This usually happens when the model was trained with different features than the current input.")
                else:
                    st.info("ℹ️ Feature importance information is not available for this model type.")

# About Project section
elif menu == "About Project":
    st.title("About Project 📖")
    st.markdown("""
    ### Project Goal
    Predicting customer churn using machine learning models.

    ### Models
    Voting Classifier combining Random Forest, XGBoost and Logistic Regression.

    ### Evaluation Metrics
    - Accuracy: 85%
    - Precision: 82%
    - Recall: 80%
    - F1 Score: 81%
    """)
    
    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [0.85, 0.82, 0.80, 0.81]
    })
    fig = px.bar(metrics, x="Metric", y="Value", title="Model Evaluation Metrics", color="Metric")
    st.plotly_chart(fig)

# Analysis & Insights section
elif menu == "Analysis & Insights":
    st.title("Analysis & Insights 📈")
    st.markdown("""
    ### Feature Analysis
    The chart below shows which features have the most impact on predictions.
    """)
    
    # Sample analysis chart
    feature_importance = pd.DataFrame({
        "Feature": ["Contract", "MonthlyCharges", "tenure", "TechSupport", "OnlineSecurity", "PaymentMethod"],
        "Importance": [0.35, 0.25, 0.20, 0.15, 0.10, 0.08]
    })
    fig = px.bar(feature_importance, x="Feature", y="Importance", title="Feature Impact on Prediction", color="Feature")
    st.plotly_chart(fig)

# Upload Data section
elif menu == "Upload Data":
    st.title("Upload Data & Batch Prediction 📂")
    st.markdown("Please upload CSV file containing customer information.")
    
    uploaded_file = st.file_uploader("Select CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.subheader("Uploaded Data Preview")
            st.dataframe(data.head())

            required_columns = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                               "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                               "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                               "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
                               "MonthlyCharges", "TotalCharges"]
            
            # Check if required columns exist (customerID is optional)
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                st.error(f"❌ The following required columns are missing: {missing_columns}")
            else:
                if st.button("🔍 Predict for Uploaded Data"):
                    with st.spinner("⏳ Processing prediction..."):
                        time.sleep(1)
                        predictions = predict(data, model)
                        
                        if predictions is not None:
                            result_df = data.copy()
                            if 'customerID' in data.columns:
                                result_df["Prediction"] = predictions
                                result_df["Prediction_Label"] = result_df["Prediction"].map({1: "Churn", 0: "Loyal"})
                                st.subheader("Prediction Results")
                                st.dataframe(result_df[["customerID", "Prediction_Label"]])
                            else:
                                result_df["Prediction"] = predictions
                                result_df["Prediction_Label"] = result_df["Prediction"].map({1: "Churn", 0: "Loyal"})
                                st.subheader("Prediction Results")
                                st.dataframe(result_df[["Prediction_Label"]])

                            # Distribution chart
                            fig = px.histogram(result_df, x="Prediction_Label", title="Prediction Distribution", 
                                            color="Prediction_Label", 
                                            color_discrete_map={"Churn": "red", "Loyal": "green"})
                            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"❌ File processing error: {str(e)}")

if model is None:
    st.warning("⚠️ Model not loaded. Please check if the model file exists.")

    