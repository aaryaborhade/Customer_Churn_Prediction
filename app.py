import streamlit as st
import pandas as pd
import joblib


# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)


# ==============================
# Load Model and Preprocessing
# ==============================

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# ==============================
# Title
# ==============================

st.title("🏦 Customer Churn Prediction System")

st.write(
    "Enter customer details to predict whether the customer "
    "is likely to leave the bank."
)


# ==============================
# User Inputs
# ==============================

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

num_products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

has_credit_card = st.selectbox(
    "Has Credit Card",
    ["Yes", "No"]
)

is_active_member = st.selectbox(
    "Is Active Member",
    ["Yes", "No"]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)


# ==============================
# Prediction
# ==============================

if st.button("🔍 Predict Churn"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [1 if has_credit_card == "Yes" else 0],
        "IsActiveMember": [
            1 if is_active_member == "Yes" else 0
        ],
        "EstimatedSalary": [estimated_salary]
    })


    # Encode categorical variables
    input_data = pd.get_dummies(
        input_data,
        columns=["Geography", "Gender"],
        drop_first=True
    )


    # Make sure input has same columns as training data
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # Scale input
    input_scaled = scaler.transform(input_data)


    # Prediction
    prediction = model.predict(input_scaled)[0]


    # Probability
    probability = model.predict_proba(input_scaled)[0][1]


    # ==============================
    # Display Result
    # ==============================

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            "🔴 Customer is likely to churn."
        )

    else:

        st.success(
            "🟢 Customer is likely to stay."
        )


    st.write(
        f"**Churn Probability: {probability * 100:.2f}%**"
    )