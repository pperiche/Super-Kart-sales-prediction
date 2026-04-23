
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load model
# -----------------------------
model = joblib.load("SuperKart/models/GradientBoosting.pkl")  # or RandomForest.pkl

st.title("SuperKart Sales Prediction App")

st.write("Enter product and store details to predict sales.")

# -----------------------------
# User Inputs
# -----------------------------

product_weight = st.number_input("Product Weight", min_value=0.0)
product_mrp = st.number_input("Product MRP", min_value=0.0)
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0)

store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
store_city = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])

product_type = st.selectbox("Product Type", [
    "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household"
])

sugar_content = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])

store_year = st.number_input("Store Establishment Year", min_value=1980, max_value=2025)

# -----------------------------
# Preprocessing
# -----------------------------

def preprocess_input():
    data = {
        "Product_Weight": product_weight,
        "Product_Allocated_Area": np.log1p(product_allocated_area),
        "Product_MRP": product_mrp,
        "Store_Establishment_Year": store_year,
        "Store_Size": {"Small":0,"Medium":1,"High":2}[store_size],
        "Store_Location_City_Type": {"Tier 3":0,"Tier 2":1,"Tier 1":2}[store_city],
    }

    df = pd.DataFrame([data])

    # One-hot encoding (must match training)
    df = pd.get_dummies(df)

    return df

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Sales"):

    input_df = preprocess_input()

    # Align columns with training data
    model_columns = model.feature_names_in_

    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]

    # Predict (log scale)
    prediction_log = model.predict(input_df)[0]

    # Convert back to actual
    prediction = np.expm1(prediction_log)

    st.success(f"Predicted Sales: {prediction:.2f}")
