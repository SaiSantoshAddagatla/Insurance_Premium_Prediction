import streamlit as st
import pickle
import numpy as np

# Loading scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('insurance_best_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Insurance Premium Estimator")

age = st.number_input("Age", min_value=1, max_value=100, value=30)
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
surgeries = st.number_input("Number of Major Surgeries", min_value=0, max_value=10, value=0)

# Calculating BMI
height_m = height / 100
bmi = weight / (height_m ** 2)

# Binary inputs
diabetes = st.selectbox("Do you have Diabetes?", ["No", "Yes"])
bp_problems = st.selectbox("Blood Pressure Problems?", ["No", "Yes"])
transplants = st.selectbox("Any Transplants?", ["No", "Yes"])
chronic_diseases = st.selectbox("Any Chronic Diseases?", ["No", "Yes"])
allergies = st.selectbox("Known Allergies?", ["No", "Yes"])
cancer_history = st.selectbox("Family History of Cancer?", ["No", "Yes"])

if st.button("Estimate Premium"):
    # Step 1: Preparing input
    num_features = np.array([[age, height, weight, bmi, surgeries]])
    cat_features = np.array([[
        1 if diabetes == "Yes" else 0,
        1 if bp_problems == "Yes" else 0,
        1 if transplants == "Yes" else 0,
        1 if chronic_diseases == "Yes" else 0,
        1 if allergies == "Yes" else 0,
        1 if cancer_history == "Yes" else 0
    ]])

    # Step 2: Scaling numerical features
    num_scaled = scaler.transform(num_features)

    # Step 3: Combining all features
    final_input = np.concatenate([num_scaled, cat_features], axis=1)

    # Step 4: Prediction
    prediction = model.predict(final_input)[0]
    st.success(f"Estimated Premium: ₹{prediction:,.2f}")
