import streamlit as st
import pickle
import numpy as np
import joblib

model = joblib.load("RF_best_Model.pkl")

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.write("This app uses a trained Random Forest model to predict whether a person has diabetes based on medical data.")

# Input
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose", min_value=40, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=40, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=7, max_value=100, value=20)
insulin = st.number_input("Insulin (mu U/ml)", min_value=2, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)

# Prediction
if st.button("Predict"):
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, dpf, age]])
    
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The model predicts this person **has diabetes** (probability: {proba:.2f})")
    else:
        st.success(f"✅ The model predicts this person **does not have diabetes** (probability: {proba:.2f})")
