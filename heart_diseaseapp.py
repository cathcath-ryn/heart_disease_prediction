import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Heart Disease Prediction App")
model = joblib.load("model.pkl")

age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex == "Male" else 0
cp = st.selectbox(
    "Chest Pain Type",
    {"Typical Angina": 0, "Atypical": 1, "Non-anginal": 2, "Asymptomatic": 3}
)
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"])
restecg = st.selectbox(
    "Resting ECG Results",
    ["Normal (0)", "ST-T wave abnormality (1)", "Left ventricular hypertrophy (2)"]
)
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
oldpeak = st.number_input("Oldpeak", 0.0, 7.0, 1.0)
slope = st.selectbox("ST Slope", ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
ca = st.number_input("Number of Major Vessels (0–4)", 0, 4, 0)
thal = st.selectbox("Thalassemia", ["Normal (1)", "Fixed defect (2)", "Reversible defect (3)"])

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    pred = model.predict(input_data)[0]

    if pred == 1:
        st.error("High risk of heart disease.")
    else:
        st.success("Low risk of heart disease.")
