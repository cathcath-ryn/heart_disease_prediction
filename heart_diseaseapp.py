import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

logo_path = "parami.jpg"

st.sidebar.markdown("Student Info")

if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)


st.sidebar.markdown("Name: **Ei Phyu Sin Win**")
st.sidebar.markdown("**Student ID: PIUS20230033**")
st.sidebar.markdown("Class: 2027")
st.sidebar.markdown("Intro to Machine Learning")
st.title("Heart Disease Prediction App")
model = joblib.load("model.pkl")

age = st.number_input("Age", 1, 120, 50)
sex = st.radio("Sex", ["Female", "Male"])
sex = 1 if sex == "Male" else 0
cp_options = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}
cp_label = st.radio("Chest Pain Type", list(cp_options.keys()))
cp = cp_options[cp_label]

trestbps = st.number_input("Resting Blood Pressure (in mmHg)", 80, 200, 120)

chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

fbs_options = {"No": 0, "Yes": 1}
fbs_label = st.radio("Fasting Blood Sugar > 120 mg/dl", list(fbs_options.keys()))
fbs = fbs_options[fbs_label]

restecg_options = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2
}
restecg_label = st.radio("Resting ECG Results", list(restecg_options.keys()))
restecg = restecg_options[restecg_label]

thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)

exang_options = {"No": 0, "Yes": 1}
exang_label = st.radio("Exercise Induced Angina", list(exang_options.keys()))
exang = exang_options[exang_label]

oldpeak = st.number_input("Oldpeak", 0.0, 7.0, 1.0)

slope_options = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope_label = st.radio("ST Slope", list(slope_options.keys()))
slope = slope_options[slope_label]

ca = st.slider("Number of Visible Major Vessels", 0, 3, 0)

thal_options = {
    "Normal": 1,
    "Fixed defect": 2,
    "Reversible defect": 3
}
thal_label = st.radio("Thalassemia", list(thal_options.keys()))
thal = thal_options[thal_label]

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    pred = model.predict(input_data)[0]

    if pred == 1:
        st.error("High risk of heart disease.")
    else:
        st.success("Low risk of heart disease.")
