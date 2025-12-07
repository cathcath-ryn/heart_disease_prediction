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
cp = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
)
cp = int(cp[-2])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
fbs = int(fbs[-2])
restecg = st.selectbox(
    "Resting ECG Results",
    ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"]
)
restecg = int(restecg[-2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang = int(exang[-2])
oldpeak = st.number_input("Oldpeak", 0.0, 7.0, 1.0)
slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
slope = int(slope[-2])
ca = st.slider("Number of Visible Major Vessels", 0, 3, 0)
thal = st.selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversible defect"])
thal = int(thal[-2])

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    pred = model.predict(input_data)[0]

    if pred == 1:
        st.error("High risk of heart disease.")
    else:
        st.success("Low risk of heart disease.")
