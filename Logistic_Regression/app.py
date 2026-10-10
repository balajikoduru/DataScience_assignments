import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# -------------------------------
# Load the trained model and scaler
# -------------------------------
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# Title and description
# -------------------------------
st.title("🩺 Diabetes Prediction System")
st.write(
    "Enter the patient's details below to predict whether "
    "the model classifies the patient as Diabetic or Not Diabetic."
)

st.info(
    "This application is a machine-learning demonstration based on "
    "the trained Logistic Regression model from the uploaded project."
)

st.divider()

# -------------------------------
# Input fields
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0.0,
        max_value=20.0,
        value=1.0,
        step=1.0
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )

with col2:
    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=900.0,
        value=80.0,
        step=1.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.001,
        format="%.3f"
    )

    age = st.number_input(
        "Age",
        min_value=1.0,
        max_value=120.0,
        value=30.0,
        step=1.0
    )

# -------------------------------
# Prediction
# -------------------------------
st.divider()

if st.button("🔍 Predict Diabetes", use_container_width=True):

    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    # Apply the same StandardScaler used during training
    scaled_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_data)[0]

    # Get probability of class 1
    probability = model.predict_proba(scaled_data)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Prediction: Diabetic")
    else:
        st.success("✅ Prediction: Not Diabetic")

    st.metric(
        "Predicted Diabetes Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    st.caption(
        "The probability shown is the Logistic Regression model's "
        "estimated probability for class 1."
    )

# -------------------------------
# Model information
# -------------------------------
with st.expander("About the Model"):
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Preprocessing:** StandardScaler")
    st.write("**Target variable:** Outcome")
    st.write(
        "**Input features:** Pregnancies, Glucose, BloodPressure, "
        "SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age"
    )

st.caption(
    "For educational/project demonstration only. This prediction should "
    "not be used as a medical diagnosis."
)
