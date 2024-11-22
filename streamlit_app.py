# Importing Important Libraries
import pickle
import streamlit as st
import numpy as np

# Load model
model_diabetes = pickle.load(open('model_diabetes_logistic.sav', 'rb'))
st.title('Diabetes Prediction')
col1, col2 = st.columns(2)

with col1:
    pregnancies_option = st.radio("Are You Pregnenat? (Yes/No)", ("Yes", "No"))
    Pregnancies = 1 if pregnancies_option == "Yes" else 0

with col2:
    Glucose = st.number_input('Enter the Glucose value')
# Importing Important Libraries
import pickle
import streamlit as st
import numpy as np

# Load the model
model_diabetes = pickle.load(open('model_diabetes_logistic.sav', 'rb'))

# Title
st.title('🩺 Diabetes Prediction App')
st.markdown(
    """
    This application predicts the likelihood of diabetes based on input parameters.
    Please enter the values below to check the prediction.
    """
)

# Input section
st.header('Enter Patient Details')

# Use expander for grouping sections
with st.expander("Basic Information"):
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies_option = st.radio(
            "Are You Pregnant? (Yes/No)",
            ("Yes", "No"),
            help="Select 'Yes' if the patient is currently pregnant."
        )
        Pregnancies = 1 if pregnancies_option == "Yes" else 0

    with col2:
        Age = st.number_input(
            'Enter Age (in years)',
            min_value=1,
            max_value=120,
            step=1,
            help="Enter the patient's age."
        )

with st.expander("Medical Measurements"):
    col1, col2 = st.columns(2)
    
    with col1:
        Glucose = st.number_input(
            'Glucose Level (mg/dL)',
            min_value=0.0,
            step=0.1,
            help="Enter the glucose concentration level."
        )

        BloodPressure = st.number_input(
            'Blood Pressure (mmHg)',
            min_value=0.0,
            step=0.1,
            help="Enter the diastolic blood pressure reading."
        )

        SkinThickness = st.number_input(
            'Skin Thickness (mm)',
            min_value=0.0,
            step=0.1,
            help="Enter the triceps skinfold thickness."
        )
    
    with col2:
        Insulin = st.number_input(
            'Insulin Level (μU/mL)',
            min_value=0.0,
            step=0.1,
            help="Enter the insulin concentration level."
        )

        BMI = st.number_input(
            'BMI (Body Mass Index)',
            min_value=0.0,
            step=0.1,
            help="Enter the patient's BMI value."
        )

        DiabetesPedigreeFunction = st.number_input(
            'Diabetes Pedigree Function',
            min_value=0.0,
            step=0.01,
            help="Enter the Diabetes Pedigree Function value."
        )

# Add prediction button and result
diabetes_diagnosis = ''

if st.button('🧪 Run Diabetes Prediction Test'):
    try:
        # Run the prediction
        diabetes_prediction = model_diabetes.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if diabetes_prediction[0] == 1:
            diabetes_diagnosis = '⚠️ The patient has diabetes.'
        else:
            diabetes_diagnosis = '✅ The patient does not have diabetes.'
        
        st.success(diabetes_diagnosis)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.caption("🔬 Developed for educational purposes. Please consult a medical professional for accurate diagnosis.")

with col1:
    BloodPressure = st.number_input('Enter the Blood Pressure value')

with col2:
    SkinThickness = st.number_input('Enter the Skin Thickness value')

with col1:
    Insulin = st.number_input('Enter the Insulin value')

with col2:
    BMI = st.number_input('Enter the BMI value')

with col1:
    DiabetesPedigreeFunction = st.number_input('Enter the Diabetes Pedigree Function value')

with col2:
    Age = st.number_input('Enter the Age value')
diabetes_diagnosis = ''

if st.button('Diabetes Prediction Test'):
    diabetes_prediction = model_diabetes.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    if diabetes_prediction[0] == 1:
        diabetes_diagnosis = 'The patient has diabetes'
    else:
        diabetes_diagnosis = 'The patient does not have diabetes'
st.success(diabetes_diagnosis)
