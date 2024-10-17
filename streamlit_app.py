import streamlit as st
import joblib
import numpy as np

# Load the model and the scaler.
model = joblib.load('modelo_logistic_regresion.pkl')
scaler = joblib.load('scaler.pkl')

# APP Tittle
st.title('Heart Disease Predictor')

# APP Description
st.write('This app predicts if a patient has heart disease.')

# User Inputs
age = st.number_input('Age', min_value=1, max_value=120, value=45)
sex = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
chest_pain = st.number_input('Intensity of Chest Pain (0-3)', min_value=0, max_value=3, value=1)
arterial_pressure = st.number_input('Mean Arterial Pressure', min_value=80, max_value=200, value=120)
total_cholesterol = st.number_input('Total Cholesterol', min_value=100, max_value=600, value=200)
blood_glucose = st.number_input('Fasting blood sugar > 120 mg/dl (No = 0 or Yes = 1)', min_value=0, max_value=1, value=0)
electrocardiogram_on_rest = st.number_input('Electrocardiogram on Rest (0-2)', min_value=0, max_value=2, value=1)
max_heart_rate = st.number_input('Maximun Heart Rate', min_value=60, max_value=220, value=150)
exercise_produced_angina = st.number_input('Excercise produced Angina (No = 0 or Yes = 1)', min_value=0, max_value=1, value=0)
unlevel_ST = st.number_input('Unlevel ST', min_value=0.0, max_value=10.0, value=1.0)
segment_st_in_ecg = st.number_input('Segment ST in ECG (0-2)', min_value=0, max_value=2, value=1)
main_vessels_coloured_by_fluorescence = st.number_input('Main vessels coloured by Fluorescence (0-3)', min_value=0, max_value=3, value=0)
thalium = st.number_input('Thallium myocardial perfusion imaging (MPI) (1-3)', min_value=1, max_value=3, value=2)

# Prediction
if st.button('Predict'):
    # Crear un array con las características ingresadas
    input_data = np.array([[age, sex, chest_pain, arterial_pressure, total_cholesterol, blood_glucose,
                            electrocardiogram_on_rest, max_heart_rate, exercise_produced_angina, unlevel_ST,
                            segment_st_in_ecg, main_vessels_coloured_by_fluorescence, thalium]])

    # Escalar los datos
    input_data_scaled = scaler.transform(input_data)

    # Hacer la predicción
    prediction = model.predict(input_data_scaled)
    
    # Mostrar el resultado
    if prediction[0] == 1:
        st.write('The patient CAN develop heart disease')
    else:
        st.write('The patient WILL NOT develop heart disease')
