import streamlit as st
import joblib
import numpy as np

# Load the model and the scaler.
model = joblib.load('model_app.pkl')
scaler = joblib.load('scaler.pkl')

# CSS backgroung image
page_bg_img = '''
<style>
.stApp {
background-image: url("https://drive.google.com/file/d/1PGCSkMau2WCdtUXRvYsc-O4hWIK5AsK2/view?usp=drive_link");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# APP Tittle
st.title('Heart Disease Predictor')

# APP Description
st.write('This app predicts if a patient has heart disease.')

# User Inputs
with st.expander("ℹ️ Age Information", expanded=False):
    st.write("The age of the patient. Typically, heart disease risk increases with age.")
age = st.number_input('Age', min_value=1, max_value=120, value=45)

with st.expander("ℹ️ Gender Information", expanded=False):
    st.write("Select the gender of the patient.")
sex = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')

with st.expander("ℹ️ Chest Pain Information", expanded=False):
    st.write("Chest pain is classified into four categories:\n"
             "0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic.")
chest_pain = st.number_input('Intensity of Chest Pain (0-3)', min_value=0, max_value=3, value=1)

with st.expander("ℹ️ Arterial Pressure Information", expanded=False):
    st.write("Mean arterial pressure is the average blood pressure in an individual's arteries during one cardiac cycle.")
arterial_pressure = st.number_input('Mean Arterial Pressure', min_value=80, max_value=200, value=120)

with st.expander("ℹ️ Cholesterol Information", expanded=False):
    st.write("Total cholesterol levels in mg/dL. High levels of cholesterol can increase the risk of heart disease.")
total_cholesterol = st.number_input('Total Cholesterol', min_value=100, max_value=600, value=200)

with st.expander("ℹ️ Blood Glucose Information", expanded=False):
    st.write("Indicates whether the patient's fasting blood sugar is higher than 120 mg/dL (0 = No, 1 = Yes).")
blood_glucose = st.number_input('Fasting blood sugar', min_value=0, max_value=1, value=0)

with st.expander("ℹ️ ECG Information", expanded=False):
    st.write("Electrocardiogram readings in rest. Possible values are 0 (normal), 1 (having ST-T wave abnormality), 2 (left ventricular hypertrophy).")
electrocardiogram_on_rest = st.number_input('Electrocardiogram on Rest', min_value=0, max_value=2, value=1)

with st.expander("ℹ️ Heart Rate Information", expanded=False):
    st.write("Maximum heart rate achieved during exercise.")
max_heart_rate = st.number_input('Maximun Heart Rate', min_value=60, max_value=220, value=150)

with st.expander("ℹ️ Angina Information", expanded=False):
    st.write("Indicates if exercise-induced angina is present (0 = No, 1 = Yes).")
exercise_produced_angina = st.number_input('Excercise produced Angina', min_value=0, max_value=1, value=0)

with st.expander("ℹ️ ST Depression Information", expanded=False):
    st.write("ST depression induced by exercise relative to rest in mm. Higher values may indicate coronary insufficiency.")
unlevel_ST = st.number_input('Unlevel ST', min_value=0.0, max_value=10.0, value=1.0)

with st.expander("ℹ️ Segment ST Information", expanded=False):
    st.write("ST segment values from the ECG. 0 = normal, 1 = mild abnormality, 2 = severe abnormality.")
segment_st_in_ecg = st.number_input('Segment ST in ECG', min_value=0, max_value=2, value=1)

with st.expander("ℹ️ Vessels Information", expanded=False):
    st.write("Number of major vessels (0-3) colored by fluoroscopy. The higher the number, the greater the risk of heart disease.")
main_vessels_coloured_by_fluorescence = st.number_input('Main vessels coloured by Fluorescence', min_value=0, max_value=3, value=0)

with st.expander("ℹ️ Thallium Scan Information", expanded=False):
    st.write("Thallium scan score for myocardial perfusion imaging (MPI). 1 = normal, 2 = moderate defect, 3 = severe defect.")
thalium = st.number_input('Thallium myocardial perfusion imaging (MPI)', min_value=1, max_value=3, value=2)

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
        st.write('The patient MAY develop heart disease')
    else:
        st.write('The patient WILL NOT develop heart disease')
