# Heart Disease Predictor

https://hd-predictor.streamlit.app/

## Overview

The Heart Disease Predictor is a machine learning project designed to predict the presence of heart disease in patients based on various health metrics. This project utilizes different classification algorithms, with a focus on K-Nearest Neighbors (KNN), to provide accurate predictions and assist in early diagnosis.

### Here you can see a sample of the data in PowerBI

![image](https://github.com/user-attachments/assets/483cf57f-f6ec-46bc-a646-ddd48207068d)

## Technologies Used

- Python
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit
- Imbalanced-learn (SMOTE)
- PowerBI

## Dataset

The dataset used in this project is sourced from the UCI Machine Learning Repository. It contains various attributes related to patient health, such as age, sex, cholesterol levels, blood pressure, and other relevant features.

### Columns in the Dataset

- `age`: Age of the patient
- `sex`: Gender of the patient (1 = male; 0 = female)
- `chest_pain`: Type of chest pain
- `arterial_pressure`: Resting blood pressure
- `total_cholesterol`: Serum cholesterol in mg/dl
- `blood_glucose`: Fasting blood sugar > 120 mg/dl
- `electrocardiogram_on_rest`: Results of the electrocardiogram
- `max_heart_rate`: Maximum heart rate achieved
- `exercise_produced_angina`: Exercise-induced angina (1 = yes; 0 = no)
- `unlevel_ST`: ST depression induced by exercise relative to rest
- `segment_st_in_ecg`: Peak exercise ST segment
- `main_vessels_coloured_by_fluorescence`: Number of major vessels colored by fluoroscopy
- `thalium`: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)
- `disease`: Diagnosis of heart disease (1 = presence; 0 = absence)

### How It Works

1.- Data Preprocessing: The dataset is cleaned and preprocessed to handle missing values and normalize features.

2.- Oversampling: Techniques like SMOTE are used to balance the dataset, ensuring that the model does not become biased towards the majority class.

3.-Model Training: The K-Nearest Neighbors algorithm is employed to train the model on the prepared dataset.

4.- Prediction: Users can input various health metrics to receive a prediction about the likelihood of heart disease.

### Here you can see a sample of the data in PowerBI

![image](https://github.com/user-attachments/assets/483cf57f-f6ec-46bc-a646-ddd48207068d)


