import pandas as pd
import numpy as np

def load_and_standardize_df(url):
    # Load the DataFrame from the URL
    df = pd.read_csv(url)

    # Standardize column names: lowercase and replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Set pandas to display all columns
    pd.set_option('display.max_columns', None)
    
    # Return the DataFrame
    return df


def generate_female_patient_data(num_rows):
    np.random.seed(42)  # For reproducibility

    # Range of values based on observations in the dataset
    age = np.random.randint(30, 71, size=num_rows)
    sex = np.zeros(num_rows, dtype=int)  # All are females
    chest_pain = np.random.randint(0, 4, size=num_rows)
    arterial_pressure = np.random.normal(loc=130, scale=15, size=num_rows).astype(int)
    total_cholesterol = np.random.normal(loc=240, scale=50, size=num_rows).astype(int)
    blood_glucose = np.random.choice([0, 1], size=num_rows)
    electrocardiogram_on_rest = np.random.choice([0, 1, 2], size=num_rows)
    max_heart_rate = np.random.normal(loc=150, scale=25, size=num_rows).astype(int)
    exercise_produced_angina = np.random.choice([0, 1], size=num_rows)
    unlevel_ST = np.random.uniform(0.0, 5.0, size=num_rows)
    segment_st_in_ecg = np.random.choice([0, 1, 2], size=num_rows)
    main_vessels_coloured_by_fluorescence = np.random.choice([0, 1, 2, 3], size=num_rows)
    thalium = np.random.choice([0, 1, 2, 3], size=num_rows)
    disease = np.random.choice([0, 1], size=num_rows)

    # Create DataFrame
    new_data = pd.DataFrame({
        'age': age,
        'sex': sex,
        'chest_pain': chest_pain,
        'arterial_pressure': arterial_pressure,
        'total_cholesterol': total_cholesterol,
        'blood_glucose': blood_glucose,
        'electrocardiogram_on_rest': electrocardiogram_on_rest,
        'max_heart_rate': max_heart_rate,
        'exercise_produced_angina': exercise_produced_angina,
        'unlevel_ST': unlevel_ST,
        'segment_st_in_ecg': segment_st_in_ecg,
        'main_vessels_coloured_by_fluorescence': main_vessels_coloured_by_fluorescence,
        'thalium': thalium,
        'disease': disease
    })

    return new_data
