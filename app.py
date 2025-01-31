import streamlit as st
import pickle
import pandas as pd

# Load model
def load_model():
    with open("/mnt/data/rf_model.pkl", "rb") as file:
        return pickle.load(file)

# Judul aplikasi
st.title("Prediksi Gender Berdasarkan Data Kesehatan")
st.write("Masukkan data pengguna untuk mendapatkan prediksi apakah gendernya **Male** atau **Female** berdasarkan fitur kesehatan.")

# Input fitur dari pengguna
age = st.number_input("Usia", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", options=["Male", "Female"])
region = st.selectbox("Wilayah", options=["Urban", "Rural"])
smoking = st.selectbox("Merokok", options=["Yes", "No"])
diabetes = st.selectbox("Diabetes", options=["Yes", "No"])
cholesterol = st.number_input("Kolesterol", min_value=100, max_value=600, value=200)
diet_quality = st.number_input("Kualitas Diet (1-10)", min_value=1, max_value=10, value=5)
alcohol = st.selectbox("Konsumsi Alkohol", options=["Low", "Moderate", "High"])
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")

# Tombol prediksi
if st.button("Prediksi"):
    try:
        # Load model
        model = load_model()
        
        # Preprocessing input (encoding categorical variables)
        gender_encoded = 1 if gender == "Male" else 0
        region_encoded = 1 if region == "Urban" else 0
        smoking_encoded = 1 if smoking == "Yes" else 0
        diabetes_encoded = 1 if diabetes == "Yes" else 0
        alcohol_encoded = {"Low": 0, "Moderate": 1, "High": 2}[alcohol]

        input_data = pd.DataFrame(
            [[age, gender_encoded, region_encoded, smoking_encoded, diabetes_encoded, cholesterol, diet_quality, alcohol_encoded, bmi]],
            columns=["Age", "Gender", "Region", "Smoking", "Diabetes", "Cholesterol", "Diet_Quality", "Alcohol", "BMI"]
        )

        # Prediksi
        prediction = model.predict(input_data)
        result = "Male" if prediction[0] == 1 else "Female"

        # Tampilkan hasil
        st.subheader("Hasil Prediksi:")
        st.write(f"Pengguna diprediksi: **{result}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
