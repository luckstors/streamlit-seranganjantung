import streamlit as st
import pickle
import pandas as pd
import requests

# URL model di GitHub
MODEL_URL = "https://raw.githubusercontent.com/luckstors/streamlit-seranganjantung/main/rf_model.pkl"
MODEL_PATH = "rf_model.pkl"

# Fungsi untuk mengunduh model jika belum ada
def download_model():
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as file:
            file.write(response.content)
    else:
        st.error("Gagal mengunduh model. Pastikan URL benar dan file tersedia.")

# Load model
def load_model():
    try:
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.warning("Model belum ada, mengunduh terlebih dahulu...")
        download_model()
        return load_model()

# Memuat model
model = load_model()

# Judul aplikasi
st.title("Prediksi Serangan Jantung Berdasarkan Data Kesehatan")

# Input fitur dari pengguna
age = st.number_input("Usia", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", options=["Male", "Female"])
region = st.selectbox("Wilayah", options=["Urban", "Rural"])
smoking = st.selectbox("Riwayat Merokok", options=["Yes", "No"])
diabetes = st.selectbox("Riwayat Diabetes", options=["Yes", "No"])
cholesterol = st.number_input("Kolesterol", min_value=100, max_value=600, value=200)
diet = st.number_input("Kualitas Diet (1-10)", min_value=1, max_value=10, value=5)
alcohol = st.selectbox("Konsumsi Alkohol", options=["Low", "Moderate", "High"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, format="%.1f")

# Tombol prediksi
if st.button("Prediksi"):
    try:
        # Preprocessing input (encoding untuk 'Gender', 'Region', dll.)
        gender_encoded = 1 if gender == "Male" else 0
        region_encoded = 1 if region == "Urban" else 0
        smoking_encoded = 1 if smoking == "Yes" else 0
        diabetes_encoded = 1 if diabetes == "Yes" else 0
        alcohol_encoded = {"Low": 0, "Moderate": 1, "High": 2}[alcohol]

        input_data = pd.DataFrame(
            [[age, gender_encoded, region_encoded, smoking_encoded, diabetes_encoded, cholesterol, diet, alcohol_encoded, bmi]],
            columns=["Age", "Gender", "Region", "Smoking", "Diabetes", "Cholesterol", "Diet", "Alcohol", "BMI"],
        )

        # Prediksi
        prediction = model.predict(input_data)
        result = "Berisiko" if prediction[0] == 1 else "Tidak Berisiko"

        # Tampilkan hasil
        st.subheader("Hasil Prediksi:")
        st.write(f"Risiko Serangan Jantung: **{result}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
