import streamlit as st
import pickle
import pandas as pd
import requests
import os

# URL model dari GitHub
MODEL_URL = "https://github.com/luckstors/streamlit-seranganjantung/raw/main/rf_model.pkl"
MODEL_PATH = "rf_model.pkl"

# Download model jika belum ada
if not os.path.exists(MODEL_PATH):
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as file:
        file.write(response.content)

# Fungsi untuk memuat model
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

# Load model
model = load_model()

# Cek fitur yang digunakan saat training
model_features = model.feature_names_in_

# Judul aplikasi
st.title("Prediksi Serangan Jantung")
st.write("Masukkan data pengguna untuk mendapatkan prediksi risiko serangan jantung.")

# Input fitur dari pengguna
age = st.number_input("Usia", min_value=0, max_value=120, value=30)
gender = st.selectbox("Jenis Kelamin", options=["Male", "Female"])
region = st.selectbox("Wilayah", options=["Urban", "Rural"])
smoking = st.selectbox("Riwayat Merokok", options=["Yes", "No"])
diabetes = st.selectbox("Riwayat Diabetes", options=["Yes", "No"])
cholesterol = st.number_input("Kolesterol", min_value=100, max_value=600, value=200)
diet = st.number_input("Kualitas Diet (1-10)", min_value=1, max_value=10, value=5)
alcohol = st.selectbox("Konsumsi Alkohol", options=["Low", "Moderate", "High"])
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")

# Konversi input ke format yang sesuai
gender_encoded = 1 if gender == "Male" else 0
region_encoded = 1 if region == "Urban" else 0
smoking_encoded = 1 if smoking == "Yes" else 0
diabetes_encoded = 1 if diabetes == "Yes" else 0
alcohol_encoded = {"Low": 0, "Moderate": 1, "High": 2}[alcohol]

# Buat DataFrame sesuai dengan model
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender_encoded],
    "Region": [region_encoded],
    "Smoking_History": [smoking_encoded],
    "Diabetes_History": [diabetes_encoded],
    "Cholesterol_Level": [cholesterol],
    "Diet_Quality": [diet],
    "Alcohol_Consumption": [alcohol_encoded],
    "BMI": [bmi]
})

# Pastikan semua fitur yang ada di model juga ada di input
for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0  # Beri nilai default

# Pastikan urutan kolom sesuai dengan model
input_data = input_data[model_features]

# Tombol prediksi
if st.button("Prediksi"):
    try:
        # Lakukan prediksi
        prediction = model.predict(input_data)
        result = "Berisiko" if prediction[0] == 1 else "Tidak Berisiko"
        
        # Tampilkan hasil
        st.subheader("Hasil Prediksi:")
        st.write(f"Prediksi: **{result}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
