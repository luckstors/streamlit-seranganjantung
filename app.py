import streamlit as st
import pickle
import pandas as pd

# Load model
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)

# Load fitur yang digunakan saat pelatihan
feature_names = [
    "Age", "Region", "Smoking_History", "Alcohol_Consumption",
    "BMI", "Cholesterol_Level", "Diabetes_History", "Hypertension_History",
    "Diet_Quality", "Extra_Column_1", "Extra_Column_2", "Extra_Column_3"
]

# Judul aplikasi
st.title("Prediksi Gender Berdasarkan Data Kesehatan")
st.write(
    "Masukkan data pengguna untuk mendapatkan prediksi apakah gendernya **Male** atau **Female** berdasarkan fitur kesehatan."
)

# Input fitur dari pengguna
age = st.number_input("Usia", min_value=0, max_value=120, value=30)
region = st.selectbox("Wilayah", options=["Urban", "Rural"])
smoking_history = st.selectbox("Riwayat Merokok", options=["Yes", "No"])
alcohol_consumption = st.selectbox("Konsumsi Alkohol", options=["Yes", "No"])
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
cholesterol = st.number_input("Kolesterol", min_value=100, max_value=600, value=200)
diabetes_history = st.selectbox("Riwayat Diabetes", options=["Yes", "No"])
hypertension_history = st.selectbox("Riwayat Hipertensi", options=["Yes", "No"])
diet_quality = st.number_input("Kualitas Diet (1-10)", min_value=1, max_value=10, value=5)
extra_1 = st.number_input("Extra Column 1", value=0.0)
extra_2 = st.number_input("Extra Column 2", value=0.0)
extra_3 = st.number_input("Extra Column 3", value=0.0)

# Tombol prediksi
if st.button("Prediksi"):
    try:
        # Load model
        model = load_model()
        
        # Preprocessing input (encoding untuk 'Region', 'Smoking_History', dll.)
        region_encoded = 1 if region == "Urban" else 0
        smoking_encoded = 1 if smoking_history == "Yes" else 0
        alcohol_encoded = 1 if alcohol_consumption == "Yes" else 0
        diabetes_encoded = 1 if diabetes_history == "Yes" else 0
        hypertension_encoded = 1 if hypertension_history == "Yes" else 0

        input_data = pd.DataFrame(
            [[
                age, region_encoded, smoking_encoded, alcohol_encoded, bmi, cholesterol,
                diabetes_encoded, hypertension_encoded, diet_quality, extra_1, extra_2, extra_3
            ]],
            columns=feature_names,
        )

        # Prediksi
        prediction = model.predict(input_data)
        result = "Male" if prediction[0] == 1 else "Female"

        # Tampilkan hasil
        st.subheader("Hasil Prediksi:")
        st.write(f"Pengguna diprediksi: **{result}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
